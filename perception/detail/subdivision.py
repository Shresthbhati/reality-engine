"""Adaptive subdivision (REAL_RECONSTRUCTION_PERCEPTION_CITY_READY
increment C; advances ledger P7-05/P7-06): octree-cell subdivision
driven by a MEASURED detail score with bounded work and measured
stopping criteria — the detail spine's "adaptive subdivision" stage.

Position in the detail chain (directive sections 8/11):

    quality -> budget -> discovery -> ROI -> refinement
      -> (this module) adaptive subdivision of regions whose measured
         structure justifies deeper reconstruction
    -> bounded local refinement work allocation

Directive rule (mission section 4): only subdivide when
D > threshold AND evidence supports it — high polygon/analysis
density only where reality contains recoverable structure.

Score discipline (measured, never benchmark-tuned):
  - curvature: perception.detail.discovery's PCA smallest-eigenvalue
    ratio — the ONE shared closed-form eigensolver (plane ~0, curved
    shell ~1/3 for an isotropic surface, edges higher). No new
    geometry math is invented here.
  - importance: optional caller-supplied measured importance in [0,1]
    per cell (e.g. a detected-component's confidence); 0 when absent.
  - D = w_curvature * curvature + w_importance * importance, in [0,1].
  - density is NOT a score term: uniform density everywhere does not
    mean detail. It is the SUPPORT gate — a cell may only subdivide
    when it holds >= 2*min_points_per_cell so every child stays
    measurable (an unsubstantiated split would fabricate resolution).

Stopping criteria (each recorded per cell, machine-readable):
  below_threshold | max_depth | min_points | budget_exhausted
Work bound: max_cells caps visited cells; the frontier is expanded
highest-measured-score first, deterministically (score, then id).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from perception.detail.discovery import _pca_curvature

DEFAULT_W_CURVATURE = 1.0
DEFAULT_W_IMPORTANCE = 0.0
DEFAULT_THRESHOLD = 0.05
DEFAULT_MAX_DEPTH = 4
DEFAULT_MIN_POINTS = 8
DEFAULT_MAX_CELLS = 4096


@dataclass(frozen=True)
class SubdivisionCell:
    """One visited cell with its measured score and stop reason."""

    cell_id: str  # octree path: "0", "0/3", "0/3/5" (stable, sortable)
    depth: int
    n_points: int
    detail_score: float  # measured, in [0, 1]
    subdivided: bool
    stop_reason: str  # "subdivided" | "below_threshold" | "max_depth" | "min_points" | "budget_exhausted"

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "depth": self.depth,
            "n_points": self.n_points,
            "detail_score": self.detail_score,
            "subdivided": self.subdivided,
            "stop_reason": self.stop_reason,
        }


@dataclass
class SubdivisionReport:
    """Every visited cell (parents included) + work accounting."""

    cells: List[SubdivisionCell] = field(default_factory=list)
    subdivided_count: int = 0
    max_depth_used: int = 0
    work_units_spent: int = 0
    budget_exhausted: bool = False
    stopping_reason: str = "all_leaves"  # "all_leaves" | "max_cells"

    def to_dict(self) -> dict:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "subdivided_count": self.subdivided_count,
            "max_depth_used": self.max_depth_used,
            "work_units_spent": self.work_units_spent,
            "budget_exhausted": self.budget_exhausted,
            "stopping_reason": self.stopping_reason,
        }


def _as_tuples(points: Sequence) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for p in points:
        if hasattr(p, "x"):
            out.append((float(p.x), float(p.y), float(p.z)))
        else:
            out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def _octant_of(p: Tuple[float, float, float], mid: Tuple[float, float, float]) -> int:
    """Deterministic octant index 0..7 (x bit 2, y bit 1, z bit 0)."""
    return (
        (2 if p[0] >= mid[0] else 0)
        + (1 if p[1] >= mid[1] else 0) * 2
        + (1 if p[2] >= mid[2] else 0)
    )


def subdivide_adaptive(
    points: Sequence,
    *,
    score_threshold: float = DEFAULT_THRESHOLD,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_points_per_cell: int = DEFAULT_MIN_POINTS,
    max_cells: int = DEFAULT_MAX_CELLS,
    w_curvature: float = DEFAULT_W_CURVATURE,
    w_importance: float = DEFAULT_W_IMPORTANCE,
    importance_fn: Optional[object] = None,
) -> SubdivisionReport:
    """Measured adaptive octree subdivision of one point region.

    importance_fn(cell_id, depth, cell_points) -> float in [0,1] is an
    OPTIONAL measured semantic term; callers must not fake it.
    """
    pts = _as_tuples(points)
    report = SubdivisionReport()
    if not pts:
        return report

    lo = (
        min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts),
    )
    hi = (
        max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts),
    )

    # Frontier: (negative score, cell_id, depth, points, lo, hi) — the
    # highest measured score expands first; ties break by cell id.
    def measure(cell_pts):
        return max(0.0, min(1.0, w_curvature * _pca_curvature(cell_pts)))

    root_id = "0"
    root_score = measure(pts)
    frontier: List[Tuple[float, str, int, List, Tuple, Tuple]] = [
        (-root_score, root_id, 0, pts, lo, hi)
    ]
    visited: Dict[str, SubdivisionCell] = {}

    def record(cell_id, depth, cell_pts, score, subdivided, reason):
        visited[cell_id] = SubdivisionCell(
            cell_id=cell_id, depth=depth, n_points=len(cell_pts),
            detail_score=score, subdivided=subdivided, stop_reason=reason,
        )

    record(root_id, 0, pts, root_score, False, "below_threshold")

    while frontier and len(visited) < max_cells:
        neg_score, cell_id, depth, cell_pts, clo, chi = heapq.heappop(frontier)
        score = -neg_score

        if depth >= max_depth:
            record(cell_id, depth, cell_pts, score, False, "max_depth")
            continue
        if len(cell_pts) < 2 * min_points_per_cell:
            record(cell_id, depth, cell_pts, score, False, "min_points")
            continue
        if score <= score_threshold:
            record(cell_id, depth, cell_pts, score, False, "below_threshold")
            continue
        if len(visited) + 8 > max_cells:
            # Not enough budget to expand: record honestly, stop cleanly.
            record(cell_id, depth, cell_pts, score, False, "budget_exhausted")
            report.budget_exhausted = True
            break

        record(cell_id, depth, cell_pts, score, True, "subdivided")
        mid = (
            (clo[0] + chi[0]) / 2.0,
            (clo[1] + chi[1]) / 2.0,
            (clo[2] + chi[2]) / 2.0,
        )
        buckets: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(8)}
        for p in cell_pts:
            buckets[_octant_of(p, mid)].append(p)
        for octant in range(8):
            child_pts = buckets[octant]
            if not child_pts:
                continue
            child_id = f"{cell_id}/{octant}"
            child_lo = (
                clo[0] if not (octant & 4) else mid[0],
                clo[1] if not (octant & 2) else mid[1],
                clo[2] if not (octant & 1) else mid[2],
            )
            child_hi = (
                chi[0] if (octant & 4) else mid[0],
                chi[1] if (octant & 2) else mid[1],
                chi[2] if (octant & 1) else mid[2],
            )
            child_score = measure(child_pts)
            if importance_fn is not None and w_importance > 0.0:
                imp = float(importance_fn(child_id, depth + 1, child_pts))
                child_score = max(0.0, min(1.0, child_score + w_importance * imp))
            heapq.heappush(
                frontier,
                (-child_score, child_id, depth + 1, child_pts, child_lo, child_hi),
            )

    # Drain any frontier the budget could not visit: recorded as
    # budget_exhausted facts, never silently dropped.
    while frontier:
        neg_score, cell_id, depth, cell_pts, _clo, _chi = heapq.heappop(frontier)
        record(cell_id, depth, cell_pts, -neg_score, False, "budget_exhausted")
        report.budget_exhausted = True

    report.cells = [visited[k] for k in sorted(visited)]
    report.subdivided_count = sum(1 for c in report.cells if c.subdivided)
    report.max_depth_used = max((c.depth for c in report.cells), default=0)
    report.work_units_spent = len(report.cells)
    if report.budget_exhausted:
        report.stopping_reason = "max_cells"
    return report
