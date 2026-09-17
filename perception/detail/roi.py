"""ROI generation (universal-perception directive PHASE 2, sections
12/14): turning detail DISCOVERY into detail WORK ORDERS.

Position in the universal pipeline:

    detail discovery (perception/detail/discovery.py)
      -> (this module) ROI generation
      -> local refinement / adaptive reconstruction

A RegionOfInterest is the unit the local-reconstruction layer will
process: it groups one or more neighboring detail cells into a single
spatially bounded work order carrying

    ROI ID / spatial bounds / evidence sources / per-cell budgets /
    an aggregate detail budget / processing status / provenance

(directive section 12's ROI record).

Honesty rules:
  - Only cells measured as detail-bearing (curvature >= threshold at
    discovery time) seed ROIs. A flat cell is never promoted.
  - The ROI's aggregate budget is the MINIMUM of its cells' budgets
    (the weakest evidence in the group bounds the group -- a single
    single-view cell must not let its neighbors claim L4).
  - Processing status starts at "pending": this module generates work
    orders, it does not fake their execution.
  - Deterministic: same input -> byte-identical ROIs in a stable
    order (sorted by first cell key).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from perception.detail.discovery import DetailCandidate
from perception.quality.detail_budget import DetailBudget

__all__ = [
    "RegionOfInterest",
    "PROCESSING_PENDING",
    "DEFAULT_PLANARITY_SEED_THRESHOLD",
    "generate_rois",
]

#: Structure-seed gate for include_structure=True: a cell whose
#: measured planarity (1 - lambda_min/lambda_max) reaches this seeds
#: an ROI even when its curvature is below the detail gate. Oriented
#: planar structure (walls, floors) is invisible to the curvature
#: signal BY CONSTRUCTION (a plane scores ~0 curvature); the real
#: room capture proved the discovery layer needs the complementary
#: signal. Same deferral as every threshold here: documented, not
#: tuned against real datasets.
DEFAULT_PLANARITY_SEED_THRESHOLD = 0.95

#: Initial status of every generated ROI. Local refinement owns the
#: transitions (pending -> queued -> refined / refused); this module
#: never emits a terminal status, because no work has happened yet.
PROCESSING_PENDING = "pending"


@dataclass(frozen=True)
class RegionOfInterest:
    """One local-reconstruction work order (directive section 12)."""

    roi_id: str
    parent_entity_id: Optional[str]   # None until entity linkage exists
    bounds: Tuple[float, float, float, float, float, float]  # x0..z1
    n_cells: int
    n_points: int
    detail_cells: Tuple[str, ...]     # cell ids seeded by discovery
    point_ids: Tuple[str, ...]        # provenance: which points support it
    budget: DetailBudget              # MIN over member cells (weakest wins)
    max_curvature: float
    status: str = PROCESSING_PENDING
    provenance: Dict[str, object] = field(default_factory=dict)
    #: Set by the refinement lifecycle (apply_outcomes): the evidence
    #: for a "refined" status. None until refinement ran.
    refinement_outcome: Optional[object] = None
    #: Set by the refinement lifecycle: the diagnostic reason for a
    #: refusal (or None while pending/refined).
    status_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "roi_id": self.roi_id,
            "parent_entity_id": self.parent_entity_id,
            "bounds": list(self.bounds),
            "n_cells": self.n_cells,
            "n_points": self.n_points,
            "detail_cells": list(self.detail_cells),
            "point_ids": list(self.point_ids),
            "budget": self.budget.to_dict(),
            "max_curvature": self.max_curvature,
            "status": self.status,
            "provenance": dict(self.provenance),
            "refinement_outcome": (
                self.refinement_outcome.to_dict()
                if self.refinement_outcome is not None
                and hasattr(self.refinement_outcome, "to_dict")
                else None
            ),
            "status_reason": self.status_reason,
        }


def _merge_bounds(
    a: Tuple[float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    return (
        min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]),
        max(a[3], b[3]), max(a[4], b[4]), max(a[5], b[5]),
    )


def _cell_bounds(
    cell_id: str, voxel_size: float
) -> Tuple[float, float, float, float, float, float]:
    i, j, k = (int(v) for v in cell_id.split(","))
    return (i * voxel_size, j * voxel_size, k * voxel_size,
            (i + 1) * voxel_size, (j + 1) * voxel_size,
            (k + 1) * voxel_size)


def _weakest(budgets: List[DetailBudget]) -> DetailBudget:
    """The MIN of member budgets: lowest justified level wins; ties
    resolved toward the lower compute tier, then weaker GSD. Group
    claims are bounded by the group's weakest evidence."""
    order = ["L0", "L1", "L2", "L3", "L4"]
    tier_order = ["none", "survey", "light", "standard", "high", "full"]
    gsds = [b.max_gsd_mm_per_px for b in budgets
            if b.max_gsd_mm_per_px is not None]
    worst = min(budgets, key=lambda b: (
        order.index(b.justified_level),
        tier_order.index(b.compute_tier),
        -(b.max_gsd_mm_per_px if b.max_gsd_mm_per_px is not None
          else float("inf")),
    ))
    return DetailBudget(
        justified_level=worst.justified_level,
        compute_tier=worst.compute_tier,
        max_gsd_mm_per_px=min(gsds) if gsds else None,
        coverage_capped=any(b.coverage_capped for b in budgets),
        basis="unsupported" if all(
            b.basis == "unsupported" for b in budgets
        ) else "measured",
    )


def generate_rois(
    candidates: List[DetailCandidate],
    voxel_size: float,
    max_cells_per_roi: int = 27,
    include_structure: bool = False,
    planarity_seed_threshold: float = DEFAULT_PLANARITY_SEED_THRESHOLD,
) -> List[RegionOfInterest]:
    """Group detail-bearing candidate cells into bounded ROIs.

    Grouping rule (documented, deterministic): cells whose voxel keys
    are 26-adjacent (Chebyshev distance <= 1 in each axis) and whose
    combined cell count stays <= `max_cells_per_roi` merge into one
    ROI; everything else is its own ROI. ROIs are returned sorted by
    their first member cell key, so output order is stable under any
    input permutation of `candidates`.

    With `include_structure=True`, cells whose measured planarity
    reaches `planarity_seed_threshold` also seed ROIs (marked
    provenance seed="structure"), letting oriented planar structure
    -- invisible to the curvature signal by construction -- enter the
    refinement pipeline. The default preserves the detail-only
    contract.
    """
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if max_cells_per_roi < 1:
        raise ValueError("max_cells_per_roi must be >= 1")

    detail_cells = [c for c in candidates if c.is_detail]
    structure_cells: List[DetailCandidate] = []
    if include_structure:
        structure_cells = [
            c for c in candidates if c.is_structure
        ]
        detail_cells = detail_cells + structure_cells
    if not detail_cells:
        return []

    def key_of(c: DetailCandidate) -> Tuple[int, int, int]:
        return tuple(int(v) for v in c.cell_id.split(","))

    detail_cells = sorted(detail_cells, key=key_of)

    # Deterministic single-pass region growing over the sorted cells:
    # each cell joins the FIRST open region it touches (or starts a
    # new one when that region is full).
    regions: List[List[DetailCandidate]] = []
    for cell in detail_cells:
        key = key_of(cell)
        placed = False
        for region in regions:
            if len(region) >= max_cells_per_roi:
                continue
            if any(
                max(abs(key[d] - key_of(o)[d]) for d in range(3)) <= 1
                for o in region
            ):
                region.append(cell)
                placed = True
                break
        if not placed:
            regions.append([cell])

    rois: List[RegionOfInterest] = []
    for region in regions:
        region = sorted(region, key=key_of)
        bounds = _cell_bounds(region[0].cell_id, voxel_size)
        for cell in region[1:]:
            bounds = _merge_bounds(bounds, _cell_bounds(cell.cell_id, voxel_size))
        point_ids: List[str] = []
        for cell in region:
            for pid in cell.point_ids:
                if pid not in point_ids:
                    point_ids.append(pid)
        budgets = [c.budget for c in region]
        is_structure_seed = any(
            (not c.is_detail) for c in region
        )
        provenance: Dict[str, object] = {
            "voxel_size": voxel_size,
            "discovery": "perception.detail.discovery.discover_detail",
            "grouping": "26-adjacent region growth, <= max_cells_per_roi",
        }
        if is_structure_seed:
            # Record WHY the ROI exists: structure, not detail
            # curvature, with the measured planarity that seeded it.
            structure_planarities = [
                c.planarity for c in region if c.is_structure
            ]
            provenance["seed"] = "structure"
            provenance["planarity"] = max(structure_planarities)
        rois.append(RegionOfInterest(
            roi_id=f"roi-{region[0].cell_id.replace(',', '-')}",
            parent_entity_id=None,
            bounds=bounds,
            n_cells=len(region),
            n_points=sum(c.n_points for c in region),
            detail_cells=tuple(c.cell_id for c in region),
            point_ids=tuple(point_ids),
            budget=_weakest(budgets),
            max_curvature=max(c.curvature for c in region),
            status=PROCESSING_PENDING,
            provenance=provenance,
        ))

    rois.sort(key=lambda r: r.detail_cells[0])
    return rois
