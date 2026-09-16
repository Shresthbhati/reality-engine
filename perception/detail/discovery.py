"""Universal detail discovery (universal-perception directive PHASE 2-3,
sections 8/11): WHERE does the evidence support recoverable detail?

Position in the universal pipeline (directive section 8):

    dense reconstruction -> multi-scale representation
      -> (this module) detail discovery
      -> ROI generation -> adaptive local reconstruction

Domain-agnostic by construction: the input is a reconstruction's 3D
points + the measured evidence-quality report. A carved facade, a
milled gear, tree bark, and a road crack are all just points here --
there is no architecture-specific (or any domain-specific) branch.

Method (all scores MEASURED, never fabricated):
  - Points are binned into a voxel grid (default 1.0 m, documented,
    deterministic binning: floor division on coordinates, cell key is
    the integer triple sorted lexicographically).
  - Per cell, surface curvature is the PCA smallest-eigenvalue ratio
    (lambda_min / sum(lambda)): ~0 for a plane, ~1/3 for isotropic
    noise, high (>= 0.1 documented) for curved structure. Computed
    with the repo's closed-form Jacobi eigen solver (no new
    dependency; numpy-free so it matches the repo's pure-Python
    deterministic style for perception math).
  - Per cell, point density is count / (voxel_size^2 * surface
    fraction proxy): reported as points-per-m2 (count / voxel_area).
  - Every candidate cell's detail budget is derived by the DOCUMENTED
    detail_budget mapping, from the cell's OWN evidence: the scene
    report's measured GSD and the per-track view counts restricted to
    the cell's points (single-view cells are coverage-capped by that
    mapping -- the hallucination gate: a flat, low-density,
    coarse-evidence cell CANNOT claim L4).

Honesty rules:
  - Cells with fewer than `min_points` points are skipped, never
    guessed (curvature of 3 points is meaningless).
  - A cell the assessor marks unsupported (no measured GSD) gets an
    honest "unsupported" budget, never a fabricated level.
  - Determinism: identical input -> byte-identical candidates in a
    stable spatial order (sorted by cell key).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from perception.quality.assessment import (
    EvidenceQualityReport,
)
from perception.quality.detail_budget import DetailBudget, detail_budget_for
from reconstruction.backend.interface import ReconstructionResult

__all__ = [
    "DetailCandidate",
    "DEFAULT_VOXEL_SIZE_M",
    "DEFAULT_CURVATURE_THRESHOLD",
    "DEFAULT_MIN_POINTS",
    "discover_detail",
]


#: Default spatial bin (m). Coarser than the finest defensible GSD in
#: almost any real capture; a per-scene choice is passed explicitly by
#: callers that know better. NOT tuned against real datasets -- the
#: same deferral as every threshold in this repo until real-capture
#: benchmarks exist.
DEFAULT_VOXEL_SIZE_M = 1.0

#: Curvature above which a cell is flagged as detail-bearing. The
#: PCA smallest-eigenvalue ratio of a plane is ~0; documented value is
#: a round number safely between the two (plane fixtures measure
#: < 0.05, cylinder fixtures > 0.1).
DEFAULT_CURVATURE_THRESHOLD = 0.1

#: Cells with fewer points than this are skipped (curvature from
#: fewer points is numerically meaningless, not merely noisy).
DEFAULT_MIN_POINTS = 8


@dataclass(frozen=True)
class DetailCandidate:
    """One evidence-backed candidate region for detail work.

    `curvature` and `density_per_m2` are measured from the cell's
    points. `budget` is derived by the documented detail_budget
    mapping from the cell's own evidence-quality measurement.
    """

    cell_id: str          # "i,j,k" integer voxel key (stable, sortable)
    centroid: Tuple[float, float, float]
    n_points: int
    curvature: float      # PCA smallest-eigenvalue ratio, 0..1
    density_per_m2: float
    is_detail: bool       # curvature >= threshold
    budget: DetailBudget
    point_ids: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "centroid": list(self.centroid),
            "n_points": self.n_points,
            "curvature": self.curvature,
            "density_per_m2": self.density_per_m2,
            "is_detail": self.is_detail,
            "budget": self.budget.to_dict(),
            "point_ids": list(self.point_ids),
        }


def _bin_key(
    position: Tuple[float, float, float], voxel: float
) -> Tuple[int, int, int]:
    """Deterministic voxel binning: floor division per axis."""
    return (
        math.floor(position[0] / voxel),
        math.floor(position[1] / voxel),
        math.floor(position[2] / voxel),
    )


def _pca_curvature(points: Sequence[Tuple[float, float, float]]) -> float:
    """PCA smallest-eigenvalue ratio of a point set (surface
    curvature proxy). Closed-form via the characteristic polynomial
    (the repo's established eigen approach): exactly deterministic,
    no iterations, no numpy dispatch overhead.

    lambda_min / (l0 + l1 + l2): plane ~0, curved shell high.
    """
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    sxx = sum((p[0] - cx) ** 2 for p in points)
    syy = sum((p[1] - cy) ** 2 for p in points)
    szz = sum((p[2] - cz) ** 2 for p in points)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points)
    sxz = sum((p[0] - cx) * (p[2] - cz) for p in points)
    syz = sum((p[1] - cy) * (p[2] - cz) for p in points)

    # Coefficients of char. poly: det(A - t I) = -t^3 + c2 t^2 - c1 t + c0
    c2 = sxx + syy + szz
    c1 = (
        sxx * syy + sxx * szz + syy * szz
        - sxy * sxy - sxz * sxz - syz * syz
    )
    c0 = (
        sxx * syy * szz + 2.0 * sxy * sxz * syz
        - sxx * syz * syz - syy * sxz * sxz - szz * sxy * sxy
    )

    # Eigenvalues solve: t^3 - c2 t^2 + c1 t - c0 = 0. Depress with
    # t = y + s, s = c2/3:  y^3 + p y + q = 0 with
    # p = c1 - 3 s^2,  q = -2 s^3 + c1 s - c0.
    #
    # The scatter matrix is real symmetric, so ALL three eigenvalues
    # are mathematically real and the cubic's discriminant is always
    # <= 0; floating-point error can push it infinitesimally positive
    # (e.g. exactly-degenerate spectra like a straight line), where a
    # Cardano branch would silently drop two roots. Clamp to the
    # mathematically guaranteed regime and solve the trigonometric
    # three-real-root form.
    s = c2 / 3.0
    p = c1 - 3.0 * s * s
    q = -2.0 * s ** 3 + c1 * s - c0
    disc = min(0.0, (q / 2.0) ** 2 + (p / 3.0) ** 3)

    denom = 2.0 * math.sqrt(-(p / 3.0) ** 3)
    if denom == 0.0:
        # p == q == 0: triple root at the mean (or the degenerate
        # total <= 0 case handled above).
        roots = [s]
    else:
        r = 2.0 * math.sqrt(-p / 3.0)
        phi = math.acos(max(-1.0, min(1.0, -q / denom))) / 3.0
        roots = [
            r * math.cos(phi) + s,
            r * math.cos(phi - 2.0 * math.pi / 3.0) + s,
            r * math.cos(phi - 4.0 * math.pi / 3.0) + s,
        ]

    total = c2
    if total <= 0.0:
        # All points identical (degenerate cell): no measurable
        # curvature; honest zero, not a guess.
        return 0.0
    lam_min = min(roots)
    return max(0.0, lam_min / total)


def _cell_budget(
    report: EvidenceQualityReport, cell_point_ids
) -> DetailBudget:
    """Derive ONE cell's budget from the scene report's measured facts
    restricted to the cell: the report's measured GSD (scene-level
    measurement -- reused, not re-guessed) and the cell's own view
    counts. Reuses detail_budget_for unchanged (no duplicated
    mapping); an empty view subset coverage-caps the cell honestly."""
    ids = set(cell_point_ids)
    cell_views = {
        tid: n for tid, n in report.view_counts.items() if tid in ids
    }
    return detail_budget_for(EvidenceQualityReport(
        gsd_mm_per_px=report.gsd_mm_per_px,
        detail_tier=report.detail_tier,
        observed_fraction=report.observed_fraction,
        view_counts=cell_views,
        unprojectable_point_ids=(),
        overclaim_count=0,
    ))


def discover_detail(
    result: ReconstructionResult,
    report: EvidenceQualityReport,
    voxel_size: float = DEFAULT_VOXEL_SIZE_M,
    curvature_threshold: float = DEFAULT_CURVATURE_THRESHOLD,
    min_points: int = DEFAULT_MIN_POINTS,
) -> List[DetailCandidate]:
    """Discover detail-bearing regions from a reconstruction's points
    and its measured evidence-quality report (P7-04).

    Deterministic: same input -> byte-identical output, candidates in
    ascending cell-key order. Cells below `min_points` are skipped.
    Every returned candidate carries a budget derived from its own
    view support under the documented budget mapping (the
    hallucination gate: weak evidence cannot claim a fine level).
    """
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")

    cells: Dict[Tuple[int, int, int], List] = {}
    for point in result.points:
        cells.setdefault(
            _bin_key(point.position, voxel_size), []
        ).append(point)

    candidates: List[DetailCandidate] = []
    for key in sorted(cells):
        cell_points = cells[key]
        if len(cell_points) < min_points:
            continue

        positions = [tuple(float(c) for c in p.position) for p in cell_points]
        n = len(positions)
        centroid = (
            sum(p[0] for p in positions) / n,
            sum(p[1] for p in positions) / n,
            sum(p[2] for p in positions) / n,
        )
        curvature = _pca_curvature(positions)
        budget = _cell_budget(report, (p.track_id for p in cell_points))

        # Density: points per m2 of the cell's footprint. For cells
        # this small the voxel-area proxy (voxel^2) is the documented
        # convention; it is comparable across cells of the same run.
        density = n / (voxel_size * voxel_size)

        candidates.append(DetailCandidate(
            cell_id=f"{key[0]},{key[1]},{key[2]}",
            centroid=centroid,
            n_points=n,
            curvature=curvature,
            density_per_m2=density,
            is_detail=curvature >= curvature_threshold,
            budget=budget,
            point_ids=tuple(p.track_id for p in cell_points),
        ))

    return candidates
