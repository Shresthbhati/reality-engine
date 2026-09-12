"""Deterministic RANSAC plane detection over reconstructed point clouds
(spec sec 17 GEOMETRIC REASONING, "planes").

The first real step of the geometric-reasoning layer the audit called the
biggest missing piece: reconstructed points enter as a ReconstructionResult,
coherent planar structure comes out as DetectedPlane records, ready for
orientation classification (see orientation.py) and WorldIR promotion (see
evidence/promote_planes.py).

Honesty rules this module follows (per CLAUDE.md / project conventions):
  - Determinism: any randomness comes from engine/core/rng's seeded
    DeterministicRNG, and output order is canonical (sorted by inlier
    count desc, then re-keyed ids), so the same points + seed always
    produce identical detections.
  - No fabrication: if the points contain no plane meeting the thresholds,
    detect_planes returns an empty list. It never invents a plane.
  - Honest bookkeeping: leftover/unassigned points are reported in the
    PlaneDetection result rather than silently dropped, and inlier counts
    always match the reported inlier id lists.
  - Noise guard: a plane survives only with at least MIN_INLIERS points
    AND through an iterative least-squares refit (see REFINE_ROUNDS);
    garbage fits through sparse noise die on one or the other.

Algorithm (standard RANSAC, dependency-free, sized for the sparse SfM
clouds COLMAP produces here -- hundreds of points, not million-point
dense scans): repeatedly sample 3 points, count consensus inliers within
PLANE_DISTANCE_TOLERANCE_M, take the best consensus over MAX_ITERATIONS,
then refine by least-squares refit + re-collection from all remaining
points until stable. Refinement matters: observed on a synthetic room, a
single refit of a slightly-contaminated consensus plane locked in a
5.6-degree tilted compromise, while refit-and-recollect rounds shed the
contaminants and converged onto the true plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from engine.core.rng import DeterministicRNG
from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructionResult

# ---- algorithm constants (module-level, documented, not magic inline numbers) ----

#: RANSAC distance in world units (meters) for a point to count as an
#: inlier of the candidate plane through three sampled points.
PLANE_DISTANCE_TOLERANCE_M = 0.08

#: A plane supported by fewer points than this is noise, not structure.
#: (Deliberately an absolute floor, not a fraction of the remaining
#: points: in any multi-structure scene each plane is a minority of the
#: cloud, and a majority requirement would make extracting even a second
#: plane mathematically impossible -- observed, not assumed.)
MIN_INLIERS = 8

#: Maximum RANSAC consensus iterations per extracted plane.
MAX_ITERATIONS = 500

#: Refit + re-collect rounds after the consensus step.
REFINE_ROUNDS = 3

#: Degenerate geometry guard: three samples spanning less than this
#: distance (in meters) define no reliable plane.
MIN_SAMPLE_SPREAD_M = 0.10


@dataclass(frozen=True)
class DetectedPlane:
    """One detected plane, in world coordinates.

    The raw normal's sign is whichever the fit produced; orienting it
    toward the cameras (needed for floor-vs-ceiling) is orientation.py's
    job, via flip_normal_toward().

    inlier_ids are sorted reconstruction track ids (deterministic order).
    inlier_rms_distance_m is the RMS residual of the inliers about the
    plane -- real fit quality, surfaced for provenance rather than hidden.
    """

    plane_id: str
    normal: Tuple[float, float, float]
    d: float  # plane equation: normal . p + d = 0
    inlier_ids: List[str]
    inlier_rms_distance_m: float = 0.0  # detection always supplies the real RMS; 0.0 = not computed (ad-hoc plane)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self) -> None:
        length = math.sqrt(sum(c * c for c in self.normal))
        if not math.isclose(length, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"plane normal must be unit length, got |n|={length!r}")
        if not self.inlier_ids:
            raise ValueError("a detected plane must have at least one inlier")
        if self.inlier_ids != sorted(self.inlier_ids):
            raise ValueError("inlier_ids must be sorted (deterministic order)")

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_ids)


@dataclass(frozen=True)
class PlaneDetection:
    """Full result of one detect_planes() run: the planes found plus the
    points they account for -- nothing silently dropped."""

    planes: List[DetectedPlane]
    unassigned_point_ids: List[str]  # sorted; in no plane
    points_total: int

    @property
    def points_in_planes(self) -> int:
        return sum(p.inlier_count for p in self.planes)


def as_vector(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a normal to unit length, raising on a degenerate input
    rather than returning an arbitrary unit vector."""
    length = math.sqrt(sum(c * c for c in normal))
    if length == 0.0 or not math.isfinite(length):
        raise ValueError(f"cannot normalize degenerate normal {normal!r}")
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def flip_normal_toward(
    normal: Tuple[float, float, float],
    d: float,
    reference_point: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], float]:
    """Return (normal, d) flipped so the normal points toward
    `reference_point`. The plane equation normal . p + d = 0 stays
    equivalent because both components negate together. Used by
    orientation.py to orient a plane's normal toward the cameras that
    observed it."""
    side = normal[0] * reference_point[0] + normal[1] * reference_point[1] + normal[2] * reference_point[2] + d
    if side >= 0.0:
        return normal, d
    return (-normal[0], -normal[1], -normal[2]), -d


def _plane_from_three_points(
    p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]
) -> Tuple[Tuple[float, float, float], float]:
    """Exact plane through three points via the cross product of the two
    edge vectors. Caller guarantees non-degenerate spread."""
    e1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    e2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    normal = as_vector((
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ))
    d = -(normal[0] * p1[0] + normal[1] * p1[1] + normal[2] * p1[2])
    return normal, d


def _jacobi_smallest_eigenvector(
    cov: Tuple[Tuple[float, float, float], Tuple[Tuple[float, float, float], Tuple[Tuple[float, float, float], Tuple[float, float, float], float], float], Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Eigenvector for the smallest eigenvalue of a symmetric 3x3 matrix,
    by cyclic Jacobi rotations (deterministic, dependency-free).

    For a point set's covariance this is the least-squares plane normal:
    the direction of least variance. Robust where a 'dominant direction
    then cross product' approximation is not -- a near-square planar
    patch has two nearly-equal large eigenvalues, and contamination can
    swing the inferred least-variance direction wildly there, which was
    observed as tilted refits before this replaced that approach.
    """
    a = [list(cov[0]), list(cov[1]), list(cov[2])]
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for _ in range(64):
        p, q, off = 0, 1, abs(a[0][1])
        if abs(a[0][2]) > off:
            p, q, off = 0, 2, abs(a[0][2])
        if abs(a[1][2]) > off:
            p, q, off = 1, 2, abs(a[1][2])
        if off <= 1e-14:
            break
        phi = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(phi), math.sin(phi)
        for k in range(3):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(3):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    evals = (a[0][0], a[1][1], a[2][2])
    idx = min(range(3), key=lambda i: evals[i])
    return as_vector((v[0][idx], v[1][idx], v[2][idx]))


def _fit_plane_to_inliers(
    positions: List[Tuple[float, float, float]]
) -> Tuple[Tuple[float, float, float], float]:
    """Least-squares plane through a point set: smallest-eigenvector of
    the covariance (via Jacobi), anchored at the centroid. Raises
    ValueError on a degenerate set (no spread => no plane)."""
    n = len(positions)
    if n < 3:
        raise ValueError("need >= 3 points to fit a plane")
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n

    xx = yy = zz = xy = xz = yz = 0.0
    for px, py, pz in positions:
        dx, dy, dz = px - cx, py - cy, pz - cz
        xx += dx * dx; yy += dy * dy; zz += dz * dz
        xy += dx * dy; xz += dx * dz; yz += dy * dz
    inv = 1.0 / n
    cov = ((xx * inv, xy * inv, xz * inv),
           (xy * inv, yy * inv, yz * inv),
           (xz * inv, yz * inv, zz * inv))

    normal = _jacobi_smallest_eigenvector(cov)
    d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz)
    return normal, d


def _rms_distance(positions: List[Tuple[float, float, float]], normal, d) -> float:
    if not positions:
        return 0.0
    total = 0.0
    for p in positions:
        dist = normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d
        total += dist * dist
    return math.sqrt(total / len(positions))


def _sample_three(rng: DeterministicRNG, count: int) -> Tuple[int, int, int]:
    """Three distinct indices, deterministic under the seeded rng.
    Re-draws on collision; the modulo fallback guarantees termination
    even for a pathologically small population."""
    i = rng.randint(0, count - 1)
    j = rng.randint(0, count - 1)
    k = rng.randint(0, count - 1)
    attempts = 0
    while len({i, j, k}) < 3 and attempts < 100:
        if i == j:
            j = rng.randint(0, count - 1)
        if j == k:
            k = rng.randint(0, count - 1)
        if i == k and len({i, j, k}) < 3:
            k = rng.randint(0, count - 1)
        attempts += 1
    if len({i, j, k}) < 3:
        k = (k + 1) % count
        if k == i:
            k = (k + 1) % count
    return i, j, k


def detect_planes(
    result: ReconstructionResult,
    seed: int,
    distance_tolerance_m: float = PLANE_DISTANCE_TOLERANCE_M,
    min_inliers: int = MIN_INLIERS,
    max_iterations: int = MAX_ITERATIONS,
) -> PlaneDetection:
    """Detect planar structure in a ReconstructionResult's points.

    Iteratively extracts the plane with the most RANSAC consensus
    inliers from the remaining points (refined by least-squares refit +
    re-collection), and stops when no plane meets min_inliers.
    Deterministic for a given (result, seed): sampling, tie-breaking,
    and ordering all come from the seeded rng and sorted ids, never from
    dict/set iteration.
    """
    positions_by_id = {p.track_id: p.position for p in result.points}
    remaining_ids = sorted(positions_by_id.keys())

    planes: List[DetectedPlane] = []
    plane_seq = 0

    while len(remaining_ids) >= max(3, min_inliers):
        rng = DeterministicRNG(seed, name=f"plane-ransac/{plane_seq}")
        remaining_sorted = sorted(remaining_ids)

        def _collect(normal, d) -> List[str]:
            return [
                pid for pid in remaining_sorted
                if abs(normal[0] * positions_by_id[pid][0]
                       + normal[1] * positions_by_id[pid][1]
                       + normal[2] * positions_by_id[pid][2]
                       + d) <= distance_tolerance_m
            ]

        # ---- consensus ----
        best_normal = None
        best_d = 0.0
        best_inliers: List[str] = []

        for _ in range(max_iterations):
            i, j, k = _sample_three(rng, len(remaining_sorted))
            p1 = positions_by_id[remaining_sorted[i]]
            p2 = positions_by_id[remaining_sorted[j]]
            p3 = positions_by_id[remaining_sorted[k]]

            spread = max(math.dist(p1, p2), math.dist(p1, p3), math.dist(p2, p3))
            if spread < MIN_SAMPLE_SPREAD_M:
                continue

            try:
                normal, d = _plane_from_three_points(p1, p2, p3)
            except ValueError:
                continue

            inliers = _collect(normal, d)
            # Strictly-greater keeps the first-encountered best on ties,
            # which the seeded rng makes deterministic.
            if len(inliers) > len(best_inliers):
                best_normal, best_d, best_inliers = normal, d, inliers

        if best_normal is None or len(best_inliers) < min_inliers:
            break

        # ---- iterative refinement: refit, then re-collect from ALL
        # remaining points (not just the current inliers) so contaminants
        # can fall out; keep each round only if it stays above the floor.
        normal, d, inliers = best_normal, best_d, best_inliers
        for _ in range(REFINE_ROUNDS):
            try:
                refit_normal, refit_d = _fit_plane_to_inliers(
                    [positions_by_id[pid] for pid in inliers]
                )
            except ValueError:
                break
            collected = _collect(refit_normal, refit_d)
            if len(collected) < min_inliers:
                break  # refinement collapsed the plane; keep last good one
            normal, d, inliers = refit_normal, refit_d, collected

        if len(inliers) < min_inliers:
            break

        # ---- final consistency refit: the reported plane must be the
        # least-squares plane OF the reported inliers (standard RANSAC
        # refit contract). Without this, a contaminated intermediate
        # collection leaves a tilted plane stored alongside a clean final
        # inlier set -- observed on the synthetic room before this fix.
        try:
            normal, d = _fit_plane_to_inliers([positions_by_id[pid] for pid in inliers])
        except ValueError:
            pass  # keep the last collect-consistent plane; inliers unchanged

        rms = _rms_distance([positions_by_id[pid] for pid in inliers], normal, d)
        planes.append(DetectedPlane(
            plane_id=f"plane-{plane_seq:03d}",
            normal=normal,
            d=d,
            inlier_ids=sorted(inliers),
            inlier_rms_distance_m=rms,
            uncertainty=Uncertainty(
                confidence=min(1.0, len(inliers) / (len(inliers) + 10.0)),
                note=(
                    f"RANSAC consensus {len(best_inliers)} points, "
                    f"refined to {len(inliers)} inliers, rms {rms:.4f} m"
                ),
            ),
        ))
        claimed = set(inliers)
        remaining_ids = [pid for pid in remaining_ids if pid not in claimed]
        plane_seq += 1

    # Sort by inlier count desc (structure first), then re-key ids to the
    # final sorted position so labeling is independent of extraction
    # order -- rebuilt as fresh frozen instances, never mutated in place.
    planes.sort(key=lambda p: (-p.inlier_count, p.plane_id))
    planes = [
        DetectedPlane(
            plane_id=f"plane-{seq:03d}",
            normal=plane.normal,
            d=plane.d,
            inlier_ids=plane.inlier_ids,
            inlier_rms_distance_m=plane.inlier_rms_distance_m,
            uncertainty=Uncertainty(
                confidence=plane.uncertainty.confidence,
                note=plane.uncertainty.note,
            ),
        )
        for seq, plane in enumerate(planes)
    ]

    assigned = {pid for plane in planes for pid in plane.inlier_ids}
    unassigned = sorted(pid for pid in positions_by_id if pid not in assigned)
    return PlaneDetection(
        planes=planes,
        unassigned_point_ids=unassigned,
        points_total=len(result.points),
    )
