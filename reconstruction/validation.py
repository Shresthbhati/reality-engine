"""Reconstruction validation (Goal F): compare two independent reconstructions
of overlapping evidence and surface disagreement instead of assuming agreement.

Minimal real slice: nearest-point matching between two ReconstructionResults
within a distance threshold. No smoothing, no averaging, no silently
picking a "winner" -- disagreement is reported, not resolved.

The nearest-match core uses a cKDTree (same dependency the repo's
consistency/fusion/quality modules already rely on), so validation stays
tractable at large-capture scale: O((N+M) log M) instead of the original
O(N*M) Python scan that could not complete on dense (10^5-point) results.
Report semantics are unchanged: first-in-a-order classification, every
a-point's nearest b-point is consumed (on agreement AND disagreement),
unmatched_b collapses duplicate track_ids into a set. For exact-tie
nearest candidates the tree returns a deterministic choice that may
differ from a naive first-min -- the report stays deterministic either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult


class UnvalidatableReconstructionError(ValueError):
    pass


@dataclass(frozen=True)
class PointDisagreement:
    point_a_id: str
    point_b_id: str
    distance: float


@dataclass(frozen=True)
class ValidationReport:
    agreements: int
    disagreements: List[PointDisagreement] = field(default_factory=list)
    unmatched_a: int = 0
    unmatched_b: int = 0

    def has_disagreements(self) -> bool:
        return len(self.disagreements) > 0


def validate_reconstructions(
    result_a: ReconstructionResult,
    result_b: ReconstructionResult,
    distance_threshold: float,
) -> ValidationReport:
    """Nearest-point match every point in result_a against result_b.

    A pair matched as each other's nearest neighbor within
    distance_threshold counts as agreement; otherwise it's either a
    disagreement (nearest neighbor exists but too far) or unmatched
    (no candidate at all, e.g. empty result).

    Raises UnvalidatableReconstructionError if either input is a failed
    reconstruction -- there is nothing honest to compare against a
    reconstruction that produced no points.
    """
    if result_a.registration_status == "failed" or result_b.registration_status == "failed":
        raise UnvalidatableReconstructionError(
            "cannot validate a failed reconstruction against anything -- "
            f"status_a={result_a.registration_status!r}, status_b={result_b.registration_status!r}"
        )

    if not result_b.points:
        # Nothing to match against: every a-point is unmatched, no b-side
        # candidates exist to consume.
        return ValidationReport(
            agreements=0,
            disagreements=[],
            unmatched_a=len(result_a.points),
            unmatched_b=0,
        )

    b_positions = np.array([p.position for p in result_b.points], dtype=np.float64)
    tree = cKDTree(b_positions)

    matched_b_ids = set()
    agreements = 0
    disagreements: List[PointDisagreement] = []

    if result_a.points:
        a_positions = np.array([p.position for p in result_a.points], dtype=np.float64)
        distances, nearest_idx = tree.query(a_positions, k=1)
        # Single a-point degenerate: query returns scalars, not arrays.
        distances = np.atleast_1d(distances)
        nearest_idx = np.atleast_1d(nearest_idx)

        for i, point_a in enumerate(result_a.points):
            nearest = result_b.points[int(nearest_idx[i])]
            distance = float(distances[i])
            if distance <= distance_threshold:
                agreements += 1
            else:
                disagreements.append(PointDisagreement(
                    point_a_id=point_a.track_id, point_b_id=nearest.track_id, distance=distance,
                ))
            matched_b_ids.add(nearest.track_id)

    unmatched_b = len({p.track_id for p in result_b.points} - matched_b_ids)

    return ValidationReport(
        agreements=agreements,
        disagreements=disagreements,
        unmatched_a=0,
        unmatched_b=unmatched_b,
    )
