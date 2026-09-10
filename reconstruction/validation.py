"""Reconstruction validation (Goal F): compare two independent reconstructions
of overlapping evidence and surface disagreement instead of assuming agreement.

Minimal real slice: nearest-point matching between two ReconstructionResults
within a distance threshold. No smoothing, no averaging, no silently
picking a "winner" -- disagreement is reported, not resolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

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


def _distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


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

    unmatched_b_ids = {p.track_id for p in result_b.points}
    agreements = 0
    disagreements: List[PointDisagreement] = []

    for point_a in result_a.points:
        if not result_b.points:
            continue
        nearest = min(result_b.points, key=lambda p: _distance(point_a.position, p.position))
        distance = _distance(point_a.position, nearest.position)
        if distance <= distance_threshold:
            agreements += 1
            unmatched_b_ids.discard(nearest.track_id)
        else:
            disagreements.append(PointDisagreement(
                point_a_id=point_a.track_id, point_b_id=nearest.track_id, distance=distance,
            ))
            unmatched_b_ids.discard(nearest.track_id)

    unmatched_a = 0 if result_b.points else len(result_a.points)
    unmatched_b = len(unmatched_b_ids) if result_b.points else len(result_b.points)

    return ValidationReport(
        agreements=agreements,
        disagreements=disagreements,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )
