"""Uncertainty conventions for the spatial compiler.

Decisions (each is checked by tests/test_spatial_uncertainty.py and the benchmark):
  * Six epistemic statuses (model.Status): directly observed, geometrically inferred, semantically inferred,
    weakly supported, conflicting, unknown.
  * Confidence is represented SEPARATELY per dimension (existence, position, dimensions, orientation, geometry,
    topology, semantic). They are measurably independent: a wall can be known to ~1 cm in position while its
    class is a single-view guess, or be certainly present while its length is only a lower bound. A single
    scalar would hide exactly that.
  * Spread (a sigma in metres/radians, from error propagation) and confidence (a probability a producer stated)
    are different facts and are never converted into one another. A dimension with no sigma is None, not 0.
  * UNKNOWN propagates as UNKNOWN: a quantity computed from an unknown input is unknown (None), never zero and
    never high confidence.
  * Downstream geometry inherits upstream sigma: an opening's across-wall position carries its host's offset
    sigma; room area carries the first-order propagated sigma of its corner nodes; a corner shared by two walls
    has the posterior sigma of the joint constraint solve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from provenance import Provenance
from world_ir.schema_v1 import ConfidenceVector

from .model import Belief, SpatialEntity, Status


def chain_sigma(*sigmas: float | None) -> float | None:
    """Independent errors add in quadrature; any unknown (None) makes the result unknown."""
    if any(s is None for s in sigmas):
        return None
    return math.sqrt(sum(s * s for s in sigmas))  # type: ignore[operator]


def fuse_inverse_variance(estimates: Sequence[tuple[float, float | None]]) -> tuple[float | None, float | None]:
    """Fuse independent (value, sigma) estimates. An estimate without a sigma cannot be weighted, so it is
    excluded rather than treated as exact; if none can be weighted the result is (None, None)."""
    use = [(v, s) for v, s in estimates if s is not None and s > 0]
    if not use:
        return None, None
    w = [1.0 / (s * s) for _, s in use]
    return sum(wi * v for wi, (v, _) in zip(w, use)) / sum(w), 1.0 / math.sqrt(sum(w))


def consistent(a: tuple[float, float | None], b: tuple[float, float | None], k: float = 3.0, floor: float = 0.0) -> bool | None:
    """Do two estimates agree within k sigma? None when a sigma is unknown (undecidable, not 'yes')."""
    if a[1] is None or b[1] is None:
        return None
    return abs(a[0] - b[0]) <= k * math.sqrt(a[1] ** 2 + b[1] ** 2 + floor ** 2)


def weakest(statuses: Sequence[Status]) -> Status:
    rank = [Status.CONFLICTING, Status.UNKNOWN, Status.WEAKLY_SUPPORTED, Status.SEMANTICALLY_INFERRED,
            Status.GEOMETRICALLY_INFERRED, Status.DIRECTLY_OBSERVED]
    return min(statuses, key=rank.index) if statuses else Status.UNKNOWN


def provenance_for(e: SpatialEntity) -> Provenance:
    """Entity-level provenance from its position belief: a measured, fitted position is ESTIMATED (never OBSERVED:
    it is a lifted, fitted quantity), a conflicted one is CONFLICT, anything weaker is INFERRED."""
    pos = e.beliefs["position"].status
    if pos is Status.CONFLICTING:
        return Provenance.CONFLICT
    if pos in (Status.DIRECTLY_OBSERVED, Status.GEOMETRICALLY_INFERRED):
        return Provenance.ESTIMATED
    return Provenance.INFERRED if pos is not Status.UNKNOWN else Provenance.UNKNOWN


def confidence_vector(e: SpatialEntity) -> ConfidenceVector:
    """Only dimensions for which a producer-stated probability exists are filled; everything else stays None.
    (identity <- existence, classification <- semantic.) Sigma-valued dimensions are NOT turned into confidences."""
    b = e.beliefs
    return ConfidenceVector(identity=b["existence"].confidence, classification=b["semantic"].confidence)


def beliefs_to_dict(beliefs: dict[str, Belief]) -> dict:
    return {k: v.to_dict() for k, v in beliefs.items()}


def beliefs_from_dict(d: dict) -> dict[str, Belief]:
    return {k: Belief.from_dict(v) for k, v in d.items()}
