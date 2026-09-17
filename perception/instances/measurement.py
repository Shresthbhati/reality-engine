"""Measurement over merged object candidates (convergence campaign,
Phase 14 "Measurement" / Phase 11 in the master directive's priority
order). Builds directly on `object_resolution.MergedObjectCandidate`
(union AABB across all contributing views) -- the natural place real
object dimensions and distances become available in this pipeline.

Reuses the existing `world_ir.schema_v1.Measurement` type (value, unit,
precision, provenance, confidence) rather than inventing a parallel
measurement type -- the same class `evidence/promote_planes.py` and
`evidence/promote_rooms.py` already use for extent/thickness/area/height
measurements, so a candidate's dimensions are the same currency as a
promoted wall's.

Precision honesty: nothing upstream of this module currently propagates
per-axis geometric uncertainty (no covariance, no per-point error) --
using a documented approximation is explicitly permitted by this
campaign's own rule ("where exact mathematical propagation is
impossible, use clearly documented approximations"). The approximation
used here: `precision = value * (1 - confidence)`, floored at 1cm. A
low-confidence candidate (few/disagreeing observations) gets a wide
precision band; a high-confidence one gets a tight one. This is a
placeholder-grade heuristic, not a calibrated statistical model -- it is
named as such rather than presented as rigorous.

All measurements are `Provenance.ESTIMATED` (derived from reconstructed/
inferred geometry, never a direct physical observation) -- consistent
with every other geometry-derived Measurement in this repo.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from engine.physics.math3 import Vec3
from provenance import Provenance
from world_ir.schema_v1 import Measurement
from uncertainty import Uncertain, BASES, UNKNOWN

from perception.instances.object_resolution import MergedObjectCandidate

#: Floor for the precision heuristic below -- a measurement is never
#: presented as exact (zero margin of error), even for a fully-confident
#: candidate.
_MIN_PRECISION_M = 0.01

#: A spread beyond this many sigmas is not noise: the observations
#: disagree about the same physical quantity. Matching
#: reconstruction/fusion/fusion.py's CONFLICT_SIGMA semantics, the
#: disagreement is preserved rather than averaged away.
_CONFLICT_SIGMA = 5.0


def _estimate_precision(value: float, confidence: float) -> float:
    return max(_MIN_PRECISION_M, abs(value) * (1.0 - confidence))


def _measured_spread(candidate: MergedObjectCandidate) -> Optional[float]:
    """RMS 3D disagreement between the candidate's contributing
    hypotheses' positions -- the MEASURED spread of independent
    observations of the same object. None when it cannot be measured:
    fewer than two views, or any hypothesis without recorded position
    data (an evidence-only hypothesis counts toward confidence but
    contributes no geometry to measure disagreement with).

    Positions already disagree about WHERE the object is; the same
    spread bounds how well any dimension derived from its bounds can
    be known, so one scalar serves every axis honestly.
    """
    hyps = candidate.source_hypotheses
    if len(hyps) < 2:
        return None
    positions = [getattr(h, "position", None) for h in hyps]
    if any(p is None for p in positions):
        return None
    cx, cy, cz = (candidate.position.x, candidate.position.y,
                  candidate.position.z)
    deltas = []
    for h in hyps:
        dx = h.position.x - cx
        dy = h.position.y - cy
        dz = h.position.z - cz
        deltas.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
    return max(rms, _MIN_PRECISION_M)


def _spread_is_conflict(candidate: MergedObjectCandidate, spread: float) -> bool:
    """Does any view disagree with its peers beyond 5 sigma?

    Mirrors reconstruction/fusion/fusion.py's CONFLICT_SIGMA semantics,
    adapted to an honest statistical reality: with only two views and
    no per-view claimed precision, noise and disagreement are
    indistinguishable (n=2 is undecidable -- never flagged). With three
    or more, leave-one-out detection works: view i is an outlier when
    its distance to the OTHER views' centroid exceeds 5x the others'
    own RMS agreement. A flagged conflict is preserved, never averaged
    away.
    """
    hyps = candidate.source_hypotheses
    if len(hyps) < 3:
        return False

    def _dist(a: Vec3, b: Vec3) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 +
                         (a.z - b.z) ** 2)

    for i in range(len(hyps)):
        others = [h for j, h in enumerate(hyps) if j != i]
        n = len(others)
        ox = sum(h.position.x for h in others) / n
        oy = sum(h.position.y for h in others) / n
        oz = sum(h.position.z for h in others) / n
        other_centroid = Vec3(ox, oy, oz)
        d = _dist(hyps[i].position, other_centroid)
        if d <= 0.0:
            continue
        sigma_others = max(
            (sum(_dist(h.position, other_centroid) ** 2 for h in others) / n)
            ** 0.5,
            _MIN_PRECISION_M,
        )
        if d > _CONFLICT_SIGMA * sigma_others:
            return True
    return False


def _measurement(value: float, confidence: float, unit: str = "meter",
                 precision: Optional[float] = None,
                 precision_note: Optional[str] = None,
                 provenance: Provenance = Provenance.ESTIMATED) -> Measurement:
    # Map provenance to uncertainty basis
    prov_to_basis = {
        Provenance.OBSERVED: "measured",
        Provenance.RECONSTRUCTED: "measured",
        Provenance.ESTIMATED: "estimated",
        Provenance.INFERRED: "inferred",
        Provenance.GENERATED: "prior",
        Provenance.UNKNOWN: UNKNOWN,
        Provenance.CONFLICT: "estimated",
    }
    basis = prov_to_basis.get(provenance, "estimated")
    
    # Use explicit precision (measured spread) when available, else heuristic
    if precision is not None:
        sigma = max(_MIN_PRECISION_M, precision)
    else:
        # Single view: use documented confidence heuristic as sigma
        sigma = max(_MIN_PRECISION_M, _estimate_precision(value, confidence))
    
    uncertain = Uncertain(value=value, sigma=sigma, basis=basis)
    
    return Measurement(
        value=value, unit=unit,
        precision=max(_MIN_PRECISION_M, precision)
            if precision is not None else _estimate_precision(value, confidence),
        provenance=provenance, confidence=confidence,
        precision_note=precision_note,
        uncertain=uncertain,
    )


def measure_dimensions(candidate: MergedObjectCandidate) -> Dict[str, Measurement]:
    """width/height/depth (AABB extents along x/y/z) + volume for one
    candidate. Keys match the naming convention `evidence/promote_planes.py`
    already uses for entity custom_properties (`*_m`, `*_m3`).

    Precision source, in honest order of preference:
    1. Measured spread (>= 2 contributing views): the RMS disagreement
       between independent observations -- evidence, not assumption.
       Provenance stays ESTIMATED unless the views disagree beyond
       5 sigma, which flags CONFLICT (disagreement preserved, never
       averaged away).
    2. Documented confidence heuristic (single view): value * (1 -
       confidence), floored at 1 cm -- named in `precision_note` so a
       consumer can tell heuristic from measurement.
    """
    width = candidate.bounds_max.x - candidate.bounds_min.x
    height = candidate.bounds_max.y - candidate.bounds_min.y
    depth = candidate.bounds_max.z - candidate.bounds_min.z
    volume = width * height * depth

    spread = _measured_spread(candidate)
    if spread is not None:
        note = f"measured spread of {len(candidate.source_hypotheses)} views"
        prov = (Provenance.CONFLICT
                if _spread_is_conflict(candidate, spread)
                else Provenance.ESTIMATED)
        if prov is Provenance.CONFLICT:
            note += " (a view disagrees with its peers beyond 5 sigma "\
                    "-- conflict preserved)"
        return {
            "width_m": _measurement(width, candidate.confidence,
                                    precision=spread, precision_note=note,
                                    provenance=prov),
            "height_m": _measurement(height, candidate.confidence,
                                     precision=spread, precision_note=note,
                                     provenance=prov),
            "depth_m": _measurement(depth, candidate.confidence,
                                    precision=spread, precision_note=note,
                                    provenance=prov),
            "volume_m3": _measurement(volume, candidate.confidence,
                                      unit="meter^3", precision=spread,
                                      precision_note=note, provenance=prov),
        }

    # Single view: documented confidence heuristic
    fallback_note = ("documented confidence heuristic"
                     " (measured spread unavailable)")
    return {
        "width_m": _measurement(width, candidate.confidence,
                                precision_note=fallback_note),
        "height_m": _measurement(height, candidate.confidence,
                                 precision_note=fallback_note),
        "depth_m": _measurement(depth, candidate.confidence,
                                precision_note=fallback_note),
        "volume_m3": _measurement(volume, candidate.confidence,
                                  unit="meter^3", precision_note=fallback_note),
    }

    return {
        "width_m": _measurement(width, candidate.confidence,
                                precision_note=fallback_note),
        "height_m": _measurement(height, candidate.confidence,
                                 precision_note=fallback_note),
        "depth_m": _measurement(depth, candidate.confidence,
                                precision_note=fallback_note),
        "volume_m3": _measurement(volume, candidate.confidence, unit="meter^3",
                                  precision_note=fallback_note),
    }


def measure_distance(a: MergedObjectCandidate, b: MergedObjectCandidate) -> Measurement:
    """Centroid-to-centroid distance between two candidates. Confidence
    is the weaker of the two contributing candidates -- a distance is
    only as trustworthy as its least-certain endpoint. Precision is the
    weaker endpoint's measured spread when both candidates have one
    (a chain of evidence is only as strong as its weakest link);
    otherwise the documented heuristic of the weaker endpoint.
    """
    dx = a.position.x - b.position.x
    dy = a.position.y - b.position.y
    dz = a.position.z - b.position.z
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    confidence = min(a.confidence, b.confidence)

    spread_a = _measured_spread(a)
    spread_b = _measured_spread(b)
    if spread_a is not None and spread_b is not None:
        # Weaker (larger) endpoint spread governs.
        spread = max(spread_a, spread_b)
        views = (len(a.source_hypotheses), len(b.source_hypotheses))
        note = f"measured spread; weaker endpoint ({max(views)} views)"
        prov = Provenance.ESTIMATED
        if (_spread_is_conflict(a, spread_a)
                or _spread_is_conflict(b, spread_b)):
            prov = Provenance.CONFLICT
            note += " (endpoint peers disagree beyond 5 sigma)"
        return _measurement(distance, confidence, precision=spread,
                            precision_note=note, provenance=prov)

    return _measurement(
        distance, confidence,
        precision_note="documented confidence heuristic (endpoint)",
    )
