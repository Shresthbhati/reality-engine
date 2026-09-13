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

from typing import Dict

from provenance import Provenance
from world_ir.schema_v1 import Measurement

from perception.instances.object_resolution import MergedObjectCandidate

#: Floor for the precision heuristic below -- a measurement is never
#: presented as exact (zero margin of error), even for a fully-confident
#: candidate.
_MIN_PRECISION_M = 0.01


def _estimate_precision(value: float, confidence: float) -> float:
    return max(_MIN_PRECISION_M, abs(value) * (1.0 - confidence))


def _measurement(value: float, confidence: float, unit: str = "meter") -> Measurement:
    return Measurement(
        value=value, unit=unit,
        precision=_estimate_precision(value, confidence),
        provenance=Provenance.ESTIMATED, confidence=confidence,
    )


def measure_dimensions(candidate: MergedObjectCandidate) -> Dict[str, Measurement]:
    """width/height/depth (AABB extents along x/y/z) + volume for one
    candidate. Keys match the naming convention `evidence/promote_planes.py`
    already uses for entity custom_properties (`*_m`, `*_m3`)."""
    width = candidate.bounds_max.x - candidate.bounds_min.x
    height = candidate.bounds_max.y - candidate.bounds_min.y
    depth = candidate.bounds_max.z - candidate.bounds_min.z
    volume = width * height * depth

    return {
        "width_m": _measurement(width, candidate.confidence),
        "height_m": _measurement(height, candidate.confidence),
        "depth_m": _measurement(depth, candidate.confidence),
        "volume_m3": _measurement(volume, candidate.confidence, unit="meter^3"),
    }


def measure_distance(a: MergedObjectCandidate, b: MergedObjectCandidate) -> Measurement:
    """Centroid-to-centroid distance between two candidates. Confidence
    is the weaker of the two contributing candidates -- a distance is
    only as trustworthy as its least-certain endpoint."""
    dx = a.position.x - b.position.x
    dy = a.position.y - b.position.y
    dz = a.position.z - b.position.z
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    confidence = min(a.confidence, b.confidence)
    return _measurement(distance, confidence)
