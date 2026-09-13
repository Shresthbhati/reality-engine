"""2D -> 3D lifting (semantic perception campaign, Phase 4): turn one
`SegmentedRegion` (a 2D mask from any `ISegmentationBackend`) into a
3D object hypothesis, using a `DepthMap` (any `IDepthBackend`) and the
real `PinholeCamera` (reconstruction/calibration/camera.py) that
observed it.

Before this module, nothing in the repo converted a 2D detection/mask
into 3D geometry at all -- `perception/instances/interface.py` only
defines `ITrackBackend`/`InstanceTrack` for 2D cross-frame tracking, and
the segmentation/depth interfaces had no consumer that combined them.
This is real, deterministic geometry composition (unproject every valid-
depth mask pixel, take the resulting point cloud's centroid and AABB) --
not a learned model, and it does not require SAM/torch/any perception
backend to be installed to run or test: it operates on the backend
*output types* (`SegmentedRegion`, `DepthMap`), which can come from a
real backend or a hand-built fixture, same as how
`evidence/promote_planes.py` operates on `ReconstructionResult` without
caring whether COLMAP or a synthetic fixture produced it.

Honesty rules:
  - A relative (non-metric) depth map cannot honestly produce a metric
    3D hypothesis -- `lift_region_to_3d` refuses (`LiftingError`) rather
    than silently treating relative depth values as meters.
  - Too few mask pixels with valid (positive, finite) depth -> returns
    `None`, the expected "insufficient evidence" outcome, not an
    exception and not a fabricated hypothesis from a handful of points.
  - The hypothesis is always `Provenance.INFERRED` (a 2D detection +
    depth reprojected into 3D is an inference, never a direct
    observation) with confidence combining the segmentation model's own
    confidence and the fraction of mask pixels that actually had usable
    depth -- both factors visible in the result, not collapsed silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from provenance import Provenance, Uncertainty
from engine.physics.math3 import Vec3
from perception.depth.interface import DepthMap
from perception.segmentation.interface import SegmentedRegion
from reconstruction.calibration.camera import PinholeCamera

#: Minimum fraction of a mask's pixels that must carry valid (positive,
#: finite) depth for a hypothesis to be honestly produced. Below this,
#: too little of the object was actually observed in depth to trust a
#: 3D extent derived from it.
DEFAULT_MIN_VALID_DEPTH_FRACTION = 0.1


class LiftingError(ValueError):
    """Raised for genuine misuse (mismatched evidence/dimensions, non-
    metric depth) -- never for the honest "not enough evidence" case,
    which returns None instead."""


@dataclass(frozen=True)
class ObjectHypothesis3D:
    """A 3D object candidate lifted from one 2D region -- a hypothesis,
    not a WorldIR entity. Promoting this into WorldIR (entity resolution
    across multiple views/frames, ontology mapping) is a separate,
    later step, matching how `promote_plane_to_entity` sits downstream
    of pure plane detection."""

    region_id: str
    evidence_id: str
    label: str
    position: Vec3  # centroid of the unprojected, valid-depth mask points
    bounds_min: Vec3
    bounds_max: Vec3
    point_count: int
    mask_pixel_count: int
    confidence: float
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    provenance: Provenance = Provenance.INFERRED

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "evidence_id": self.evidence_id,
            "label": self.label,
            "position": self.position.to_dict(),
            "bounds_min": self.bounds_min.to_dict(),
            "bounds_max": self.bounds_max.to_dict(),
            "point_count": self.point_count,
            "mask_pixel_count": self.mask_pixel_count,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_dict(),
            "provenance": self.provenance.value,
        }


def lift_region_to_3d(
    region: SegmentedRegion,
    depth: DepthMap,
    camera: PinholeCamera,
    min_valid_fraction: float = DEFAULT_MIN_VALID_DEPTH_FRACTION,
) -> Optional[ObjectHypothesis3D]:
    """Lift one 2D mask into a 3D object hypothesis, or None when there
    isn't enough valid-depth evidence to honestly produce one.

    Raises LiftingError when the inputs themselves are invalid: mismatched
    evidence_id, mismatched mask/depth-map dimensions, or a non-metric
    (relative) depth map -- these are misuse, not "insufficient evidence".
    """
    if region.evidence_id != depth.evidence_id:
        raise LiftingError(
            f"region.evidence_id ({region.evidence_id!r}) != depth.evidence_id "
            f"({depth.evidence_id!r}) -- refusing to lift a mask against the wrong frame's depth"
        )
    if depth.unit != "meters":
        raise LiftingError(
            f"depth map for {depth.evidence_id} is unit={depth.unit!r} (relative), not 'meters' -- "
            "lifting a relative depth map would silently invent metric scale"
        )
    if len(region.mask) != depth.height or (region.mask and len(region.mask[0]) != depth.width):
        raise LiftingError(
            f"mask dimensions {len(region.mask)}x{len(region.mask[0]) if region.mask else 0} "
            f"do not match depth map dimensions {depth.height}x{depth.width}"
        )

    mask_pixel_count = 0
    points: List[Vec3] = []
    for row in range(depth.height):
        mask_row = region.mask[row]
        depth_row = depth.values[row]
        for col in range(depth.width):
            if not mask_row[col]:
                continue
            mask_pixel_count += 1
            d = depth_row[col]
            if not math.isfinite(d) or d <= 0.0:
                continue
            # Pixel-center convention: pixel (col, row) covers the
            # continuous image-plane square [col, col+1) x [row, row+1).
            points.append(camera.unproject(col + 0.5, row + 0.5, d))

    if mask_pixel_count == 0:
        return None
    valid_fraction = len(points) / mask_pixel_count
    if not points or valid_fraction < min_valid_fraction:
        return None

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    n = len(points)
    centroid = Vec3(sum(xs) / n, sum(ys) / n, sum(zs) / n)
    bounds_min = Vec3(min(xs), min(ys), min(zs))
    bounds_max = Vec3(max(xs), max(ys), max(zs))

    confidence = max(0.0, min(1.0, region.confidence * valid_fraction))

    return ObjectHypothesis3D(
        region_id=region.region_id,
        evidence_id=region.evidence_id,
        label=region.label,
        position=centroid,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        point_count=n,
        mask_pixel_count=mask_pixel_count,
        confidence=confidence,
        uncertainty=Uncertainty(
            confidence=confidence,
            note=f"{n}/{mask_pixel_count} mask pixels had valid depth (segmentation confidence {region.confidence:.2f})",
        ),
    )
