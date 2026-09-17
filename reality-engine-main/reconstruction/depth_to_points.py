"""Depth map -> point cloud (convergence/reconstruction-hardening
campaigns, Phase 8: "depth -> point cloud"). Previously flagged as the
concrete next step in both `docs/REAL_CAPTURE_VERTICAL_SLICE_AUDIT.md`
and `docs/RECONSTRUCTION_HARDENING_AUDIT.md`: `PinholeCamera.unproject()`
existed but nothing called it over a `DepthMap`'s pixel grid.

Output type is the existing `reconstruction.backend.interface.
ReconstructedPoint` -- not a new point-cloud type -- so this function's
output composes directly with everything that already consumes camera-
reconstruction points (`perception/geometry/planes.py`'s RANSAC,
`engine/compiler/world_compiler.py`). A depth-derived point and a COLMAP
sparse point are the same currency downstream; only their provenance/
source differs, both already carried on `ReconstructedPoint` via
`source_evidence_ids`.

Honesty rules (matching `perception/instances/lifting.py`'s already-
established pattern for this exact class of operation):
  - A relative (non-metric) depth map is refused (`DepthToPointsError`),
    never silently unprojected as if it were metric.
  - A pixel with non-finite or non-positive depth is skipped, never
    given a fabricated position.
  - `stride` provides simple, honest decimation (take every Nth pixel)
    for downsampling a dense depth map -- this is NOT voxel-grid
    downsampling (which needs spatial binning across possibly-merged
    point clouds, a separate, larger feature); it is named as pixel-
    stride decimation specifically so it is not mistaken for that.
"""

from __future__ import annotations

import math
from typing import List

from perception.depth.interface import DepthMap
from reconstruction.backend.interface import ReconstructedPoint
from reconstruction.calibration.camera import PinholeCamera


class DepthToPointsError(ValueError):
    pass


def depth_map_to_points(depth: DepthMap, camera: PinholeCamera, stride: int = 1) -> List[ReconstructedPoint]:
    """Unproject every valid pixel of `depth` (subsampled by `stride`)
    through `camera` into world-space `ReconstructedPoint`s.

    Raises DepthToPointsError for a non-metric depth map or a non-positive
    stride. Skips (never fabricates) pixels with non-finite/non-positive
    depth. track_id is deterministic (derived from evidence id + pixel
    coordinates), so the same depth map + camera always yields identical
    output.
    """
    if depth.unit != "meters":
        raise DepthToPointsError(
            f"depth map for {depth.evidence_id} is unit={depth.unit!r} (relative), not 'meters' -- "
            "unprojecting a relative depth map would silently invent metric scale"
        )
    if stride < 1:
        raise DepthToPointsError(f"stride must be >= 1, got {stride}")

    points: List[ReconstructedPoint] = []
    for row in range(0, depth.height, stride):
        depth_row = depth.values[row]
        for col in range(0, depth.width, stride):
            d = depth_row[col]
            if not math.isfinite(d) or d <= 0.0:
                continue
            world_point = camera.unproject(col + 0.5, row + 0.5, d)
            points.append(ReconstructedPoint(
                position=world_point.as_tuple(),
                track_id=f"depth-{depth.evidence_id}-{row:05d}-{col:05d}",
                source_evidence_ids=[depth.evidence_id],
                uncertainty=depth.uncertainty,
            ))
    return points
