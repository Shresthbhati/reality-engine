"""Epipolar consistency for multi-view object identity (P7-01): given
two calibrated `PinholeCamera`s and a claimed-same-object 2D detection
point in each, check the pair is geometrically consistent with the
cameras' known relative pose -- reusing this repo's own camera model
(`reconstruction/calibration/camera.py`) and quaternion math
(`engine/physics/math3.py`), no new geometry library.

This is `object_resolution.merge_hypotheses`'s missing geometric
discriminator (see `appearance.py` for the color half): two detections
whose *3D reconstructions* happen to land close together (proximity
gate) can still be genuinely different physical points that a pair of
real cameras could never have observed as the same feature -- this
module is the standard calibrated two-view epipolar test for that.

Method: transform each detection's own (undistorted, unprojected)
pixel into a camera-local normalized ray via the camera's own
`project`/`unproject`, so no separate distortion code is needed here;
build the essential matrix E = [t]_x R from the two cameras'
extrinsics; measure the point-to-epipolar-line distance in camera B's
normalized plane, scaled to approximate pixels by camera B's focal
length. Near zero for a true correspondence; large for two points that
don't share the cameras' epipolar geometry.
"""

from __future__ import annotations

import math
from typing import Tuple

from engine.physics.math3 import Vec3
from reconstruction.calibration.camera import PinholeCamera

#: Point-to-epipolar-line distance (approx. pixels, scaled by camera
#: B's focal length) at or under which two detections are treated as
#: geometrically consistent. Not tuned against a real dataset -- same
#: "starting point a caller can override" honesty as
#: object_resolution.DEFAULT_MERGE_DISTANCE_M.
DEFAULT_EPIPOLAR_TOLERANCE_PX = 5.0

Mat3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


def _normalized_ray(camera: PinholeCamera, u: float, v: float) -> Vec3:
    """Pixel (u, v) -> camera-LOCAL normalized point (x, y, 1), reusing
    the camera's own unproject/world_to_camera round trip instead of
    reimplementing distortion handling."""
    world_at_unit_depth = camera.unproject(u, v, depth=1.0)
    return camera.extrinsics.world_to_camera(world_at_unit_depth)


def _relative_rotation(camera_a: PinholeCamera, camera_b: PinholeCamera) -> Mat3:
    """Columns are camera_a-local basis vectors expressed in camera_b's
    local frame (p_b = R_rel @ p_a). Built by rotating each basis vector
    through the two cameras' extrinsics rather than deriving a
    quaternion-multiply formula by hand."""
    rot_a = camera_a.extrinsics.rotation
    rot_b_conj = camera_b.extrinsics.rotation.conjugate()

    def column(basis: Vec3) -> Vec3:
        return rot_b_conj.rotate(rot_a.rotate(basis))

    c0 = column(Vec3(1.0, 0.0, 0.0))
    c1 = column(Vec3(0.0, 1.0, 0.0))
    c2 = column(Vec3(0.0, 0.0, 1.0))
    return (
        (c0.x, c1.x, c2.x),
        (c0.y, c1.y, c2.y),
        (c0.z, c1.z, c2.z),
    )


def _relative_translation(camera_a: PinholeCamera, camera_b: PinholeCamera) -> Vec3:
    """Camera A's position expressed in camera B's local frame minus
    camera B's own local origin -- i.e. t_rel with p_b = R_rel @ p_a + t_rel."""
    delta = camera_a.extrinsics.position - camera_b.extrinsics.position
    return camera_b.extrinsics.rotation.conjugate().rotate(delta)


def _matmul(m: Mat3, v: Vec3) -> Vec3:
    return Vec3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )


def _skew(v: Vec3) -> Mat3:
    return (
        (0.0, -v.z, v.y),
        (v.z, 0.0, -v.x),
        (-v.y, v.x, 0.0),
    )


def _essential_matrix(camera_a: PinholeCamera, camera_b: PinholeCamera) -> Mat3:
    r_rel = _relative_rotation(camera_a, camera_b)
    t_rel = _relative_translation(camera_a, camera_b)
    skew_t = _skew(t_rel)
    # E = [t]_x @ R
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(skew_t[i][k] * r_rel[k][j] for k in range(3)))
        rows.append(tuple(row))
    return tuple(rows)  # type: ignore[return-value]


def epipolar_residual_px(
    camera_a: PinholeCamera,
    pixel_a: Tuple[float, float],
    camera_b: PinholeCamera,
    pixel_b: Tuple[float, float],
) -> float:
    """Point-to-epipolar-line distance for (pixel_a in camera_a's image,
    pixel_b in camera_b's image), approximated to pixel units via camera
    B's focal length. 0.0 for an exact correspondence; grows with how
    far the pair departs from the two cameras' known epipolar geometry."""
    x_a = _normalized_ray(camera_a, *pixel_a)
    x_b = _normalized_ray(camera_b, *pixel_b)

    e = _essential_matrix(camera_a, camera_b)
    line = _matmul(e, x_a)  # epipolar line in camera B's normalized plane: l . x_b = 0
    denom = math.hypot(line.x, line.y)
    if denom < 1e-12:
        # Degenerate (cameras share an optical center along this ray) --
        # cannot honestly measure a distance; treat as maximally
        # inconsistent rather than silently reporting zero residual.
        return math.inf

    normalized_distance = abs(line.x * x_b.x + line.y * x_b.y + line.z) / denom
    focal_scale = 0.5 * (camera_b.intrinsics.fx + camera_b.intrinsics.fy)
    return normalized_distance * focal_scale


def epipolar_consistent(
    camera_a: PinholeCamera,
    pixel_a: Tuple[float, float],
    camera_b: PinholeCamera,
    pixel_b: Tuple[float, float],
    tolerance_px: float = DEFAULT_EPIPOLAR_TOLERANCE_PX,
) -> bool:
    return epipolar_residual_px(camera_a, pixel_a, camera_b, pixel_b) <= tolerance_px
