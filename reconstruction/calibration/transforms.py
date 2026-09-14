"""Calibration/frame system unification (P2-02): rigid transforms with
explicit frame names, composable transformation chains, GNSS frames
(WGS84 geodetic <-> ECEF <-> ENU), declared sensor rigs, and
reprojection-residual quality statistics.

Builds on what exists rather than replacing it: rotation/translation
math reuses `engine.physics.math3.Quat`/`Vec3` (the same primitives
`camera.py` documents as the repo's one rotation implementation);
camera geometry reuses `CameraIntrinsics`/`CameraExtrinsics`/
`PinholeCamera` verbatim (the same types PR #28's SensorDescriptor
declares). This module adds what the ledger names as open: transform
CHAINS, GNSS FRAMES (ENU/ECEF), and REPROJECTION-RESIDUAL quality
metrics.

CONVENTIONS (the no-hidden-assumptions discipline, stated once):

- `RigidTransform(from_frame, to_frame)`: `p_to = R(p_from) + t`.
  `rotation` maps directions from `from_frame` into `to_frame`;
  `translation` is expressed in `to_frame` coordinates. Frame names are
  explicit and travel with the transform -- composition validates
  connectivity (`A->B` then `B->C` composes; `A->B` then `X->C` raises)
  so a mis-ordered chain fails loudly instead of silently producing
  nonsense geometry.
- GNSS frames: WGS84 geodetic (lat/lon degrees, ellipsoidal height
  meters -- NOT mean sea level) <-> ECEF (meters, Earth-fixed) <-> ENU
  (meters, East-North-Up tangent plane at a declared reference
  origin). ENU is only meaningful WITH its reference origin; every ENU
  function takes the reference explicitly. Height is ellipsoidal
  throughout -- a geoid separation model is out of scope and must be
  applied by a caller who knows their device's datum (recorded, not
  guessed here).
- `SensorRig`: the DECLARED calibration of a multi-sensor rig -- each
  sensor's pose in the rig frame plus (for cameras) intrinsics. Every
  entry records where its calibration came from ("declared" factory
  data, "estimated" by a procedure, "assumed" by a human). Absent
  calibration is recorded as absent -- never invented.
- Reprojection residuals are in PIXELS: ||observed_pixel -
  projected_pixel||. Points that cannot project (behind the camera)
  are COUNTED and reported, never silently dropped and never given a
  fabricated residual.

All types are frozen dataclasses with to_dict/from_dict roundtrips,
matching every other calibration/evidence type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.camera import (
    CameraIntrinsics,
    PinholeCamera,
)


class CalibrationTransformError(ValueError):
    """A transform/chain/rig operation violated the documented frame or
    calibration contract (frame mismatch, unknown sensor, empty
    observations, non-finite geometry)."""


# ---------------------------------------------------------------------------
# WGS84 constants
# ---------------------------------------------------------------------------

#: WGS84 semi-major axis (meters).
_WGS84_A_M = 6378137.0

#: WGS84 flattening (dimensionless).
_WGS84_F = 1.0 / 298.257223563

#: Derived first eccentricity squared.
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

#: Semi-minor axis (meters), for the near-pole height special case.
_WGS84_B_M = _WGS84_A_M * (1.0 - _WGS84_F)

#: Newton iteration cap for ECEF -> geodetic. Converges to well under
#: a millimeter in ~4 iterations for any non-polar point.
_GEODETIC_ITERATIONS = 12


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise CalibrationTransformError(f"{name} must be finite, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Rigid transforms and chains
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidTransform:
    """A rigid body transform between two NAMED frames:
    ``p_to = rotation(p_from) + translation``.

    `rotation` is a unit `Quat` mapping directions from `from_frame`
    into `to_frame`; `translation` is expressed in `to_frame`. Frame
    names are required and non-empty so every transform carries its
    own semantics; `compose`/`inverse` preserve them (an inverse of
    A->B is B->A). Points transform with translation, directions do
    not -- `apply_direction` exists so callers never have to remember
    which is which.
    """

    from_frame: str
    to_frame: str
    rotation: Quat = field(default_factory=Quat.identity)
    translation: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        for name in (self.from_frame, self.to_frame):
            if not isinstance(name, str) or not name:
                raise CalibrationTransformError("frame names must be non-empty strings")
        norm = self.rotation.normalized()
        if (
            abs(norm.w - self.rotation.w) > 1e-9
            or abs(norm.x - self.rotation.x) > 1e-9
            or abs(norm.y - self.rotation.y) > 1e-9
            or abs(norm.z - self.rotation.z) > 1e-9
        ):
            raise CalibrationTransformError("rotation must be a unit quaternion (within 1e-9)")
        for component in (self.translation.x, self.translation.y, self.translation.z):
            _require_finite(component, "translation components")

    def apply(self, point_from: Vec3) -> Vec3:
        """Transform a POINT (translation applies)."""
        return self.rotation.rotate(point_from) + self.translation

    def apply_direction(self, direction_from: Vec3) -> Vec3:
        """Transform a DIRECTION (rotation only -- directions have no
        position, so translation must not apply)."""
        return self.rotation.rotate(direction_from)

    def compose(self, outer: "RigidTransform") -> "RigidTransform":
        """`self` maps A->B and `outer` maps B->C; returns A->C.
        Raises when the frames do not connect -- a mis-ordered chain is
        a real error, not something to silently compute anyway."""
        if outer.from_frame != self.to_frame:
            raise CalibrationTransformError(
                f"cannot compose {self.from_frame}->{self.to_frame} with "
                f"{outer.from_frame}->{outer.to_frame}: outer must start at {self.to_frame!r}"
            )
        return RigidTransform(
            from_frame=self.from_frame,
            to_frame=outer.to_frame,
            rotation=outer.rotation.multiply(self.rotation),
            translation=outer.rotation.rotate(self.translation) + outer.translation,
        )

    def inverse(self) -> "RigidTransform":
        """The exact inverse: A->B becomes B->A (unit-quaternion
        conjugate is the exact inverse rotation)."""
        rotation_inv = self.rotation.conjugate()
        return RigidTransform(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            rotation=rotation_inv,
            translation=rotation_inv.rotate(self.translation * -1.0),
        )

    def to_dict(self) -> dict:
        return {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "rotation": self.rotation.to_dict(),
            "translation": self.translation.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "RigidTransform":
        return RigidTransform(
            from_frame=data["from_frame"],
            to_frame=data["to_frame"],
            rotation=Quat.from_dict(data["rotation"]),
            translation=Vec3.from_dict(data["translation"]),
        )


@dataclass(frozen=True)
class CalibrationChain:
    """An EXPLICIT transformation chain: ordered `RigidTransform`s whose
    frames connect (each step's `to_frame` is the next step's
    `from_frame`). The chain validates connectivity at construction --
    a gap is a construction error, never a silent broken mapping --
    and composes to a single equivalent transform via `total()`.

    This is the contract VIO/registration (P3-01/P4-01) consume: give
    me points in `first.from_frame`, get points in `last.to_frame`,
    with every intermediate frame named and inspectable.
    """

    steps: Tuple[RigidTransform, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise CalibrationTransformError("a calibration chain must contain at least one step")
        for first, second in zip(self.steps, self.steps[1:]):
            if first.to_frame != second.from_frame:
                raise CalibrationTransformError(
                    f"chain gap: {first.from_frame}->{first.to_frame} is not "
                    f"followed by a transform from {first.to_frame!r} "
                    f"(got {second.from_frame!r})"
                )

    @property
    def from_frame(self) -> str:
        return self.steps[0].from_frame

    @property
    def to_frame(self) -> str:
        return self.steps[-1].to_frame

    def total(self) -> RigidTransform:
        """Compose all steps into one equivalent RigidTransform."""
        result = self.steps[0]
        for step in self.steps[1:]:
            result = result.compose(step)
        return result

    def apply(self, point: Vec3) -> Vec3:
        return self.total().apply(point)

    def to_dict(self) -> dict:
        return {"steps": [step.to_dict() for step in self.steps]}

    @staticmethod
    def from_dict(data: dict) -> "CalibrationChain":
        return CalibrationChain(steps=tuple(RigidTransform.from_dict(s) for s in data["steps"]))


# ---------------------------------------------------------------------------
# GNSS frames: WGS84 geodetic <-> ECEF <-> ENU
# ---------------------------------------------------------------------------


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> Vec3:
    """WGS84 geodetic (degrees, ellipsoidal height in meters) -> ECEF
    (meters). Standard closed form with the prime-vertical radius of
    curvature N. Height is ELLIPSOIDAL -- geoid separation is a
    caller concern (recorded, not modeled here)."""
    lat = math.radians(_require_finite(lat_deg, "lat_deg"))
    lon = math.radians(_require_finite(lon_deg, "lon_deg"))
    height = _require_finite(height_m, "height_m")
    if not (-90.0 <= lat_deg <= 90.0):
        raise CalibrationTransformError(f"lat_deg out of range [-90, 90]: {lat_deg!r}")
    sin_lat = math.sin(lat)
    n = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return Vec3(
        (n + height) * math.cos(lat) * math.cos(lon),
        (n + height) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - _WGS84_E2) + height) * sin_lat,
    )


def ecef_to_geodetic(point_ecef: Vec3) -> Tuple[float, float, float]:
    """ECEF (meters) -> WGS84 geodetic (lat_deg, lon_deg, ellipsoidal
    height_m) by Newton iteration on (lat, height); exact for the
    longitude. Near the poles (p_xy ~ 0) the closed-form latitude
    iteration degenerates, so it is special-cased with the
    semi-minor axis. Roundtrips `geodetic_to_ecef` to sub-millimeter
    for realistic capture geometries."""
    for component in (point_ecef.x, point_ecef.y, point_ecef.z):
        _require_finite(component, "ECEF components")
    lon_deg = math.degrees(math.atan2(point_ecef.y, point_ecef.x))
    p_xy = math.hypot(point_ecef.x, point_ecef.y)
    if p_xy < 1e-9:
        # On the rotation axis: latitude is +/-90 and the height is the
        # distance to the semi-minor axis endpoint.
        lat_deg = 90.0 if point_ecef.z >= 0.0 else -90.0
        height = abs(point_ecef.z) - _WGS84_B_M
        return (lat_deg, lon_deg, height)
    lat = math.atan2(point_ecef.z, p_xy)
    height = 0.0
    for _ in range(_GEODETIC_ITERATIONS):
        sin_lat = math.sin(lat)
        n = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        height_new = p_xy / math.cos(lat) - n
        lat_new = math.atan2(point_ecef.z, p_xy * (1.0 - _WGS84_E2 * n / (n + height_new)))
        if abs(lat_new - lat) < 1e-13 and abs(height_new - height) < 1e-9:
            lat, height = lat_new, height_new
            break
        lat, height = lat_new, height_new
    return (math.degrees(lat), lon_deg, height)


def _enu_rotation_matrix(lat_deg: float, lon_deg: float) -> Tuple[Tuple[float, float, float], ...]:
    """Rows of the ECEF->ENU rotation at (lat, lon): row 0 = East unit
    vector in ECEF, row 1 = North, row 2 = Up."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    return (
        (-sin_lon, cos_lon, 0.0),
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
        (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat),
    )


def _quat_from_matrix_rows(rows: Tuple[Tuple[float, float, float], ...]) -> Quat:
    """Quaternion q with q.rotate(v) == M v for the row-major matrix M
    (standard Shepperd conversion). Used to express the ENU rotation as
    the repo's one rotation primitive (`Quat`) instead of introducing a
    second rotation representation."""
    m00, m01, m02 = rows[0]
    m10, m11, m12 = rows[1]
    m20, m21, m22 = rows[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quat(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s).normalized()
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return Quat((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s).normalized()
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return Quat((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s).normalized()
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return Quat((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s).normalized()


def ecef_to_enu_transform(ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> RigidTransform:
    """The RigidTransform mapping ECEF points into the ENU tangent plane
    at the reference origin (a declared GNSS anchor: its geodetic
    coordinates). `from_frame='ecef'`, `to_frame=f'enu@{lat},{lon}'` --
    the frame NAME carries its reference origin, because an ENU frame
    without its origin is meaningless (the no-hidden-assumptions rule
    applied to frame identity itself).

    The desired mapping is ``p_enu = R (p_ecef - ref)``; since
    RigidTransform applies rotation before translation
    (``p_to = R p + t``), the stored translation is the rewritten
    ``-R ref`` -- the transform therefore composes like any other."""
    for value, name in ((ref_lat_deg, "ref_lat_deg"), (ref_lon_deg, "ref_lon_deg"), (ref_alt_m, "ref_alt_m")):
        _require_finite(value, name)
    if not (-90.0 <= ref_lat_deg <= 90.0):
        raise CalibrationTransformError(f"ref_lat_deg out of range [-90, 90]: {ref_lat_deg!r}")
    ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    rotation = _quat_from_matrix_rows(_enu_rotation_matrix(ref_lat_deg, ref_lon_deg))
    return RigidTransform(
        from_frame="ecef",
        to_frame=f"enu@{ref_lat_deg:.7f},{ref_lon_deg:.7f}",
        rotation=rotation,
        translation=rotation.rotate(ref_ecef) * -1.0,
    )


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    height_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Vec3:
    """One geodetic point -> ENU meters relative to the reference
    origin (composes geodetic_to_ecef with the ENU transform)."""
    point_ecef = geodetic_to_ecef(lat_deg, lon_deg, height_m)
    return ecef_to_enu_transform(ref_lat_deg, ref_lon_deg, ref_alt_m).apply(point_ecef)


def enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Tuple[float, float, float]:
    """ENU meters (relative to the reference origin) -> WGS84 geodetic
    (the inverse path: ENU -> ECEF -> geodetic)."""
    transform = ecef_to_enu_transform(ref_lat_deg, ref_lon_deg, ref_alt_m)
    point_ecef = transform.inverse().apply(Vec3(east_m, north_m, up_m))
    return ecef_to_geodetic(point_ecef)


# ---------------------------------------------------------------------------
# Declared sensor rigs
# ---------------------------------------------------------------------------

#: Calibration provenance values. Every rig entry records where its
#: calibration came from; "assumed" exists so a human's guess is
#: LABELED as a guess rather than laundered into factory data.
CALIBRATION_SOURCES = ("declared", "estimated", "assumed")


@dataclass(frozen=True)
class CalibrationEntry:
    """One sensor's declared calibration inside a rig: its pose in the
    rig frame (`T_sensor_rig` maps points FROM the sensor's frame INTO
    the rig frame -- i.e. `from_frame=sensor_id`,
    `to_frame=rig_frame`), optional camera intrinsics, and WHERE the
    calibration came from. A missing pose is recorded as None -- an
    uncalibrated sensor in a rig is a real, reportable state."""

    sensor_id: str
    sensor_type: str  # "camera" | "imu" | "gnss" | "lidar" | "depth" | ...
    transform: Optional[RigidTransform] = None  # sensor frame -> rig frame
    intrinsics: Optional[CameraIntrinsics] = None  # cameras only
    source: str = "declared"  # declared | estimated | assumed
    quality: Dict[str, float] = field(default_factory=dict)  # declared scalars only

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise CalibrationTransformError("sensor_id must be a non-empty string")
        if not isinstance(self.sensor_type, str) or not self.sensor_type:
            raise CalibrationTransformError("sensor_type must be a non-empty string")
        if self.source not in CALIBRATION_SOURCES:
            raise CalibrationTransformError(
                f"source {self.source!r} not in {CALIBRATION_SOURCES} -- "
                "calibration provenance is never unlabeled"
            )
        if self.transform is not None and self.transform.to_frame == self.transform.from_frame:
            raise CalibrationTransformError(
                "a sensor's rig transform cannot map its own frame to itself"
            )
        for key, value in self.quality.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise CalibrationTransformError(f"quality[{key!r}] must be a finite scalar")

    def to_dict(self) -> dict:
        d: dict = {"sensor_id": self.sensor_id, "sensor_type": self.sensor_type, "source": self.source}
        if self.transform is not None:
            d["transform"] = self.transform.to_dict()
        if self.intrinsics is not None:
            d["intrinsics"] = self.intrinsics.to_dict()
        if self.quality:
            d["quality"] = dict(self.quality)
        return d

    @staticmethod
    def from_dict(data: dict) -> "CalibrationEntry":
        return CalibrationEntry(
            sensor_id=data["sensor_id"],
            sensor_type=data["sensor_type"],
            transform=RigidTransform.from_dict(data["transform"]) if data.get("transform") else None,
            intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]) if data.get("intrinsics") else None,
            source=data.get("source", "declared"),
            quality=dict(data.get("quality", {})),
        )


@dataclass(frozen=True)
class SensorRig:
    """A DECLARED multi-sensor rig: every sensor's frame related to one
    rig frame, with explicit calibration provenance. Loaded from a rig
    declaration (JSON, same discipline as sensor_identity.json); the
    rig does not estimate anything -- it records what was declared and
    answers frame-chain questions about it.

    `chain(sensor_id)` returns the explicit CalibrationChain from the
    sensor's frame to the rig frame (single declared hop today; the
    chain type composes multi-hop paths the moment transforms between
    rig frames exist -- the seam P3-01/VIO consumes)."""

    rig_frame: str
    entries: Tuple[CalibrationEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rig_frame, str) or not self.rig_frame:
            raise CalibrationTransformError("rig_frame must be a non-empty string")
        seen: set = set()
        for entry in self.entries:
            if entry.sensor_id in seen:
                raise CalibrationTransformError(f"duplicate sensor_id in rig: {entry.sensor_id!r}")
            seen.add(entry.sensor_id)

    def sensor_ids(self) -> Tuple[str, ...]:
        return tuple(entry.sensor_id for entry in self.entries)

    def entry(self, sensor_id: str) -> CalibrationEntry:
        for entry in self.entries:
            if entry.sensor_id == sensor_id:
                return entry
        raise CalibrationTransformError(
            f"sensor {sensor_id!r} is not part of this rig (members: {list(self.sensor_ids())})"
        )

    def chain(self, sensor_id: str) -> CalibrationChain:
        """Sensor frame -> rig frame, as an explicit chain. Raises for
        an unknown sensor; raises for a rig member with NO declared
        pose (absent calibration is reported, never invented)."""
        entry = self.entry(sensor_id)
        if entry.transform is None:
            raise CalibrationTransformError(
                f"sensor {sensor_id!r} has no declared rig transform -- "
                "uncalibrated is a recorded state, not a guessable one"
            )
        return CalibrationChain(steps=(entry.transform,))

    def to_dict(self) -> dict:
        return {"rig_frame": self.rig_frame, "entries": [entry.to_dict() for entry in self.entries]}

    @staticmethod
    def from_dict(data: dict) -> "SensorRig":
        return SensorRig(
            rig_frame=data["rig_frame"],
            entries=tuple(CalibrationEntry.from_dict(e) for e in data["entries"]),
        )

    @staticmethod
    def from_file(path: str) -> "SensorRig":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise CalibrationTransformError(f"{path}: invalid JSON ({exc})") from exc
        return SensorRig.from_dict(data)


# ---------------------------------------------------------------------------
# Reprojection residual statistics (calibration quality metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReprojectionResidualStats:
    """Pixel-residual statistics for one camera's observations. `n` is
    the number of projected observations; `n_unprojectable` counts
    points the camera cannot see (behind the image plane) -- counted
    and reported, never silently dropped and never assigned a
    fabricated residual. All statistics are over the residual NORM
    (pixels); `mean_u`/`mean_v` are the signed per-axis means (a bias
    indicator). No confidence is attached -- these are measured facts
    of the given observations."""

    camera_id: str
    n: int
    n_unprojectable: int
    mean_px: float
    median_px: float
    p95_px: float
    max_px: float
    mean_u: float
    mean_v: float

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "n": self.n,
            "n_unprojectable": self.n_unprojectable,
            "mean_px": self.mean_px,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "max_px": self.max_px,
            "mean_u": self.mean_u,
            "mean_v": self.mean_v,
        }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile on a pre-sorted sequence: the smallest
    value covering `fraction` of the data (index ceil(f*n)-1).
    Documented nearest-rank, not interpolated -- deterministic and
    dependency-free."""
    n = len(sorted_values)
    rank = max(1, math.ceil(fraction * n))
    return sorted_values[min(rank, n) - 1]


def reprojection_residual_stats(
    camera_id: str,
    camera: PinholeCamera,
    observations: Sequence[Tuple[Vec3, Tuple[float, float]]],
) -> ReprojectionResidualStats:
    """Residual statistics for one camera: for each (world_point,
    observed_pixel) pair, project the point and measure the pixel-space
    residual ||observed - projected||. Points that cannot project
    (behind the camera plane -- `PinholeCamera.project` returns None)
    increment `n_unprojectable`. Raises on an EMPTY observation set
    (statistics over nothing would be fabricated) and on non-finite
    observations."""
    if not observations:
        raise CalibrationTransformError(
            "no observations given: residual statistics over an empty set "
            "do not exist and will not be fabricated"
        )
    norms: list = []
    sum_u = 0.0
    sum_v = 0.0
    n_unprojectable = 0
    for point, (u_obs, v_obs) in observations:
        for value, name in ((point.x, "point.x"), (point.y, "point.y"), (point.z, "point.z")):
            _require_finite(value, name)
        _require_finite(u_obs, "observed u")
        _require_finite(v_obs, "observed v")
        projected = camera.project(point)
        if projected is None:
            n_unprojectable += 1
            continue
        u_proj, v_proj = projected
        du = u_obs - u_proj
        dv = v_obs - v_proj
        norms.append(math.hypot(du, dv))
        sum_u += du
        sum_v += dv
    if not norms:
        raise CalibrationTransformError(
            f"every observation for {camera_id!r} was unprojectable; no residual "
            "statistics exist (the count is reported, statistics are not invented)"
        )
    norms.sort()
    n = len(norms)
    return ReprojectionResidualStats(
        camera_id=camera_id,
        n=n,
        n_unprojectable=n_unprojectable,
        mean_px=sum(norms) / n,
        median_px=norms[n // 2] if n % 2 == 1 else 0.5 * (norms[n // 2 - 1] + norms[n // 2]),
        p95_px=_percentile(norms, 0.95),
        max_px=norms[-1],
        mean_u=sum_u / n,
        mean_v=sum_v / n,
    )


def per_camera_reprojection_residual_stats(
    cameras: Dict[str, PinholeCamera],
    observations_by_camera: Dict[str, Sequence[Tuple[Vec3, Tuple[float, float]]]],
) -> Dict[str, ReprojectionResidualStats]:
    """Residual statistics per camera. Observations for a camera id
    absent from `cameras` raise (an observation without a camera is a
    data error, not a skippable row); a camera with no observations is
    simply absent from the result (nothing was measured)."""
    unknown = set(observations_by_camera) - set(cameras)
    if unknown:
        raise CalibrationTransformError(
            f"observations reference cameras with no calibration: {sorted(unknown)}"
        )
    result: Dict[str, ReprojectionResidualStats] = {}
    for camera_id, camera in cameras.items():
        observations = observations_by_camera.get(camera_id)
        if not observations:
            continue
        result[camera_id] = reprojection_residual_stats(camera_id, camera, observations)
    return result
