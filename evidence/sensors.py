"""Real, normalized sensor-stream ingestion for IMU and GNSS sidecar
data (Priority 2 / spec sec 5-8: "recognizing imu/ is not the same
thing as ingesting and using IMU measurements").

``evidence/multi_source.py``'s composite-capture detection already
finds IMU/GPS/DEPTH/CALIBRATION/TELEMETRY sidecar files and records
their paths on a ``SourceRecord`` -- but only as a file manifest
(``components["imu"] -> [list of paths]``), never parsed. This module
is the actual parser: it turns IMU and GNSS sidecar files into real,
typed, unit-explicit sample streams a downstream consumer (time sync,
VIO, registration -- none of which exist yet) can use without having
to write its own file-format parsing first.

FILE FORMAT (the one this repo parses; documented here because there
was no existing convention anywhere in the codebase to match against):

JSON Lines (``.jsonl``) -- one JSON object per line, UTF-8, blank lines
and lines starting with ``#`` ignored. Chosen over CSV/binary because
it is self-describing (field names travel with the data, so a missing
or renamed field fails loudly instead of silently shifting columns)
and trivially streamable/appendable, matching how phone/drone capture
tooling typically logs sensor data incrementally during a capture.

IMU record (every field required unless noted):

    {
        "t": 12.345,                  // seconds, monotonic within the
                                       // file, sensor's own clock --
                                       // NOT yet normalized/synced to
                                       // any other stream (that is
                                       // Priority 3, time sync; not
                                       // done here)
        "accel_mps2": [ax, ay, az],   // specific force, body frame,
                                       // meters/second^2, INCLUDES
                                       // gravity (raw accelerometer
                                       // convention -- a stationary
                                       // sensor on a level surface
                                       // reads ~[0, 0, 9.81] or
                                       // ~[0, 0, -9.81] depending on
                                       // axis convention; this module
                                       // does not resolve which,
                                       // that is the calibration
                                       // subsystem's job)
        "gyro_rps": [gx, gy, gz],     // angular velocity, body frame,
                                       // radians/second
        "mag_ut": [mx, my, mz]        // OPTIONAL: magnetometer,
                                       // body frame, microtesla
    }

GNSS record (every field required unless noted):

    {
        "t": 12.345,                  // seconds, sensor's own clock
        "lat_deg": 37.421,            // WGS84 latitude, degrees
        "lon_deg": -122.084,          // WGS84 longitude, degrees
        "alt_m": 15.2,                // ellipsoidal height, meters
                                       // (NOT mean-sea-level -- caller
                                       // must know which their device
                                       // reports; not resolved here)
        "accuracy_m": 3.0,            // OPTIONAL: horizontal accuracy
                                       // estimate, meters (1-sigma or
                                       // device-defined; not
                                       // standardized across GNSS
                                       // chipsets, so this is recorded
                                       // as reported, not reinterpreted)
        "fix_type": "3d",             // OPTIONAL: "none"|"2d"|"3d"|
                                       // "dgps"|"rtk"; defaults to
                                       // "unknown" if absent
        "satellites": 11,             // OPTIONAL: satellite count
        "velocity_mps": [vx, vy, vz]  // OPTIONAL: ENU velocity,
                                       // meters/second, if the device
                                       // reports it
    }

CALIBRATION record (one JSON object per file, NOT JSONL -- a
calibration is a single artifact, not a time series):

    {
        "intrinsics": {                // required: reuses
                                        // reconstruction.calibration.
                                        // camera.CameraIntrinsics'
                                        // OWN dict shape verbatim
                                        // (fx, fy, cx, cy in pixels;
                                        // width, height in pixels;
                                        // k1,k2,p1,p2,k3 Brown-Conrady
                                        // distortion, dimensionless)
            "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0,
            "width": 1920, "height": 1080,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0
        },
        "extrinsics": {                // OPTIONAL: camera-to-world
                                        // placement, if this
                                        // calibration also recorded a
                                        // known pose (a rig with a
                                        // fixed known mount, say);
                                        // reuses CameraExtrinsics'
                                        // dict shape (position xyz,
                                        // rotation quaternion wxyz)
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        }
    }

TELEMETRY record (JSONL, one record per platform-state sample -- drone
flight-controller logs are the primary source this schema targets;
distinct from GNSS because a telemetry record also carries attitude
and gimbal pointing, not just position):

    {
        "t": 12.345,                   // seconds, sensor clock
        "position": {                  // OPTIONAL: platform GPS fix,
                                        // same fields/units as a GNSS
                                        // record's required fields
            "lat_deg": 37.4, "lon_deg": -122.1, "alt_m": 50.0
        },
        "attitude_deg": [roll, pitch, yaw],  // OPTIONAL: platform body
                                        // attitude, Euler angles,
                                        // degrees, roll-pitch-yaw
                                        // order, right-handed -- ONE
                                        // specific convention, not a
                                        // universal solution (flight
                                        // controllers disagree on
                                        // sign/axis conventions; a
                                        // real consumer must know
                                        // which controller produced
                                        // the log)
        "gimbal_attitude_deg": [pitch, roll, yaw],  // OPTIONAL: camera
                                        // gimbal pointing relative to
                                        // the platform body, degrees
        "battery_pct": 87.5,           // OPTIONAL: 0-100
        "flight_mode": "auto"          // OPTIONAL: free-text, as
                                        // reported by the platform
    }

WHAT THIS MODULE DELIBERATELY DOES NOT DO (named, not hidden):

- No coordinate-frame conversion (WGS84 -> ECEF -> local ENU is a
  separate, explicit CRS layer -- spec sec 8 -- not built here; a
  GNSSSample's lat/lon/alt are exactly what the sidecar file said).
- No time synchronization across streams/devices (spec sec 10 --
  ``t`` is per-file, per-sensor-clock, not globally comparable yet).
- No IMU integration/trajectory estimation (spec sec 7/11 -- this is
  parsing only, not VIO).
- No binary/vendor-specific format support (ROS bags, RealSense,
  proprietary phone logs) -- only the JSONL/JSON conventions
  documented above. A real vendor adapter is future work behind the
  same ``SensorStream``/``CalibrationRecord`` output types.
- No DEPTH sidecar parsing. Unlike IMU/GNSS/telemetry (small numeric
  records that map cleanly onto one JSON schema), a depth SIDECAR is
  an image-shaped artifact whose format varies by device in ways this
  module cannot honestly paper over: 16-bit PNG vs. raw binary vs.
  EXR, and -- critically -- the depth-value-to-meters scale factor and
  invalid-pixel convention are device-specific (ARKit/ARCore/
  RealSense/Kinect all differ). ``perception/depth/interface.py``
  already has a real ``DepthMap`` type and MiDaS-produced depth
  reaches WorldIR through the existing pipeline; a sidecar DEPTH-image
  decoder needs its own device-format decision and is deliberately not
  attempted blind in this pass.

Honest failure: a malformed record (missing required field, wrong
type, non-finite number) raises ``SensorParseError`` naming the file,
line number, and reason -- never silently dropped or coerced. An empty
or all-comment file produces an empty ``SensorStream`` (0 samples),
which is different from a missing/unreadable file (``OSError``, not
swallowed).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from provenance import Provenance

Vec3 = Tuple[float, float, float]


class SensorParseError(ValueError):
    """A sidecar file's content violated the documented JSONL schema.

    Distinct from OSError (file missing/unreadable) and from
    json.JSONDecodeError (malformed JSON syntax, which this also
    reports as a SensorParseError with line context) -- callers that
    want to distinguish "no file" from "bad file" can catch OSError
    separately.
    """


class GNSSFixType(str, Enum):
    NONE = "none"
    FIX_2D = "2d"
    FIX_3D = "3d"
    DGPS = "dgps"
    RTK = "rtk"
    UNKNOWN = "unknown"


def _require_vec3(record: dict, key: str, path: str, line_no: int) -> Vec3:
    value = record.get(key)
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        raise SensorParseError(
            f"{path}:{line_no}: {key!r} must be a 3-element array, got {value!r}"
        )
    out = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise SensorParseError(f"{path}:{line_no}: {key!r} component {component!r} is not numeric")
        f = float(component)
        if not math.isfinite(f):
            raise SensorParseError(f"{path}:{line_no}: {key!r} component {f!r} is not finite")
        out.append(f)
    return (out[0], out[1], out[2])


def _require_float(record: dict, key: str, path: str, line_no: int) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SensorParseError(f"{path}:{line_no}: {key!r} must be a number, got {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise SensorParseError(f"{path}:{line_no}: {key!r} value {f!r} is not finite")
    return f


def _optional_vec3(record: dict, key: str, path: str, line_no: int) -> Optional[Vec3]:
    if key not in record or record[key] is None:
        return None
    return _require_vec3(record, key, path, line_no)


def _iter_jsonl_records(path: str):
    """Yield (line_no, dict) for every non-blank, non-comment line.

    Raises SensorParseError with file+line context on malformed JSON
    or a line that doesn't decode to a JSON object.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SensorParseError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise SensorParseError(f"{path}:{line_no}: expected a JSON object, got {type(record).__name__}")
            yield line_no, record


@dataclass(frozen=True)
class IMUSample:
    """One IMU reading. Units and frame documented at module level."""

    t: float
    accel_mps2: Vec3
    gyro_rps: Vec3
    mag_ut: Optional[Vec3] = None

    def to_dict(self) -> dict:
        d = {"t": self.t, "accel_mps2": list(self.accel_mps2), "gyro_rps": list(self.gyro_rps)}
        if self.mag_ut is not None:
            d["mag_ut"] = list(self.mag_ut)
        return d


@dataclass(frozen=True)
class GNSSSample:
    """One GNSS fix. Units and frame documented at module level."""

    t: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    accuracy_m: Optional[float] = None
    fix_type: GNSSFixType = GNSSFixType.UNKNOWN
    satellites: Optional[int] = None
    velocity_mps: Optional[Vec3] = None

    def to_dict(self) -> dict:
        d = {
            "t": self.t, "lat_deg": self.lat_deg, "lon_deg": self.lon_deg, "alt_m": self.alt_m,
            "fix_type": self.fix_type.value,
        }
        if self.accuracy_m is not None:
            d["accuracy_m"] = self.accuracy_m
        if self.satellites is not None:
            d["satellites"] = self.satellites
        if self.velocity_mps is not None:
            d["velocity_mps"] = list(self.velocity_mps)
        return d


@dataclass(frozen=True)
class TelemetrySample:
    """One platform-telemetry sample. Every field but `t` is optional
    since a real flight-controller log may report any subset (some
    platforms omit gimbal state, some omit GPS on an indoor flight).
    Units, frame, and the specific Euler convention are documented at
    module level."""

    t: float
    position: Optional["GNSSPosition"] = None
    attitude_deg: Optional[Vec3] = None
    gimbal_attitude_deg: Optional[Vec3] = None
    battery_pct: Optional[float] = None
    flight_mode: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"t": self.t}
        if self.position is not None:
            d["position"] = self.position.to_dict()
        if self.attitude_deg is not None:
            d["attitude_deg"] = list(self.attitude_deg)
        if self.gimbal_attitude_deg is not None:
            d["gimbal_attitude_deg"] = list(self.gimbal_attitude_deg)
        if self.battery_pct is not None:
            d["battery_pct"] = self.battery_pct
        if self.flight_mode is not None:
            d["flight_mode"] = self.flight_mode
        return d


@dataclass(frozen=True)
class GNSSPosition:
    """A bare lat/lon/alt fix embedded in a TelemetrySample -- same
    fields/units as GNSSSample's required fields, but without the
    fix-quality metadata a dedicated GNSS stream carries (a telemetry
    log's embedded GPS field is rarely as detailed as a real GNSS
    receiver's own log)."""

    lat_deg: float
    lon_deg: float
    alt_m: float

    def to_dict(self) -> dict:
        return {"lat_deg": self.lat_deg, "lon_deg": self.lon_deg, "alt_m": self.alt_m}


@dataclass(frozen=True)
class CalibrationRecord:
    """One camera calibration, reusing the existing, tested
    reconstruction.calibration.camera types verbatim rather than
    reinventing intrinsics/extrinsics -- a calibration sidecar file is
    just those types' own JSON shape on disk."""

    source_path: str
    intrinsics: object  # reconstruction.calibration.camera.CameraIntrinsics
    extrinsics: Optional[object] = None  # ...CameraExtrinsics, if present
    provenance: Provenance = Provenance.OBSERVED

    def to_dict(self) -> dict:
        d = {
            "source_path": self.source_path,
            "provenance": self.provenance.value,
            "intrinsics": self.intrinsics.to_dict(),
        }
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics.to_dict()
        return d


@dataclass(frozen=True)
class SensorStream:
    """A parsed, normalized run of samples from one sensor sidecar file.

    `kind` is "imu" or "gnss" (matches evidence.multi_source
    CaptureComponent values so a caller can key results the same way).
    `samples` is exactly the parse order (== file order); IMU/GNSS
    sidecar files are not assumed sorted by `t`, and this module does
    not silently reorder them -- reordering would hide a real logging
    fault (e.g. clock reset) that time-sync diagnostics need to see.
    """

    kind: str
    source_path: str
    samples: Tuple[object, ...] = field(default_factory=tuple)
    provenance: Provenance = Provenance.OBSERVED

    def __len__(self) -> int:
        return len(self.samples)

    def is_monotonic(self) -> bool:
        """True if every sample's `t` is >= the previous one's. A
        diagnostic, not enforced at parse time (a non-monotonic file is
        real data about a real logging fault, not a parse error)."""
        ts = [s.t for s in self.samples]
        return all(a <= b for a, b in zip(ts, ts[1:]))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "source_path": self.source_path,
            "provenance": self.provenance.value,
            "samples": [s.to_dict() for s in self.samples],
        }


def parse_imu_jsonl(path: str) -> SensorStream:
    """Parse one IMU sidecar file (JSONL, schema documented at module
    level) into a real SensorStream of IMUSample. Raises
    SensorParseError on any malformed record; raises OSError if the
    file cannot be opened."""
    samples: List[IMUSample] = []
    for line_no, record in _iter_jsonl_records(path):
        samples.append(IMUSample(
            t=_require_float(record, "t", path, line_no),
            accel_mps2=_require_vec3(record, "accel_mps2", path, line_no),
            gyro_rps=_require_vec3(record, "gyro_rps", path, line_no),
            mag_ut=_optional_vec3(record, "mag_ut", path, line_no),
        ))
    return SensorStream(kind="imu", source_path=path, samples=tuple(samples))


def parse_gnss_jsonl(path: str) -> SensorStream:
    """Parse one GNSS sidecar file (JSONL, schema documented at module
    level) into a real SensorStream of GNSSSample. Raises
    SensorParseError on any malformed record; raises OSError if the
    file cannot be opened."""
    samples: List[GNSSSample] = []
    for line_no, record in _iter_jsonl_records(path):
        lat = _require_float(record, "lat_deg", path, line_no)
        if not (-90.0 <= lat <= 90.0):
            raise SensorParseError(f"{path}:{line_no}: lat_deg {lat!r} out of range [-90, 90]")
        lon = _require_float(record, "lon_deg", path, line_no)
        if not (-180.0 <= lon <= 180.0):
            raise SensorParseError(f"{path}:{line_no}: lon_deg {lon!r} out of range [-180, 180]")
        fix_raw = record.get("fix_type", "unknown")
        try:
            fix_type = GNSSFixType(fix_raw)
        except ValueError:
            raise SensorParseError(f"{path}:{line_no}: fix_type {fix_raw!r} is not a recognized value")
        satellites = record.get("satellites")
        if satellites is not None and (not isinstance(satellites, int) or isinstance(satellites, bool) or satellites < 0):
            raise SensorParseError(f"{path}:{line_no}: satellites {satellites!r} must be a non-negative integer")
        accuracy = record.get("accuracy_m")
        if accuracy is not None:
            accuracy = _require_float(record, "accuracy_m", path, line_no)
        samples.append(GNSSSample(
            t=_require_float(record, "t", path, line_no),
            lat_deg=lat,
            lon_deg=lon,
            alt_m=_require_float(record, "alt_m", path, line_no),
            accuracy_m=accuracy,
            fix_type=fix_type,
            satellites=satellites,
            velocity_mps=_optional_vec3(record, "velocity_mps", path, line_no),
        ))
    return SensorStream(kind="gnss", source_path=path, samples=tuple(samples))


def parse_telemetry_jsonl(path: str) -> SensorStream:
    """Parse one telemetry sidecar file (JSONL, schema documented at
    module level) into a real SensorStream of TelemetrySample. Raises
    SensorParseError on any malformed record; raises OSError if the
    file cannot be opened. Every field but `t` is optional (see
    TelemetrySample), so this validates only the fields actually
    present in each record."""
    samples: List[TelemetrySample] = []
    for line_no, record in _iter_jsonl_records(path):
        position = None
        raw_position = record.get("position")
        if raw_position is not None:
            if not isinstance(raw_position, dict):
                raise SensorParseError(f"{path}:{line_no}: 'position' must be an object, got {raw_position!r}")
            lat = _require_float(raw_position, "lat_deg", path, line_no)
            if not (-90.0 <= lat <= 90.0):
                raise SensorParseError(f"{path}:{line_no}: position.lat_deg {lat!r} out of range [-90, 90]")
            lon = _require_float(raw_position, "lon_deg", path, line_no)
            if not (-180.0 <= lon <= 180.0):
                raise SensorParseError(f"{path}:{line_no}: position.lon_deg {lon!r} out of range [-180, 180]")
            position = GNSSPosition(
                lat_deg=lat, lon_deg=lon,
                alt_m=_require_float(raw_position, "alt_m", path, line_no),
            )
        battery = record.get("battery_pct")
        if battery is not None:
            battery = _require_float(record, "battery_pct", path, line_no)
            if not (0.0 <= battery <= 100.0):
                raise SensorParseError(f"{path}:{line_no}: battery_pct {battery!r} out of range [0, 100]")
        flight_mode = record.get("flight_mode")
        if flight_mode is not None and not isinstance(flight_mode, str):
            raise SensorParseError(f"{path}:{line_no}: flight_mode must be a string, got {flight_mode!r}")
        samples.append(TelemetrySample(
            t=_require_float(record, "t", path, line_no),
            position=position,
            attitude_deg=_optional_vec3(record, "attitude_deg", path, line_no),
            gimbal_attitude_deg=_optional_vec3(record, "gimbal_attitude_deg", path, line_no),
            battery_pct=battery,
            flight_mode=flight_mode,
        ))
    return SensorStream(kind="telemetry", source_path=path, samples=tuple(samples))


def parse_calibration_json(path: str) -> CalibrationRecord:
    """Parse one calibration sidecar file (single JSON object, schema
    documented at module level) into a real CalibrationRecord, reusing
    reconstruction.calibration.camera.CameraIntrinsics/CameraExtrinsics
    for the actual validation (positive focal lengths, finite values,
    positive image dimensions -- their own __post_init__ already
    enforces this; not reimplemented here). Raises SensorParseError on
    any malformed content; raises OSError if the file cannot be
    opened."""
    from reconstruction.calibration.camera import (
        CameraExtrinsics,
        CameraIntrinsics,
        CameraIntrinsicsError,
    )

    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SensorParseError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SensorParseError(f"{path}: expected a JSON object, got {type(data).__name__}")
    if "intrinsics" not in data:
        raise SensorParseError(f"{path}: missing required key 'intrinsics'")
    try:
        intrinsics = CameraIntrinsics.from_dict(data["intrinsics"])
    except (KeyError, TypeError, CameraIntrinsicsError) as exc:
        raise SensorParseError(f"{path}: invalid 'intrinsics': {exc}") from exc
    extrinsics = None
    if data.get("extrinsics") is not None:
        try:
            extrinsics = CameraExtrinsics.from_dict(data["extrinsics"])
        except (KeyError, TypeError) as exc:
            raise SensorParseError(f"{path}: invalid 'extrinsics': {exc}") from exc
    return CalibrationRecord(source_path=path, intrinsics=intrinsics, extrinsics=extrinsics)


#: Parser dispatch keyed by evidence.multi_source.CaptureComponent value
#: ("imu"/"gps"/"telemetry") -- a caller resolving a SourceRecord's
#: `components` dict can look up the right parser without an if/elif
#: chain. Calibration is intentionally NOT here: it parses to a single
#: CalibrationRecord per file, not a SensorStream of samples, so it has
#: its own dispatch function (parse_component_calibrations) below.
PARSERS_BY_COMPONENT = {
    "imu": parse_imu_jsonl,
    "gps": parse_gnss_jsonl,
    "telemetry": parse_telemetry_jsonl,
}


def parse_component_streams(paths: Sequence[str], component: str) -> List[SensorStream]:
    """Parse every `.jsonl` file in `paths` for the given component
    ("imu"/"gps"/"telemetry"); non-`.jsonl` files are skipped (recorded
    nowhere -- caller sees a shorter result list than `paths`, honest
    about what was actually parseable) rather than raising, since a
    sidecar directory may legitimately mix a manufacturer's raw log
    alongside a `.jsonl` export this module understands.
    """
    parser = PARSERS_BY_COMPONENT.get(component)
    if parser is None:
        raise ValueError(f"no parser registered for component {component!r}")
    return [parser(p) for p in paths if p.endswith(".jsonl")]


def parse_component_calibrations(paths: Sequence[str]) -> List[CalibrationRecord]:
    """Parse every `.json` file in `paths` (calibration component) into
    a real CalibrationRecord. Non-`.json` files are skipped, same
    honest-subset behavior as parse_component_streams."""
    return [parse_calibration_json(p) for p in paths if p.endswith(".json")]
