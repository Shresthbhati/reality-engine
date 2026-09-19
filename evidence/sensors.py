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

DECLARED SENSOR IDENTITY (P1-02 canonical sensor model): every sensor
exposes sensor_id, source_id, capture_id, clock_id, timestamp, frame,
intrinsics, extrinsics, units, coordinate_system, quality, and
provenance -- with no hidden assumptions. Timestamps and provenance
live on the samples and streams; the DECLARED identity half lives in
an optional per-component sidecar, ``<component>/sensor_identity.json``
(named distinctly because depth/ already owns manifest.json for its
scale declaration, Decision 020), loaded via ``load_sensor_identity``
and attached to parsed evidence via ``attach_sensor_identities`` or
the session's sensor_streams/calibrations/depth_frames accessors.
Schema (all keys optional except sensor_id; absent = undeclared =
recorded as None, never defaulted):

    {
        "sensor_id": "imu-main",        // required, non-empty
        "source_id": "src-phone-01",     // owning source (joins P1-01 identity)
        "capture_id": "cap-001",         // acquisition session on the device
        "clock_id": "clk-phone",         // joins evidence.clocks (P2-01)
        "frame": "body",                 // sensor's measurement frame label
        "coordinate_system": "ENU",      // e.g. ENU | ECEF | WGS84
        "units": {"accel": "m/s^2"},    // quantity -> declared unit string
        "quality": {"range_g": 8},      // JSON scalars only, validated
        "intrinsics": {...},             // optional CameraIntrinsics dict
        "extrinsics": {...},             // optional CameraExtrinsics dict
        "provenance": "OBSERVED",        // Provenance enum value
        "kind": "imu"                    // optional; must match its directory
    }

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
- No DEPTH sidecar parsing HERE. Unlike IMU/GNSS/telemetry (small
  numeric records that map cleanly onto one JSON schema), a depth
  SIDECAR is an image-shaped artifact whose format varies by device
  (16-bit PNG vs. raw binary vs. EXR) and whose depth-value-to-meters
  scale factor and invalid-pixel convention are device-specific. It
  now has its own module with that decision made explicitly:
  ``evidence/depth_frames.py`` (16-bit PNG first, per Decision 020 in
  docs/engineering/DESIGN_DECISIONS.md, explicit depth_scale required
  from a depth/manifest.json or the caller -- never guessed), with
  ``MultiSourceSession.depth_frames()`` as its on-demand entry point,
  mirroring sensor_streams()/calibrations() here.

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
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from provenance import Provenance

Vec3 = Tuple[float, float, float]


class SensorParseError(ValueError):
    """A sensor sidecar file violated the documented format."""
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


class SensorIdentityError(SensorParseError):
    """A sensor identity manifest (<component>/manifest.json) violated
    the documented identity schema. Subclasses SensorParseError so
    callers catching parse errors catch identity errors too."""


#: Identity keys a manifest may declare. `sensor_id` is required (an
#: identity that names no sensor is not an identity); everything else
#: is optional and stays None/{} when absent -- undeclared is a
#: recorded fact, never defaulted.
_IDENTITY_REQUIRED_KEY = "sensor_id"
_IDENTITY_STR_KEYS = (
    "sensor_id", "source_id", "capture_id", "clock_id",
    "frame", "coordinate_system", "kind",
)


def _identity_scalar(value: object, key: str, path: str) -> object:
    """Quality-map values must be JSON scalars (bool/int/float/str).
    Nested structures and non-finite floats are rejected: quality is
    DECLARED metadata that must roundtrip losslessly, not a place to
    stash uninterpretable blobs."""
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SensorIdentityError(f"{path}: quality[{key!r}] must be finite, got {value!r}")
        return value
    raise SensorIdentityError(
        f"{path}: quality[{key!r}] must be a JSON scalar (bool/int/float/str), "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class SensorDescriptor:
    """DECLARED sensor identity (P1-02 canonical sensor model).

    Every sensor exposes sensor_id, source_id, capture_id, clock_id,
    timestamp, frame, intrinsics, extrinsics, units,
    coordinate_system, quality, and provenance -- with no hidden
    assumptions. This type is the declared half of that contract: what
    the capture DECLARED about a sensor, loaded from a per-component
    ``<component>/manifest.json`` sidecar. Timestamps are the other
    half and live on the samples themselves (and, synchronized, in
    evidence.clocks.TimeAlignment -- original sensor timestamps are
    never overwritten).

    Honesty rules:
    - ``sensor_id`` is required; everything else is optional. An
      absent field stays None/{} -- "undeclared" is recorded as such,
      never filled with a guess.
    - ``units`` maps a quantity name to its declared unit string
      (e.g. {"accel": "m/s^2"}); the sample field names already carry
      unit-suffixed conventions, this records what the device itself
      declared.
    - ``quality`` values must be JSON scalars (validated): declared
      quality metrics travel losslessly.
    - ``kind`` (when declared) must match the component it is loaded
      for -- an imu/manifest.json declaring kind "gnss" is a real
      capture error and fails loudly.
    - ``intrinsics``/``extrinsics`` reuse the existing, tested
      reconstruction.calibration.camera types verbatim (same rule as
      CalibrationRecord).

    Immutable like every evidence-layer type; attached to parsed
    evidence by the session, never derived from file contents.
    """

    sensor_id: str
    source_id: Optional[str] = None
    capture_id: Optional[str] = None
    clock_id: Optional[str] = None
    frame: Optional[str] = None
    coordinate_system: Optional[str] = None
    units: Dict[str, str] = field(default_factory=dict)
    quality: Dict[str, object] = field(default_factory=dict)
    intrinsics: Optional[object] = None  # reconstruction.calibration.camera.CameraIntrinsics
    extrinsics: Optional[object] = None  # ...CameraExtrinsics, if declared
    provenance: Provenance = Provenance.OBSERVED
    kind: Optional[str] = None  # component this identity declares itself for
    declared_path: str = ""  # where this identity was declared (lineage)

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise SensorIdentityError("sensor_id must be a non-empty string")
        for value in (self.source_id, self.capture_id, self.clock_id, self.frame, self.coordinate_system, self.kind):
            if value is not None and (not isinstance(value, str) or not value):
                raise SensorIdentityError("identity string fields must be non-empty strings or None")
        for key, unit in self.units.items():
            if not isinstance(key, str) or not isinstance(unit, str) or not unit:
                raise SensorIdentityError(f"units[{key!r}] must map to a non-empty unit string")
        for key, value in self.quality.items():
            _identity_scalar(value, key, "SensorDescriptor")

    def units_dict(self) -> Dict[str, str]:
        return dict(self.units)

    def quality_dict(self) -> Dict[str, object]:
        return dict(self.quality)

    def to_dict(self) -> dict:
        d: dict = {"sensor_id": self.sensor_id}
        for name in ("source_id", "capture_id", "clock_id", "frame", "coordinate_system", "kind"):
            value = getattr(self, name)
            if value is not None:
                d[name] = value
        if self.units:
            d["units"] = self.units_dict()
        if self.quality:
            d["quality"] = self.quality_dict()
        if self.intrinsics is not None:
            d["intrinsics"] = self.intrinsics.to_dict()
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics.to_dict()
        d["provenance"] = self.provenance.value
        if self.declared_path:
            d["declared_path"] = self.declared_path
        return d

    @staticmethod
    def from_dict(data: dict, declared_path: str = "") -> "SensorDescriptor":
        if not isinstance(data, dict):
            raise SensorIdentityError(f"{declared_path}: expected a JSON object, got {type(data).__name__}")
        if _IDENTITY_REQUIRED_KEY not in data:
            raise SensorIdentityError(f"{declared_path}: missing required key '{_IDENTITY_REQUIRED_KEY}'")
        kwargs: dict = {}
        for name in _IDENTITY_STR_KEYS:
            if name == _IDENTITY_REQUIRED_KEY:
                continue
            value = data.get(name)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise SensorIdentityError(f"{declared_path}: {name} must be a non-empty string, got {value!r}")
                kwargs[name] = value
        units = data.get("units", {})
        if not isinstance(units, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in units.items()
        ):
            raise SensorIdentityError(f"{declared_path}: 'units' must map string quantity names to non-empty unit strings")
        quality = data.get("quality", {})
        if not isinstance(quality, dict):
            raise SensorIdentityError(f"{declared_path}: 'quality' must be an object")
        quality = {k: _identity_scalar(v, k, declared_path) for k, v in quality.items()}
        try:
            provenance = Provenance(data.get("provenance", Provenance.OBSERVED.value))
        except ValueError as exc:
            raise SensorIdentityError(
                f"{declared_path}: provenance {data.get('provenance')!r} is not a recognized value"
            ) from exc
        # Intrinsics/extrinsics reuse the tested camera types verbatim
        # (same rule as CalibrationRecord) so to_dict/from_dict are
        # losslessly symmetric -- a descriptor with declared camera
        # geometry roundtrips with its geometry intact.
        intrinsics = None
        if data.get("intrinsics") is not None:
            from reconstruction.calibration.camera import CameraIntrinsics, CameraIntrinsicsError

            try:
                intrinsics = CameraIntrinsics.from_dict(data["intrinsics"])
            except (KeyError, TypeError, CameraIntrinsicsError) as exc:
                raise SensorIdentityError(f"{declared_path}: invalid 'intrinsics': {exc}") from exc
        extrinsics = None
        if data.get("extrinsics") is not None:
            from reconstruction.calibration.camera import CameraExtrinsics

            try:
                extrinsics = CameraExtrinsics.from_dict(data["extrinsics"])
            except (KeyError, TypeError) as exc:
                raise SensorIdentityError(f"{declared_path}: invalid 'extrinsics': {exc}") from exc
        return SensorDescriptor(
            sensor_id=data[_IDENTITY_REQUIRED_KEY],
            units=dict(units),
            quality=quality,
            provenance=provenance,
            declared_path=declared_path or str(data.get("declared_path", "")),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            **kwargs,
        )


SENSOR_MANIFEST_NAME = "sensor_identity.json"


def load_sensor_identity(directory: str, component: str) -> Optional[SensorDescriptor]:
    """Load the DECLARED sensor identity for one component from
    ``<directory>/<component>/sensor_identity.json``. Returns None when
    no identity file exists -- undeclared identity is a normal,
    recordable state (the evidence stays parseable, identity stays
    None), not an error. Raises SensorIdentityError when the file
    exists but violates the schema, including a declared `kind` that
    mismatches the component directory it lives in.

    Named sensor_identity.json (not manifest.json) because the depth
    component directory already owns manifest.json for its SCALE
    manifest (Decision 020) -- one name, one meaning.

    This is the P1-02 seam: identity is DECLARED by the capture, never
    inferred from file contents and never defaulted -- the same
    no-hidden-assumptions rule as depth scale (Decision 020) and
    synchronization (P2-01).
    """
    manifest_path = os.path.join(directory, component, SENSOR_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SensorIdentityError(f"{manifest_path}: invalid JSON ({exc})") from exc
    descriptor = SensorDescriptor.from_dict(data, declared_path=manifest_path)
    if descriptor.kind is not None and descriptor.kind != component:
        raise SensorIdentityError(
            f"{manifest_path}: declared kind {descriptor.kind!r} does not match "
            f"the component directory it was declared in ({component!r})"
        )
    return descriptor


@dataclass(frozen=True)
class IMUSample:
    """One IMU reading. Units and frame documented at module level.

    ``identity`` is the DECLARED SensorDescriptor attached by the
    session from the capture's identity sidecar (None when the capture
    declared none -- undeclared is recorded, never defaulted).
    """

    t: float
    accel_mps2: Vec3
    gyro_rps: Vec3
    mag_ut: Optional[Vec3] = None
    identity: Optional["SensorDescriptor"] = None

    def to_dict(self) -> dict:
        d = {"t": self.t, "accel_mps2": list(self.accel_mps2), "gyro_rps": list(self.gyro_rps)}
        if self.mag_ut is not None:
            d["mag_ut"] = list(self.mag_ut)
        if self.identity is not None:
            d["identity"] = self.identity.to_dict()
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
    identity: Optional["SensorDescriptor"] = None  # declared identity; None = undeclared

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
        if self.identity is not None:
            d["identity"] = self.identity.to_dict()
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
    identity: Optional["SensorDescriptor"] = None  # declared identity; None = undeclared

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
        if self.identity is not None:
            d["identity"] = self.identity.to_dict()
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
    identity: Optional["SensorDescriptor"] = None  # declared identity; None = undeclared

    def to_dict(self) -> dict:
        d = {
            "source_path": self.source_path,
            "provenance": self.provenance.value,
            "intrinsics": self.intrinsics.to_dict(),
        }
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics.to_dict()
        if self.identity is not None:
            d["identity"] = self.identity.to_dict()
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
    identity: Optional["SensorDescriptor"] = None  # declared identity; None = undeclared

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
            "identity": self.identity.to_dict() if self.identity is not None else None,
        }

    def synchronized(self, clock_id: Optional[str] = None, metadata: Optional[dict] = None):
        """Map this stream onto the global timeline via evidence.clocks
        (P2.1). Returns a TimeAlignment; the stream itself is NEVER
        modified -- original sensor timestamps are retained on every
        aligned sample (t_global = a * t_sensor + b).

        Clock identity resolution order (P1-02): the explicit
        `clock_id` argument wins; otherwise the stream's DECLARED
        identity clock_id (from the capture's identity sidecar); when
        neither declares a clock, the sentinel "<undeclared>" is used
        and synchronization degrades honestly to UNSYNCHRONIZED
        samples (reasons recorded in diagnostics) -- unless the
        metadata itself declares a shared clock, which needs no clock
        identity. A stream is never silently re-stamped.
        """
        from evidence.clocks import synchronize_stream

        resolved = clock_id or (self.identity.clock_id if self.identity is not None else None) or "<undeclared>"
        return synchronize_stream(self, resolved, metadata)


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


def attach_sensor_identities(
    streams: List[SensorStream],
    source_root: str,
    component: str,
) -> List[SensorStream]:
    """Attach DECLARED sensor identity (P1-02) to parsed streams and
    their samples: load `<source_root>/<component>/sensor_identity.json`
    once and attach it to each stream and every sample. When the
    capture declared no identity, streams return unchanged (identity
    stays None -- undeclared is a recorded fact, not a gap to fill).
    New stream objects only; the input list is never mutated."""
    identity = load_sensor_identity(source_root, component)
    if identity is None:
        return streams
    return [
        replace(
            stream,
            identity=identity,
            samples=tuple(replace(sample, identity=identity) for sample in stream.samples),
        )
        for stream in streams
    ]


def parse_component_calibrations(paths: Sequence[str]) -> List[CalibrationRecord]:
    """Parse every `.json` file in `paths` (calibration component) into
    a real CalibrationRecord. Non-`.json` files are skipped, same
    honest-subset behavior as parse_component_streams. The component's
    identity sidecar (sensor_identity.json, P1-02) is metadata, not
    calibration evidence, and is skipped for the same reason -- it is
    loaded separately via load_sensor_identity."""
    return [
        parse_calibration_json(p)
        for p in paths
        if p.endswith(".json") and os.path.basename(p) != SENSOR_MANIFEST_NAME
    ]


# ------------------------------------------------------------------
# DEPTH: 16-bit grayscale PNG, decoded via stdlib zlib/struct only
# (no PIL/OpenCV dependency -- matches perception/depth/interface.py's
# stated "zero dependencies beyond the stdlib" discipline).
#
# FORMAT: one JSON descriptor per depth frame (not JSONL -- decoding
# an image per line would be absurd, and a depth frame, like a
# calibration, is one artifact):
#
#     {
#         "image": "0001_depth.png",  // required, relative to this
#                                      // JSON file's own directory:
#                                      // MUST be an 8-bit-per-channel-
#                                      // sample grayscale (PNG color
#                                      // type 0), bit depth 16,
#                                      // non-interlaced PNG -- the one
#                                      // concrete convention this
#                                      // module supports (chosen
#                                      // because it needs no
#                                      // third-party decoder: PNG's
#                                      // DEFLATE compression is
#                                      // decodable with stdlib zlib)
#         "evidence_id": "photo-0001",  // required: which RGB evidence
#                                      // this depth corresponds to
#         "scale_to_meters": 0.001,   // OPTIONAL: raw 16-bit sample *
#                                      // this = meters. Omitted/0.0 ->
#                                      // unit stays "relative" (the
#                                      // raw integer values, unscaled)
#         "invalid_value": 0          // OPTIONAL: raw samples equal to
#                                      // this become float('nan') in
#                                      // the output (no-return /
#                                      // invalid pixel convention)
#     }
#
# WHAT THIS DOES NOT SUPPORT (named, not hidden): 8-bit depth PNGs,
# color/RGB-encoded depth, EXR, raw binary dumps, or any vendor-
# specific packed format (e.g. ARKit's disparity encoding). A real
# multi-vendor depth ingestion layer needs per-device adapters behind
# this same DepthMap output type -- not attempted here.
# ------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DepthDecodeError(SensorParseError):
    """The referenced image was not a 16-bit grayscale, non-interlaced
    PNG this module can decode."""


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_png_scanlines(raw: bytes, width: int, height: int, bytes_per_pixel: int) -> bytes:
    """Reverse PNG's per-scanline filtering (spec sec 9), yielding the
    unfiltered pixel bytes (no filter-type bytes)."""
    stride = width * bytes_per_pixel
    out = bytearray(height * stride)
    prev = bytearray(stride)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        cur = bytearray(raw[offset:offset + stride])
        offset += stride
        if len(cur) != stride:
            raise DepthDecodeError(f"truncated scanline {row}: got {len(cur)} bytes, expected {stride}")
        for x in range(stride):
            a = cur[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
            b = prev[x]
            c = prev[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
            if filter_type == 0:
                pass
            elif filter_type == 1:
                cur[x] = (cur[x] + a) & 0xFF
            elif filter_type == 2:
                cur[x] = (cur[x] + b) & 0xFF
            elif filter_type == 3:
                cur[x] = (cur[x] + (a + b) // 2) & 0xFF
            elif filter_type == 4:
                cur[x] = (cur[x] + _paeth_predictor(a, b, c)) & 0xFF
            else:
                raise DepthDecodeError(f"unsupported PNG filter type {filter_type} on scanline {row}")
        out[row * stride:(row + 1) * stride] = cur
        prev = cur
    return bytes(out)


def _decode_16bit_grayscale_png(path: str) -> Tuple[int, int, List[List[int]]]:
    """Decode a 16-bit grayscale, non-interlaced PNG into
    (width, height, values[row][col]) of raw 0-65535 integers. Raises
    DepthDecodeError for any PNG feature this decoder doesn't support
    (wrong color type/bit depth, interlacing, missing IHDR, corrupt
    IDAT stream) -- never guesses or produces a partially-wrong image.
    """
    import struct
    import zlib

    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != _PNG_SIGNATURE:
        raise DepthDecodeError(f"{path}: not a PNG file (bad signature)")

    width = height = bit_depth = color_type = interlace = None
    idat_chunks: List[bytes] = []
    offset = 8
    while offset < len(data):
        if offset + 8 > len(data):
            raise DepthDecodeError(f"{path}: truncated chunk header at offset {offset}")
        length, = struct.unpack(">I", data[offset:offset + 4])
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 8 + length + 4  # + 4 for the CRC we don't verify
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None:
        raise DepthDecodeError(f"{path}: missing IHDR chunk")
    if color_type != 0:
        raise DepthDecodeError(f"{path}: unsupported PNG color type {color_type} (only grayscale/0 supported)")
    if bit_depth != 16:
        raise DepthDecodeError(f"{path}: unsupported PNG bit depth {bit_depth} (only 16-bit supported)")
    if interlace != 0:
        raise DepthDecodeError(f"{path}: interlaced PNGs are not supported")
    if not idat_chunks:
        raise DepthDecodeError(f"{path}: no IDAT chunks (empty image data)")

    try:
        decompressed = zlib.decompress(b"".join(idat_chunks))
    except zlib.error as exc:
        raise DepthDecodeError(f"{path}: corrupt PNG image data ({exc})") from exc

    unfiltered = _unfilter_png_scanlines(decompressed, width, height, bytes_per_pixel=2)

    values: List[List[int]] = []
    for row in range(height):
        row_bytes = unfiltered[row * width * 2:(row + 1) * width * 2]
        values.append(list(struct.unpack(f">{width}H", row_bytes)))
    return width, height, values


def parse_depth_json(path: str) -> "DepthMap":
    """Parse one depth sidecar descriptor (single JSON object, schema
    documented above this function) into a real
    perception.depth.interface.DepthMap. Raises SensorParseError for a
    malformed descriptor, DepthDecodeError for a referenced PNG this
    module cannot decode, OSError if either file cannot be opened.
    """
    from perception.depth.interface import DepthMap
    from provenance import Uncertainty

    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SensorParseError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SensorParseError(f"{path}: expected a JSON object, got {type(data).__name__}")
    image_name = data.get("image")
    if not isinstance(image_name, str) or not image_name:
        raise SensorParseError(f"{path}: missing or invalid required key 'image'")
    evidence_id = data.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise SensorParseError(f"{path}: missing or invalid required key 'evidence_id'")
    scale_to_meters = data.get("scale_to_meters", 0.0)
    if not isinstance(scale_to_meters, (int, float)) or isinstance(scale_to_meters, bool):
        raise SensorParseError(f"{path}: 'scale_to_meters' must be a number")
    invalid_value = data.get("invalid_value")
    if invalid_value is not None and (not isinstance(invalid_value, int) or isinstance(invalid_value, bool)):
        raise SensorParseError(f"{path}: 'invalid_value' must be an integer")

    image_path = os.path.join(os.path.dirname(path), image_name)
    width, height, raw_values = _decode_16bit_grayscale_png(image_path)

    scale = float(scale_to_meters)
    unit = "meters" if scale > 0.0 else "relative"
    rows: List[List[float]] = []
    for row in raw_values:
        rows.append([
            float("nan") if invalid_value is not None and v == invalid_value
            else (v * scale if scale > 0.0 else float(v))
            for v in row
        ])
    return DepthMap(
        evidence_id=evidence_id, width=width, height=height, values=rows, unit=unit,
        uncertainty=Uncertainty(confidence=1.0, note=f"decoded from {image_path}"),
    )


def parse_component_depth_maps(paths: Sequence[str]) -> List["DepthMap"]:
    """Parse every `.json` file in `paths` (depth component) into a
    real DepthMap. Non-`.json` files (the referenced `.png` images
    themselves, if listed alongside their descriptor) are skipped."""
    return [parse_depth_json(p) for p in paths if p.endswith(".json")]
