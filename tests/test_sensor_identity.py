"""Tests for the P1-02 canonical sensor model: DECLARED sensor identity
(evidence.sensors.SensorDescriptor + <component>/sensor_identity.json),
its attachment to parsed evidence, and its interplay with the P2-01
clock model (clock_id resolution) and Decision 020 depth scale rules.

Spec rule under test: every sensor exposes sensor_id, source_id,
capture_id, clock_id, timestamp, frame, intrinsics, extrinsics, units,
coordinate_system, quality, and provenance -- with no hidden
assumptions. Undeclared is a recorded fact (None), never defaulted;
identity is declared by the capture, never inferred from contents.
"""

import json
import os

import pytest

from evidence.sensors import (
    SensorDescriptor,
    SensorIdentityError,
    attach_sensor_identities,
    load_sensor_identity,
    parse_component_streams,
)
from evidence.multi_source import MultiSourceSession


def _jpeg(seed_byte: int = 0x01) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([seed_byte]) * 64


def _intrinsics_dict() -> dict:
    return {
        "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0,
        "width": 1920, "height": 1080,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    }


def _write_identity(root, component: str, data: dict) -> None:
    comp_dir = os.path.join(str(root), component)
    os.makedirs(comp_dir, exist_ok=True)
    with open(os.path.join(comp_dir, "sensor_identity.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestSensorDescriptor:
    def test_sensor_id_required_and_nonempty(self):
        with pytest.raises(SensorIdentityError):
            SensorDescriptor(sensor_id="")
        with pytest.raises(SensorIdentityError):
            SensorDescriptor.from_dict({}, declared_path="x")

    def test_undeclared_fields_stay_none(self):
        d = SensorDescriptor(sensor_id="s1")
        assert d.source_id is None
        assert d.capture_id is None
        assert d.clock_id is None
        assert d.frame is None
        assert d.coordinate_system is None
        assert d.units == {}
        assert d.quality == {}
        assert d.intrinsics is None
        assert d.extrinsics is None

    def test_roundtrip_preserves_declared_fields(self):
        d = SensorDescriptor(
            sensor_id="imu-main",
            source_id="src-phone-01",
            capture_id="cap-001",
            clock_id="clk-phone",
            frame="body",
            coordinate_system="ENU",
            units={"accel": "m/s^2", "gyro": "rad/s"},
            quality={"range_g": 8, "vod": True},
        )
        restored = SensorDescriptor.from_dict(d.to_dict(), declared_path="roundtrip")
        assert restored.sensor_id == d.sensor_id
        for name in ("source_id", "capture_id", "clock_id", "frame", "coordinate_system"):
            assert getattr(restored, name) == getattr(d, name)
        assert restored.units == d.units
        assert restored.quality == d.quality

    def test_camera_geometry_roundtrips_typed(self):
        from reconstruction.calibration.camera import CameraIntrinsics

        data = {
            "sensor_id": "cam-main",
            "intrinsics": _intrinsics_dict(),
            "extrinsics": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        }
        d = SensorDescriptor.from_dict(data, declared_path="x")
        assert isinstance(d.intrinsics, CameraIntrinsics)
        assert d.extrinsics is not None
        restored = SensorDescriptor.from_dict(d.to_dict(), declared_path="y")
        assert restored.intrinsics.fx == d.intrinsics.fx
        assert restored.extrinsics is not None

    def test_units_and_quality_validated(self):
        with pytest.raises(SensorIdentityError):
            SensorDescriptor(sensor_id="s", units={"accel": ""})
        with pytest.raises(SensorIdentityError):
            SensorDescriptor.from_dict({"sensor_id": "s", "quality": {"bad": [1, 2]}}, declared_path="x")
        with pytest.raises(SensorIdentityError):
            SensorDescriptor.from_dict({"sensor_id": "s", "units": {"a": 3}}, declared_path="x")

    def test_quality_nonfinite_float_rejected(self):
        raw = '{"sensor_id": "s", "quality": {"bad": Infinity}}'
        with pytest.raises(SensorIdentityError):
            SensorDescriptor.from_dict(json.loads(raw), declared_path="x")

    def test_bad_provenance_rejected(self):
        with pytest.raises(SensorIdentityError):
            SensorDescriptor.from_dict({"sensor_id": "s", "provenance": "FABRICATED"}, declared_path="x")


class TestLoadSensorIdentity:
    def test_absent_identity_returns_none(self, tmp_path):
        assert load_sensor_identity(str(tmp_path), "imu") is None

    def test_load_records_declared_path_lineage(self, tmp_path):
        _write_identity(tmp_path, "imu", {"sensor_id": "imu-main", "clock_id": "clk-a"})
        d = load_sensor_identity(str(tmp_path), "imu")
        assert d is not None
        assert d.sensor_id == "imu-main"
        assert d.declared_path.endswith(os.path.join("imu", "sensor_identity.json"))

    def test_kind_mismatch_fails_loudly(self, tmp_path):
        _write_identity(tmp_path, "imu", {"sensor_id": "s", "kind": "gnss"})
        with pytest.raises(SensorIdentityError, match="kind"):
            load_sensor_identity(str(tmp_path), "imu")

    def test_declared_kind_matching_component_is_accepted(self, tmp_path):
        _write_identity(tmp_path, "imu", {"sensor_id": "s", "kind": "imu"})
        d = load_sensor_identity(str(tmp_path), "imu")
        assert d is not None and d.kind == "imu"

    def test_invalid_json_is_an_identity_error(self, tmp_path):
        comp_dir = os.path.join(str(tmp_path), "imu")
        os.makedirs(comp_dir)
        with open(os.path.join(comp_dir, "sensor_identity.json"), "w") as f:
            f.write("{not json")
        with pytest.raises(SensorIdentityError):
            load_sensor_identity(str(tmp_path), "imu")

    def test_invalid_intrinsics_raise_identity_error(self, tmp_path):
        _write_identity(tmp_path, "calibration", {"sensor_id": "s", "intrinsics": {"fx": "oops"}})
        with pytest.raises(SensorIdentityError):
            load_sensor_identity(str(tmp_path), "calibration")


class TestAttachSensorIdentities:
    def _imu_stream(self, tmp_path):
        comp = os.path.join(str(tmp_path), "imu")
        os.makedirs(comp, exist_ok=True)
        log = os.path.join(comp, "log.jsonl")
        with open(log, "w") as f:
            f.write('{"t": 0.0, "accel_mps2": [0,0,9.81], "gyro_rps": [0,0,0]}\n')
        return parse_component_streams([log], "imu")

    def test_attach_reaches_streams_and_samples_immutably(self, tmp_path):
        _write_identity(tmp_path, "imu", {"sensor_id": "imu-9", "clock_id": "clk-a", "frame": "body"})
        streams = self._imu_stream(tmp_path)
        attached = attach_sensor_identities(streams, str(tmp_path), "imu")
        assert attached[0] is not streams[0]  # new objects, input untouched
        assert streams[0].identity is None
        assert attached[0].identity.sensor_id == "imu-9"
        assert attached[0].samples[0].identity is attached[0].identity

    def test_no_identity_returns_same_objects(self, tmp_path):
        streams = self._imu_stream(tmp_path)
        attached = attach_sensor_identities(streams, str(tmp_path), "imu")
        assert attached == streams
        assert attached[0] is streams[0]
        assert attached[0].identity is None


def _build_composite_capture(root) -> str:
    """A real composite layout: photos/ (visual) + imu/gps/depth/
    calibration sidecars, each with a declared identity."""
    root = str(root)
    os.makedirs(os.path.join(root, "photos"))
    with open(os.path.join(root, "photos", "a.jpg"), "wb") as f:
        f.write(_jpeg())
    os.makedirs(os.path.join(root, "imu"))
    with open(os.path.join(root, "imu", "log.jsonl"), "w") as f:
        f.write('{"t": 0.0, "accel_mps2": [0,0,9.81], "gyro_rps": [0,0,0]}\n')
    _write_identity(root, "imu", {
        "sensor_id": "imu-main", "source_id": "src-phone-01", "capture_id": "cap-001",
        "clock_id": "clk-phone", "frame": "body", "coordinate_system": "ENU",
        "units": {"accel": "m/s^2", "gyro": "rad/s"}, "quality": {"range_g": 8},
    })
    os.makedirs(os.path.join(root, "gps"))
    with open(os.path.join(root, "gps", "fix.jsonl"), "w") as f:
        f.write('{"t": 1.0, "lat_deg": 37.0, "lon_deg": -122.0, "alt_m": 15.0}\n')
    _write_identity(root, "gps", {"sensor_id": "gnss-main", "clock_id": "clk-gnss", "coordinate_system": "WGS84"})
    os.makedirs(os.path.join(root, "depth"))
    from PIL import Image

    img = Image.new("I;16", (2, 2))
    img.putdata([2000, 2000, 2000, 0])
    img.save(os.path.join(root, "depth", "f001.png"))
    with open(os.path.join(root, "depth", "manifest.json"), "w") as f:
        json.dump({"depth_scale": 0.001, "units": "millimeter", "invalid_value": 0}, f)
    _write_identity(root, "depth", {"sensor_id": "depth-main", "clock_id": "clk-depth", "frame": "camera", "units": {"depth": "millimeter"}})
    os.makedirs(os.path.join(root, "calibration"))
    with open(os.path.join(root, "calibration", "cam.json"), "w") as f:
        json.dump({"intrinsics": _intrinsics_dict()}, f)
    _write_identity(root, "calibration", {"sensor_id": "cam-main", "capture_id": "cap-001"})
    return root


class TestSessionIdentityAttachment:
    def test_session_attaches_declared_identity_to_every_component(self, tmp_path):
        capture = _build_composite_capture(tmp_path / "capture")
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(capture)
        assert record.status.value == "ingested"

        imu = session.sensor_streams(record.source_id, "imu")
        assert imu[0].identity.sensor_id == "imu-main"
        assert imu[0].identity.capture_id == "cap-001"
        assert imu[0].identity.units["accel"] == "m/s^2"
        assert imu[0].samples[0].identity.clock_id == "clk-phone"

        gnss = session.sensor_streams(record.source_id, "gps")
        assert gnss[0].identity.sensor_id == "gnss-main"
        assert gnss[0].identity.coordinate_system == "WGS84"

        depth = session.depth_frames(record.source_id)
        assert depth[0].identity.sensor_id == "depth-main"
        assert depth[0].identity.frame == "camera"
        assert depth[0].meters_at(0, 0) == 2.0  # scale unaffected by identity

        calib = session.calibrations(record.source_id)
        assert calib[0].identity is not None
        assert calib[0].identity.sensor_id == "cam-main"
        assert calib[0].intrinsics.fx == 1200.0

    def test_undeclared_identity_stays_none(self, tmp_path):
        capture = tmp_path / "plain"
        capture.mkdir()
        (capture / "photos").mkdir()
        (capture / "photos" / "a.jpg").write_bytes(_jpeg())
        (capture / "imu").mkdir()
        (capture / "imu" / "log.jsonl").write_text(
            '{"t": 0.0, "accel_mps2": [0,0,9.81], "gyro_rps": [0,0,0]}\n'
        )
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(capture))
        imu = session.sensor_streams(record.source_id, "imu")
        assert imu[0].identity is None
        assert imu[0].samples[0].identity is None

    def test_synchronized_uses_declared_clock_id(self, tmp_path):
        capture = _build_composite_capture(tmp_path / "capture")
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(capture)
        imu = session.sensor_streams(record.source_id, "imu")[0]
        alignment = imu.synchronized(metadata={"shared_clock": True})
        assert alignment.clock_id == "clk-phone"
        assert alignment.samples[0].global_t == 0.0
        assert alignment.samples[0].original_t == 0.0  # originals retained

    def test_undeclared_clock_degrades_honestly(self, tmp_path):
        capture = tmp_path / "noclock"
        capture.mkdir()
        (capture / "photos").mkdir()
        (capture / "photos" / "a.jpg").write_bytes(_jpeg())
        (capture / "imu").mkdir()
        (capture / "imu" / "log.jsonl").write_text(
            '{"t": 0.0, "accel_mps2": [0,0,9.81], "gyro_rps": [0,0,0]}\n'
        )
        _write_identity(str(capture), "imu", {"sensor_id": "imu-x"})  # no clock_id
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(capture))
        imu = session.sensor_streams(record.source_id, "imu")[0]
        alignment = imu.synchronized()
        assert alignment.clock_id == "<undeclared>"
        assert alignment.diagnostics.method == "none"
        assert alignment.samples[0].global_t is None
        # explicit argument still wins over the sentinel
        explicit = imu.synchronized("clk-explicit", metadata={"shared_clock": True})
        assert explicit.clock_id == "clk-explicit"

    def test_identity_never_substitutes_for_depth_scale(self, tmp_path):
        capture = tmp_path / "noscale"
        capture.mkdir()
        (capture / "photos").mkdir()
        (capture / "photos" / "a.jpg").write_bytes(_jpeg())
        (capture / "depth").mkdir()
        from PIL import Image

        img = Image.new("I;16", (2, 2))
        img.putdata([2000, 2000, 2000, 2000])
        img.save(str(capture / "depth" / "f001.png"))
        _write_identity(str(capture), "depth", {"sensor_id": "depth-main", "units": {"depth": "millimeter"}})
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(capture))
        from evidence.sensors import SensorParseError

        with pytest.raises(Exception) as excinfo:
            session.depth_frames(record.source_id)
        # identity declares units but scale still requires the manifest:
        # no hidden assumptions, ever.
        assert "depth_scale" in str(excinfo.value) or "DepthScaleUnavailable" in type(excinfo.value).__name__

    def test_depth_frame_roundtrip_preserves_identity(self, tmp_path):
        capture = _build_composite_capture(tmp_path / "capture")
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(capture)
        frame = session.depth_frames(record.source_id)[0]
        restored = type(frame).from_dict(frame.to_dict())
        assert restored.identity is not None
        assert restored.identity.sensor_id == "depth-main"
        assert restored.identity.clock_id == "clk-depth"

    def test_session_roundtrip_unaffected_by_identity(self, tmp_path):
        capture = _build_composite_capture(tmp_path / "capture")
        session = MultiSourceSession(session_id="sess1")
        session.add_source(capture)
        restored = MultiSourceSession.from_dict(session.to_dict())
        assert len(restored.sources()) == 1
        # identity is attached on demand from the capture directory,
        # which still exists -- re-attachment works after restore.
        record = restored.sources()[0]
        imu = restored.sensor_streams(record.source_id, "imu")
        assert imu[0].identity.sensor_id == "imu-main"
