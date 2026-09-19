"""Tests for evidence/sensors.py (real IMU/GNSS sidecar parsing,
Priority 2 / spec sec 5-8) and its wiring into
evidence.multi_source.MultiSourceSession.sensor_streams().
"""

from __future__ import annotations

import json
import struct
import zlib
from binascii import crc32

import pytest

from evidence.sensors import (
    DepthDecodeError,
    GNSSFixType,
    SensorParseError,
    parse_calibration_json,
    parse_component_calibrations,
    parse_component_depth_maps,
    parse_component_streams,
    parse_depth_json,
    parse_gnss_jsonl,
    parse_imu_jsonl,
    parse_telemetry_jsonl,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc32(chunk_type + data))


def _encode_16bit_grayscale_png(values):
    """Minimal, dependency-free encoder for test fixtures: filter type
    0 (None) on every scanline -- valid PNG, decodable by this repo's
    own _decode_16bit_grayscale_png without needing Pillow installed."""
    height = len(values)
    width = len(values[0]) if values else 0
    raw = bytearray()
    for row in values:
        raw.append(0)  # filter type 0
        for v in row:
            raw += struct.pack(">H", v)
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


class TestIMUParsing:
    def test_parses_required_fields(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [
            {"t": 0.0, "accel_mps2": [0.0, 0.0, 9.81], "gyro_rps": [0.0, 0.0, 0.0]},
            {"t": 0.01, "accel_mps2": [0.1, 0.0, 9.80], "gyro_rps": [0.01, 0.0, 0.0]},
        ])

        stream = parse_imu_jsonl(str(path))

        assert stream.kind == "imu"
        assert len(stream) == 2
        assert stream.samples[0].t == 0.0
        assert stream.samples[0].accel_mps2 == (0.0, 0.0, 9.81)
        assert stream.samples[1].gyro_rps == (0.01, 0.0, 0.0)
        assert stream.is_monotonic() is True

    def test_optional_magnetometer_is_none_when_absent(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [{"t": 0.0, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]}])

        stream = parse_imu_jsonl(str(path))

        assert stream.samples[0].mag_ut is None

    def test_magnetometer_parsed_when_present(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [{
            "t": 0.0, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0],
            "mag_ut": [12.5, -3.2, 45.0],
        }])

        stream = parse_imu_jsonl(str(path))

        assert stream.samples[0].mag_ut == (12.5, -3.2, 45.0)

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        path.write_text(
            "# imu log\n\n"
            + json.dumps({"t": 0.0, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]})
            + "\n\n"
        )

        stream = parse_imu_jsonl(str(path))

        assert len(stream) == 1

    def test_empty_file_yields_zero_samples_not_an_error(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        path.write_text("")

        stream = parse_imu_jsonl(str(path))

        assert len(stream) == 0

    def test_missing_required_field_raises_with_file_and_line(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [{"t": 0.0, "accel_mps2": [0, 0, 9.81]}])  # gyro_rps missing

        with pytest.raises(SensorParseError, match=r"imu\.jsonl:1.*gyro_rps"):
            parse_imu_jsonl(str(path))

    def test_wrong_length_vector_raises(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [{"t": 0.0, "accel_mps2": [0, 0], "gyro_rps": [0, 0, 0]}])

        with pytest.raises(SensorParseError, match="3-element array"):
            parse_imu_jsonl(str(path))

    def test_non_finite_value_raises(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        path.write_text('{"t": 0.0, "accel_mps2": [0, 0, NaN], "gyro_rps": [0, 0, 0]}\n')

        with pytest.raises(SensorParseError, match="not finite"):
            parse_imu_jsonl(str(path))

    def test_malformed_json_line_raises_with_line_number(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        path.write_text('{"t": 0.0, "accel_mps2": [0, 0, 9.81]\n')  # truncated

        with pytest.raises(SensorParseError, match=r"imu\.jsonl:1"):
            parse_imu_jsonl(str(path))

    def test_missing_file_raises_oserror_not_sensorparseerror(self, tmp_path):
        with pytest.raises(OSError):
            parse_imu_jsonl(str(tmp_path / "nope.jsonl"))

    def test_non_monotonic_timestamps_are_a_diagnostic_not_a_parse_error(self, tmp_path):
        path = tmp_path / "imu.jsonl"
        _write_jsonl(path, [
            {"t": 1.0, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]},
            {"t": 0.5, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]},
        ])

        stream = parse_imu_jsonl(str(path))  # must not raise

        assert stream.is_monotonic() is False


class TestGNSSParsing:
    def test_parses_required_and_optional_fields(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{
            "t": 0.0, "lat_deg": 37.4219, "lon_deg": -122.084, "alt_m": 15.2,
            "accuracy_m": 3.0, "fix_type": "3d", "satellites": 11,
            "velocity_mps": [0.1, 0.2, 0.0],
        }])

        stream = parse_gnss_jsonl(str(path))

        sample = stream.samples[0]
        assert sample.lat_deg == 37.4219
        assert sample.fix_type is GNSSFixType.FIX_3D
        assert sample.satellites == 11
        assert sample.velocity_mps == (0.1, 0.2, 0.0)

    def test_optional_fields_default_sensibly(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 0.0, "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0}])

        sample = parse_gnss_jsonl(str(path)).samples[0]

        assert sample.accuracy_m is None
        assert sample.fix_type is GNSSFixType.UNKNOWN
        assert sample.satellites is None
        assert sample.velocity_mps is None

    def test_latitude_out_of_range_raises(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 0.0, "lat_deg": 91.0, "lon_deg": 0.0, "alt_m": 0.0}])

        with pytest.raises(SensorParseError, match="out of range"):
            parse_gnss_jsonl(str(path))

    def test_longitude_out_of_range_raises(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 0.0, "lat_deg": 0.0, "lon_deg": 200.0, "alt_m": 0.0}])

        with pytest.raises(SensorParseError, match="out of range"):
            parse_gnss_jsonl(str(path))

    def test_unrecognized_fix_type_raises(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 0.0, "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "fix_type": "quantum"}])

        with pytest.raises(SensorParseError, match="fix_type"):
            parse_gnss_jsonl(str(path))

    def test_negative_satellite_count_raises(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 0.0, "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "satellites": -1}])

        with pytest.raises(SensorParseError, match="satellites"):
            parse_gnss_jsonl(str(path))


class TestTelemetryParsing:
    def test_parses_full_record(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        _write_jsonl(path, [{
            "t": 0.0,
            "position": {"lat_deg": 37.4, "lon_deg": -122.1, "alt_m": 50.0},
            "attitude_deg": [1.0, 2.0, 3.0],
            "gimbal_attitude_deg": [-45.0, 0.0, 0.0],
            "battery_pct": 87.5,
            "flight_mode": "auto",
        }])

        stream = parse_telemetry_jsonl(str(path))

        sample = stream.samples[0]
        assert sample.position.lat_deg == 37.4
        assert sample.attitude_deg == (1.0, 2.0, 3.0)
        assert sample.gimbal_attitude_deg == (-45.0, 0.0, 0.0)
        assert sample.battery_pct == 87.5
        assert sample.flight_mode == "auto"

    def test_all_optional_fields_absent_is_valid(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        _write_jsonl(path, [{"t": 0.0}])

        sample = parse_telemetry_jsonl(str(path)).samples[0]

        assert sample.position is None
        assert sample.attitude_deg is None
        assert sample.gimbal_attitude_deg is None
        assert sample.battery_pct is None
        assert sample.flight_mode is None

    def test_position_out_of_range_raises(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        _write_jsonl(path, [{"t": 0.0, "position": {"lat_deg": 200.0, "lon_deg": 0.0, "alt_m": 0.0}}])

        with pytest.raises(SensorParseError, match="out of range"):
            parse_telemetry_jsonl(str(path))

    def test_battery_out_of_range_raises(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        _write_jsonl(path, [{"t": 0.0, "battery_pct": 150.0}])

        with pytest.raises(SensorParseError, match="battery_pct"):
            parse_telemetry_jsonl(str(path))

    def test_non_string_flight_mode_raises(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        _write_jsonl(path, [{"t": 0.0, "flight_mode": 5}])

        with pytest.raises(SensorParseError, match="flight_mode"):
            parse_telemetry_jsonl(str(path))


class TestCalibrationParsing:
    def _write_calibration(self, path, extrinsics=None):
        data = {"intrinsics": {
            "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0,
            "width": 1920, "height": 1080,
        }}
        if extrinsics is not None:
            data["extrinsics"] = extrinsics
        path.write_text(json.dumps(data))

    def test_parses_intrinsics_only(self, tmp_path):
        path = tmp_path / "calib.json"
        self._write_calibration(path)

        record = parse_calibration_json(str(path))

        assert record.intrinsics.fx == 1200.0
        assert record.intrinsics.width == 1920
        assert record.extrinsics is None

    def test_parses_intrinsics_and_extrinsics(self, tmp_path):
        path = tmp_path / "calib.json"
        self._write_calibration(path, extrinsics={
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        })

        record = parse_calibration_json(str(path))

        assert record.extrinsics is not None
        assert record.extrinsics.position.x == 1.0

    def test_missing_intrinsics_key_raises(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"not_intrinsics": {}}))

        with pytest.raises(SensorParseError, match="intrinsics"):
            parse_calibration_json(str(path))

    def test_invalid_intrinsics_values_raise(self, tmp_path):
        """Reuses CameraIntrinsics' OWN validation (fx must be > 0) --
        proves this module doesn't reimplement or weaken it."""
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"intrinsics": {
            "fx": -1.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0,
            "width": 1920, "height": 1080,
        }}))

        with pytest.raises(SensorParseError, match="invalid 'intrinsics'"):
            parse_calibration_json(str(path))

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text("{not valid json")

        with pytest.raises(SensorParseError, match="invalid JSON"):
            parse_calibration_json(str(path))

    def test_component_calibrations_skips_non_json_files(self, tmp_path):
        json_path = tmp_path / "calib.json"
        self._write_calibration(json_path)
        other_path = tmp_path / "notes.txt"
        other_path.write_text("not calibration")

        records = parse_component_calibrations([str(json_path), str(other_path)])

        assert len(records) == 1


class TestDepthParsing:
    def _write_descriptor(self, dir_path, image_name="depth.png", **overrides):
        data = {"image": image_name, "evidence_id": "photo-0001"}
        data.update(overrides)
        (dir_path / "depth.json").write_text(json.dumps(data))
        return dir_path / "depth.json"

    def test_decodes_a_hand_encoded_png_exactly(self, tmp_path):
        values = [[0, 1000, 65535], [100, 200, 300]]
        (tmp_path / "depth.png").write_bytes(_encode_16bit_grayscale_png(values))
        descriptor = self._write_descriptor(tmp_path)

        depth_map = parse_depth_json(str(descriptor))

        assert depth_map.width == 3
        assert depth_map.height == 2
        assert depth_map.evidence_id == "photo-0001"
        assert depth_map.unit == "relative"
        assert depth_map.values == [[0.0, 1000.0, 65535.0], [100.0, 200.0, 300.0]]

    def test_scale_to_meters_converts_and_sets_unit(self, tmp_path):
        values = [[1000, 2000]]
        (tmp_path / "depth.png").write_bytes(_encode_16bit_grayscale_png(values))
        descriptor = self._write_descriptor(tmp_path, scale_to_meters=0.001)

        depth_map = parse_depth_json(str(descriptor))

        assert depth_map.unit == "meters"
        assert depth_map.values == [[1.0, 2.0]]

    def test_invalid_value_becomes_nan(self, tmp_path):
        import math

        values = [[0, 500, 0]]
        (tmp_path / "depth.png").write_bytes(_encode_16bit_grayscale_png(values))
        descriptor = self._write_descriptor(tmp_path, invalid_value=0)

        depth_map = parse_depth_json(str(descriptor))

        assert math.isnan(depth_map.values[0][0])
        assert depth_map.values[0][1] == 500.0
        assert math.isnan(depth_map.values[0][2])

    def test_missing_image_key_raises(self, tmp_path):
        (tmp_path / "depth.json").write_text(json.dumps({"evidence_id": "photo-0001"}))

        with pytest.raises(SensorParseError, match="image"):
            parse_depth_json(str(tmp_path / "depth.json"))

    def test_missing_evidence_id_raises(self, tmp_path):
        (tmp_path / "depth.png").write_bytes(_encode_16bit_grayscale_png([[0]]))
        (tmp_path / "depth.json").write_text(json.dumps({"image": "depth.png"}))

        with pytest.raises(SensorParseError, match="evidence_id"):
            parse_depth_json(str(tmp_path / "depth.json"))

    def test_referenced_image_missing_raises_oserror(self, tmp_path):
        descriptor = self._write_descriptor(tmp_path, image_name="nope.png")

        with pytest.raises(OSError):
            parse_depth_json(str(descriptor))

    def test_wrong_color_type_raises_depth_decode_error(self, tmp_path):
        # color type 2 (RGB) instead of 0 (grayscale)
        ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
        idat = zlib.compress(bytes([0, 0, 0, 0, 0, 0, 0]))
        png = (
            b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
        )
        (tmp_path / "depth.png").write_bytes(png)
        descriptor = self._write_descriptor(tmp_path)

        with pytest.raises(DepthDecodeError, match="color type"):
            parse_depth_json(str(descriptor))

    def test_wrong_bit_depth_raises(self, tmp_path):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
        idat = zlib.compress(bytes([0, 0]))
        png = (
            b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
        )
        (tmp_path / "depth.png").write_bytes(png)
        descriptor = self._write_descriptor(tmp_path)

        with pytest.raises(DepthDecodeError, match="bit depth"):
            parse_depth_json(str(descriptor))

    def test_interlaced_png_raises(self, tmp_path):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 0, 0, 0, 1)  # interlace=1
        idat = zlib.compress(bytes([0, 0, 0]))
        png = (
            b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
        )
        (tmp_path / "depth.png").write_bytes(png)
        descriptor = self._write_descriptor(tmp_path)

        with pytest.raises(DepthDecodeError, match="[Ii]nterlaced"):
            parse_depth_json(str(descriptor))

    def test_not_a_png_raises(self, tmp_path):
        (tmp_path / "depth.png").write_bytes(b"not a png file")
        descriptor = self._write_descriptor(tmp_path)

        with pytest.raises(DepthDecodeError, match="signature"):
            parse_depth_json(str(descriptor))

    def test_corrupt_zlib_stream_raises(self, tmp_path):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 0, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", b"not valid zlib data")
            + _png_chunk(b"IEND", b"")
        )
        (tmp_path / "depth.png").write_bytes(png)
        descriptor = self._write_descriptor(tmp_path)

        with pytest.raises(DepthDecodeError, match="corrupt"):
            parse_depth_json(str(descriptor))

    def test_filter_types_sub_up_average_paeth_all_decode_correctly(self, tmp_path):
        """Each PNG filter type reconstructs to the same known image --
        proves the unfiltering math itself, not just filter-type-0."""
        import struct as _struct
        import zlib as _zlib

        width, height = 3, 2
        values = [[100, 200, 300], [400, 500, 600]]
        raw_unfiltered = bytearray()
        for row in values:
            for v in row:
                raw_unfiltered += _struct.pack(">H", v)
        stride = width * 2

        for filter_type in (0, 1, 2, 3, 4):
            filtered = bytearray()
            prev = bytearray(stride)
            for row_idx in range(height):
                cur = bytearray(raw_unfiltered[row_idx * stride:(row_idx + 1) * stride])
                out_row = bytearray(stride)
                for x in range(stride):
                    a = cur[x - 2] if x >= 2 else 0
                    b = prev[x]
                    c = prev[x - 2] if x >= 2 else 0
                    if filter_type == 0:
                        out_row[x] = cur[x]
                    elif filter_type == 1:
                        out_row[x] = (cur[x] - a) & 0xFF
                    elif filter_type == 2:
                        out_row[x] = (cur[x] - b) & 0xFF
                    elif filter_type == 3:
                        out_row[x] = (cur[x] - (a + b) // 2) & 0xFF
                    elif filter_type == 4:
                        p = a + b - c
                        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                        pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                        out_row[x] = (cur[x] - pred) & 0xFF
                filtered.append(filter_type)
                filtered += out_row
                prev = cur
            ihdr = _struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
            idat = _zlib.compress(bytes(filtered))
            png = (
                b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
                + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
            )
            (tmp_path / f"depth_f{filter_type}.png").write_bytes(png)
            descriptor = self._write_descriptor(tmp_path, image_name=f"depth_f{filter_type}.png")

            depth_map = parse_depth_json(str(descriptor))

            assert depth_map.values == [[float(v) for v in row] for row in values], f"filter type {filter_type} failed"

    def test_cross_validated_against_pillow_encoded_png(self, tmp_path):
        """Decode a PNG encoded by a REAL, independent library (Pillow)
        -- proves this module's decoder isn't just self-consistent with
        its own encoder, it reads what a real PNG tool actually
        produces. Skipped if Pillow (an optional perception dependency,
        not a core one) isn't installed."""
        pil_image = pytest.importorskip("PIL.Image")
        np = pytest.importorskip("numpy")

        arr = np.array([[0, 1000, 5000, 65535], [100, 200, 300, 400]], dtype=np.uint16)
        img = pil_image.fromarray(arr, mode="I;16")
        png_path = tmp_path / "pillow_depth.png"
        img.save(str(png_path))
        descriptor = self._write_descriptor(tmp_path, image_name="pillow_depth.png")

        depth_map = parse_depth_json(str(descriptor))

        assert depth_map.values == [[0.0, 1000.0, 5000.0, 65535.0], [100.0, 200.0, 300.0, 400.0]]


class TestDepthComponentDispatch:
    def test_parse_component_depth_maps_skips_non_json_files(self, tmp_path):
        values = [[0, 1]]
        (tmp_path / "depth.png").write_bytes(_encode_16bit_grayscale_png(values))
        (tmp_path / "depth.json").write_text(json.dumps({"image": "depth.png", "evidence_id": "e1"}))

        results = parse_component_depth_maps([str(tmp_path / "depth.json"), str(tmp_path / "depth.png")])

        assert len(results) == 1
        assert results[0].evidence_id == "e1"


class TestRoundTrip:
    def test_to_dict_round_trips_the_recorded_values(self, tmp_path):
        path = tmp_path / "gnss.jsonl"
        _write_jsonl(path, [{"t": 1.0, "lat_deg": 1.0, "lon_deg": 2.0, "alt_m": 3.0, "fix_type": "rtk"}])

        stream = parse_gnss_jsonl(str(path))
        d = stream.to_dict()

        assert d["kind"] == "gnss"
        assert d["samples"][0]["fix_type"] == "rtk"
        assert d["provenance"] == "OBSERVED"


class TestComponentDispatch:
    def test_parse_component_streams_skips_non_jsonl_files(self, tmp_path):
        jsonl_path = tmp_path / "log.jsonl"
        _write_jsonl(jsonl_path, [{"t": 0.0, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]}])
        csv_path = tmp_path / "raw.csv"
        csv_path.write_text("t,ax,ay,az\n0,0,0,9.81\n")

        streams = parse_component_streams([str(jsonl_path), str(csv_path)], "imu")

        assert len(streams) == 1
        assert streams[0].source_path == str(jsonl_path)

    def test_unknown_component_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no parser registered"):
            parse_component_streams([], "depth")


class TestMultiSourceSessionWiring:
    """The full chain: add_source() detects sidecar files -> records
    them on SourceRecord.components -> sensor_streams() resolves and
    parses them for real."""

    def _phone_capture_folder(self, tmp_path):
        folder = tmp_path / "phone_capture_001"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "0001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x01" * 64)
        (folder / "imu").mkdir()
        _write_jsonl(folder / "imu" / "log.jsonl", [
            {"t": 0.0, "accel_mps2": [0.0, 0.0, 9.81], "gyro_rps": [0.0, 0.0, 0.0]},
            {"t": 0.01, "accel_mps2": [0.05, 0.0, 9.80], "gyro_rps": [0.02, 0.0, 0.0]},
        ])
        (folder / "gps").mkdir()
        _write_jsonl(folder / "gps" / "track.jsonl", [
            {"t": 0.0, "lat_deg": 37.4, "lon_deg": -122.1, "alt_m": 10.0, "fix_type": "3d"},
        ])
        (folder / "calibration").mkdir()
        (folder / "calibration" / "cam0.json").write_text(json.dumps({"intrinsics": {
            "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0, "width": 1920, "height": 1080,
        }}))
        return folder

    def test_add_source_records_detected_components_on_the_record(self, tmp_path):
        """Regression: SourceRecord.components existed on the dataclass
        but add_source() never actually populated it -- sidecar files
        were detected (for SourceType classification) and discarded."""
        from evidence.multi_source import CaptureComponent, MultiSourceSession

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="s1")

        record = session.add_source(str(folder), source=None)

        assert record.components  # not {} -- the bug this fixes
        assert record.components[CaptureComponent.IMU.value] == ["imu/log.jsonl"]
        assert record.components[CaptureComponent.GPS.value] == ["gps/track.jsonl"]
        assert record.components[CaptureComponent.RGB.value] == ["rgb/0001.jpg"]

    def test_sensor_streams_parses_the_real_imu_data(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(folder))

        streams = session.sensor_streams(record.source_id, "imu")

        assert len(streams) == 1
        assert len(streams[0]) == 2
        assert streams[0].samples[0].accel_mps2 == (0.0, 0.0, 9.81)

    def test_sensor_streams_parses_the_real_gnss_data(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(folder))

        streams = session.sensor_streams(record.source_id, "gps")

        assert len(streams) == 1
        assert streams[0].samples[0].lat_deg == 37.4

    def test_calibrations_parses_the_real_intrinsics(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(folder))

        calibrations = session.calibrations(record.source_id)

        assert len(calibrations) == 1
        assert calibrations[0].intrinsics.fx == 1200.0
        assert calibrations[0].intrinsics.width == 1920

    def test_calibrations_empty_when_no_calibration_files(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        photo_only = tmp_path / "just_photos2"
        photo_only.mkdir()
        (photo_only / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x01" * 64)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(photo_only))

        assert session.calibrations(record.source_id) == []

    def test_sensor_streams_returns_empty_list_when_no_component_files(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        photo_only = tmp_path / "just_photos"
        photo_only.mkdir()
        (photo_only / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x01" * 64)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(photo_only))

        assert session.sensor_streams(record.source_id, "imu") == []

    def test_components_and_sensor_streams_survive_a_round_trip(self, tmp_path):
        """The detected components must also serialize -- to_dict()/
        from_dict() already had the field; this proves it now carries
        real content, not just an always-empty {}."""
        from evidence.multi_source import MultiSourceSession

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="s1")
        record = session.add_source(str(folder))

        restored = MultiSourceSession.from_dict(session.to_dict())
        restored_record = restored._sources[record.source_id]

        assert restored_record.components == record.components
        streams = restored.sensor_streams(record.source_id, "imu")
        assert len(streams) == 1
