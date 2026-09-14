"""Tests for evidence/depth_frames.py (16-bit PNG depth sidecar
parsing per Decision 020) and its wiring into
evidence.multi_source.MultiSourceSession.depth_frames().

Rules under test (docs/future/sensor-ingestion/DEPTH_INGESTION.md):
raw integers are never meters without an explicit scale; invalid
pixels are masked, not zero-filled; unknown formats/scales fail
loudly naming the file; nothing is silently skipped.
"""

from __future__ import annotations

import json
import os
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from evidence.depth_frames import (
    DepthFrame,
    DepthFrameError,
    DepthScaleUnavailable,
    DepthSidecarManifest,
    parse_depth_frames,
    parse_depth_png,
)
from evidence.sensors import SensorParseError
from provenance import Provenance


def _write_depth_png(path, rows, mode="I;16"):
    arr = np.array(rows, dtype="<u2")
    # Pillow emits a deprecation warning for mode="I;16" here; I;16
    # PNGs are exactly what real depth devices emit and what we parse.
    Image.fromarray(arr, mode=mode).save(path)


MM_ROWS = [[1000, 2000, 3000, 0], [1500, 2500, 3500, 500], [0, 0, 4000, 4500]]


class TestExplicitScaleParsing:
    def test_parses_raw_values_and_scale(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        frame = parse_depth_png(path, depth_scale=0.001)

        assert frame.width == 4 and frame.height == 3
        assert frame.depth_dtype == "uint16"
        assert frame.depth_scale == 0.001
        assert frame.units == "meter"
        assert frame.raw[0] == (1000, 2000, 3000, 0)
        assert frame.provenance == Provenance.OBSERVED

    def test_meters_at_applies_scale(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        frame = parse_depth_png(path, depth_scale=0.001)

        assert frame.meters_at(0, 0) == 1.0
        assert frame.meters_at(2, 2) == 4.0

    def test_invalid_pixel_is_masked_not_zero(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        frame = parse_depth_png(path, depth_scale=0.001)

        # 0 is the invalid marker: reading it must be None, not 0.0 m
        assert frame.meters_at(0, 3) is None
        assert frame.meters_at(2, 0) is None
        assert frame.invalid_pixel_count == 3

    def test_iter_meters_yields_only_valid_pixels(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        frame = parse_depth_png(path, depth_scale=0.001)

        meters = sorted(frame.iter_meters())
        assert len(meters) == 9  # 12 pixels - 3 invalid
        assert meters[0] == (0, 0, 1.0)
        assert all(isinstance(m, float) for _, _, m in meters)

    def test_explicit_invalid_value_override(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, [[65535, 1000]])

        frame = parse_depth_png(path, depth_scale=0.001, invalid_value=65535)

        assert frame.meters_at(0, 0) is None
        assert frame.meters_at(0, 1) == 1.0
        assert frame.invalid_pixel_count == 1

    def test_frame_id_defaults_to_stem(self, tmp_path):
        path = str(tmp_path / "frame_0042.png")
        _write_depth_png(path, [[5]])

        frame = parse_depth_png(path, depth_scale=1.0)

        assert frame.frame_id == "frame_0042"

    def test_frame_id_override(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, [[5]])

        frame = parse_depth_png(path, depth_scale=1.0, frame_id="rgb_0042")

        assert frame.frame_id == "rgb_0042"

    def test_roundtrip_to_dict_from_dict(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        frame = parse_depth_png(path, depth_scale=0.001)

        assert DepthFrame.from_dict(frame.to_dict()) == frame


class TestScaleRefusal:
    def test_no_scale_anywhere_raises_naming_file(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        with pytest.raises(DepthScaleUnavailable, match="frame_0001"):
            parse_depth_png(path)

    def test_zero_scale_rejected(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        with pytest.raises(DepthFrameError, match="depth_scale"):
            parse_depth_png(path, depth_scale=0.0)

    def test_negative_scale_rejected(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        with pytest.raises(DepthFrameError, match="depth_scale"):
            parse_depth_png(path, depth_scale=-0.001)

    def test_nonfinite_scale_rejected(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)

        with pytest.raises(DepthFrameError, match="depth_scale"):
            parse_depth_png(path, depth_scale=float("nan"))

    def test_depthframe_constructor_enforces_scale_rule_too(self):
        # The dataclass itself refuses to exist unscaled, so no other
        # construction path can sneak raw-integers-as-meters past the
        # parser's gate.
        with pytest.raises(ValueError, match="depth_scale"):
            DepthFrame(
                frame_id="x", source_path="x.png", width=1, height=1,
                raw=((1000,),), depth_dtype="uint16", depth_scale=0.0,
                invalid_value=0, units="meter",
            )

    def test_unknown_units_rejected(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, [[1000]])

        with pytest.raises(ValueError, match="units"):
            parse_depth_png(path, depth_scale=1.0, units="fathoms")


class TestManifestResolution:
    def test_manifest_global_scale_resolves(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, MM_ROWS)
        manifest = DepthSidecarManifest.load(self._write_manifest(tmp_path, {"depth_scale": 0.001}))

        frame = parse_depth_png(path, manifest=manifest)

        assert frame.depth_scale == 0.001
        assert frame.meters_at(0, 0) == 1.0

    def test_per_file_override_wins_over_global(self, tmp_path):
        path = str(tmp_path / "frame_0002.png")
        _write_depth_png(path, [[2000]])
        manifest = DepthSidecarManifest.load(self._write_manifest(
            tmp_path, {"depth_scale": 0.001, "per_file": {"frame_0002.png": {"depth_scale": 0.0005}}}
        ))

        frame = parse_depth_png(path, manifest=manifest)

        assert frame.depth_scale == 0.0005
        assert frame.meters_at(0, 0) == 1.0

    def test_manifest_invalid_value_and_units_carry(self, tmp_path):
        path = str(tmp_path / "frame_0001.png")
        _write_depth_png(path, [[65535, 2000]])
        manifest = DepthSidecarManifest.load(self._write_manifest(
            tmp_path, {"depth_scale": 0.001, "invalid_value": 65535, "units": "millimeter"}
        ))

        frame = parse_depth_png(path, manifest=manifest)

        assert frame.meters_at(0, 0) is None
        assert frame.units == "millimeter"

    def test_manifest_without_global_scale_and_file_not_listed(self, tmp_path):
        path = str(tmp_path / "frame_0003.png")
        _write_depth_png(path, [[1000]])
        manifest = DepthSidecarManifest.load(self._write_manifest(
            tmp_path, {"per_file": {"frame_0001.png": {"depth_scale": 0.001}}}
        ))

        with pytest.raises(DepthScaleUnavailable, match="frame_0003"):
            parse_depth_png(path, manifest=manifest)

    def test_manifest_bad_scale_rejected_at_load(self, tmp_path):
        path = self._write_manifest(tmp_path, {"depth_scale": 0.0})

        with pytest.raises(DepthFrameError, match="depth_scale"):
            DepthSidecarManifest.load(path)

    def test_manifest_bad_units_rejected_at_load(self, tmp_path):
        path = self._write_manifest(tmp_path, {"units": "parsec"})

        with pytest.raises(DepthFrameError, match="units"):
            DepthSidecarManifest.load(path)

    def test_manifest_bad_per_file_entry_rejected_at_load(self, tmp_path):
        path = self._write_manifest(tmp_path, {"per_file": {"f.png": {"depth_scale": -1}}})

        with pytest.raises(DepthFrameError, match="per_file"):
            DepthSidecarManifest.load(path)

    def test_manifest_invalid_json_rejected(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json")

        with pytest.raises(DepthFrameError, match="invalid JSON"):
            DepthSidecarManifest.load(str(path))

    @staticmethod
    def _write_manifest(tmp_path, data) -> str:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)


class TestFormatRejection:
    def test_8bit_gray_rejected_as_not_depth(self, tmp_path):
        path = str(tmp_path / "preview.png")
        Image.fromarray(np.zeros((4, 3), dtype=np.uint8), mode="L").save(path)

        with pytest.raises(DepthFrameError, match="not a supported depth encoding"):
            parse_depth_png(path, depth_scale=0.001)

    def test_rgb_color_rejected(self, tmp_path):
        path = str(tmp_path / "photo.png")
        Image.fromarray(np.zeros((4, 3, 3), dtype=np.uint8), mode="RGB").save(path)

        with pytest.raises(DepthFrameError, match="not a supported depth encoding"):
            parse_depth_png(path, depth_scale=0.001)

    def test_truncated_png_rejected(self, tmp_path):
        path = str(tmp_path / "cut.png")
        _write_depth_png(path, MM_ROWS)
        raw = open(path, "rb").read()
        open(path, "wb").write(raw[: len(raw) // 2])

        with pytest.raises(DepthFrameError, match="decode failed|unreadable"):
            parse_depth_png(path, depth_scale=0.001)

    def test_corrupt_idat_rejected(self, tmp_path):
        # Valid header chunks, garbage IDAT payload: header parses,
        # zlib stream does not. Exercises the decode-failure branch
        # distinctly from truncation.
        path = str(tmp_path / "corrupt.png")
        width, height = 2, 2

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
        fake_idat = zlib.compress(b"\x00" + b"\x00" * (width * height * 2 - 1))[:-4]  # truncated stream
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", fake_idat)
               + chunk(b"IEND", b""))
        open(path, "wb").write(png)

        with pytest.raises(DepthFrameError):
            parse_depth_png(path, depth_scale=0.001)

    def test_missing_file_raises_oserror_not_depthframeerror(self, tmp_path):
        with pytest.raises(OSError):
            parse_depth_png(str(tmp_path / "absent.png"), depth_scale=0.001)


class TestBatchParsing:
    def test_parse_depth_frames_skips_non_png(self, tmp_path):
        png = str(tmp_path / "a.png")
        txt = str(tmp_path / "notes.txt")
        _write_depth_png(png, [[1000]])
        open(txt, "w").write("not a depth file")

        frames = parse_depth_frames([png, txt], depth_scale=0.001)

        assert len(frames) == 1
        assert frames[0].frame_id == "a"

    def test_parse_depth_frames_empty_input(self):
        assert parse_depth_frames([], depth_scale=0.001) == []

    def test_parse_depth_frames_with_manifest(self, tmp_path):
        p1 = str(tmp_path / "f1.png")
        p2 = str(tmp_path / "f2.png")
        _write_depth_png(p1, [[1000]])
        _write_depth_png(p2, [[2000]])
        manifest = DepthSidecarManifest.load(self._write_manifest(
            tmp_path, {"depth_scale": 0.001, "per_file": {"f2.png": {"depth_scale": 0.0005}}}
        ))

        frames = parse_depth_frames([p1, p2], manifest=manifest)

        assert [f.depth_scale for f in frames] == [0.001, 0.0005]

    @staticmethod
    def _write_manifest(tmp_path, data) -> str:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)


class TestErrorHierarchy:
    def test_depthframeerror_is_sensorparseerror(self):
        # Callers already catching SensorParseError for IMU/GNSS/
        # telemetry sidecars catch depth parsing failures too.
        assert issubclass(DepthFrameError, SensorParseError)


class TestSessionWiring:
    """depth_frames() on MultiSourceSession: on-demand parsing, honest
    empty when no depth files, manifest auto-discovery."""

    def _session_with_depth(self, tmp_path, manifest_data=None):
        from evidence.multi_source import MultiSourceSession

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir(exist_ok=True)
        _write_depth_png(str(depth_dir / "frame_0001.png"), MM_ROWS)
        rgb_dir = tmp_path / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((4, 3, 3), dtype=np.uint8), mode="RGB").save(str(rgb_dir / "rgb_0001.png"))
        if manifest_data is not None:
            (depth_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        session = MultiSourceSession(session_id="s1")
        session.add_source(str(tmp_path))
        return session

    def test_returns_empty_for_source_without_depth(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        # A real source with no depth/ directory: empty result is
        # honest (nothing recorded), distinct from an unknown source
        # id (KeyError, per depth_frames()' contract).
        (tmp_path / "rgb").mkdir()
        Image.fromarray(np.zeros((4, 3, 3), dtype=np.uint8), mode="RGB").save(str(tmp_path / "rgb" / "x.png"))
        session = MultiSourceSession(session_id="s1")
        session.add_source(str(tmp_path))
        source_id = session.sources()[0].source_id

        assert session.depth_frames(source_id) == []

    def test_manifest_in_source_resolves_scale(self, tmp_path):
        session = self._session_with_depth(tmp_path, {"depth_scale": 0.001})
        source_id = session.sources()[0].source_id

        frames = session.depth_frames(source_id)

        assert len(frames) == 1
        assert frames[0].depth_scale == 0.001
        assert frames[0].meters_at(0, 0) == 1.0

    def test_without_manifest_raises_scale_unavailable(self, tmp_path):
        session = self._session_with_depth(tmp_path)
        source_id = session.sources()[0].source_id

        with pytest.raises(DepthScaleUnavailable, match="frame_0001"):
            session.depth_frames(source_id)

    def test_parse_error_names_file_and_propagates(self, tmp_path):
        from evidence.multi_source import MultiSourceSession

        (tmp_path / "rgb").mkdir()
        Image.fromarray(np.zeros((4, 3, 3), dtype=np.uint8), mode="RGB").save(str(tmp_path / "rgb" / "x.png"))
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        Image.fromarray(np.zeros((4, 3), dtype=np.uint8), mode="L").save(str(depth_dir / "bad.png"))
        # Provide a manifest so the scale gate passes and the file's
        # format problem is what surfaces (scale refusal precedes
        # decode by design).
        (depth_dir / "manifest.json").write_text(json.dumps({"depth_scale": 0.001}), encoding="utf-8")

        session = MultiSourceSession(session_id="s1")
        session.add_source(str(tmp_path))
        source_id = session.sources()[0].source_id

        with pytest.raises(DepthFrameError, match="bad.png"):
            session.depth_frames(source_id)
