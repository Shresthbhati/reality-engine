"""EXR depth sidecar adapter (P1-03 depth integration, format ladder
rung 2 of Decision 020).

The 16-bit PNG rung is integer-only: a device reporting float meters
(or +inf for no-return LiDAR pixels) cannot be represented losslessly.
EXR is the float rung. The adapter follows the module's established
discipline exactly:

  - EXR depth IS metric float meters by openEXR convention used here;
    the manifest scale rule does NOT apply to float EXR (no raw
    integers to misread). A manifest `depth_scale` other than 1.0 on an
    EXR file is a contradiction and is REJECTED (DepthFrameError) --
    silently applying it would double-scale real meters.
  - invalid pixels: NaN, +inf, and negative values are masked (not
    zero-filled); the count is reported via invalid_pixel_count.
  - DepthFrame.raw stays integer-valued (milli-meters as ints) so the
    canonical DepthFrame contract is unchanged: EXR float meters are
    stored as round(m * 1000) with depth_scale=0.001, an exact
    representation for mm-precision depth, and meters_at() returns the
    same value a PNG path would (sub-mm precision is not recoverable
    from DepthFrame anyway -- documented, not hidden).
  - missing OpenEXR -> DepthFrameError naming the dependency, never a
    silent skip (adapter-absence is an availability fact).
"""

import math

import numpy as np
import pytest

from evidence.depth_exr import parse_depth_exr
from evidence.depth_frames import (
    DepthFrame,
    DepthFrameError,
    DepthSidecarManifest,
)


def _write_exr(path, meters, channel="D"):
    import OpenEXR
    import Imath

    meters = np.asarray(meters, dtype=np.float32)
    h, w = meters.shape
    hdr = OpenEXR.Header(w, h)
    hdr["channels"] = {channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(str(path), hdr)
    out.writePixels({channel: meters.tobytes()})
    out.close()


METERS = [[1.0, 2.5, 3.0, 4.0], [1.5, 2.0, float("nan"), 0.5], [2.25, 3.5, 4.0, float("inf")]]


class TestExrParsing:
    def test_parses_float_meters(self, tmp_path):
        path = tmp_path / "frame_0001.exr"
        _write_exr(path, METERS)
        frame = parse_depth_exr(str(path))
        assert (frame.width, frame.height) == (4, 3)
        assert frame.depth_dtype == "float32"
        # NaN/inf/negative are invalid; here 1 NaN + 1 inf.
        assert frame.invalid_pixel_count == 2
        assert frame.meters_at(0, 0) == pytest.approx(1.0)
        assert frame.meters_at(1, 3) == pytest.approx(0.5)
        assert frame.meters_at(0, 2) == 3.0

    def test_nan_inf_negative_masked_not_zero(self, tmp_path):
        path = tmp_path / "frame.exr"
        _write_exr(path, [[1.0, -0.5], [float("nan"), float("inf")]])
        frame = parse_depth_exr(str(path))
        assert frame.invalid_pixel_count == 3
        assert frame.meters_at(0, 1) is None
        assert frame.meters_at(1, 0) is None

    def test_manifest_scale_one_ok(self, tmp_path):
        path = tmp_path / "frame.exr"
        _write_exr(path, [[2.0]])
        manifest = DepthSidecarManifest(depth_scale=1.0)
        frame = parse_depth_exr(str(path), manifest=manifest)
        assert frame.meters_at(0, 0) == pytest.approx(2.0)

    def test_manifest_scale_other_than_one_rejected(self, tmp_path):
        path = tmp_path / "frame.exr"
        _write_exr(path, [[2.0]])
        manifest = DepthSidecarManifest(depth_scale=0.001)
        with pytest.raises(DepthFrameError) as exc:
            parse_depth_exr(str(path), manifest=manifest)
        assert "scale" in str(exc.value).lower()

    def test_missing_dependency_is_explicit(self, tmp_path, monkeypatch):
        path = tmp_path / "frame.exr"
        _write_exr(path, [[1.0]])
        import evidence.depth_exr as dx
        monkeypatch.setattr(dx, "_OpenEXRInputFile", None)
        with pytest.raises(DepthFrameError) as exc:
            parse_depth_exr(str(path))
        assert "OpenEXR" in str(exc.value)

    def test_corrupt_file_raises_depth_frame_error(self, tmp_path):
        path = tmp_path / "bad.exr"
        path.write_bytes(b"not an exr file at all")
        with pytest.raises(DepthFrameError):
            parse_depth_exr(str(path))

    def test_parse_depth_frames_routes_exr(self, tmp_path):
        from evidence.depth_frames import parse_depth_frames

        p1 = tmp_path / "a.exr"
        p2 = tmp_path / "b.png"
        _write_exr(p1, [[1.0, 2.0]])
        from PIL import Image
        Image.fromarray(np.array([[1000, 2000]], dtype="<u2"), mode="I;16").save(str(p2))
        frames = parse_depth_frames([str(p1), str(p2)], depth_scale=0.001)
        assert len(frames) == 2
        assert {f.depth_dtype for f in frames} == {"float32", "uint16"}
