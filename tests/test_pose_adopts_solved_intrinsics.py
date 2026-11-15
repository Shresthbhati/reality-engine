"""3a: with no EXIF and no earlier model, the pose stage adopts the focal length COLMAP just estimated (trust: estimated)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.s04_pose import adopt_solved_intrinsics
from engine.pipeline.vertical_slice import VerticalSliceOptions
from reconstruction.colmap_intrinsics import agreed_intrinsics, read_colmap_cameras

SIZE = (1536, 1024)


def _model(root: Path, lines):
    d = root / "staging" / "sparse" / "0"
    d.mkdir(parents=True)
    (d / "cameras.txt").write_text("# Camera list\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return SimpleNamespace(backend=SimpleNamespace(session=SimpleNamespace(staging=root / "staging")))


def _ctx(root: Path, **opts):
    options = VerticalSliceOptions(image_size=SIZE, **opts)
    return StageContext(job_id="j", workdir=root, artifacts={}, params={}, state={"vertical_slice_options": options})


def test_adopts_agreeing_single_camera_estimate_as_estimated(tmp_path):
    pose = _model(tmp_path, ["1 SIMPLE_RADIAL 1536 1024 1800.5 768 512 0.01"])
    ctx = _ctx(tmp_path)
    rec = adopt_solved_intrinsics(ctx, pose)
    opts = ctx.state["vertical_slice_options"]
    assert opts.intrinsics == (1800.5, 1800.5, 768.0, 512.0) and opts.intrinsics_trust == "estimated"
    assert rec["source"] == "colmap_estimated_this_run" and rec["trust"] == "estimated"


def test_trusted_or_existing_intrinsics_are_never_overridden(tmp_path):
    pose = _model(tmp_path, ["1 SIMPLE_RADIAL 1536 1024 1800.5 768 512 0.01"])
    ctx = _ctx(tmp_path, intrinsics=(1000.0, 1000.0, 768.0, 512.0))
    assert adopt_solved_intrinsics(ctx, pose) is None
    assert ctx.state["vertical_slice_options"].intrinsics == (1000.0, 1000.0, 768.0, 512.0)


def test_per_image_cameras_with_modest_spread_use_the_median_and_report_the_spread(tmp_path):
    # what COLMAP wrote for the ten generated living-room photos: one camera per image, focals ~3% apart
    pose = _model(tmp_path, ["1 SIMPLE_RADIAL 1536 1024 1016.1 768 512 -0.11",
                             "2 SIMPLE_RADIAL 1536 1024 994.9 768 512 -0.12",
                             "3 SIMPLE_RADIAL 1536 1024 1026.4 768 512 -0.12"])
    ctx = _ctx(tmp_path)
    rec = adopt_solved_intrinsics(ctx, pose)
    assert ctx.state["vertical_slice_options"].intrinsics[0] == 1016.1  # the median focal
    assert 0.02 < rec["focal_spread"] < 0.04 and rec["trust"] == "estimated"


def test_disagreeing_cameras_wrong_size_or_fisheye_are_refused(tmp_path):
    cases = (
        ["1 SIMPLE_RADIAL 1536 1024 1800 768 512 0", "2 SIMPLE_RADIAL 1536 1024 1500 768 512 0"],  # focals 18% apart
        ["1 SIMPLE_RADIAL 800 600 1000 400 300 0"],                                               # not the images' size
        ["1 OPENCV_FISHEYE 1536 1024 900 900 768 512 0 0 0 0"],                                   # fisheye
    )
    for i, lines in enumerate(cases):
        root = tmp_path / f"case{i}"
        root.mkdir()
        ctx = _ctx(root)
        assert adopt_solved_intrinsics(ctx, _model(root, lines)) is None
        assert ctx.state["vertical_slice_options"].intrinsics is None


def test_no_colmap_session_or_unreadable_model_leaves_intrinsics_unset(tmp_path):
    ctx = _ctx(tmp_path)
    assert adopt_solved_intrinsics(ctx, SimpleNamespace(backend=SimpleNamespace())) is None  # test backend: no session
    pose = SimpleNamespace(backend=SimpleNamespace(session=SimpleNamespace(staging=tmp_path / "nope")))
    assert adopt_solved_intrinsics(ctx, pose) is None
    assert ctx.state["vertical_slice_options"].intrinsics is None


def test_uniform_image_size_gates_the_shared_camera(tmp_path):
    from PIL import Image

    from reconstruction.backend.colmap_backend import _uniform_image_size

    def img(name, size):
        p = tmp_path / name
        Image.new("RGB", size).save(p)
        return p

    a, b, c = img("a.png", (64, 48)), img("b.png", (64, 48)), img("c.png", (32, 24))
    assert _uniform_image_size([a, b]) is True            # one device -> one shared camera
    assert _uniform_image_size([a, b, c]) is False        # mixed sizes -> keep per-image cameras
    assert _uniform_image_size([a]) is False              # one photo says nothing about sharing
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    assert _uniform_image_size([a, bad]) is False         # unreadable -> default, not a guess


def test_shared_reader_and_agreement_rule():
    assert agreed_intrinsics([], SIZE) is None
    cams = [{"model": "PINHOLE", "width": 1536, "height": 1024, "params": [1200.0, 1210.0, 768.0, 512.0]}]
    assert agreed_intrinsics(cams, SIZE) == (1200.0, 1210.0, 768.0, 512.0)
    assert read_colmap_cameras(Path("does-not-exist")) == []
