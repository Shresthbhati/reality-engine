"""Synthetic-video tests for reconstruction.video_frames (no real footage, no ffmpeg)."""
import hashlib
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from reconstruction import video_frames as vf  # noqa: E402

FPS, W, H = 20, 320, 240
PAN_PX = 4  # px per frame = 80 px/s
# segments (seconds): pan, blurred pan, static, pan
SEGMENTS = [("pan", 2.0), ("blur", 1.5), ("static", 1.5), ("pan", 2.0)]


def _scene():
    rng = np.random.default_rng(7)
    big = (rng.random((H, 2400)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 1.2)
    for x in range(0, 2400, 90):
        cv2.rectangle(big, (x, 40 + (x % 7) * 15), (x + 40, 120 + (x % 5) * 20), int(60 + x % 150), -1)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def _write_video(path, codec):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
    if not wr.isOpened():
        return None
    scene, x, labels = _scene(), 0, []
    for kind, secs in SEGMENTS:
        for _ in range(int(secs * FPS)):
            if kind != "static":
                x += PAN_PX
            fr = scene[:, x:x + W].copy()
            if kind == "blur":
                fr = cv2.blur(fr, (31, 31))
            wr.write(fr)
            labels.append(kind)
    wr.release()
    return labels


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    for codec, ext in (("MJPG", "avi"), ("mp4v", "mp4")):
        p = d / f"walk.{ext}"
        labels = _write_video(p, codec)
        if labels and p.is_file() and p.stat().st_size > 0 and cv2.VideoCapture(str(p)).read()[0]:
            return p, labels
    pytest.skip("no OpenCV VideoWriter codec (MJPG/mp4v) works on this machine")


def test_info(video):
    p, labels = video
    info = vf.read_video_info(p)
    assert (info["width"], info["height"]) == (W, H)
    assert info["fps"] == pytest.approx(FPS, abs=0.5)
    assert abs(info["frames"] - len(labels)) <= 1 and info["rotation"] == 0


def test_keyframes_behaviour(video):
    p, labels = video
    r = vf.sample_keyframes(p)
    idx = [k.index for k in r.frames]
    kinds = [labels[i] for i in idx]
    assert len(idx) >= 8, r.stats
    # no kept frame is blurred; the static stretch yields at most the one frame that entered it
    assert "blur" not in kinds
    assert kinds.count("static") <= 1
    # blurred and duplicate frames were rejected WITH reasons
    reasons = " ".join(x["reason"] for x in r.rejected)
    assert "motion_blur" in reasons and "near_duplicate" in reasons
    assert any(labels[x["index"]] == "blur" and x["reason"].startswith("motion_blur") for x in r.rejected)
    # sharp enough and spaced: min gap in time, and real motion between consecutive kept frames
    assert all(k.sharpness >= r.stats["blur_floor"] for k in r.frames)
    assert all(b.time_s - a.time_s >= 0.25 - 1e-6 for a, b in zip(r.frames, r.frames[1:]))
    assert all(k.shift_px is None or k.shift_px >= vf.MIN_MOTION_PX for k in r.frames[1:])
    # the pan resumes after the static stretch: frames exist from the last segment
    assert max(idx) > len(labels) - int(2 * FPS)


def test_max_frames_and_target(video):
    p, _ = video
    assert len(vf.sample_keyframes(p, max_frames=5).frames) <= 5
    r = vf.sample_keyframes(p, target_count=6)
    assert len(r.frames) <= 6 and any("over_budget" in x["reason"] for x in r.rejected)


def test_deterministic(video, tmp_path):
    p, _ = video
    a = vf.write_keyframes(p, tmp_path / "a")
    b = vf.write_keyframes(p, tmp_path / "b")
    assert [(k.index, k.time_s, k.sharpness, k.shift_px) for k in a.frames] == \
           [(k.index, k.time_s, k.sharpness, k.shift_px) for k in b.frames]
    for ka, kb in zip(a.frames, b.frames):
        assert open(ka.path, "rb").read() == open(kb.path, "rb").read()


def test_write_and_manifest(video, tmp_path):
    p, _ = video
    res = vf.write_keyframes(p, tmp_path / "o")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert m["source_sha256"] == sha and m["video"]["fps"] == pytest.approx(FPS, abs=0.5)
    assert len(m["frames"]) == len(res.frames)
    for k, e in zip(res.frames, m["frames"]):
        assert e["file"] == f"video-{sha[:8]}-f{k.index:06d}.jpg" and e["time_s"] == k.time_s
        img = cv2.imread(k.path)
        assert img is not None and img.shape[:2] == (H, W)


def test_diagnostics_short_and_fast(tmp_path):
    p = tmp_path / "short.avi"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not wr.isOpened():
        pytest.skip("MJPG writer unavailable")
    scene = _scene()
    for i in range(30):  # 1.5 s, 60 px/frame: far too fast
        wr.write(scene[:, i * 60:i * 60 + W])
    wr.release()
    r = vf.sample_keyframes(p)
    text = " ".join(r.diagnostics)
    assert "only" in text and "usable keyframes" in text  # short + too few; never padded
    assert len(r.frames) == r.stats["kept"]


def test_bad_paths(tmp_path):
    with pytest.raises(vf.VideoError, match="not found"):
        vf.sample_keyframes(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    with pytest.raises(vf.VideoError):
        vf.read_video_info(junk)
