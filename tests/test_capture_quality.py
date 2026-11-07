"""F1 capture guidance: blur/exposure/overlap verdicts, video keyframe choice, wall coverage + hints.

All images are generated here (textured scene, then Gaussian-blurred / clipped); nothing is a real photo, so the
thresholds are exercised, not validated."""

from __future__ import annotations

import math

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from reconstruction import capture_quality as cq  # noqa: E402


def _scene(seed=0, size=(480, 640)) -> "np.ndarray":
    """Textured grey 'room': random blobs + lines, deterministic."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 128, np.uint8)
    for _ in range(120):
        c = (int(rng.integers(0, size[1])), int(rng.integers(0, size[0])))
        cv2.circle(img, c, int(rng.integers(3, 25)), int(rng.integers(20, 235)), -1)
    for _ in range(40):
        cv2.line(img, (int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))),
                 (int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))), int(rng.integers(0, 255)), 2)
    return img


def _jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_sharp_vs_blurred_vs_exposure_verdicts():
    sharp = _scene()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    blown = np.clip(sharp.astype(int) + 140, 0, 255).astype(np.uint8)
    dark = (sharp * 0.1).astype(np.uint8)
    res = cq.assess_capture([("sharp", _jpg(sharp)), ("blurred", _jpg(blurred)),
                             ("blown", _jpg(blown)), ("dark", _jpg(dark)), ("junk", b"not an image")])
    p = res["photos"]
    print({k: (v["verdict"], (v["metrics"] or {}).get("blur_laplacian_var")) for k, v in p.items()})
    assert p["sharp"]["metrics"]["blur_laplacian_var"] > 10 * p["blurred"]["metrics"]["blur_laplacian_var"]
    assert p["blurred"]["verdict"] == cq.DROP and p["blurred"]["reasons"]
    assert p["blown"]["verdict"] in (cq.WARN, cq.DROP)
    assert any("overexposed" in r or "bright" in r for r in p["blown"]["reasons"])
    assert p["dark"]["verdict"] in (cq.WARN, cq.DROP)
    assert p["junk"]["verdict"] == cq.DROP and "decoded" in p["junk"]["reasons"][0]
    assert res["provisional"] is True and "provisional" in res["provisional_note"].lower()


def test_overlap_proxy_related_vs_unrelated_vs_identical():
    a = _scene(1)
    shifted = np.roll(a, 60, axis=1)                 # same scene, sideways baseline
    other = _scene(99)                               # unrelated scene
    related = cq.pair_overlap_proxy(a, shifted)
    unrelated = cq.pair_overlap_proxy(a, other)
    identical = cq.pair_overlap_proxy(a, a)
    print(related, unrelated, identical)
    assert related["good_matches"] > cq.MIN_GOOD_MATCHES_WARN
    assert related["median_shift_fraction"] > cq.MIN_MEDIAN_SHIFT_FRACTION
    assert unrelated["good_matches"] < related["good_matches"] / 2
    assert identical["median_shift_fraction"] < cq.MIN_MEDIAN_SHIFT_FRACTION


def test_batch_flags_an_unrelated_photo_and_a_duplicate():
    a = _scene(1)
    res = cq.assess_capture([("a", _jpg(a)), ("b", _jpg(np.roll(a, 60, axis=1))),
                             ("dup", _jpg(a)), ("lost", _jpg(_scene(7)))])
    p = res["photos"]
    print({k: v["verdict"] for k, v in p.items()})
    assert p["b"]["verdict"] == cq.OK
    # borderline match counts on random synthetic blobs: warn or drop are both honest, ok is not
    assert p["lost"]["verdict"] in (cq.WARN, cq.DROP)
    assert any("overlap" in r.lower() or "share" in r.lower() for r in p["lost"]["reasons"])
    assert any("identical" in r.lower() for r in p["dup"]["reasons"])
    assert p["lost"]["overlap"]["method"].startswith("ORB proxy")
    assert res["summary"]["total"] == 4 and res["summary"][cq.WARN] + res["summary"][cq.DROP] >= 2


def test_summarize_stored_counts_only_known_verdicts():
    s = cq.summarize_stored([{"verdict": "ok"}, {"verdict": "warn"}, None, {"verdict": "will_be_dropped"}, {}])
    assert (s["total"], s["ok"], s["warn"], s["will_be_dropped"]) == (3, 1, 1, 1)


# ------------------------------------------------------------------------------------------------ video --------
def _write_video(path, sharp_every=5, frames=60, still=False):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert w.isOpened()
    for i in range(frames):
        base = _scene(0 if still else i // sharp_every, size=(240, 320))
        frame = base if (i % sharp_every == 0) else cv2.GaussianBlur(base, (0, 0), 4)
        w.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    w.release()
    return str(path)


def test_keyframes_prefer_sharp_frames_and_stay_spread(tmp_path):
    path = _write_video(tmp_path / "clip.avi")
    scores = cq.score_video_frames(path, 60, budget=6)
    idx = cq.pick_keyframes(scores, budget=6)
    sharp_frames = {i for i in range(60) if i % 5 == 0}
    print("picked", idx)
    assert len(idx) >= 4
    assert all(i in sharp_frames for i in idx), "every keyframe should be a sharp (every 5th) frame"
    assert idx == sorted(idx) and idx[-1] - idx[0] > 30


def test_static_video_does_not_return_redundant_frames(tmp_path):
    path = _write_video(tmp_path / "still.avi", sharp_every=1, frames=40, still=True)
    idx = cq.pick_keyframes(cq.score_video_frames(path, 40, budget=8), budget=8)
    assert len(idx) == 1  # no motion: one frame is all the baseline there is


def test_api_frame_picker_keeps_contract_and_falls_back(tmp_path, monkeypatch):
    from apps.api import routes_reconstructions as rr
    from evidence.importers import _enumerate_video_candidates

    path = _write_video(tmp_path / "clip.avi")
    cands, _ = _enumerate_video_candidates(path)
    sel, label = rr._pick_video_frames(path, cands)
    assert label.startswith("sharpness_ranked_windows") and 2 <= len(sel) <= rr.VIDEO_FRAME_BUDGET
    assert all(hasattr(c, "frame_index") and hasattr(c, "timestamp_s") for c in sel)

    monkeypatch.setattr(cq, "score_video_frames", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    sel2, label2 = rr._pick_video_frames(path, cands)
    assert label2.startswith("uniform_time_sampling") and len(sel2) >= 2


# -------------------------------------------------------------------------------------- wall coverage ---------
def _cam(x, z, dx, dz, eid="e"):
    th = math.atan2(dx, dz)                           # yaw about +Y: local +Z -> (sin, 0, cos)
    return {"evidence_id": eid, "position_m": [x, 1.5, z], "rotation_wxyz": [math.cos(th / 2), 0, math.sin(th / 2), 0]}


def test_wall_coverage_seen_unseen_and_hints_without_compass():
    square = [[-3, 0, -3], [3, 0, -3], [3, 0, 3], [-3, 0, 3]]          # A B C D; wall 2 is x=+3, wall 4 is x=-3
    cams = [_cam(-2.5, z, 1, 0, f"c{z}") for z in (-1, 0, 1)]           # three photos all facing the x=+3 wall
    res = cq.wall_coverage(square, cams)
    by = {w["index"]: w for w in res["walls"]}
    print({i: (w["status"], w["covered_fraction"], w["multi_view_fraction"]) for i, w in by.items()})
    assert by[2]["status"] == "seen" and by[2]["hint"] is None
    assert by[4]["status"] == "not_seen"
    assert by[4]["hint"].startswith("Take 3 more photos of wall 4 (between corner D and A)")
    text = " ".join(res["hints"]).lower()
    assert not any(w in text for w in ("north", "south", "east", "west"))
    assert any("compass" in n or "north reference" in n for n in res["notes"])
    assert res["summary"]["seen"] == 1 and len(res["corners"]) == 4 and len(res["cameras"]) == 3


def test_wall_coverage_other_walls_block_the_view():
    ell = [[0, 0, 0], [6, 0, 0], [6, 0, 2], [2, 0, 2], [2, 0, 6], [0, 0, 6]]
    # camera in the tall arm aiming through the inner corner's blocking wall at the far short wall (x=6)
    cams = [_cam(1, 5, 5, -4, "a"), _cam(1.2, 5.2, 5, -4, "b")]
    res = cq.wall_coverage(ell, cams)
    far = next(w for w in res["walls"] if w["p0"][0] == 6 and w["p1"][0] == 6)
    assert far["status"] == "not_seen", far


def test_wall_coverage_reports_unavailable_without_footprint():
    assert cq.wall_coverage([], [])["available"] is False
    assert cq.wall_coverage([[0, 0, 0], [1, 0, 0]], [])["reason"]


def test_a_room_built_from_the_best_single_photo_is_partial_not_unplaced():
    """Real world: 10 photos, 1 used for the learned layout. The 3D view must show that room, not a 'not placed' wall."""
    from reconstruction.scene_identity import capture_quality

    q = capture_quality(10, ["ev1"], 4374, structure={"room": 1, "floor": 1, "wall": 4},
                        layout_shell={"chosen": "ev1.jpg"})
    assert q["state"] == "partial" and q["source"] == "single_photo_layout"
    assert "best single photo" in q["message"] and "ev1.jpg" in q["message"]
    assert any("overlap" in a.lower() for a in q["advice"])


def test_without_a_layout_shell_the_verdict_is_unchanged():
    from reconstruction.scene_identity import capture_quality

    assert capture_quality(10, ["ev1"], 4374)["state"] == "insufficient"


def test_a_fused_room_says_how_many_photos_it_used_and_which_could_not_be_registered():
    from reconstruction.scene_identity import capture_quality

    q = capture_quality(10, ["ev1"], 4374, structure={"room": 1, "floor": 1, "wall": 4},
                        layout_shell={"chosen": "a.jpg", "photos_used": 4, "refused": {"b.jpg": "no consistent feature matches"}})
    assert q["state"] == "partial" and q["registered"] == 4
    assert "4 of 10" in q["message"] and "b.jpg" in q["message"]
