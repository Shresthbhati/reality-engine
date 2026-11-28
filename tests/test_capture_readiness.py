"""Capture readiness (reconstruction/capture_readiness.py): synthetic stress cases with positive AND negative expectations.

Every image is generated here (layered textured planes, so a camera shift produces parallax). Nothing is a real photograph: the
thresholds are exercised, not validated. CPU only."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from reconstruction import capture_readiness as cr  # noqa: E402
from world_ir.artifact_store import MemoryArtifactStore  # noqa: E402

W, H = 800, 600


def _texture(seed: int, size=(1400, 2600)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full(size, 120, np.uint8)
    for _ in range(900):
        c = (int(rng.integers(0, size[1])), int(rng.integers(0, size[0])))
        cv2.circle(img, c, int(rng.integers(4, 30)), int(rng.integers(10, 245)), -1)
    for _ in range(200):
        cv2.line(img, (int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))),
                 (int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))), int(rng.integers(0, 255)), 3)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


class Scene:
    """Two depth layers: far layer shifts by x, near layer by 3x (parallax). ``planar`` uses the far layer only."""

    def __init__(self, seed: int):
        self.bg, self.fg = _texture(seed), _texture(seed + 1000)
        rng = np.random.default_rng(seed + 7)
        self.mask = np.zeros(self.bg.shape, np.uint8)
        for _ in range(40):
            x, y = int(rng.integers(0, 2300)), int(rng.integers(0, 1200))
            cv2.rectangle(self.mask, (x, y), (x + int(rng.integers(80, 260)), y + int(rng.integers(80, 260))), 1, -1)

    def view(self, x: float, planar: bool = False) -> np.ndarray:
        x = int(x)
        far = self.bg[300:300 + H, 200 + x:200 + x + W].copy()
        if planar:
            return far
        x3 = 200 + 3 * x
        near = self.fg[300:300 + H, x3:x3 + W]
        m = self.mask[300:300 + H, x3:x3 + W].astype(bool)
        far[m] = near[m]
        return far


def _save(tmp_path, name: str, img, ext="png", exif=None) -> str:
    p = tmp_path / f"{name}.{ext}"
    if exif:
        from PIL import Image

        im = Image.fromarray(img)
        ex = Image.Exif()
        for k, v in exif.items():
            ex[k] = v
        im.save(p, exif=ex)
    else:
        assert cv2.imwrite(str(p), img)
    return str(p)


def _seq(tmp_path, scene: Scene, n=5, step=70, tag="a", **kw):
    return [(f"{tag}{i}", _save(tmp_path, f"{tag}{i}", scene.view(i * step, **kw))) for i in range(n)]


def _run(items, **kw):
    return cr.assess_readiness(items, labels={i: f"img-{i}" for i, _ in items}, **kw)


def _diag(rep, iid):
    return next(r for r in rep["images"] if r["id"] == iid)["diagnosis"]


@pytest.fixture(scope="module")
def scene_a():
    return Scene(1)


def test_overlapping_sequence_with_parallax_is_ready(tmp_path, scene_a):
    rep = _run(_seq(tmp_path, scene_a))
    assert rep["graph"]["n_components"] == 1 and rep["graph"]["verified_pairs"] >= 4
    assert rep["status"] == cr.READY, rep["user_message"]
    assert rep["frame"]["shared_frame"] is None          # not attempted yet: no frame is claimed
    assert rep["diversity"]["overall"]["verdict"] == "adequate"


def test_isolated_unrelated_image_is_named_and_requested(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a) + [("odd", _save(tmp_path, "odd", Scene(99).view(0)))]
    rep = _run(items)
    d = _diag(rep, "odd")
    assert d["cause"] == "isolated" and d["certainty"] == "measured" and d["evidence"]["n_verified_links"] == 0
    assert rep["status"] == cr.NEEDS_MORE and "img-odd" in rep["user_message"] and "60%" in rep["user_message"]
    assert all(_diag(rep, f"a{i}")["cause"] is None for i in range(5))   # the connected ones are not blamed


def test_two_disconnected_groups(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 3, tag="a") + _seq(tmp_path, Scene(50), 3, tag="b")
    rep = _run(items)
    assert rep["graph"]["n_components"] == 2 and rep["status"] == cr.NEEDS_MORE
    assert any(a["kind"] == "disconnected_groups" for a in rep["actions"])
    assert "separate groups" in rep["user_message"]


def test_duplicate_is_detected_and_not_counted(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 4)
    items.append(("dup", _save(tmp_path, "dup", scene_a.view(70), ext="jpg")))     # re-encoded copy of a1
    rep = _run(items)
    assert any(r["state"] == "confirmed" for r in rep["duplicates"])
    causes = {_diag(rep, i)["cause"] for i in ("a1", "dup")}
    assert "duplicate" in causes
    assert rep["status"] == cr.READY and rep["graph"]["n_components"] == 1   # the duplicate is not an isolated image


def test_blurred_image_is_diagnosed(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 4)
    items.append(("blur", _save(tmp_path, "blur", cv2.GaussianBlur(scene_a.view(140), (0, 0), 9))))
    rep = _run(items)
    d = _diag(rep, "blur")
    assert d["cause"] == "blurred" and d["predicted_registration"] == "unlikely"
    assert rep["status"] == cr.NEEDS_MORE and "Retake img-blur" in rep["user_message"]


def test_mixed_resolution_is_reported_and_still_links(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 4)
    big = cv2.resize(scene_a.view(70), None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    items.append(("big", _save(tmp_path, "big", big)))
    rep = _run(items)
    assert rep["intrinsics"]["mixed_size"] and len(rep["intrinsics"]["size_groups"]) == 2
    assert any("different pixel sizes" in h for h in rep["intrinsics"]["hints"])
    assert _diag(rep, "big")["evidence"]["n_verified_links"] >= 1        # positive: same content at another size still matches
    assert any(a["kind"] == "mixed_size" for a in rep["actions"])


def test_rotated_crop_changes_aspect_but_links(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 4)
    rot = cv2.rotate(scene_a.view(70), cv2.ROTATE_90_CLOCKWISE)           # 600x800 portrait of the same content
    items.append(("rot", _save(tmp_path, "rot", rot)))
    rep = _run(items)
    assert len(rep["intrinsics"]["aspect_groups"]) == 2
    assert _diag(rep, "rot")["evidence"]["n_verified_links"] >= 1


def test_mixed_exif_camera_and_focal_prior(tmp_path, scene_a):
    items = [("c0", _save(tmp_path, "c0", scene_a.view(0), ext="jpg", exif={0x010F: "MakerA", 0x0110: "Cam1"})),
             ("c1", _save(tmp_path, "c1", scene_a.view(70), ext="jpg", exif={0x010F: "MakerB", 0x0110: "Cam2"})),
             ("c2", _save(tmp_path, "c2", scene_a.view(140), ext="jpg"))]
    rep = _run(items)
    assert len(rep["intrinsics"]["camera_groups_exif"]) == 2 and rep["intrinsics"]["n_with_exif"] == 2
    assert any("more than one camera" in h for h in rep["intrinsics"]["hints"])
    assert rep["intrinsics"]["distortion"]["state"] == "unknown"          # distortion is never claimed


def test_no_exif_is_reported(tmp_path, scene_a):
    rep = _run(_seq(tmp_path, scene_a, 3))
    assert rep["intrinsics"]["n_with_focal"] == 0 and any(a["kind"] == "no_exif" for a in rep["actions"])


def test_planar_rotation_like_views_are_low_parallax_not_ready(tmp_path, scene_a):
    rep = _run(_seq(tmp_path, scene_a, 5, step=40, planar=True))
    assert rep["graph"]["n_components"] == 1
    assert rep["diversity"]["overall"]["verdict"] in ("mostly_low_parallax", "narrow_baseline")
    assert rep["status"] == cr.NEEDS_MORE and any(a["kind"] == "low_parallax" for a in rep["actions"])


def test_weak_bridge_between_groups_is_flagged(tmp_path):
    s = Scene(7)
    # two tight groups joined by ONE pair that overlaps only in a narrow strip (few inliers)
    items = [("g0", _save(tmp_path, "g0", s.view(0))), ("g1", _save(tmp_path, "g1", s.view(40))),
             ("g2", _save(tmp_path, "g2", s.view(80))), ("g3", _save(tmp_path, "g3", s.view(120)))]
    rep = _run(items)
    # the sequence links fully here; build the bridge case from the graph facts instead: remove edges to leave a single weak one
    base = rep
    base["pairs"] = [p for p in base["pairs"] if {p["a"], p["b"]} in ({"g0", "g1"}, {"g2", "g3"}, {"g1", "g2"})]
    for p in base["pairs"]:
        if {p["a"], p["b"]} == {"g1", "g2"}:
            p["inliers"], p["inlier_ratio"] = 34, 0.4          # verified but barely
    rep2 = cr.after_registration(base, ["g0", "g1", "g2", "g3"])
    assert rep2["graph"]["weak_bridges"] and rep2["graph"]["weak_bridges"][0]["inliers"] == 34


def test_budget_exhausted_says_unknown_never_ready(tmp_path, scene_a):
    rep = _run(_seq(tmp_path, scene_a, 4), budget_s=0.0)
    assert rep["status"] == cr.UNKNOWN and rep["pairing"]["budget_exhausted"]
    assert all(p["status"] == "not_evaluated" for p in rep["pairs"])
    assert _diag(rep, "a0")["cause"] == "unknown" and any("could not decide" in rep["user_message"] for _ in [0])
    assert rep["cannot_decide"]


def test_partial_pair_coverage_is_unknown_not_ready(tmp_path, scene_a):
    rep = _run(_seq(tmp_path, scene_a, 5), max_pairs=3)            # forces window mode with only some pairs
    assert rep["pairing"]["mode"] == "window"
    assert rep["status"] in (cr.UNKNOWN, cr.NEEDS_MORE) and rep["status"] != cr.READY


def test_single_image_and_unreadable_inputs(tmp_path, scene_a):
    one = _run(_seq(tmp_path, scene_a, 1))
    assert one["status"] == cr.NEEDS_MORE and "at least two" in one["reason"]
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    bad = cr.assess_readiness([("j", str(junk)), ("missing", str(tmp_path / "nope.png"))])
    assert bad["status"] == cr.BLOCKED and bad["user_message"].startswith("Cannot assess")


def test_after_registration_labels_unregistered_and_never_invents_a_frame(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 4) + [("odd", _save(tmp_path, "odd", Scene(99).view(0)))]
    pre = _run(items)
    post = cr.after_registration(pre, ["a0", "a1", "a2"])
    assert post["stage"] == "post_registration" and post["frame"]["shared_frame"] is True
    assert post["frame"]["unregistered_ids"] == ["a3", "odd"] and "independent single-image" in post["frame"]["unregistered_label"]
    assert _diag(post, "odd")["cause"] == "isolated" and _diag(post, "a3")["cause"] == "other"       # linked, yet unplaced: unexplained
    assert post["status"] == cr.NEEDS_MORE and "NOT placed in a shared 3D frame" in post["user_message"]
    one = cr.after_registration(pre, ["a0"])
    assert one["frame"]["shared_frame"] is False
    none = cr.after_registration(pre, [])
    assert none["frame"]["shared_frame"] is False and none["status"] == cr.NEEDS_MORE
    full = cr.after_registration(pre, [i for i, _ in items])
    assert full["status"] == cr.READY and full["frame"]["unregistered_ids"] == []


def test_disconnected_group_cause_after_registration(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 3, tag="a") + _seq(tmp_path, Scene(50), 3, tag="b")
    post = cr.after_registration(_run(items), ["a0", "a1", "a2"])
    assert _diag(post, "b0")["cause"] == "disconnected_group"


def test_artifact_hash_is_deterministic_and_stored(tmp_path, scene_a):
    items = _seq(tmp_path, scene_a, 3)
    store = MemoryArtifactStore()
    r1, r2 = cr.store_report(_run(items), store), cr.store_report(_run(items), store)
    assert r1["artifact"]["sha256"] == r2["artifact"]["sha256"] and r1["artifact"]["uri"].startswith("artifact://")
    assert store.get(r1["artifact"]["uri"]) == cr.canonical_bytes(r1)
    entry = cr.stage_entry(r1)
    assert entry["status"] == r1["status"] and entry["artifact"]["sha256"] == r1["artifact"]["sha256"]


def _model(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[800.0, 0, 400], [0, 800.0, 300], [0, 0, 1]])
    cams = {f"c{i}": {"K": K, "R": np.eye(3), "t": np.array([-0.5 * i, 0.0, 0.0])} for i in range(4)}
    pts = []
    for _ in range(60):
        X = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(3, 6)])
        track = []
        for cid, c in cams.items():
            uv = K @ ((c["R"] @ X + c["t"]) / (c["R"] @ X + c["t"])[2])
            track.append((cid, uv[0] + rng.normal(0, noise), uv[1] + rng.normal(0, noise)))
        pts.append({"xyz": X.tolist(), "track": track})
    return {"cameras": cams, "points": pts}


def test_pose_consistency_good_and_bad_and_absent():
    good = cr.pose_consistency(_model())
    assert good["state"] == "ok" and good["verdict"] == "consistent" and good["reprojection_px_median"] < 0.01
    bad = cr.pose_consistency(_model(noise=12.0))
    assert bad["verdict"] in ("weak", "inconsistent") and bad["reprojection_px_p95"] > 4
    none = cr.pose_consistency({"cameras": {"c0": {}}, "points": []})
    assert none["state"] == "not_available"


def test_load_colmap_text_model_roundtrip(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n1 PINHOLE 800 600 800 800 400 300\n")
    (tmp_path / "images.txt").write_text("# i\n1 1 0 0 0 0 0 0 1 ev_a.png\n100 200 7\n2 1 0 0 0 -0.5 0 0 1 ev_b.png\n50 200 7\n")
    (tmp_path / "points3D.txt").write_text("# p\n7 0 0 4 1 2 3 0.5 1 0 2 0\n")
    m = cr.load_colmap_text_model(tmp_path)
    assert set(m["cameras"]) == {"ev_a", "ev_b"} and len(m["points"]) == 1 and len(m["points"][0]["track"]) == 2
    assert cr.pose_consistency(m)["state"] == "ok"
    assert cr.load_colmap_text_model(tmp_path / "missing") is None


def test_for_job_never_raises_and_flags_missing_files(tmp_path):
    rep = cr.for_job([("e1", str(tmp_path / "x.png"), "x.png")], store=None)
    assert rep["status"] == cr.BLOCKED and rep["artifact"]["sha256"]
    assert cr.for_job_after(None, []) is None
