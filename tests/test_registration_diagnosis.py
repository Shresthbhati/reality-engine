"""Registration diagnosis: parsing COLMAP's database/model/log, per-image rejection reasons, root-cause rules and the
user-facing guidance. Everything runs on a hand-built COLMAP database + text model (headless, no COLMAP); the one test
that drives the real binary skips when it is absent."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from reconstruction import registration_diagnosis as rd

MAXID = 2147483647
F = 500.0


def _pair(a, b):
    return a * MAXID + b


def _build_db(path: Path, n_images=8, edges=None, keypoints=200):
    """images 1..n; ``edges`` = {(a, b): (raw_matches, inlier_pairs[(i, j)...], config)}."""
    con = sqlite3.connect(path)
    con.executescript("""
    create table cameras (camera_id integer primary key, model integer, width integer, height integer, params blob, prior_focal_length integer);
    create table images (image_id integer primary key, name text, camera_id integer);
    create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);
    create table matches (pair_id integer primary key, rows integer, cols integer, data blob);
    create table two_view_geometries (pair_id integer primary key, rows integer, cols integer, data blob, config integer,
                                      F blob, E blob, H blob, qvec blob, tvec blob, camera1 integer, camera2 integer);
    """)
    con.execute("insert into cameras values (1, 2, 640, 480, ?, 0)", (np.array([F, 320.0, 240.0, 0.0]).tobytes(),))
    rng = np.random.default_rng(0)
    for i in range(1, n_images + 1):
        con.execute("insert into images values (?, ?, 1)", (i, f"IMG_{i:04d}.jpg"))
        kp = rng.uniform([0, 0], [640, 480], size=(keypoints, 2)).astype(np.float32)
        con.execute("insert into keypoints values (?, ?, 2, ?)", (i, keypoints, kp.tobytes()))
    for a in range(1, n_images + 1):
        for b in range(a + 1, n_images + 1):
            raw, inl, cfg = (edges or {}).get((a, b), (0, [], 0))
            con.execute("insert into matches values (?, ?, 2, ?)", (_pair(a, b), raw, b"" if not raw else np.zeros((raw, 2), np.uint32).tobytes()))
            arr = np.array(inl, dtype=np.uint32).reshape(-1, 2)
            con.execute("insert into two_view_geometries values (?, ?, 2, ?, ?, null, null, null, null, null, 1, 1)",
                        (_pair(a, b), len(arr), arr.tobytes() if len(arr) else b"", cfg))
    con.commit()
    con.close()
    return path


def _write_model(d: Path, registered=(1, 2, 3), n_points=60, keypoints=200, linked_to=None):
    """Text model: cameras ``registered`` look at a plane z=5, spaced 1 unit along x; point k is keypoint k of each image.
    Projections are exact, so the reprojection error must come back ~0."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "cameras.txt").write_text(f"1 SIMPLE_RADIAL 640 480 {F} 320 240 0\n")
    rng = np.random.default_rng(1)
    X = np.c_[rng.uniform(-1, 3, n_points), rng.uniform(-1, 1, n_points), np.full(n_points, 5.0)]
    lines, tracks = [], {k: [] for k in range(n_points)}
    for iid in registered:
        t = np.array([-(iid - 1) * 1.0, 0, 0])       # x_cam = X + t  -> camera centre at +(iid-1) on x
        pc = X + t
        uv = np.c_[F * pc[:, 0] / pc[:, 2] + 320, F * pc[:, 1] / pc[:, 2] + 240]
        pts2d = []
        for k in range(keypoints):
            if k < n_points:
                pts2d.append(f"{uv[k, 0]} {uv[k, 1]} {k + 1}")
                tracks[k].append((iid, k))
            else:
                pts2d.append("1.0 1.0 -1")
        lines.append(f"{iid} 1 0 0 0 {t[0]} {t[1]} {t[2]} 1 IMG_{iid:04d}.jpg")
        lines.append(" ".join(pts2d))
    (d / "images.txt").write_text("\n".join(lines) + "\n")
    (d / "points3D.txt").write_text("\n".join(
        f"{k + 1} {X[k, 0]} {X[k, 1]} {X[k, 2]} 9 9 9 0.1 " + " ".join(f"{i} {j}" for i, j in tracks[k]) for k in range(n_points)) + "\n")
    return d


@pytest.fixture()
def scene(tmp_path):
    inl = lambda n: [(i, i) for i in range(n)]  # noqa: E731  verified inlier matches: keypoint i <-> keypoint i
    edges = {
        (1, 2): (120, inl(100), 3), (2, 3): (120, inl(90), 3), (1, 3): (100, inl(80), 3),   # registered chain
        (4, 5): (60, inl(40), 3),                                                              # a pair nobody else matches
        (1, 7): (40, [(i, i) for i in range(100, 125)], 3),                                    # edge to model, no 3D reach
        (2, 8): (150, inl(55), 6),                                                             # 55 keypoints that DO hit 3D points
    }
    db = rd.read_database(_build_db(tmp_path / "db.db", edges=edges))
    model = rd.read_model(_write_model(tmp_path / "m"))
    return db, model


def test_database_pairs_inlier_counts_and_ratio(scene):
    db, _ = scene
    assert len(db["images"]) == 8 and db["cameras"][1]["params"][0] == F
    assert db["pairs"][(1, 2)]["inliers"] == 100 and db["raw_matches"][(1, 2)] == 120
    rows = {(r["a"], r["b"]): r for r in rd.pair_table(db)}
    assert rows[("IMG_0001.jpg", "IMG_0002.jpg")]["inlier_ratio"] == pytest.approx(100 / 120, abs=1e-3)
    assert rows[("IMG_0002.jpg", "IMG_0008.jpg")]["config"] == "planar_or_panoramic"


def test_graph_components_isolation_and_rotation_share(scene):
    db, _ = scene
    g = rd.graph_summary(db)
    assert [len(c) for c in g["components"]] == [5, 2, 1]
    assert "IMG_0006.jpg" in g["isolated_images"] and "IMG_0001.jpg" not in g["isolated_images"]
    assert g["edges_ge_15"] == 6 and g["edges_ge_30"] == 5   # (1,7) has 25 inliers
    assert g["rotation_only_share"] == pytest.approx(1 / 6, abs=1e-3)


def test_model_reprojection_is_zero_and_triangulation_angle_sane(scene):
    _, model = scene
    mm = rd.model_metrics(model)
    errs = [r["reproj_rms_px"] for r in mm["per_image"].values()]
    assert len(errs) == 3 and max(errs) < 0.01          # own projection reproduces the exact synthetic geometry
    assert all(r["observations"] == 60 for r in mm["per_image"].values())
    assert 5 < mm["median_tri_angle_deg"] < 40            # 2-unit baseline at depth 5


def test_every_unregistered_image_gets_a_derived_reason_with_numbers(scene):
    db, model = scene
    reg = {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"}
    u = rd.unregistered_reasons(db, [model], reg)
    assert set(u) == {f"IMG_000{i}.jpg" for i in (4, 5, 6, 7, 8)}
    assert u["IMG_0006.jpg"]["reason"] == "no_verified_overlap" and u["IMG_0006.jpg"]["best_pair_inliers"] == 0
    assert u["IMG_0004.jpg"]["reason"] == "disconnected_component" and u["IMG_0005.jpg"]["reason"] == "disconnected_component"
    assert u["IMG_0007.jpg"]["reason"] == "insufficient_2d3d_inliers" and u["IMG_0007.jpg"]["candidate_2d3d_best_model"] == 0
    # image 8 shares 55 inliers with image 2, whose matched keypoints (0..54) all carry a 3D point
    assert u["IMG_0008.jpg"]["reason"] == "pose_estimation_rejected" and u["IMG_0008.jpg"]["candidate_2d3d_best_model"] == 55


def test_parse_mapper_log_initial_pair_order_and_no_failure_line():
    log = "\n".join([
        "I Finding good initial image pair", "I Registering initial image pair #20 and #23",
        "I Registering image #26 (num_reg_frames=3)", "I => Image sees 136 / 283 points",
        "I Registering image #9 (num_reg_frames=4)", "I => Image sees 145 / 263 points",
        "I Keeping successful reconstruction", "I => No good initial image pair found.",
        "I Discarding reconstruction due to no initial pair", "I => Relaxing the initialization constraints."])
    p = rd.parse_mapper_log(log)
    assert p["initial_pairs"] == [[20, 23]] and [r["image_id"] for r in p["registered_order"]] == [26, 9]
    assert p["registered_order"][0]["sees"] == 136 and p["no_initial_pair"] == 1 and p["relaxed_initialisation"] == 1


def _report(db, model, reg, quals=None, attempts=None):
    qual = quals or {im["name"]: {"blur_laplacian_var": 300.0, "gradient_anisotropy": 1.2, "edge_density": 0.05,
                                  "width": 640, "height": 480} for im in db["images"].values()}
    return rd.analyse(db, [model] if reg else [], attempts or [{"attempt": 0, "args": [], "registered": len(reg), "points": 60}],
                      {"attempt0": {"initial_pairs": [[1, 2]] if reg else []}}, qual, {}, {n: (640, 480) for n in qual}, 0)


def test_low_texture_is_named_with_its_numbers_and_becomes_guidance(tmp_path):
    db = rd.read_database(_build_db(tmp_path / "t.db", n_images=6, keypoints=40))      # 40 keypoints per image
    quals = {im["name"]: {"blur_laplacian_var": 8.0, "gradient_anisotropy": 1.1, "edge_density": 0.002, "width": 640, "height": 480}
             for im in db["images"].values()}
    rep = _report(db, None, set(), quals)
    causes = {c["cause"]: c for c in rep["root_causes"]}
    assert causes["low_texture"]["evidence"]["median_keypoints"] == 40 and causes["low_texture"]["confidence"] == "high"
    assert "no_initial_pair" in causes and "insufficient_overlap" in causes
    g = rd.capture_guidance(rep)
    kinds = {i["kind"]: i for i in g}
    assert "40 keypoints" in kinds["low_texture"]["basis"] and "textured" in kinds["low_texture"]["message"]
    assert all(set(i) == {"kind", "message", "basis"} for i in g)       # the shape engine.pipeline.guidance emits


def test_motion_blur_needs_directional_blur_not_just_softness(tmp_path):
    db = rd.read_database(_build_db(tmp_path / "t.db", n_images=6))
    soft = {im["name"]: {"blur_laplacian_var": 20.0, "gradient_anisotropy": 1.2, "edge_density": 0.05, "width": 640, "height": 480}
            for im in db["images"].values()}
    streak = {k: dict(v, gradient_anisotropy=4.0) for k, v in soft.items()}
    assert "motion_blur" not in {c["cause"] for c in _report(db, None, set(), soft)["root_causes"]}
    assert "motion_blur" in {c["cause"] for c in _report(db, None, set(), streak)["root_causes"]}


def test_planar_label_is_not_rotation_only_when_the_model_has_baseline(scene, tmp_path):
    db, model = scene
    rep = _report(db, model, {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"})
    assert "rotation_only_or_planar" not in {c["cause"] for c in rep["root_causes"]}      # tri angle ~20 deg
    # same graph, but every pair labelled panoramic and no model: that IS the rotation-only signature
    edges = {(a, b): (50, [(i, i) for i in range(40)], 5) for a in range(1, 5) for b in range(a + 1, 5)}
    db2 = rd.read_database(_build_db(tmp_path / "r.db", n_images=4, edges=edges))
    c2 = {c["cause"]: c for c in _report(db2, None, set())["root_causes"]}
    assert c2["rotation_only_or_planar"]["evidence"]["panoramic_pair_share"] == 1.0


def test_no_findings_or_guidance_when_registration_is_fine(scene):
    db, model = scene
    names = {im["name"] for im in db["images"].values()}
    rep = rd.analyse(db, [model], [{"attempt": 0, "args": [], "registered": 8, "points": 60}], {}, {}, {}, {}, 0)
    rep["registered"]["count"] = len(names)          # all placed
    assert rd.capture_guidance(rep) == [] and rd.classify_causes(rep) == []


def test_findings_are_correction_engine_diagnostics(scene):
    db, model = scene
    rep = _report(db, model, {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"})
    types = {d["type"] for d in rep["diagnostics"]}
    assert "UNREGISTERED_IMAGE" in types                  # same type reconstruction/correction.py already consumes
    json.dumps(rep["diagnostics"])


def test_subset_selection_is_deterministic_and_chain_is_weaker_than_dense(scene):
    db, _ = scene
    a = rd.select_subset(db, 4, "dense_overlap")
    assert a == rd.select_subset(db, 4, "dense_overlap") and len(a) == 4 and len(set(a)) == 4
    assert rd.select_subset(db, 4, "stride") == ["IMG_0001.jpg", "IMG_0003.jpg", "IMG_0005.jpg", "IMG_0007.jpg"]
    assert len(rd.select_subset(db, 3, "sparse_chain")) == 3


def _colmap():
    for c in (os.environ.get("COLMAP_BINARY"), shutil.which("colmap"), "C:/Users/shres/tools/colmap-extracted/bin/colmap"):
        if c and (shutil.which(c) or Path(c).exists()) and rd.colmap_available(c):
            return c
    return None


@pytest.mark.skipif(_colmap() is None, reason="COLMAP binary not available")
def test_real_colmap_run_end_to_end_on_small_textured_images(tmp_path):
    import cv2

    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (900, 1200), dtype=np.uint8), (0, 0), 1.6)
    files = []
    for k in range(5):
        x = 60 + 70 * k                           # lateral shifts of one textured wall: overlapping views
        p = tmp_path / "src" / f"p{k}.jpg"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), tex[200:680, x:x + 640])
        files.append(p)
    rep = rd.diagnose_image_set(files, tmp_path / "work", _colmap(), label="unit")
    assert rep["registered"]["of"] == 5 and set(rep["images"]) == {f.name for f in files}
    assert all(v["keypoints"] > 0 for v in rep["images"].values())
    assert rep["graph"]["edges_ge_15"] > 0 and (tmp_path / "work" / "diagnosis.json").is_file()
    assert (tmp_path / "work" / "summary.txt").read_text().startswith("[unit] registered")
