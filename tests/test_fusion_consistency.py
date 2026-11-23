"""Fusion consistency, input validation, provenance and mesh diagnostics (engine/multiengine/depth_fusion.py).

Synthetic scene with exact ground truth (the analytic room of tests/test_depth_fusion.py: floor z=0, ceiling z=2.5, walls x=4, y=3).
Everything before and after the TSDF step is pure numpy, so these tests run WITHOUT Open3D: the TSDF step (``_integrate_and_extract``) is
replaced by an analytic room mesh. They prove the screening / probes / consistency / provenance / diagnostics logic, not the TSDF itself;
``test_real_open3d_fusion_end_to_end`` runs the real integrator where Open3D is installed (skipped otherwise)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from engine.math import Quat, Vec3
from engine.multiengine import depth_fusion as df
from engine.multiengine.depth_fusion import (
    FusionInputError,
    FusionView,
    _quat_to_R,
    fuse_depth_to_surface,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, matrix_to_quat

W, H, FX = 160, 120, 140.0
PLANES = [((0, 0, 1), 0.0), ((0, 0, 1), 2.5), ((1, 0, 0), 4.0), ((0, 1, 0), 3.0)]
POSES = [((1.0, 1.0, 1.5), (4, 3, 1.2)), ((2.0, 0.6, 1.4), (4, 1.5, 1.3)), ((0.6, 2.0, 1.5), (2, 3, 1.2)), ((1.6, 1.6, 1.2), (4, 3, 0.5)),
         ((2.5, 1.0, 1.3), (4, 1.0, 1.2))]


def _cam_from_R(pos, R) -> PinholeCamera:
    q = matrix_to_quat(R)
    return PinholeCamera(CameraIntrinsics(FX, FX, W / 2, H / 2, W, H), CameraExtrinsics(Vec3(*pos), Quat(q.w, q.x, q.y, q.z)))


def _R_look(pos, look) -> np.ndarray:
    f = np.asarray(look, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f], axis=1)


def _camera(pos, look) -> PinholeCamera:
    return _cam_from_R(pos, _R_look(pos, look))


def _depth(cam: PinholeCamera) -> np.ndarray:
    e = cam.extrinsics
    R = _quat_to_R((e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z))
    t = np.array([e.position.x, e.position.y, e.position.z])
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / FX, (v - H / 2) / FX, np.ones_like(u, float)], -1) @ R.T
    best = np.full((H, W), np.inf)
    for n, c in PLANES:
        n = np.asarray(n, float)
        denom = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (c - n @ t) / denom
        best = np.minimum(best, np.where((denom != 0) & (s > 0), s, np.inf))
    best[~np.isfinite(best)] = 0.0
    return best.astype(np.float32)


def _views(n=5):
    out = []
    for i, (p, lk) in enumerate(POSES[:n]):
        cam = _camera(p, lk)
        out.append(FusionView(f"v{i}", np.full((H, W, 3), 150, np.uint8), _depth(cam), cam))
    return out


def _replace(views, i, **kw):
    v = views[i]
    d = dict(source_id=v.source_id, color=v.color, depth=v.depth, camera=v.camera)
    d.update(kw)
    views = list(views)
    views[i] = FusionView(**d)
    return views


def _room_mesh(step=0.1):
    """The analytic room as a mesh with normals pointing INTO the room (toward the cameras)."""
    verts, faces = [], []

    def plane(origin, a, b, na, nb):
        base = len(verts)
        for i in range(na + 1):
            for j in range(nb + 1):
                verts.append(np.asarray(origin) + a * i / na + b * j / nb)
        for i in range(na):
            for j in range(nb):
                p = base + i * (nb + 1) + j
                faces.extend([[p, p + nb + 1, p + 1], [p + 1, p + nb + 1, p + nb + 2]])
    n = lambda L: int(round(L / step))  # noqa: E731
    plane((0, 0, 0), np.array([4.0, 0, 0]), np.array([0, 3.0, 0]), n(4), n(3))          # floor, normal +z
    plane((0, 0, 2.5), np.array([0, 3.0, 0]), np.array([4.0, 0, 0]), n(3), n(4))        # ceiling, normal -z
    plane((4, 0, 0), np.array([0, 0, 2.5]), np.array([0, 3.0, 0]), n(2.5), n(3))        # wall x=4, normal -x
    plane((0, 3, 0), np.array([4.0, 0, 0]), np.array([0, 0, 2.5]), n(4), n(2.5))        # wall y=3, normal -y
    return np.array(verts), np.array(faces)


@pytest.fixture
def fake_tsdf(monkeypatch):
    """Replace ONLY the Open3D step; everything else (screening, probes, consistency, masks, provenance, diagnostics) is the real code."""
    captured: dict = {}

    def fake(views, kept_depth, *, should_stop=None, **kw):
        captured["kept_depth"] = list(kept_depth)
        integrated, interrupted = [], None
        for k, v in enumerate(views):
            if should_stop is not None and should_stop(k):
                interrupted = f"stop requested before view {k} ({v.source_id})"
                break
            integrated.append(k)
        verts, faces = _room_mesh() if integrated else (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        return {"integrated": integrated, "interrupted": interrupted, "vertices": verts, "faces": faces, "colors": None, "raw_triangles": len(faces),
                "removed_floating_specks_triangles": 0, "removed_zero_area_triangles": 0, "suppressed_coincident_triangles": 0}

    monkeypatch.setattr(df, "_integrate_and_extract", fake)
    return captured


# ------------------------------------------------------------------------------------------------------------------ positive path
def test_clean_fusion_is_complete_and_every_view_is_traceable(fake_tsdf):
    res = fuse_depth_to_surface(_views())
    assert res.status == "ran" and res.complete and res.error_code is None and not res.report["views_excluded"]
    assert all(p["contributed_to_surface"] and p["depth_sha256"] for p in res.report["per_view"])
    prov = json.loads(res.provenance_json)
    assert prov["schema"] == "fusion_provenance/v1" and "NOT per voxel" in prov["granularity"] and prov["complete"] is True
    assert [v["source_id"] for v in prov["views"]] == [f"v{i}" for i in range(5)]
    assert all(v["pose"]["position"] and len(v["depth_sha256"]) == 64 for v in prov["views"])
    import hashlib

    assert prov["mesh_sha256"] == hashlib.sha256(res.mesh_ply).hexdigest() == res.report["provenance"]["mesh_sha256"]
    assert prov["vertex_table"]["sha256"] == hashlib.sha256(res.provenance_npz).hexdigest()
    assert res.report["provenance"]["json_sha256"] == hashlib.sha256(res.provenance_json).hexdigest()
    import io

    tab = np.load(io.BytesIO(res.provenance_npz))
    assert tab["contrib"].shape == (len(res.vertices), 5) and tab["contrib"].any(axis=0).all()
    assert np.nanmedian(np.abs(tab["residual"])) < 0.05                      # supporting views really lie on those vertices
    assert prov["components"][0]["views"]                                    # component -> supporting views
    r = res.report
    assert r["geometry_confidence"]["mean_vertex_confidence"] > 0 and r["unsupported_regions"]["statement"]
    assert 0 < r["mesh_support"]["vertices_supported_by_ge2_views"] <= r["mesh_support"]["vertices_supported_by_ge1_view"] <= 1


def test_the_room_mesh_faces_the_cameras_and_flipping_it_is_detected(fake_tsdf):
    res = fuse_depth_to_surface(_views())
    o = res.report["mesh"]["orientation_vs_viewing_rays"]
    assert o["triangles_evaluated"] > 100 and o["flipped_fraction"] < 0.02 and not o["globally_inverted"]
    import io

    contrib = np.load(io.BytesIO(res.provenance_npz))["contrib"]
    centres = np.stack([df._pose(v.camera)[1] for v in _views()])
    flipped = df.orientation_vs_views(res.vertices, res.faces[:, ::-1].copy(), contrib, centres)
    assert flipped["globally_inverted"] and flipped["flipped_fraction"] > 0.98
    one = res.faces.copy()
    seen = np.flatnonzero(contrib[one].all(axis=1).any(axis=1))[:50]       # reverse 50 triangles that some view actually supports
    one[seen] = one[seen, ::-1]
    local = df.orientation_vs_views(res.vertices, one, contrib, centres)
    assert 0 < local["flipped_triangles"] <= 50 and not local["globally_inverted"]   # a handful of reversed triangles is local, not global


# ------------------------------------------------------------------------------------------------------------------ wrong camera transforms
def test_a_reversed_camera_is_excluded_with_a_typed_code_and_the_result_is_partial(fake_tsdf):
    views = _views()
    p, lk = POSES[2]
    R = _R_look(p, lk) @ np.diag([-1.0, 1.0, -1.0])      # camera looks the opposite way
    views = _replace(views, 2, camera=_cam_from_R(p, R))
    res = fuse_depth_to_surface(views)
    ex = res.report["views_excluded"]
    assert [e["source"] for e in ex] == ["v2"] and ex[0]["code"] == "WRONG_CAMERA_TRANSFORM"
    assert res.status == "partial" and res.complete is False and "v2" in res.report["unsupported_regions"]["views_missing"]
    assert json.loads(res.provenance_json)["complete"] is False
    with pytest.raises(FusionInputError) as e:
        fuse_depth_to_surface(views, strict=True)
    assert e.value.code == "WRONG_CAMERA_TRANSFORM"


def test_world_to_camera_given_as_camera_to_world_is_detected_for_one_view(fake_tsdf):
    views = _views()
    p, lk = POSES[1]
    R = _R_look(p, lk)
    t = np.asarray(p, float)
    bad = _cam_from_R(tuple(-R.T @ t), R.T)               # the inverse transform stored where camera-to-world belongs
    res = fuse_depth_to_surface(_replace(views, 1, camera=bad))
    ex = res.report["views_excluded"]
    assert ex and ex[0]["source"] == "v1" and ex[0]["code"] == "WRONG_CAMERA_TRANSFORM"
    assert res.report["input_validation"]["probe"]["views"][1]["suggested_convention"].startswith("inverse_world_to_camera")
    assert res.status == "partial"


def test_a_whole_pose_set_in_the_wrong_convention_is_refused_not_fused(fake_tsdf):
    views = []
    for v, (p, lk) in zip(_views(), POSES, strict=True):
        R = _R_look(p, lk)
        views.append(FusionView(v.source_id, v.color, v.depth, _cam_from_R(tuple(-R.T @ np.asarray(p, float)), R.T)))
    res = fuse_depth_to_surface(views)
    assert res.status == "skipped" and res.error_code == "WRONG_CAMERA_TRANSFORM" and res.mesh_ply is None
    assert res.report["input_validation"]["wrong_convention_for_all_views"].startswith("inverse_world_to_camera")


# ------------------------------------------------------------------------------------------------------------------ depth scale
def test_a_view_with_a_different_depth_scale_is_excluded_and_its_factor_estimated(fake_tsdf):
    views = _views()
    views = _replace(views, 3, depth=views[3].depth * 2.0)
    res = fuse_depth_to_surface(views)
    ex = res.report["views_excluded"]
    assert [e["source"] for e in ex] == ["v3"] and ex[0]["code"] == "DEPTH_SCALE_MISMATCH"
    est = res.report["input_validation"]["probe"]["views"][3]["estimated_depth_scale_factor"]
    assert est == pytest.approx(0.5, rel=0.1)                # multiplying its depth by ~0.5 makes it agree with the rest
    assert res.status == "partial" and res.report["views_integrated"] == ["v0", "v1", "v2", "v4"]


def test_two_views_that_disagree_by_a_scale_cannot_be_attributed_and_nothing_is_fused(fake_tsdf):
    views = _views(2)
    views = _replace(views, 1, depth=views[1].depth * 2.0)
    res = fuse_depth_to_surface(views)
    assert res.status == "skipped" and res.mesh_ply is None and res.error_code == "DEPTH_SCALE_MISMATCH"
    assert res.report["input_validation"].get("scale_attribution_ambiguous") is True


# ------------------------------------------------------------------------------------------------------------------ corrupted / missing / single / disconnected
def test_scattered_nan_inf_and_negative_depth_is_counted_and_fusion_still_runs(fake_tsdf):
    views = _views()
    d = views[0].depth.copy()
    d[10, 10:20] = np.nan
    d[11, 10:20] = np.inf
    d[12, 10:20] = -1.0
    res = fuse_depth_to_surface(_replace(views, 0, depth=d))
    pv = res.report["per_view"][0]
    assert pv["invalid_nonfinite"] == 20 and pv["invalid_nonpositive"] == 10 and res.status == "ran"
    assert d[10, 10] != d[10, 10]                              # the caller's array was not modified


def test_mostly_corrupt_or_missing_depth_excludes_the_view_with_a_typed_code(fake_tsdf):
    views = _views()
    d = views[1].depth.copy()
    d[: int(H * 0.8)] = np.nan
    res = fuse_depth_to_surface(_replace(views, 1, depth=d))
    assert res.report["views_excluded"][0]["code"] == "CORRUPT_DEPTH" and res.status == "partial"
    res = fuse_depth_to_surface(_replace(views, 1, depth=None))
    assert res.report["views_excluded"][0]["code"] == "MISSING_ARTIFACT"
    res = fuse_depth_to_surface(_replace(views, 1, depth=np.zeros(5)))
    assert res.report["views_excluded"][0]["code"] == "CORRUPT_DEPTH"
    res = fuse_depth_to_surface(_replace(views, 1, color=np.zeros((3, 3, 3), np.uint8)))
    assert res.report["views_excluded"][0]["code"] == "SHAPE_MISMATCH"
    with pytest.raises(FusionInputError) as e:
        fuse_depth_to_surface(_replace(views, 1, depth=None), strict=True)
    assert e.value.code == "MISSING_ARTIFACT" and e.value.source == "v1"


def test_missing_and_corrupt_artifact_files_raise_typed_errors(tmp_path):
    from PIL import Image

    cam = _camera(*POSES[0])
    Image.new("RGB", (W, H)).save(tmp_path / "c.png")
    with pytest.raises(FusionInputError) as e:
        df.load_view("a", tmp_path / "c.png", tmp_path / "missing.npy", cam)
    assert e.value.code == "MISSING_ARTIFACT"
    (tmp_path / "bad.npy").write_bytes(b"not a numpy file")
    with pytest.raises(FusionInputError) as e:
        df.load_view("a", tmp_path / "c.png", tmp_path / "bad.npy", cam)
    assert e.value.code == "CORRUPT_DEPTH"
    np.save(tmp_path / "d.npy", _depth(cam))
    v = df.load_view("a", tmp_path / "c.png", tmp_path / "d.npy", cam)
    import hashlib

    assert v.depth_sha256 == hashlib.sha256((tmp_path / "d.npy").read_bytes()).hexdigest()   # provenance names the exact artifact file


def test_a_single_view_and_all_unusable_views_are_never_fused(fake_tsdf):
    one = fuse_depth_to_surface(_views()[:1])
    assert one.status == "skipped" and "two views" in one.reason and one.error_code == "SINGLE_VIEW" and one.mesh_ply is None
    nothing = fuse_depth_to_surface(_replace(_replace(_views(2), 0, depth=None), 1, depth=None))
    assert nothing.status == "skipped" and nothing.error_code == "MISSING_ARTIFACT"


def test_a_view_with_no_verified_overlap_is_excluded_not_fused_on_trust(fake_tsdf):
    views = _views()
    far = _camera((100, 100, 1.5), (101, 100, 1.5))
    views.append(FusionView("far", np.full((H, W, 3), 90, np.uint8), np.full((H, W), 3.0, np.float32), far))   # a wall 3 units away, in a different place
    res = fuse_depth_to_surface(views)
    ex = res.report["views_excluded"]
    assert [(e["source"], e["code"]) for e in ex] == [("far", "DISCONNECTED_VIEW_GRAPH")] and res.status == "partial"
    assert res.report["input_validation"]["view_graph"]["connected"] is False
    assert res.report["input_validation"]["view_graph"]["components"][1] == ["far"]


# ------------------------------------------------------------------------------------------------------------------ occlusion, inconsistency, confidence
def test_depth_discontinuity_pixels_are_masked_and_counted():
    d = np.full((20, 20), 2.0)
    d[5:10, 5:10] = 1.0                                         # a foreground box: a depth jump on its border
    m = df._discontinuity_mask(d, 0.10)
    assert m[5, 5] and m[4, 5] and m[9, 9] and m[10, 9] and not m[7, 7] and not m[0, 0]   # both sides of the jump, not the flat interior
    assert not df._discontinuity_mask(np.linspace(2.0, 2.2, 400).reshape(20, 20), 0.10).any()   # a smooth slope is not an occlusion boundary


def test_foreground_occluder_boundary_is_masked_and_its_interior_is_rejected_as_inconsistent(fake_tsdf):
    views = _views()
    d = views[0].depth.copy()
    d[40:80, 60:100] *= 0.5                                      # a phantom foreground patch no other view saw
    res = fuse_depth_to_surface(_replace(views, 0, depth=d))
    pv = res.report["per_view"][0]
    assert pv["rejected_occlusion_boundary"] > 0 and pv["rejected_as_inconsistent"] > 0
    kept0 = fake_tsdf["kept_depth"][0]
    assert (kept0[45:75, 65:95] == 0).mean() > 0.3                # the contradicted part of the phantom never reaches the integrator
    # (patch pixels no other view can see stay "unsupported single view": kept and counted as such, not claimed as consistent)
    assert pv["unsupported_single_view"] > 0
    assert kept0[10:30, 10:30].any()                              # the rest of the view does
    assert pv["integrated_pixels"] + pv["rejected_as_inconsistent"] + pv["rejected_low_confidence"] <= pv["depth_pixels"]


def test_caller_confidence_gates_integration_and_derived_confidence_is_recorded(fake_tsdf):
    views = _views()
    conf = np.ones((H, W), np.float32)
    conf[:, :40] = 0.05
    res = fuse_depth_to_surface(_replace(views, 2, confidence=conf))
    pv = res.report["per_view"][2]
    assert pv["confidence_source"] == "caller" and pv["rejected_low_confidence"] > 0
    assert (fake_tsdf["kept_depth"][2][:, :40] == 0).all()
    assert res.report["per_view"][0]["confidence_source"] == "derived_from_cross_view_consistency" and 0.25 <= res.report["per_view"][0]["mean_confidence_integrated"] <= 1
    bad = fuse_depth_to_surface(_replace(views, 2, confidence=np.ones((3, 3))))
    assert bad.report["views_excluded"][0]["code"] == "SHAPE_MISMATCH"


# ------------------------------------------------------------------------------------------------------------------ interruption / atomic persistence
def test_an_interrupted_fusion_is_partial_never_complete(fake_tsdf, tmp_path):
    res = fuse_depth_to_surface(_views(), should_stop=lambda k: k >= 3)
    assert res.status == "partial" and res.complete is False and res.error_code == "INTERRUPTED"
    assert res.report["views_integrated"] == ["v0", "v1", "v2"] and res.report["views_not_integrated"] == ["v3", "v4"]
    assert res.report["complete"] is False and any("interrupted" in d for d in res.report["degradations"])
    m = df.persist_fusion(res, tmp_path, "fused")
    assert m["complete"] is False and m["mesh"]["path"].endswith("fused.partial.ply")        # a partial surface is never named like a complete one
    assert not (tmp_path / "fused.ply").exists()
    none = fuse_depth_to_surface(_views(), should_stop=lambda k: True)
    assert none.status == "skipped" and none.error_code == "INTERRUPTED" and none.mesh_ply is None


def test_a_crash_mid_integration_keeps_the_done_views_as_a_partial_result(monkeypatch):
    # an exception inside the integrator loop is converted by the real function; emulate it with the same contract the real one returns
    def boom(views, kept, **kw):
        verts, faces = _room_mesh()
        return {"integrated": [0, 1], "interrupted": "integration failed at view 2 (v2): RuntimeError: x", "vertices": verts, "faces": faces, "colors": None,
                "raw_triangles": len(faces), "removed_floating_specks_triangles": 0, "removed_zero_area_triangles": 0, "suppressed_coincident_triangles": 0}

    monkeypatch.setattr(df, "_integrate_and_extract", boom)
    res = fuse_depth_to_surface(_views())
    assert res.status == "partial" and not res.complete and res.report["views_not_integrated"] == ["v2", "v3", "v4"]


def test_persist_is_atomic_and_leaves_no_half_written_mesh(fake_tsdf, tmp_path, monkeypatch):
    res = fuse_depth_to_surface(_views())
    out = tmp_path / "out"
    m = df.persist_fusion(res, out, "fused")
    import hashlib

    assert (out / "fused.ply").read_bytes() == res.mesh_ply and m["mesh"]["sha256"] == hashlib.sha256(res.mesh_ply).hexdigest() and m["complete"] is True
    assert sorted(p.name for p in out.iterdir()) == ["fused.ply", "fused.provenance.json", "fused.provenance.npz"]      # no temp files left

    out2 = tmp_path / "out2"
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:                                     # the mesh rename, after both sidecars were placed
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(df.os, "replace", flaky)
    with pytest.raises(OSError):
        df.persist_fusion(res, out2, "fused")
    assert list(out2.iterdir()) == []                           # sidecars rolled back, temps removed: nothing half-written
    with pytest.raises(FusionInputError):
        df.persist_fusion(fuse_depth_to_surface(_views()[:1]), out2, "x")   # a skipped result has nothing to persist


# ------------------------------------------------------------------------------------------------------------------ mesh diagnostics (pure meshes)
def _grid(n=10, hole=None):
    v = np.array([[i, j, 0.0] for i in range(n + 1) for j in range(n + 1)])
    f = []
    for i in range(n):
        for j in range(n):
            if hole and hole[0] <= i < hole[1] and hole[0] <= j < hole[1]:
                continue
            p = i * (n + 1) + j
            f.extend([[p, p + n + 1, p + 1], [p + 1, p + n + 1, p + n + 2]])
    return v, np.array(f)


def test_boundary_loops_count_holes_and_their_area():
    v, f = _grid(10, hole=(4, 6))                                # a 2 x 2 hole in a 10 x 10 sheet
    r = df._validate_mesh(v, f, voxel=1.0)
    bl = r["boundary_loops"]
    assert bl["count"] == 2 and bl["largest"][0]["approx_area_gauge2"] == pytest.approx(100.0) and bl["largest"][1]["approx_area_gauge2"] == pytest.approx(4.0)
    assert all(lp["closed"] for lp in bl["largest"]) and r["valid"] and r["components"] == 1
    assert df._validate_mesh(*_grid(10), voxel=1.0)["boundary_loops"]["count"] == 1      # only the outer border


def test_non_manifold_edges_and_inconsistent_winding_are_reported_as_problems():
    v, f = _grid(2)
    f2 = np.vstack([f, [[0, 1, 5], [1, 0, 6]]])                   # two more triangles on the edge (0, 1): three triangles share it
    assert df._validate_mesh(v, f2)["non_manifold_edges"] >= 1 and not df._validate_mesh(v, f2)["valid"]
    g = f.copy()
    g[1] = g[1][::-1]                                            # one triangle wound the other way next to its neighbour
    r = df._validate_mesh(v, g)
    assert r["inconsistent_winding_edges"] > 0 and any("winding" in p for p in r["problems"])
    assert df._validate_mesh(v, f)["inconsistent_winding_edges"] == 0


def test_coincident_sheets_are_detected_and_the_duplicate_suppressed():
    v, f = _grid(6)
    v2 = np.vstack([v, v + [0, 0, 0.01]])                        # the same sheet again, 0.01 above it
    f2 = np.vstack([f, f + len(v)])
    co = df._coincident_triangles(v2, f2, 0.1)
    assert co["same_direction_pairs"] > 0 and co["suppressed_triangles"] == len(f)
    assert not co["remove"][: len(f)].any() and co["remove"][len(f):].all()                # the first sheet is kept, its twin removed
    back = df._coincident_triangles(v2, np.vstack([f, f[:, ::-1] + len(v)]), 0.1)
    assert back["back_to_back_pairs"] > 0 and back["suppressed_triangles"] == 0              # a thin shell is reported, never deleted
    assert df._coincident_triangles(v, f, 0.1)["suppressed_triangles"] == 0                  # a single clean sheet: nothing suppressed


def test_disconnected_components_are_reported_with_their_supporting_views():
    v, f = _grid(4)
    v2 = np.vstack([v, v + [50, 0, 0]])
    f2 = np.vstack([f, f + len(v)])
    contrib = np.zeros((len(v2), 3), np.uint8)
    contrib[: len(v), 0] = 1
    contrib[: len(v), 1] = 1
    contrib[len(v):, 2] = 1
    comps = df._components_with_views(v2, f2, contrib, ["a", "b", "c"], np.ones(contrib.shape, np.float16))
    assert len(comps) == 2 and {tuple(x["source"] for x in c["views"]) for c in comps} == {("a", "b"), ("c",)}
    assert df._validate_mesh(v2, f2)["components"] == 2


# ------------------------------------------------------------------------------------------------------------------ real Open3D (skipped where not installed)
def test_real_open3d_fusion_end_to_end_keeps_valid_inputs_complete():
    pytest.importorskip("open3d")
    res = fuse_depth_to_surface(_views())
    assert res.status == "ran" and res.complete and not res.report["views_excluded"], res.report.get("reason")
    assert res.report["mesh"]["valid"] and res.provenance_json and res.report["mesh"]["orientation_vs_viewing_rays"]["flipped_fraction"] is not None
