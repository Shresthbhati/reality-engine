"""Depth fusion (Open3D TSDF) against an ANALYTIC scene with exact ground truth: floor z=0, ceiling z=2.5, walls x=4 and y=3 (working frame, Z-up).
Synthetic by construction: these tests prove the algorithm and the camera/scale conventions, not anything about real photographs."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("trimesh")

from engine.math import Quat, Vec3
from engine.multiengine.depth_fusion import FusionView, _quat_to_R, fuse_depth_to_surface
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, matrix_to_quat

W, H, FX = 160, 120, 140.0
PLANES = [((0, 0, 1), 0.0), ((0, 0, 1), 2.5), ((1, 0, 0), 4.0), ((0, 1, 0), 3.0)]  # (normal, offset): n . x = c


def _camera(pos, look, scale=1.0) -> PinholeCamera:
    f = np.asarray(look, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f], axis=1)  # columns: camera x (right), y (down), z (forward) in world = camera-to-world
    q = matrix_to_quat(R)
    return PinholeCamera(CameraIntrinsics(FX, FX, W / 2, H / 2, W, H),
                         CameraExtrinsics(Vec3(*(np.asarray(pos) * scale)), Quat(q.w, q.x, q.y, q.z)))


def _depth(cam: PinholeCamera, scale=1.0) -> np.ndarray:
    """Exact depth (along camera z) of the analytic room, by ray / plane intersection."""
    e = cam.extrinsics
    q = e.rotation
    R = _quat_to_R((q.w, q.x, q.y, q.z))
    t = np.array([e.position.x, e.position.y, e.position.z])
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / FX, (v - H / 2) / FX, np.ones_like(u, float)], -1) @ R.T
    best = np.full((H, W), np.inf)
    for n, c in PLANES:
        n = np.asarray(n, float)
        denom = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (c * scale - n @ t) / denom
        s = np.where((denom != 0) & (s > 0), s, np.inf)
        best = np.minimum(best, s)
    best[~np.isfinite(best)] = 0.0
    return best.astype(np.float32)


def _views(scale=1.0):
    poses = [((1.0, 1.0, 1.5), (4, 3, 1.2)), ((2.0, 0.6, 1.4), (4, 1.5, 1.3)), ((0.6, 2.0, 1.5), (2, 3, 1.2)), ((1.6, 1.6, 1.2), (4, 3, 0.5)),
             ((2.5, 1.0, 1.3), (4, 1.0, 1.2))]
    out = []
    for i, (p, lk) in enumerate(poses):
        cam = _camera(p, lk, scale)
        out.append(FusionView(f"v{i}", np.full((H, W, 3), 150, np.uint8), _depth(cam, scale), cam))
    return out


def _dist_to_planes(vertices: np.ndarray, scale=1.0) -> np.ndarray:
    return np.min([np.abs(vertices @ np.asarray(n, float) - c * scale) for n, c in PLANES], axis=0)


def test_fused_surface_lands_on_the_true_planes_so_the_camera_convention_is_right():
    res = fuse_depth_to_surface(_views())
    assert res.status == "ran" and res.vertices is not None and len(res.faces) > 500
    voxel = res.report["voxel_length_gauge"]
    d = _dist_to_planes(res.vertices)
    assert np.median(d) <= 1.0 * voxel and np.percentile(d, 95) <= 3.0 * voxel  # a wrong extrinsic convention would scatter the surface
    assert res.report["scale_state"] == "SFM_GAUGE" and res.report["physically_metric"] is False


def test_the_fusion_is_scale_equivariant_gauge_units_are_never_treated_as_metres():
    a = fuse_depth_to_surface(_views(1.0))
    b = fuse_depth_to_surface(_views(7.0))
    assert b.report["voxel_length_gauge"] == pytest.approx(7.0 * a.report["voxel_length_gauge"], rel=1e-3)
    assert np.median(_dist_to_planes(b.vertices, 7.0)) <= 1.0 * b.report["voxel_length_gauge"]  # same quality at 7x the numbers


def test_depth_that_contradicts_the_other_views_is_rejected_and_does_not_enter_the_surface():
    views = _views()
    bad = views[0].depth.copy()
    bad[40:80, 60:100] *= 0.5  # a patch that claims a surface far in FRONT of the true wall: every other view saw through that spot
    views[0] = FusionView("v0", views[0].color, bad, views[0].camera)
    res = fuse_depth_to_surface(views)
    pv0 = next(p for p in res.report["per_view"] if p["source"] == "v0")
    assert pv0["rejected_as_inconsistent"] > 0
    assert np.median(_dist_to_planes(res.vertices)) <= res.report["voxel_length_gauge"]  # the phantom surface is not in the mesh


def test_every_contributing_view_is_measured_and_a_single_view_is_never_fused():
    res = fuse_depth_to_surface(_views())
    assert all(p["contributed_to_surface"] and p["mesh_vertices_on_this_depth"] > 0 for p in res.report["per_view"])
    assert res.report["mesh_support"]["vertices_supported_by_ge2_views"] > 0.3  # the overlap is real, and it is measured
    one = fuse_depth_to_surface(_views()[:1])
    assert one.status == "skipped" and "two views" in one.reason and one.mesh_ply is None


def test_the_mesh_is_valid_and_persistable_as_ply():
    res = fuse_depth_to_surface(_views())
    m = res.report["mesh"]
    assert m["valid"] and m["finite"] and m["degenerate_triangles"] == 0 and m["duplicate_triangles"] == 0 and m["non_manifold_edges"] == 0
    assert res.mesh_ply[:3] == b"ply"
    import io

    import trimesh

    back = trimesh.load(io.BytesIO(res.mesh_ply), file_type="ply")
    assert len(back.faces) == len(res.faces) and np.allclose(back.vertices, res.vertices, atol=1e-5)  # reload gives back what was fused
