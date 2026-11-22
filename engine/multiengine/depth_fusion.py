"""Depth-to-geometry fusion: aligned per-image depth + calibrated cameras -> one fused surface (Open3D uniform TSDF + marching cubes).

(``ScalableTSDFVolume`` was tried first and extracts zero triangles in the installed Open3D 0.20 Windows build for any parameters, even a
single flat wall; ``UniformTSDFVolume`` works, with bounds taken from the data.)

What makes this production-honest rather than a demo:

* Scale discipline. The input depth is already aligned to the SfM gauge, so every length here (voxel size, truncation, tolerances) is a
  FRACTION of the scene's own median depth: the surface is in SfM gauge units, never metres. ``scale_state`` is returned as
  ``SFM_GAUGE`` and nothing converts it.
* Camera conventions are the repository's: OpenCV camera frame (x right, y down, z forward, depth = z), ``rotation`` is camera-to-world.
  Open3D wants world-to-camera, so the extrinsic is the inverse. The reprojection test in ``tests/test_depth_fusion.py`` pins this.
* Inconsistent depth is REJECTED before integration. Every depth pixel is reprojected into every other view: agreement within a
  relative tolerance is *support*, a surface that lies behind the other view's observed surface is *occluded* (neutral), and a surface
  that lies in front of what the other view saw (a free-space violation) is a *conflict*. A pixel with more conflicts than support is not
  integrated. Pixels seen by one view only are kept and counted as ``unsupported`` -- they are not evidence of consistency.
* Contribution is MEASURED, not assumed. Per view: pixels integrated / rejected, and how many of the final mesh vertices lie on that
  view's depth. A depth map that never touched the surface is reported as such.
* The mesh is validated (degenerate / duplicate triangles, non-finite vertices, manifoldness, components) and persisted by the caller.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

DEFAULT_VOXEL_FRACTION = 0.01     # voxel edge = 1% of the median scene depth (gauge units)
DEFAULT_TRUNC_VOXELS = 4.0        # SDF truncation = 4 voxels
DEFAULT_DEPTH_TRUNC_FACTOR = 4.0  # ignore depth beyond 4x the median depth
DEFAULT_CONSISTENCY_REL_TOL = 0.03  # two views agree on a surface when their depths differ by < 3% of that depth
MIN_CLUSTER_TRIANGLES = 50        # connected pieces smaller than this are reported and removed (floating specks)
MIN_RESOLUTION, MAX_RESOLUTION = 64, 256  # voxels per axis of the (cubic) volume: 256^3 keeps memory ~0.5 GB; a cap makes the voxel coarser, and is reported


@dataclass
class FusionView:
    source_id: str
    color: np.ndarray   # H x W x 3 uint8 at the DEPTH resolution
    depth: np.ndarray   # H x W float, SfM-gauge depth along the camera z axis (<=0 or non-finite = no measurement)
    camera: Any         # world_ir.camera.PinholeCamera (camera-to-world extrinsics), intrinsics sized to the ORIGINAL photograph


@dataclass
class FusionResult:
    status: str                       # "ran" | "skipped"
    reason: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    mesh_ply: bytes | None = None     # binary PLY, vertices in the SAME frame as the input cameras
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


def _intrinsics_at(cam: Any, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy of ``cam`` rescaled to a depth map of ``shape`` (h, w)."""
    K = cam.intrinsics
    sx, sy = shape[1] / K.width, shape[0] / K.height
    return K.fx * sx, K.fy * sy, K.cx * sx, K.cy * sy


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(cam: Any) -> tuple[np.ndarray, np.ndarray]:
    """(R camera-to-world 3x3, camera centre in world)."""
    e = cam.extrinsics
    return _quat_to_R((e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z)), np.array([e.position.x, e.position.y, e.position.z], float)


def _backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, R: np.ndarray, t: np.ndarray):
    """All valid pixels -> world points. Returns (points N x 3, flat pixel indices N)."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.ravel())
    v, u = np.divmod(idx, w)
    z = depth.ravel()[idx].astype(float)
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    return pc @ R.T + t, idx


def _project(points: np.ndarray, fx: float, fy: float, cx: float, cy: float, R: np.ndarray, t: np.ndarray, shape: tuple[int, int]):
    """World points -> (u, v ints, z, in_image mask) for a camera with camera-to-world (R, t)."""
    pc = (points - t) @ R           # R^T (X - t) in row-vector form
    z = pc[:, 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    u = np.round(pc[:, 0] / zs * fx + cx).astype(np.int64)
    v = np.round(pc[:, 1] / zs * fy + cy).astype(np.int64)
    h, w = shape
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, inside


def cross_view_consistency(views: Sequence[FusionView], tol: float) -> list[dict[str, Any]]:
    """Per view: which valid depth pixels are supported / occluded / in conflict with the OTHER views. ``keep`` is the integration mask."""
    cams = []
    for v in views:
        fx, fy, cx, cy = _intrinsics_at(v.camera, v.depth.shape)
        R, t = _pose(v.camera)
        cams.append((fx, fy, cx, cy, R, t))
    out = []
    for i, vi in enumerate(views):
        fx, fy, cx, cy, R, t = cams[i]
        pts, idx = _backproject(vi.depth, fx, fy, cx, cy, R, t)
        support = np.zeros(len(idx), np.int32)
        conflict = np.zeros(len(idx), np.int32)
        for j, vj in enumerate(views):
            if i == j:
                continue
            fxj, fyj, cxj, cyj, Rj, tj = cams[j]
            u, v, z, inside = _project(pts, fxj, fyj, cxj, cyj, Rj, tj, vj.depth.shape)
            dj = np.zeros(len(idx))
            dj[inside] = vj.depth[v[inside], u[inside]]
            seen = inside & np.isfinite(dj) & (dj > 0)
            rel = np.where(seen, (z - dj) / np.where(seen, dj, 1.0), 0.0)
            support += (seen & (np.abs(rel) <= tol)).astype(np.int32)
            conflict += (seen & (rel < -tol)).astype(np.int32)      # in front of what view j saw along that ray: free-space violation
            # rel > tol: behind j's observed surface = occluded in j = neutral
        keep = conflict <= support
        out.append({"idx": idx, "keep": keep, "support": support, "conflict": conflict,
                    "valid": int(len(idx)), "supported": int((support > 0).sum()), "conflicting": int((conflict > support).sum()),
                    "unsupported": int(((support == 0) & (conflict == 0)).sum())})
    return out


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Measured validity of the fused mesh (nothing here is a threshold; the numbers are reported and gated structurally by the caller)."""
    rep: dict[str, Any] = {"vertices": int(len(vertices)), "triangles": int(len(faces))}
    if len(faces) == 0:
        return {**rep, "valid": False, "problems": ["no triangles were extracted"]}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    problems: list[str] = []
    finite = bool(np.isfinite(vertices).all())
    if not finite:
        problems.append("non-finite vertex coordinates")
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = int((area <= 1e-12).sum())
    if degenerate:
        problems.append(f"{degenerate} degenerate triangle(s)")
    dup = int(len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0)))
    if dup:
        problems.append(f"{dup} duplicate triangle(s)")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold, boundary = int((counts > 2).sum()), int((counts == 1).sum())
    if nonmanifold:
        problems.append(f"{nonmanifold} non-manifold edge(s)")
    n = len(vertices)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    comp_sizes = np.sort(np.unique(face_label, return_counts=True)[1])[::-1]
    return {**rep, "valid": not problems, "problems": problems, "finite": finite, "degenerate_triangles": degenerate, "duplicate_triangles": dup,
            "non_manifold_edges": nonmanifold, "boundary_edges": boundary, "components": int(len(comp_sizes)),
            "largest_component_triangles": int(comp_sizes[0]), "surface_area_gauge2": round(float(area.sum()), 6),
            "bounds_gauge": [np.min(vertices, axis=0).round(4).tolist(), np.max(vertices, axis=0).round(4).tolist()]}


def fuse_depth_to_surface(views: Sequence[FusionView], *, voxel_fraction: float = DEFAULT_VOXEL_FRACTION,
                          trunc_voxels: float = DEFAULT_TRUNC_VOXELS, depth_trunc_factor: float = DEFAULT_DEPTH_TRUNC_FACTOR,
                          consistency_rel_tol: float = DEFAULT_CONSISTENCY_REL_TOL, min_cluster_triangles: int = MIN_CLUSTER_TRIANGLES) -> FusionResult:
    """Fuse ``views`` into one surface. Nothing is returned for fewer than two views: a single depth map cannot be checked for consistency
    and a one-view "fusion" would only re-wrap that depth map."""
    base: dict[str, Any] = {"method": "open3d.UniformTSDFVolume (integrate + marching cubes)", "scale_state": "SFM_GAUGE", "scale_units": "sfm_units",
                            "physically_metric": False, "views_offered": len(views),
                            "parameters": {"voxel_fraction_of_median_depth": voxel_fraction, "sdf_trunc_voxels": trunc_voxels,
                                           "depth_trunc_factor": depth_trunc_factor, "consistency_rel_tol": consistency_rel_tol}}
    usable = [v for v in views if np.isfinite(v.depth).any() and (v.depth[np.isfinite(v.depth)] > 0).any()]
    base["views_with_valid_depth"] = len(usable)
    if len(usable) < 2:
        return FusionResult("skipped", "fewer than two views with valid aligned depth: no cross-view consistency can be established", base)
    try:
        import open3d as o3d
        import trimesh
    except Exception as exc:  # noqa: BLE001
        return FusionResult("skipped", f"fusion library unavailable ({type(exc).__name__}: {exc})", base)

    allz = np.concatenate([v.depth[np.isfinite(v.depth) & (v.depth > 0)] for v in usable])
    z_med = float(np.median(allz))
    voxel = voxel_fraction * z_med
    trunc = trunc_voxels * voxel
    base.update({"median_depth_gauge": round(z_med, 6), "voxel_length_gauge": round(voxel, 6), "sdf_trunc_gauge": round(trunc, 6)})

    cons = cross_view_consistency(usable, consistency_rel_tol)
    # Volume bounds from the KEPT depth itself (1st-99th percentile per axis, so a few stray pixels cannot inflate the volume), in gauge units.
    kept_pts = []
    for v, c in zip(usable, cons, strict=True):
        fx, fy, cx, cy = _intrinsics_at(v.camera, v.depth.shape)
        R, t = _pose(v.camera)
        pts, idx = _backproject(v.depth, fx, fy, cx, cy, R, t)
        kept_pts.append(pts[c["keep"]][::7])
    allp = np.concatenate(kept_pts)
    if len(allp) < 100:
        return FusionResult("ran", "too few consistent depth pixels to define a volume", base)
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    side = float(np.max(hi - lo)) + 2.0 * (trunc_voxels * voxel)
    resolution = int(np.clip(np.ceil(side / voxel), MIN_RESOLUTION, MAX_RESOLUTION))
    voxel = side / resolution                      # the EFFECTIVE voxel (gauge units); capped resolution makes it coarser than requested
    trunc = trunc_voxels * voxel
    origin = (lo + hi) / 2.0 - side / 2.0
    base.update({"voxel_length_gauge": round(voxel, 6), "sdf_trunc_gauge": round(trunc, 6), "volume_resolution": resolution,
                 "volume_side_gauge": round(side, 6), "resolution_capped": bool(resolution == MAX_RESOLUTION)})
    volume = o3d.pipelines.integration.UniformTSDFVolume(length=side, resolution=resolution, sdf_trunc=trunc,
                                                         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=origin)
    per_view: list[dict[str, Any]] = []
    for v, c in zip(usable, cons, strict=True):
        h, w = v.depth.shape
        depth = np.zeros(h * w, np.float32)
        kept = c["idx"][c["keep"]]
        depth[kept] = v.depth.ravel()[kept]
        depth = depth.reshape(h, w)
        fx, fy, cx, cy = _intrinsics_at(v.camera, (h, w))
        R, t = _pose(v.camera)
        T_cw = np.eye(4)
        T_cw[:3, :3], T_cw[:3, 3] = R, t
        extrinsic = np.linalg.inv(T_cw)  # Open3D integrates with WORLD-TO-CAMERA
        color = o3d.geometry.Image(np.ascontiguousarray(v.color.astype(np.uint8)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, o3d.geometry.Image(depth), depth_scale=1.0,
                                                                  depth_trunc=depth_trunc_factor * z_med, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy), extrinsic)
        per_view.append({"source": v.source_id, "depth_pixels": c["valid"], "integrated_pixels": int(c["keep"].sum()),
                         "rejected_as_inconsistent": int((~c["keep"]).sum()), "supported_by_another_view": c["supported"],
                         "unsupported_single_view": c["unsupported"]})

    mesh = volume.extract_triangle_mesh()
    base["raw_triangles_before_cleanup"] = int(len(mesh.triangles))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    removed_specks = 0
    if len(mesh.triangles):
        tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
        tri_clusters, cluster_n = np.asarray(tri_clusters), np.asarray(cluster_n)
        small = np.flatnonzero(cluster_n < min_cluster_triangles)
        if len(small):
            removed_specks = int(cluster_n[small].sum())
            mesh.remove_triangles_by_mask(np.isin(tri_clusters, small))
            mesh.remove_unreferenced_vertices()
    removed_zero_area = 0
    if len(mesh.triangles):  # marching cubes can leave numerically zero-area slivers that Open3D's own cleanup keeps: remove them explicitly
        tv, tf = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        tri = tv[tf]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        zero = area <= 1e-12
        removed_zero_area = int(zero.sum())
        if removed_zero_area:
            mesh.remove_triangles_by_mask(zero)
            mesh.remove_unreferenced_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    validity = _validate_mesh(vertices, faces)
    validity["removed_floating_specks_triangles"] = removed_specks
    validity["removed_zero_area_triangles"] = removed_zero_area
    report = {**base, "per_view": per_view, "mesh": validity}
    if len(faces) == 0:
        return FusionResult("ran", "fusion executed but the fused volume contains no surface", report)

    # contribution: how many mesh vertices lie ON each view's depth (a depth map that never touched the surface shows up as 0)
    tol_abs = 2.0 * voxel
    support_count = np.zeros(len(vertices), np.int32)
    for v, pv in zip(usable, per_view, strict=True):
        fx, fy, cx, cy = _intrinsics_at(v.camera, v.depth.shape)
        R, t = _pose(v.camera)
        u, vv, z, inside = _project(vertices, fx, fy, cx, cy, R, t, v.depth.shape)
        d = np.zeros(len(vertices))
        d[inside] = v.depth[vv[inside], u[inside]]
        on = inside & np.isfinite(d) & (d > 0) & (np.abs(z - d) <= tol_abs + consistency_rel_tol * d)
        support_count += on.astype(np.int32)
        pv["mesh_vertices_on_this_depth"] = int(on.sum())
        pv["contributed_to_surface"] = bool(on.any() and pv["integrated_pixels"] > 0)
    report["mesh_support"] = {"vertices_supported_by_ge1_view": round(float((support_count >= 1).mean()), 4),
                              "vertices_supported_by_ge2_views": round(float((support_count >= 2).mean()), 4),
                              "views_that_did_not_contribute": [p["source"] for p in per_view if not p["contributed_to_surface"]]}

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.has_vertex_colors():
        tm.visual.vertex_colors = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    tm.export(buf, file_type="ply")
    return FusionResult("ran", None, report, buf.getvalue(), vertices, faces)
