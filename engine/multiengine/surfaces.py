"""Observed surfaces of the evidence world: floor / ceiling / wall / support planes and an observed-surface mesh.

Input is the fused point cloud of every image's ALIGNED depth (SfM-gauge units, job frame, +Y up). Output are ordinary WorldIR
entities + geometries, so they persist, reload, view and export through the same path as objects.

What is and is not claimed (also stored on every geometry):

* ``observed``  -- the convex hull of the inlier points projected onto the plane: only where depth actually saw the surface;
* ``unknown``   -- everything beyond the hull. It is NOT extended to meet other planes, closed into a room or filled: hidden surfaces
  are never invented. A plane's ``observed_extent`` is a lower bound of the real surface;
* the mesh uses ball pivoting, which preserves holes (Poisson would close them with inferred surface);
* scale is the SfM gauge: planes carry ``scale_state: sfm_gauge`` and no metric claim;
* classification needs the up axis; the job frame is +Y up after canonicalization. Planes that are neither horizontal nor vertical
  are reported as ``skipped`` with the reason, not forced into a class.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

UP = np.array([0.0, 1.0, 0.0])
VOXEL_FRACTION = 0.01  # of the median scene depth
PLANE_DIST_FRACTION = 0.012
MIN_INLIER_FRACTION = 0.04  # of the downsampled cloud
MAX_PLANES = 10
HORIZONTAL_COS, VERTICAL_COS = 0.9, 0.25  # |n . up|; untuned starting values


@dataclass
class PlaneFit:
    normal: np.ndarray
    offset: float  # n . p = offset
    inliers: np.ndarray  # Nx3
    kind: str  # floor | ceiling | wall | support_surface | skipped
    rms: float
    hull: np.ndarray  # Mx3 observed boundary polygon on the plane
    area: float
    reason: str = ""


def fused_cloud(aligned: Mapping[str, tuple[np.ndarray, Any]], *, stride: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points Nx3 in the job frame, source index per point, camera centres Kx3) from every image's aligned depth map."""
    from engine.multiengine.spatial_stage import _quat_to_R  # c2w rotation from the camera's quaternion

    pts, src_idx, centres = [], [], []
    for k, (_src, (depth, cam)) in enumerate(sorted(aligned.items())):
        h, w = depth.shape
        K = cam.intrinsics
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[v, u]
        ok = np.isfinite(z) & (z > 0.05)
        if not ok.any():
            continue
        pc = np.stack([(u[ok] - K.cx) * z[ok] / K.fx, (v[ok] - K.cy) * z[ok] / K.fy, z[ok]], 1)
        e = cam.extrinsics
        R = _quat_to_R((e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z))
        pw = pc @ R.T + np.array([e.position.x, e.position.y, e.position.z])
        pts.append(pw)
        src_idx.append(np.full(len(pw), k))
        centres.append([e.position.x, e.position.y, e.position.z])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros((0, 3))
    return np.vstack(pts), np.concatenate(src_idx), np.asarray(centres)


def _hull(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy.spatial import ConvexHull

    n = normal / np.linalg.norm(normal)
    a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    c = points.mean(0)
    uv = np.stack([(points - c) @ a, (points - c) @ b], 1)
    try:
        hull = ConvexHull(uv)
    except Exception:  # noqa: BLE001 -- degenerate (collinear) inliers: no observed extent to report
        return np.zeros((0, 3)), 0.0
    poly = uv[hull.vertices]
    return c + poly[:, :1] * a + poly[:, 1:] * b, float(hull.volume)  # for 2D input, ConvexHull.volume is the area


def fit_planes(points: np.ndarray, centres: np.ndarray, *, scene_depth: float, seed: int = 0) -> list[PlaneFit]:
    """Iterative RANSAC planes with Open3D on a voxel-downsampled cloud; classified against the +Y up axis."""
    import open3d as o3d

    if len(points) < 200:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(VOXEL_FRACTION * scene_depth)
    total = len(pcd.points)
    o3d.utility.random.seed(seed)
    cam_up = float(np.median(centres @ UP)) if len(centres) else 0.0  # sensor height along up: floor is below it, ceiling above
    planes: list[PlaneFit] = []
    rest = pcd
    for _ in range(MAX_PLANES):
        if len(rest.points) < max(50, MIN_INLIER_FRACTION * total):
            break
        model, idx = rest.segment_plane(PLANE_DIST_FRACTION * scene_depth, 3, 1500)
        if len(idx) < MIN_INLIER_FRACTION * total:
            break
        n = np.asarray(model[:3], float)
        norm = np.linalg.norm(n)
        n, d = n / norm, -model[3] / norm
        inl = np.asarray(rest.select_by_index(idx).points)
        rest = rest.select_by_index(idx, invert=True)
        cos = abs(float(n @ UP))
        level = float(np.median(inl @ UP))
        reason = ""
        if cos >= HORIZONTAL_COS:
            kind = "floor" if level < cam_up - 0.25 * scene_depth else "ceiling" if level > cam_up + 0.25 * scene_depth else "support_surface"
        elif cos <= VERTICAL_COS:
            kind = "wall"
        else:
            kind, reason = "skipped", f"|n.up| = {cos:.2f} is neither horizontal (>= {HORIZONTAL_COS}) nor vertical (<= {VERTICAL_COS})"
        hull, area = _hull(inl, n)
        planes.append(PlaneFit(n, float(d), inl, kind, float(np.sqrt(np.mean((inl @ n - d) ** 2))), hull, area, reason))
    return planes


def observed_mesh(points: np.ndarray, centres: np.ndarray, *, scene_depth: float) -> tuple[bytes, int, int] | None:
    """Ball-pivoting mesh of the fused cloud as PLY bytes (holes stay holes), or None when it cannot be built."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.02 * scene_depth)
    if len(pcd.points) < 100:
        return None
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08 * scene_depth, max_nn=30))
    if len(centres):
        pcd.orient_normals_towards_camera_location(np.median(centres, axis=0))
    base = float(np.median(np.asarray(pcd.compute_nearest_neighbor_distance())))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([1.5 * base, 3 * base, 6 * base]))
    if len(mesh.triangles) == 0:
        return None
    fd, path = tempfile.mkstemp(suffix=".ply")
    os.close(fd)
    try:
        o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
        with open(path, "rb") as fh:
            return fh.read(), len(mesh.vertices), len(mesh.triangles)
    finally:
        os.remove(path)


def build_surface_entities(aligned: Mapping[str, tuple[np.ndarray, Any]], matrix: Any, run_id: str, *, with_mesh: bool = True):
    """``(entities, geometries, notes, report)`` for the evidence world's observed surfaces. A geometry failure is recorded in
    ``report``, never faked and never raised."""
    from provenance import Provenance
    from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
    from world_ir.v2_extensions import ExternalReference

    report: dict[str, Any] = {"status": "ran", "planes": [], "mesh": None,
                              "claims": "observed hulls only; nothing beyond the observed extent is inferred; SfM-gauge units"}
    if not aligned:
        return [], [], [], {**report, "status": "skipped", "reason": "no aligned depth maps"}
    pts, src_idx, centres = fused_cloud(aligned)
    if len(pts) < 500:
        return [], [], [], {**report, "status": "skipped", "reason": f"only {len(pts)} depth points"}
    scene_depth = float(np.median(np.concatenate([d[np.isfinite(d) & (d > 0.05)].ravel()[::50] for d, _ in aligned.values()])))
    names = sorted(aligned)
    cloud_ref, _ = matrix.artifacts.put(json.dumps({"points": np.round(pts[::10], 4).tolist(), "frame": "job (+Y up)", "units": "sfm_units"}).encode("utf-8"))
    ents: list[Any] = []
    geoms: list[Any] = []
    counts: dict[str, int] = {}
    for p in fit_planes(pts, centres, scene_depth=scene_depth):
        if p.kind == "skipped":
            report["planes"].append({"kind": "skipped", "reason": p.reason, "inliers": len(p.inliers)})
            continue
        if len(p.hull) < 3:
            report["planes"].append({"kind": p.kind, "reason": "no observed extent (degenerate inliers)", "inliers": len(p.inliers)})
            continue
        counts[p.kind] = counts.get(p.kind, 0) + 1
        sid = f"surf-{p.kind}-{counts[p.kind]}"
        lo, hi = p.hull.min(0), p.hull.max(0)
        sample = np.random.default_rng(0).choice(len(src_idx), min(len(src_idx), 5000), replace=False)
        seen_by = sorted({names[int(i)] for i in np.unique(src_idx[sample])})
        geom = Geometry(
            id=f"geom-{sid}", type=GeometryType.PLANE, lod_level=0, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi),
            provenance=Provenance.ESTIMATED, confidence=None,
            quality_metrics={"frame_id": "world", "scale_state": "sfm_gauge", "units": "sfm_units", "plane": {"normal": p.normal.tolist(), "offset": p.offset},
                             "observed_hull": np.round(p.hull, 5).tolist(), "observed_area": round(p.area, 5), "inlier_count": int(len(p.inliers)),
                             "rms_residual": round(p.rms, 6), "unknown": "the surface beyond the observed hull was not seen and is not inferred",
                             "method": "Open3D RANSAC plane on the fused aligned-depth cloud"})
        etype = {"floor": EntityType.FLOOR, "ceiling": EntityType.CEILING, "wall": EntityType.WALL, "support_surface": EntityType.SUPPORT_SURFACE}[p.kind]
        c = p.inliers.mean(0)
        ents.append(Entity(
            id=sid, type=etype, name=p.kind.replace("_", " ").title(),
            transform={"position": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}, "frame_id": "world", "scale_state": "sfm_gauge",
                       "units": "sfm_units", "physically_metric": False},
            geometry_ids=[geom.id], provenance=Provenance.ESTIMATED, confidence=None, semantic_labels=[f"architecture/{p.kind}"],
            custom_properties={"surface": {"kind": p.kind, "observed": True, "extent": "observed hull only (lower bound)", "images_contributing": seen_by,
                                            "run_id": run_id}},
            external_references=(ExternalReference("model_output", cloud_ref, "fused_depth_point_cloud",
                                                    checksum="sha256:" + cloud_ref.removeprefix("artifact://"), provenance=Provenance.ESTIMATED),)))
        geoms.append(geom)
        report["planes"].append({"id": sid, "kind": p.kind, "inliers": int(len(p.inliers)), "observed_area": round(p.area, 4), "rms": round(p.rms, 5)})
    if with_mesh:
        try:
            built = observed_mesh(pts, centres, scene_depth=scene_depth)
        except Exception as exc:  # noqa: BLE001 -- recorded, never faked
            built, report["mesh"] = None, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if built is not None:
            data, nv, nt = built
            uri, digest = matrix.artifacts.put(data)
            g = Geometry(id="geom-observed-mesh", type=GeometryType.MESH, lod_level=0, vertex_count=nv, triangle_count=nt, data_uri=uri,
                         data_hash=digest, bounds_min=Vector3(*pts.min(0)), bounds_max=Vector3(*pts.max(0)), provenance=Provenance.ESTIMATED,
                         confidence=None, quality_metrics={"frame_id": "world", "scale_state": "sfm_gauge", "method": "Open3D ball pivoting (holes preserved)",
                                                            "unknown": "unobserved regions are holes, not filled"})
            mc = (pts.min(0) + pts.max(0)) / 2
            ents.append(Entity(id="surf-observed-mesh", type=EntityType.STRUCTURE, name="Observed surface mesh", geometry_ids=[g.id],
                               transform={"position": {"x": float(mc[0]), "y": float(mc[1]), "z": float(mc[2])}, "frame_id": "world",
                                          "scale_state": "sfm_gauge", "units": "sfm_units", "physically_metric": False},
                               provenance=Provenance.ESTIMATED, confidence=None, semantic_labels=["architecture/observed_surface_mesh"],
                               custom_properties={"surface": {"kind": "observed_mesh", "observed": True, "run_id": run_id}}))
            geoms.append(g)
            report["mesh"] = {"status": "built", "vertices": nv, "triangles": nt, "sha256": digest}
        elif report["mesh"] is None:
            report["mesh"] = {"status": "not_built", "reason": "too few points or no triangles formed"}
    return ents, geoms, [f"surfaces: {len(ents)} observed surface entities; nothing beyond the observed extent is inferred"], report
