"""Real-data TSDF fusion harness (Sprint 1B, P1).

Runs ``engine.multiengine.depth_fusion.fuse_depth_to_surface`` on depth maps that COLMAP's PatchMatch
stereo measured from REAL photographs, using the same cameras COLMAP solved. Nothing here is analytic
or synthetic: the depth is photogrammetric, the cameras are the sparse model's, and the result is
checked against COLMAP's own ``stereo_fusion`` cloud (an independent surface from the same evidence).

Inputs are a COLMAP dense workspace produced by ``reconstruction.dense_pipeline.run_dense_mvs``:
  dense/images/<name>                     undistorted photographs
  dense/sparse/{cameras,images}.bin|txt   undistorted cameras + poses
  dense/stereo/depth_maps/<name>.geometric.bin   PatchMatch depth (COLMAP format)
  dense/fused.ply                         COLMAP's own fused cloud (independent reference)

What is verified and recorded, not assumed:
  * COLMAP depth maps store normalised depth: z = depth * ||ray|| with ray = ((u-cx)/fx, (v-cy)/fy, 1).
    This is checked against the sparse points before integration (the check is reported either way).
  * ``rotation`` is camera-to-world; Open3D integrates with world-to-camera, so the extrinsic is inverted.
  * every length is a fraction of the scene's own median depth: the surface is in SfM gauge units.
  * the mesh and its provenance sidecar are written atomically (temp + fsync + rename) with sha256.

    python scripts/sprint1/real_tsdf_fusion.py --dense <dense_dir> --out <out_dir>
    python scripts/sprint1/real_tsdf_fusion.py --verify <out_dir>      # fresh process: rehash + reload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> str:
    """temp file in the same directory + fsync + atomic rename; returns the sha256 of the bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_colmap_depth(path: Path) -> np.ndarray:
    """COLMAP depth map (``.bin``): ASCII header ``width&height&channels&`` then row-major float32."""
    raw = path.read_bytes()
    end = raw.index(b"&")
    width = int(raw[:end])
    raw = raw[end + 1:]
    end = raw.index(b"&")
    height = int(raw[:end])
    raw = raw[end + 1:]
    end = raw.index(b"&")
    channels = int(raw[:end])
    data = raw[end + 1:]
    need = width * height * channels * 4
    if len(data) < need:
        raise ValueError(f"{path.name}: truncated depth map ({len(data)} < {need} bytes)")
    arr = np.frombuffer(data[:need], dtype="<f4").reshape(height, width, channels)
    return arr[:, :, 0]


def model_views(dense: Path):
    """[(name, PinholeCamera, world-to-camera 4x4, cam center)] from the dense workspace model."""
    import pycolmap

    from engine.math import Quat, Vec3
    from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

    rec = pycolmap.Reconstruction(str(dense / "sparse"))
    views = []
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        params = [float(p) for p in cam.params]
        model = str(cam.model.name)
        if model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError(f"unsupported camera model {model} for {img.name}")
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world  # pycolmap 4.x: method
        c2w = cfw.inverse()
        R = np.asarray(c2w.rotation.matrix(), float)
        t = np.asarray(c2w.translation, float)
        c_w2c = np.eye(4)
        c_w2c[:3, :3] = np.asarray(cfw.rotation.matrix(), float)
        c_w2c[:3, 3] = np.asarray(cfw.translation, float)
        # pycolmap quaternion is (x, y, z, w); PinholeCamera/engine Quat is (w, x, y, z)
        qx, qy, qz, qw = (float(v) for v in c2w.rotation.quat)
        pinhole = PinholeCamera(
            CameraIntrinsics(fx, fy, cx, cy, int(cam.width), int(cam.height)),
            CameraExtrinsics(Vec3(float(t[0]), float(t[1]), float(t[2])), Quat(qw, qx, qy, qz)),
        )
        views.append({"name": img.name, "camera": pinhole, "w2c": c_w2c, "centre": -R.T @ t,
                      "width": int(cam.width), "height": int(cam.height), "model": model})
    return views


def normalised_depth_to_z(depth: np.ndarray, K, shape) -> np.ndarray:
    """COLMAP stores depth / ||ray||; z-depth multiplies the ray norm back. Invalid values stay <= 0."""
    h, w = shape
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ray_norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    out = depth.astype(np.float64) * ray_norm
    out[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return out


def validate_depth_against_sparse(dense: Path, views, depth_by_name: dict, z_by_name: dict) -> dict:
    """Independent check of the depth convention: project points observed by each camera and compare z to the
    depth map at that pixel. A wrong convention/units/order shows up as a large median |z - depth|."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(dense / "sparse"))
    errs, rels, samples = [], [], 0
    for img in rec.images.values():
        if img.name not in z_by_name:
            continue
        cam = rec.cameras[img.camera_id]
        params = [float(p) for p in cam.params]
        fx, fy, cx, cy = (params[0], params[1], params[2], params[3]) if len(params) >= 4 else (params[0], params[0], params[1], params[2])
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = np.asarray(cfw.rotation.matrix(), float)
        t = np.asarray(cfw.translation, float)
        z = z_by_name[img.name]
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            X = np.asarray(rec.points3D[p2d.point3D_id].xyz, float)
            xc = R @ X + t
            if xc[2] <= 0:
                continue
            u = int(round(params[0] * xc[0] / xc[2] + params[2] if len(params) >= 4 else fx * xc[0] / xc[2] + cx))
            v = int(round(params[1] * xc[1] / xc[2] + params[3] if len(params) >= 4 else fy * xc[1] / xc[2] + cy))
            if 0 <= v < z.shape[0] and 0 <= u < z.shape[1] and z[v, u] > 0:
                errs.append(abs(z[v, u] - xc[2]))
                rels.append(abs(z[v, u] - xc[2]) / max(xc[2], 1e-9))
                samples += 1
        if samples > 20000:
            break
    if not errs:
        return {"status": "NOT_MEASURED", "reason": "no sparse point projected onto a valid depth pixel"}
    arr, r = np.array(errs), np.array(rels)
    return {"status": "measured", "samples": int(len(arr)), "median_abs_error_gauge": round(float(np.median(arr)), 6),
            "p90_abs_error_gauge": round(float(np.percentile(arr, 90)), 6),
            "median_relative_error": round(float(np.median(r)), 6),
            "note": "z-depth read from COLMAP's depth map at each sparse point's projection vs that point's camera z; "
                    "a wrong depth convention, unit or flip shows up here"}


def run(dense: Path, out: Path) -> int:
    from PIL import Image

    from engine.multiengine.depth_fusion import FusionView, fuse_depth_to_surface

    t0 = time.perf_counter()
    views = model_views(dense)
    if not views:
        print(json.dumps({"status": "FAILED", "reason": "dense workspace model has no images"}))
        return 1
    depth_dir = dense / "stereo" / "depth_maps"
    if not depth_dir.is_dir():
        print(json.dumps({"status": "FAILED", "reason": f"no depth maps at {depth_dir}"}))
        return 1

    fusion_views, depth_meta, z_by_name = [], [], {}
    for v in sorted(views, key=lambda x: x["name"]):
        cands = sorted(depth_dir.glob(f"{v['name']}*.bin"))
        geo = [c for c in cands if "geometric" in c.name]
        chosen = (geo or cands)
        if not chosen:
            depth_meta.append({"view": v["name"], "status": "no depth map"})
            continue
        path = chosen[0]
        d = read_colmap_depth(path)
        if d.shape != (v["height"], v["width"]):
            depth_meta.append({"view": v["name"], "status": "size mismatch",
                               "depth_shape": list(d.shape), "camera": [v["width"], v["height"]]})
            continue
        K = v["camera"].intrinsics
        z = normalised_depth_to_z(d, (K.fx, K.fy, K.cx, K.cy), d.shape)
        img_path = dense / "images" / v["name"]
        if not img_path.is_file():
            depth_meta.append({"view": v["name"], "status": f"undistorted image missing: {img_path.name}"})
            continue
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape[:2] != d.shape:
            rgb = np.asarray(Image.fromarray(rgb).resize((d.shape[1], d.shape[0])))
        fusion_views.append(FusionView(v["name"], rgb, z, v["camera"]))
        z_by_name[v["name"]] = z
        depth_meta.append({"view": v["name"], "depth_file": path.name, "shape": list(d.shape),
                           "valid_fraction": round(float(((d > 0) & np.isfinite(d)).mean()), 4),
                           "normalised_median": round(float(np.median(d[d > 0])), 6) if (d > 0).any() else None,
                           "z_median_gauge": round(float(np.median(z[z > 0])), 6) if (z > 0).any() else None})

    report: dict = {"status": "ran", "views_with_depth": len(fusion_views), "depth_maps": depth_meta,
                    "depth_convention_check": validate_depth_against_sparse(dense, views, None, z_by_name),
                    "inputs": {"dense_workspace": str(dense)}}
    res = fuse_depth_to_surface(fusion_views)
    report.update({"fusion_status": res.status, "fusion_reason": res.reason, "fusion": res.report})
    out.mkdir(parents=True, exist_ok=True)
    if res.mesh_ply is None:
        report["status"] = "FAILED_NO_MESH"
        atomic_write(out / "fusion_report.json", json.dumps(report, indent=1, default=float).encode())
        print(json.dumps({"status": report["status"], "reason": res.reason, "report": str(out / "fusion_report.json")}))
        return 1

    mesh_path = out / "fused_tsdf.ply"
    mesh_sha = atomic_write(mesh_path, res.mesh_ply)
    sidecar = {
        "format": "ply", "frame": "dense workspace model frame (same as COLMAP fused.ply)",
        "scale_state": "SFM_GAUGE", "scale_units": "sfm_units", "physically_metric": False,
        "method": "open3d.UniformTSDFVolume integrate + extract_triangle_mesh",
        "mesh_sha256": mesh_sha, "vertices": int(len(res.vertices)), "triangles": int(len(res.faces)),
        "per_view_provenance": res.report.get("per_view"),
        "mesh_support": res.report.get("mesh_support"), "mesh": res.report.get("mesh"),
        "depth_convention_check": report["depth_convention_check"],
    }
    sidecar_sha = atomic_write(out / "fused_tsdf.provenance.json",
                               json.dumps(sidecar, indent=1, sort_keys=True, default=float).encode())
    report.update({"mesh_path": str(mesh_path), "mesh_sha256": mesh_sha,
                   "provenance_path": str(out / "fused_tsdf.provenance.json"), "provenance_sha256": sidecar_sha,
                   "seconds": round(time.perf_counter() - t0, 2)})

    # Independent reference: COLMAP's own stereo_fusion cloud from the same depth maps.
    ref = dense / "fused.ply"
    if ref.is_file():
        report["reference_comparison"] = compare_to_reference(np.asarray(res.vertices), ref)

    atomic_write(out / "fusion_report.json", json.dumps(report, indent=1, default=float).encode())
    print(json.dumps({"status": "ran", "vertices": int(len(res.vertices)), "triangles": int(len(res.faces)),
                      "mesh_sha256": mesh_sha, "seconds": report["seconds"],
                      "depth_check": report["depth_convention_check"],
                      "reference": report.get("reference_comparison")}, indent=1, default=float))
    return 0


def compare_to_reference(vertices: np.ndarray, ref_ply: Path) -> dict:
    """Nearest-neighbour distance from every TSDF vertex to COLMAP's own fused cloud, in gauge units, relative to
    the mesh's own voxel-scale. This is a floor/consistency check, not a ground-truth accuracy claim."""
    import trimesh
    from scipy.spatial import cKDTree

    cloud = trimesh.load(str(ref_ply), process=False)
    pts = np.asarray(cloud.vertices, dtype=float)
    if len(pts) == 0:
        return {"status": "NOT_MEASURED", "reason": "reference cloud empty"}
    tree = cKDTree(pts)
    d, _ = tree.query(vertices, k=1)
    diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    return {"status": "measured", "reference": ref_ply.name, "reference_points": int(len(pts)),
            "median_distance_gauge": round(float(np.median(d)), 6),
            "p90_distance_gauge": round(float(np.percentile(d, 90)), 6),
            "median_relative_to_scene_diagonal": round(float(np.median(d) / diag), 6) if diag else None,
            "note": "distance to COLMAP's independent fusion of the same depth evidence; agreement is consistency, not physical accuracy"}


def verify(out: Path) -> int:
    """Fresh process: recompute the sha256 of the persisted artifacts and reload the mesh. Fails on any mismatch."""
    mesh = out / "fused_tsdf.ply"
    prov = out / "fused_tsdf.provenance.json"
    problems = []
    if not mesh.is_file() or not prov.is_file():
        print(json.dumps({"status": "FAILED", "reason": "mesh or provenance sidecar missing"}))
        return 1
    sidecar = json.loads(prov.read_text(encoding="utf-8"))
    got_mesh = sha256(mesh)
    got_prov = sha256(prov)
    if got_mesh != sidecar.get("mesh_sha256"):
        problems.append(f"mesh sha256 {got_mesh} != recorded {sidecar.get('mesh_sha256')}")
    import trimesh

    loaded = trimesh.load(str(mesh), process=False)
    if len(loaded.vertices) != sidecar.get("vertices") or len(loaded.faces) != sidecar.get("triangles"):
        problems.append(f"reloaded counts {len(loaded.vertices)}/{len(loaded.faces)} != recorded "
                        f"{sidecar.get('vertices')}/{sidecar.get('triangles')}")
    res = {"status": "PASS" if not problems else "FAILED", "problems": problems,
           "mesh_sha256": got_mesh, "provenance_sha256": got_prov,
           "vertices": int(len(loaded.vertices)), "triangles": int(len(loaded.faces)),
           "scale_state": sidecar.get("scale_state"), "physically_metric": sidecar.get("physically_metric")}
    atomic_write(out / "fresh_process_verify.json", json.dumps(res, indent=1).encode())
    print(json.dumps(res, indent=1))
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense")
    ap.add_argument("--out")
    ap.add_argument("--verify")
    a = ap.parse_args(argv)
    if a.verify:
        return verify(Path(a.verify))
    if not a.dense or not a.out:
        ap.error("--dense and --out are required (or use --verify)")
    return run(Path(a.dense), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
