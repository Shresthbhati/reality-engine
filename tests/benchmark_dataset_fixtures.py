"""Tiny GENERATED fixtures in the on-disk layout each dataset's official documentation describes (see the adapter docstrings for the sources and
for what is UNVERIFIED). They are synthetic stand-ins -- passing tests on them proves the parsers follow the documented layout, nothing more."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _png(path: Path, seed: int, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _blob(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(np.random.default_rng(seed).integers(0, 255, 64, dtype=np.uint8)))


def _savetxt(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr)


def _pose(i: int) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [i * 0.3, 0.1 * i * i, 1.5]
    return m


def write_ply_quad(path: Path, scale: float = 1.0, z: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = [(0, 0, z), (4 * scale, 0, z), (4 * scale, 3 * scale, z), (0, 3 * scale, z)]
    path.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\n"
                    "property list uchar int vertex_indices\nend_header\n" + "\n".join(f"{a} {b} {c}" for a, b, c in v) + "\n3 0 1 2\n3 0 2 3\n", encoding="ascii")


def make_scannet(root: Path, n_frames=5) -> str:
    sid = "scene0000_00"
    d = root / "scans" / sid
    for i in range(n_frames):
        _blob(d / "exported" / "color" / f"{i}.jpg", i)
        _blob(d / "exported" / "depth" / f"{i}.png", 100 + i)
        _savetxt(d / "exported" / "pose" / f"{i}.txt", _pose(i) if i != 3 else np.full((4, 4), -np.inf))   # frame 3: invalid pose
    K = np.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 577.8, 578.0, 320.0, 240.0
    _savetxt(d / "exported" / "intrinsic" / "intrinsic_color.txt", K)
    (d / f"{sid}.txt").write_text("colorHeight = 480\ncolorWidth = 640\naxisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
    write_ply_quad(d / f"{sid}_vh_clean_2.ply")
    (d / f"{sid}_vh_clean.aggregation.json").write_text(json.dumps({"sceneId": sid, "segGroups": [{"id": 0, "objectId": 0, "segments": [1], "label": "couch"}]}))
    (d / f"{sid}_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"sceneId": sid, "segIndices": [1, 1, 1, 1]}))
    (d / f"{sid}.sens").write_bytes(b"sens")          # present in the real layout; never read
    # a second scan of the same physical space
    _blob(root / "scans" / "scene0000_01" / "x.txt", 9)
    return sid


def make_scannetpp(root: Path, n_frames=4, with_scans=True) -> str:
    sid = "0a1b2c3d4e"
    d = root / "data" / sid
    imgdir = d / "dslr" / "resized_undistorted_images"
    frames = []
    for i in range(n_frames):
        _blob(imgdir / f"DSC{i:05d}.JPG", i)
        t = _pose(i)
        frames.append({"file_path": f"DSC{i:05d}.JPG", "transform_matrix": t.tolist(), "is_bad": i == 2})
    (d / "dslr" / "nerfstudio").mkdir(parents=True)
    (d / "dslr" / "nerfstudio" / "transforms_undistorted.json").write_text(json.dumps(
        {"fl_x": 1500.0, "fl_y": 1501.0, "cx": 960.0, "cy": 640.0, "w": 1920, "h": 1280, "camera_model": "OPENCV", "frames": frames, "test_frames": []}))
    if with_scans:
        write_ply_quad(d / "scans" / "mesh_aligned_0.05.ply")
        write_ply_quad(d / "scans" / "pc_aligned.ply")
        write_ply_quad(d / "scans" / "mesh_aligned_0.05_semantic.ply")
        (d / "scans" / "segments_anno.json").write_text(json.dumps([{"label": "wall", "segments": [1]}]))
    (root / "split").mkdir(exist_ok=True)
    (root / "split" / "nvs_sem_val.txt").write_text(sid + "\n")
    return sid


def make_arkit(root: Path, n_frames=4) -> str:
    vid = "47333462"
    d = root / "Training" / vid
    fr = d / f"{vid}_frames"
    traj = []
    for i in range(n_frames):
        ts = 6845.7 + i * 0.1
        stem = f"{vid}_{ts:.3f}"
        _png(fr / "lowres_wide" / f"{stem}.png", i)
        _blob(fr / "lowres_depth" / f"{stem}.png", 50 + i)
        (fr / "lowres_wide_intrinsics").mkdir(parents=True, exist_ok=True)
        (fr / "lowres_wide_intrinsics" / f"{stem}.pincam").write_text("64 48 50.0 50.0 32.0 24.0")
        traj.append(f"{ts:.3f} 0 0 0 {-0.3 * i:.3f} 0.0 -1.5")           # identity rotation, world-to-camera translation
    (fr / "lowres_wide.traj").write_text("\n".join(traj) + "\n")
    write_ply_quad(d / f"{vid}_3dod_mesh.ply")
    (d / f"{vid}_3dod_annotation.json").write_text(json.dumps({"data": [
        {"label": "sofa", "uid": "u1", "axesLengths": [2.0, 1.0, 0.8], "centroid": [1, 1, 0.4], "normalizedAxes": [1, 0, 0, 0, 1, 0, 0, 0, 1]}]}))
    return vid


def make_structured3d(root: Path, n_pos=3) -> str:
    sid = "scene_00007"
    d = root / sid
    for k in range(n_pos):
        pos = d / "2D_rendering" / "485142" / "perspective" / "full" / str(k)
        _png(pos / "rgb_rawlight.png", k, (80, 40))
        _blob(pos / "depth.png", 70 + k)
        # eye in mm; view along +x, up +z; half-fov 45 deg horizontally, ~26.565 deg vertically (tan = 0.5)
        pos.joinpath("camera_pose.txt").write_text(f"{1000 * k} 2000 1500 1 0 0 0 0 1 0.785398163 0.463647609 1")
    (d / "annotation_3d.json").write_text(json.dumps({"junctions": [], "lines": [], "planes": [{"ID": 0, "type": "floor", "normal": [0, 0, 1], "offset": 0}],
                                                      "semantics": []}))
    (d / "bbox_3d.json").write_text(json.dumps([{"ID": 1, "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "coeffs": [500, 500, 400], "centroid": [1000, 1000, 400]}]))
    return sid


def make_3rscan(root: Path, n_frames=4) -> tuple[str, str]:
    ref, resc = "531cff08-0021-28f6-8e08-ba2eeb945e09", "531cff10-0021-28f6-8f94-80db8fdbbbee"
    for sid, off in ((ref, 0), (resc, 50)):
        d = root / sid
        for i in range(n_frames):
            _blob(d / "sequence" / f"frame-{i:06d}.color.jpg", off + i)
            _savetxt(d / "sequence" / f"frame-{i:06d}.pose.txt", _pose(i))
        (d / "sequence" / "_info.txt").write_text("m_colorWidth = 960\nm_colorHeight = 540\n"
                                                  "m_calibrationColorIntrinsic = 756 0 480 0 0 756 270 0 0 0 1 0 0 0 0 1\n")
        (d / "mesh.refined.v2.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        (d / "semseg.v2.json").write_text(json.dumps({"scan_id": sid, "segGroups": [{"id": 1, "label": "window", "obb": {"centroid": [0, 0, 0]}}]}))
    (root / "3RScan.json").write_text(json.dumps([{"reference": ref, "type": "train", "scans": [{"reference": resc, "transform": list(range(16))}]}]))
    return ref, resc


def make_multiscan(root: Path, n_frames=4, with_frames=True) -> str:
    sid = "scene_00000_00"
    d = root / sid
    d.mkdir(parents=True)
    lines = []
    for i in range(n_frames):
        t = _pose(i)
        lines.append(json.dumps({"timestamp": 1000 + i, "intrinsics": [500.0, 0, 0, 0, 501.0, 0, 320.0, 240.0, 1.0],
                                 "transform": t.flatten(order="F").tolist(), "quaternion": [1, 0, 0, 0], "euler_angles": [0, 0, 0]}))
        if with_frames:
            _blob(d / "frames" / f"{i:06d}.jpg", i)
    (d / f"{sid}.jsonl").write_text("\n".join(lines) + "\n")
    (d / f"{sid}.mp4").write_bytes(b"mp4")
    write_ply_quad(d / f"{sid}.ply")
    (d / f"{sid}.annotations.json").write_text(json.dumps({"version": "multiscan@0.0.1", "scanId": sid, "objects": [
        {"objectId": 1, "label": "door.1", "mobilityType": "fixed", "partIds": [1], "obb": {"centroid": [0, 0, 1], "axesLengths": [1, 0.1, 2]}}],
        "parts": [{"partId": 1, "label": "door.1", "articulations": [{"type": "rotation", "axis": [0, 0, 1]}]}]}))
    (root / "scene_00000_01").mkdir()
    return sid
