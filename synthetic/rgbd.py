"""Synthetic RGB-D: render a scene through a pinhole camera and degrade the depth like a real sensor would.

SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING. This module generates data for the RGB-D pipeline
(16-bit PNG sidecar -> ``DepthFrame`` -> unprojection -> WorldIR); it is not a model of any particular device and
proves nothing about one.

The degradations, all seeded and reported in ``RgbdFrame.invalid_breakdown``:

    noise         range-dependent Gaussian: sigma(z) = sigma0 + sigma_z2 * z^2 (grows quadratically, as stereo/ToF do)
    holes         circular dropout blobs (specular / absorbing patches)
    out of range  nearer than ``min_range_m`` or farther than ``max_range_m``
    glass         windows return no valid depth (probabilistically)
    flying pixels pixels that jump > ``edge_jump_m`` from a 4-neighbour (depth discontinuities)
    salt          isolated random invalid pixels
    no hit        rays that leave the scene

Invalid pixels are written as raw value 0, the invalid marker ``evidence.depth_frames`` already masks (never zero
metres). Conventions: camera frame is OpenCV (x right, y down, z forward); pixel centres are ``(col+.5, row+.5)``,
the convention ``reconstruction.depth_to_points`` unprojects with; poses are camera-to-world with wxyz quaternions,
the convention ``reconstruction.calibration.camera.camera_from_pose`` reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from synthetic import SYNTHETIC_RGBD_LABEL
from synthetic.scene import Scene, raycast

PHYSICAL_PENDING = "PHYSICAL DEVICE VERIFICATION PENDING"
DEPTH_SCALE_M_PER_UNIT = 0.001   # millimetres in 16 bits: 65.5 m range at 1 mm


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_hfov(width: int, height: int, hfov_deg: float = 80.0) -> "Intrinsics":
        f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return Intrinsics(width, height, float(f), float(f), width / 2.0, height / 2.0)


@dataclass(frozen=True)
class Pose:
    """Camera-to-world. ``R`` columns are the camera's right, down and forward axes in world coordinates."""

    R: Tuple[Tuple[float, float, float], ...]
    t: Tuple[float, float, float]

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.R, dtype=float)

    def quat_wxyz(self) -> Tuple[float, float, float, float]:
        m = self.matrix
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        q = np.array([w, x, y, z])
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        return tuple(float(c) for c in q)

    def to_dict(self) -> dict:
        return {"R": [list(r) for r in self.R], "t": list(self.t), "quat_wxyz": list(self.quat_wxyz())}


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> Pose:
    eye_a, tgt, up_a = (np.asarray(a, dtype=float) for a in (eye, target, up))
    f = tgt - eye_a
    f /= np.linalg.norm(f)
    right = np.cross(f, up_a)
    if np.linalg.norm(right) < 1e-6:                         # looking straight along ``up``: pick any right axis
        right = np.cross(f, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f], axis=1)
    return Pose(tuple(tuple(float(c) for c in row) for row in R), tuple(float(c) for c in eye_a))


@dataclass(frozen=True)
class SensorModel:
    sigma0_m: float = 0.002
    sigma_z2: float = 0.0008
    hole_fraction: float = 0.03
    hole_radius_px: Tuple[int, int] = (3, 9)
    min_range_m: float = 0.3
    max_range_m: float = 8.0
    glass_invalid_prob: float = 0.85
    edge_jump_m: float = 0.15
    salt_fraction: float = 0.002
    rgb_noise_levels: float = 2.0
    seed: int = 0

    @staticmethod
    def ideal() -> "SensorModel":
        """No degradation at all: the renderer's exact depth. For tests that need geometric truth."""
        return SensorModel(0.0, 0.0, 0.0, (1, 1), 0.0, 1e9, 0.0, 1e9, 0.0, 0.0)


@dataclass
class RgbdFrame:
    index: int
    rgb: np.ndarray                  # H x W x 3 uint8
    depth_true: np.ndarray           # H x W float32 metres, NaN = ray left the scene
    depth_measured: np.ndarray       # H x W float32 metres, NaN = invalid pixel
    quad_index: np.ndarray           # H x W int32 (ground-truth surface id, -1 = none)
    intrinsics: Intrinsics
    pose: Pose
    invalid_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth_measured)


def _hash01(ix, iy, iz, seed=1) -> np.ndarray:
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791) ^ seed
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def render_frame(scene: Scene, intr: Intrinsics, pose: Pose, sensor: SensorModel = SensorModel(),
                 index: int = 0) -> RgbdFrame:
    rng = np.random.default_rng([sensor.seed, index])
    w, h = intr.width, intr.height
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    R = pose.matrix
    dirs = cam @ R.T
    t, qi = raycast(scene, pose.t, dirs)
    hit = np.isfinite(t)
    depth_true = np.where(hit, t, np.nan).astype(np.float32)

    # ---- RGB: lambert against the view ray + a world-anchored texture so images carry features
    rgb = np.zeros((h * w, 3), dtype=float)
    if hit.any():
        qn = np.array([scene.quads[i].normal for i in range(len(scene.quads))])
        col = np.array([scene.quads[i].color for i in range(len(scene.quads))], dtype=float)
        idx = qi[hit]
        pts = np.asarray(pose.t) + t[hit, None] * dirs[hit]
        view = dirs[hit] / np.linalg.norm(dirs[hit], axis=1, keepdims=True)
        lam = np.abs((qn[idx] * view).sum(1))
        cell = np.floor(pts / 0.07).astype(np.int64)
        tex = 0.82 + 0.36 * _hash01(cell[:, 0], cell[:, 1], cell[:, 2])
        rgb[hit] = col[idx] * (0.35 + 0.65 * lam)[:, None] * tex[:, None]
    if sensor.rgb_noise_levels > 0:
        rgb += rng.normal(0.0, sensor.rgb_noise_levels, rgb.shape) * hit[:, None]
    rgb_img = np.clip(rgb, 0, 255).astype(np.uint8).reshape(h, w, 3)

    # ---- depth degradation
    d = depth_true.astype(np.float64).reshape(h, w).copy()
    breakdown = {"no_hit": int((~hit).sum())}
    sigma = sensor.sigma0_m + sensor.sigma_z2 * np.nan_to_num(d) ** 2
    if sensor.sigma0_m > 0 or sensor.sigma_z2 > 0:
        d = d + rng.normal(0.0, 1.0, d.shape) * sigma
    invalid = ~np.isfinite(depth_true.reshape(h, w))
    out_of_range = np.isfinite(depth_true.reshape(h, w)) & (
        (depth_true.reshape(h, w) < sensor.min_range_m) | (depth_true.reshape(h, w) > sensor.max_range_m))
    breakdown["out_of_range"] = int(out_of_range.sum())
    invalid |= out_of_range
    labels = np.array([q.label for q in scene.quads] + ["none"])
    glass = (labels[qi.reshape(h, w)] == "glass") & (rng.random((h, w)) < sensor.glass_invalid_prob)
    breakdown["glass"] = int((glass & ~invalid).sum())
    invalid |= glass
    holes = np.zeros((h, w), dtype=bool)
    target = sensor.hole_fraction * h * w
    covered = 0
    while covered < target and sensor.hole_fraction > 0:
        r = int(rng.integers(sensor.hole_radius_px[0], sensor.hole_radius_px[1] + 1))
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        covered += int((blob & ~holes).sum())
        holes |= blob
    breakdown["holes"] = int((holes & ~invalid).sum())
    invalid |= holes
    if sensor.edge_jump_m < 1e8:
        dt = depth_true.reshape(h, w).astype(np.float64)
        jump = np.zeros((h, w), dtype=bool)
        for axis in (0, 1):
            diff = np.abs(np.diff(dt, axis=axis))
            big = np.nan_to_num(diff, nan=0.0) > sensor.edge_jump_m
            if axis == 0:
                jump[1:, :] |= big
                jump[:-1, :] |= big
            else:
                jump[:, 1:] |= big
                jump[:, :-1] |= big
        breakdown["flying"] = int((jump & ~invalid).sum())
        invalid |= jump
    salt = rng.random((h, w)) < sensor.salt_fraction
    breakdown["salt"] = int((salt & ~invalid).sum())
    invalid |= salt
    measured = np.where(invalid | (d <= 0), np.nan, d).astype(np.float32)
    return RgbdFrame(index, rgb_img, depth_true.reshape(h, w), measured, qi.reshape(h, w), intr, pose, breakdown)


# ------------------------------------------------------------------------------------------ capture writer

def write_capture(directory, frames: Sequence[RgbdFrame], *, name: str = "synthetic_rgbd",
                  declare_depth_scale: bool = True, measured_baseline: bool = True,
                  depth_scale: float = DEPTH_SCALE_M_PER_UNIT, relative_unit_m: float = 0.0037,
                  seed: int = 0) -> Path:
    """Write ``frames`` as a capture directory the real pipeline reads (``reality compile``):

        images/img_000.jpg ...        RGB
        depth/img_000.png ...         16-bit depth, raw 0 = invalid
        depth/manifest.json           {"depth_scale": ...}   -- omitted when ``declare_depth_scale`` is False
        manifest.json                 dataset, intrinsics, image size, scale reference, SYNTHETIC label
        ground_truth/poses.json       the camera-to-world poses the frames were rendered from
        ground_truth/depth_true.npz   exact depth per frame

    ``declare_depth_scale=False`` writes depth whose raw unit is an arbitrary ``relative_unit_m`` and no scale
    declaration: the parser must refuse it, which is the 'relative scale only' case. ``measured_baseline=False``
    omits the measured camera baseline, so the world's scale stays RELATIVE and device metres must not be unprojected
    into it."""
    from PIL import Image

    root = Path(directory)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    (root / "ground_truth").mkdir(exist_ok=True)
    unit = depth_scale if declare_depth_scale else relative_unit_m
    entries, poses, truth = [], {}, {}
    for f in frames:
        stem = f"img_{f.index:03d}"
        Image.fromarray(f.rgb, "RGB").save(root / "images" / f"{stem}.jpg", quality=95)
        raw = np.where(np.isfinite(f.depth_measured), np.rint(np.nan_to_num(f.depth_measured) / unit), 0)
        raw = np.clip(raw, 0, 65535).astype("<u2")
        Image.fromarray(raw, mode="I;16").save(root / "depth" / f"{stem}.png")
        entries.append({"file": f"{stem}.jpg"})
        poses[stem] = f.pose.to_dict()
        truth[stem] = f.depth_true
    if declare_depth_scale:
        (root / "depth" / "manifest.json").write_text(
            json.dumps({"depth_scale": depth_scale, "invalid_value": 0, "units": "meter"}, indent=1), encoding="utf-8")
    i0 = frames[0].intrinsics
    manifest = {
        "dataset": name,
        "images": entries,
        "intrinsics_px": {"fx": i0.fx, "fy": i0.fy, "cx": i0.cx, "cy": i0.cy},
        "image_size": [i0.width, i0.height],
        "scale_reference": {"method": "synthetic_ground_truth_baseline"},
        "synthetic": {"label": SYNTHETIC_RGBD_LABEL, "pending": PHYSICAL_PENDING,
                      "generator": "synthetic.rgbd", "seed": seed,
                      "note": "rendered from a planar scene; not a physical RGB-D device"},
    }
    if measured_baseline and len(frames) >= 2:
        # the baseline is between the first frame and the frame whose camera is FARTHEST from it (a panorama repeats
        # one eye position, which would be a zero baseline); the true distance is the 'measured' tape reading
        far = max(frames[1:], key=lambda f: float(np.linalg.norm(np.subtract(f.pose.t, frames[0].pose.t))))
        dist = float(np.linalg.norm(np.subtract(far.pose.t, frames[0].pose.t)))
        if dist > 0.0:
            manifest["measured_baselines"] = [{
                "evidence_id_a": "img_000", "evidence_id_b": f"img_{far.index:03d}", "distance_m": dist}]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (root / "ground_truth" / "poses.json").write_text(json.dumps(poses, indent=1), encoding="utf-8")
    np.savez_compressed(root / "ground_truth" / "depth_true.npz", **truth)
    return root


class GroundTruthBackend:
    """Reconstruction-backend stand-in (``REALITY_TEST_BACKEND=synthetic.rgbd:GroundTruthBackend``) for synthetic
    captures: it returns the poses the frames were RENDERED from and a sparse cloud on the true surface, read from
    the capture's ``ground_truth/``. It exists so the RGB-D leg of the pipeline can be exercised without COLMAP; it is
    never a reconstruction and is never selected outside tests."""

    SPARSE_STRIDE = 12

    def reconstruct(self, evidence):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        from provenance import Uncertainty
        from reconstruction.backend.interface import (
            ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult)

        images_dir = Path(url2pathname(urlparse(evidence[0].source_uri).path)).parent
        gt = images_dir.parent / "ground_truth"
        poses = json.loads((gt / "poses.json").read_text(encoding="utf-8"))
        manifest = json.loads((images_dir.parent / "manifest.json").read_text(encoding="utf-8"))
        k = manifest["intrinsics_px"]
        truth = np.load(gt / "depth_true.npz")
        cams, pts = [], []
        for item in evidence:
            p = poses[item.id]
            R = np.array(p["R"])
            cams.append(ReconstructedCameraPose(evidence_id=item.id, position=tuple(p["t"]),
                                                rotation=tuple(p["quat_wxyz"]), uncertainty=Uncertainty(confidence=0.9)))
            d = truth[item.id]
            for r in range(0, d.shape[0], self.SPARSE_STRIDE):
                for c in range(0, d.shape[1], self.SPARSE_STRIDE):
                    z = d[r, c]
                    if np.isfinite(z):
                        cam_pt = np.array([(c + .5 - k["cx"]) / k["fx"] * z, (r + .5 - k["cy"]) / k["fy"] * z, z])
                        pts.append(ReconstructedPoint(
                            position=tuple(float(v) for v in R @ cam_pt + np.array(p["t"])),
                            track_id=f"trk-{item.id}-{r:03d}-{c:03d}-sparse", source_evidence_ids=[item.id],
                            uncertainty=Uncertainty(confidence=0.9)))
        return ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
