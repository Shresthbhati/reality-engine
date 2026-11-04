"""Learned feature matcher: DISK / ALIKED + LightGlue (Apache 2.0 via Kornia).

Uses learned feature detectors (DISK or ALIKED) and deep transformer-based
matcher LightGlue (Philipp Lindenberger & Paul-Edouard Sarlin) to establish
robust, dense correspondences across wide baselines and viewpoint variations.

Both models and checkpoints are licensed under Apache 2.0 (commercial-friendly).
Runs on CUDA with fp16 when available (via perception.device.select_device),
with a deterministic CPU fallback.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from reconstruction.matching.interface import MatchResult


class LightGlueMatcher:
    license = "Apache-2.0"
    non_commercial = False

    def __init__(
        self,
        paths: Dict[str, Path],
        max_side: int = 960,
        ransac_px: float = 1.0,
        device: Optional[str] = None,
        features: str = "disk",
    ):
        self.paths = dict(paths)
        self.max_side = max_side
        self.ransac_px = ransac_px
        self._custom_device = device
        self.features = features.lower()
        self.name = f"lightglue@{self.features}"

        self._detector = None
        self._matcher = None
        self._device = None
        # eid -> (keypoints_np, keypoints_tensor, descriptors_tensor, (width, height))
        self._feat: Dict[str, Tuple[np.ndarray, "torch.Tensor", "torch.Tensor", Tuple[int, int]]] = {}

    def _ensure_models(self):
        if self._detector is not None:
            return

        import torch
        import kornia.feature as kf
        from engine.core.device import select_device

        if self._custom_device:
            self._device = torch.device(self._custom_device)
        else:
            dev_info = select_device()
            self._device = torch.device(dev_info.device)

        if self.features == "aliked":
            self._detector = kf.ALIKED.from_pretrained("aliked-n16").to(self._device).eval()
            self._matcher = kf.LightGlue(features="aliked").to(self._device).eval()
        else:
            self._detector = kf.DISK.from_pretrained("depth").to(self._device).eval()
            self._matcher = kf.LightGlue(features="disk").to(self._device).eval()

    def _features(self, eid: str):
        if eid not in self._feat:
            import cv2
            import torch

            self._ensure_models()

            path = self.paths[eid]
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"unreadable image for evidence {eid}: {path}")

            h, w = img.shape[:2]
            scale = self.max_side / max(h, w)
            if scale < 1.0:
                target_w, target_h = int(round(w * scale)), int(round(h * scale))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                h, w = target_h, target_w

            # Convert BGR -> RGB -> normalized tensor (1, 3, H, W)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self._device) / 255.0

            if self.features == "aliked":
                with torch.no_grad():
                    feat = self._detector(t_img)[0]
                    kpts = feat.keypoints
                    descs = feat.descriptors
                    kpts_np = kpts.cpu().numpy().astype(np.float32)
                    if scale < 1.0:
                        kpts_np /= np.float32(scale)
                    self._feat[eid] = (kpts_np, kpts, descs, (w, h))
            else:
                # DISK U-Net requires input dimensions divisible by 16; pad if needed
                pad_h = (16 - (h % 16)) % 16
                pad_w = (16 - (w % 16)) % 16
                if pad_h > 0 or pad_w > 0:
                    t_img = torch.nn.functional.pad(t_img, (0, pad_w, 0, pad_h), mode="replicate")

                with torch.no_grad():
                    features = self._detector(t_img)[0]
                    kpts = features.keypoints  # (N, 2) in (x, y) coordinates
                    descs = features.descriptors  # (N, D)

                    # Filter out any keypoints detected in the padded region
                    if pad_h > 0 or pad_w > 0:
                        valid = (kpts[:, 0] < float(w)) & (kpts[:, 1] < float(h))
                        kpts = kpts[valid]
                        descs = descs[valid]

                    # Store keypoints as float32 numpy array and descriptors on device
                    kpts_np = kpts.cpu().numpy().astype(np.float32)
                    if scale < 1.0:
                        kpts_np /= np.float32(scale)

                    self._feat[eid] = (kpts_np, kpts, descs, (w, h))

        return self._feat[eid]

    def match(self, a: str, b: str) -> MatchResult:
        import cv2
        import torch

        t0 = time.perf_counter()
        empty = np.zeros((0, 2), dtype=np.float32)

        def result(n_match: int, n_in: int, p1=empty, p2=empty) -> MatchResult:
            return MatchResult(
                a=a,
                b=b,
                points_a=p1,
                points_b=p2,
                matches=n_match,
                inliers=n_in,
                inlier_ratio=n_in / n_match if n_match else 0.0,
                runtime_s=time.perf_counter() - t0,
                matcher=self.name,
            )

        (kpts_a_orig, kpts_a_dev, descs_a, (w_a, h_a)) = self._features(a)
        (kpts_b_orig, kpts_b_dev, descs_b, (w_b, h_b)) = self._features(b)

        if len(kpts_a_dev) < 8 or len(kpts_b_dev) < 8:
            return result(0, 0)

        data = {
            "image0": {
                "keypoints": kpts_a_dev.unsqueeze(0),
                "descriptors": descs_a.unsqueeze(0),
                "image_size": torch.tensor([[w_a, h_a]], device=self._device),
            },
            "image1": {
                "keypoints": kpts_b_dev.unsqueeze(0),
                "descriptors": descs_b.unsqueeze(0),
                "image_size": torch.tensor([[w_b, h_b]], device=self._device),
            },
        }

        with torch.no_grad():
            pred = self._matcher(data)
            matches_dev = pred["matches"][0]
            if len(matches_dev) == 0:
                return result(0, 0)

            matches_np = matches_dev.cpu().numpy()

        idx_a = matches_np[:, 0]
        idx_b = matches_np[:, 1]
        p1 = kpts_a_orig[idx_a]
        p2 = kpts_b_orig[idx_b]
        n_matches = len(p1)

        if n_matches < 8:
            return result(n_matches, 0)

        # Epipolar RANSAC verification
        _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, self.ransac_px, 0.999, 5000)
        if mask is None:
            return result(n_matches, 0)

        keep = mask.ravel().astype(bool)
        n_inliers = int(keep.sum())
        return result(n_matches, n_inliers, p1[keep], p2[keep])


class ALIKEDLightGlueMatcher(LightGlueMatcher):
    """ALIKED local features + LightGlue matcher behind IMatcher."""
    name = "lightglue@aliked"
    license = "Apache-2.0"
    non_commercial = False

    def __init__(self, paths: Dict[str, Path], **kwargs):
        kwargs["features"] = "aliked"
        super().__init__(paths, **kwargs)
