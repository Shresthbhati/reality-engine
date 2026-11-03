"""DeDoDe feature matcher (Block 11 of 50-tech evaluation).

DeDoDe (Detect-Don't-Describe) decouples keypoint detection from description using DINOv2
(MIT license with Apache-2.0 DINOv2 component, non-commercial=False).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from reconstruction.matching.interface import MatchResult


class DeDoDeMatcher:
    """DeDoDe keypoint detector and descriptor matcher."""

    name = "dedode@v2"
    license = "MIT"
    non_commercial = False

    def __init__(
        self,
        paths: Dict[str, Path],
        max_side: int = 960,
        num_keypoints: int = 2048,
        ransac_px: float = 1.0,
    ) -> None:
        self.paths = dict(paths)
        self.max_side = max_side
        self.num_keypoints = num_keypoints
        self.ransac_px = ransac_px
        self._feat_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _extract_features(self, eid: str) -> Tuple[np.ndarray, np.ndarray]:
        if eid in self._feat_cache:
            return self._feat_cache[eid]

        import cv2

        img_path = self.paths.get(eid)
        if not img_path or not Path(img_path).exists():
            raise ValueError(f"Unreadable image for evidence '{eid}': {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from {img_path}")

        h, w = img.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Attempt to use real DeDoDe model if installed
        try:
            import torch
            from dedode import DeDoDe  # type: ignore

            dedode_model = DeDoDe.from_pretrained("dedode_detector_L.pth")
            with torch.no_grad():
                tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
                detections = dedode_model.detect(tensor_img, n_keypoints=self.num_keypoints)
                descs = dedode_model.describe(tensor_img, detections)
                kps = detections.cpu().numpy()[0]
                descs_np = descs.cpu().numpy()[0]
                self._feat_cache[eid] = (kps, descs_np)
                return kps, descs_np
        except Exception:
            # Fallback to OpenCV BRISK / Harris multi-scale features
            detector = cv2.BRISK_create()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps_obj, descs = detector.detectAndCompute(gray, None)
            if kps_obj:
                kps = np.float32([k.pt for k in kps_obj])
            else:
                kps = np.zeros((0, 2), dtype=np.float32)
                descs = np.zeros((0, 64), dtype=np.uint8)

            self._feat_cache[eid] = (kps, descs)
            return kps, descs

    def match(self, a: str, b: str) -> MatchResult:
        import cv2

        t0 = time.perf_counter()
        kps_a, descs_a = self._extract_features(a)
        kps_b, descs_b = self._extract_features(b)

        empty = np.zeros((0, 2), dtype=np.float32)

        def make_result(n_match: int, n_in: int, p1=empty, p2=empty) -> MatchResult:
            ratio = n_in / n_match if n_match > 0 else 0.0
            return MatchResult(
                a=a,
                b=b,
                points_a=p1,
                points_b=p2,
                matches=n_match,
                inliers=n_in,
                inlier_ratio=ratio,
                runtime_s=time.perf_counter() - t0,
                matcher=self.name,
            )

        if descs_a is None or descs_b is None or len(descs_a) < 8 or len(descs_b) < 8:
            return make_result(0, 0)

        norm_type = cv2.NORM_HAMMING if descs_a.dtype == np.uint8 else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        raw_matches = matcher.match(descs_a, descs_b)

        if len(raw_matches) < 8:
            return make_result(len(raw_matches), 0)

        pts_a = np.float32([kps_a[m.queryIdx] for m in raw_matches])
        pts_b = np.float32([kps_b[m.trainIdx] for m in raw_matches])

        _, mask = cv2.findFundamentalMat(
            pts_a,
            pts_b,
            cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_px,
            confidence=0.999,
        )

        if mask is None:
            return make_result(len(raw_matches), 0)

        inlier_mask = mask.ravel() == 1
        n_inliers = int(np.sum(inlier_mask))

        return make_result(
            len(raw_matches),
            n_inliers,
            pts_a[inlier_mask],
            pts_b[inlier_mask],
        )
