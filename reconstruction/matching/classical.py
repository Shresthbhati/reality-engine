"""Shared descriptor-matching core plus the honest classical OpenCV matchers (AKAZE, BRISK).

These are registered under their TRUE names. Learned matchers (XFeat, DeDoDe) reuse `DescriptorMatcher.match` but
supply their own `_extract`; they never fall back to these detectors under a learned name.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from reconstruction.matching.interface import MatcherUnavailableError, MatchResult


def torch_hub_dir() -> Path:
    home = os.environ.get("TORCH_HOME")
    return (Path(home) if home else Path.home() / ".cache" / "torch") / "hub"


def torch_hub_checkpoints() -> Path:
    return torch_hub_dir() / "checkpoints"


class DescriptorMatcher:
    """Extract (keypoints, descriptors) per image, mutual-NN match, epipolar-RANSAC verify."""

    name = ""
    license = ""
    non_commercial = False

    def __init__(self, paths: Dict[str, Path], max_side: int = 960, ransac_px: float = 1.0) -> None:
        self.paths = dict(paths)
        self.max_side = max_side
        self.ransac_px = ransac_px
        self._feat_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, eid: str) -> np.ndarray:
        import cv2

        p = self.paths.get(eid)
        if not p or not Path(p).exists():
            raise ValueError(f"Unreadable image for evidence '{eid}': {p}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from {p}")
        scale = self.max_side / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    def _extract(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _features(self, eid: str) -> Tuple[np.ndarray, np.ndarray]:
        if eid not in self._feat_cache:
            self._feat_cache[eid] = self._extract(self._load(eid))
        return self._feat_cache[eid]

    def match(self, a: str, b: str) -> MatchResult:
        import cv2

        t0 = time.perf_counter()
        kps_a, descs_a = self._features(a)
        kps_b, descs_b = self._features(b)
        empty = np.zeros((0, 2), dtype=np.float32)

        def result(n_match: int, n_in: int, p1=empty, p2=empty) -> MatchResult:
            return MatchResult(a=a, b=b, points_a=p1, points_b=p2, matches=n_match, inliers=n_in,
                               inlier_ratio=n_in / n_match if n_match > 0 else 0.0,
                               runtime_s=time.perf_counter() - t0, matcher=self.name)

        if descs_a is None or descs_b is None or len(descs_a) < 8 or len(descs_b) < 8:
            return result(0, 0)
        norm = cv2.NORM_HAMMING if descs_a.dtype == np.uint8 else cv2.NORM_L2
        raw = cv2.BFMatcher(norm, crossCheck=True).match(descs_a, descs_b)
        if len(raw) < 8:
            return result(len(raw), 0)
        pa = np.float32([kps_a[m.queryIdx] for m in raw])
        pb = np.float32([kps_b[m.trainIdx] for m in raw])
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, ransacReprojThreshold=self.ransac_px,
                                         confidence=0.999)
        if mask is None:
            return result(len(raw), 0)
        keep = mask.ravel() == 1
        return result(len(raw), int(keep.sum()), pa[keep], pb[keep])


class _CvDetectorMatcher(DescriptorMatcher):
    """OpenCV binary-descriptor matcher. The detector is absent from some OpenCV builds (e.g. 5.x), so it is probed."""

    _factory = ""
    _desc_bytes = 64

    @classmethod
    def probe(cls) -> Tuple[bool, str]:
        try:
            import cv2
        except ImportError as e:
            return False, f"OpenCV not installed: {e}"
        if not hasattr(cv2, cls._factory):
            return False, f"this OpenCV build ({cv2.__version__}) has no cv2.{cls._factory}"
        return True, "available"

    def __init__(self, paths: Dict[str, Path], **kwargs) -> None:
        ok, msg = self.probe()
        if not ok:
            raise MatcherUnavailableError(msg)
        super().__init__(paths, **kwargs)

    def _extract(self, img_bgr):
        import cv2

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = getattr(cv2, self._factory)().detectAndCompute(gray, None)
        if not kps:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, self._desc_bytes), dtype=np.uint8)
        return np.float32([k.pt for k in kps]), descs


class AkazeMatcher(_CvDetectorMatcher):
    name = "akaze@opencv"
    license = "BSD-3-Clause"
    _factory = "AKAZE_create"
    _desc_bytes = 61


class BriskMatcher(_CvDetectorMatcher):
    name = "brisk@opencv"
    license = "BSD-3-Clause"
    _factory = "BRISK_create"
    _desc_bytes = 64


class OrbMatcher(_CvDetectorMatcher):
    name = "orb@opencv"
    license = "BSD-3-Clause"
    _factory = "ORB_create"
    _desc_bytes = 32


__all__ = ["AkazeMatcher", "BriskMatcher", "OrbMatcher", "DescriptorMatcher", "MatcherUnavailableError",
           "torch_hub_checkpoints", "torch_hub_dir"]
