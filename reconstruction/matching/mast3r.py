"""MASt3R matcher backend (Task 13.9).

Upstream: naver/mast3r (Grounding Image Matching in 3D with MASt3R).
Licence: CC BY-NC 4.0 (Non-commercial).

Decision D2 & Blueprint Task 13.9:
Permissive matchers (SIFT, ALIKED+LightGlue) are used by default.
MASt3R is gated behind the explicit environment variable:
    REALITY_ALLOW_NC_MODELS=1
If the flag is not set, MASt3R is refused with MatcherUnavailableError
naming the licence. If the flag is set, it probes for local weights
and dependencies.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from reconstruction.matching.interface import IMatcher, MatcherUnavailableError, MatchResult

MASt3R_LICENCE = "CC BY-NC 4.0"
MASt3R_MODEL_NAME = "mast3r@naver"


def allow_nc_models() -> bool:
    """Return True if non-commercial models are explicitly enabled."""
    return os.environ.get("REALITY_ALLOW_NC_MODELS", "").strip() == "1"


def torch_hub_checkpoints() -> Path:
    home = os.environ.get("TORCH_HOME")
    return (Path(home) if home else Path.home() / ".cache" / "torch") / "hub" / "checkpoints"


class Mast3rMatcher(IMatcher):
    """MASt3R dense feature matching backend (CC BY-NC 4.0)."""

    name: str = MASt3R_MODEL_NAME
    license: str = MASt3R_LICENCE
    non_commercial: bool = True

    def __init__(self, paths: Dict[str, Path], **kwargs):
        self.paths = dict(paths)
        self.image_paths = self.paths
        self.ransac_px = float(kwargs.get("ransac_px", 1.0))

        if not allow_nc_models():
            raise MatcherUnavailableError(
                f"{self.name} is non-commercial ({MASt3R_LICENCE}); "
                "set REALITY_ALLOW_NC_MODELS=1 to enable it"
            )

        avail, reason = self.probe()
        if not avail:
            raise MatcherUnavailableError(reason)

    @classmethod
    def probe(cls) -> Tuple[bool, str]:
        """Check whether MASt3R is licensed and available to run."""
        if not allow_nc_models():
            return (
                False,
                f"MASt3R is non-commercial ({MASt3R_LICENCE}); requires REALITY_ALLOW_NC_MODELS=1",
            )

        # Probe for torch and required model checkpoint or torchhub module
        try:
            import torch  # noqa: F401
        except ImportError as e:
            return False, f"MASt3R requires torch: {e}"

        weights_env = os.environ.get("MASt3R_WEIGHTS_PATH")
        default_weights = Path("weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        cache_weights = torch_hub_checkpoints() / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

        has_weights = (
            (weights_env and Path(weights_env).is_file())
            or default_weights.is_file()
            or cache_weights.is_file()
            or os.environ.get("REALITY_MOCK_MASt3R", "").strip() == "1"
        )

        if not has_weights:
            return (
                False,
                "MASt3R backend not implemented / weights not cached in weights/ or ~/.cache (set MASt3R_WEIGHTS_PATH)",
            )

        return True, "available"

    def match(self, a: str, b: str) -> MatchResult:
        """Match two images using MASt3R dense matching and geometric verification."""
        t0 = time.perf_counter()

        if a not in self.paths:
            raise MatcherUnavailableError(f"image {a!r} not in paths registry")
        if b not in self.paths:
            raise MatcherUnavailableError(f"image {b!r} not in paths registry")

        path_a = self.paths[a]
        path_b = self.paths[b]

        # Load images
        try:
            from engine.core.image_io import open_oriented

            img_a = np.array(open_oriented(path_a).convert("RGB"))
            img_b = np.array(open_oriented(path_b).convert("RGB"))
        except Exception as exc:
            raise MatcherUnavailableError(f"Failed to load images for MASt3R: {exc}") from exc

        # Extract correspondences
        pts_a, pts_b = self._extract_correspondences(img_a, img_b)

        # Geometric verification via epipolar RANSAC
        inliers, inlier_mask = self._verify_geometry(pts_a, pts_b)

        if inliers > 0:
            verified_a = pts_a[inlier_mask]
            verified_b = pts_b[inlier_mask]
            inlier_ratio = float(inliers / max(1, len(pts_a)))
        else:
            verified_a = np.zeros((0, 2), dtype=np.float32)
            verified_b = np.zeros((0, 2), dtype=np.float32)
            inlier_ratio = 0.0

        runtime_s = max(1e-4, time.perf_counter() - t0)

        return MatchResult(
            a=a,
            b=b,
            points_a=verified_a,
            points_b=verified_b,
            matches=len(pts_a),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            runtime_s=runtime_s,
            matcher=self.name,
        )

    def _extract_correspondences(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract dense patch correspondences between two views."""
        if os.environ.get("REALITY_MOCK_MASt3R", "").strip() == "1":
            h_a, w_a = img_a.shape[:2]
            h_b, w_b = img_b.shape[:2]
            n_pts = 30
            xs = np.linspace(w_a * 0.15, w_a * 0.85, n_pts, dtype=np.float32)
            ys = np.linspace(h_a * 0.15, h_a * 0.85, n_pts, dtype=np.float32)
            pts_a = np.column_stack([xs, ys])
            pts_b = np.column_stack([xs * (w_b / w_a) + 1.0, ys * (h_b / h_a) + 0.5])
            return pts_a, pts_b

        import torch

        weights = os.environ.get("MASt3R_WEIGHTS_PATH")
        if not weights or not Path(weights).is_file():
            weights = Path("weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")

        try:
            from mast3r.fast_reciprocal import fast_reciprocal_matching  # type: ignore
            from mast3r.model import AsymmetricMASt3R  # type: ignore

            if self._model is None:
                self._model = AsymmetricMASt3R.from_pretrained(str(weights))
                self._model.eval()

            with torch.no_grad():
                res = fast_reciprocal_matching(self._model, img_a, img_b)
                pts_a = res["pts_a"].cpu().numpy()
                pts_b = res["pts_b"].cpu().numpy()
                return pts_a, pts_b
        except Exception as exc:
            raise MatcherUnavailableError(f"MASt3R upstream model execution failed: {exc}") from exc

    def _verify_geometry(
        self, pts_a: np.ndarray, pts_b: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Verify epipolar geometry between corresponding points via RANSAC."""
        if len(pts_a) < 8:
            return 0, np.zeros(len(pts_a), dtype=bool)

        try:
            import cv2

            _, mask = cv2.findFundamentalMat(
                pts_a, pts_b, cv2.FM_RANSAC, ransacReprojThreshold=self.ransac_px, confidence=0.99
            )
            if mask is None:
                return 0, np.zeros(len(pts_a), dtype=bool)
            inlier_mask = mask.ravel().astype(bool)
            return int(np.sum(inlier_mask)), inlier_mask
        except Exception:
            return 0, np.zeros(len(pts_a), dtype=bool)
