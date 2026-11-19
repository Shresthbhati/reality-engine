"""LightGlue and ALIKED / DISK Neural Feature Matcher Adapter.

Uses LightGlue with ALIKED (or DISK) via kornia/lightglue for deep learned
feature extraction and transformer-based correspondence matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from PIL import Image

from engine.adapters.base import (
    BaseEngineAdapter,
    DependencyMissingError,
    InputValidationError,
)


class LightGlueMatcherAdapter(BaseEngineAdapter):
    """Adapter for ALIKED + LightGlue feature extraction and matching."""

    def __init__(
        self,
        feature_type: str = "aliked",
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name=f"lightglue_{feature_type}",
            task_category="feature_matching",
            licence="Apache-2.0",
            device=device,
        )
        self.feature_type = feature_type.lower()
        self._extractor = None
        self._matcher = None

    def verify_dependencies(self) -> tuple[bool, str]:
        import importlib.util
        missing = [pkg for pkg in ("torch", "lightglue", "kornia") if importlib.util.find_spec(pkg) is None]
        if missing:
            return False, f"Missing dependencies: {', '.join(missing)}"
        return True, "torch, kornia, and lightglue available"

    def load_model(self) -> None:
        if self._matcher is not None:
            return

        if torch is None:
            raise DependencyMissingError("PyTorch is required for LightGlueMatcherAdapter")
        from lightglue import ALIKED, DISK, LightGlue

        dev = self._custom_device or ("cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu")
        try:
            if self.feature_type == "aliked":
                self._extractor = ALIKED(max_num_keypoints=2048).to(dev).eval()
                self._matcher = LightGlue(features="aliked").to(dev).eval()
            elif self.feature_type == "disk":
                self._extractor = DISK(max_num_keypoints=2048).to(dev).eval()
                self._matcher = LightGlue(features="disk").to(dev).eval()
            else:
                raise ValueError(f"Unsupported LightGlue feature type: {self.feature_type}")
            self._model = self._matcher
        except Exception as e:
            raise DependencyMissingError(f"Failed to load LightGlue ({self.feature_type}): {e}") from e

    def validate_inputs(self, inputs: Any) -> None:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise InputValidationError("Inputs must be a pair of images (image_0, image_1).")

    def _execute_inference(
        self,
        inputs: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        from lightglue.utils import load_image

        dev = self._custom_device or ("cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu")

        tensors = []
        for inp in inputs:
            if isinstance(inp, (str, Path)):
                t = load_image(str(inp)).to(dev)
            elif isinstance(inp, Image.Image):
                arr = np.array(inp.convert("RGB")) / 255.0
                t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).to(dev)
            elif isinstance(inp, np.ndarray):
                arr = inp / 255.0 if inp.max() > 1.0 else inp
                t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).to(dev)
            else:
                raise InputValidationError(f"Unsupported image type: {type(inp)}")
            tensors.append(t)

        with torch.no_grad():
            feats0 = self._extractor.extract(tensors[0])
            feats1 = self._extractor.extract(tensors[1])

            matches01 = self._matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [
                rb.cpu().numpy() for rb in (feats0, feats1, matches01)
            ]

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches01["matches"]
        scores = matches01["scores"]

        num_matches = len(matches)
        mean_score = float(np.mean(scores)) if num_matches > 0 else 0.0

        # Run geometric verification via RANSAC Fundamental Matrix
        inlier_ratio = 0.0
        num_inliers = 0
        if num_matches >= 8:
            matched_pts0 = kpts0[matches[:, 0]]
            matched_pts1 = kpts1[matches[:, 1]]
            _, mask = cv2.findFundamentalMat(matched_pts0, matched_pts1, cv2.FM_RANSAC, 2.0, 0.99)
            if mask is not None:
                num_inliers = int(np.sum(mask))
                inlier_ratio = float(num_inliers / num_matches)

        structured_results = {
            "num_keypoints_image0": len(kpts0),
            "num_keypoints_image1": len(kpts1),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "mean_matching_score": mean_score,
            "matches": matches.tolist() if num_matches > 0 else [],
            "scores": scores.tolist() if num_matches > 0 else [],
        }

        diagnostics = {
            "feature_type": self.feature_type,
            "ransac_verified": num_inliers >= 8,
            "keypoints_density_ratio": len(kpts0) / max(1, len(kpts1)),
        }

        return structured_results, matches01, diagnostics
