"""Heuristic depth prior (Blueprint Task 4.4, [15](15_ML_MODEL_STRATEGY.md)).

Produces per-pixel depth-like values in an ARBITRARY RELATIVE SCALE from image
intensity, vertical position, and image gradients -- NOT from a learned model
and NOT in metres.

The values are monotone with distance (larger = further) and are shaped into an
indoor-looking 0.3-20 range, but nothing ties that range to this scene: an earlier
version of this file returned them with `unit="meters"`, which let a fabricated
metric scale reach anything that trusted the unit. They are now labelled
`unit="relative"`, `quantity="depth"`, and `scale_state="relative"`, and
`reconstruction.depth_to_points.metricize_relative_depth` refuses them (it needs
inverse depth). This backend is a diagnostic prior, not a measurement source.

Honesty contract:
  - name = "heuristic_depth_prior" (never "metric3d_v2" or any upstream model name)
  - unit = "relative", quantity = "depth": no metric scale is claimed
  - confidence = 0.50 (low -- this is a scene-statistics prior, not inference)
  - Never claims to execute Metric3D, Depth Anything, ZoeDepth, or any learned model

License: Apache-2.0 (commercial-friendly).
Device: CPU only (pure numpy, no GPU required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from provenance import Uncertainty
from perception.depth.interface import DepthMap, IDepthBackend

logger = logging.getLogger(__name__)


class HeuristicDepthPriorBackend(IDepthBackend):
    """Heuristic depth prior producing per-pixel depth estimates in meters.

    Computes depth from image luminance, vertical perspective gradient, and
    image structure -- a scene-statistics prior, NOT learned model inference.

    Attributes:
        name: Backend model identifier ('heuristic_depth_prior')
        license: Apache-2.0
        unit: 'meters'
        scale_state: 'estimated'
    """

    name = "heuristic_depth_prior"
    license = "Apache-2.0"
    unit = "relative"
    quantity = "depth"
    scale_state = "relative"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        default_focal_ratio: float = 1.2,
    ):
        self._checkpoint = checkpoint
        self._custom_device = device
        self._default_focal_ratio = default_focal_ratio

    def _estimate_focal_length(self, width: int, height: int, intrinsics: Optional[Dict[str, float]] = None) -> float:
        """Returns effective focal length in pixels.

        Uses supplied camera intrinsics if available, otherwise falls back to
        canonical 35mm equivalent focal ratio default (~1.2 * max(W, H)).
        """
        if intrinsics and "fx" in intrinsics:
            return float(intrinsics["fx"])
        return float(max(width, height) * self._default_focal_ratio)

    def estimate_depth_array(
        self,
        rgb_image: np.ndarray,
        focal_length_px: Optional[float] = None,
    ) -> np.ndarray:
        """Per-pixel depth-like values in an arbitrary relative scale.

        Uses a vertical perspective gradient prior (ground plane recedes with
        y) plus image structure variation, clipped to 0.3-20 for a plausible
        indoor range. The numbers are relative and must not be read as metres.

        Args:
            rgb_image: (H, W, 3) uint8 numpy array
            focal_length_px: optional focal length in pixels

        Returns:
            (H, W) float32 numpy array, all values > 0, larger = further
        """
        h, w = rgb_image.shape[:2]
        f_px = focal_length_px or self._estimate_focal_length(w, h)

        # Convert image to grayscale luminance for geometric gradient depth prior
        gray = (
            0.2989 * rgb_image[:, :, 0]
            + 0.5870 * rgb_image[:, :, 1]
            + 0.1140 * rgb_image[:, :, 2]
        ) / 255.0

        # Vertical perspective gradient prior: ground plane recedes with y, ceiling converges
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        cy = h / 2.0
        cx = w / 2.0

        # Perspective distance from camera optical centre
        dist_from_horizon = np.abs(y_grid - cy) / max(cy, 1.0)
        base_depth = 1.8 + 2.4 * (1.0 - np.clip(dist_from_horizon, 0.0, 0.85))

        # Add image structure variation (edges and textures reflect surface geometry)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        depth_m = base_depth + 0.4 * (1.0 - np.clip(grad_mag * 3.0, 0.0, 1.0))

        # Focal ratio scale correction
        scale_correction = f_px / max(w, h, 1) / self._default_focal_ratio
        depth_m = depth_m * float(scale_correction)

        # Shaped into an indoor-looking range. This is a display/normalisation choice, not
        # a metric claim: the 0.3-20 span is arbitrary, so unit is "relative".
        return np.clip(depth_m, 0.3, 20.0).astype(np.float32)

    def estimate_depth(self, evidence: List[Any]) -> List[DepthMap]:
        """Produce heuristic DepthMaps (unit='relative', arbitrary scale) for each valid photo/video item."""
        results: List[DepthMap] = []
        import cv2

        self.skipped: List[dict] = []
        for item in evidence:
            kind_val = str(getattr(getattr(item, "kind", ""), "value", getattr(item, "kind", ""))).lower()
            if "photo" not in kind_val and "video" not in kind_val:
                continue

            path = None
            if hasattr(item, "file_path") and item.file_path:
                path = Path(item.file_path)
            elif hasattr(item, "source_uri") and item.source_uri:
                path = Path(item.source_uri)

            if not path or not path.exists():
                logger.warning("EvidenceItem '%s' has no valid on-disk image file; skipping", item.id)
                self.skipped.append({"evidence_id": str(item.id),
                                     "reason": f"no readable on-disk image: {path}"})
                continue

            bgr = cv2.imread(str(path))
            if bgr is None:
                logger.warning("Could not decode image at '%s'", path)
                self.skipped.append({"evidence_id": str(item.id),
                                     "reason": f"image could not be decoded: {path}"})
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            intrinsics = getattr(item, "intrinsics", None)
            f_px = self._estimate_focal_length(w, h, intrinsics)
            depth_arr = self.estimate_depth_array(rgb, focal_length_px=f_px)

            mean_prior = float(np.mean(depth_arr))
            spread = 0.02 * mean_prior
            valid = np.isfinite(depth_arr) & (depth_arr > 0.0)

            dm = DepthMap(
                evidence_id=item.id,
                width=w,
                height=h,
                values=np.ascontiguousarray(np.where(valid, depth_arr, np.nan), dtype=np.float32),
                unit="relative",
                quantity="depth",
                valid_mask=np.ascontiguousarray(valid),
                uncertainty=Uncertainty(
                    confidence=0.50,
                    note=("heuristic depth prior in an arbitrary relative scale (scene statistics, "
                          f"not model inference, not metres; mean {mean_prior:.3f}, spread {spread:.3f})"),
                ),
            )
            results.append(dm)

        return results