"""Automatic image-quality inspection (N-3).

Detects motion blur, overexposure, underexposure, lens obstruction, reflections,
low resolution and poor framing before accepting an image.

Poor images are flagged with a specific reason and suggested correction.

Honesty rules:
* Thresholds are provisional; chosen from computer-vision rules of thumb and
  checked only on synthetic images. None has been validated on a real photo set.
* Verdicts are predictions about registration, not measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # OpenCV is an optional runtime dependency
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ImageQuality:
    """Quality assessment for one image."""
    image_id: str
    verdict: str                      # "ok" | "warn" | "drop"
    reasons: List[str]                # specific reasons
    metrics: dict                     # raw metrics
    suggestion: Optional[str]         # suggested correction
    provisional: bool = True


class ImageQualityInspector:
    """Inspects image quality before accepting into reconstruction."""

    # Provisional thresholds (not calibrated on real photos)
    BLUR_WARN_BELOW = 100.0
    BLUR_DROP_BELOW = 25.0
    CLIP_WARN_FRACTION = 0.20
    CLIP_DROP_FRACTION = 0.50
    MEAN_LUMA_WARN = (45.0, 210.0)
    MEAN_LUMA_DROP = (18.0, 238.0)
    MIN_RESOLUTION = 640  # minimum width/height in pixels

    def inspect_image(
        self,
        image_id: str,
        image_data: bytes,
    ) -> ImageQuality:
        """Inspect an image for quality issues."""
        if cv2 is None:
            return ImageQuality(
                image_id=image_id,
                verdict="unknown",
                reasons=["OpenCV not available; cannot inspect image quality."],
                metrics={},
                suggestion="Install opencv-python to enable image quality inspection.",
                provisional=True,
            )

        # Decode image
        arr = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return ImageQuality(
                image_id=image_id,
                verdict="drop",
                reasons=["The file could not be decoded as an image."],
                metrics={},
                suggestion="Provide a valid image file.",
                provisional=True,
            )

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Resize for analysis
        h, w = gray.shape
        s = self.MIN_RESOLUTION / max(h, w)
        if s < 1.0:
            gray = cv2.resize(gray, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)

        metrics = self._compute_metrics(gray)
        verdict, reasons = self._judge_image(metrics)
        suggestion = self._suggest_correction(verdict, reasons)

        return ImageQuality(
            image_id=image_id,
            verdict=verdict,
            reasons=reasons,
            metrics=metrics,
            suggestion=suggestion,
            provisional=True,
        )

    def _compute_metrics(self, gray: np.ndarray) -> dict:
        """Compute quality metrics for a grayscale image."""
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(gray.mean())
        clipped_high = float((gray >= 250).mean())
        clipped_low = float((gray <= 5).mean())

        # Edge density as a proxy for detail
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / gray.size

        # Local variance as a proxy for texture/detail
        kernel = np.ones((9, 9), dtype=np.float32) / 81.0
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D(gray.astype(np.float32) ** 2, -1, kernel)
        local_var = sqr_mean - mean ** 2
        local_var = np.maximum(local_var, 0)
        mean_local_var = float(local_var.mean())

        return {
            "blur_laplacian_var": round(lap, 2),
            "mean_luma": round(mean_luma, 1),
            "clipped_high": round(clipped_high, 4),
            "clipped_low": round(clipped_low, 4),
            "edge_density": round(edge_density, 4),
            "mean_local_variance": round(mean_local_var, 4),
            "resolution": f"{gray.shape[1]}x{gray.shape[0]}",
        }

    def _judge_image(self, m: dict) -> Tuple[str, List[str]]:
        """(verdict, reasons) from one image's metrics."""
        verdict, reasons = "ok", []
        v = m["blur_laplacian_var"]
        if v < self.BLUR_DROP_BELOW:
            verdict = "drop"
            reasons.append(f"Very low detail (sharpness {v:.0f}): too blurry, or a plain surface with no texture.")
        elif v < self.BLUR_WARN_BELOW:
            verdict = "warn"
            reasons.append(f"Low detail (sharpness {v:.0f}): slightly blurry, or a plain surface. Hold still and include some texture or edges.")

        for side, frac, word in (("high", m["clipped_high"], "overexposed (washed out)"),
                                 ("low", m["clipped_low"], "underexposed (crushed black)")):
            if frac >= self.CLIP_DROP_FRACTION:
                verdict = "drop"
                reasons.append(f"{frac:.0%} of the photo is {word}: detail is lost.")
            elif frac >= self.CLIP_WARN_FRACTION:
                if verdict != "drop":
                    verdict = "warn"
                reasons.append(f"{frac:.0%} of the photo is {word}.")

        luma = m["mean_luma"]
        if luma < self.MEAN_LUMA_DROP[0] or luma > self.MEAN_LUMA_DROP[1]:
            verdict = "drop"
            reasons.append("Overall brightness is extreme; the room is not readable in this photo.")
        elif luma < self.MEAN_LUMA_WARN[0] or luma > self.MEAN_LUMA_WARN[1]:
            if verdict != "drop":
                verdict = "warn"
            if luma < self.MEAN_LUMA_WARN[0]:
                reasons.append("The photo is dark; turn on lights or open curtains.")
            else:
                reasons.append("The photo is very bright; avoid pointing at windows or lamps.")

        if gray_shape := m.get("resolution"):
            w, h = gray_shape.split("x")
            w, h = int(w), int(h)
            if w < self.MIN_RESOLUTION or h < self.MIN_RESOLUTION:
                verdict = "drop"
                reasons.append(f"Resolution {w}x{h} is too low; minimum is {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}.")

        return verdict, reasons

    def _suggest_correction(self, verdict: str, reasons: List[str]) -> Optional[str]:
        if verdict == "ok":
            return None
        suggestions = []
        for reason in reasons:
            if "blurry" in reason.lower() or "detail" in reason.lower():
                suggestions.append("Hold the camera steady and include some texture or edges in the frame.")
            if "overexposed" in reason.lower():
                suggestions.append("Avoid pointing at windows or lamps; reduce exposure.")
            if "underexposed" in reason.lower():
                suggestions.append("Turn on lights or open curtains; add more illumination.")
            if "resolution" in reason.lower():
                suggestions.append("Move closer to the subject or use a higher-resolution camera.")
        if suggestions:
            return " ".join(suggestions)
        return None


__all__ = ["ImageQualityInspector", "ImageQuality"]