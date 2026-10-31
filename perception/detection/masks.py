"""Pixel masks for transient objects, so reconstruction ignores people/vehicles."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from perception.detection.policy import classify


def _region(shape: Tuple[int, int], det, margin: int) -> np.ndarray:
    h, w = shape[:2]
    if det.mask is not None:
        m = np.asarray(det.mask).astype(bool)
        if m.shape != (h, w):
            raise ValueError(f"detection mask shape {m.shape} != image {(h, w)}")
        return m
    out = np.zeros((h, w), bool)
    x0, y0, x1, y1 = det.box_xyxy
    x0, y0 = max(int(np.floor(x0)) - margin, 0), max(int(np.floor(y0)) - margin, 0)
    x1, y1 = min(int(np.ceil(x1)) + margin, w), min(int(np.ceil(y1)) + margin, h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = True
    return out


def transient_mask(image_shape, detections: Sequence, margin: int = 4) -> np.ndarray:
    """True = pixel belongs to a transient object. Kept structure wins overlaps."""
    h, w = image_shape[:2]
    trans = np.zeros((h, w), bool)
    keep = np.zeros((h, w), bool)
    for d in detections:
        c = classify(d.label)
        if c == "drop":
            trans |= _region((h, w), d, margin)
        elif c == "keep":
            keep |= _region((h, w), d, 0)
    return trans & ~keep


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray, fill: str = "inpaint"):
    """Return (masked_image, valid_pixels); valid_pixels is True where feature matching may look."""
    if fill not in ("inpaint", "black", "transparent"):
        raise ValueError(f"unknown fill '{fill}'")
    mask = np.asarray(mask, bool)
    img = np.ascontiguousarray(image_rgb)
    if fill == "transparent":
        out = np.dstack([img, np.where(mask, 0, 255).astype(np.uint8)])
    elif fill == "black":
        out = img.copy()
        out[mask] = 0
    else:
        out = cv2.inpaint(img, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA) if mask.any() else img.copy()
    return out, ~mask


def mask_coverage(mask: np.ndarray) -> float:
    m = np.asarray(mask, bool)
    return float(m.mean()) if m.size else 0.0
