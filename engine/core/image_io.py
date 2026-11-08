"""Image I/O and orientation normalization utility (core layer).

Provides safe image loading with EXIF orientation normalization and memory-bomb
mitigation (pixel capping). Belongs to engine.core so any pipeline, evidence,
perception, or reconstruction module can consume it without inter-sibling coupling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

#: Decoded-pixel cap (width * height). Phone cameras reach ~50-200 MP; an
#: unbounded image otherwise exhausts memory (RGB decode = 3 bytes/px).
#: Override with REALITY_ENGINE_MAX_IMAGE_PIXELS.
DEFAULT_MAX_IMAGE_PIXELS = 200_000_000
_ORIENTATION_TAG = 0x0112


class ImageTooLarge(ValueError):
    """Header-declared pixel count exceeds the cap; raised before decoding."""


def max_image_pixels() -> int:
    """Return the configured maximum allowed pixels for image decoding."""
    try:
        v = int(os.environ.get("REALITY_ENGINE_MAX_IMAGE_PIXELS", ""))
        return v if v > 0 else DEFAULT_MAX_IMAGE_PIXELS
    except ValueError:
        return DEFAULT_MAX_IMAGE_PIXELS


def open_oriented(path: Union[str, Path], max_pixels: Optional[int] = None):
    """Open an image for geometry/depth/detection: EXIF orientation applied
    (so pixels match the upright scene the intrinsics describe) and the
    pixel cap enforced from the header, before any decode.

    Returns a fully loaded PIL image. Raises ``ImageTooLarge`` over the cap
    and standard PIL errors for undecodable files.
    """
    from PIL import Image, ImageOps

    cap = max_pixels or max_image_pixels()
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w * h > cap:
                raise ImageTooLarge(f"image is {w}x{h} ({w * h:,} px); decoded-pixel cap is {cap:,}")
            img.load()
            return ImageOps.exif_transpose(img)
    except Image.DecompressionBombError as exc:
        raise ImageTooLarge(f"decompression bomb refused ({exc})") from exc
