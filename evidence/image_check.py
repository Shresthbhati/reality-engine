"""Image evidence inspection: is this decodable, what is it, what does it carry.

Runs BEFORE any expensive reconstruction stage. Bytes that are not a
decodable image never reach COLMAP (which otherwise burns minutes on
them), and non-photographic material (floor plans, infographics, CGI
renders) is classified rather than silently mixed into observed
geometry -- the distinction survives into provenance.

Classification is deliberately conservative and says HOW it decided:
  * ``photograph``            camera EXIF present (make/model) -- best available proof
  * ``photograph_unverified`` decodes, natural colour statistics, no EXIF; TREATED as a
                              photograph but not verified (a CGI render can look like this)
  * ``flat_graphic``          few flat colours: diagram / plan / infographic
Callers may override with a declared class (the uploader knows best):
floor_plan, render, infographic, historical_photo, photograph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Classes whose pixels are observations of a real scene and may feed geometry.
GEOMETRY_CLASSES = frozenset({"photograph", "photograph_unverified"})
DECLARABLE_CLASSES = frozenset(
    {"photograph", "floor_plan", "render", "infographic", "historical_photo"}
)

MIN_SIDE_PX = 64


@dataclass(frozen=True)
class ImageFacts:
    ok: bool
    reason: Optional[str] = None
    width: int = 0
    height: int = 0
    evidence_class: str = "unknown"
    class_basis: str = ""
    #: variance of the Laplacian on a downscaled grayscale copy; a relative
    #: sharpness ranking between images, NOT a calibrated blur score.
    sharpness: Optional[float] = None
    exif: dict = field(default_factory=dict)
    #: (lat, lon) only when the file itself carries GPS EXIF -- never inferred.
    gps: Optional[tuple] = None

    @property
    def geometry_eligible(self) -> bool:
        return self.ok and self.evidence_class in GEOMETRY_CLASSES

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "width": self.width,
            "height": self.height,
            "evidence_class": self.evidence_class,
            "class_basis": self.class_basis,
            "sharpness": self.sharpness,
            "exif": self.exif,
            "gps": list(self.gps) if self.gps else None,
            "geometry_eligible": self.geometry_eligible,
        }


def _rational(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _gps_from_exif(exif) -> Optional[tuple]:
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def _deg(dms, ref):
        d, m, s = (_rational(x) for x in dms)
        if None in (d, m, s):
            return None
        val = d + m / 60.0 + s / 3600.0
        return -val if ref in ("S", "W") else val

    lat = _deg(gps[2], gps.get(1, "N"))
    lon = _deg(gps[4], gps.get(3, "E"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0.0 and lon == 0.0:
        return None  # the classic "no fix" placeholder is not a location
    return (lat, lon)


def inspect_image(path: Path | str, declared_class: Optional[str] = None) -> ImageFacts:
    """Decode + classify one image file. Never raises: a bad file is a
    result (``ok=False`` with the measured reason)."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            exif_obj = img.getexif()
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return ImageFacts(ok=False, reason=f"not a decodable image ({type(exc).__name__})")
    except Exception as exc:  # decompression bombs, truncated files
        return ImageFacts(ok=False, reason=f"image decode failed ({type(exc).__name__}: {exc})")

    if min(width, height) < MIN_SIDE_PX:
        return ImageFacts(
            ok=False, width=width, height=height,
            reason=f"image too small ({width}x{height}); need at least {MIN_SIDE_PX}px per side",
        )

    exif: dict = {}
    make = exif_obj.get(0x010F) if exif_obj else None
    model = exif_obj.get(0x0110) if exif_obj else None
    if make or model:
        exif["make"] = str(make).strip() if make else None
        exif["model"] = str(model).strip() if model else None
    try:
        sub = exif_obj.get_ifd(0x8769) if exif_obj else {}
    except Exception:
        sub = {}
    focal35 = _rational(sub.get(0xA405)) if sub else None
    if focal35:
        exif["focal_length_35mm"] = focal35
    gps = _gps_from_exif(exif_obj) if exif_obj else None

    # small grayscale copy for statistics
    scale = 256.0 / max(width, height)
    small = rgb.resize((max(8, int(width * scale)), max(8, int(height * scale))))
    arr = np.asarray(small, dtype=np.float32)
    gray = arr.mean(axis=2)
    lap = (
        gray[1:-1, 1:-1] * 4
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    sharpness = float(lap.var())

    # flat-colour statistics: share of pixels in the 6 most common quantized colours
    q = (arr // 32).astype(np.int32)
    key = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    counts = np.bincount(key.ravel(), minlength=512)
    top6 = float(np.sort(counts)[-6:].sum() / key.size)

    if declared_class:
        if declared_class not in DECLARABLE_CLASSES:
            return ImageFacts(
                ok=False, width=width, height=height,
                reason=f"unknown declared evidence class {declared_class!r}",
            )
        evidence_class, basis = declared_class, "declared by uploader"
    elif make or model:
        evidence_class, basis = "photograph", "camera EXIF (make/model) present"
    elif top6 > 0.90:
        evidence_class = "flat_graphic"
        basis = f"{top6:.0%} of pixels in 6 flat colours (diagram/plan/infographic-like)"
    else:
        evidence_class = "photograph_unverified"
        basis = "decodes with natural colour statistics; no camera EXIF (not verified)"

    return ImageFacts(
        ok=True, width=width, height=height, evidence_class=evidence_class,
        class_basis=basis, sharpness=sharpness, exif=exif, gps=gps,
    )


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
