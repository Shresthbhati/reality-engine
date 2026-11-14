"""Camera intrinsics COLMAP itself estimated, read back from a sparse model. Nothing here is invented.

Shared by the API job runner (a world's earlier committed model) and the pose stage (the model this run just solved), so
photographs without EXIF can still feed depth, perception and meshing -- labelled ``estimated``, never ``trusted``.
"""

from __future__ import annotations

import statistics
import struct
from pathlib import Path
from typing import Optional, Sequence, Tuple


def read_colmap_cameras(model_dir: Path) -> list[dict]:
    """Camera intrinsics from a sparse model's cameras.txt (or cameras.bin).
    Returns [{model, width, height, params}]; [] when unreadable."""
    out: list[dict] = []
    txt = model_dir / "cameras.txt"
    try:
        if txt.is_file():
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                out.append({"model": parts[1], "width": int(parts[2]), "height": int(parts[3]),
                            "params": [float(v) for v in parts[4:]]})
            return out
        binf = model_dir / "cameras.bin"
        if binf.is_file():
            names = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                     4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
                     8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}
            data = binf.read_bytes()
            (n,), off = struct.unpack_from("<Q", data, 0), 8
            for _ in range(n):
                _cid, mid, w, h = struct.unpack_from("<iiQQ", data, off)
                off += 24
                name, k = names[mid]
                out.append({"model": name, "width": w, "height": h,
                            "params": list(struct.unpack_from(f"<{k}d", data, off))})
                off += 8 * k
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        return []
    return out


def _pinhole_values(cams: Sequence[dict], size: Tuple[int, int]) -> list:
    """(fx, fy, cx, cy) per camera; [] when any camera is fisheye, has the wrong size or no usable parameters."""
    vals = []
    for c in cams:
        p, m = c["params"], c["model"]
        if "FISHEYE" in m or (c["width"], c["height"]) != tuple(size):
            return []
        if m in ("PINHOLE", "OPENCV", "FULL_OPENCV") and len(p) >= 4:
            vals.append((p[0], p[1], p[2], p[3]))
        elif len(p) >= 3:            # single-focal models: f, cx, cy, ...
            vals.append((p[0], p[0], p[1], p[2]))
    return vals


def agreed_intrinsics(cams: Sequence[dict], size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    """(fx, fy, cx, cy) only when every camera agrees (1%), is a pinhole-like (non-fisheye) model and has the images'
    real size. Else None. Radial distortion terms are ignored (the pinhole approximation is why trust stays ``estimated``)."""
    vals = _pinhole_values(cams, size)
    if not vals or any(abs(a - b) > 0.01 * abs(a) for v in vals[1:] for a, b in zip(vals[0], v)):
        return None
    return vals[0]


def median_intrinsics(
    cams: Sequence[dict], size: Tuple[int, int], max_spread: float = 0.10
) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
    """((fx, fy, cx, cy), focal_spread) using the per-camera MEDIAN, for photographs that do not share one physical
    camera (COLMAP then fits one per image). ``focal_spread`` = (max - min) / median of fx; refused (None) above
    ``max_spread`` -- a wildly varying focal means the model is unreliable, not that a median is safe."""
    vals = _pinhole_values(cams, size)
    if not vals:
        return None
    cols = list(zip(*vals))
    med = tuple(float(statistics.median(c)) for c in cols)
    spread = (max(cols[0]) - min(cols[0])) / med[0] if med[0] else float("inf")
    if spread > max_spread:
        return None
    return med, float(spread)
