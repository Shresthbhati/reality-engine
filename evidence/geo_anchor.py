"""EXIF GPS extraction and geo anchor derivation (Blueprint 16 & 30 Task 9.3).

Extracts geographic anchors from capture evidence (photos, metadata, sensor streams)
and attaches them to WorldIR.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import ExifTags, Image

from world_ir.geo import GeoAnchor
from world_ir.world_v1 import WorldIR


def _rational_to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_deg(dms: Any, ref: Optional[str]) -> Optional[float]:
    if not dms or len(dms) < 3:
        return None
    try:
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
        deg = d + (m / 60.0) + (s / 3600.0)
        if ref and str(ref).strip().upper() in ("S", "W"):
            deg = -deg
        return deg
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_gps_from_image_path(image_path: Union[Path, str]) -> Optional[Dict[str, float]]:
    """Extract GPS latitude, longitude, and optional altitude from image EXIF."""
    p = Path(image_path)
    if not p.is_file():
        return None

    try:
        with Image.open(p) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(0x8825)
            if not gps_ifd:
                return None

            lat_ref = gps_ifd.get(1)
            lat_dms = gps_ifd.get(2)
            lon_ref = gps_ifd.get(3)
            lon_dms = gps_ifd.get(4)

            lat = _dms_to_deg(lat_dms, lat_ref)
            lon = _dms_to_deg(lon_dms, lon_ref)

            if lat is None or lon is None:
                return None

            res: Dict[str, float] = {"latitude": lat, "longitude": lon}
            alt = _rational_to_float(gps_ifd.get(6))
            if alt is not None:
                alt_ref = gps_ifd.get(5, 0)
                if alt_ref == 1:
                    alt = -alt
                res["altitude_m"] = alt

            return res
    except Exception:
        return None


def extract_geo_anchor_from_evidence(
    items: Sequence[Any],
    heading_deg: float = 0.0,
) -> Optional[GeoAnchor]:
    """Derive a canonical GeoAnchor from a collection of evidence items or metadata.

    Accepts:
    - Sequence of image file paths (Path or str).
    - Sequence of metadata dicts with "latitude"/"longitude" or "latitude_deg"/"longitude_deg" or "gps".
    - Objects with .metadata or .sensor_metadata attributes.

    Returns:
    - GeoAnchor if valid GPS measurements are present; None if no GPS exists.
    """
    coords: List[Tuple[float, float, float]] = []

    for item in items:
        # Case 1: Image file path
        if isinstance(item, (str, Path)):
            p = Path(item)
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".png"):
                gps = extract_gps_from_image_path(p)
                if gps:
                    coords.append((gps["latitude"], gps["longitude"], gps.get("altitude_m", 0.0)))
            continue

        # Case 2: Dict
        meta: Dict[str, Any] = {}
        if isinstance(item, dict):
            meta = item
        elif hasattr(item, "sensor_metadata") and isinstance(item.sensor_metadata, dict):
            meta = item.sensor_metadata
        elif hasattr(item, "metadata") and isinstance(item.metadata, dict):
            meta = item.metadata

        lat: Optional[float] = None
        lon: Optional[float] = None
        alt: float = 0.0

        if "latitude" in meta and "longitude" in meta:
            lat = _rational_to_float(meta["latitude"])
            lon = _rational_to_float(meta["longitude"])
            alt = _rational_to_float(meta.get("altitude_m", 0.0)) or 0.0
        elif "latitude_deg" in meta and "longitude_deg" in meta:
            lat = _rational_to_float(meta["latitude_deg"])
            lon = _rational_to_float(meta["longitude_deg"])
            alt = _rational_to_float(meta.get("altitude_m", 0.0)) or 0.0
        elif "gps" in meta and isinstance(meta["gps"], (tuple, list)) and len(meta["gps"]) >= 2:
            lat = _rational_to_float(meta["gps"][0])
            lon = _rational_to_float(meta["gps"][1])
            if len(meta["gps"]) >= 3:
                alt = _rational_to_float(meta["gps"][2]) or 0.0

        if lat is not None and lon is not None:
            # Reject null island / invalid ranges
            if not (lat == 0.0 and lon == 0.0) and (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0):
                coords.append((lat, lon, alt))

    if not coords:
        return None

    # Compute mean centroid
    mean_lat = sum(c[0] for c in coords) / len(coords)
    mean_lon = sum(c[1] for c in coords) / len(coords)
    mean_alt = sum(c[2] for c in coords) / len(coords)

    # Calculate spatial dispersion radius as accuracy estimate
    if len(coords) > 1:
        # 1 deg lat ~= 111,000 meters
        lat_scale = 111_000.0
        lon_scale = 111_000.0 * math.cos(math.radians(mean_lat))
        dispersions = [
            math.sqrt(((c[0] - mean_lat) * lat_scale) ** 2 + ((c[1] - mean_lon) * lon_scale) ** 2)
            for c in coords
        ]
        accuracy_m = round(max(5.0, sum(dispersions) / len(dispersions)), 2)
    else:
        accuracy_m = 5.0  # default consumer GNSS uncertainty

    return GeoAnchor(
        latitude=round(mean_lat, 7),
        longitude=round(mean_lon, 7),
        altitude_m=round(mean_alt, 2),
        heading_deg=round(heading_deg, 2),
        crs="EPSG:4326",
        accuracy_m=accuracy_m,
        source="exif_gps",
    )


def attach_geo_anchor_to_world(world: WorldIR, geo_anchor: GeoAnchor) -> WorldIR:
    """Attach a GeoAnchor to a WorldIR instance while preserving local metric geometry.

    Local coordinates in world.geometries and world.entities remain 100% untouched.
    """
    world.geo_anchor = geo_anchor
    return world


__all__ = [
    "extract_gps_from_image_path",
    "extract_geo_anchor_from_evidence",
    "attach_geo_anchor_to_world",
]
