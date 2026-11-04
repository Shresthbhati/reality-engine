"""Geographic anchor and coordinate transforms (Blueprint 11, 16 & 30 Task 9.3).

Explicitly anchors a WorldIR's local metric coordinate frame to the physical Earth.

Invariants (Blueprint 16 & 30 Task 9.3):
1. Local geometry is in meters (local metric frame). Never put raw lat/lon into mesh coordinates.
2. Geographic anchor stores latitude, longitude, altitude_m, heading_deg, crs, accuracy_m.
3. Transforms between local metric frame (ENU: East-North-Up) and geographic coordinates (WGS84 / EPSG:4326)
   are explicit, bidirectional, and powered by PROJ (pyproj).
4. GPS is treated as a prior / anchor, not a deterministic 6DoF pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    pyproj = None
    HAS_PYPROJ = False


@dataclass(frozen=True)
class GeoAnchor:
    """Geographic anchor anchoring a WorldIR local coordinate frame to the Earth."""

    latitude: float
    longitude: float
    altitude_m: float = 0.0
    heading_deg: float = 0.0  # clockwise from North (degrees)
    crs: str = "EPSG:4326"    # WGS84
    accuracy_m: Optional[float] = None
    source: str = "exif_gps"  # "exif_gps" | "gnss_sensor" | "manual" | "cad_site"

    def __post_init__(self):
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "altitude_m": round(self.altitude_m, 2),
            "heading_deg": round(self.heading_deg, 2),
            "crs": self.crs,
            "source": self.source,
        }
        if self.accuracy_m is not None:
            d["accuracy_m"] = round(self.accuracy_m, 2)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoAnchor":
        return cls(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            altitude_m=float(data.get("altitude_m", 0.0)),
            heading_deg=float(data.get("heading_deg", 0.0)),
            crs=str(data.get("crs", "EPSG:4326")),
            accuracy_m=float(data["accuracy_m"]) if data.get("accuracy_m") is not None else None,
            source=str(data.get("source", "unknown")),
        )

    def _get_transformers(self) -> Tuple[Any, Any]:
        """Build pyproj forward and inverse transformers for local Transverse Mercator."""
        if not HAS_PYPROJ:
            raise RuntimeError("pyproj is required for exact geographic coordinate transformations")
        local_proj_crs = (
            f"+proj=tmerc +lat_0={self.latitude} +lon_0={self.longitude} "
            f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        to_local = pyproj.Transformer.from_crs("EPSG:4326", local_proj_crs, always_xy=True)
        to_geo = pyproj.Transformer.from_crs(local_proj_crs, "EPSG:4326", always_xy=True)
        return to_local, to_geo

    def local_to_geographic(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Transform local metric coordinates (x, y, z in meters) to (lat, lon, alt).

        Accounts for heading rotation (heading_deg clockwise from True North):
        Local +Y is along heading, Local +X is to the right.
        """
        # Step 1: Heading rotation to East-North-Up (ENU)
        theta = math.radians(self.heading_deg)
        east_m = x * math.cos(theta) + y * math.sin(theta)
        north_m = -x * math.sin(theta) + y * math.cos(theta)

        # Step 2: Projection from local tangent plane to WGS84
        _, to_geo = self._get_transformers()
        lon, lat = to_geo.transform(east_m, north_m)
        alt = self.altitude_m + z
        return (lat, lon, alt)

    def geographic_to_local(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Transform geographic coordinates (lat, lon, alt) to local metric (x, y, z in meters)."""
        to_local, _ = self._get_transformers()
        east_m, north_m = to_local.transform(lon, lat)

        # Inverse heading rotation
        theta = math.radians(self.heading_deg)
        x = east_m * math.cos(theta) - north_m * math.sin(theta)
        y = east_m * math.sin(theta) + north_m * math.cos(theta)
        z = alt - self.altitude_m
        return (x, y, z)

    def bounding_box_to_geojson_polygon(
        self,
        bounds_min: Tuple[float, float, float],
        bounds_max: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """Convert a local 3D bounding box to a GeoJSON Polygon Feature in WGS84 (lng, lat)."""
        min_x, min_y, _ = bounds_min
        max_x, max_y, _ = bounds_max

        # 4 corners in counter-clockwise order on floor plane
        corners_local = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),  # closed
        ]

        coords_geo = []
        for lx, ly in corners_local:
            lat, lon, _ = self.local_to_geographic(lx, ly, 0.0)
            coords_geo.append([round(lon, 7), round(lat, 7)])

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords_geo],
            },
            "properties": {
                "anchor_lat": round(self.latitude, 7),
                "anchor_lon": round(self.longitude, 7),
                "altitude_m": round(self.altitude_m, 2),
                "heading_deg": round(self.heading_deg, 2),
                "crs": self.crs,
                "source": self.source,
            },
        }


__all__ = ["GeoAnchor"]
