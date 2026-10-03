"""GeoJSON (GIS) exporter: one Polygon FEATURE per box entity -- its plan-view (XY) footprint.

Honesty rules:
  - WorldIR carries no georeference. By default the coordinates are the world's own LOCAL METRES and the document says
    so (top-level "worldir.frame", a named non-geographic "crs"). That is NOT RFC 7946 (which requires WGS84 lon/lat):
    a GIS must be told where the world is.
  - When the caller supplies ``origin_lonlat`` (the lon/lat of local (0, 0)) the footprints are converted to WGS84 with
    a small-area equirectangular approximation, ASSUMING local +x = east and +y = north (only the caller knows that).
    Accuracy degrades with distance from the origin (fine for a site, wrong for a country); the document records both
    the assumption and the approximation.
  - Height and base elevation are properties, not geometry (GeoJSON footprints are 2-D here).
  - Entities without a position or real bounds are skipped with a reason (see exporters/boxes.py).
"""

from __future__ import annotations

import json
import math
from typing import Optional, Tuple

from exporters.boxes import collect_boxes, make_report

_EARTH_RADIUS_M = 6378137.0


def _to_lonlat(x: float, y: float, origin: Tuple[float, float]) -> list:
    lon0, lat0 = origin
    lat = lat0 + math.degrees(y / _EARTH_RADIUS_M)
    lon = lon0 + math.degrees(x / (_EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return [round(lon, 9), round(lat, 9)]


def export_to_geojson(world, origin_lonlat: Optional[Tuple[float, float]] = None) -> dict:
    return _build(world, origin_lonlat)[0]


def _build(world, origin_lonlat):
    boxes, skipped, reasons = collect_boxes(world)
    if origin_lonlat is not None and not (-180.0 <= origin_lonlat[0] <= 180.0 and -90.0 <= origin_lonlat[1] <= 90.0):
        raise ValueError(f"origin_lonlat {origin_lonlat!r} is not a valid (lon, lat)")
    features = []
    for b in boxes:
        (x0, y0, z0), (x1, y1, z1) = b.min, b.max
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]       # counter-clockwise, closed
        coords = [_to_lonlat(x, y, origin_lonlat) if origin_lonlat else [round(x, 6), round(y, 6)] for x, y in ring]
        features.append({
            "type": "Feature", "id": b.id,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"name": b.name, "worldir_type": b.type, "worldir_provenance": b.provenance,
                           "worldir_confidence": b.confidence, "base_elevation_m": round(z0, 6),
                           "height_m": round(z1 - z0, 6)},
        })
    if origin_lonlat is None:
        meta = {"frame": "local metres, Z-up; NOT georeferenced",
                "note": "coordinates are the world's own local metres, not WGS84 lon/lat; supply origin_lonlat to project"}
        doc = {"type": "FeatureCollection",
               "crs": {"type": "name", "properties": {"name": "urn:worldir:crs:local-metres"}},
               "worldir": meta, "features": features}
    else:
        doc = {"type": "FeatureCollection",
               "worldir": {"frame": "WGS84 lon/lat (EPSG:4326)", "origin_lonlat": list(origin_lonlat),
                           "assumption": "local +x = east, +y = north",
                           "approximation": "small-area equirectangular; accuracy degrades with distance from the origin"},
               "features": features}
    return doc, boxes, skipped, reasons


def export_to_geojson_with_report(world, artifact_store=None, origin_lonlat: Optional[Tuple[float, float]] = None):
    doc, boxes, skipped, reasons = _build(world, origin_lonlat)
    content = json.dumps(doc, sort_keys=True, indent=1)
    return content, make_report("geojson", world, content, boxes, skipped, reasons)
