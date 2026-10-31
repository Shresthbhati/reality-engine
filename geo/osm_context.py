"""OpenStreetMap (Overpass) context -> WorldIR.

Map data is not a measurement: every entity is INFERRED (never OBSERVED), confidence is capped below 1, and any
dimension filled from a default rather than a tag is flagged ESTIMATED. Nothing is ever faked when Overpass is
unreachable -- a typed error is raised instead.

Frame: x = east, z = south (north = -z), y = up (canonical Y-up). Origin is the anchor (lat, lon).
Licence: ODbL, (c) OpenStreetMap contributors.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pyproj import Proj

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_URLS = (OVERPASS_URL, "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")
HEADERS = {"User-Agent": "RealityEngine/0.1 (research prototype)", "Accept": "application/json"}
ATTRIBUTION = "(c) OpenStreetMap contributors"
SOURCE = "OpenStreetMap"
LEVEL_HEIGHT_M = 3.0
DEFAULT_BUILDING_HEIGHT_M = 6.0
BRIDGE_DECK_PER_LAYER_M = 5.0
CENTER_ONLY_FOOTPRINT_M = 12.0  # when Overpass gave only a centre point, not an outline
MAX_CONFIDENCE = 0.9  # map data is not a measurement

ROAD_WIDTH_M = {"motorway": 14.0, "trunk": 12.0, "primary": 10.0, "secondary": 8.0, "tertiary": 7.0,
                "residential": 6.0, "service": 4.0, "unclassified": 5.0, "living_street": 5.0, "track": 3.0}
SIDEWALK_WIDTH_M = 2.0
RAIL_WIDTH_M = 3.0
BRIDGE_WIDTH_M = 6.0
BARRIER_WIDTH_M = 0.3
RAIL_KIND = {"subway": "metro_line", "light_rail": "metro_line", "rail": "rail", "tram": "rail"}
# kind -> (entity type, footprint side m, height m) for point features; sizes are defaults, flagged ESTIMATED.
POINT_DEFAULTS = {
    "lamp_post": (EntityType.INFRASTRUCTURE, 0.3, 5.0), "bin": (EntityType.INFRASTRUCTURE, 0.5, 0.9),
    "post_box": (EntityType.INFRASTRUCTURE, 0.5, 1.2), "bench": (EntityType.INFRASTRUCTURE, 1.5, 0.8),
    "traffic_sign": (EntityType.INFRASTRUCTURE, 0.3, 2.5), "fire_hydrant": (EntityType.INFRASTRUCTURE, 0.3, 0.8),
    "station": (EntityType.INFRASTRUCTURE, 20.0, 6.0), "station_entrance": (EntityType.INFRASTRUCTURE, 3.0, 3.0),
    "barrier": (EntityType.STRUCTURE, 1.5, 1.0), "tree": (EntityType.VEGETATION, 3.0, 6.0),
}


class OverpassError(RuntimeError):
    """Base for Overpass failures."""


class OverpassUnavailable(OverpassError):
    """Network failure, timeout, server error or unparseable response."""


class OverpassRateLimited(OverpassError):
    """HTTP 429 from Overpass."""


def build_query(lat: float, lon: float, radius_m: float, timeout_s: int = 50) -> str:
    a = f"(around:{radius_m:g},{lat:.6f},{lon:.6f})"
    return (
        f"[out:json][timeout:{timeout_s}];(\n"
        f'way["building"]{a};\nway["highway"]{a};\n'
        f'way["railway"~"^(subway|light_rail|rail|tram)$"]{a};\nway["bridge"]{a};\nway["barrier"]{a};\n'
        f'node["railway"~"^(station|subway_entrance)$"]{a};\nnode["public_transport"="station"]{a};\n'
        f'node["highway"~"^(street_lamp|traffic_signals|stop)$"]{a};\n'
        f'node["amenity"~"^(waste_basket|waste_disposal|post_box|bench)$"]{a};\n'
        f'node["barrier"]{a};\nnode["traffic_sign"]{a};\nnode["emergency"="fire_hydrant"]{a};\n'
        f'node["natural"="tree"]{a};\n);\nout geom tags;'
    )


def _cache_path(cache_dir: Path, lat: float, lon: float, radius_m: float) -> Path:
    key = f"{round(lat, 4):.4f}_{round(lon, 4):.4f}_{radius_m:g}"
    return Path(cache_dir) / f"overpass_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def fetch_overpass(lat: float, lon: float, radius_m: float, *, client: httpx.Client | None = None,
                   cache_dir: Path | str | None = None, timeout_s: float = 60.0) -> tuple[dict, str]:
    """Return (overpass json, source) where source is 'cache' or 'overpass'. Never fabricates data."""
    path = _cache_path(Path(cache_dir), lat, lon, radius_m) if cache_dir else None
    if path and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")), "cache"
        except (OSError, ValueError):
            pass  # corrupt cache entry: refetch
    own = client is None
    client = client or httpx.Client(timeout=timeout_s)
    resp, last_error = None, None
    try:
        # the public instances are often overloaded: fall through the mirrors, report the last failure if all fail
        for url in OVERPASS_URLS:
            try:
                r = client.post(url, data={"data": build_query(lat, lon, radius_m)}, headers=HEADERS)
            except httpx.HTTPError as exc:
                last_error = OverpassUnavailable(f"Overpass unreachable: {type(exc).__name__}: {exc}")
                continue
            if r.status_code == 429:
                last_error = OverpassRateLimited("Overpass rate limit reached (HTTP 429); retry later")
                continue
            if r.status_code != 200:
                last_error = OverpassUnavailable(f"Overpass returned HTTP {r.status_code}")
                continue
            resp = r
            break
    finally:
        if own:
            client.close()
    if resp is None:
        raise last_error
    try:
        data = resp.json()
    except ValueError as exc:
        raise OverpassUnavailable("Overpass returned a non-JSON response") from exc
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        raise OverpassUnavailable("Overpass response has no 'elements' list")
    if "runtime error" in str(data.get("remark", "")):
        raise OverpassUnavailable(f"Overpass runtime error: {data['remark']}")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data, "overpass"


# ---- conversion ----------------------------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _num(v) -> float | None:
    m = _NUM.search(str(v)) if v is not None else None
    return float(m.group()) if m else None


def _dim(value: float, state: str, basis: str) -> dict:
    return {"value": round(value, 3), "unit": "m", "state": state, "basis": basis}


def _width_dim(tags: dict, default: float, default_basis: str) -> tuple[float, dict]:
    w = _num(tags.get("width"))
    return (w, _dim(w, "INFERRED", "osm_tag:width")) if w else (default, _dim(default, "ESTIMATED", default_basis))


_TAG_KEYS = ("name", "building", "building:levels", "height", "highway", "railway", "bridge", "tunnel", "layer",
             "barrier", "amenity", "natural", "width", "public_transport", "emergency", "traffic_sign", "ref")


def osm_to_world(elements: list[dict], lat: float, lon: float, name: str, *, radius_m: float | None = None,
                 fetched_at: str | None = None) -> WorldIR:
    proj = Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")

    def xz(la: float, lo: float) -> tuple[float, float]:
        e, n = proj(lo, la)
        return round(e, 3), round(-n, 3)

    world = WorldIR(name=name)
    world.global_provenance = Provenance.INFERRED
    world.metadata["compiled_from"] = {"source": SOURCE, "up": [0, 1, 0]}
    # the local tangent-plane projection is in metres; sizes taken from tags/defaults stay ESTIMATED (caps every state)
    world.metadata["scale"] = {"state": "metric", "method": "geographic projection (OpenStreetMap)",
                               "dimension_state": "ESTIMATED"}
    world.metadata["geo"] = {
        "anchor_lat": lat, "anchor_lon": lon, "radius_m": radius_m, "source": SOURCE, "licence": "ODbL",
        "attribution": ATTRIBUTION,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def add(el, etype, kind, label, conf, dims, extra):
        tags = el.get("tags") or {}
        props = {"osm_id": el.get("id"), "osm_type": el.get("type"),
                 "osm_tags": {k: tags[k] for k in _TAG_KEYS if k in tags},
                 "source": SOURCE, "imported": True, "kind": kind, "dimensions": dims, **extra}
        eid = f"osm-{el.get('type')}-{el.get('id')}"
        world.entities[eid] = Entity(
            id=eid, type=etype, name=tags.get("name") or label, provenance=Provenance.INFERRED,
            confidence=round(min(MAX_CONFIDENCE, max(0.05, conf)), 2), semantic_labels=[kind], custom_properties=props)

    def pts_of(el) -> list[tuple[float, float]]:
        return [xz(g["lat"], g["lon"]) for g in el.get("geometry") or [] if "lat" in g and "lon" in g]

    def center_of(el) -> tuple[float, float] | None:
        if "lat" in el and "lon" in el:
            return xz(el["lat"], el["lon"])
        c = el.get("center")
        return xz(c["lat"], c["lon"]) if c else None

    def box(points, half, h, y0=0.0):
        xs, zs = [p[0] for p in points], [p[1] for p in points]
        return {"bounds_min": [min(xs) - half, y0, min(zs) - half], "bounds_max": [max(xs) + half, y0 + h, max(zs) + half]}

    def line(el, etype, kind, label, width_dim, conf, extra=None, extra_dims=None, y0=0.0, h=0.2):
        pts = pts_of(el)
        if len(pts) < 2:
            return
        w = width_dim["value"]
        add(el, etype, kind, label, conf, {"width": width_dim, **(extra_dims or {})},
            {**(extra or {}), "polyline_xyz_m": [[x, y0, z] for x, z in pts], **box(pts, w / 2, h, y0)})

    for el in elements:
        t = el.get("tags") or {}
        if el.get("type") == "way":
            if "building" in t:
                pts = pts_of(el)
                est_fp = len(pts) < 3
                if est_fp:
                    c = center_of(el)
                    if c is None:
                        continue
                    s = CENTER_ONLY_FOOTPRINT_M / 2
                    pts = [(c[0] - s, c[1] - s), (c[0] + s, c[1] - s), (c[0] + s, c[1] + s), (c[0] - s, c[1] + s)]
                elif pts[0] == pts[-1]:
                    pts = pts[:-1]
                h_tag, lv = _num(t.get("height")), _num(t.get("building:levels"))
                if h_tag:
                    height, hs, hb = h_tag, "INFERRED", "osm_tag:height"
                elif lv:
                    height, hs, hb = lv * LEVEL_HEIGHT_M, "ESTIMATED", f"building:levels*{LEVEL_HEIGHT_M:g}m"
                else:
                    height, hs, hb = DEFAULT_BUILDING_HEIGHT_M, "ESTIMATED", "default_building_height"
                conf = 0.4 + (0.15 if h_tag else 0) + (0.15 if lv else 0) - (0.1 if est_fp else 0)
                dims = {"height": _dim(height, hs, hb)}
                if est_fp:
                    dims["footprint_side"] = _dim(CENTER_ONLY_FOOTPRINT_M, "ESTIMATED", "centre_point_only")
                add(el, EntityType.BUILDING, "building", "Building", conf, dims,
                    {"footprint_polygon_xyz_m": [[x, 0.0, z] for x, z in pts], "footprint_estimated": est_fp,
                     **box(pts, 0, height)})
            elif t.get("highway") == "steps":
                _, wd = _width_dim(t, 1.5, "default_steps_width")
                line(el, EntityType.STAIRS, "steps", "Steps", wd, 0.5, h=2.0)
            elif t.get("bridge") not in (None, "no"):
                layer = _num(t.get("layer")) or 1.0
                _, wd = _width_dim(t, BRIDGE_WIDTH_M, "default_bridge_width")
                deck = _dim(layer * BRIDGE_DECK_PER_LAYER_M, "ESTIMATED", "layer*5m")
                line(el, EntityType.INFRASTRUCTURE, "bridge", "Bridge", wd, 0.45,
                     {"carries": t.get("highway") or t.get("railway"), "layer": layer}, {"deck_elevation": deck},
                     y0=deck["value"], h=1.0)
            elif t.get("railway") in RAIL_KIND:
                tunnel = t.get("tunnel") not in (None, "no")
                line(el, EntityType.INFRASTRUCTURE, RAIL_KIND[t["railway"]], t["railway"].replace("_", " ").title(),
                     _dim(RAIL_WIDTH_M, "ESTIMATED", "default_rail_width"), 0.45,
                     {"railway": t["railway"], "tunnel": tunnel, "layer": _num(t.get("layer")),
                      "underground": tunnel}, h=0.3)
            elif "barrier" in t:
                line(el, EntityType.STRUCTURE, t["barrier"] if t["barrier"] != "yes" else "barrier", "Barrier",
                     _dim(BARRIER_WIDTH_M, "ESTIMATED", "default_barrier_width"), 0.5, h=1.0)
            elif "highway" in t:
                hw = t["highway"]
                if hw in ("footway", "pedestrian", "path", "cycleway") or t.get("footway") == "sidewalk":
                    w, wd = _width_dim(t, SIDEWALK_WIDTH_M, "default_sidewalk_width")
                    line(el, EntityType.SIDEWALK, "sidewalk", "Sidewalk", wd, 0.5 + (0.1 if "width" in t else 0))
                else:
                    w, wd = _width_dim(t, ROAD_WIDTH_M.get(hw, 5.0), f"default_width_for_highway={hw}")
                    line(el, EntityType.ROAD, "road", "Road", wd, 0.5 + (0.1 if "width" in t else 0))
            continue
        if el.get("type") != "node":
            continue
        c = center_of(el)
        if c is None:
            continue
        kind = None
        if t.get("railway") == "station" or t.get("public_transport") == "station":
            kind = "station"
        elif t.get("railway") == "subway_entrance":
            kind = "station_entrance"
        elif t.get("highway") == "street_lamp":
            kind = "lamp_post"
        elif t.get("amenity") in ("waste_basket", "waste_disposal"):
            kind = "bin"
        elif t.get("amenity") == "post_box":
            kind = "post_box"
        elif t.get("amenity") == "bench":
            kind = "bench"
        elif "barrier" in t:
            kind = "barrier"
        elif t.get("highway") in ("traffic_signals", "stop") or "traffic_sign" in t:
            kind = "traffic_sign"
        elif t.get("emergency") == "fire_hydrant":
            kind = "fire_hydrant"
        elif t.get("natural") == "tree":
            kind = "tree"
        if kind is None:
            continue
        etype, side, dh = POINT_DEFAULTS[kind]
        h_tag = _num(t.get("height"))
        h, hs, hb = (h_tag, "INFERRED", "osm_tag:height") if h_tag else (dh, "ESTIMATED", f"default_{kind}_height")
        extra = {"position_xyz_m": [c[0], 0.0, c[1]], **box([c], side / 2, h)}
        label_kind = t["barrier"] if kind == "barrier" and t["barrier"] != "yes" else kind
        if kind == "barrier":
            extra["barrier_kind"] = t["barrier"]
        add(el, etype, label_kind, kind.replace("_", " ").title(), 0.6,
            {"height": _dim(h, hs, hb), "footprint_side": _dim(side, "ESTIMATED", f"default_{kind}_size")}, extra)
    return world


def counts_by_kind(world: WorldIR) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in world.entities.values():
        k = e.custom_properties.get("kind", "unknown")
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def estimated_count(world: WorldIR) -> int:
    return sum(1 for e in world.entities.values()
               if any(d.get("state") == "ESTIMATED" for d in (e.custom_properties.get("dimensions") or {}).values()))
