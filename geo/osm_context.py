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
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pyproj import Proj

from provenance import Provenance, Uncertainty
from world_ir.schema_v1 import Entity, EntityType
from world_ir.statement_state import StatementState
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
    "shrub": (EntityType.VEGETATION, 1.2, 1.0),
}

# ---- roofs ---------------------------------------------------------------------------------------------------------
OSM_ROOF_SHAPES = {"flat", "gabled", "hipped", "pyramidal", "skillion", "dome", "half-hipped", "gambrel", "mansard",
                   "round"}
# building=* values that default to a pitched (gabled) roof when no roof tags exist; everything else defaults to flat.
GABLED_DEFAULT_BUILDINGS = {"house", "detached", "semidetached_house", "residential", "bungalow", "terrace", "farm",
                            "cabin", "hut"}
ROOF_LEVEL_HEIGHT_M = 2.5
DEFAULT_ROOF_PITCH_DEG = 30.0
ROOF_HEIGHT_RANGE_M = (0.5, 6.0)

# ---- landscape -----------------------------------------------------------------------------------------------------
# (osm key, osm value) -> kind. Order of the keys below is the precedence when a way carries several.
# Narrow readings (ambiguous tags): allotments -> garden (cultivated plots); recreation_ground/common -> park;
# greenfield/scrub/grassland/heath/wetland -> meadow (open low vegetation; never filled with generated trees).
AREA_KIND = {
    ("leisure", "garden"): "garden", ("leisure", "park"): "park", ("leisure", "common"): "park",
    ("leisure", "playground"): "playground", ("leisure", "pitch"): "pitch",
    ("landuse", "grass"): "lawn", ("landuse", "village_green"): "lawn", ("landuse", "flowerbed"): "flowerbed",
    ("landuse", "meadow"): "meadow", ("landuse", "greenfield"): "meadow", ("landuse", "forest"): "woodland",
    ("landuse", "orchard"): "orchard", ("landuse", "allotments"): "garden",
    ("landuse", "recreation_ground"): "park", ("landuse", "cemetery"): "cemetery",
    ("natural", "wood"): "woodland", ("natural", "scrub"): "meadow", ("natural", "grassland"): "meadow",
    ("natural", "heath"): "meadow", ("natural", "wetland"): "meadow", ("natural", "water"): "water",
}
AREA_KEYS = ("leisure", "landuse", "natural")
FILL_SPACING_M = {"woodland": 6.0, "orchard": 5.0}
MAX_FILL_PER_POLYGON = 200
MAX_FILL_PER_WORLD = 1500
TREE_ROW_SPACING_M = 7.0
MAX_TREE_ROW_TREES = 60
GENERATED_LABEL = "Generated (not captured)"
GENERATED_MAX_CONFIDENCE = 0.3
CONIFER_GENERA = {"pinus", "picea", "abies", "larix", "cedrus", "cupressus", "taxus", "juniperus", "thuja",
                  "araucaria", "pseudotsuga", "tsuga", "cryptomeria", "sequoia"}
HEDGE_HEIGHT_M, HEDGE_WIDTH_M = 1.5, 0.8


class OverpassError(RuntimeError):
    """Base for Overpass failures."""


class OverpassUnavailable(OverpassError):
    """Network failure, timeout, server error or unparseable response."""


class OverpassRateLimited(OverpassError):
    """HTTP 429 from Overpass."""


def _alt(key: str) -> str:
    return "^(" + "|".join(sorted(v for k, v in AREA_KIND if k == key)) + ")$"


_LEISURE, _LANDUSE, _NATURAL = _alt("leisure"), _alt("landuse"), _alt("natural")


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
        f'node["natural"~"^(tree|shrub)$"]{a};\nway["natural"="tree_row"]{a};\n'
        f'way["leisure"~"{_LEISURE}"]{a};\nway["landuse"~"{_LANDUSE}"]{a};\nway["natural"~"{_NATURAL}"]{a};\n'
        f'relation["type"="multipolygon"]["leisure"~"{_LEISURE}"]{a};\n'
        f'relation["type"="multipolygon"]["landuse"~"{_LANDUSE}"]{a};\n'
        f'relation["type"="multipolygon"]["natural"~"{_NATURAL}"]{a};\n);\nout geom tags;'
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
             "barrier", "amenity", "natural", "width", "public_transport", "emergency", "traffic_sign", "ref",
             "roof:shape", "roof:height", "roof:levels", "roof:angle", "roof:orientation", "leisure", "landuse",
             "diameter_crown", "leaf_type", "genus")


def _area_kind(t: dict) -> str | None:
    for key in AREA_KEYS:
        k = AREA_KIND.get((key, t.get(key)))
        if k:
            return k
    return None


def _closed(el: dict) -> bool:
    g = el.get("geometry") or []
    return len(g) >= 4 and g[0] == g[-1]


def _canopy_style(t: dict) -> tuple[str, str]:
    leaf, genus = str(t.get("leaf_type", "")).lower(), str(t.get("genus", "")).split()[0].lower() if t.get("genus") else ""
    if leaf == "needleleaved":
        return "cone", "osm_tag:leaf_type=needleleaved"
    if leaf == "broadleaved":
        return "round", "osm_tag:leaf_type=broadleaved"
    if genus in CONIFER_GENERA:
        return "cone", f"osm_tag:genus={genus} (conifer)"
    return "round", "default: round canopy (leaf_type not mapped)"


def _poly_area(pts) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))) / 2


def _obbox(pts) -> tuple[float, float]:
    """(long side, short side) of the minimum-area oriented bounding box over the polygon's edge directions."""
    best = None
    for a, b in zip(pts, pts[1:] + pts[:1]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d < 1e-9:
            continue
        ux, uz = (b[0] - a[0]) / d, (b[1] - a[1]) / d
        us = [p[0] * ux + p[1] * uz for p in pts]
        vs = [-p[0] * uz + p[1] * ux for p in pts]
        w, h = max(us) - min(us), max(vs) - min(vs)
        if best is None or w * h < best[0]:
            best = (w * h, max(w, h), min(w, h))
    return (best[1], best[2]) if best else (0.0, 0.0)


def _in_poly(x: float, z: float, pts) -> bool:
    inside = False
    for (x1, z1), (x2, z2) in zip(pts, pts[1:] + pts[:1]):
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1) + x1:
            inside = not inside
    return inside


def _roof_for(t: dict, pts, centre_only: bool) -> dict:
    """Roof spec for a building. OSM roof tags are mapper estimates, so the height is ESTIMATED either way.
    Ambiguity: unknown roof:shape values map to flat; roof:height wins over roof:levels over roof:angle over the pitch
    default. Footprints that are only a centre point stay flat (no outline to put a roof on)."""
    b = t.get("building", "yes")
    raw = str(t.get("roof:shape", "")).strip().lower().replace("_", "-")
    if centre_only:
        return {"shape": "flat", "height_m": 0.0, "basis": "flat: footprint is a centre point only"}
    if raw:
        shape = raw if raw in OSM_ROOF_SHAPES else "flat"
        basis = "OSM roof:shape tag" if raw in OSM_ROOF_SHAPES else f"roof:shape={raw} not recognised, drawn flat"
    elif b in GABLED_DEFAULT_BUILDINGS:
        shape, basis = "gabled", f"default for building={b} (not mapped)"
    else:
        return {"shape": "flat", "height_m": 0.0, "basis": f"default for building={b} (not mapped)"}
    if shape == "flat":
        return {"shape": "flat", "height_m": 0.0, "basis": basis}
    _, span = _obbox(pts)
    h_tag, lv, ang = _num(t.get("roof:height")), _num(t.get("roof:levels")), _num(t.get("roof:angle"))
    if h_tag:
        h, hb = h_tag, "OSM roof:height tag"
    elif lv:
        h, hb = lv * ROOF_LEVEL_HEIGHT_M, f"roof:levels*{ROOF_LEVEL_HEIGHT_M:g}m"
    else:
        deg = ang if ang and 0 < ang < 80 else DEFAULT_ROOF_PITCH_DEG
        run = span if shape in ("skillion", "dome") else span / 2
        h = math.tan(math.radians(deg)) * run
        hb = f"roof:angle tag {deg:g} deg" if ang and 0 < ang < 80 else f"default {DEFAULT_ROOF_PITCH_DEG:g} deg pitch"
    h = min(max(h, ROOF_HEIGHT_RANGE_M[0]), ROOF_HEIGHT_RANGE_M[1])
    orient = str(t.get("roof:orientation", "along")).strip().lower()
    return {"shape": shape, "height_m": round(h, 3), "basis": f"{basis}; height from {hb}",
            "orientation": orient if orient in ("along", "across") else "along"}


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

    def add(el, etype, kind, label, conf, dims, extra, eid=None):
        tags = el.get("tags") or {}
        props = {"osm_id": el.get("id"), "osm_type": el.get("type"),
                 "osm_tags": {k: tags[k] for k in _TAG_KEYS if k in tags},
                 "source": SOURCE, "imported": True, "kind": kind, "dimensions": dims, **extra}
        eid = eid or f"osm-{el.get('type')}-{el.get('id')}"
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

    fill_left = [MAX_FILL_PER_WORLD]

    def gen_tree(eid, parent_eid, parent_kind, pos, h, r, basis, extra=None):
        """A tree invented to illustrate a mapped wood / orchard / tree row. NEVER an imported fact: flagged exactly like
        world_ir.connectors output (generated_by, ESTIMATED, confidence <= 0.3, label, asserted False)."""
        conf = GENERATED_MAX_CONFIDENCE
        props = {
            "generated_by": "osm_context", "label": GENERATED_LABEL, "asserted": False, "state": "ESTIMATED",
            "confidence": conf, "imported": False, "kind": "tree", "derived_from": parent_eid,
            "derived_from_kind": parent_kind,
            "dimensions": {"height": {"value": round(h, 3), "unit": "m", "state": "ESTIMATED", "basis": basis},
                           "footprint_side": {"value": round(2 * r, 3), "unit": "m", "state": "ESTIMATED", "basis": basis}},
            "canopy": {"style": "round", "radius_m": round(r, 3), "height_m": round(h * 0.6, 3), "basis": basis,
                       "radius_state": "ESTIMATED"},
            "position_xyz_m": [round(pos[0], 3), 0.0, round(pos[1], 3)], **(extra or {}),
            **box([pos], r, h),
        }
        world.entities[eid] = Entity(
            id=eid, type=EntityType.VEGETATION, name="Tree (generated)", provenance=Provenance.GENERATED,
            confidence=conf, uncertainty=Uncertainty(confidence=conf, note="generated illustration, not reconstructed"),
            statement_state=StatementState.PROCEDURAL, semantic_labels=["generated", "tree"], custom_properties=props)

    def tree_row(el):
        pts = pts_of(el)
        if len(pts) < 2:
            return
        add(el, EntityType.VEGETATION, "tree_row", "Tree row", 0.5, {"width": _dim(3.0, "ESTIMATED", "default_tree_row_width")},
            {"polyline_xyz_m": [[x, 0.0, z] for x, z in pts], **box(pts, 1.5, 0.2)})
        parent = f"osm-{el.get('type')}-{el.get('id')}"
        n, carry = 0, 0.0
        for a, b in zip(pts, pts[1:]):  # fixed spacing along the line, deterministic; first tree at the first node
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            d = carry
            while d <= seg + 1e-9 and n < MAX_TREE_ROW_TREES:
                f = d / seg if seg else 0.0
                gen_tree(f"gen-tree-{el.get('type')}-{el.get('id')}-{n:03d}", parent, "tree_row",
                         (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), 6.0, 2.0,
                         f"default tree for a tree row, {TREE_ROW_SPACING_M:g} m spacing (not mapped)")
                n += 1
                d += TREE_ROW_SPACING_M
            carry = d - seg

    def fill_trees(eid, osm_el, kind, ring):
        spacing = FILL_SPACING_M[kind]
        xs, zs = [p[0] for p in ring], [p[1] for p in ring]
        w, d = max(xs) - min(xs), max(zs) - min(zs)
        step = max(spacing, math.sqrt(max(w * d, 1.0) / 4000))  # bound the candidate grid on huge polygons
        rng = random.Random(int(osm_el.get("id") or 0))  # seeded from the OSM id: same input -> same trees
        cand = []
        gx = min(xs) + step / 2
        while gx < max(xs):
            gz = min(zs) + step / 2
            while gz < max(zs):
                x, z = gx + rng.uniform(-0.3, 0.3) * step, gz + rng.uniform(-0.3, 0.3) * step
                if _in_poly(x, z, ring):
                    cand.append((x, z, rng.uniform(0.85, 1.15)))
                gz += step
            gx += step
        cap = min(MAX_FILL_PER_POLYGON, fill_left[0])
        if len(cand) > cap:  # even subsample, so a cap never bares one end of the polygon
            cand = [cand[int(i * len(cand) / cap)] for i in range(cap)] if cap > 0 else []
        for n, (x, z, s) in enumerate(cand):
            gen_tree(f"gen-tree-{eid}-{n:03d}", eid, kind, (x, z), (7.0 if kind == "woodland" else 4.5) * s,
                     (2.2 if kind == "woodland" else 1.8) * s,
                     f"default tree generated inside mapped {kind}, ~{step:g} m spacing (not mapped)")
        fill_left[0] -= len(cand)

    def add_area(el, tag_el, kind, ring, t, eid=None, extra=None):
        if len(ring) < 3 or _poly_area(ring) < 0.5:
            return
        eid = eid or f"osm-{el.get('type')}-{el.get('id')}"
        area = _poly_area(ring)
        etype = EntityType.WATER if kind == "water" else EntityType.VEGETATION
        add(tag_el, etype, kind, kind.title(), 0.45, {"area": {"value": round(area, 1), "unit": "m2", "state": "ESTIMATED",
            "basis": "polygon area of the OSM outline"}},
            {"footprint_polygon_xyz_m": [[x, 0.0, z] for x, z in ring], **(extra or {}), **box(ring, 0, 0.05)}, eid=eid)
        if kind in FILL_SPACING_M and fill_left[0] > 0:
            fill_trees(eid, tag_el, kind, ring)

    for el in elements:
        t = el.get("tags") or {}
        if el.get("type") == "relation":
            akind = _area_kind(t)
            if not akind:
                continue
            rings = []
            for m in el.get("members") or []:
                g = [(p["lat"], p["lon"]) for p in m.get("geometry") or [] if p and "lat" in p and "lon" in p]
                # narrow: only outer members that are already closed rings; split/assembled rings are not stitched
                if m.get("role") == "outer" and len(g) >= 4 and g[0] == g[-1]:
                    rings.append([xz(*p) for p in g[:-1]])
            inner = sum(1 for m in el.get("members") or [] if m.get("role") == "inner")
            for n, ring in enumerate(rings):
                eid = f"osm-relation-{el.get('id')}" + (f"-{n + 1}" if n else "")
                add_area(el, el, akind, ring, t, eid, {"holes_ignored": inner, "outer_rings": len(rings)})
            continue
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
                roof = _roof_for(t, pts, est_fp)
                rh = roof["height_m"]
                # OSM `height` is the total height including the roof, so the walls stop lower; levels/default
                # heights are wall heights and the roof sits on top of them.
                eave = max(height - rh, 0.6 * height) if (h_tag and rh) else height
                roof = {**roof, "eave_height_m": round(eave, 3)}
                if rh:
                    dims["roof_height"] = _dim(rh, "ESTIMATED", roof["basis"])
                add(el, EntityType.BUILDING, "building", "Building", conf, dims,
                    {"footprint_polygon_xyz_m": [[x, 0.0, z] for x, z in pts], "footprint_estimated": est_fp,
                     "roof": roof, **box(pts, 0, eave + rh)})
            elif (akind := _area_kind(t)) and _closed(el):
                ring = pts_of(el)[:-1]
                add_area(el, el, akind, ring, t)
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
            elif t.get("barrier") == "hedge":
                w, wd = _width_dim(t, HEDGE_WIDTH_M, "default_hedge_width")
                ht = _num(t.get("height"))
                hd = _dim(ht, "INFERRED", "osm_tag:height") if ht else _dim(HEDGE_HEIGHT_M, "ESTIMATED", "default_hedge_height")
                line(el, EntityType.VEGETATION, "hedge", "Hedge", wd, 0.5, {"barrier_kind": "hedge"},
                     {"height": hd}, h=hd["value"])
            elif t.get("natural") == "tree_row":
                tree_row(el)
            elif "barrier" in t:
                line(el, EntityType.STRUCTURE, t["barrier"] if t["barrier"] != "yes" else "barrier", "Barrier",
                     _dim(BARRIER_WIDTH_M, "ESTIMATED", "default_barrier_width"), 0.5,
                     {"barrier_kind": t["barrier"]}, h=1.0)
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
        elif t.get("natural") in ("tree", "shrub"):
            kind = t["natural"]
        if kind is None:
            continue
        etype, side, dh = POINT_DEFAULTS[kind]
        h_tag = _num(t.get("height"))
        h, hs, hb = (h_tag, "INFERRED", "osm_tag:height") if h_tag else (dh, "ESTIMATED", f"default_{kind}_height")
        dims = {"height": _dim(h, hs, hb), "footprint_side": _dim(side, "ESTIMATED", f"default_{kind}_size")}
        extra = {}
        if kind in ("tree", "shrub"):
            crown = _num(t.get("diameter_crown"))
            if crown and crown > 0:  # canopy diameter drives the drawn footprint
                side = crown
                dims["footprint_side"] = _dim(crown, "INFERRED", "osm_tag:diameter_crown")
            style, sb = _canopy_style(t)
            ch = h * (0.7 if style == "cone" else 0.6)
            extra["canopy"] = {"style": style, "radius_m": round(side / 2, 3), "height_m": round(ch, 3), "basis": sb,
                               "radius_state": "INFERRED" if crown else "ESTIMATED"}
        extra.update({"position_xyz_m": [c[0], 0.0, c[1]], **box([c], side / 2, h)})
        label_kind = t["barrier"] if kind == "barrier" and t["barrier"] != "yes" else kind
        if kind == "barrier":
            extra["barrier_kind"] = t["barrier"]
        add(el, etype, label_kind, kind.replace("_", " ").title(), 0.6, dims, extra)
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
