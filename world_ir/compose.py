"""Compose several worlds (rooms, houses) placed on a Site into ONE new world.

Decisions (docs/superpowers/plans/2026-10-31-rooms-to-neighbourhoods.md): sources are never modified; every entity id
becomes ``<world_id>/<id>``; each source is rigidly placed (yaw about the vertical axis, then translated so the
source's own footprint centre lands on the placement centre); precedence on overlap is captured > imported >
generated; nothing is invented -- connectors are flagged GENERATED and containers carry no geometry.

Site frame: x = east, z = south. Yaw is a rotation of the (east, south) plane by ``yaw_deg`` (matrix
[[cos, -sin], [sin, cos]]), so +90 turns east into south (clockwise on the map).

Up axis: a source is Z-up if ``metadata.compiled_from.up == [0, 0, 1]`` else Y-up. The composed world uses the FIRST
source's convention; other sources are converted by an exact proper rotation (Z-up -> Y-up is (x,y,z) -> (x,z,-y)).
A world whose up is neither, or whose entities disagree with its declared up, is refused with ComposeError.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from typing import Iterable, Sequence

import numpy as np

from provenance import Provenance

from .connectors import Connector, is_generated
from .schema_v1 import Entity, EntityType, Geometry, Relationship, RelationshipKind
from .statement_state import StatementState
from .world_v1 import WorldIR

KINDS = ("house", "neighbourhood")
_ID_KEY = re.compile(r"(^|_)ids?$")
_POINT_KEYS = {"position", "start_xyz_m", "end_xyz_m"}
_DIR_KEYS = {"up", "direction", "outward", "e1", "e2", "normal"}
_PLAN2D_KEYS = {"footprint_polygon_m", "polygon_m", "plan_obb_centre_m"}
_P_Z_TO_Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
_VEC = {"y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
_STRUCTURE_SKIP = {EntityType.BUILDING, EntityType.STOREY, EntityType.PLOT, EntityType.BLOCK, EntityType.LOCALITY,
                   EntityType.DISTRICT}


class ComposeError(ValueError):
    """The sources cannot be composed exactly (mixed up axes, scaled sources, ...)."""


def _num3(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in v)


def _up_axis(world: WorldIR, label: str) -> str:
    up = (world.metadata.get("compiled_from") or {}).get("up")
    if up is None:
        axis = "y"
    elif _num3(up) and np.allclose(up, _VEC["z"], atol=1e-6):
        axis = "z"
    elif _num3(up) and np.allclose(up, _VEC["y"], atol=1e-6):
        axis = "y"
    else:
        raise ComposeError(f"{label}: up axis {up} is neither Y-up [0,1,0] nor Z-up [0,0,1]; cannot convert exactly")
    for e in world.entities.values():
        for u in _ups(e.custom_properties):
            if not np.allclose(u, _VEC[axis], atol=1e-3):
                raise ComposeError(f"{label}: entity '{e.id}' has up {u} but the world is {axis.upper()}-up "
                                   "(mixed up axes); cannot convert exactly")
    return axis


def _ups(obj) -> Iterable[list]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "up" and _num3(v):
                yield list(v)
            else:
                yield from _ups(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _ups(v)


class _Tf:
    """p' = M p + t (rigid, M orthogonal) plus id namespacing for one source."""

    def __init__(self, M, t, wid):
        self.M, self.t, self.wid = M, t, wid

    def pt(self, v):
        return [float(x) for x in self.M @ np.asarray(v, float) + self.t]

    def dir(self, v):
        return [float(x) for x in self.M @ np.asarray(v, float)]

    def ns(self, i):
        return f"{self.wid}/{i}" if isinstance(i, str) and i else i

    def aabb(self, lo, hi):
        pts = [self.pt([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


def _ns_val(v, tf):
    if isinstance(v, str):
        return tf.ns(v)
    if isinstance(v, list):
        return [tf.ns(i) if isinstance(i, str) else i for i in v]
    return v


def _props(d: dict, tf: _Tf, basis, warn: list):
    """Transform one custom_properties-like dict in place (recursive)."""
    for k, v in list(d.items()):
        if k in ("bounds_min", "bounds_max"):
            continue
        if k in _DIR_KEYS and _num3(v):
            d[k] = tf.dir(v)
        elif k in _POINT_KEYS or k.endswith("_xyz_m"):
            if _num3(v):
                d[k] = tf.pt(v)
            elif isinstance(v, list) and v and all(_num3(p) for p in v):
                d[k] = [tf.pt(p) for p in v]
        elif k in _PLAN2D_KEYS:
            if basis is None:
                warn.append(f"2D plan field '{k}' left unchanged: no footprint basis to re-express it in")
                continue
            s = [float(tf.t @ (tf.M @ np.asarray(b, float))) for b in basis]
            if isinstance(v, list) and len(v) == 2 and all(isinstance(i, (int, float)) for i in v):
                d[k] = [v[0] + s[0], v[1] + s[1]]
            elif isinstance(v, list):
                d[k] = [[p[0] + s[0], p[1] + s[1]] for p in v]
        elif _ID_KEY.search(k):
            d[k] = _ns_val(v, tf)
        elif isinstance(v, dict):
            _props(v, tf, basis, warn)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _props(item, tf, basis, warn)
    if _num3(d.get("bounds_min")) and _num3(d.get("bounds_max")):
        d["bounds_min"], d["bounds_max"] = tf.aabb(d["bounds_min"], d["bounds_max"])


def _xyz_dict(v):
    return {"x": v[0], "y": v[1], "z": v[2]}


def _footprint_points(d: dict) -> list:
    pts = []
    for e in d["entities"].values():
        if e["type"] == EntityType.ROOM.value:
            for p in (e.get("custom_properties") or {}).get("footprint_polygon_xyz_m") or []:
                if _num3(p):
                    pts.append(p)
    if not pts:
        for g in d["geometries"].values():
            for k in ("bounds_min", "bounds_max"):
                if g.get(k):
                    pts.append([g[k]["x"], g[k]["y"], g[k]["z"]])
    return pts


def _transform_source(world: WorldIR, wid: str, src_axis: str, tgt_axis: str, tx: float, tz: float, yaw: float,
                      warn: list) -> dict:
    d = copy.deepcopy(world.to_dict())
    h1, h2 = (0, 2) if tgt_axis == "y" else (0, 1)
    P = np.eye(3) if src_axis == tgt_axis else (_P_Z_TO_Y if src_axis == "z" else _P_Z_TO_Y.T)
    th = math.radians(yaw)
    R = np.eye(3)
    R[h1, h1], R[h1, h2], R[h2, h1], R[h2, h2] = math.cos(th), -math.sin(th), math.sin(th), math.cos(th)
    pts = _footprint_points(d)
    if pts:
        q = np.array([P @ np.asarray(p, float) for p in pts])
        lo, hi = q.min(axis=0), q.max(axis=0)
        c = (lo + hi) / 2.0
        c[3 - h1 - h2] = 0.0  # keep the vertical coordinate: only the horizontal centre is placed
    else:
        c = np.zeros(3)
        warn.append(f"{wid}: no footprint or geometry bounds; placed by its own origin")
    C = np.zeros(3)
    C[h1], C[h2] = tx, tz
    tf = _Tf(R @ P, C - R @ c, wid)
    ns = tf.ns

    rooms = {}
    for e in d["entities"].values():
        b = (e.get("custom_properties") or {}).get("footprint_basis")
        if isinstance(b, dict) and _num3(b.get("e1")) and _num3(b.get("e2")):
            rooms[e["id"]] = (b["e1"], b["e2"])

    ents = {}
    for e in d["entities"].values():
        cp = e.setdefault("custom_properties", {})
        basis = rooms.get(e["id"]) or rooms.get(cp.get("room_id"))
        _props(cp, tf, basis, warn)
        cp["source_world_id"] = wid
        e["id"] = ns(e["id"])
        for k in ("geometry_ids", "material_ids", "surface_ids", "component_ids"):
            e[k] = [ns(i) for i in e.get(k, [])]
        for r in e.get("relationships", []):
            r["target_id"] = ns(r["target_id"])
        for o in e.get("observations", []):
            o["id"] = ns(o["id"])
        for ev in e.get("temporal_events", []):
            ev["id"], ev["entity_id"] = ns(ev["id"]), ns(ev["entity_id"])
        t = e.get("transform")
        if isinstance(t, dict):
            if isinstance(t.get("position"), dict):
                p = t["position"]
                t["position"] = _xyz_dict(tf.pt([p["x"], p["y"], p["z"]]))
            extra = set(t) - {"position"}
            if extra:
                warn.append(f"entity '{e['id']}' transform keys {sorted(extra)} were not re-based")
        ents[e["id"]] = e
    d["entities"] = ents

    geoms = {}
    for g in d["geometries"].values():
        g["id"] = ns(g["id"])
        if g.get("bounds_min") and g.get("bounds_max"):
            lo, hi = tf.aabb([g["bounds_min"][a] for a in "xyz"], [g["bounds_max"][a] for a in "xyz"])
            g["bounds_min"], g["bounds_max"] = _xyz_dict(lo), _xyz_dict(hi)
        if g.get("data_uri"):
            warn.append(f"geometry '{g['id']}' references external data '{g['data_uri']}' that is not re-based")
        for o in g.get("observations", []):
            o["id"] = ns(o["id"])
        geoms[g["id"]] = g
    d["geometries"] = geoms

    for sect in ("materials", "components"):
        d[sect] = {ns(k): {**v, "id": ns(v["id"])} for k, v in d[sect].items()}
    for m in d["materials"].values():
        for o in m.get("observations", []):
            o["id"] = ns(o["id"])
    surfs = {}
    for s in d["surfaces"].values():
        s["id"], s["geometry_id"], s["material_id"] = ns(s["id"]), ns(s["geometry_id"]), ns(s["material_id"])
        if s.get("normal_direction"):
            s["normal_direction"] = _xyz_dict(tf.dir([s["normal_direction"][a] for a in "xyz"]))
        surfs[s["id"]] = s
    d["surfaces"] = surfs
    d["observations"] = {ns(k): {**v, "id": ns(v["id"])} for k, v in d["observations"].items()}
    evs = {}
    for ev in d["temporal_events"].values():
        ev["id"], ev["entity_id"] = ns(ev["id"]), ns(ev["entity_id"])
        evs[ev["id"]] = ev
    d["temporal_events"] = evs
    for cr in d["causal_relations"]:
        cr["cause_id"], cr["effect_id"] = ns(cr["cause_id"]), ns(cr["effect_id"])
    if d["transforms"]:
        warn.append(f"{wid}: {len(d['transforms'])} entity transform matrices were namespaced but not re-based")
        d["transforms"] = {ns(k): v for k, v in d["transforms"].items()}
    d["branches"], d["scenarios"] = {}, {}
    return d


# ---- containers -------------------------------------------------------------------------------------------------

def _container(eid, etype, name, wid=None) -> Entity:
    cp = {"composed_by": "compose", "organizational": True, "asserted": False}
    if wid:
        cp["source_world_id"] = wid
    return Entity(id=eid, type=etype, name=name, provenance=Provenance.INFERRED, confidence=0.5,
                  statement_state=StatementState.DERIVED, custom_properties=cp)


def _link(parent: Entity, child: Entity):
    parent.relationships.append(Relationship(RelationshipKind.CONTAINS, child.id, confidence=0.5, provenance=Provenance.INFERRED,
                                             metadata={"derived_from": "compose"}))
    child.relationships.append(Relationship(RelationshipKind.PART_OF, parent.id, confidence=0.5, provenance=Provenance.INFERRED,
                                            metadata={"derived_from": "compose"}))


def _add_containers(out: WorldIR, sources, kind: str, name: str, warn: list):
    plots = []
    for wid, sname in sources:
        mine = [e for e in out.entities.values() if e.custom_properties.get("source_world_id") == wid]
        rooms = [e for e in mine if e.type is EntityType.ROOM]
        buildings = [e for e in mine if e.type is EntityType.BUILDING]
        if not rooms and not buildings:
            warn.append(f"{wid} ('{sname}') has no rooms or building; placed without a plot or building")
            continue
        if not buildings:
            b = _container(f"compose/building/{wid}", EntityType.BUILDING, sname, wid)
            out.entities[b.id] = b
            for r in rooms:
                _link(b, r)
            buildings = [b]
        plot = _container(f"compose/plot/{wid}" if kind == "neighbourhood" else "compose/plot",
                          EntityType.PLOT, sname if kind == "neighbourhood" else name, wid if kind == "neighbourhood" else None)
        if plot.id not in out.entities:
            out.entities[plot.id] = plot
        plot = out.entities[plot.id]
        for b in buildings:
            _link(plot, b)
        if all(p.id != plot.id for p in plots):
            plots.append(plot)
    if kind == "neighbourhood" and plots:
        block = _container("compose/block", EntityType.BLOCK, f"{name} block")
        loc = _container("compose/locality", EntityType.LOCALITY, name)
        out.entities[block.id], out.entities[loc.id] = block, loc
        for p in plots:
            _link(block, p)
        _link(loc, block)


# ---- generated-vs-captured precedence --------------------------------------------------------------------------

def _rect(lo, hi, h):
    return [(lo[h[0]], lo[h[1]]), (hi[h[0]], lo[h[1]]), (hi[h[0]], hi[h[1]]), (lo[h[0]], hi[h[1]])]


def _footprints(e: Entity, geoms: dict, h) -> list:
    """Plan-view polygons of an entity: its footprint polygon, wall segment, bounds, else its geometry bounds."""
    cp = e.custom_properties
    poly = cp.get("footprint_polygon_xyz_m")
    if isinstance(poly, list) and len(poly) >= 3 and all(_num3(p) for p in poly):
        return [[(p[h[0]], p[h[1]]) for p in poly]]
    ws = cp.get("wall_slab")
    if isinstance(ws, dict) and _num3(ws.get("start_xyz_m")) and _num3(ws.get("end_xyz_m")):
        a, b = ws["start_xyz_m"], ws["end_xyz_m"]
        dx, dz = b[h[0]] - a[h[0]], b[h[1]] - a[h[1]]
        n = math.hypot(dx, dz) or 1.0
        t = max(float(ws.get("thickness_m") or 0.1), 0.05) / 2.0
        ox, oz = -dz / n * t, dx / n * t
        return [[(a[h[0]] + ox, a[h[1]] + oz), (b[h[0]] + ox, b[h[1]] + oz),
                 (b[h[0]] - ox, b[h[1]] - oz), (a[h[0]] - ox, a[h[1]] - oz)]]
    if _num3(cp.get("bounds_min")) and _num3(cp.get("bounds_max")):
        return [_rect(cp["bounds_min"], cp["bounds_max"], h)]
    out = []
    for gid in e.geometry_ids:
        g = geoms.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            out.append(_rect([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z],
                             [g.bounds_max.x, g.bounds_max.y, g.bounds_max.z], h))
    return out


def _area(poly) -> float:
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly))) / 2.0


def _clip_area(subject, clip) -> float:
    """Area of subject ∩ clip, clip CONVEX (Sutherland-Hodgman). Subject may be concave (area stays correct)."""
    if _area(clip) < 0:
        clip = clip[::-1]
    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                k = sp / (sp - sq)
                out.append((p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1])))
        if not out:
            return 0.0
    return abs(_area(out)) if len(out) >= 3 else 0.0


def _drop_overlapping_generated(out: WorldIR, axis: str, warn: list):
    h = (0, 2) if axis == "y" else (0, 1)
    captured = []
    for e in out.entities.values():
        if is_generated(e) or e.custom_properties.get("composed_by") or e.type in _STRUCTURE_SKIP:
            continue
        for poly in _footprints(e, out.geometries, h):
            captured.append((e.id, poly))
    dropped = {}
    for e in list(out.entities.values()):
        if not is_generated(e):
            continue
        # ponytail: plan-view only (vertical overlap ignored); add a height test if overpasses ever matter
        for gp in _footprints(e, out.geometries, h):
            hit = next((cid for cid, cp in captured if _clip_area(cp, gp) > 1e-3), None)
            if hit:
                dropped[e.id] = hit
                break
    for eid, hit in dropped.items():
        for gid in out.entities[eid].geometry_ids:
            out.geometries.pop(gid, None)
        del out.entities[eid]
        warn.append(f"dropped generated '{eid}': overlaps captured/imported '{hit}' (captured > generated)")
    if dropped:
        for e in out.entities.values():
            e.relationships = [r for r in e.relationships if r.target_id not in dropped]


# ---- public API -------------------------------------------------------------------------------------------------

def compose_worlds(sources: Sequence[tuple], *, name: str, kind: str, connectors: Sequence[Connector] = ()) -> WorldIR:
    """sources: (world_id, name, WorldIR, translate_x_m, translate_z_m, yaw_deg, scale[, version_id])."""
    if kind not in KINDS:
        raise ComposeError(f"kind must be one of {KINDS}")
    if not sources:
        raise ComposeError("nothing to compose: no sources")
    ids = [s[0] for s in sources]
    if len(set(ids)) != len(ids):
        raise ComposeError("a world can appear only once per composition (ids are namespaced by world id)")
    axes = [_up_axis(s[2], f"world '{s[0]}'") for s in sources]
    tgt = axes[0]
    warn: list[str] = []
    out = WorldIR(id="composed-" + hashlib.sha1(f"{kind}|{name}|{'|'.join(ids)}".encode()).hexdigest()[:16], name=name)
    provenance = []
    for s, ax in zip(sources, axes):
        wid, sname, world, tx, tz, yaw, scale = s[:7]
        vid = s[7] if len(s) > 7 else None
        if abs(float(scale) - 1.0) > 1e-9:
            raise ComposeError(f"world '{wid}' placed at scale {scale}: only scale 1 can be composed exactly "
                               "(scalar dimensions are not rescaled)")
        d = _transform_source(world, wid, ax, tgt, float(tx), float(tz), float(yaw), warn)
        w = WorldIR.from_dict(d)
        for sect in ("entities", "geometries", "materials", "surfaces", "components", "temporal_events",
                     "observations", "transforms"):
            getattr(out, sect).update(getattr(w, sect))
        out.causal_relations.extend(w.causal_relations)
        provenance.append({"world_id": wid, "version_id": vid, "name": sname, "translate": [float(tx), float(tz)],
                           "yaw": float(yaw), "scale": float(scale), "up_converted": ax != tgt})
    _add_containers(out, [(s[0], s[1]) for s in sources], kind, name, warn)
    for c in connectors:
        for g in c.geometries:
            out.geometries[g.id] = g
        for e in c.entities:
            out.entities[e.id] = e
    _drop_overlapping_generated(out, tgt, warn)
    out.global_confidence = min(s[2].global_confidence for s in sources)
    out.metadata = {
        "composed_from": provenance,
        "compose": {"kind": kind, "name": name, "source_count": len(sources)},
        "compiled_from": {"up": list(_VEC[tgt]), "composed": True},
        "compose_warnings": warn,
    }
    return out
