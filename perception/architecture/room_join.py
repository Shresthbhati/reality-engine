"""Join rooms through the openings they share (plan F7).

A walk-through capture produces one room per space. Two rooms on the same storey that share a partition with a door are
one connected home; without this pass they stay unrelated islands even though the evidence for the connection is a
detected opening on both sides of the same wall.

The join is evidence-only:

* the two wall slabs must be opposite-parallel, within one partition thickness, and overlap along their length;
* at least one DOOR/WINDOW/OPENING rectangle must coincide in world space (centre within ``OPENING_MATCH_M`` and vertical
  overlap), i.e. two detections of the SAME physical opening -- or one opening plus a blank host wall, which is still a
  physical passage;
* the result is a CONNECTS relationship between the two ROOMs plus ``connects_room_ids`` on the opening and
  ``adjacent_room_ids`` on the rooms. No geometry is created and no doorway is invented.

Two rooms that merely touch are not joined: a shared wall without an opening is not a passage.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from world_ir import EntityType, Relationship, RelationshipKind

#: openings whose world-space centres are this close are two detections of one physical opening
OPENING_MATCH_M = 0.45
#: walls must be within this plan distance to be two faces of one partition
MAX_PARTITION_M = 0.45
#: parallel walls must overlap this much along their length to bound the same passage
MIN_OVERLAP_M = 0.8
#: |dot(direction_a, direction_b)| for opposite/parallel walls
PARALLEL_DOT = 0.95


def _vec3(v) -> Optional[Tuple[float, float, float]]:
    if not v or len(v) != 3:
        return None
    return (float(v[0]), float(v[1]), float(v[2]))


def _wall(world, ent) -> Optional[dict]:
    cp = ent.custom_properties or {}
    slab = cp.get("wall_slab") or {}
    start, end = _vec3(slab.get("start_xyz_m")), _vec3(slab.get("end_xyz_m"))
    if start is None or end is None:
        return None
    length = math.dist(start, end)
    if length < 1e-6:
        return None
    d = tuple((end[i] - start[i]) / length for i in range(3))
    return {"entity": ent, "room_id": cp.get("room_id"), "start": start, "end": end, "dir": d, "length": length,
            "thickness": float(cp.get("thickness_m") or 0.1), "height": float(cp.get("height_m") or 0.0),
            "openings": list(slab.get("openings") or [])}


def _opening_centre(wall: dict, opening: dict) -> Tuple[float, float, float]:
    off = float(opening.get("offset_m") or 0.0) + float(opening.get("width_m") or 0.0) / 2.0
    sill = float(opening.get("sill_m") or 0.0)
    height = float(opening.get("height_m") or 0.0)
    p = [wall["start"][i] + wall["dir"][i] * off for i in range(3)]
    # vertical component is world axis 2 in the slab frame's own "up"; use the opening's sill mid-height along the same
    # direction as the wall's up (the slab stores it; fall back to world Z, the canonical frame)
    cp = wall["entity"].custom_properties or {}
    up = _vec3((cp.get("wall_slab") or {}).get("up")) or (0.0, 0.0, 1.0)
    mid = sill + height / 2.0
    return (p[0] + up[0] * mid, p[1] + up[1] * mid, p[2] + up[2] * mid)


def _parallel_and_close(a: dict, b: dict) -> Tuple[bool, float]:
    dot = abs(sum(a["dir"][i] * b["dir"][i] for i in range(3)))
    if dot < PARALLEL_DOT:
        return False, 0.0
    # perpendicular plan distance between the two lines
    rel = [b["start"][i] - a["start"][i] for i in range(3)]
    cross = math.dist(rel, [sum(rel[i] * a["dir"][i] for i in range(3)) * a["dir"][j] for j in range(3)])
    limit = MAX_PARTITION_M + (a["thickness"] + b["thickness"]) / 2.0
    if cross > limit:
        return False, 0.0
    # overlap of b's span projected on a's direction
    sb = sorted(sum((p[i] - a["start"][i]) * a["dir"][i] for i in range(3)) for p in (b["start"], b["end"]))
    overlap = max(0.0, min(a["length"], sb[1]) - max(0.0, sb[0]))
    return overlap >= MIN_OVERLAP_M, overlap


def _shared_openings(a: dict, b: dict) -> List[dict]:
    shared = []
    for oa in a["openings"]:
        ca = _opening_centre(a, oa)
        for ob in b["openings"]:
            cb = _opening_centre(b, ob)
            if math.dist(ca, cb) > OPENING_MATCH_M:
                continue
            va = (float(oa.get("sill_m") or 0.0), float(oa.get("sill_m") or 0.0) + float(oa.get("height_m") or 0.0))
            vb = (float(ob.get("sill_m") or 0.0), float(ob.get("sill_m") or 0.0) + float(ob.get("height_m") or 0.0))
            if min(va[1], vb[1]) - max(va[0], vb[0]) <= 0.05:
                continue
            shared.append({"a": oa.get("entity_id"), "b": ob.get("entity_id"), "kind": oa.get("kind") or ob.get("kind"),
                           "centre_xyz_m": [round(v, 4) for v in ca]})
    if shared:
        return shared
    # One side measured the opening and the other side is blank (plain paint is often detected on one face only). The
    # opening's world rectangle is still evidence of a passage through the partition; match it against the other wall's
    # plane, span and height. A one-sided match is recorded with the missing side as None, never invented.
    for source, host, key in ((a, b, "a"), (b, a, "b")):
        for o in source["openings"]:
            c = _opening_centre(source, o)
            rel = [c[i] - host["start"][i] for i in range(3)]
            along = sum(rel[i] * host["dir"][i] for i in range(3))
            cp = host["entity"].custom_properties or {}
            up = _vec3((cp.get("wall_slab") or {}).get("up")) or (0.0, 0.0, 1.0)
            vertical = sum(rel[i] * up[i] for i in range(3))
            perp = math.sqrt(max(0.0, sum(v * v for v in rel) - along * along - vertical * vertical))
            if (-0.1 <= along <= host["length"] + 0.1
                    and perp <= MAX_PARTITION_M + (source["thickness"] + host["thickness"]) / 2.0
                    and 0.0 <= vertical <= host["height"]):
                shared.append({key: o.get("entity_id"), ("b" if key == "a" else "a"): None,
                               "kind": o.get("kind"), "centre_xyz_m": [round(v, 4) for v in c],
                               "one_sided": True})
    return shared


def _connect(world, room_a, room_b, metadata: dict) -> bool:
    existing = {(r.kind, r.target_id) for r in room_a.relationships}
    if (RelationshipKind.CONNECTS, room_b.id) in existing:
        return False
    room_a.relationships.append(Relationship(kind=RelationshipKind.CONNECTS, target_id=room_b.id, confidence=0.6,
                                             metadata=metadata))
    room_b.relationships.append(Relationship(kind=RelationshipKind.CONNECTS, target_id=room_a.id, confidence=0.6,
                                             metadata=metadata))
    return True


def join_rooms_through_openings(world) -> dict:
    """Detect shared openings between rooms, link the rooms, and record the evidence. Returns the report (also stored
    at ``world.metadata['room_joins']``)."""
    report: dict = {"joins": [], "warnings": []}
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM
             and (e.custom_properties or {}).get("footprint_polygon_xyz_m")
             and not (e.custom_properties or {}).get("superseded_by")]
    by_room: Dict[str, List[dict]] = {r.id: [] for r in rooms}
    for ent in world.entities.values():
        if ent.type != EntityType.WALL or not (ent.custom_properties or {}).get("wall_slab"):
            continue
        rid = ent.custom_properties.get("room_id")
        if rid in by_room:
            w = _wall(world, ent)
            if w is not None:
                by_room[rid].append(w)
    for a, b in combinations(rooms, 2):
        best = None
        for wa in by_room.get(a.id, []):
            for wb in by_room.get(b.id, []):
                ok, overlap = _parallel_and_close(wa, wb)
                if not ok:
                    continue
                shared = _shared_openings(wa, wb)
                if not shared:
                    continue
                cand = {"room_a": a.id, "room_b": b.id, "wall_a": wa["entity"].id, "wall_b": wb["entity"].id,
                        "overlap_m": round(overlap, 3), "shared_openings": shared}
                if best is None or overlap > best["overlap_m"]:
                    best = cand
        if best is None:
            continue
        metadata = {"derived_from": "shared_opening", "wall_a": best["wall_a"], "wall_b": best["wall_b"],
                    "openings": [s["a"] or s["b"] for s in best["shared_openings"] if (s["a"] or s["b"])]}
        if _connect(world, a, b, metadata):
            for room in (a, b):
                cp = room.custom_properties
                adj = [x for x in (cp.get("adjacent_room_ids") or []) if isinstance(x, str)]
                other = b.id if room.id == a.id else a.id
                if other not in adj:
                    cp["adjacent_room_ids"] = adj + [other]
            for s in best["shared_openings"]:
                for oid in (s["a"], s["b"]):
                    opening = world.entities.get(oid) if oid else None
                    if opening is not None:
                        opening.custom_properties["connects_room_ids"] = [a.id, b.id]
                        opening.custom_properties["shared_doorway"] = True
            report["joins"].append(best)
        else:
            report["warnings"].append(f"{a.id} <-> {b.id}: already connected")
    world.metadata["room_joins"] = report
    return report
