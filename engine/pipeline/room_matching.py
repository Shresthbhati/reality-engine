"""Match proposals between the rooms of one world (milestone M4; decision: the app PROPOSES with a confidence, the user CONFIRMS).

Two kinds of proposal, both read-only (``state: "proposed"``; nothing in the world is changed by asking):

* ``door_joins``: a door of one room and a door of another that could be the same opening (similar width and height). Confirming
  one calls ``engine.pipeline.rooms.connect_rooms``.
* ``revisits``: two rooms whose measured size, ceiling height and openings agree, i.e. the second may be a second visit to the
  first. Confirming one is the user's statement "this is the same room"; the evidence is shown so the decision is informed.

Confidence is how well the MEASURED properties agree; it is not a probability. Rooms are compared by their own measurements
(footprint size, ceiling height, openings), not by position: separate sessions have no shared frame.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

from engine.pipeline.rooms import DOOR_HEIGHT_TOL_M, DOOR_WIDTH_TOL_M, room_prefix

REVISIT_MIN_SCORE = 0.6
OPENING_REL_TOL = 0.25   # widths within 25 % and sills within 0.25 m count as the same opening
OPENING_SILL_TOL_M = 0.25
WEIGHTS = {"dimensions": 0.4, "ceiling_height": 0.2, "openings": 0.4, "visual": 0.5}


def _val(d, key):
    v = (d or {}).get(key)
    return float(v["value"]) if isinstance(v, dict) and "value" in v else None


def room_signature(world, prefix: str) -> Optional[Dict[str, object]]:
    """Measured properties of the room under ``prefix``: plan size (long/short), ceiling height, openings; None when it has no room."""
    room = next((e for e in world.entities.values() if e.type.value == "room" and room_prefix(e.id) == prefix), None)
    if room is None:
        return None
    dims = (room.custom_properties or {}).get("dimensions")
    u, v = _val(dims, "layout_dimension_u_m"), _val(dims, "layout_dimension_v_m")
    openings = []
    for e in world.entities.values():
        slab = (e.custom_properties or {}).get("wall_slab")
        if e.type.value == "wall" and room_prefix(e.id) == prefix and slab:
            openings += [{"kind": o["kind"], "width_m": float(o["width_m"]), "height_m": float(o["height_m"]), "sill_m": float(o["sill_m"])}
                         for o in slab.get("openings", [])]
    return {"room": prefix, "long_m": max(u, v) if u and v else None, "short_m": min(u, v) if u and v else None,
            "height_m": _val(dims, "height_m"), "openings": sorted(openings, key=lambda o: (o["kind"], o["width_m"]))}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(a, b, 1e-9)


def similarity(a: Dict[str, object], b: Dict[str, object], visual: Optional[float] = None) -> Dict[str, object]:
    """Agreement of two room signatures: components in 0..1 (None when not comparable) and their weighted ``score``.

    ``visual`` (feature matches between the rooms' photos) joins as one more component when it is available.
    """
    comp: Dict[str, Optional[float]] = {"dimensions": None, "ceiling_height": None, "openings": None, "visual": visual}
    if None not in (a["long_m"], b["long_m"], a["short_m"], b["short_m"]):
        comp["dimensions"] = max(0.0, 1.0 - (_rel(a["long_m"], b["long_m"]) + _rel(a["short_m"], b["short_m"])) / 2.0)
    if a["height_m"] and b["height_m"]:
        comp["ceiling_height"] = max(0.0, 1.0 - _rel(a["height_m"], b["height_m"]))
    oa, ob = list(a["openings"]), list(b["openings"])
    if oa or ob:
        left, matched = list(ob), 0
        for o in oa:
            hit = next((p for p in left if p["kind"] == o["kind"] and _rel(o["width_m"], p["width_m"]) <= OPENING_REL_TOL
                        and abs(o["sill_m"] - p["sill_m"]) <= OPENING_SILL_TOL_M), None)
            if hit is not None:
                left.remove(hit)
                matched += 1
        comp["openings"] = 2.0 * matched / (len(oa) + len(ob))
    used = {k: v for k, v in comp.items() if v is not None}
    score = sum(WEIGHTS[k] * v for k, v in used.items()) / sum(WEIGHTS[k] for k in used) if used else 0.0
    return {"score": round(score, 3), **{k: (None if v is None else round(v, 3)) for k, v in comp.items()}}


def propose_matches(world, visual=None) -> Dict[str, List[dict]]:
    """Ranked, read-only proposals: ``{"door_joins": [...], "revisits": [...]}`` (best first).

    ``visual(room_a, room_b)`` may return ``{score, best_inliers, pairs_tested}`` (``room_visual.visual_agreement``) or None.
    """
    prefixes = sorted({room_prefix(e.id) for e in world.entities.values() if e.type.value == "room"})
    sigs = {p: room_signature(world, p) for p in prefixes}
    revisits: List[dict] = []
    for pa, pb in combinations(prefixes, 2):
        vis = visual(pa, pb) if visual is not None else None
        sim = similarity(sigs[pa], sigs[pb], vis["score"] if vis else None)
        if sim["score"] >= REVISIT_MIN_SCORE:
            revisits.append({"room_a": pa, "room_b": pb, "confidence": sim["score"], "state": "proposed",
                             "evidence": {k: sim[k] for k in ("dimensions", "ceiling_height", "openings", "visual")},
                             **({"visual_detail": vis} if vis else {}),
                             "note": "measured size, ceiling height and openings agree; confirm only if this is the same room"})
    doors = [e for e in world.entities.values() if e.type.value == "door" and not (e.custom_properties or {}).get("connects_to")]
    joins: List[dict] = []
    for da, db in combinations(sorted(doors, key=lambda e: e.id), 2):
        if room_prefix(da.id) == room_prefix(db.id):
            continue
        wa, wb = _val(da.custom_properties.get("dimensions"), "width_m"), _val(db.custom_properties.get("dimensions"), "width_m")
        ha, hb = _val(da.custom_properties.get("dimensions"), "height_m"), _val(db.custom_properties.get("dimensions"), "height_m")
        wd = abs(wa - wb) if None not in (wa, wb) else 0.0
        hd = abs(ha - hb) if None not in (ha, hb) else 0.0
        if wd > DOOR_WIDTH_TOL_M or hd > DOOR_HEIGHT_TOL_M:
            continue
        joins.append({"door_a": da.id, "door_b": db.id, "room_a": room_prefix(da.id), "room_b": room_prefix(db.id),
                      "width_diff_m": round(wd, 3), "height_diff_m": round(hd, 3),
                      "confidence": round(1.0 - 0.5 * wd / DOOR_WIDTH_TOL_M - 0.5 * hd / DOOR_HEIGHT_TOL_M, 3), "state": "proposed",
                      "note": "similar opening size only; nothing measured that these doors are the same, confirm to place the rooms"})
    return {"door_joins": sorted(joins, key=lambda p: -p["confidence"]), "revisits": sorted(revisits, key=lambda p: -p["confidence"])}


# ---- change set between a room and a possible second visit (read-only; the user reviews it, then accepts) ----------------------------

MATCHED_REL = 0.12      # element sizes within 12 % are the same element
UPDATED_REL = 0.35      # within 35 %: the same element, measured differently or changed
CHANGE_STATES = ("matched", "updated", "uncertain", "conflicting", "new")


def _elements(world, prefix: str) -> List[dict]:
    out: List[dict] = []
    for e in world.entities.values():
        if room_prefix(e.id) != prefix:
            continue
        cp = e.custom_properties or {}
        slab = cp.get("wall_slab")
        if e.type.value == "wall" and slab:
            out.append({"id": e.id, "kind": "wall", "size": {"length": float(slab["length_m"]), "height": float(slab["height_m"])}})
            for o in slab.get("openings", []):
                out.append({"id": o.get("entity_id") or f"{e.id}:opening", "kind": "opening",
                            "size": {"width": float(o["width_m"]), "height": float(o["height_m"])}, "label": o["kind"]})
        elif e.type.value == "furniture" or str(cp.get("entity_class", "")).startswith("furniture."):
            d = cp.get("dimensions") or {}
            out.append({"id": e.id, "kind": "furniture", "label": cp.get("category"),
                        "size": {k: _val(d, f"{k}_m") for k in ("length", "width", "height") if _val(d, f"{k}_m") is not None}})
        elif str(cp.get("entity_class", "")).startswith(("electrical.", "hvac.")):
            h = _val(cp.get("dimensions"), "height_above_floor_m")
            out.append({"id": e.id, "kind": "fixture", "label": cp["entity_class"], "size": {"height": h} if h is not None else {}})
    return out


def _size_diff(a: dict, b: dict) -> float:
    keys = set(a["size"]) & set(b["size"])
    return max((_rel(a["size"][k], b["size"][k]) for k in keys), default=0.0)


def change_set(world, old_prefix: str, new_prefix: str) -> Dict[str, object]:
    """What differs between room ``old_prefix`` and ``new_prefix`` (a possible second visit), by element kind and size.

    ``matched`` / ``updated`` (same element, size within 35 %) / ``conflicting`` (same kind, size differs more) / ``new`` (only in the new
    visit) / ``uncertain`` (seen before, not seen now: never assumed removed). Positions are not compared.
    """
    old, new = _elements(world, old_prefix), _elements(world, new_prefix)
    changes: List[dict] = []
    left = list(new)
    for a in old:
        cands = [b for b in left if b["kind"] == a["kind"] and b.get("label") == a.get("label")]
        if not cands:
            changes.append({"kind": a["kind"], "state": "uncertain", "old": a["id"], "new": None, "label": a.get("label")})
            continue
        b = min(cands, key=lambda c: _size_diff(a, c))
        left.remove(b)
        d = _size_diff(a, b)
        state = "matched" if d <= MATCHED_REL else "updated" if d <= UPDATED_REL else "conflicting"
        changes.append({"kind": a["kind"], "state": state, "old": a["id"], "new": b["id"], "label": a.get("label"), "size_diff": round(d, 3)})
    for b in left:
        changes.append({"kind": b["kind"], "state": "new", "old": None, "new": b["id"], "label": b.get("label")})
    summary: Dict[str, int] = {}
    for c in changes:
        summary[c["state"]] = summary.get(c["state"], 0) + 1
    return {"old_room": old_prefix, "new_room": new_prefix, "state": "proposed", "summary": summary, "changes": changes,
            "note": ("kinds and sizes only: positions are not compared because rooms captured in separate sessions have no shared frame; "
                     "nothing is applied until the user accepts")}
