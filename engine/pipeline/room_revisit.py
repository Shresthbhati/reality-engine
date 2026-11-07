"""Accept a second visit (milestone M4): the new capture of a room replaces the old room; the old room's door joins are kept.

Nothing here decides that two captures are the same room: the user does (``room_matching`` only proposes). Accepting:

* removes the old room's elements and keeps everything else (other rooms, observations / evidence, history in the version store);
* re-makes each door join the old room had through the new room's matching door (same opening size). The first re-join places the new
  room; further ones are checked against that placement and kept only when they agree (<= 0.5 m, facing each other);
* reports every join that could not be kept (``lost_connections``) instead of keeping a pointer to a removed door;
* records the decision and the change summary in ``world.metadata["room_revisits"]``.

It is all-or-nothing: the world changes only when every step succeeded.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from engine.pipeline.room_matching import change_set, room_signature
from engine.pipeline.rooms import (
    DOOR_HEIGHT_TOL_M, DOOR_WIDTH_TOL_M, RoomConnectionError, _frame, connect_rooms, room_prefix,
)

VERIFY_TOL_M = 0.5


class RoomRevisitError(ValueError):
    """The revisit cannot be accepted; the world is left unchanged."""


def _doors(world, prefix: str) -> List:
    return [e for e in world.entities.values() if e.type.value == "door" and room_prefix(e.id) == prefix]


def accept_revisit(world, old_prefix: str, new_prefix: str) -> Dict[str, object]:
    if old_prefix == new_prefix:
        raise RoomRevisitError("old and new are the same room")
    for p in (old_prefix, new_prefix):
        if room_signature(world, p) is None:
            raise RoomRevisitError(f"no room {p!r} in this world")
    work = copy.deepcopy(world)
    summary = change_set(work, old_prefix, new_prefix)["summary"]

    # the joins the old room had, with the opening size each door had
    pairs = []
    for d in _doors(work, old_prefix):
        other_id = (d.custom_properties or {}).get("connects_to")
        if other_id and other_id in work.entities:
            f = _frame(work, d)
            pairs.append({"old_door": d.id, "other_door": other_id, "width": f["width"], "height": f["height"]})

    # remove the old room (entities, their geometries, and anything pointing at them)
    gone = {e.id for e in work.entities.values() if room_prefix(e.id) == old_prefix}
    geo = {g for i in gone for g in work.entities[i].geometry_ids}
    for i in gone:
        del work.entities[i]
    for g in geo:
        work.geometries.pop(g, None)
    for e in work.entities.values():
        e.relationships = [r for r in (e.relationships or []) if r.target_id not in gone]
        cp = e.custom_properties or {}
        if cp.get("connects_to") in gone:
            cp.pop("connects_to")
        if cp.get("distinct_from"):
            cp["distinct_from"] = [x for x in cp["distinct_from"] if x not in gone]
            if not cp["distinct_from"]:
                cp.pop("distinct_from")
    conns = work.metadata.get("room_connections", [])
    work.metadata["room_connections"] = [c for c in conns if not set(c["doors"]) & gone]

    # re-make the joins through the new room's doors
    rejoined: List[dict] = []
    lost: List[dict] = []
    placed = False
    free = _doors(work, new_prefix)
    for pr in pairs:
        cands = [d for d in free if not (d.custom_properties or {}).get("connects_to")]
        fits = []
        for d in cands:
            f = _frame(work, d)
            if f["width"] is not None and pr["width"] is not None and abs(f["width"] - pr["width"]) > DOOR_WIDTH_TOL_M:
                continue
            if f["height"] is not None and pr["height"] is not None and abs(f["height"] - pr["height"]) > DOOR_HEIGHT_TOL_M:
                continue
            fits.append((abs((f["width"] or 0) - (pr["width"] or 0)) + abs((f["height"] or 0) - (pr["height"] or 0)), d))
        if not fits:
            lost.append({"old_door": pr["old_door"], "other_door": pr["other_door"], "reason": "no door of the new room matches the old door's size"})
            continue
        new_door = min(fits, key=lambda t: t[0])[1]
        other = work.entities[pr["other_door"]]
        if not placed:
            try:
                connect_rooms(work, other.id, new_door.id)
            except RoomConnectionError as exc:
                lost.append({"old_door": pr["old_door"], "other_door": pr["other_door"], "reason": str(exc)})
                continue
            placed = True
            rejoined.append({"old_door": pr["old_door"], "other_door": other.id, "new_door": new_door.id, "state": "placed"})
            continue
        fa, fb = _frame(work, other), _frame(work, new_door)
        gap = float(np.linalg.norm(fa["floor_centre"] - fb["floor_centre"]))
        facing = float(fa["outward"] @ fb["outward"]) < -0.9
        if gap > VERIFY_TOL_M or not facing:
            lost.append({"old_door": pr["old_door"], "other_door": other.id, "reason": f"inconsistent with the first re-join ({gap:.2f} m apart)"})
            continue
        for d, o in ((other, new_door), (new_door, other)):
            d.custom_properties["connects_to"] = o.id
            d.custom_properties["distinct_from"] = sorted(set(d.custom_properties.get("distinct_from") or ()) | {o.id})
        rejoined.append({"old_door": pr["old_door"], "other_door": other.id, "new_door": new_door.id, "state": "verified", "gap_m": round(gap, 3)})

    room = next((e for e in work.entities.values() if e.type.value == "room" and room_prefix(e.id) == new_prefix), None)
    if room is not None:
        room.custom_properties["revisit_of"] = old_prefix
    record = {"old_room": old_prefix, "new_room": new_prefix, "state": "accepted", "entities_removed": len(gone),
              "rejoined": rejoined, "lost_connections": lost, "change_summary": summary}
    work.metadata.setdefault("room_revisits", []).append(record)

    world.entities, world.geometries, world.metadata = work.entities, work.geometries, work.metadata
    return {k: record[k] for k in ("entities_removed", "rejoined", "lost_connections", "change_summary")}
