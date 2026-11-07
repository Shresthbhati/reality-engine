"""Join rooms that were captured in separate sessions into ONE apartment, through a shared door (milestone M4).

Each session is reconstructed in its own frame under its own id prefix (``boot1-...``, ``boot2-...``). Nothing measured how the rooms
sit relative to each other, so their placement is a claim that someone must make: the user says "this door of room 1 is that door of
room 2". Given that, the second room moves RIGIDLY (a turn about the vertical axis and a shift): its door lands on the first room's
door and the two rooms lie on opposite sides of it. No wall, opening or furniture of either room is re-measured or resized.

The connection is recorded (``world.metadata["room_connections"]`` and ``connects_to`` on both doors) with its state, so a later,
better alignment can replace it.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

DOOR_WIDTH_TOL_M = 0.3
DOOR_HEIGHT_TOL_M = 0.3
CONNECTION_STATE = "user_confirmed_doors_coincide"


class RoomConnectionError(ValueError):
    """The two doors cannot be the same opening; the world is left unchanged."""


def room_prefix(entity_id: str) -> str:
    """``boot2-plane-01`` -> ``boot2``: the session/room an entity belongs to (one reconstruction per prefix)."""
    return entity_id.split("-", 1)[0]


def _door(world, door_id: str):
    e = world.entities.get(door_id)
    if e is None or e.type.value != "door":
        raise RoomConnectionError(f"{door_id!r} is not a door of this world")
    return e


def _frame(world, door) -> Dict[str, object]:
    """Door centre on the floor (x, y, z), the wall's outward direction in plan (x, z), and the opening's size."""
    cp = door.custom_properties or {}
    wall = world.entities.get(cp.get("host_wall_id", ""))
    slab = (wall.custom_properties or {}).get("wall_slab") if wall is not None else None
    if not slab:
        raise RoomConnectionError(f"door {door.id!r} has no host wall with a solid slab: its orientation is unknown")
    g = world.geometries[door.geometry_ids[0]]
    dims = cp.get("dimensions") or {}
    return {
        "floor_centre": np.array([(g.bounds_min.x + g.bounds_max.x) / 2, float(slab["start_xyz_m"][1]), (g.bounds_min.z + g.bounds_max.z) / 2]),
        "outward": np.array([slab["outward"][0], slab["outward"][2]], float),
        "width": float(dims["width_m"]["value"]) if "width_m" in dims else None,
        "height": float(dims["height_m"]["value"]) if "height_m" in dims else None,
    }


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _pt(p, R, t) -> List[float]:
    return [float(v) for v in R @ np.asarray(p, float) + t]


def _move_room(world, prefix: str, R: np.ndarray, t: np.ndarray, theta: float) -> int:
    """Apply the rigid move to every entity (and its geometry) whose id carries ``prefix``; returns how many entities moved."""
    from world_ir import Vector3

    moved, seen = 0, set()
    for e in world.entities.values():
        if room_prefix(e.id) != prefix:
            continue
        moved += 1
        pos = (e.transform or {}).get("position")
        if pos:
            x, y, z = _pt([pos["x"], pos["y"], pos["z"]], R, t)
            e.transform["position"] = {"x": x, "y": y, "z": z}
        for gid in e.geometry_ids:
            g = world.geometries.get(gid)
            if g is None or gid in seen:
                continue
            seen.add(gid)
            lo, hi = g.bounds_min, g.bounds_max
            corners = [_pt([x, y, z], R, t) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
            g.bounds_min = Vector3(*[min(c[i] for c in corners) for i in range(3)])
            g.bounds_max = Vector3(*[max(c[i] for c in corners) for i in range(3)])
        cp = e.custom_properties or {}
        slab = cp.get("wall_slab")
        if slab:
            slab["start_xyz_m"], slab["end_xyz_m"] = _pt(slab["start_xyz_m"], R, t), _pt(slab["end_xyz_m"], R, t)
            for k in ("direction", "outward", "up"):
                slab[k] = _pt(slab[k], R, np.zeros(3))
        surf = cp.get("slab")
        if surf and "polygon_xyz_m" in surf:
            surf["polygon_xyz_m"] = [_pt(p, R, t) for p in surf["polygon_xyz_m"]]
            surf["up"] = _pt(surf["up"], R, np.zeros(3))
        basis = cp.get("footprint_basis")
        if basis and "e1" in basis and "footprint_polygon_m" in cp:
            e1n, e2n = R @ np.asarray(basis["e1"], float), R @ np.asarray(basis["e2"], float)
            cp["footprint_polygon_m"] = [[round(a + float(t @ e1n), 4), round(b + float(t @ e2n), 4)] for a, b in cp["footprint_polygon_m"]]
            basis["e1"], basis["e2"] = [float(v) for v in e1n], [float(v) for v in e2n]
        if isinstance(cp.get("yaw_rad"), (int, float)):
            cp["yaw_rad"] = float(cp["yaw_rad"]) - theta  # a direction (cos, sin) in (x, z) turned by Ry(theta)
    return moved


def connect_rooms(world, door_a_id: str, door_b_id: str, *, width_tol_m: float = DOOR_WIDTH_TOL_M,
                  height_tol_m: float = DOOR_HEIGHT_TOL_M) -> Dict[str, object]:
    """Move room B (the door_b's room) so its door coincides with door A and the rooms lie on opposite sides of it.

    Raises ``RoomConnectionError`` (world untouched) when both doors are in one room, either is already connected, or their widths /
    heights differ by more than the tolerance (then they are not the same opening). Returns the applied rotation / translation.
    """
    da, db = _door(world, door_a_id), _door(world, door_b_id)
    pa, pb = room_prefix(da.id), room_prefix(db.id)
    if pa == pb:
        raise RoomConnectionError("both doors belong to the same room")
    for d in (da, db):
        if (d.custom_properties or {}).get("connects_to"):
            raise RoomConnectionError(f"door {d.id!r} is already connected to {d.custom_properties['connects_to']!r}")
    fa, fb = _frame(world, da), _frame(world, db)
    if fa["width"] is not None and fb["width"] is not None and abs(fa["width"] - fb["width"]) > width_tol_m:
        raise RoomConnectionError(f"door widths differ ({fa['width']:.2f} m vs {fb['width']:.2f} m): not the same opening")
    if fa["height"] is not None and fb["height"] is not None and abs(fa["height"] - fb["height"]) > height_tol_m:
        raise RoomConnectionError(f"door heights differ ({fa['height']:.2f} m vs {fb['height']:.2f} m): not the same opening")
    phi_a, phi_b = math.atan2(fa["outward"][1], fa["outward"][0]), math.atan2(fb["outward"][1], fb["outward"][0])
    theta = phi_b - (phi_a + math.pi)  # Ry(theta) turns direction angle phi to phi - theta: B's outward must point opposite to A's
    theta = math.atan2(math.sin(theta), math.cos(theta))
    R = _ry(theta)
    t = fa["floor_centre"] - R @ fb["floor_centre"]
    moved = _move_room(world, pb, R, t, theta)
    for d, other in ((da, db), (db, da)):
        d.custom_properties["connects_to"] = other.id
        # one physical opening, one entity per room (each is the opening of ITS room's wall): they overlap on purpose, which the
        # world validator's duplicate guard would otherwise reject
        d.custom_properties["distinct_from"] = sorted(set(d.custom_properties.get("distinct_from") or ()) | {other.id})
    record = {"doors": [da.id, db.id], "rooms": [pa, pb], "rotation_deg": round(math.degrees(theta), 3),
              "translation_m": [round(float(v), 4) for v in t], "state": CONNECTION_STATE, "entities_moved": moved,
              "note": "placement assumes the two doors are the same opening; nothing else measured how the rooms sit"}
    world.metadata.setdefault("room_connections", []).append(record)
    return record
