"""D3: reconcile the dimensions of a room against each other and say so when they disagree.

Every check compares two numbers the pipeline produced by DIFFERENT routes; a disagreement is evidence of a defect, so it
is reported as a warning with both values -- never corrected, never hidden. Nothing here measures anything new.

* ``area_vs_extent``: for a rectangular footprint, floor area must equal width x length (from the polygon edges AND from
  the solved layout). Non-rectangular rooms (L, U, notched) have no such identity and are reported ``not_applicable``,
  not compared against their bounding box.
* ``net_wall_area``: net wall area = gross (length x height) - openings, per wall and summed; an opening must lie inside
  its host wall (width <= length, sill + height <= height).
* ``dimension_states``: a derived quantity (area) may not claim a stronger D1 state than the dimensions it is derived
  from (floor width / length).
"""

from __future__ import annotations

import math
from typing import List

from world_ir import EntityType

AREA_REL_TOL = 0.03
AREA_ABS_TOL_M2 = 0.1
LAYOUT_AREA_REL_TOL = 0.05
RIGHT_ANGLE_TOL_DEG = 5.0
FIT_TOL_M = 0.02
_STATE_RANK = {"ESTIMATED": 0, "RECONSTRUCTED": 1, "MEASURED": 2, "VERIFIED": 3}


def _rectangle_sides(poly) -> tuple | None:
    pts = [tuple(map(float, p[:2])) for p in poly]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) != 4:
        return None
    edges = [(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]) for i in range(4)]
    lens = [math.hypot(*e) for e in edges]
    if min(lens) < 1e-6:
        return None
    for i in range(4):
        a, b = edges[i], edges[(i + 1) % 4]
        cos = (a[0] * b[0] + a[1] * b[1]) / (lens[i] * lens[(i + 1) % 4])
        if abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 90.0) > RIGHT_ANGLE_TOL_DEG:
            return None
    return (lens[0] + lens[2]) / 2.0, (lens[1] + lens[3]) / 2.0


def _state(ent, key):
    d = (ent.custom_properties.get("dimensions") or {}).get(key)
    return d.get("state") if isinstance(d, dict) else None


def _check_room(world, room) -> dict:
    cp = room.custom_properties
    checks: List[dict] = []
    warnings: List[str] = []

    def add(name, ok, detail, **values):
        checks.append({"check": name, "ok": ok, "detail": detail, **values})
        if ok is False:
            warnings.append(f"{room.id}: {name}: {detail}")

    area = cp.get("floor_area_m2")
    sides = _rectangle_sides(cp["footprint_polygon_m"]) if cp.get("footprint_polygon_m") else None
    cons = cp.get("dimensional_consistency") or {}
    if area is None or sides is None:
        add("area_vs_extent", None, "footprint is not a rectangle: no width x length identity applies (not compared to a bounding box)"
            if area is not None else "no floor area recorded")
    else:
        w, l = sides
        tol = max(AREA_ABS_TOL_M2, AREA_REL_TOL * area)
        ok = abs(area - w * l) <= tol
        add("area_vs_extent", ok, f"floor area {area:.3f} m2 vs footprint {w:.3f} x {l:.3f} = {w * l:.3f} m2 (tol {tol:.3f})",
            area_m2=area, width_m=round(w, 4), length_m=round(l, 4))
        if cons.get("length_m") and cons.get("width_m"):
            lw = cons["length_m"] * cons["width_m"]
            ok = abs(area - lw) <= max(AREA_ABS_TOL_M2, LAYOUT_AREA_REL_TOL * area)
            add("area_vs_layout_extent", ok,
                f"floor area {area:.3f} m2 vs solved layout {cons['width_m']:.3f} x {cons['length_m']:.3f} = {lw:.3f} m2",
                area_m2=area, layout_area_m2=round(lw, 4))

    gross = openings = net = 0.0
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL
             and e.custom_properties.get("wall_slab") and e.custom_properties.get("room_id") == room.id]
    for wall in walls:
        s = wall.custom_properties["wall_slab"]
        L, H = float(s["length_m"]), float(s["height_m"])
        o_area = 0.0
        for o in s.get("openings") or []:
            ow, oh, off, sill = (float(o[k]) for k in ("width_m", "height_m", "offset_m", "sill_m"))
            o_area += ow * oh
            if off < -FIT_TOL_M or off + ow > L + FIT_TOL_M or sill < -FIT_TOL_M or sill + oh > H + FIT_TOL_M:
                add("opening_fits_host", False,
                    f"{o.get('kind', 'opening')} {o.get('entity_id')} ({ow:.2f} x {oh:.2f} m at {off:.2f}/{sill:.2f}) "
                    f"does not fit wall {wall.id} ({L:.2f} x {H:.2f} m)")
        wall_net = L * H - o_area
        if wall_net < -1e-6:
            add("net_wall_area", False, f"openings ({o_area:.3f} m2) exceed wall {wall.id} gross area ({L * H:.3f} m2)")
        gross, openings, net = gross + L * H, openings + o_area, net + wall_net
    if walls:
        ok = abs(net - (gross - openings)) < 1e-6 and net >= 0
        add("net_wall_area", ok, f"net wall area {net:.3f} m2 = gross {gross:.3f} - openings {openings:.3f}",
            gross_m2=round(gross, 4), openings_m2=round(openings, 4), net_m2=round(net, 4))
        cp["gross_wall_area_m2"], cp["opening_area_m2"], cp["net_wall_area_m2"] = (round(gross, 4), round(openings, 4), round(net, 4))

    s_area = _STATE_RANK.get(_state(room, "floor_area_m2"))
    s_in = [_STATE_RANK.get(_state(room, k)) for k in ("floor_dimension_u_m", "floor_dimension_v_m")]
    if s_area is not None and all(v is not None for v in s_in):
        ok = s_area <= min(s_in)
        add("dimension_states", ok, f"floor area is {_state(room, 'floor_area_m2')} but its width/length are "
            f"{_state(room, 'floor_dimension_u_m')}/{_state(room, 'floor_dimension_v_m')}")
    return {"room_id": room.id, "checks": checks, "warnings": warnings}


def reconcile_room_dimensions(world) -> dict:
    """Run the D3 checks for every room; store the result at ``world.metadata['dimension_consistency']``."""
    rooms = [_check_room(world, e) for e in world.entities.values()
             if e.type == EntityType.ROOM and "superseded_by" not in e.custom_properties
             and e.custom_properties.get("footprint_polygon_m")]
    report = {"rooms": rooms, "warnings": [w for r in rooms for w in r["warnings"]]}
    world.metadata["dimension_consistency"] = report
    return report
