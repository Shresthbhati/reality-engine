"""Per-room footprints for multi-room / multi-storey captures (plan S3 completeness + F7).

``room_semantics.refine_interior_semantics`` treats one capture as one space: it measures the anchor room from the whole
cloud and marks every other ROOM as a fragment of it. That is right for a single room split into plane fragments, and
wrong for a building: a two-storey capture has two real rooms, and before this pass only the first had a footprint -- so
only the first got solid walls, and the second was marked ``superseded_by`` the first.

This pass runs after room semantics and before the solid-wall promotion. It measures each room that still lacks a
measured footprint from its OWN storey floor:

* floor height comes from the room's STOREY (``floor_height_m``) -- not from the lowest floor in the whole cloud;
* the footprint is the largest connected component of the plan-view floor-band points at that height (the room's own
  floor; a wall line from another storey is neither filled nor large enough to win);
* height and ceiling layers come from the high points that fall inside that footprint;
* floor and wall colours are measured from the points of that room only (near the floor, and in the wall band just
  inside the outline);
* a room that cannot be measured stays unmeasured with a warning -- no outline is invented.

Nothing is invented: a room with too little floor evidence keeps no footprint, and therefore gets no solid walls.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from world_ir import EntityType

from perception.architecture.room_semantics import (
    _basis, _cell_size, _ceiling_peaks, _inside_distance, _plan, _Raster, _shoelace,
)

#: a room needs at least this many floor-band points to have a measurable outline
MIN_FLOOR_POINTS = 40
#: floor band half-width (m)
FLOOR_BAND_M = 0.12
#: points above the floor that belong to the room's upper structure
MAX_ROOM_HEIGHT_M = 4.5


def _storey_of(world, entity) -> Optional[object]:
    for rel in entity.relationships:
        if rel.kind.value == "part_of" and rel.target_id in world.entities:
            target = world.entities[rel.target_id]
            if target.type == EntityType.STOREY:
                return target
    return None


def _edge_heights(poly: np.ndarray, ab: np.ndarray, hz: np.ndarray, floor_h: float) -> List[Optional[float]]:
    """Measured wall-top height for each footprint edge, from points within 15 cm of that edge.

    ``None`` when an edge has no point support: the solid-wall promotion then refuses to extrude it, instead of using a
    plausible default height. The 98th percentile ignores a few stray returns above the ceiling."""
    out: List[Optional[float]] = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 1e-6:
            out.append(None)
            continue
        tangent = d / L
        normal = np.array([-tangent[1], tangent[0]])
        rel = ab - a
        dist = np.abs(rel @ normal)
        along = rel @ tangent
        sel = (dist < 0.15) & (along >= 0.0) & (along <= L) & (hz > floor_h + 0.2) & (hz < floor_h + MAX_ROOM_HEIGHT_M)
        if int(sel.sum()) < 8:
            out.append(None)
            continue
        h = float(np.percentile(hz[sel], 98)) - floor_h
        out.append(round(h, 4) if h > 0.5 else None)
    return out


def _colour(colors: np.ndarray, idx: np.ndarray) -> Optional[List[int]]:
    if not len(idx):
        return None
    c = colors[idx]
    c = c[c[:, 0] >= 0]
    return [int(v) for v in c.mean(axis=0)] if len(c) else None


def _measure_room(world, room, ab: np.ndarray, hz: np.ndarray, colors: np.ndarray, e1, e2, u,
                  structure_prefix: str) -> dict:
    """Measure one room from its storey floor. Returns a report entry; mutates the room's custom properties only on
    success."""
    cp = room.custom_properties
    storey = _storey_of(world, room)
    floor_h = cp.get("floor_height_m")
    if floor_h is None and storey is not None:
        floor_h = (storey.custom_properties or {}).get("floor_height_m")
    if floor_h is None:
        return {"room_id": room.id, "status": "skipped", "reason": "no storey floor height"}
    floor_h = float(floor_h)
    band = np.abs(hz - floor_h) < FLOOR_BAND_M
    F = ab[band]
    if len(F) < MIN_FLOOR_POINTS:
        return {"room_id": room.id, "status": "skipped",
                "reason": f"only {len(F)} floor-band points at h={floor_h:.3f} m (need {MIN_FLOOR_POINTS})"}
    cell = _cell_size(F)
    raster = _Raster(F, cell)
    occ = raster.occupancy(F, close_m=0.35)
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    if n < 2:
        return {"room_id": room.id, "status": "skipped", "reason": "floor occupancy empty"}
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"room_id": room.id, "status": "skipped", "reason": "floor outline not found"}
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, max(1.5, 0.15 / cell), True).reshape(-1, 2)
    if len(approx) < 3:
        return {"room_id": room.id, "status": "skipped", "reason": "floor outline degenerate"}
    mask = np.zeros_like(comp)
    cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
    poly = raster.to_metres(approx.astype(float))
    area = _shoelace(poly)
    if area <= 0.5:
        return {"room_id": room.id, "status": "skipped", "reason": f"measured floor area {area:.2f} m2 is not a room"}

    height = hz - floor_h
    inside = _inside_distance(raster, mask, ab) > 0.02
    cand = inside & (height > 1.2) & (height < MAX_ROOM_HEIGHT_M)
    peaks = _ceiling_peaks(height[cand]) if cand.sum() >= 60 else []
    room_height = max(peaks) if peaks else None
    if room_height is None:
        return {"room_id": room.id, "status": "skipped", "reason": "no ceiling measured above the floor"}

    def to_world(ab_poly, h):
        return [[round(float(v), 4) for v in (a * e1 + b * e2 + h * u)] for a, b in ab_poly]

    cp["floor_height_m"] = round(floor_h, 4)
    cp["footprint_polygon_m"] = [[round(float(a), 4), round(float(b), 4)] for a, b in poly]
    cp["footprint_polygon_xyz_m"] = to_world(poly, floor_h)
    cp["floor_area_m2"] = round(area, 4)
    cp["height_m"] = round(float(room_height), 4)
    cp["ceiling_height_max_m"] = round(float(room_height), 4)
    cp["footprint_edge_heights_m"] = _edge_heights(poly, ab, hz, floor_h)
    cp["footprint_method"] = (f"per-room plan-view occupancy of {int(band.sum())} floor-band points "
                              f"(|h-{floor_h:.3f}|<{FLOOR_BAND_M} m), cell {cell:.3f} m, storey floor")
    cp["footprint_basis"] = {"e1": e1.tolist(), "e2": e2.tolist(), "up": u.tolist()}
    floor_col = _colour(colors, np.flatnonzero(band & (inside | (_inside_distance(raster, mask, ab) > -0.10))))
    if floor_col:
        cp["floor_color_rgb"] = floor_col
    wall_zone = inside & (height > 0.2) & (height < room_height - 0.15) & \
        (_inside_distance(raster, mask, ab) < 0.25)
    wall_col = _colour(colors, np.flatnonzero(wall_zone))
    if wall_col:
        cp["wall_color_rgb"] = wall_col

    # ceiling layers: every measured peak becomes one layer over the same footprint (the contour is the room's)
    layers = []
    ceiling_entities = [e for e in world.entities.values() if e.type == EntityType.CEILING
                        and any(r.target_id == room.id and r.kind.value == "part_of" for r in e.relationships)]
    for k, peak in enumerate(peaks):
        layer_cand = inside & (np.abs(height - peak) < 0.08)
        coverage = float(layer_cand.sum()) / max(1, int(inside.sum()))
        if coverage < 0.15:
            continue
        layers.append({
            "entity_ids": [ceiling_entities[0].id] if len(ceiling_entities) == 1 else [],
            "role": "ceiling" if k == len(peaks) - 1 else "soffit",
            "height_above_floor_m": round(float(peak), 4),
            "plan_coverage_fraction": round(coverage, 4),
            "polygon_m": cp["footprint_polygon_m"],
            "polygon_xyz_m": to_world(poly, floor_h + peak),
            "color_rgb": _colour(colors, np.flatnonzero(layer_cand)) or wall_col,
        })
    if layers:
        cp["ceiling_layers"] = layers

    # this room is a real measured space, not a fragment of the anchor
    cp.pop("superseded_by", None)
    if "fragment_of_room" in room.semantic_labels:
        room.semantic_labels.remove("fragment_of_room")
    return {"room_id": room.id, "status": "measured", "area_m2": round(area, 4), "height_m": round(float(room_height), 4),
            "floor_height_m": round(floor_h, 4), "ceiling_layers": len(layers), "cell_m": round(cell, 4)}


def refine_room_footprints(world, result, up, *, structure_prefix: str = "struct") -> dict:
    """Measure every ROOM entity that has no footprint yet, from its own storey floor.

    Called after ``refine_interior_semantics`` and before the solid-wall promotion. Single-room worlds are untouched
    (their footprint is already measured by room semantics)."""
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    report: dict = {"rooms": [], "warnings": []}
    if len(rooms) < 2 or not result.points:
        return report
    basis = None
    for r in rooms:
        b = (r.custom_properties or {}).get("footprint_basis")
        if b:
            basis = b
            break
    if basis:
        e1 = np.asarray(basis["e1"], float)
        e2 = np.asarray(basis["e2"], float)
        u = np.asarray(basis["up"], float)
    else:
        e1, e2, u = _basis(up)
    P = np.array([p.position for p in result.points], float)
    pl = _plan(P, e1, e2, u)
    ab, hz = pl[:, :2], pl[:, 2]
    colors = np.array([(p.color if getattr(p, "color", None) is not None else (-1, -1, -1)) for p in result.points], float)
    for room in rooms:
        cp = room.custom_properties
        if cp.get("footprint_polygon_xyz_m") or cp.get("superseded_by"):
            # a superseded room has to be re-measured before it may claim its own footprint; a measured one is left alone
            if cp.get("footprint_polygon_xyz_m"):
                continue
        try:
            rep = _measure_room(world, room, ab, hz, colors, e1, e2, u, structure_prefix)
        except Exception as exc:  # noqa: BLE001 -- one room failing must not lose the others
            rep = {"room_id": room.id, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        report["rooms"].append(rep)
        if rep["status"] != "measured":
            report["warnings"].append(f"{room.id}: no per-room footprint ({rep['reason']}); no solid walls for it")
    return report
