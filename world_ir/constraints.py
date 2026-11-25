"""P111 architectural constraint checks for the ``compile_validate`` stage.

Returned as data (``ConstraintViolation``), never raised: an ``error`` violation makes the stage ``degraded``
(see ``violations_to_segment_fields``), a ``warning`` is only counted in the stage metrics.

Inventory (what already existed, so this module adds only the gaps):
  - ``validation._check_invariants`` invariant 7 blocks an opening with no surface, but only once the world has any
    surface and it accepts floors/ceilings as hosts; ``topology.revalidate_merged_topology`` (dangling_opening,
    opening_exceeds_host_wall) is a merge-time pass not run by ``compile_validate`` and only compares widths.
  - ``validation._check_cross_entity_consistency`` checks opening HEIGHT against the host wall (warning).
  - Nothing compared an opening's position along the wall, a wall base to its storey floor, or stair/storey links.
Added here: opening_without_host_wall, opening_outside_host_wall, wall_base_not_at_storey_floor,
stair_not_connecting_two_storeys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from shapely.geometry import LineString
from shapely.ops import polygonize

from world_ir import WorldIR
from world_ir.schema_v1 import EntityType, RelationshipKind
from world_ir.validation import _entity_aabb

_OPENINGS = (EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING)


@dataclass(frozen=True)
class ConstraintConfig:
    """Tolerances in metres. ``floor_tolerance_m`` matches perception.architecture.room_graph's floor-height band."""
    floor_tolerance_m: float = 0.05
    extent_tolerance_m: float = 0.05


@dataclass(frozen=True)
class ConstraintViolation:
    code: str
    entity_ids: Tuple[str, ...]
    message: str
    severity: str  # "error" | "warning"


def _host_id(world: WorldIR, opening) -> Optional[str]:
    cp = opening.custom_properties or {}
    host = cp.get("host_wall_id") or cp.get("surface_id")
    if host:
        return host
    for rel in opening.relationships:
        if rel.kind is RelationshipKind.ATTACHED_TO and rel.target_id in world.entities:
            return rel.target_id
    return None


def _long_axis_span(box) -> Tuple[int, float, float]:
    lo, hi = box
    axis = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1
    return axis, lo[axis], hi[axis]


def _wall_storey(world: WorldIR, wall_id: str) -> Optional[str]:
    sid = (world.entities[wall_id].custom_properties or {}).get("storey_id")
    if sid in world.entities:
        return sid
    rooms = {e.id for e in world.entities.values()
             if wall_id in ((e.custom_properties or {}).get("boundary_element_ids") or ())}
    for e in sorted(world.entities.values(), key=lambda x: x.id):
        if e.type is EntityType.STOREY and any(
                r.kind is RelationshipKind.CONTAINS and (r.target_id == wall_id or r.target_id in rooms)
                for r in e.relationships):
            return e.id
    return None


def _stair_storeys(world: WorldIR, stair_id: str) -> Set[str]:
    out = {r.target_id for r in world.entities[stair_id].relationships
           if r.target_id in world.entities and world.entities[r.target_id].type is EntityType.STOREY}
    for e in world.entities.values():
        if e.type is EntityType.STOREY and any(r.target_id == stair_id for r in e.relationships):
            out.add(e.id)
    return out


def _wall_segment_2d(world: WorldIR, wall) -> Optional[LineString]:
    cp = wall.custom_properties or {}
    slab = cp.get("wall_slab") or {}
    s, e = slab.get("start_xyz_m"), slab.get("end_xyz_m")
    if isinstance(s, (list, tuple)) and isinstance(e, (list, tuple)) and len(s) >= 2 and len(e) >= 2:
        p1, p2 = (float(s[0]), float(s[1])), (float(e[0]), float(e[1]))
        if p1 != p2:
            return LineString([p1, p2])
    box = _entity_aabb(world, wall)
    if box is None:
        return None
    lo, hi = box
    dx, dy = hi[0] - lo[0], hi[1] - lo[1]
    if dx >= dy:
        ym = (lo[1] + hi[1]) / 2.0
        return LineString([(lo[0], ym), (hi[0], ym)]) if hi[0] > lo[0] else None
    xm = (lo[0] + hi[0]) / 2.0
    return LineString([(xm, lo[1]), (xm, hi[1])]) if hi[1] > lo[1] else None


def check_architectural_constraints(world: WorldIR, config: ConstraintConfig = ConstraintConfig()) -> List[ConstraintViolation]:
    out: List[ConstraintViolation] = []
    tol = config.extent_tolerance_m
    for eid in sorted(world.entities):
        e = world.entities[eid]
        if e.type in _OPENINGS:
            host_id = _host_id(world, e)
            host = world.entities.get(host_id) if host_id else None
            if host is None or host.type is not EntityType.WALL:
                out.append(ConstraintViolation(
                    "opening_without_host_wall", (eid,),
                    f"opening {eid} has no host wall (host={host_id!r})", "error"))
                continue
            ob, wb = _entity_aabb(world, e), _entity_aabb(world, host)
            if ob is None or wb is None:
                continue
            axis, w_lo, w_hi = _long_axis_span(wb)
            if ob[0][axis] < w_lo - tol or ob[1][axis] > w_hi + tol:
                out.append(ConstraintViolation(
                    "opening_outside_host_wall", (eid, host_id),
                    f"opening {eid} spans {ob[0][axis]:.2f}..{ob[1][axis]:.2f}m along host wall {host_id} "
                    f"({w_lo:.2f}..{w_hi:.2f}m)", "error"))
        elif e.type is EntityType.WALL:
            sid, box = _wall_storey(world, eid), _entity_aabb(world, e)
            floor = (world.entities[sid].custom_properties or {}).get("floor_height_m") if sid else None
            if box is None or not isinstance(floor, (int, float)):
                continue
            if abs(box[0][2] - float(floor)) > config.floor_tolerance_m:
                out.append(ConstraintViolation(
                    "wall_base_not_at_storey_floor", (eid, sid),
                    f"wall {eid} base z={box[0][2]:.2f}m is not at storey {sid} floor {float(floor):.2f}m "
                    f"(tolerance {config.floor_tolerance_m}m)", "error"))
    storeys = [i for i, x in world.entities.items() if x.type is EntityType.STOREY]
    for eid in sorted(i for i, x in world.entities.items() if x.type is EntityType.STAIRS):
        linked = _stair_storeys(world, eid)
        if len(linked) < 2:
            # a single-storey world has nothing to connect to: flag it, but do not degrade the job
            out.append(ConstraintViolation(
                "stair_not_connecting_two_storeys", (eid,),
                f"stair {eid} is linked to {len(linked)} storey(s); a stair must connect two",
                "error" if len(storeys) >= 2 else "warning"))

    # Contradictory relationships & containment cycles (NetworkX)
    contains_graph = nx.DiGraph()
    for eid in sorted(world.entities):
        e = world.entities[eid]
        for rel in e.relationships:
            if rel.target_id == eid:
                out.append(ConstraintViolation(
                    "contradictory_relationship", (eid,),
                    f"entity {eid} has self-referential relationship {rel.kind.value}", "error"))
            elif rel.target_id not in world.entities:
                out.append(ConstraintViolation(
                    "broken_entity_reference", (eid,),
                    f"entity {eid} references missing target {rel.target_id}", "warning"))
            elif rel.kind is RelationshipKind.CONTAINS:
                contains_graph.add_edge(eid, rel.target_id)
    for cycle in nx.simple_cycles(contains_graph):
        out.append(ConstraintViolation(
            "contradictory_relationship", tuple(sorted(cycle)),
            f"containment cycle detected among {sorted(cycle)}", "error"))

    # Non-Manhattan / irregular room enclosure check (Shapely)
    for eid in sorted(i for i, x in world.entities.items() if x.type is EntityType.ROOM):
        room = world.entities[eid]
        b_ids = list((room.custom_properties or {}).get("boundary_element_ids") or ())
        if len(b_ids) >= 2:
            segs = []
            for bid in b_ids:
                w = world.entities.get(bid)
                if w is not None:
                    seg = _wall_segment_2d(world, w)
                    if seg is not None:
                        segs.append(seg)
            if len(segs) >= 2:
                polys = list(polygonize(segs))
                if not polys:
                    out.append(ConstraintViolation(
                        "room_enclosure_incomplete", (eid,),
                        f"room {eid} boundary walls ({len(segs)} segments) do not form a closed polygon",
                        "warning"))

    # Wall near-miss connectivity gaps
    wall_segs: List[Tuple[str, LineString]] = []
    for eid in sorted(i for i, x in world.entities.items() if x.type is EntityType.WALL):
        seg = _wall_segment_2d(world, world.entities[eid])
        if seg is not None:
            wall_segs.append((eid, seg))
    if len(wall_segs) >= 2:
        for i, (wid, seg) in enumerate(wall_segs):
            min_d = min(seg.distance(other) for j, (_, other) in enumerate(wall_segs) if j != i)
            if tol < min_d <= 0.5:
                out.append(ConstraintViolation(
                    "wall_connectivity_gap", (wid,),
                    f"wall {wid} has a {min_d:.2f}m gap to its nearest neighbor wall",
                    "warning"))

    return out


def violations_to_segment_fields(violations: List[ConstraintViolation]) -> Tuple[List[str], Dict[str, float]]:
    """Stage fields: only errors become stage warnings (=> ``degraded``); both severities are counted in metrics."""
    errors = [v for v in violations if v.severity == "error"]
    warnings = [f"constraint {v.code}: {v.message}" for v in errors]
    metrics = {"constraint_errors": float(len(errors)),
               "constraint_warnings": float(len(violations) - len(errors))}
    return warnings, metrics
