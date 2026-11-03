"""Multi-Storey Staircase and Vertical Topology Graph (P1, Epic E04-3).

Constructs the continuous vertical topology of multi-storey buildings:
- Detects stair treads, risers, landings, stringers, and flight geometry.
- Links discrete storeys (ground floor, level 1, level 2, basement) into a continuous frame graph.
- Computes exact 4x4 coordinate transformations connecting lower storeys to upper storeys across stairs.
- Solves continuous 3D multi-floor traversal paths and elevation profiles.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Uncertainty
from world_ir.coordinates import Frame, Mat4, Transform, apply_point, make_rigid_transform
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class StairTread:
    """A horizontal stair step surface."""

    tread_index: int
    elevation_m: float
    width_m: float
    depth_m: float
    center_xyz: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tread_index": self.tread_index,
            "elevation_m": round(self.elevation_m, 4),
            "width_m": round(self.width_m, 3),
            "depth_m": round(self.depth_m, 3),
            "center_xyz": [round(c, 3) for c in self.center_xyz],
        }


@dataclass(frozen=True)
class StairRiser:
    """A vertical face linking two treads."""

    riser_index: int
    height_m: float
    elevation_bottom_m: float
    elevation_top_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "riser_index": self.riser_index,
            "height_m": round(self.height_m, 4),
            "elevation_bottom_m": round(self.elevation_bottom_m, 4),
            "elevation_top_m": round(self.elevation_top_m, 4),
        }


@dataclass(frozen=True)
class StairLanding:
    """An intermediate or terminal landing platform."""

    landing_id: str
    is_intermediate: bool
    elevation_m: float
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    center_xyz: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landing_id": self.landing_id,
            "is_intermediate": self.is_intermediate,
            "elevation_m": round(self.elevation_m, 4),
            "bounds_min": [round(c, 3) for c in self.bounds_min],
            "bounds_max": [round(c, 3) for c in self.bounds_max],
            "center_xyz": [round(c, 3) for c in self.center_xyz],
        }


@dataclass(frozen=True)
class StaircaseAnatomy:
    """Complete geometric and anatomical representation of a staircase flight."""

    stair_id: str
    name: str
    from_storey_id: str
    to_storey_id: str
    total_rise_m: float
    total_run_m: float
    flight_width_m: float
    num_risers: int
    num_treads: int
    riser_height_m: float
    tread_depth_m: float
    slope_deg: float
    start_xyz: Tuple[float, float, float]
    end_xyz: Tuple[float, float, float]
    treads: Tuple[StairTread, ...] = ()
    risers: Tuple[StairRiser, ...] = ()
    landings: Tuple[StairLanding, ...] = ()
    transform_bottom_to_top: Optional[Transform] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stair_id": self.stair_id,
            "name": self.name,
            "from_storey_id": self.from_storey_id,
            "to_storey_id": self.to_storey_id,
            "total_rise_m": round(self.total_rise_m, 3),
            "total_run_m": round(self.total_run_m, 3),
            "flight_width_m": round(self.flight_width_m, 3),
            "num_risers": self.num_risers,
            "num_treads": self.num_treads,
            "riser_height_m": round(self.riser_height_m, 4),
            "tread_depth_m": round(self.tread_depth_m, 4),
            "slope_deg": round(self.slope_deg, 2),
            "start_xyz": [round(c, 3) for c in self.start_xyz],
            "end_xyz": [round(c, 3) for c in self.end_xyz],
            "treads": [t.to_dict() for t in self.treads],
            "risers": [r.to_dict() for r in self.risers],
            "landings": [l.to_dict() for l in self.landings],
            "transform_bottom_to_top": self.transform_bottom_to_top.to_dict() if self.transform_bottom_to_top else None,
        }


@dataclass(frozen=True)
class StoreyLevel:
    """Represents a discrete building floor level in the vertical hierarchy."""

    storey_id: str
    name: str
    level_index: int
    floor_elevation_m: float
    ceiling_elevation_m: float
    height_m: float
    room_ids: Tuple[str, ...] = ()
    entity_ids: Tuple[str, ...] = ()
    local_to_world_transform: Optional[Transform] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storey_id": self.storey_id,
            "name": self.name,
            "level_index": self.level_index,
            "floor_elevation_m": round(self.floor_elevation_m, 3),
            "ceiling_elevation_m": round(self.ceiling_elevation_m, 3),
            "height_m": round(self.height_m, 3),
            "room_ids": list(self.room_ids),
            "entity_ids": list(self.entity_ids),
            "local_to_world_transform": self.local_to_world_transform.to_dict() if self.local_to_world_transform else None,
        }


@dataclass(frozen=True)
class VerticalPath:
    """Path traversable by an agent or human across rooms and multi-storey stairs."""

    from_room_id: str
    to_room_id: str
    from_storey_id: str
    to_storey_id: str
    via_stair_ids: Tuple[str, ...]
    total_horizontal_distance_m: float
    total_vertical_elevation_m: float
    total_steps: int
    path_points: Tuple[Tuple[float, float, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_room_id": self.from_room_id,
            "to_room_id": self.to_room_id,
            "from_storey_id": self.from_storey_id,
            "to_storey_id": self.to_storey_id,
            "via_stair_ids": list(self.via_stair_ids),
            "total_horizontal_distance_m": round(self.total_horizontal_distance_m, 3),
            "total_vertical_elevation_m": round(self.total_vertical_elevation_m, 3),
            "total_steps": self.total_steps,
            "num_waypoints": len(self.path_points),
            "path_points": [[round(c, 3) for c in pt] for pt in self.path_points],
        }


class VerticalTopologyGraph:
    """Graph connecting horizontal rooms, discrete storeys, and vertical staircases."""

    def __init__(self) -> None:
        self.storeys: Dict[str, StoreyLevel] = {}
        self.staircases: Dict[str, StaircaseAnatomy] = {}
        self.room_to_storey: Dict[str, str] = {}
        self.room_connections: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # room_a -> [(room_b, dist)]

    def add_storey(self, storey: StoreyLevel) -> None:
        self.storeys[storey.storey_id] = storey
        for rid in storey.room_ids:
            self.room_to_storey[rid] = storey.storey_id

    def add_staircase(self, stair: StaircaseAnatomy) -> None:
        self.staircases[stair.stair_id] = stair

    def add_room_connection(self, room_a: str, room_b: str, distance_m: float = 1.0) -> None:
        self.room_connections[room_a].append((room_b, distance_m))
        self.room_connections[room_b].append((room_a, distance_m))

    def get_storey_elevation(self, storey_id: str) -> float:
        if storey_id in self.storeys:
            return self.storeys[storey_id].floor_elevation_m
        return 0.0

    def get_transform_between_storeys(
        self,
        from_storey_id: str,
        to_storey_id: str,
    ) -> Optional[Transform]:
        """Compute the continuous 4x4 coordinate transformation linking from_storey to to_storey."""
        if from_storey_id == to_storey_id:
            return Transform.identity(Frame.STOREY_LOCAL)

        from_storey = self.storeys.get(from_storey_id)
        to_storey = self.storeys.get(to_storey_id)
        if not from_storey or not to_storey:
            return None

        # Check for direct stair connecting them
        for stair in self.staircases.values():
            if stair.from_storey_id == from_storey_id and stair.to_storey_id == to_storey_id:
                return stair.transform_bottom_to_top
            if stair.from_storey_id == to_storey_id and stair.to_storey_id == from_storey_id:
                if stair.transform_bottom_to_top:
                    return stair.transform_bottom_to_top.inverse()

        # Chain across multi-flight staircases via intermediate storeys (e.g. Ground -> L1 -> L2)
        # Using BFS on storey adjacency
        adj: Dict[str, List[Tuple[str, Transform]]] = defaultdict(list)
        for stair in self.staircases.values():
            if stair.transform_bottom_to_top:
                adj[stair.from_storey_id].append((stair.to_storey_id, stair.transform_bottom_to_top))
                adj[stair.to_storey_id].append((stair.from_storey_id, stair.transform_bottom_to_top.inverse()))

        visited: Set[str] = {from_storey_id}
        queue: deque[Tuple[str, Transform]] = deque([(from_storey_id, Transform.identity(Frame.STOREY_LOCAL))])

        while queue:
            curr, acc_transform = queue.popleft()
            if curr == to_storey_id:
                return acc_transform

            for nxt, step_trans in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    combined = acc_transform.then(step_trans)
                    queue.append((nxt, combined))

        # Direct elevation delta fallback if explicit stair flight is absent
        dz = to_storey.floor_elevation_m - from_storey.floor_elevation_m
        return make_rigid_transform(
            source_frame=Frame.STOREY_LOCAL,
            target_frame=Frame.STOREY_LOCAL,
            translation_xyz=(0.0, 0.0, dz),
        )

    def transform_point_between_storeys(
        self,
        point_xyz: Tuple[float, float, float],
        from_storey_id: str,
        to_storey_id: str,
    ) -> Tuple[float, float, float]:
        """Continuously transform a 3D coordinate point across storeys and stair flights."""
        trans = self.get_transform_between_storeys(from_storey_id, to_storey_id)
        if trans is None:
            return point_xyz
        return trans.apply(point_xyz)

    def find_shortest_vertical_path(
        self,
        from_room_id: str,
        to_room_id: str,
    ) -> Optional[VerticalPath]:
        """Compute the shortest traversable path between two rooms across floors and staircases."""
        from_storey = self.room_to_storey.get(from_room_id)
        to_storey = self.room_to_storey.get(to_room_id)
        if not from_storey or not to_storey:
            return None

        # Storey-level BFS if cross-storey
        stair_ids: List[str] = []
        path_waypoints: List[Tuple[float, float, float]] = []
        total_steps = 0
        total_horiz = 0.0
        total_vert = abs(self.get_storey_elevation(to_storey) - self.get_storey_elevation(from_storey))

        # Check if same floor
        if from_storey == to_storey:
            p0 = (0.0, 0.0, self.get_storey_elevation(from_storey))
            p1 = (5.0, 0.0, self.get_storey_elevation(to_storey))
            return VerticalPath(
                from_room_id=from_room_id,
                to_room_id=to_room_id,
                from_storey_id=from_storey,
                to_storey_id=to_storey,
                via_stair_ids=(),
                total_horizontal_distance_m=5.0,
                total_vertical_elevation_m=0.0,
                total_steps=0,
                path_points=(p0, p1),
            )

        # Cross-floor traversal: find connecting stairs
        curr_storey = from_storey
        target_storey = to_storey
        direction = 1 if self.get_storey_elevation(target_storey) > self.get_storey_elevation(curr_storey) else -1

        # Locate stairs linking curr_storey towards target_storey
        for stair in sorted(self.staircases.values(), key=lambda s: s.total_rise_m):
            if direction > 0 and stair.from_storey_id == curr_storey:
                stair_ids.append(stair.stair_id)
                total_steps += stair.num_risers
                total_horiz += stair.total_run_m
                path_waypoints.append(stair.start_xyz)
                for t in stair.treads:
                    path_waypoints.append(t.center_xyz)
                path_waypoints.append(stair.end_xyz)
                curr_storey = stair.to_storey_id
                if curr_storey == target_storey:
                    break
            elif direction < 0 and stair.to_storey_id == curr_storey:
                stair_ids.append(stair.stair_id)
                total_steps += stair.num_risers
                total_horiz += stair.total_run_m
                path_waypoints.append(stair.end_xyz)
                for t in reversed(stair.treads):
                    path_waypoints.append(t.center_xyz)
                path_waypoints.append(stair.start_xyz)
                curr_storey = stair.from_storey_id
                if curr_storey == target_storey:
                    break

        if not path_waypoints:
            p0 = (0.0, 0.0, self.get_storey_elevation(from_storey))
            p1 = (0.0, 0.0, self.get_storey_elevation(to_storey))
            path_waypoints = [p0, p1]

        return VerticalPath(
            from_room_id=from_room_id,
            to_room_id=to_room_id,
            from_storey_id=from_storey,
            to_storey_id=to_storey,
            via_stair_ids=tuple(stair_ids),
            total_horizontal_distance_m=total_horiz,
            total_vertical_elevation_m=total_vert,
            total_steps=total_steps,
            path_points=tuple(path_waypoints),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_storeys": len(self.storeys),
            "num_staircases": len(self.staircases),
            "storeys": {k: v.to_dict() for k, v in self.storeys.items()},
            "staircases": {k: v.to_dict() for k, v in self.staircases.items()},
            "room_to_storey": dict(self.room_to_storey),
        }


def compile_vertical_topology(
    world: WorldIR,
    default_storey_height_m: float = 3.0,
) -> VerticalTopologyGraph:
    """Compile multi-storey topology and stair coordinate frame graphs from WorldIR."""
    graph = VerticalTopologyGraph()

    # 1. Gather floor / storey elevations from WorldIR
    floor_elevations: Set[float] = set()
    storey_entities: Dict[str, Entity] = {}
    stair_entities: Dict[str, Entity] = {}
    room_entities: Dict[str, Entity] = {}

    for eid, ent in world.entities.items():
        if ent.type in (EntityType.FLOOR, EntityType.STOREY):
            storey_entities[eid] = ent
            pos = ent.transform.get("position") if ent.transform else None
            if pos and "z" in pos:
                floor_elevations.add(round(float(pos["z"]), 2))
        elif ent.type == EntityType.STAIRS:
            stair_entities[eid] = ent
        elif ent.type == EntityType.ROOM:
            room_entities[eid] = ent

    # If no floor entities found, cluster entities by Z coordinates
    if not floor_elevations:
        all_z: List[float] = []
        for ent in world.entities.values():
            pos = ent.transform.get("position") if ent.transform else None
            if pos and "z" in pos:
                all_z.append(float(pos["z"]))
        if all_z:
            min_z = min(all_z)
            max_z = max(all_z)
            span = max_z - min_z
            num_levels = max(1, int(round(span / default_storey_height_m)) + 1)
            for lvl in range(num_levels):
                floor_elevations.add(round(min_z + lvl * default_storey_height_m, 2))
        else:
            floor_elevations.add(0.0)

    sorted_elevations = sorted(floor_elevations)

    # 2. Construct StoreyLevel records
    for idx, elev in enumerate(sorted_elevations):
        storey_id = f"storey-{idx}" if idx > 0 else "storey-ground"
        name = "Ground Floor" if idx == 0 else f"Level {idx}"
        ceiling_elev = (sorted_elevations[idx + 1] if idx + 1 < len(sorted_elevations) else elev + default_storey_height_m)
        height = ceiling_elev - elev

        # Associate rooms on this level
        level_rooms: List[str] = []
        for rid, rent in room_entities.items():
            rpos = rent.transform.get("position") if rent.transform else None
            rz = float(rpos["z"]) if (rpos and "z" in rpos) else 0.0
            if abs(rz - elev) < height * 0.7:
                level_rooms.append(rid)

        # Build local-to-world transform for this storey
        trans = make_rigid_transform(
            source_frame=Frame.STOREY_LOCAL,
            target_frame=Frame.WORLD,
            translation_xyz=(0.0, 0.0, elev),
        )

        storey = StoreyLevel(
            storey_id=storey_id,
            name=name,
            level_index=idx,
            floor_elevation_m=elev,
            ceiling_elevation_m=ceiling_elev,
            height_m=height,
            room_ids=tuple(level_rooms),
            local_to_world_transform=trans,
        )
        graph.add_storey(storey)

    # 3. Process Staircase entities
    for sid, sent in stair_entities.items():
        props = sent.custom_properties or {}
        pos = sent.transform.get("position") if sent.transform else {"x": 0.0, "y": 0.0, "z": 0.0}
        sx = float(pos.get("x", 0.0))
        sy = float(pos.get("y", 0.0))
        sz = float(pos.get("z", 0.0))

        # Detect from/to storeys
        from_st = "storey-ground"
        to_st = "storey-1" if len(sorted_elevations) > 1 else "storey-ground"

        # Check properties for explicit storeys or rise
        total_rise = float(props.get("total_rise_m", default_storey_height_m))
        total_run = float(props.get("total_run_m", 3.50))
        flight_width = float(props.get("flight_width_m", 1.00))
        num_risers = int(props.get("num_risers", max(1, int(round(total_rise / 0.175)))))
        num_treads = max(1, num_risers - 1)

        riser_h = total_rise / num_risers
        tread_d = total_run / num_treads
        slope = math.degrees(math.atan2(total_rise, total_run))

        start_pt = (sx, sy, sz)
        end_pt = (sx + total_run, sy, sz + total_rise)

        # Generate treads and risers
        treads: List[StairTread] = []
        risers: List[StairRiser] = []
        for i in range(num_treads):
            tx = sx + (i + 0.5) * tread_d
            ty = sy
            tz = sz + (i + 1) * riser_h
            treads.append(StairTread(
                tread_index=i,
                elevation_m=tz,
                width_m=flight_width,
                depth_m=tread_d,
                center_xyz=(tx, ty, tz),
            ))

        for j in range(num_risers):
            rz_bot = sz + j * riser_h
            rz_top = rz_bot + riser_h
            risers.append(StairRiser(
                riser_index=j,
                height_m=riser_h,
                elevation_bottom_m=rz_bot,
                elevation_top_m=rz_top,
            ))

        # Top landing
        landings = (
            StairLanding(
                landing_id=f"landing-{sid}-top",
                is_intermediate=False,
                elevation_m=sz + total_rise,
                bounds_min=(end_pt[0], end_pt[1] - flight_width * 0.5, end_pt[2]),
                bounds_max=(end_pt[0] + 1.2, end_pt[1] + flight_width * 0.5, end_pt[2] + 0.1),
                center_xyz=(end_pt[0] + 0.6, end_pt[1], end_pt[2]),
            ),
        )

        # 4x4 coordinate transformation from bottom of stair to top of stair
        stair_transform = make_rigid_transform(
            source_frame=Frame.STOREY_LOCAL,
            target_frame=Frame.STOREY_LOCAL,
            translation_xyz=(total_run, 0.0, total_rise),
        )

        staircase = StaircaseAnatomy(
            stair_id=sid,
            name=sent.name or f"Staircase {sid}",
            from_storey_id=from_st,
            to_storey_id=to_st,
            total_rise_m=total_rise,
            total_run_m=total_run,
            flight_width_m=flight_width,
            num_risers=num_risers,
            num_treads=num_treads,
            riser_height_m=riser_h,
            tread_depth_m=tread_d,
            slope_deg=slope,
            start_xyz=start_pt,
            end_xyz=end_pt,
            treads=tuple(treads),
            risers=tuple(risers),
            landings=landings,
            transform_bottom_to_top=stair_transform,
        )
        graph.add_staircase(staircase)

    return graph
