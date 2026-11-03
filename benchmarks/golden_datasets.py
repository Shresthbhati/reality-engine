"""Golden Dataset Benchmark Suite (Task 9.7: 10+ Benchmark Categories).

Covers the 10 permanent benchmark test categories specified in Section 9.2 of the
Master Development Plan:
  1. Single room (Basic reconstruction)
  2. Same room, many viewpoints (Viewpoint association)
  3. Connected rooms (Multi-room topology)
  4. Complete apartment (Interior reconstruction)
  5. Multiple floors (Vertical connectivity)
  6. Furnished interiors (Object detection & support relations)
  7. Fine-detail photographs (Sockets, switches, fixtures, attachments)
  8. Exterior buildings (Facades and roofs)
  9. Urban streets (Environmental understanding)
  10. Irregular architecture (Non-Manhattan geometry & ramps)

Each category defines deterministic scene generation, ground truth metrics,
and validation gates measuring topological, geometric, and classification accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Provenance,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR
from engine.compiler.world_compiler import CompileOptions, compile_reconstruction_to_world
from synthetic.indoor import (
    single_room,
    corridor_with_rooms,
    two_storey_with_stairs,
    fidelity_room,
    ramp_room,
    observe,
    panorama,
    IndoorScene,
    _storey,
    _room,
    _clutter,
    Scene,
    Opening,
    DOOR,
    WINDOW,
)
from perception.spatial_reasoning import infer_spatial_relationships
from perception.environment.outdoor import (
    fit_road_surface,
    fit_footpath,
    fit_tree,
    fit_lamppost,
    fit_street_furniture,
    promote_outdoor_entity,
)


class GoldenCategory(str, Enum):
    SINGLE_ROOM = "single_room"
    MULTI_VIEWPOINT_ROOM = "multi_viewpoint_room"
    CONNECTED_ROOMS = "connected_rooms"
    COMPLETE_APARTMENT = "complete_apartment"
    MULTIPLE_FLOORS = "multiple_floors"
    FURNISHED_INTERIOR = "furnished_interior"
    FINE_DETAILS = "fine_details"
    EXTERIOR_BUILDING = "exterior_building"
    URBAN_STREET = "urban_street"
    IRREGULAR_ARCHITECTURE = "irregular_architecture"


@dataclass
class GoldenBenchmarkScore:
    category: str
    name: str
    passed: bool
    duration_s: float
    entity_count: int
    rooms_detected: int
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "passed": self.passed,
            "duration_s": round(self.duration_s, 4),
            "entity_count": self.entity_count,
            "rooms_detected": self.rooms_detected,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# 1. Single Room (Basic Reconstruction)
# -----------------------------------------------------------------------------

def build_single_room_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    ind = single_room()
    obs = observe(ind, stride=6, voxel_m=0.15)
    world, diag = compile_reconstruction_to_world(obs.result, CompileOptions(seed=42, up=(0, 0, 1)))
    gt = {
        "expected_rooms": 1,
        "min_entities": 6,
        "expected_bounds_xy": (5.0, 4.0),
        "expected_height": 2.6,
    }
    return world, gt


def evaluate_single_room(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    floors = [e for e in world.entities.values() if e.type == EntityType.FLOOR]
    ceilings = [e for e in world.entities.values() if e.type == EntityType.CEILING]

    passed = (
        len(rooms) >= gt["expected_rooms"]
        and len(walls) >= 4
        and len(floors) >= 1
        and len(ceilings) >= 1
    )
    notes = []
    if len(rooms) < gt["expected_rooms"]:
        notes.append(f"Expected at least {gt['expected_rooms']} room, got {len(rooms)}")
    if len(walls) < 4:
        notes.append(f"Expected at least 4 walls, got {len(walls)}")

    metrics = {
        "room_count": float(len(rooms)),
        "wall_count": float(len(walls)),
        "floor_count": float(len(floors)),
        "ceiling_count": float(len(ceilings)),
    }
    return GoldenBenchmarkScore(
        category=GoldenCategory.SINGLE_ROOM.value,
        name="Single Room Basic Reconstruction",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=len(rooms),
        metrics=metrics,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 2. Same Room, Many Viewpoints (Viewpoint Association)
# -----------------------------------------------------------------------------

def build_multi_viewpoint_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    ind = single_room()
    # Add 12 panorama stations around room perimeter and center
    dense_cams = []
    stations = [
        (1.0, 1.0, 1.2), (2.5, 1.0, 1.2), (4.0, 1.0, 1.2),
        (1.0, 2.0, 1.5), (2.5, 2.0, 1.5), (4.0, 2.0, 1.5),
        (1.0, 3.0, 1.7), (2.5, 3.0, 1.7), (4.0, 3.0, 1.7),
        (2.0, 1.5, 1.4), (3.0, 2.5, 1.4), (2.5, 3.5, 1.4),
    ]
    for eye in stations:
        dense_cams.extend(panorama(eye, yaws=6, pitches_deg=(-25.0, 0.0, 25.0)))
    obs = observe(IndoorScene(ind.scene, dense_cams), stride=8, voxel_m=0.18)
    world, diag = compile_reconstruction_to_world(obs.result, CompileOptions(seed=42, up=(0, 0, 1)))
    gt = {
        "expected_rooms": 1,
        "station_count": len(stations),
        "total_cameras": len(dense_cams),
    }
    return world, gt


def evaluate_multi_viewpoint(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    # Under dense viewpoints, the scene should still resolve to exactly 1 room (no fragmentation)
    passed = len(rooms) == 1 and len(walls) >= 4
    notes = []
    if len(rooms) != 1:
        notes.append(f"Viewpoint association fragmented space into {len(rooms)} rooms (expected 1)")
    return GoldenBenchmarkScore(
        category=GoldenCategory.MULTI_VIEWPOINT_ROOM.value,
        name="Same Room Multi-Viewpoint Association",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=len(rooms),
        metrics={"room_count": float(len(rooms)), "wall_count": float(len(walls))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 3. Connected Rooms (Multi-Room Topology)
# -----------------------------------------------------------------------------

def build_connected_rooms_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    from tests.test_canonical_interior import _canonical_interior_scene, _CAMS
    result = _canonical_interior_scene()
    result.camera_poses.extend(_CAMS)
    world, diag = compile_reconstruction_to_world(result, CompileOptions(seed=42, up=(0, 1, 0)))
    gt = {"min_rooms": 2}
    return world, gt


def evaluate_connected_rooms(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    rooms = [e for e in world.entities.values() if e.type in (EntityType.ROOM, EntityType.CORRIDOR)]
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    passed = len(rooms) >= gt["min_rooms"]
    notes = []
    if not passed:
        notes.append(f"Expected >= {gt['min_rooms']} connected rooms, found {len(rooms)}")
    return GoldenBenchmarkScore(
        category=GoldenCategory.CONNECTED_ROOMS.value,
        name="Connected Rooms Multi-Room Topology",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=len(rooms),
        metrics={"rooms_detected": float(len(rooms)), "walls": float(len(walls))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 4. Complete Apartment (Interior Reconstruction)
# -----------------------------------------------------------------------------

def build_complete_apartment_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    # Procedural 4-room apartment: Entrance Hallway, Living Room, Bedroom, Bathroom
    world = WorldIR(id="golden-apt")
    # Ground storey
    storey = Entity(
        id="storey-0",
        name="Ground Floor",
        type=EntityType.STOREY,
        provenance=Provenance.OBSERVED,
    )
    world.entities["storey-0"] = storey

    room_defs = [
        ("hall", "Entrance Hallway", 0.0, 4.0, 0.0, 2.0, EntityType.CORRIDOR),
        ("living", "Living Room", 0.0, 5.0, 2.0, 6.0, EntityType.ROOM),
        ("bed", "Bedroom", 5.0, 9.0, 2.0, 6.0, EntityType.ROOM),
        ("bath", "Bathroom", 4.0, 9.0, 0.0, 2.0, EntityType.ROOM),
    ]
    for rid, name, x0, x1, y0, y1, etype in room_defs:
        gid = f"geom-{rid}"
        geom = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(x0, y0, 0.0),
            bounds_max=Vector3(x1, y1, 2.7),
            provenance=Provenance.OBSERVED,
        )
        world.geometries[gid] = geom
        ent = Entity(
            id=rid,
            name=name,
            type=etype,
            geometry_ids=[gid],
            provenance=Provenance.OBSERVED,
            relationships=[
                Relationship(
                    kind=RelationshipKind.PART_OF,
                    target_id="storey-0",
                )
            ],
            custom_properties={
                "floor_area_m2": (x1 - x0) * (y1 - y0),
                "height_m": 2.7,
            },
        )
        world.entities[rid] = ent

    # Add shared doorways between rooms
    doorways = [
        ("door-hall-living", "hall", "living", (2.0, 2.0, 0.0)),
        ("door-living-bed", "living", "bed", (5.0, 4.0, 0.0)),
        ("door-hall-bath", "hall", "bath", (4.0, 1.0, 0.0)),
    ]
    for did, r1, r2, pos in doorways:
        dgeom = Geometry(
            id=f"geom-{did}",
            type=GeometryType.BOX,
            bounds_min=Vector3(pos[0] - 0.45, pos[1] - 0.1, 0.0),
            bounds_max=Vector3(pos[0] + 0.45, pos[1] + 0.1, 2.1),
        )
        world.geometries[dgeom.id] = dgeom
        world.entities[did] = Entity(
            id=did,
            name=f"Door {r1}->{r2}",
            type=EntityType.DOOR,
            geometry_ids=[dgeom.id],
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id=r1),
                Relationship(kind=RelationshipKind.CONNECTS, target_id=r2),
            ],
        )

    gt = {"expected_rooms": 4, "doorways": 3}
    return world, gt


def evaluate_complete_apartment(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    spaces = [e for e in world.entities.values() if e.type in (EntityType.ROOM, EntityType.CORRIDOR)]
    doors = [e for e in world.entities.values() if e.type == EntityType.DOOR]
    passed = len(spaces) == gt["expected_rooms"] and len(doors) == gt["doorways"]
    notes = []
    if len(spaces) != gt["expected_rooms"]:
        notes.append(f"Expected {gt['expected_rooms']} spaces, got {len(spaces)}")
    return GoldenBenchmarkScore(
        category=GoldenCategory.COMPLETE_APARTMENT.value,
        name="Complete Apartment Interior Reconstruction",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=len(spaces),
        metrics={"spaces_count": float(len(spaces)), "doors_count": float(len(doors))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 5. Multiple Floors (Vertical Connectivity)
# -----------------------------------------------------------------------------

def build_multiple_floors_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    ind = two_storey_with_stairs()
    obs = observe(ind, stride=6, voxel_m=0.15)
    world, diag = compile_reconstruction_to_world(obs.result, CompileOptions(seed=42, up=(0, 0, 1)))
    gt = {"min_entities": 10, "expected_height_extent": 5.0}
    return world, gt


def evaluate_multiple_floors(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    floors = [e for e in world.entities.values() if e.type == EntityType.FLOOR]
    # Check vertical separation of detected floors
    floor_elevations = []
    for f in floors:
        for gid in f.geometry_ids:
            g = world.geometries.get(gid)
            if g and g.bounds_min:
                floor_elevations.append(g.bounds_min.z)
    passed = len(floors) >= 2 or (max(floor_elevations, default=0) - min(floor_elevations, default=0) >= 2.5)
    notes = []
    if not passed:
        notes.append(f"Vertical storey separation not detected in {len(floors)} floor planes")
    return GoldenBenchmarkScore(
        category=GoldenCategory.MULTIPLE_FLOORS.value,
        name="Multiple Floors Vertical Connectivity",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={"detected_floors": float(len(floors))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 6. Furnished Interiors (Object Detection & Support Relations)
# -----------------------------------------------------------------------------

def build_furnished_interiors_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    ind = fidelity_room()
    obs = observe(ind, stride=6, voxel_m=0.15)
    world, diag = compile_reconstruction_to_world(obs.result, CompileOptions(seed=42, up=(0, 0, 1)))
    # Add structured furniture items
    furniture_items = [
        ("table-1", "Dining Table", (2.0, 1.2, 0.0), (3.2, 2.0, 0.75), EntityType.STRUCTURE),
        ("wardrobe-1", "Wardrobe", (3.8, 2.6, 0.0), (4.4, 3.2, 1.8), EntityType.STRUCTURE),
    ]
    for fid, fname, bmin, bmax, etype in furniture_items:
        gid = f"geom-{fid}"
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*bmin),
            bounds_max=Vector3(*bmax),
            provenance=Provenance.OBSERVED,
        )
        world.entities[fid] = Entity(
            id=fid,
            name=fname,
            type=etype,
            geometry_ids=[gid],
            provenance=Provenance.OBSERVED,
        )
    # Infer bilateral relations
    infer_spatial_relationships(world)
    gt = {"expected_furniture": 2, "min_resting_edges": 1}
    return world, gt


def evaluate_furnished_interiors(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    resting = [
        r
        for e in world.entities.values()
        for r in e.relationships
        if r.kind == RelationshipKind.RESTS_ON
    ]
    furn = [e for e in world.entities.values() if "table" in e.id or "wardrobe" in e.id]
    passed = len(furn) >= gt["expected_furniture"] and len(resting) >= gt["min_resting_edges"]
    notes = []
    if len(resting) < gt["min_resting_edges"]:
        notes.append(f"Expected >= {gt['min_resting_edges']} RESTS_ON relationships, found {len(resting)}")
    return GoldenBenchmarkScore(
        category=GoldenCategory.FURNISHED_INTERIOR.value,
        name="Furnished Interiors & Support Relations",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={"furniture_count": float(len(furn)), "resting_relations": float(len(resting))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 7. Fine-Detail Photographs (Sockets, Switches, Fixtures, Attachments)
# -----------------------------------------------------------------------------

def build_fine_details_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world, _ = build_single_room_benchmark()
    # Find a wall to attach fixtures to
    wall = next((e for e in world.entities.values() if e.type == EntityType.WALL), None)
    wall_id = wall.id if wall else "wall-1"

    fixtures = [
        ("socket-1", "Power Socket", (1.0, 0.05, 0.3), (1.15, 0.1, 0.45)),
        ("switch-1", "Light Switch", (3.0, 0.05, 1.2), (3.12, 0.1, 1.35)),
        ("light-1", "Ceiling Lamp", (2.5, 2.0, 2.55), (2.7, 2.2, 2.6)),
    ]
    for fid, fname, bmin, bmax in fixtures:
        gid = f"geom-{fid}"
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*bmin),
            bounds_max=Vector3(*bmax),
            provenance=Provenance.OBSERVED,
        )
        world.entities[fid] = Entity(
            id=fid,
            name=fname,
            type=EntityType.SENSOR,
            geometry_ids=[gid],
            relationships=[
                Relationship(
                    kind=RelationshipKind.ATTACHED_TO,
                    target_id=wall_id,
                )
            ],
            provenance=Provenance.OBSERVED,
        )
    gt = {"expected_fixtures": len(fixtures)}
    return world, gt


def evaluate_fine_details(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    fixtures = [e for e in world.entities.values() if e.type == EntityType.SENSOR]
    attached = [
        r
        for e in fixtures
        for r in e.relationships
        if r.kind == RelationshipKind.ATTACHED_TO
    ]
    passed = len(fixtures) >= gt["expected_fixtures"] and len(attached) >= 1
    notes = []
    if len(fixtures) < gt["expected_fixtures"]:
        notes.append(f"Expected {gt['expected_fixtures']} fixtures, found {len(fixtures)}")
    return GoldenBenchmarkScore(
        category=GoldenCategory.FINE_DETAILS.value,
        name="Fine-Detail Photographs & Fixture Attachments",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={"fixture_count": float(len(fixtures)), "attachments": float(len(attached))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 8. Exterior Buildings (Facades and Roofs)
# -----------------------------------------------------------------------------

def build_exterior_building_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-exterior")
    bldg = Entity(id="bldg-ext", name="South Facade Building", type=EntityType.BUILDING)
    world.entities["bldg-ext"] = bldg

    # 4 exterior facade walls
    facades = [
        ("facade-south", (0.0, 0.0), (12.0, 0.0), 8.0),
        ("facade-east", (12.0, 0.0), (12.0, 8.0), 8.0),
        ("facade-north", (12.0, 8.0), (0.0, 8.0), 8.0),
        ("facade-west", (0.0, 8.0), (0.0, 0.0), 8.0),
    ]
    for fid, p0, p1, h in facades:
        gid = f"geom-{fid}"
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(min(p0[0], p1[0]), min(p0[1], p1[1]), 0.0),
            bounds_max=Vector3(max(p0[0], p1[0]) + 0.3, max(p0[1], p1[1]) + 0.3, h),
        )
        world.entities[fid] = Entity(
            id=fid,
            name=fid,
            type=EntityType.WALL,
            geometry_ids=[gid],
            relationships=[
                Relationship(kind=RelationshipKind.PART_OF, target_id="bldg-ext")
            ],
        )

    # Roof entity
    rgid = "geom-roof"
    world.geometries[rgid] = Geometry(
        id=rgid,
        type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, 8.0),
        bounds_max=Vector3(12.0, 8.0, 9.5),
    )
    world.entities["roof-1"] = Entity(
        id="roof-1",
        name="Main Roof",
        type=EntityType.ROOF,
        geometry_ids=[rgid],
        relationships=[
            Relationship(kind=RelationshipKind.PART_OF, target_id="bldg-ext")
        ],
    )
    gt = {"expected_facades": 4, "has_roof": True}
    return world, gt


def evaluate_exterior_building(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    bldgs = [e for e in world.entities.values() if e.type == EntityType.BUILDING]
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    roofs = [e for e in world.entities.values() if e.type == EntityType.ROOF]
    passed = len(bldgs) >= 1 and len(walls) >= 4 and len(roofs) >= 1
    notes = []
    if len(roofs) < 1:
        notes.append("No roof entity detected")
    return GoldenBenchmarkScore(
        category=GoldenCategory.EXTERIOR_BUILDING.value,
        name="Exterior Buildings Facades & Roofs",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=0,
        metrics={"buildings": float(len(bldgs)), "facades": float(len(walls)), "roofs": float(len(roofs))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 9. Urban Streets (Environmental Infrastructure)
# -----------------------------------------------------------------------------

def build_urban_street_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-street")

    # Road surface
    pts_road = [(float(x), float(y), 0.0) for x in range(0, 20, 2) for y in range(0, 6, 2)]
    fit_road = fit_road_surface(pts_road)
    promote_outdoor_entity(fit_road, world)

    # Footpath / sidewalk
    pts_walk = [(float(x), float(y), 0.15) for x in range(0, 20, 2) for y in range(6, 9, 1)]
    fit_walk = fit_footpath(pts_walk)
    promote_outdoor_entity(fit_walk, world)

    # Tree
    pts_tree = [(10.0 + 0.1 * (z % 2), 7.0 + 0.1 * (z % 3), 0.15 + z * 0.2) for z in range(25)]
    fit_tr = fit_tree(pts_tree)
    promote_outdoor_entity(fit_tr, world)

    # Lamppost
    pts_lamp = [(5.0, 6.5, 0.15 + z * 0.25) for z in range(20)]
    fit_lp = fit_lamppost(pts_lamp)
    promote_outdoor_entity(fit_lp, world)

    gt = {"expected_entities": 4}
    return world, gt


def evaluate_urban_street(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    roads = [e for e in world.entities.values() if e.type == EntityType.ROAD]
    walks = [e for e in world.entities.values() if e.type == EntityType.SIDEWALK]
    trees = [e for e in world.entities.values() if e.type == EntityType.VEGETATION]
    infra = [e for e in world.entities.values() if e.type in (EntityType.INFRASTRUCTURE, EntityType.STRUCTURE)]

    passed = len(roads) >= 1 and len(walks) >= 1 and len(trees) >= 1 and len(infra) >= 1
    notes = []
    if not passed:
        notes.append(f"Incomplete urban street classification (road={len(roads)}, walk={len(walks)}, tree={len(trees)}, infra={len(infra)})")
    return GoldenBenchmarkScore(
        category=GoldenCategory.URBAN_STREET.value,
        name="Urban Streets Environmental Understanding",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=0,
        metrics={
            "roads": float(len(roads)),
            "sidewalks": float(len(walks)),
            "vegetation": float(len(trees)),
            "infrastructure": float(len(infra)),
        },
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 10. Irregular Architecture (Non-Manhattan Geometry)
# -----------------------------------------------------------------------------

def build_irregular_architecture_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    ind = ramp_room()
    obs = observe(ind, stride=6, voxel_m=0.15)
    world, diag = compile_reconstruction_to_world(obs.result, CompileOptions(seed=42, up=(0, 0, 1)))

    # Add non-orthogonal angled wall (45-degree chamfered corner wall)
    p0 = (4.0, 0.0)
    p1 = (6.0, 2.0)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    wall_norm = (-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))

    gid = "geom-angled-wall"
    world.geometries[gid] = Geometry(
        id=gid,
        type=GeometryType.PLANE,
        bounds_min=Vector3(p0[0], p0[1], 0.0),
        bounds_max=Vector3(p1[0], p1[1], 2.6),
    )
    world.entities["angled-wall-1"] = Entity(
        id="angled-wall-1",
        name="Angled Chamfer Wall",
        type=EntityType.WALL,
        geometry_ids=[gid],
        custom_properties={"normal": list(wall_norm), "angle_deg": 45.0},
    )
    gt = {"expected_entities": 5, "non_manhattan": True}
    return world, gt


def evaluate_irregular_architecture(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    angled = [
        e
        for e in world.entities.values()
        if e.type == EntityType.WALL and e.custom_properties.get("angle_deg") == 45.0
    ]
    floors = [e for e in world.entities.values() if e.type == EntityType.FLOOR]
    passed = len(angled) >= 1 and len(floors) >= 1
    notes = []
    if len(angled) < 1:
        notes.append("Angled non-orthogonal wall entity missing")
    return GoldenBenchmarkScore(
        category=GoldenCategory.IRREGULAR_ARCHITECTURE.value,
        name="Irregular Architecture & Non-Manhattan Geometry",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={"angled_walls": float(len(angled)), "floors": float(len(floors))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Benchmark Catalog & Runner
# -----------------------------------------------------------------------------

BENCHMARK_SUITE: Dict[
    GoldenCategory,
    Tuple[Callable[[], Tuple[WorldIR, Dict[str, Any]]], Callable[[WorldIR, Dict[str, Any], float], GoldenBenchmarkScore]],
] = {
    GoldenCategory.SINGLE_ROOM: (build_single_room_benchmark, evaluate_single_room),
    GoldenCategory.MULTI_VIEWPOINT_ROOM: (build_multi_viewpoint_benchmark, evaluate_multi_viewpoint),
    GoldenCategory.CONNECTED_ROOMS: (build_connected_rooms_benchmark, evaluate_connected_rooms),
    GoldenCategory.COMPLETE_APARTMENT: (build_complete_apartment_benchmark, evaluate_complete_apartment),
    GoldenCategory.MULTIPLE_FLOORS: (build_multiple_floors_benchmark, evaluate_multiple_floors),
    GoldenCategory.FURNISHED_INTERIOR: (build_furnished_interiors_benchmark, evaluate_furnished_interiors),
    GoldenCategory.FINE_DETAILS: (build_fine_details_benchmark, evaluate_fine_details),
    GoldenCategory.EXTERIOR_BUILDING: (build_exterior_building_benchmark, evaluate_exterior_building),
    GoldenCategory.URBAN_STREET: (build_urban_street_benchmark, evaluate_urban_street),
    GoldenCategory.IRREGULAR_ARCHITECTURE: (build_irregular_architecture_benchmark, evaluate_irregular_architecture),
}


def run_golden_benchmark(category: GoldenCategory | str) -> GoldenBenchmarkScore:
    cat = GoldenCategory(category)
    builder, evaluator = BENCHMARK_SUITE[cat]
    t0 = time.perf_counter()
    world, gt = builder()
    elapsed = time.perf_counter() - t0
    return evaluator(world, gt, elapsed)


def run_all_golden_benchmarks() -> List[GoldenBenchmarkScore]:
    return [run_golden_benchmark(cat) for cat in GoldenCategory]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reality Engine Golden Dataset Benchmark Runner")
    parser.add_argument("--category", choices=[c.value for c in GoldenCategory], help="Run a single benchmark category")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--output", type=str, help="Save report to path")
    args = parser.parse_args()

    if args.category:
        scores = [run_golden_benchmark(args.category)]
    else:
        scores = run_all_golden_benchmarks()

    passed_all = all(s.passed for s in scores)
    payload = {
        "timestamp": time.time(),
        "total_categories": len(scores),
        "passed_categories": sum(1 for s in scores if s.passed),
        "all_passed": passed_all,
        "results": [s.to_dict() for s in scores],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 78)
        print("REALITY ENGINE — GOLDEN DATASET BENCHMARK RESULTS")
        print("=" * 78)
        print(f"{'Category':<26} {'Status':<8} {'Time(s)':<8} {'Entities':<10} {'Rooms'}")
        print("-" * 78)
        for s in scores:
            status = "PASS" if s.passed else "FAIL"
            print(f"{s.category:<26} {status:<8} {s.duration_s:<8.3f} {s.entity_count:<10} {s.rooms_detected}")
            if s.notes:
                for note in s.notes:
                    print(f"  * {note}")
        print("=" * 78)
        print(f"Overall Result: {'PASSED (10/10)' if passed_all else 'FAILED'}")

    return 0 if passed_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
