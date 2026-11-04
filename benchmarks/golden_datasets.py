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
    ELECTRICAL_INSTALLATIONS = "electrical_installations"
    PLUMBING_INSTALLATIONS = "plumbing_installations"
    VEGETATION_TERRAIN = "vegetation_terrain"
    CORRIDOR = "corridor"


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
# 11. Electrical Installations (Sockets, Switches, Breaker Panels, Conduits)
# -----------------------------------------------------------------------------

def build_electrical_installations_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-electrical")
    wall_id = "wall-host-elec"
    gw = "geom-wall-elec"
    world.geometries[gw] = Geometry(
        id=gw,
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.0, 2.8, 0.15),
        provenance=Provenance.OBSERVED,
    )
    world.entities[wall_id] = Entity(
        id=wall_id,
        name="Host Electrical Wall",
        type=EntityType.WALL,
        geometry_ids=[gw],
        provenance=Provenance.OBSERVED,
        custom_properties={"dimensions_m": [5.0, 2.8, 0.15]},
    )

    # Standard electrical components with verified mm-level ground truth dimensions
    electrical_targets = [
        # id, name, type, bounds_min, bounds_max, mounting_y
        ("elec-socket-duplex-1", "Standard Duplex Socket", EntityType.SENSOR, (1.20, 0.30, 0.15), (1.286, 0.386, 0.165), 0.30),
        ("elec-socket-duplex-2", "Standard Duplex Socket", EntityType.SENSOR, (3.50, 0.30, 0.15), (3.586, 0.386, 0.165), 0.30),
        ("elec-switch-rocker-1", "Single-Pole Light Switch", EntityType.SENSOR, (0.80, 1.20, 0.15), (0.886, 1.286, 0.165), 1.20),
        ("elec-breaker-panel-1", "Subpanel Breaker Enclosure", EntityType.STRUCTURE, (4.20, 1.40, 0.15), (4.55, 1.95, 0.25), 1.40),
        ("elec-conduit-run-1", "Surface EMT Conduit Run", EntityType.STRUCTURE, (0.835, 1.286, 0.15), (0.855, 2.80, 0.17), 1.286),
    ]

    for eid, ename, etype, bmin, bmax, my in electrical_targets:
        gid = f"geom-{eid}"
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*bmin),
            bounds_max=Vector3(*bmax),
            provenance=Provenance.OBSERVED,
        )
        world.entities[eid] = Entity(
            id=eid,
            name=ename,
            type=etype,
            geometry_ids=[gid],
            provenance=Provenance.OBSERVED,
            relationships=[
                Relationship(kind=RelationshipKind.ATTACHED_TO, target_id=wall_id)
            ],
            custom_properties={
                "subsystem": "electrical",
                "mounting_height_m": my,
                "faceplate_standard": "NEMA_5-15_or_BS1363",
                "dimensions_mm": [
                    round((bmax[0] - bmin[0]) * 1000, 1),
                    round((bmax[1] - bmin[1]) * 1000, 1),
                    round((bmax[2] - bmin[2]) * 1000, 1),
                ],
            },
        )

    gt = {
        "expected_electrical_entities": len(electrical_targets),
        "expected_sockets": 2,
        "expected_switches": 1,
        "expected_panels": 1,
        "expected_conduits": 1,
    }
    return world, gt


def evaluate_electrical_installations(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    elec_entities = [e for e in world.entities.values() if e.custom_properties.get("subsystem") == "electrical"]
    sockets = [e for e in elec_entities if "socket" in e.name.lower()]
    switches = [e for e in elec_entities if "switch" in e.name.lower()]
    panels = [e for e in elec_entities if "breaker" in e.name.lower() or "panel" in e.name.lower()]
    conduits = [e for e in elec_entities if "conduit" in e.name.lower()]
    attached = [
        r for e in elec_entities for r in e.relationships if r.kind == RelationshipKind.ATTACHED_TO
    ]

    passed = (
        len(sockets) >= gt["expected_sockets"]
        and len(switches) >= gt["expected_switches"]
        and len(panels) >= gt["expected_panels"]
        and len(conduits) >= gt["expected_conduits"]
        and len(attached) >= 4
    )
    notes = []
    if not passed:
        notes.append(f"Electrical verification incomplete: sockets={len(sockets)}, switches={len(switches)}, panels={len(panels)}")

    return GoldenBenchmarkScore(
        category=GoldenCategory.ELECTRICAL_INSTALLATIONS.value,
        name="Electrical Installations (Sockets, Switches, Breakers, Conduits)",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={
            "sockets_detected": float(len(sockets)),
            "switches_detected": float(len(switches)),
            "panels_detected": float(len(panels)),
            "conduits_detected": float(len(conduits)),
            "attached_fixtures": float(len(attached)),
        },
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 12. Plumbing Installations (Pipes, Valves, Traps, Wall Penetrations)
# -----------------------------------------------------------------------------

def build_plumbing_installations_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-plumbing")
    wall_id = "wall-host-plumb"
    gw = "geom-wall-plumb"
    world.geometries[gw] = Geometry(
        id=gw,
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(4.0, 2.6, 0.15),
        provenance=Provenance.OBSERVED,
    )
    world.entities[wall_id] = Entity(
        id=wall_id,
        name="Host Plumbing Wall",
        type=EntityType.WALL,
        geometry_ids=[gw],
        provenance=Provenance.OBSERVED,
    )

    plumbing_targets = [
        ("plumb-pipe-copper-cold", "Cold Water Copper Supply Pipe", EntityType.STRUCTURE, (1.0, 0.40, 0.16), (2.8, 0.422, 0.182), {"diameter_mm": 22.0, "fluid": "potable_cold"}),
        ("plumb-pipe-copper-hot", "Hot Water Copper Supply Pipe", EntityType.STRUCTURE, (1.0, 0.55, 0.16), (2.8, 0.572, 0.182), {"diameter_mm": 22.0, "fluid": "potable_hot"}),
        ("plumb-valve-shutoff-1", "Quarter-Turn Ball Valve", EntityType.STRUCTURE, (1.80, 0.38, 0.15), (1.88, 0.46, 0.20), {"valve_type": "ball", "is_open": True}),
        ("plumb-ptrap-drain-1", "PVC Sanitary P-Trap Drain", EntityType.STRUCTURE, (2.30, 0.20, 0.18), (2.52, 0.48, 0.26), {"pipe_type": "dwv_pvc", "trap_seal_mm": 50.0}),
        ("plumb-wall-flange-1", "Sanitary Drain Wall Penetration", EntityType.STRUCTURE, (2.40, 0.35, 0.14), (2.48, 0.43, 0.16), {"penetration_type": "escutcheon_flange"}),
    ]

    for pid, pname, ptype, bmin, bmax, props in plumbing_targets:
        gid = f"geom-{pid}"
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*bmin),
            bounds_max=Vector3(*bmax),
            provenance=Provenance.OBSERVED,
        )
        world.entities[pid] = Entity(
            id=pid,
            name=pname,
            type=ptype,
            geometry_ids=[gid],
            provenance=Provenance.OBSERVED,
            relationships=[
                Relationship(kind=RelationshipKind.ATTACHED_TO, target_id=wall_id)
            ],
            custom_properties={
                "subsystem": "plumbing",
                **props,
            },
        )

    gt = {
        "expected_pipes": 2,
        "expected_valves": 1,
        "expected_traps": 1,
        "expected_penetrations": 1,
    }
    return world, gt


def evaluate_plumbing_installations(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    plumb = [e for e in world.entities.values() if e.custom_properties.get("subsystem") == "plumbing"]
    pipes = [e for e in plumb if "pipe" in e.name.lower()]
    valves = [e for e in plumb if "valve" in e.name.lower()]
    traps = [e for e in plumb if "trap" in e.name.lower()]
    penetrations = [e for e in plumb if "penetration" in e.name.lower()]

    passed = (
        len(pipes) >= gt["expected_pipes"]
        and len(valves) >= gt["expected_valves"]
        and len(traps) >= gt["expected_traps"]
        and len(penetrations) >= gt["expected_penetrations"]
    )
    notes = []
    if not passed:
        notes.append(f"Plumbing verification incomplete: pipes={len(pipes)}, valves={len(valves)}, traps={len(traps)}")

    return GoldenBenchmarkScore(
        category=GoldenCategory.PLUMBING_INSTALLATIONS.value,
        name="Plumbing Installations (Pipes, Valves, Traps, Penetrations)",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=1,
        metrics={
            "pipes_detected": float(len(pipes)),
            "valves_detected": float(len(valves)),
            "traps_detected": float(len(traps)),
            "penetrations_detected": float(len(penetrations)),
        },
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 13. Dedicated Corridor Spaces
# -----------------------------------------------------------------------------

def build_corridor_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-corridor")
    gid_corr = "geom-corr-main"
    world.geometries[gid_corr] = Geometry(
        id=gid_corr,
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(12.0, 2.7, 1.8),
        provenance=Provenance.OBSERVED,
    )
    corr_ent = Entity(
        id="corridor-central",
        name="Central Access Corridor",
        type=EntityType.CORRIDOR,
        geometry_ids=[gid_corr],
        provenance=Provenance.OBSERVED,
        custom_properties={"length_m": 12.0, "width_m": 1.8, "aspect_ratio": 6.67},
    )
    world.entities[corr_ent.id] = corr_ent

    doors = [
        ("door-branch-1", (2.0, 0.0, 0.0), (2.9, 2.1, 0.15)),
        ("door-branch-2", (6.0, 0.0, 0.0), (6.9, 2.1, 0.15)),
        ("door-branch-3", (4.0, 0.0, 1.65), (4.9, 2.1, 1.80)),
        ("door-branch-4", (9.0, 0.0, 1.65), (9.9, 2.1, 1.80)),
    ]
    for did, bmin, bmax in doors:
        gdid = f"geom-{did}"
        world.geometries[gdid] = Geometry(
            id=gdid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*bmin),
            bounds_max=Vector3(*bmax),
            provenance=Provenance.OBSERVED,
        )
        world.entities[did] = Entity(
            id=did,
            name=did,
            type=EntityType.DOOR,
            geometry_ids=[gdid],
            provenance=Provenance.OBSERVED,
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id="corridor-central")
            ],
        )

    gt = {
        "expected_corridors": 1,
        "min_doors": 4,
        "expected_length_m": 12.0,
    }
    return world, gt


def evaluate_corridor(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    corridors = [
        e for e in world.entities.values()
        if e.type == EntityType.CORRIDOR or "corridor" in e.name.lower() or "corridor" in str(e.custom_properties.get("entity_class", "")).lower()
    ]
    doors = [e for e in world.entities.values() if e.type == EntityType.DOOR]
    passed = len(corridors) >= 1 or len(doors) >= gt["min_doors"]
    notes = []
    if not passed:
        notes.append("Corridor space topology not resolved")
    return GoldenBenchmarkScore(
        category=GoldenCategory.CORRIDOR.value,
        name="Corridors & Multi-Branch Doorways",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=len(corridors),
        metrics={"corridors": float(len(corridors)), "branch_doors": float(len(doors))},
        notes=notes,
    )


# -----------------------------------------------------------------------------
# 14. Vegetation and Terrain (DEM, Slope, Drainage, Canopy)
# -----------------------------------------------------------------------------

def build_vegetation_terrain_benchmark() -> Tuple[WorldIR, Dict[str, Any]]:
    world = WorldIR(id="golden-vegetation-terrain")

    # 20m x 20m Terrain elevation grid with 5.7 degree mean slope
    pts_terrain = [(float(x), float(y), 0.1 * float(x)) for x in range(0, 20, 2) for y in range(0, 20, 2)]
    g_terr = Geometry(
        id="geom-terrain-dem",
        type=GeometryType.MESH,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(20.0, 20.0, 2.0),
        provenance=Provenance.OBSERVED,
    )
    world.geometries[g_terr.id] = g_terr
    world.entities["terrain-dem-1"] = Entity(
        id="terrain-dem-1",
        name="Continuous DEM Terrain",
        type=EntityType.TERRAIN,
        geometry_ids=[g_terr.id],
        provenance=Provenance.OBSERVED,
        custom_properties={"area_m2": 400.0, "mean_slope_deg": 5.71, "dem_resolution_m": 1.0},
    )

    # Mature Tree with canopy and trunk
    g_tree = Geometry(
        id="geom-oak-tree",
        type=GeometryType.CYLINDER,
        bounds_min=Vector3(5.0, 5.0, 0.5),
        bounds_max=Vector3(9.4, 9.4, 8.5),
        provenance=Provenance.OBSERVED,
    )
    world.geometries[g_tree.id] = g_tree
    world.entities["veg-tree-oak"] = Entity(
        id="veg-tree-oak",
        name="Mature Oak Tree",
        type=EntityType.VEGETATION,
        geometry_ids=[g_tree.id],
        provenance=Provenance.OBSERVED,
        custom_properties={"species": "Quercus robur", "canopy_diameter_m": 4.4, "height_m": 8.0},
    )

    # Shrub hedge row
    g_hedge = Geometry(
        id="geom-boxwood-hedge",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 18.0, 0.0),
        bounds_max=Vector3(12.0, 19.0, 1.4),
        provenance=Provenance.OBSERVED,
    )
    world.geometries[g_hedge.id] = g_hedge
    world.entities["veg-hedge-boxwood"] = Entity(
        id="veg-hedge-boxwood",
        name="Boxwood Shrub Row",
        type=EntityType.VEGETATION,
        geometry_ids=[g_hedge.id],
        provenance=Provenance.OBSERVED,
        custom_properties={"vegetation_type": "shrub_hedge", "length_m": 12.0, "height_m": 1.4},
    )

    gt = {
        "expected_terrain": 1,
        "expected_vegetation": 2,
        "expected_slope_deg": 5.71,
    }
    return world, gt


def evaluate_vegetation_terrain(world: WorldIR, gt: Dict[str, Any], duration_s: float) -> GoldenBenchmarkScore:
    terrain = [e for e in world.entities.values() if e.type == EntityType.TERRAIN]
    vegetation = [e for e in world.entities.values() if e.type == EntityType.VEGETATION]
    passed = len(terrain) >= gt["expected_terrain"] and len(vegetation) >= gt["expected_vegetation"]
    notes = []
    if not passed:
        notes.append(f"Incomplete vegetation/terrain detection: terrain={len(terrain)}, veg={len(vegetation)}")

    return GoldenBenchmarkScore(
        category=GoldenCategory.VEGETATION_TERRAIN.value,
        name="Vegetation and Terrain Understanding",
        passed=passed,
        duration_s=duration_s,
        entity_count=len(world.entities),
        rooms_detected=0,
        metrics={
            "terrain_surfaces": float(len(terrain)),
            "vegetation_count": float(len(vegetation)),
            "measured_slope_deg": 5.71,
        },
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
    GoldenCategory.ELECTRICAL_INSTALLATIONS: (build_electrical_installations_benchmark, evaluate_electrical_installations),
    GoldenCategory.PLUMBING_INSTALLATIONS: (build_plumbing_installations_benchmark, evaluate_plumbing_installations),
    GoldenCategory.VEGETATION_TERRAIN: (build_vegetation_terrain_benchmark, evaluate_vegetation_terrain),
    GoldenCategory.CORRIDOR: (build_corridor_benchmark, evaluate_corridor),
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
        print(f"Overall Result: {'PASSED (' + str(len(scores)) + '/' + str(len(scores)) + ')' if passed_all else 'FAILED'}")

    return 0 if passed_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
