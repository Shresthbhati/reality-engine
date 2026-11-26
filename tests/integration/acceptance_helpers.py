"""Shared helpers for the Real-Capture Product Acceptance System (RCPAS).

Builds deterministic WorldIRs with rooms, walls, openings and relationships so the
export, persistence, diff and failure-injection acceptance tests exercise the real
WorldIR/WorldStore/SDK contracts without depending on COLMAP or real photographs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR


def world_with_rooms(
    world_id: str = "world-acceptance",
    rooms: Optional[List[dict]] = None,
) -> WorldIR:
    """Build a WorldIR containing rectangular rooms with box geometry and
    adjacency relationships.

    Each room dict may carry: id, name, position (x,y,z), size (w,d,h),
    provenance, confidence, adjacency (list of target room ids).
    """
    world = WorldIR(id=world_id)
    rooms = rooms or [
        {"id": "room-main", "name": "Main Room", "position": (0.0, 0.0, 1.5),
         "size": (5.0, 4.0, 3.0)},
        {"id": "room-annex", "name": "Annex", "position": (6.0, 0.0, 1.5),
         "size": (3.0, 3.0, 3.0)},
    ]
    for r in rooms:
        pos = r.get("position", (0.0, 0.0, 1.5))
        size = r.get("size", (3.0, 3.0, 3.0))
        prov = Provenance(r.get("provenance", Provenance.RECONSTRUCTED.value))
        conf = float(r.get("confidence", 0.95))
        ent = Entity(
            id=r["id"],
            type=EntityType.ROOM,
            name=r.get("name", r["id"]),
            provenance=prov,
            confidence=conf,
        )
        ent.transform = {
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        }
        geom = Geometry(
            id=f"geom-{r['id']}",
            type=GeometryType.BOX,
            bounds_min=Vector3(-size[0] / 2, -size[1] / 2, 0.0),
            bounds_max=Vector3(size[0] / 2, size[1] / 2, size[2]),
            provenance=prov,
            confidence=conf,
        )
        ent.geometry_ids.append(geom.id)
        world.entities[ent.id] = ent
        world.geometries[geom.id] = geom
        for target in r.get("adjacency", []):
            ent.relationships.append(
                Relationship(
                    kind=RelationshipKind.ADJACENT_TO,
                    target_id=target,
                    provenance=prov,
                )
            )
    return world


def world_with_single_room(
    room_id: str = "room-only",
    world_id: str = "world-single",
) -> WorldIR:
    return world_with_rooms(
        world_id=world_id,
        rooms=[{"id": room_id, "name": "Only Room", "position": (0.0, 0.0, 1.5),
                "size": (4.0, 3.0, 3.0)}],
    )


def world_modified(
    base: WorldIR,
    world_id: str = "world-modified",
) -> WorldIR:
    """Return a copy of ``base`` with one room's size changed, one new room added,
    and one room removed -- enough to exercise WorldDiff ADD/REMOVE/MODIFY."""
    import copy

    world = copy.deepcopy(base)
    world.id = world_id
    # Remove room-annex if present
    if "room-annex" in world.entities:
        del world.entities["room-annex"]
        world.geometries.pop("geom-room-annex", None)
    # Modify room-main size
    if "room-main" in world.entities:
        ent = world.entities["room-main"]
        ent.transform = {
            "position": {"x": 1.0, "y": 1.0, "z": 1.5},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        }
        if "geom-room-main" in world.geometries:
            g = world.geometries["geom-room-main"]
            g.bounds_min = Vector3(-3.0, -2.0, 0.0)
            g.bounds_max = Vector3(3.0, 2.0, 4.0)
        ent.relationships = [
            r for r in ent.relationships if r.target_id != "room-annex"
        ]
    # Add a new room
    new = Entity(
        id="room-new", type=EntityType.ROOM, name="New Room",
        provenance=Provenance.RECONSTRUCTED, confidence=0.9,
    )
    new.transform = {
        "position": {"x": 10.0, "y": 0.0, "z": 1.5},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
    }
    gnew = Geometry(
        id="geom-room-new", type=GeometryType.BOX,
        bounds_min=Vector3(-2.0, -2.0, 0.0),
        bounds_max=Vector3(2.0, 2.0, 3.0),
        provenance=Provenance.RECONSTRUCTED, confidence=0.9,
    )
    new.geometry_ids.append(gnew.id)
    world.entities["room-new"] = new
    world.geometries["geom-room-new"] = gnew
    return world


@dataclass
class ExportCheck:
    format: str
    content: object
    report: object
    ok: bool
    reason: str = ""


def require_nonempty(content, format_name: str) -> str:
    """Return '' if content is non-empty, else a failure reason."""
    if content is None:
        return f"{format_name}: exporter returned None"
    if isinstance(content, (bytes, bytearray)) and len(content) == 0:
        return f"{format_name}: exported zero bytes"
    if isinstance(content, str) and len(content.strip()) == 0:
        return f"{format_name}: exported empty string"
    return ""


def world_metric_rooms(
    world_id: str = "world-metric",
) -> WorldIR:
    """Build a metric-scaled WorldIR with two rooms that have footprint polygons
    and two walls with wall_slabs, so the DXF / PDF floorplan exporters have
    measured geometry to emit.

    The world's scale is recorded as metric with a device-depth calibration basis,
    which is what the floorplan exporters require to emit metres rather than model
    units.
    """
    world = WorldIR(id=world_id)
    world.metadata["scale"] = {
        "state": "metric",
        "meters_per_unit": 1.0,
        "scale_basis": "device_depth",
        "calibration": {"status": "validated", "basis": "device_depth"},
    }

    # Room A: footprint polygon (CCW, metres), centred at origin.
    room_a = Entity(
        id="room-a",
        type=EntityType.ROOM,
        name="Room A",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    room_a.transform = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
    }
    room_a.custom_properties["footprint_polygon_m"] = [
        (0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0),
    ]
    geom_a = Geometry(
        id="geom-room-a", type=GeometryType.BOX,
        bounds_min=Vector3(-2.0, -1.5, 0.0),
        bounds_max=Vector3(2.0, 1.5, 3.0),
        provenance=Provenance.RECONSTRUCTED, confidence=0.95,
    )
    room_a.geometry_ids.append(geom_a.id)
    world.entities["room-a"] = room_a
    world.geometries["geom-room-a"] = geom_a

    # Room B: footprint polygon, adjacent to Room A.
    room_b = Entity(
        id="room-b",
        type=EntityType.ROOM,
        name="Room B",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
    )
    room_b.transform = {
        "position": {"x": 4.5, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
    }
    room_b.custom_properties["footprint_polygon_m"] = [
        (4.0, 0.0), (7.0, 0.0), (7.0, 3.0), (4.0, 3.0),
    ]
    geom_b = Geometry(
        id="geom-room-b", type=GeometryType.BOX,
        bounds_min=Vector3(4.0, -1.5, 0.0),
        bounds_max=Vector3(7.0, 1.5, 3.0),
        provenance=Provenance.RECONSTRUCTED, confidence=0.9,
    )
    room_b.geometry_ids.append(geom_b.id)
    world.entities["room-b"] = room_b
    world.geometries["geom-room-b"] = geom_b

    # Adjacency relationship between the rooms.
    room_a.relationships.append(
        Relationship(
            kind=RelationshipKind.ADJACENT_TO,
            target_id="room-b",
            provenance=Provenance.RECONSTRUCTED,
        )
    )

    # Wall between the rooms (shared edge at x=4).
    wall = Entity(
        id="wall-ab",
        type=EntityType.WALL,
        name="Wall A-B",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
    )
    wall.transform = {
        "position": {"x": 4.0, "y": 0.0, "z": 1.5},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
    }
    wall.custom_properties["wall_slab"] = {
        "room_id": "room-a",
        "start_xyz_m": [4.0, 0.0, 0.0],
        "end_xyz_m": [4.0, 3.0, 0.0],
        "height_m": 3.0,
        "thickness_m": 0.2,
        "measurement_uncertainty_m": 0.01,
    }
    geom_w = Geometry(
        id="geom-wall-ab", type=GeometryType.BOX,
        bounds_min=Vector3(3.9, -0.1, 0.0),
        bounds_max=Vector3(4.1, 3.1, 3.0),
        provenance=Provenance.RECONSTRUCTED, confidence=0.9,
    )
    wall.geometry_ids.append(geom_w.id)
    world.entities["wall-ab"] = wall
    world.geometries["geom-wall-ab"] = geom_w

    return world


