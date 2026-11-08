"""Real-Time Multi-Room Semantic Navigation & Automated Walkthrough Tour Engine (Track B).

Provides:
- Room waypoint extraction and spatial containment detection from WorldIR.
- Topological pathfinding between rooms and doorways.
- Automated cinematic architectural tour generation with keyframe spline pathing.
- Spatial bookmarks management for camera vantage points and POIs.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


@dataclass
class RoomWaypoint:
    """A semantic room or architectural space in the building."""

    room_id: str
    name: str
    room_type: str
    center: list[float]  # [x, y, z]
    bounds_min: list[float]  # [min_x, min_y, min_z]
    bounds_max: list[float]  # [max_x, max_y, max_z]
    floor_elevation_m: float
    ceiling_height_m: float
    openings: list[dict] = field(default_factory=list)
    area_sqm: float = 0.0

    def contains_point(self, pos: Sequence[float], margin: float = 0.15) -> bool:
        """Check if 3D position [x, y, z] falls inside room volume."""
        x, y, z = pos[0], pos[1], pos[2]
        within_xz = (
            (self.bounds_min[0] - margin) <= x <= (self.bounds_max[0] + margin)
            and (self.bounds_min[2] - margin) <= z <= (self.bounds_max[2] + margin)
        )
        within_y = (self.floor_elevation_m - 0.3) <= y <= (self.floor_elevation_m + self.ceiling_height_m + 0.3)
        return within_xz and within_y


@dataclass
class SpatialBookmark:
    """A user or system-saved spatial perspective / point of interest."""

    bookmark_id: str
    name: str
    room_id: Optional[str]
    room_name: Optional[str]
    position: list[float]  # [x, y, z]
    yaw_rad: float
    pitch_rad: float
    fov_deg: float = 60.0
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "bookmark_id": self.bookmark_id,
            "name": self.name,
            "room_id": self.room_id,
            "room_name": self.room_name,
            "position": self.position,
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "fov_deg": self.fov_deg,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpatialBookmark:
        return cls(
            bookmark_id=data.get("bookmark_id", f"bmk_{uuid.uuid4().hex[:8]}"),
            name=data.get("name", "Spatial Viewpoint"),
            room_id=data.get("room_id"),
            room_name=data.get("room_name"),
            position=list(data.get("position", [0.0, 1.65, 0.0])),
            yaw_rad=float(data.get("yaw_rad", 0.0)),
            pitch_rad=float(data.get("pitch_rad", 0.0)),
            fov_deg=float(data.get("fov_deg", 60.0)),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class TourKeyframe:
    """A timed camera vantage stop along the architectural walkthrough tour."""

    keyframe_id: str
    room_id: str
    room_name: str
    position: list[float]  # [x, y, z]
    yaw_rad: float
    pitch_rad: float
    dwell_seconds: float
    transition_seconds: float
    narration: str
    kind: str  # "entry" | "panorama" | "feature" | "transition"

    def to_dict(self) -> dict:
        return {
            "keyframe_id": self.keyframe_id,
            "room_id": self.room_id,
            "room_name": self.room_name,
            "position": self.position,
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "dwell_seconds": self.dwell_seconds,
            "transition_seconds": self.transition_seconds,
            "narration": self.narration,
            "kind": self.kind,
        }


@dataclass
class ArchitecturalTour:
    """An automated cinematic architectural walkthrough tour."""

    tour_id: str
    title: str
    description: str
    total_duration_seconds: float
    keyframes: list[TourKeyframe]
    waypoint_path: list[list[float]]  # interpolated dense camera path [x, y, z]
    rooms_visited: list[str]

    def to_dict(self) -> dict:
        return {
            "tour_id": self.tour_id,
            "title": self.title,
            "description": self.description,
            "total_duration_seconds": self.total_duration_seconds,
            "keyframes": [k.to_dict() for k in self.keyframes],
            "waypoint_path": self.waypoint_path,
            "rooms_visited": self.rooms_visited,
        }


@dataclass
class NavigationData:
    """Consolidated semantic navigation and tour payload for frontend viewers."""

    world_id: str
    rooms: list[RoomWaypoint]
    doors: list[dict]
    recommended_tour: ArchitecturalTour
    suggested_bookmarks: list[SpatialBookmark]
    bounds: dict

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "rooms": [
                {
                    "room_id": r.room_id,
                    "name": r.name,
                    "room_type": r.room_type,
                    "center": r.center,
                    "bounds_min": r.bounds_min,
                    "bounds_max": r.bounds_max,
                    "floor_elevation_m": r.floor_elevation_m,
                    "ceiling_height_m": r.ceiling_height_m,
                    "area_sqm": r.area_sqm,
                    "openings": r.openings,
                }
                for r in self.rooms
            ],
            "doors": self.doors,
            "recommended_tour": self.recommended_tour.to_dict(),
            "suggested_bookmarks": [b.to_dict() for b in self.suggested_bookmarks],
            "bounds": self.bounds,
        }


# ==============================================================================
# Room Extraction & Spatial Containment
# ==============================================================================

def _get_props(ent: Any) -> dict:
    if hasattr(ent, "custom_properties") and isinstance(ent.custom_properties, dict):
        return ent.custom_properties
    if hasattr(ent, "properties") and isinstance(ent.properties, dict):
        return ent.properties
    return {}


def extract_rooms_from_worldir(world: WorldIR) -> Tuple[list[RoomWaypoint], list[dict]]:
    """Extract room waypoints and opening connectors from WorldIR entities and metadata."""
    rooms: list[RoomWaypoint] = []
    doors: list[dict] = []

    # 1. Check for explicit Room / Space entities
    room_entities = [
        ent
        for ent in world.entities.values()
        if ent.type in (EntityType.ROOM, EntityType.CORRIDOR)
        or str(getattr(ent, "type", "")).lower() in ("room", "corridor", "space")
    ]

    # Collect doors / openings
    door_entities = [
        ent
        for ent in world.entities.values()
        if ent.type in (EntityType.DOOR, EntityType.OPENING)
    ]

    for d in door_entities:
        pos = [0.0, 0.0, 0.0]
        if d.transform and "position" in d.transform:
            p = d.transform["position"]
            pos = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
        d_props = _get_props(d)
        doors.append({
            "door_id": d.id,
            "name": d.name or d.id,
            "position": pos,
            "width_m": float(d_props.get("width_m", 0.9)),
            "connected_spaces": list(d_props.get("connected_spaces", [])),
        })

    # 2. Extract from SpaceGraph in metadata if available
    sg_dict = world.metadata.get("interior_space_graph", {})
    sg_rooms = sg_dict.get("rooms", {}) if isinstance(sg_dict, dict) else {}

    if room_entities:
        for ent in room_entities:
            pos = [0.0, 0.0, 0.0]
            if ent.transform and "position" in ent.transform:
                p = ent.transform["position"]
                pos = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
            
            ent_props = _get_props(ent)
            # Dimensions
            width = float(ent_props.get("width_m", ent_props.get("dimension_x", 4.0)))
            depth = float(ent_props.get("depth_m", ent_props.get("dimension_z", 4.0)))
            height = float(ent_props.get("height_m", 2.6))
            
            b_min = [pos[0] - width / 2.0, pos[1], pos[2] - depth / 2.0]
            b_max = [pos[0] + width / 2.0, pos[1] + height, pos[2] + depth / 2.0]
            area = round(width * depth, 2)

            rooms.append(
                RoomWaypoint(
                    room_id=ent.id,
                    name=ent.name or ent.id.replace("_", " ").title(),
                    room_type=ent_props.get("room_type", "room"),
                    center=pos,
                    bounds_min=b_min,
                    bounds_max=b_max,
                    floor_elevation_m=pos[1],
                    ceiling_height_m=height,
                    area_sqm=area,
                )
            )
    elif sg_rooms:
        for rid, rdata in sg_rooms.items():
            cx = float(rdata.get("centroid_x", 0.0))
            cz = float(rdata.get("centroid_z", 0.0))
            elev = float(rdata.get("elevation_m", 0.0))
            w = float(rdata.get("width_m", 4.0))
            d = float(rdata.get("depth_m", 4.0))
            h = float(rdata.get("height_m", 2.6))
            
            rooms.append(
                RoomWaypoint(
                    room_id=rid,
                    name=rdata.get("name", rid.replace("_", " ").title()),
                    room_type=rdata.get("room_type", "room"),
                    center=[cx, elev, cz],
                    bounds_min=[cx - w / 2.0, elev, cz - d / 2.0],
                    bounds_max=[cx + w / 2.0, elev + h, cz + d / 2.0],
                    floor_elevation_m=elev,
                    ceiling_height_m=h,
                    area_sqm=round(w * d, 2),
                )
            )

    # 3. Fallback: If no explicit room objects found, cluster wall/floor entities to infer space bounds
    if not rooms:
        wall_entities = [
            ent
            for ent in world.entities.values()
            if ent.type in (EntityType.WALL, EntityType.FLOOR, EntityType.STRUCTURE)
        ]
        if wall_entities:
            xs, ys, zs = [], [], []
            for w in wall_entities:
                if w.transform and "position" in w.transform:
                    p = w.transform["position"]
                    xs.append(p.get("x", 0.0))
                    ys.append(p.get("y", 0.0))
                    zs.append(p.get("z", 0.0))
            if xs:
                min_x, max_x = min(xs) - 1.0, max(xs) + 1.0
                min_y, max_y = min(ys), max(ys) + 2.8
                min_z, max_z = min(zs) - 1.0, max(zs) + 1.0
                cx = (min_x + max_x) / 2.0
                cz = (min_z + max_z) / 2.0
                rooms.append(
                    RoomWaypoint(
                        room_id="room_main",
                        name="Main Interior Space",
                        room_type="general",
                        center=[cx, min_y, cz],
                        bounds_min=[min_x, min_y, min_z],
                        bounds_max=[max_x, max_y, max_z],
                        floor_elevation_m=min_y,
                        ceiling_height_m=max(2.5, max_y - min_y),
                        area_sqm=round(abs(max_x - min_x) * abs(max_z - min_z), 2),
                    )
                )

    # 4. Default ultimate fallback if world is completely empty
    if not rooms:
        rooms.append(
            RoomWaypoint(
                room_id="room_default",
                name="Primary Space",
                room_type="general",
                center=[0.0, 0.0, 0.0],
                bounds_min=[-3.0, 0.0, -3.0],
                bounds_max=[3.0, 2.7, 3.0],
                floor_elevation_m=0.0,
                ceiling_height_m=2.7,
                area_sqm=36.0,
            )
        )

    # Match doors to adjacent rooms
    for d in doors:
        dx, dz = d["position"][0], d["position"][2]
        for r in rooms:
            if r.contains_point([dx, r.floor_elevation_m + 0.5, dz], margin=0.6):
                if d["door_id"] not in [op.get("door_id") for op in r.openings]:
                    r.openings.append(d)

    return rooms, doors


def detect_room_at_position(
    pos: Sequence[float],
    rooms: Sequence[RoomWaypoint],
) -> Optional[RoomWaypoint]:
    """Identify which room the player or camera is currently inside."""
    for r in rooms:
        if r.contains_point(pos):
            return r
    # Secondary fallback: find closest room center within horizontal reach
    closest = None
    min_dist = float("inf")
    for r in rooms:
        d = math.hypot(pos[0] - r.center[0], pos[2] - r.center[2])
        if d < min_dist:
            min_dist = d
            closest = r
    if closest and min_dist <= 6.0:
        return closest
    return None


# ==============================================================================
# Automated Architectural Walkthrough Tour Generation
# ==============================================================================

_ROOM_TYPE_ORDER = [
    "entrance",
    "foyer",
    "hallway",
    "corridor",
    "living_room",
    "living",
    "salon",
    "kitchen",
    "dining",
    "bedroom",
    "master_bedroom",
    "office",
    "study",
    "bathroom",
    "powder_room",
    "balcony",
    "terrace",
    "patio",
]


def _sort_rooms_for_tour(rooms: Sequence[RoomWaypoint]) -> list[RoomWaypoint]:
    """Sort rooms into an intuitive architectural progression."""
    def _rank(r: RoomWaypoint) -> int:
        rtype = r.room_type.lower()
        rname = r.name.lower()
        for idx, pattern in enumerate(_ROOM_TYPE_ORDER):
            if pattern in rtype or pattern in rname:
                return idx
        return 100

    return sorted(rooms, key=_rank)


def generate_architectural_tour(
    world: WorldIR,
    project_name: str = "Architectural Walkthrough",
) -> Tuple[ArchitecturalTour, list[SpatialBookmark]]:
    """Build an automated cinematic multi-room walkthrough tour with bookmarks."""
    rooms, doors = extract_rooms_from_worldir(world)
    sorted_rooms = _sort_rooms_for_tour(rooms)

    keyframes: list[TourKeyframe] = []
    bookmarks: list[SpatialBookmark] = []
    visited_rooms: list[str] = []
    eye_height = 1.65

    for idx, room in enumerate(sorted_rooms):
        visited_rooms.append(room.name)
        rx, rz = room.center[0], room.center[2]
        fy = room.floor_elevation_m + eye_height

        # 1. Entry Keyframe
        entry_pos = [rx, fy, rz]
        yaw = 0.0
        pitch = -0.05

        if room.openings:
            first_door = room.openings[0]["position"]
            entry_pos = [first_door[0], fy, first_door[2]]
            # Look towards center of room from door
            dx = rx - first_door[0]
            dz = rz - first_door[2]
            yaw = math.atan2(-dx, -dz)
        else:
            # Stand near room min bounds looking toward center
            entry_pos = [
                room.bounds_min[0] + 0.5,
                fy,
                room.bounds_min[2] + 0.5,
            ]
            dx = rx - entry_pos[0]
            dz = rz - entry_pos[2]
            yaw = math.atan2(-dx, -dz)

        k_entry = TourKeyframe(
            keyframe_id=f"kf_{room.room_id}_entry",
            room_id=room.room_id,
            room_name=room.name,
            position=entry_pos,
            yaw_rad=round(yaw, 3),
            pitch_rad=round(pitch, 3),
            dwell_seconds=2.0,
            transition_seconds=2.5 if idx > 0 else 0.0,
            narration=f"Entering {room.name} ({room.area_sqm} m²).",
            kind="entry",
        )
        keyframes.append(k_entry)

        # 2. Room Center Panoramic Keyframe
        center_pos = [rx, fy, rz]
        # Diagonal vantage look for best spatial perception
        diag_x = room.bounds_max[0] - rx
        diag_z = room.bounds_max[2] - rz
        pan_yaw = math.atan2(-diag_x, -diag_z)

        k_center = TourKeyframe(
            keyframe_id=f"kf_{room.room_id}_pan",
            room_id=room.room_id,
            room_name=room.name,
            position=center_pos,
            yaw_rad=round(pan_yaw, 3),
            pitch_rad=-0.08,
            dwell_seconds=3.0,
            transition_seconds=2.0,
            narration=f"Exploring {room.name} architectural layout and finished volumes.",
            kind="panorama",
        )
        keyframes.append(k_center)

        # 3. Create a spatial bookmark for each room center
        bmk = SpatialBookmark(
            bookmark_id=f"bmk_{room.room_id}",
            name=f"{room.name} Vantage",
            room_id=room.room_id,
            room_name=room.name,
            position=center_pos,
            yaw_rad=round(pan_yaw, 3),
            pitch_rad=-0.08,
            fov_deg=65.0,
            tags=[room.room_type, "room_center"],
        )
        bookmarks.append(bmk)

    # Calculate total duration
    total_sec = sum(k.dwell_seconds + k.transition_seconds for k in keyframes)

    # Generate dense interpolated waypoint path for visual mini-map trace
    waypoint_path = _interpolate_dense_path(keyframes, samples_per_transition=10)

    tour = ArchitecturalTour(
        tour_id=f"tour_{uuid.uuid4().hex[:8]}",
        title=f"{project_name} Walkthrough",
        description=f"Automated architectural walkthrough tour visiting {len(sorted_rooms)} spaces.",
        total_duration_seconds=round(total_sec, 1),
        keyframes=keyframes,
        waypoint_path=waypoint_path,
        rooms_visited=visited_rooms,
    )

    return tour, bookmarks


def _interpolate_dense_path(
    keyframes: Sequence[TourKeyframe],
    samples_per_transition: int = 10,
) -> list[list[float]]:
    """Compute dense [x, y, z] points along keyframe path for mini-map path line."""
    if not keyframes:
        return []
    path: list[list[float]] = []

    for i in range(len(keyframes) - 1):
        p0 = keyframes[i].position
        p1 = keyframes[i + 1].position
        for s in range(samples_per_transition):
            t = s / float(samples_per_transition)
            # Smooth cosine interpolation
            t_smooth = (1.0 - math.cos(t * math.pi)) / 2.0
            x = p0[0] + (p1[0] - p0[0]) * t_smooth
            y = p0[1] + (p1[1] - p0[1]) * t_smooth
            z = p0[2] + (p1[2] - p0[2]) * t_smooth
            path.append([round(x, 3), round(y, 3), round(z, 3)])

    path.append(keyframes[-1].position)
    return path


def interpolate_tour_pose(
    tour: ArchitecturalTour,
    elapsed_seconds: float,
) -> dict:
    """Evaluate camera position, orientation, and current room at any timestamp t.

    Returns dict with keys: position, yaw_rad, pitch_rad, current_room, narration, progress_pct.
    """
    if not tour.keyframes:
        return {
            "position": [0.0, 1.65, 0.0],
            "yaw_rad": 0.0,
            "pitch_rad": 0.0,
            "current_room": "Space",
            "narration": "",
            "progress_pct": 0.0,
        }

    clamped_t = max(0.0, min(tour.total_duration_seconds, elapsed_seconds))
    progress = (
        (clamped_t / tour.total_duration_seconds) * 100.0
        if tour.total_duration_seconds > 0
        else 100.0
    )

    accumulated = 0.0
    for i, kf in enumerate(tour.keyframes):
        # 1. Transition phase into this keyframe
        if kf.transition_seconds > 0:
            if accumulated + kf.transition_seconds > clamped_t:
                prev_kf = tour.keyframes[max(0, i - 1)]
                alpha = (clamped_t - accumulated) / kf.transition_seconds
                # Smoothstep Hermite curve
                s = alpha * alpha * (3.0 - 2.0 * alpha)
                pos = [
                    prev_kf.position[c] + (kf.position[c] - prev_kf.position[c]) * s
                    for c in range(3)
                ]
                # Angle interpolation taking shortest angular path
                yaw = _interpolate_angle(prev_kf.yaw_rad, kf.yaw_rad, s)
                pitch = prev_kf.pitch_rad + (kf.pitch_rad - prev_kf.pitch_rad) * s
                return {
                    "position": [round(c, 3) for c in pos],
                    "yaw_rad": round(yaw, 3),
                    "pitch_rad": round(pitch, 3),
                    "current_room": kf.room_name,
                    "narration": f"Navigating to {kf.room_name}...",
                    "progress_pct": round(progress, 1),
                }
            accumulated += kf.transition_seconds

        # 2. Dwell phase at this keyframe
        if accumulated + kf.dwell_seconds > clamped_t:
            return {
                "position": kf.position,
                "yaw_rad": kf.yaw_rad,
                "pitch_rad": kf.pitch_rad,
                "current_room": kf.room_name,
                "narration": kf.narration,
                "progress_pct": round(progress, 1),
            }
        accumulated += kf.dwell_seconds

    # Reached end of tour
    last = tour.keyframes[-1]
    return {
        "position": last.position,
        "yaw_rad": last.yaw_rad,
        "pitch_rad": last.pitch_rad,
        "current_room": last.room_name,
        "narration": "Walkthrough tour complete.",
        "progress_pct": 100.0,
    }


def _interpolate_angle(a0: float, a1: float, t: float) -> float:
    """Interpolate angle in radians along the shortest circular path."""
    diff = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    return a0 + diff * t


# ==============================================================================
# Consolidated Navigation Package Builder
# ==============================================================================

def build_world_navigation_data(
    world: WorldIR,
    project_name: str = "Architectural Space",
) -> NavigationData:
    """Build full navigation manifest for a world."""
    rooms, doors = extract_rooms_from_worldir(world)
    tour, suggested_bmk = generate_architectural_tour(world, project_name=project_name)

    # Compute aggregate bounding volume
    all_min_x = [r.bounds_min[0] for r in rooms]
    all_max_x = [r.bounds_max[0] for r in rooms]
    all_min_z = [r.bounds_min[2] for r in rooms]
    all_max_z = [r.bounds_max[2] for r in rooms]

    bounds = {
        "min_x": min(all_min_x) if all_min_x else -5.0,
        "max_x": max(all_max_x) if all_max_x else 5.0,
        "min_z": min(all_min_z) if all_min_z else -5.0,
        "max_z": max(all_max_z) if all_max_z else 5.0,
    }

    # Load custom bookmarks if stored in world metadata
    custom_bmks = []
    if "spatial_bookmarks" in world.metadata:
        for b_dict in world.metadata["spatial_bookmarks"]:
            try:
                custom_bmks.append(SpatialBookmark.from_dict(b_dict))
            except Exception:
                pass

    bookmarks = custom_bmks if custom_bmks else suggested_bmk

    return NavigationData(
        world_id=world.id,
        rooms=rooms,
        doors=doors,
        recommended_tour=tour,
        suggested_bookmarks=bookmarks,
        bounds=bounds,
    )
