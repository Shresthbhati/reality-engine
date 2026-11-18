"""Multi-View Room Association Engine (Programme 1: Multi-View Intelligence).

Determines:
1. Physical containment: which room a camera/photograph is physically inside (3D point-in-polyhedron).
2. Visual co-visibility: which rooms and wall surfaces are visible in the camera's view frustum.
3. Cross-view room identity: connects disparate photographs to their canonical physical room entities
   across multi-room floorplans and multi-session captures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPose3D:
    """A reconstructed or calibrated 3D camera station."""

    evidence_id: str
    position: tuple[float, float, float]
    look_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)  # normalized forward vector
    up_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)
    hfov_deg: float = 60.0
    vfov_deg: float = 45.0
    max_view_depth_m: float = 12.0


@dataclass
class RoomBoundary:
    """A 3D spatial boundary representing a physical room."""

    room_id: str
    name: str
    polygon_xy: list[tuple[float, float]]  # closed plan polygon vertices
    floor_z: float
    ceiling_z: float
    wall_element_ids: list[str] = field(default_factory=list)
    boundary_walls: list[dict[str, Any]] = field(default_factory=list)  # [{"id", "start", "end"}]


@dataclass
class PhotoRoomAssociation:
    """Association of a photograph/camera with rooms and surfaces."""

    evidence_id: str
    inside_room_id: str | None
    visible_room_ids: list[str]
    visible_wall_ids: list[str]
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "inside_room_id": self.inside_room_id,
            "visible_room_ids": list(self.visible_room_ids),
            "visible_wall_ids": list(self.visible_wall_ids),
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


def point_in_polygon_2d(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Robust 2D point-in-polygon test using ray casting with epsilon boundary tolerance."""
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if point lies directly on edge (within 1cm)
        dx, dy = x2 - x1, y2 - y1
        edge_len_sq = dx * dx + dy * dy
        if edge_len_sq > 1e-12:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / edge_len_sq))
            px = x1 + t * dx
            py = y1 + t * dy
            dist_sq = (x - px) ** 2 + (y - py) ** 2
            if dist_sq < 1e-4:  # <= 1cm to edge
                return True

        # Ray casting
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside

    return inside


class MultiViewRoomAssociator:
    """Associates multi-view photographs and camera poses to physical rooms."""

    def __init__(
        self,
        elevation_margin_m: float = 0.25,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self.elevation_margin = elevation_margin_m
        self.up = tuple(float(x) for x in up)
        self.is_y_up = abs(self.up[1]) > 0.5

    def is_camera_inside_room(self, camera: CameraPose3D, room: RoomBoundary) -> bool:
        """Verify whether camera 3D coordinate is physically enclosed within the room volume."""
        cx, cy, cz = camera.position
        if self.is_y_up:
            elev = cy
            plan_pt = (cx, cz)
        else:
            elev = cz
            plan_pt = (cx, cy)

        # 1. Height check with margin (accommodating tripods or hands slightly elevated)
        if elev < (room.floor_z - self.elevation_margin) or elev > (room.ceiling_z + self.elevation_margin):
            return False

        # 2. Plan boundary test
        return point_in_polygon_2d(plan_pt, room.polygon_xy)

    def associate_camera(
        self,
        camera: CameraPose3D,
        rooms: Sequence[RoomBoundary],
        doors: Sequence[dict[str, Any]] | None = None,
    ) -> PhotoRoomAssociation:
        """Associate a single camera with rooms: determine enclosure and visible spaces."""
        inside_room_id: str | None = None
        for room in rooms:
            if self.is_camera_inside_room(camera, room):
                inside_room_id = room.room_id
                break

        visible_rooms: set[str] = set()
        if inside_room_id:
            visible_rooms.add(inside_room_id)

        visible_walls: list[str] = []
        cam_pos = np.asarray(camera.position, dtype=np.float64)
        look_dir = np.asarray(camera.look_dir, dtype=np.float64)
        look_norm = np.linalg.norm(look_dir)
        if look_norm > 1e-12:
            look_dir = look_dir / look_norm

        half_fov_rad = math.radians(camera.hfov_deg / 2.0)
        cos_half_fov = math.cos(half_fov_rad)

        # Inspect which room walls are facing and inside the view cone
        for room in rooms:
            z_mid = (room.floor_z + room.ceiling_z) / 2.0
            for w in room.boundary_walls:
                wid = w.get("id", "wall")
                start = np.asarray(w.get("start", [0.0, 0.0]), dtype=np.float64)
                end = np.asarray(w.get("end", [0.0, 0.0]), dtype=np.float64)

                # Sample 5 points along wall segment
                wall_visible = False
                for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                    pt_xy = (1.0 - t) * start + t * end
                    pt_3d = np.array([pt_xy[0], pt_xy[1], z_mid])
                    vec = pt_3d - cam_pos
                    dist = float(np.linalg.norm(vec))
                    if 0.1 <= dist <= camera.max_view_depth_m:
                        dir_v = vec / dist
                        if float(np.dot(look_dir, dir_v)) >= cos_half_fov:
                            wall_visible = True
                            break

                if wall_visible:
                    visible_walls.append(wid)
                    visible_rooms.add(room.room_id)

        # Check door portals: if looking directly at an open door, the connected room is visible
        if doors:
            for d in doors:
                center = np.asarray(d.get("center", [0.0, 0.0, 1.0]), dtype=np.float64)
                vec_to_door = center - cam_pos
                dist_d = float(np.linalg.norm(vec_to_door))
                if dist_d < camera.max_view_depth_m:
                    dir_to_door = vec_to_door / (dist_d + 1e-12)
                    if float(np.dot(look_dir, dir_to_door)) >= cos_half_fov:
                        # Add connected rooms across this portal
                        for r_ref in d.get("connects_rooms", []):
                            visible_rooms.add(r_ref)

        confidence = 0.95 if inside_room_id else (0.70 if visible_rooms else 0.10)
        note = "inside_room" if inside_room_id else ("portal_or_exterior_view" if visible_rooms else "outside_or_unregistered")

        return PhotoRoomAssociation(
            evidence_id=camera.evidence_id,
            inside_room_id=inside_room_id,
            visible_room_ids=sorted(visible_rooms),
            visible_wall_ids=visible_walls,
            confidence=confidence,
            notes=note,
        )

    def associate_photographs(
        self,
        cameras: Sequence[CameraPose3D],
        rooms: Sequence[RoomBoundary],
        doors: Sequence[dict[str, Any]] | None = None,
    ) -> list[PhotoRoomAssociation]:
        """Associate all captured cameras with physical rooms across the building floorplan."""
        return [self.associate_camera(cam, rooms, doors) for cam in cameras]

    def build_cross_view_covisibility(
        self,
        associations: Sequence[PhotoRoomAssociation],
    ) -> dict[tuple[str, str], float]:
        """Compute co-visibility matrix between photographs (0.0 to 1.0 score).

        Two photographs share high co-visibility if they reside in the same room and view
        the same wall surfaces.
        """
        covis: dict[tuple[str, str], float] = {}
        n = len(associations)

        for i in range(n):
            a1 = associations[i]
            w1 = set(a1.visible_wall_ids)
            for j in range(i + 1, n):
                a2 = associations[j]
                w2 = set(a2.visible_wall_ids)

                score = 0.0
                # Same room base score
                if a1.inside_room_id and a2.inside_room_id and a1.inside_room_id == a2.inside_room_id:
                    score += 0.40

                # Shared walls Jaccard overlap
                if w1 and w2:
                    intersection = len(w1 & w2)
                    union = len(w1 | w2)
                    jaccard = intersection / union if union > 0 else 0.0
                    score += 0.60 * jaccard

                key = (min(a1.evidence_id, a2.evidence_id), max(a1.evidence_id, a2.evidence_id))
                covis[key] = round(min(1.0, score), 3)

        return covis
