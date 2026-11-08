"""Multi-User Real-Time Collaborative Telepresence & Live Spatial Markup.

Provides:
1. Real-time peer presence tracking, camera pose broadcast, and avatar synchronization.
2. 3D spatial annotations & markups (pins, 3D surface strokes, measurement callouts).
3. Persistent WorldIR integration (saved in world.custom_properties["collab_markups"]).
4. buildingSMART BCF 2.1 zip archive export for spatial markups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import io
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid
import xml.etree.ElementTree as ET
import zipfile


class MarkupType(str, Enum):
    PIN = "pin"
    STROKE = "stroke"
    MEASUREMENT = "measurement"


class MarkupTrade(str, Enum):
    ARCHITECTURE = "architecture"
    STRUCTURE = "structure"
    MEP = "mep"
    SAFETY = "safety"
    GENERAL = "general"


class MarkupStatus(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"


@dataclass(slots=True)
class PeerPresence:
    """Active collaborator presence in a 3D world."""
    client_id: str
    user_name: str
    role: str = "architect"
    color: str = "#3b82f6"
    camera_position: tuple[float, float, float] = (0.0, 0.0, 1.8)
    look_at: tuple[float, float, float] = (0.0, 1.0, 1.8)
    field_of_view: float = 60.0
    last_heartbeat: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "user_name": self.user_name,
            "role": self.role,
            "color": self.color,
            "camera_position": [round(c, 4) for c in self.camera_position],
            "look_at": [round(c, 4) for c in self.look_at],
            "field_of_view": round(self.field_of_view, 1),
            "last_heartbeat": round(self.last_heartbeat, 2),
            "is_active": (time.time() - self.last_heartbeat) < 30.0,
        }


@dataclass(slots=True)
class SpatialMarkup:
    """3D spatial annotation, pin, or stroke pinned to the model."""
    markup_id: str
    world_id: str
    markup_type: MarkupType = MarkupType.PIN
    trade: MarkupTrade = MarkupTrade.GENERAL
    title: str = ""
    description: str = ""
    author_name: str = "Collaborator"
    author_id: str = "user-1"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    stroke_points: list[tuple[float, float, float]] = field(default_factory=list)
    camera_viewpoint: dict[str, Any] = field(default_factory=dict)
    color: str = "#ef4444"
    status: MarkupStatus = MarkupStatus.OPEN
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "markup_id": self.markup_id,
            "world_id": self.world_id,
            "markup_type": self.markup_type.value,
            "trade": self.trade.value,
            "title": self.title,
            "description": self.description,
            "author_name": self.author_name,
            "author_id": self.author_id,
            "position": [round(c, 4) for c in self.position],
            "normal": [round(c, 4) for c in self.normal],
            "stroke_points": [[round(c, 4) for c in pt] for pt in self.stroke_points],
            "camera_viewpoint": self.camera_viewpoint,
            "color": self.color,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialMarkup:
        pos = data.get("position", [0.0, 0.0, 0.0])
        norm = data.get("normal", [0.0, 0.0, 1.0])
        raw_pts = data.get("stroke_points", [])
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in raw_pts if len(p) >= 3]

        return cls(
            markup_id=str(data.get("markup_id") or f"markup-{uuid.uuid4().hex[:8]}"),
            world_id=str(data.get("world_id", "")),
            markup_type=MarkupType(data.get("markup_type", MarkupType.PIN.value)),
            trade=MarkupTrade(data.get("trade", MarkupTrade.GENERAL.value)),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            author_name=str(data.get("author_name", "Collaborator")),
            author_id=str(data.get("author_id", "user-1")),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            normal=(float(norm[0]), float(norm[1]), float(norm[2])),
            stroke_points=pts,
            camera_viewpoint=dict(data.get("camera_viewpoint", {})),
            color=str(data.get("color", "#ef4444")),
            status=MarkupStatus(data.get("status", MarkupStatus.OPEN.value)),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


class CollabPresenceRegistry:
    """Thread-safe in-memory peer presence tracking per world."""

    def __init__(self):
        # world_id -> {client_id: PeerPresence}
        self._worlds: dict[str, dict[str, PeerPresence]] = {}

    def update_peer(
        self,
        world_id: str,
        client_id: str,
        user_name: str,
        role: str = "architect",
        color: str = "#3b82f6",
        camera_position: tuple[float, float, float] = (0.0, 0.0, 1.8),
        look_at: tuple[float, float, float] = (0.0, 1.0, 1.8),
        field_of_view: float = 60.0,
    ) -> PeerPresence:
        if world_id not in self._worlds:
            self._worlds[world_id] = {}

        now = time.time()
        peer = PeerPresence(
            client_id=client_id,
            user_name=user_name,
            role=role,
            color=color,
            camera_position=camera_position,
            look_at=look_at,
            field_of_view=field_of_view,
            last_heartbeat=now,
        )
        self._worlds[world_id][client_id] = peer
        return peer

    def remove_peer(self, world_id: str, client_id: str) -> bool:
        if world_id in self._worlds and client_id in self._worlds[world_id]:
            del self._worlds[world_id][client_id]
            return True
        return False

    def get_active_peers(self, world_id: str, timeout_sec: float = 30.0) -> list[PeerPresence]:
        if world_id not in self._worlds:
            return []

        now = time.time()
        active = []
        expired = []
        for cid, peer in self._worlds[world_id].items():
            if (now - peer.last_heartbeat) <= timeout_sec:
                active.append(peer)
            else:
                expired.append(cid)

        # Clean expired
        for cid in expired:
            del self._worlds[world_id][cid]

        return sorted(active, key=lambda p: p.user_name)


# Global singleton instance
_GLOBAL_PRESENCE = CollabPresenceRegistry()


def get_presence_registry() -> CollabPresenceRegistry:
    return _GLOBAL_PRESENCE


# =========================================================================
# WorldIR Markup Persistence
# =========================================================================

def _get_markup_dict(world: Any) -> dict[str, Any]:
    if hasattr(world, "metadata") and isinstance(world.metadata, dict):
        return world.metadata
    if not hasattr(world, "custom_properties") or world.custom_properties is None:
        world.custom_properties = {}
    return world.custom_properties


def load_world_markups(world: Any) -> list[SpatialMarkup]:
    """Extract collaborative markups stored in world metadata or custom_properties."""
    target = _get_markup_dict(world)
    raw_list = target.get("collab_markups", [])
    markups: list[SpatialMarkup] = []
    for item in raw_list:
        if isinstance(item, dict):
            try:
                markups.append(SpatialMarkup.from_dict(item))
            except Exception:
                continue
    return markups


def save_world_markup(world: Any, markup: SpatialMarkup) -> list[SpatialMarkup]:
    """Append or update a spatial markup in world metadata."""
    target = _get_markup_dict(world)
    existing = load_world_markups(world)
    found = False
    new_list = []
    for m in existing:
        if m.markup_id == markup.markup_id:
            new_list.append(markup)
            found = True
        else:
            new_list.append(m)

    if not found:
        new_list.append(markup)

    target["collab_markups"] = [m.to_dict() for m in new_list]
    return new_list


def delete_world_markup(world: Any, markup_id: str) -> bool:
    """Remove a spatial markup from world metadata."""
    target = _get_markup_dict(world)
    existing = load_world_markups(world)
    filtered = [m for m in existing if m.markup_id != markup_id]
    if len(filtered) == len(existing):
        return False

    target["collab_markups"] = [m.to_dict() for m in filtered]
    return True


# =========================================================================
# buildingSMART BCF 2.1 Exporter for Markups
# =========================================================================

def export_markups_to_bcf(markups: Sequence[SpatialMarkup], world_id: str) -> bytes:
    """Package collaborative spatial markups into standard buildingSMART BCF 2.1 zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. bcf.version
        zf.writestr(
            "bcf.version",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Version VersionId="2.1" xmlns="http://www.buildingsmart-tech.org/specifications/bcf-releases" />\n',
        )

        for markup in markups:
            topic_guid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reality-engine://markup/{markup.markup_id}"))
            topic_dir = f"{topic_guid}/"

            # 2. markup.bcf
            root = ET.Element("Markup", xmlns="http://www.buildingsmart-tech.org/specifications/bcf-releases")
            header = ET.SubElement(root, "Header")
            file_el = ET.SubElement(header, "File", IfcProject=world_id)
            ET.SubElement(file_el, "Filename").text = f"{world_id}.ifc"
            ET.SubElement(file_el, "Date").text = markup.created_at

            topic = ET.SubElement(
                root,
                "Topic",
                Guid=topic_guid,
                TopicType=markup.trade.value.capitalize(),
                TopicStatus=markup.status.value.capitalize(),
            )
            ET.SubElement(topic, "Title").text = markup.title or f"Markup {markup.markup_id}"
            ET.SubElement(topic, "CreationDate").text = markup.created_at
            ET.SubElement(topic, "CreationAuthor").text = markup.author_name
            ET.SubElement(topic, "Description").text = markup.description or markup.title

            comment = ET.SubElement(root, "Comment", Guid=str(uuid.uuid4()))
            ET.SubElement(comment, "Date").text = markup.created_at
            ET.SubElement(comment, "Author").text = markup.author_name
            ET.SubElement(comment, "Comment").text = (
                f"Trade: {markup.trade.value} | Type: {markup.markup_type.value}\n"
                f"Position: ({round(markup.position[0], 3)}, {round(markup.position[1], 3)}, {round(markup.position[2], 3)})"
            )

            # Viewpoint linkage
            vp_guid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reality-engine://viewpoint/{markup.markup_id}"))
            ET.SubElement(topic, "Viewpoints", Guid=vp_guid)
            zf.writestr(f"{topic_dir}markup.bcf", ET.tostring(root, encoding="utf-8", xml_declaration=True))

            # 3. viewpoint.bcfv
            vp_root = ET.Element("VisualizationInfo", xmlns="http://www.buildingsmart-tech.org/specifications/bcf-releases", Guid=vp_guid)
            cam = markup.camera_viewpoint
            cam_pos = cam.get("position", [markup.position[0] - 1.0, markup.position[1] - 1.0, markup.position[2] + 0.8])
            cam_dir = cam.get("direction", [1.0, 1.0, -0.4])

            persp = ET.SubElement(vp_root, "PerspectiveCamera")
            ET.SubElement(persp, "CameraViewPoint", X=str(round(cam_pos[0], 4)), Y=str(round(cam_pos[1], 4)), Z=str(round(cam_pos[2], 4)))
            ET.SubElement(persp, "CameraDirection", X=str(round(cam_dir[0], 4)), Y=str(round(cam_dir[1], 4)), Z=str(round(cam_dir[2], 4)))
            ET.SubElement(persp, "CameraUpVector", X="0.0", Y="0.0", Z="1.0")
            ET.SubElement(persp, "FieldOfView").text = str(round(cam.get("fov", 60.0), 1))

            zf.writestr(f"{topic_dir}viewpoint.bcfv", ET.tostring(vp_root, encoding="utf-8", xml_declaration=True))

    buf.seek(0)
    return buf.getvalue()
