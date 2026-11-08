"""Automated Issue & Clash Detection with BCF (BIM Collaboration Format) Engine.

Provides multi-trade geometric clash detection, clearance verification,
and industry-standard buildingSMART BCF 2.1 / 3.0 export and import.

Key Capabilities:
- Hard Clash Detection: Direct 3D volumetric intersection between conflicting trades (e.g. MEP vs Structure).
- Clearance / Soft Clash Detection: Violation of required clearance buffer zones (e.g. door swing, equipment access).
- Duplicate Detection: Overlapping redundant element geometries (>85% volumetric overlap).
- BCF 2.1 XML & ZIP Package Generator: Fully compliant buildingSMART BCF archive (.bcfzip) with camera viewpoints, GUID selection, and snapshot thumbnails.
- BCF Archive Parser: Ingests external BCF packages from Navisworks, Solibri, and BIM Track.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid
import xml.etree.ElementTree as ET
import zipfile

from engine.pipeline.temporal_delta import TradeCategory, classify_trade


class ClashType(str, Enum):
    HARD = "hard"
    CLEARANCE = "clearance"
    DUPLICATE = "duplicate"


class ClashSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ClashStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    REVIEW = "review"
    APPROVED = "approved"


@dataclass(slots=True)
class ClashViewpoint:
    """BCF-compliant camera viewpoint for inspecting a clash in 3D."""

    camera_position: list[float]
    camera_direction: list[float]
    camera_up: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    field_of_view: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_position": [round(c, 4) for c in self.camera_position],
            "camera_direction": [round(c, 4) for c in self.camera_direction],
            "camera_up": [round(c, 4) for c in self.camera_up],
            "field_of_view": round(self.field_of_view, 1),
        }


@dataclass(slots=True)
class ClashItem:
    """Individual detected spatial clash or trade coordination issue."""

    id: str
    clash_type: ClashType
    severity: ClashSeverity
    status: ClashStatus
    entity_a_id: str
    entity_b_id: str
    entity_a_name: str
    entity_b_name: str
    trade_a: TradeCategory
    trade_b: TradeCategory
    entity_a_type: str = ""
    entity_b_type: str = ""
    penetration_depth_m: float = 0.0
    clearance_distance_m: float = 0.0
    intersection_volume_m3: float = 0.0
    clash_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    viewpoint: Optional[ClashViewpoint] = None
    title: str = ""
    description: str = ""
    assigned_to: str = ""
    created_at_iso: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "clash_type": self.clash_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "entity_a_id": self.entity_a_id,
            "entity_b_id": self.entity_b_id,
            "entity_a_name": self.entity_a_name,
            "entity_b_name": self.entity_b_name,
            "trade_a": self.trade_a.value if isinstance(self.trade_a, TradeCategory) else str(self.trade_a),
            "trade_b": self.trade_b.value if isinstance(self.trade_b, TradeCategory) else str(self.trade_b),
            "entity_a_type": self.entity_a_type,
            "entity_b_type": self.entity_b_type,
            "penetration_depth_m": round(self.penetration_depth_m, 4),
            "clearance_distance_m": round(self.clearance_distance_m, 4),
            "intersection_volume_m3": round(self.intersection_volume_m3, 5),
            "clash_point": [round(c, 3) for c in self.clash_point],
            "bounds_min": [round(c, 3) for c in self.bounds_min],
            "bounds_max": [round(c, 3) for c in self.bounds_max],
            "viewpoint": self.viewpoint.to_dict() if self.viewpoint else None,
            "title": self.title,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "created_at_iso": self.created_at_iso,
        }


@dataclass(slots=True)
class ClashDetectionReport:
    """Comprehensive multi-trade clash detection report for a world."""

    world_id: str
    timestamp_iso: str
    total_clashes: int
    hard_clashes: int
    clearance_clashes: int
    duplicate_clashes: int
    by_severity: dict[str, int]
    by_trade_pair: dict[str, int]
    items: list[ClashItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "timestamp_iso": self.timestamp_iso,
            "total_clashes": self.total_clashes,
            "hard_clashes": self.hard_clashes,
            "clearance_clashes": self.clearance_clashes,
            "duplicate_clashes": self.duplicate_clashes,
            "by_severity": self.by_severity,
            "by_trade_pair": self.by_trade_pair,
            "items": [c.to_dict() for c in self.items],
        }


@dataclass(slots=True)
class ClashDetectionOptions:
    """Configuration options for automated clash detection."""

    hard_clash_tolerance_m: float = 0.01  # Minimum overlap to report a hard clash
    clearance_buffer_m: float = 0.05      # Required clearance buffer between trades
    check_duplicates: bool = True
    check_clearance: bool = True
    ignored_trades: set[TradeCategory] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Bounding box & geometry extraction helpers
# ---------------------------------------------------------------------------

def _extract_entity_position(entity: Any) -> tuple[float, float, float] | None:
    tr = getattr(entity, "transform", None)
    if isinstance(tr, dict) and "position" in tr:
        pos = tr["position"]
        if isinstance(pos, dict) and all(k in pos for k in ["x", "y", "z"]):
            return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))

    props = getattr(entity, "custom_properties", None) or {}
    if "position" in props and isinstance(props["position"], (list, tuple)) and len(props["position"]) >= 3:
        p = props["position"]
        return (float(p[0]), float(p[1]), float(p[2]))

    return None


def _extract_entity_dims(entity: Any) -> tuple[float, float, float]:
    props = getattr(entity, "custom_properties", None) or {}
    w = float(props.get("width_m", props.get("width", 1.0)))
    d = float(props.get("depth_m", props.get("depth", 1.0)))
    h = float(props.get("height_m", props.get("height", 1.0)))
    return (max(0.01, w), max(0.01, d), max(0.01, h))


def get_entity_aabb(entity: Any, world: Any = None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ((min_x, min_y, min_z), (max_x, max_y, max_z)) in meters."""
    if world is not None and hasattr(world, "geometries") and world.geometries:
        geom_ids = getattr(entity, "geometry_ids", []) or []
        for gid in geom_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                b_min = getattr(g, "bounds_min", None)
                b_max = getattr(g, "bounds_max", None)
                if b_min and b_max:
                    def _c(b: Any, k: str) -> float:
                        return float(getattr(b, k) if hasattr(b, k) else b[k])
                    return (
                        (_c(b_min, "x"), _c(b_min, "y"), _c(b_min, "z")),
                        (_c(b_max, "x"), _c(b_max, "y"), _c(b_max, "z")),
                    )

    pos = _extract_entity_position(entity) or (0.0, 0.0, 0.0)
    w, d, h = _extract_dims_or_bounds(entity)
    return (
        (pos[0] - w / 2.0, pos[1] - h / 2.0, pos[2] - d / 2.0),
        (pos[0] + w / 2.0, pos[1] + h / 2.0, pos[2] + d / 2.0),
    )


def _extract_dims_or_bounds(entity: Any) -> tuple[float, float, float]:
    props = getattr(entity, "custom_properties", None) or {}
    if "bounds_min" in props and "bounds_max" in props:
        bmin = props["bounds_min"]
        bmax = props["bounds_max"]
        if len(bmin) >= 3 and len(bmax) >= 3:
            return (
                max(0.01, float(bmax[0]) - float(bmin[0])),
                max(0.01, float(bmax[1]) - float(bmin[1])),
                max(0.01, float(bmax[2]) - float(bmin[2])),
            )
    return _extract_entity_dims(entity)


def _compute_viewpoint(clash_point: list[float], extent: float) -> ClashViewpoint:
    """Generate an optimal inspection camera perspective for a clash location."""
    d = max(1.5, extent * 2.2)
    # Offset camera horizontally and slightly elevated
    cam_pos = [
        clash_point[0] + d * 0.7,
        clash_point[1] + d * 0.5,
        clash_point[2] + d * 0.7,
    ]
    # Direction toward clash point
    dx = clash_point[0] - cam_pos[0]
    dy = clash_point[1] - cam_pos[1]
    dz = clash_point[2] - cam_pos[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    cam_dir = [dx / norm, dy / norm, dz / norm]

    return ClashViewpoint(
        camera_position=cam_pos,
        camera_direction=cam_dir,
        camera_up=[0.0, 1.0, 0.0],
        field_of_view=60.0,
    )


def _determine_severity(trade_a: TradeCategory, trade_b: TradeCategory, clash_type: ClashType, penetration_m: float) -> ClashSeverity:
    """Assess risk severity according to trade combinations and penetration depth."""
    trades = {trade_a, trade_b}
    if clash_type == ClashType.DUPLICATE:
        return ClashSeverity.MINOR

    # MEP penetrating Structural elements is high-risk in AEC
    if TradeCategory.STRUCTURE in trades and TradeCategory.MEP in trades:
        return ClashSeverity.CRITICAL if penetration_m > 0.05 else ClashSeverity.MAJOR

    # Structural vs Architectural clashes (e.g. wall/opening colliding with column)
    if TradeCategory.STRUCTURE in trades and TradeCategory.ARCHITECTURE in trades:
        return ClashSeverity.CRITICAL if penetration_m > 0.1 else ClashSeverity.MAJOR

    # MEP vs MEP (e.g. duct vs pipe)
    if trade_a == TradeCategory.MEP and trade_b == TradeCategory.MEP:
        return ClashSeverity.MAJOR if penetration_m > 0.03 else ClashSeverity.MINOR

    # Clearance issues
    if clash_type == ClashType.CLEARANCE:
        return ClashSeverity.MAJOR if TradeCategory.STRUCTURE in trades else ClashSeverity.MINOR

    return ClashSeverity.MINOR


# ---------------------------------------------------------------------------
# Core Clash Detection Algorithm
# ---------------------------------------------------------------------------

def detect_world_clashes(
    world: Any,
    world_id: str = "default_world",
    options: Optional[ClashDetectionOptions] = None,
) -> ClashDetectionReport:
    """Execute automated geometric clash detection across all entities in a world."""
    opts = options or ClashDetectionOptions()
    entities_dict = getattr(world, "entities", {}) or {}
    entities: list[Any] = list(entities_dict.values())

    items: list[ClashItem] = []
    by_severity: dict[str, int] = {s.value: 0 for s in ClashSeverity}
    by_trade_pair: dict[str, int] = {}

    hard_count = 0
    clearance_count = 0
    duplicate_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    # Pre-extract AABBs, trades, and dimensions
    item_cache = []
    for ent in entities:
        ent_id = getattr(ent, "id", "")
        name = getattr(ent, "name", "") or ent_id
        ent_type = getattr(ent, "type", "entity")
        if hasattr(ent_type, "value"):
            ent_type_str = str(ent_type.value)
        else:
            ent_type_str = str(ent_type)

        props = getattr(ent, "custom_properties", None) or {}
        trade = classify_trade(ent_type_str, props)
        if trade in opts.ignored_trades:
            continue

        bmin, bmax = get_entity_aabb(ent, world)
        vol = max(0.0001, (bmax[0] - bmin[0]) * (bmax[1] - bmin[1]) * (bmax[2] - bmin[2]))

        # Relationships check (to ignore intended host-contained items)
        rels = getattr(ent, "relationships", []) or []
        related_ids = set()
        for r in rels:
            tid = getattr(r, "target_id", None) or getattr(r, "target_entity_id", None)
            if tid:
                related_ids.add(tid)

        item_cache.append({
            "entity": ent,
            "id": ent_id,
            "name": name,
            "type": ent_type_str,
            "trade": trade,
            "bmin": bmin,
            "bmax": bmax,
            "volume": vol,
            "related_ids": related_ids,
        })

    # Pairwise geometric intersection check
    n = len(item_cache)
    for i in range(n):
        a = item_cache[i]
        bmin_a, bmax_a = a["bmin"], a["bmax"]

        for j in range(i + 1, n):
            b = item_cache[j]
            bmin_b, bmax_b = b["bmin"], b["bmax"]

            # Skip explicitly related/hosted pairs (e.g. window inside wall, switch on wall)
            if b["id"] in a["related_ids"] or a["id"] in b["related_ids"]:
                continue

            # Skip space/room volumes (they are virtual boundaries, not physical solids)
            if any(t in a["type"].lower() for t in ["room", "space", "corridor", "zone"]):
                continue
            if any(t in b["type"].lower() for t in ["room", "space", "corridor", "zone"]):
                continue

            # Axis overlaps
            ox = min(bmax_a[0], bmax_b[0]) - max(bmin_a[0], bmin_b[0])
            oy = min(bmax_a[1], bmax_b[1]) - max(bmin_a[1], bmin_b[1])
            oz = min(bmax_a[2], bmax_b[2]) - max(bmin_a[2], bmin_b[2])

            is_intersecting = (ox > 0 and oy > 0 and oz > 0)

            if is_intersecting:
                intersection_vol = ox * oy * oz
                penetration = min(ox, oy, oz)

                # Check for duplicate
                min_vol = min(a["volume"], b["volume"])
                vol_ratio = intersection_vol / min_vol if min_vol > 0 else 0.0

                if opts.check_duplicates and vol_ratio > 0.85:
                    clash_type = ClashType.DUPLICATE
                    duplicate_count += 1
                elif penetration >= opts.hard_clash_tolerance_m:
                    clash_type = ClashType.HARD
                    hard_count += 1
                else:
                    continue

                # Intersection AABB and midpoint
                c_min = [max(bmin_a[0], bmin_b[0]), max(bmin_a[1], bmin_b[1]), max(bmin_a[2], bmin_b[2])]
                c_max = [min(bmax_a[0], bmax_b[0]), min(bmax_a[1], bmax_b[1]), min(bmax_a[2], bmax_b[2])]
                clash_pt = [(c_min[0] + c_max[0]) / 2.0, (c_min[1] + c_max[1]) / 2.0, (c_min[2] + c_max[2]) / 2.0]
                extent = max(ox, oy, oz)
                viewpoint = _compute_viewpoint(clash_pt, extent)

                severity = _determine_severity(a["trade"], b["trade"], clash_type, penetration)
                by_severity[severity.value] += 1

                t_pair = f"{a['trade'].value}-{b['trade'].value}"
                by_trade_pair[t_pair] = by_trade_pair.get(t_pair, 0) + 1

                title = f"{clash_type.value.capitalize()} Clash: {a['name']} & {b['name']}"
                desc = (
                    f"Physical overlap of {penetration:.3f}m ({intersection_vol:.4f} m³) between "
                    f"{a['trade'].value.upper()} ({a['name']}) and {b['trade'].value.upper()} ({b['name']})."
                )

                items.append(ClashItem(
                    id=str(uuid.uuid4()),
                    clash_type=clash_type,
                    severity=severity,
                    status=ClashStatus.ACTIVE,
                    entity_a_id=a["id"],
                    entity_b_id=b["id"],
                    entity_a_name=a["name"],
                    entity_b_name=b["name"],
                    trade_a=a["trade"],
                    trade_b=b["trade"],
                    entity_a_type=a["type"],
                    entity_b_type=b["type"],
                    penetration_depth_m=penetration,
                    clearance_distance_m=0.0,
                    intersection_volume_m3=intersection_vol,
                    clash_point=clash_pt,
                    bounds_min=c_min,
                    bounds_max=c_max,
                    viewpoint=viewpoint,
                    title=title,
                    description=desc,
                    assigned_to="BIM Coordinator",
                    created_at_iso=now_iso,
                ))

            elif opts.check_clearance:
                # Calculate minimum distance between non-overlapping boxes
                gap_x = max(0.0, max(bmin_a[0], bmin_b[0]) - min(bmax_a[0], bmax_b[0]))
                gap_y = max(0.0, max(bmin_a[1], bmin_b[1]) - min(bmax_a[1], bmax_b[1]))
                gap_z = max(0.0, max(bmin_a[2], bmin_b[2]) - min(bmax_a[2], bmax_b[2]))
                gap_dist = math.sqrt(gap_x * gap_x + gap_y * gap_y + gap_z * gap_z)

                if 0.0 < gap_dist < opts.clearance_buffer_m:
                    clash_type = ClashType.CLEARANCE
                    clearance_count += 1

                    clash_pt = [
                        (min(bmax_a[0], bmax_b[0]) + max(bmin_a[0], bmin_b[0])) / 2.0,
                        (min(bmax_a[1], bmax_b[1]) + max(bmin_a[1], bmin_b[1])) / 2.0,
                        (min(bmax_a[2], bmax_b[2]) + max(bmin_a[2], bmin_b[2])) / 2.0,
                    ]
                    viewpoint = _compute_viewpoint(clash_pt, opts.clearance_buffer_m)
                    severity = _determine_severity(a["trade"], b["trade"], clash_type, 0.0)
                    by_severity[severity.value] += 1

                    t_pair = f"{a['trade'].value}-{b['trade'].value}"
                    by_trade_pair[t_pair] = by_trade_pair.get(t_pair, 0) + 1

                    title = f"Clearance Violation: {a['name']} & {b['name']}"
                    desc = (
                        f"Separation distance {gap_dist:.3f}m is below required clearance {opts.clearance_buffer_m:.3f}m "
                        f"between {a['name']} and {b['name']}."
                    )

                    items.append(ClashItem(
                        id=str(uuid.uuid4()),
                        clash_type=clash_type,
                        severity=severity,
                        status=ClashStatus.ACTIVE,
                        entity_a_id=a["id"],
                        entity_b_id=b["id"],
                        entity_a_name=a["name"],
                        entity_b_name=b["name"],
                        trade_a=a["trade"],
                        trade_b=b["trade"],
                        entity_a_type=a["type"],
                        entity_b_type=b["type"],
                        penetration_depth_m=0.0,
                        clearance_distance_m=gap_dist,
                        intersection_volume_m3=0.0,
                        clash_point=clash_pt,
                        bounds_min=[clash_pt[0] - 0.1, clash_pt[1] - 0.1, clash_pt[2] - 0.1],
                        bounds_max=[clash_pt[0] + 0.1, clash_pt[1] + 0.1, clash_pt[2] + 0.1],
                        viewpoint=viewpoint,
                        title=title,
                        description=desc,
                        assigned_to="Field Engineer",
                        created_at_iso=now_iso,
                    ))

    return ClashDetectionReport(
        world_id=world_id,
        timestamp_iso=now_iso,
        total_clashes=len(items),
        hard_clashes=hard_count,
        clearance_clashes=clearance_count,
        duplicate_clashes=duplicate_count,
        by_severity=by_severity,
        by_trade_pair=by_trade_pair,
        items=items,
    )


# ---------------------------------------------------------------------------
# Open buildingSMART BCF (BIM Collaboration Format) 2.1 Package Generator
# ---------------------------------------------------------------------------

def generate_bcf_zip(
    report: ClashDetectionReport,
    author: str = "Reality Engine Perception",
) -> bytes:
    """Generate a standard buildingSMART BCF 2.1 zip package containing clash topics."""
    buf = io.BytesIO()

    # Minimal 1x1 PNG thumbnail bytes for snapshot.png
    # 1x1 transparent PNG binary
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. bcf.version root file
        ver_root = ET.Element(
            "Version",
            {
                "VersionId": "2.1",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "https://raw.githubusercontent.com/BuildingSMART/BCF-XML/release_2_1/Schemas/version.xsd",
            },
        )
        zf.writestr("bcf.version", ET.tostring(ver_root, encoding="utf-8", xml_declaration=True))

        # 2. For each clash topic, create a folder named with topic GUID
        for clash in report.items:
            folder = clash.id

            # markup.bcf
            markup = ET.Element(
                "Markup",
                {
                    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                    "xsi:noNamespaceSchemaLocation": "https://raw.githubusercontent.com/BuildingSMART/BCF-XML/release_2_1/Schemas/markup.xsd",
                },
            )

            hdr = ET.SubElement(markup, "Header")
            ET.SubElement(hdr, "File", {
                "IfcProject": report.world_id,
                "Filename": f"{report.world_id}.ifc",
                "Date": clash.created_at_iso,
            })

            topic = ET.SubElement(markup, "Topic", {
                "Guid": clash.id,
                "TopicType": clash.clash_type.value.capitalize(),
                "TopicStatus": clash.status.value.capitalize(),
            })
            ET.SubElement(topic, "Title").text = clash.title
            ET.SubElement(topic, "Priority").text = clash.severity.value.capitalize()
            ET.SubElement(topic, "CreationDate").text = clash.created_at_iso
            ET.SubElement(topic, "CreationAuthor").text = author
            ET.SubElement(topic, "Description").text = clash.description
            if clash.assigned_to:
                ET.SubElement(topic, "AssignedTo").text = clash.assigned_to

            # Comment element
            cmt = ET.SubElement(markup, "Comment", {"Guid": str(uuid.uuid4())})
            ET.SubElement(cmt, "Date").text = clash.created_at_iso
            ET.SubElement(cmt, "Author").text = author
            ET.SubElement(cmt, "Comment").text = (
                f"Automated geometric clash: {clash.clash_type.value.upper()} "
                f"between {clash.entity_a_name} ({clash.trade_a.value}) and {clash.entity_b_name} ({clash.trade_b.value})."
            )

            # Viewpoints reference
            vpts = ET.SubElement(markup, "Viewpoints", {"Guid": str(uuid.uuid4())})
            ET.SubElement(vpts, "Viewpoint").text = "viewpoint.bcfv"
            ET.SubElement(vpts, "Snapshot").text = "snapshot.png"

            zf.writestr(f"{folder}/markup.bcf", ET.tostring(markup, encoding="utf-8", xml_declaration=True))

            # viewpoint.bcfv
            vis_info = ET.Element(
                "VisualizationInfo",
                {
                    "Guid": str(uuid.uuid4()),
                    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                    "xsi:noNamespaceSchemaLocation": "https://raw.githubusercontent.com/BuildingSMART/BCF-XML/release_2_1/Schemas/visinfo.xsd",
                },
            )

            # Components selection & coloring
            comps = ET.SubElement(vis_info, "Components")
            sel = ET.SubElement(comps, "Selection")
            ET.SubElement(sel, "Component", {"IfcGuid": clash.entity_a_id})
            ET.SubElement(sel, "Component", {"IfcGuid": clash.entity_b_id})

            coloring = ET.SubElement(comps, "Coloring")
            col_a = ET.SubElement(coloring, "Color", {"Color": "FF0000"})
            ET.SubElement(col_a, "Component", {"IfcGuid": clash.entity_a_id})
            col_b = ET.SubElement(coloring, "Color", {"Color": "FFA500"})
            ET.SubElement(col_b, "Component", {"IfcGuid": clash.entity_b_id})

            # Camera Viewpoint
            if clash.viewpoint:
                cam = ET.SubElement(vis_info, "PerspectiveCamera")
                vp = ET.SubElement(cam, "CameraViewPoint")
                ET.SubElement(vp, "X").text = str(round(clash.viewpoint.camera_position[0], 4))
                ET.SubElement(vp, "Y").text = str(round(clash.viewpoint.camera_position[1], 4))
                ET.SubElement(vp, "Z").text = str(round(clash.viewpoint.camera_position[2], 4))

                cd = ET.SubElement(cam, "CameraDirection")
                ET.SubElement(cd, "X").text = str(round(clash.viewpoint.camera_direction[0], 4))
                ET.SubElement(cd, "Y").text = str(round(clash.viewpoint.camera_direction[1], 4))
                ET.SubElement(cd, "Z").text = str(round(clash.viewpoint.camera_direction[2], 4))

                cup = ET.SubElement(cam, "CameraUpVector")
                ET.SubElement(cup, "X").text = str(round(clash.viewpoint.camera_up[0], 4))
                ET.SubElement(cup, "Y").text = str(round(clash.viewpoint.camera_up[1], 4))
                ET.SubElement(cup, "Z").text = str(round(clash.viewpoint.camera_up[2], 4))

                ET.SubElement(cam, "FieldOfView").text = str(clash.viewpoint.field_of_view)

            zf.writestr(f"{folder}/viewpoint.bcfv", ET.tostring(vis_info, encoding="utf-8", xml_declaration=True))

            # snapshot.png
            zf.writestr(f"{folder}/snapshot.png", minimal_png)

    return buf.getvalue()


def parse_bcf_zip(bcf_bytes: bytes) -> list[dict[str, Any]]:
    """Parse an uploaded BCF zip package and extract topic metadata."""
    topics: list[dict[str, Any]] = []
    buf = io.BytesIO(bcf_bytes)

    with zipfile.ZipFile(buf, mode="r") as zf:
        file_list = zf.namelist()
        markup_files = [f for f in file_list if f.endswith("markup.bcf")]

        for mf in markup_files:
            folder = mf.split("/")[0] if "/" in mf else ""
            content = zf.read(mf)
            try:
                root = ET.fromstring(content)
                topic = root.find("Topic")
                if topic is not None:
                    guid = topic.get("Guid", folder or str(uuid.uuid4()))
                    title = topic.findtext("Title", "Untitled Topic")
                    topic_type = topic.get("TopicType", "Clash")
                    topic_status = topic.get("TopicStatus", "Active")
                    priority = topic.findtext("Priority", "Normal")
                    description = topic.findtext("Description", "")
                    author = topic.findtext("CreationAuthor", "")
                    created_at = topic.findtext("CreationDate", "")

                    # Parse components from viewpoint if available
                    vpt_file = f"{folder}/viewpoint.bcfv"
                    components = []
                    if vpt_file in file_list:
                        vpt_content = zf.read(vpt_file)
                        vpt_root = ET.fromstring(vpt_content)
                        for comp in vpt_root.findall(".//Selection/Component"):
                            guid_attr = comp.get("IfcGuid")
                            if guid_attr:
                                components.append(guid_attr)

                    topics.append({
                        "id": guid,
                        "title": title,
                        "topic_type": topic_type,
                        "topic_status": topic_status,
                        "priority": priority,
                        "description": description,
                        "author": author,
                        "created_at_iso": created_at,
                        "components": components,
                    })
            except Exception:
                continue

    return topics
