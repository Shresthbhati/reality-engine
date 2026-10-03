"""Shared input of the box-based world compilers (GIS, ROS/Gazebo, Habitat, SUMO, Unreal, Godot).

Same discipline as exporters/cityjson, gltf and blender: only entities with a transform POSITION and a BOX/PLANE
geometry with REAL bounds are exported; degenerate axes floor to a minimum dimension; everything else is skipped WITH
A REASON, never given an invented size. Rotation is not applied (no WorldIR producer of these geometries writes one;
the other exporters make the same choice).

WorldIR is Z-up, right-handed, in the world's own metres. Each writer converts to its target's convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from exporters.report import ExportReport, content_hash
from world_ir.schema_v1 import GeometryType

_EXPORTABLE = frozenset({GeometryType.BOX, GeometryType.PLANE})
MIN_DIMENSION_M = 0.01


@dataclass(frozen=True)
class PlacedBox:
    id: str
    name: str
    type: str
    provenance: str
    confidence: float
    center: Tuple[float, float, float]   # world metres, Z-up
    size: Tuple[float, float, float]     # full extents along x, y, z (metres)

    @property
    def min(self) -> Tuple[float, float, float]:
        return tuple(c - s / 2.0 for c, s in zip(self.center, self.size))

    @property
    def max(self) -> Tuple[float, float, float]:
        return tuple(c + s / 2.0 for c, s in zip(self.center, self.size))


def collect_boxes(world) -> Tuple[List[PlacedBox], List[str], List[str]]:
    """(boxes, skipped ids, skip reasons) in the world's entity order -- deterministic."""
    boxes: List[PlacedBox] = []
    skipped: List[str] = []
    reasons: List[str] = []
    for entity in world.entities.values():
        pos = (entity.transform or {}).get("position") if isinstance(entity.transform, dict) else None
        if not pos:
            skipped.append(entity.id)
            reasons.append("no transform position")
            continue
        geom = next((world.geometries[g] for g in entity.geometry_ids
                     if g in world.geometries and world.geometries[g].type in _EXPORTABLE), None)
        if geom is None:
            skipped.append(entity.id)
            reasons.append("no BOX/PLANE geometry with real bounds attached")
            continue
        if geom.bounds_min is None or geom.bounds_max is None:
            skipped.append(entity.id)
            reasons.append("geometry has no real bounds_min/bounds_max")
            continue
        size = tuple(max(MIN_DIMENSION_M, hi - lo) for lo, hi in (
            (geom.bounds_min.x, geom.bounds_max.x), (geom.bounds_min.y, geom.bounds_max.y),
            (geom.bounds_min.z, geom.bounds_max.z)))
        center = (pos["x"] + (geom.bounds_min.x + geom.bounds_max.x) / 2.0,
                  pos["y"] + (geom.bounds_min.y + geom.bounds_max.y) / 2.0,
                  pos["z"] + (geom.bounds_min.z + geom.bounds_max.z) / 2.0)
        boxes.append(PlacedBox(
            id=entity.id, name=entity.name or entity.id, type=getattr(entity.type, "value", str(entity.type)),
            provenance=entity.provenance.value, confidence=entity.confidence, center=center, size=size))
    return boxes, skipped, reasons


def make_report(fmt_name: str, world, content, boxes, skipped, reasons) -> ExportReport:
    return ExportReport(
        format=fmt_name, world_id=world.id, world_version=getattr(world, "version", 0) or 0,
        entities_exported=tuple(b.id for b in boxes), entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons), content_hash=content_hash(content))


def fmt(v: float) -> str:
    """Stable, compact number formatting (byte-identical output for the same world)."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s
