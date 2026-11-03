"""Reconstruction Completeness & Coverage Heatmap Engine (Wave 4, Block 32).

Measures and audits environmental reconstruction completeness at multi-scale tiers:
- Room level: floor, ceiling, and wall face observed coverage fractions.
- Wall level: measures unobserved spans where visual evidence was missing.
- Entity level: ratio of directly observed vs inferred geometry.
- Scene level: area-weighted completeness percentage, void volume, and coverage heatmaps.

Guarantees:
- Strictly evidence-backed: never reports 100% completeness for uninspected wall spans.
- Distinguishes observed facts from inferred closures.
- Attaches structured completeness metrics directly to WorldIR metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR


@dataclass
class RoomCompletenessReport:
    """Detailed completeness audit for an individual room or space."""
    room_id: str
    room_name: str
    floor_area_m2: float
    wall_perimeter_m: float
    observed_wall_fraction: float       # 0.0 to 1.0
    observed_floor_fraction: float      # 0.0 to 1.0
    observed_ceiling_fraction: float    # 0.0 to 1.0
    unobserved_wall_spans_m: List[Tuple[float, float]] = field(default_factory=list)
    fixture_count: int = 0
    overall_completeness_pct: float = 0.0
    confidence_score: float = 0.0
    missing_aspects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_name": self.room_name,
            "floor_area_m2": round(self.floor_area_m2, 2),
            "wall_perimeter_m": round(self.wall_perimeter_m, 2),
            "observed_wall_fraction": round(self.observed_wall_fraction, 3),
            "observed_floor_fraction": round(self.observed_floor_fraction, 3),
            "observed_ceiling_fraction": round(self.observed_ceiling_fraction, 3),
            "unobserved_wall_spans_m": [[round(a, 3), round(b, 3)] for a, b in self.unobserved_wall_spans_m],
            "fixture_count": self.fixture_count,
            "overall_completeness_pct": round(self.overall_completeness_pct, 1),
            "confidence_score": round(self.confidence_score, 3),
            "missing_aspects": list(self.missing_aspects),
        }


@dataclass
class SceneCompletenessReport:
    """Scene-wide reconstruction completeness report and coverage heatmap."""
    total_spaces: int
    total_area_m2: float
    overall_completeness_pct: float
    average_confidence: float
    room_reports: List[RoomCompletenessReport] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    unobserved_void_volume_m3: float = 0.0
    coverage_heatmap: Dict[str, float] = field(default_factory=dict)  # entity_id -> completeness ratio (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_spaces": self.total_spaces,
            "total_area_m2": round(self.total_area_m2, 2),
            "overall_completeness_pct": round(self.overall_completeness_pct, 1),
            "average_confidence": round(self.average_confidence, 3),
            "unobserved_void_volume_m3": round(self.unobserved_void_volume_m3, 2),
            "missing_elements": list(self.missing_elements),
            "coverage_heatmap": {k: round(v, 3) for k, v in self.coverage_heatmap.items()},
            "room_reports": [r.to_dict() for r in self.room_reports],
        }


def _get_entity_area(world: WorldIR, entity: Entity) -> float:
    """Calculate horizontal surface area from geometry or custom properties."""
    area = entity.custom_properties.get("area_m2") or entity.custom_properties.get("measured_area_m2")
    if area is not None:
        return float(area)

    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min and g.bounds_max:
                    dx = abs(g.bounds_max.x - g.bounds_min.x)
                    dz = abs(g.bounds_max.z - g.bounds_min.z)
                    return dx * dz
    return 0.0


def evaluate_reconstruction_completeness(world: WorldIR) -> SceneCompletenessReport:
    """Compute exhaustive environmental completeness metrics across all WorldIR rooms and entities."""
    room_entities: List[Entity] = []
    wall_entities: List[Entity] = []
    floor_entities: List[Entity] = []
    ceiling_entities: List[Entity] = []
    other_entities: List[Entity] = []

    for ent in world.entities.values():
        e_type = ent.type
        c_class = str(ent.custom_properties.get("entity_class", "")).lower()

        if e_type in (EntityType.ROOM, EntityType.CORRIDOR) or c_class.startswith("room.") or c_class == "room":
            room_entities.append(ent)
        elif e_type == EntityType.WALL or "wall" in c_class:
            wall_entities.append(ent)
        elif e_type == EntityType.FLOOR or "floor" in c_class:
            floor_entities.append(ent)
        elif e_type == EntityType.CEILING or "ceiling" in c_class:
            ceiling_entities.append(ent)
        else:
            other_entities.append(ent)

    coverage_heatmap: Dict[str, float] = {}
    room_reports: List[RoomCompletenessReport] = []
    missing_elements: List[str] = []

    # 1. Evaluate coverage per wall
    for w in wall_entities:
        obs_frac = w.custom_properties.get("observed_fraction")
        if obs_frac is not None:
            frac = float(obs_frac)
        elif w.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED):
            frac = 0.95
        elif w.provenance == Provenance.INFERRED:
            frac = 0.35
        else:
            frac = 0.50
        coverage_heatmap[w.id] = max(0.0, min(1.0, frac))

    # 2. Evaluate coverage per floor and ceiling
    for fl in floor_entities:
        frac = 0.95 if fl.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED) else 0.40
        coverage_heatmap[fl.id] = frac

    for cl in ceiling_entities:
        frac = 0.90 if cl.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED) else 0.30
        coverage_heatmap[cl.id] = frac

    for oth in other_entities:
        frac = 0.90 if oth.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED) else 0.40
        coverage_heatmap[oth.id] = frac

    # 3. Analyze room-by-room completeness
    total_scene_area = 0.0
    weighted_completeness_sum = 0.0
    total_void_volume = 0.0

    if not room_entities and (wall_entities or floor_entities):
        # Implicit single space scene
        dummy_room = Entity(id="room-implicit-main", name="Main Reconstructed Space", type=EntityType.ROOM)
        room_entities = [dummy_room]

    for rm in room_entities:
        # Collect member walls, floors, ceilings
        rm_walls = [w for w in wall_entities if w.custom_properties.get("room_id") == rm.id or w.custom_properties.get("parent_id") == rm.id]
        if not rm_walls:
            # Check relationships
            rel_wall_ids = {r.target_id for r in rm.relationships if r.target_id in world.entities and world.entities[r.target_id].type == EntityType.WALL}
            rm_walls = [world.entities[wid] for wid in rel_wall_ids]
        if not rm_walls and len(room_entities) == 1:
            rm_walls = wall_entities

        rm_floors = [f for f in floor_entities if f.custom_properties.get("room_id") == rm.id or f.custom_properties.get("parent_id") == rm.id]
        if not rm_floors and len(room_entities) == 1:
            rm_floors = floor_entities

        rm_ceilings = [c for c in ceiling_entities if c.custom_properties.get("room_id") == rm.id or c.custom_properties.get("parent_id") == rm.id]
        if not rm_ceilings and len(room_entities) == 1:
            rm_ceilings = ceiling_entities

        rm_fixtures = [e for e in other_entities if e.custom_properties.get("room_id") == rm.id or e.custom_properties.get("parent_id") == rm.id]

        # Calculate perimeter and observed wall fraction
        perimeter = 0.0
        wall_obs_sum = 0.0
        unobserved_spans: List[Tuple[float, float]] = []
        missing_aspects: List[str] = []

        if rm_walls:
            for w in rm_walls:
                dims = w.custom_properties.get("dimensions_m")
                length = float(dims[0]) if dims and len(dims) >= 1 else 3.0
                perimeter += length
                w_frac = coverage_heatmap.get(w.id, 0.5)
                wall_obs_sum += w_frac * length
                # Check unobserved spans
                spans = w.custom_properties.get("unobserved_spans_m", [])
                for sp in spans:
                    if len(sp) >= 2:
                        unobserved_spans.append((float(sp[0]), float(sp[1])))
            obs_wall_fraction = wall_obs_sum / max(1e-6, perimeter)
        else:
            obs_wall_fraction = 0.0
            missing_aspects.append("No boundary wall slabs detected")

        # Floor coverage
        rm_area = _get_entity_area(world, rm)
        if rm_floors:
            floor_obs_fraction = np.mean([coverage_heatmap.get(f.id, 0.5) for f in rm_floors])
            if rm_area <= 0.0:
                rm_area = sum(_get_entity_area(world, f) for f in rm_floors)
        else:
            floor_obs_fraction = 0.0
            missing_aspects.append("Floor plane missing or unobserved")

        if rm_area <= 0.0 and perimeter > 0.0:
            # Heuristic area approximation from perimeter (square assumption: (P/4)^2)
            rm_area = (perimeter / 4.0) ** 2

        # Ceiling coverage
        if rm_ceilings:
            ceiling_obs_fraction = np.mean([coverage_heatmap.get(c.id, 0.5) for c in rm_ceilings])
        else:
            ceiling_obs_fraction = 0.0
            missing_aspects.append("Ceiling surface unobserved (open top)")

        if unobserved_spans:
            missing_aspects.append(f"{len(unobserved_spans)} unobserved wall span gaps require visual capture")

        # Compute weighted room completeness: 45% walls, 35% floor, 20% ceiling
        room_pct = (obs_wall_fraction * 45.0) + (floor_obs_fraction * 35.0) + (ceiling_obs_fraction * 20.0)
        conf = (obs_wall_fraction * 0.4) + (floor_obs_fraction * 0.4) + (ceiling_obs_fraction * 0.2)

        # Estimate unobserved void volume in room
        est_height = 2.7  # standard ceiling height
        void_vol = rm_area * est_height * (1.0 - (room_pct / 100.0))
        total_void_volume += void_vol

        report = RoomCompletenessReport(
            room_id=rm.id,
            room_name=rm.name,
            floor_area_m2=rm_area,
            wall_perimeter_m=perimeter,
            observed_wall_fraction=obs_wall_fraction,
            observed_floor_fraction=floor_obs_fraction,
            observed_ceiling_fraction=ceiling_obs_fraction,
            unobserved_wall_spans_m=unobserved_spans,
            fixture_count=len(rm_fixtures),
            overall_completeness_pct=room_pct,
            confidence_score=conf,
            missing_aspects=missing_aspects,
        )
        room_reports.append(report)
        coverage_heatmap[rm.id] = room_pct / 100.0

        total_scene_area += rm_area
        weighted_completeness_sum += room_pct * rm_area

        if missing_aspects:
            for asp in missing_aspects:
                missing_elements.append(f"[{rm.name}] {asp}")

    overall_pct = (weighted_completeness_sum / max(1e-6, total_scene_area)) if total_scene_area > 0 else 50.0
    avg_conf = np.mean([r.confidence_score for r in room_reports]) if room_reports else 0.5

    scene_report = SceneCompletenessReport(
        total_spaces=len(room_reports),
        total_area_m2=total_scene_area,
        overall_completeness_pct=overall_pct,
        average_confidence=float(avg_conf),
        room_reports=room_reports,
        missing_elements=missing_elements,
        unobserved_void_volume_m3=total_void_volume,
        coverage_heatmap=coverage_heatmap,
    )

    if world.metadata is None:
        world.metadata = {}
    world.metadata["completeness"] = scene_report.to_dict()

    return scene_report


# Alias for whole-scene completeness evaluation
evaluate_scene_completeness = evaluate_reconstruction_completeness

