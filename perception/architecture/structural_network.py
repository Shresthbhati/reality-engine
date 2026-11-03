"""Structural Network & Load-Bearing Understanding Engine (Wave 2, Block 12).

Analyzes detected columns, beams, structural walls, and floor/ceiling slabs:
- Constructs a structural connectivity graph linking columns, beams, and slabs.
- Infers load-bearing relationships (SUPPORTS, RESTS_ON).
- Evaluates structural bay spans (column-to-column and beam spans).
- Distinguishes visually inferred structural roles from verified engineering facts.

MANDATORY CONSTITUTIONAL SAFETY RULE:
Visual evidence from photographs/scans cannot prove structural capacity or engineering
safety. All inferred structural roles carry:
  provenance = Provenance.INFERRED
  custom_properties["is_visually_inferred_structural_role"] = True
  custom_properties["engineering_safety_certified"] = False
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance, Uncertainty
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


class StructuralRole(str, Enum):
    """Visually inferred structural role of an architectural element."""
    VERTICAL_SUPPORT = "vertical_support"      # Columns, load-bearing piers
    HORIZONTAL_SPAN = "horizontal_span"        # Beams, lintels, joists
    BEARING_WALL = "bearing_wall"              # Solid walls carrying vertical load
    PARTITION_NON_BEARING = "partition_wall"   # Thin internal partition wall
    DIAPHRAGM_SLAB = "diaphragm_slab"          # Floor/ceiling horizontal slab
    FOUNDATION_BASE = "foundation_base"        # Ground-contact base or footing
    UNKNOWN = "unknown"


@dataclass
class StructuralBay:
    """A structural bay defined by supporting columns/posts."""
    bay_id: str
    column_ids: List[str]
    span_x_m: float
    span_z_m: float
    area_m2: float
    supported_beam_ids: List[str] = field(default_factory=list)


@dataclass
class StructuralAnalysisReport:
    """Report detailing inferred structural layout and geometry checks."""
    column_count: int
    beam_count: int
    bearing_wall_count: int
    slab_count: int
    bays: List[StructuralBay] = field(default_factory=list)
    relationships_added: int = 0
    max_unsupported_span_m: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_count": self.column_count,
            "beam_count": self.beam_count,
            "bearing_wall_count": self.bearing_wall_count,
            "slab_count": self.slab_count,
            "bay_count": len(self.bays),
            "relationships_added": self.relationships_added,
            "max_unsupported_span_m": round(self.max_unsupported_span_m, 3),
            "warnings": list(self.warnings),
            "disclaimer": "Visually inferred architectural analysis only. Not certified for structural safety or load capacity.",
        }


def _get_entity_center(world: WorldIR, entity: Entity) -> Optional[np.ndarray]:
    """Calculate world-space center of an entity from its geometry or custom properties."""
    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min and g.bounds_max:
                    return np.array([
                        (g.bounds_min.x + g.bounds_max.x) / 2.0,
                        (g.bounds_min.y + g.bounds_max.y) / 2.0,
                        (g.bounds_min.z + g.bounds_max.z) / 2.0,
                    ])
    # Fallback to custom properties
    pos = entity.custom_properties.get("position")
    if pos and isinstance(pos, dict):
        return np.array([float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))])
    return None


def _get_entity_bounds(world: WorldIR, entity: Entity) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get min and max bounds for an entity."""
    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min and g.bounds_max:
                    return (
                        np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z]),
                        np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]),
                    )
    return None


def analyze_structural_system(world: WorldIR) -> StructuralAnalysisReport:
    """Analyze structural elements in WorldIR, build connectivity graph, and infer load paths."""
    columns: List[Entity] = []
    beams: List[Entity] = []
    walls: List[Entity] = []
    floors: List[Entity] = []
    ceilings: List[Entity] = []

    # 1. Categorize structural entities
    for ent in world.entities.values():
        e_type = ent.type
        c_class = str(ent.custom_properties.get("entity_class", "")).lower()

        if e_type == EntityType.COLUMN or "column" in c_class:
            columns.append(ent)
        elif e_type == EntityType.BEAM or "beam" in c_class:
            beams.append(ent)
        elif e_type == EntityType.WALL or "wall" in c_class:
            walls.append(ent)
        elif e_type == EntityType.FLOOR or "floor" in c_class:
            floors.append(ent)
        elif e_type == EntityType.CEILING or "ceiling" in c_class:
            ceilings.append(ent)

    # 2. Tag entities with visually inferred structural roles
    for col in columns:
        col.custom_properties["structural_role"] = StructuralRole.VERTICAL_SUPPORT.value
        col.custom_properties["is_visually_inferred_structural_role"] = True
        col.custom_properties["engineering_safety_certified"] = False

    for beam in beams:
        beam.custom_properties["structural_role"] = StructuralRole.HORIZONTAL_SPAN.value
        beam.custom_properties["is_visually_inferred_structural_role"] = True
        beam.custom_properties["engineering_safety_certified"] = False

    # Discriminate bearing walls vs thin partition walls based on thickness
    bearing_walls: List[Entity] = []
    for wall in walls:
        thick = wall.custom_properties.get("thickness_m", 0.10)
        # Standard architectural heuristic: walls >= 0.20m (8 in) are likely load-bearing / external
        if thick >= 0.18:
            wall.custom_properties["structural_role"] = StructuralRole.BEARING_WALL.value
            bearing_walls.append(wall)
        else:
            wall.custom_properties["structural_role"] = StructuralRole.PARTITION_NON_BEARING.value
        wall.custom_properties["is_visually_inferred_structural_role"] = True
        wall.custom_properties["engineering_safety_certified"] = False

    for fl in floors:
        fl.custom_properties["structural_role"] = StructuralRole.DIAPHRAGM_SLAB.value
        fl.custom_properties["is_visually_inferred_structural_role"] = True

    # 3. Infer load path relationships (SUPPORTS / RESTS_ON)
    relationships_added = 0
    warnings: List[str] = []

    # Beams supported by Columns
    for beam in beams:
        b_bounds = _get_entity_bounds(world, beam)
        if not b_bounds:
            continue
        b_min, b_max = b_bounds

        supporting_columns: List[Entity] = []
        for col in columns:
            c_bounds = _get_entity_bounds(world, col)
            if not c_bounds:
                continue
            c_min, c_max = c_bounds

            # Check if column top touches or is immediately beneath beam bottom (tolerance 0.15m)
            vertical_contact = abs(c_max[1] - b_min[1]) <= 0.15 or (c_max[1] >= b_min[1] and c_min[1] <= b_min[1])
            # Check horizontal overlap in XZ
            horizontal_overlap = not (c_max[0] < b_min[0] or c_min[0] > b_max[0] or c_max[2] < b_min[2] or c_min[2] > b_max[2])

            if vertical_contact and horizontal_overlap:
                supporting_columns.append(col)
                # Column SUPPORTS Beam
                col.relationships.append(Relationship(
                    kind=RelationshipKind.SUPPORTS,
                    target_id=beam.id,
                    confidence=0.85,
                    provenance=Provenance.INFERRED,
                    metadata={"load_path": "vertical_axial_support"},
                ))
                # Beam RESTS_ON Column
                beam.relationships.append(Relationship(
                    kind=RelationshipKind.RESTS_ON,
                    target_id=col.id,
                    confidence=0.85,
                    provenance=Provenance.INFERRED,
                    metadata={"load_path": "bearing_seat"},
                ))
                relationships_added += 2

        # Check beam span length
        beam_length = max(b_max[0] - b_min[0], b_max[2] - b_min[2])
        if beam_length > 8.0 and len(supporting_columns) < 2:
            warnings.append(f"Beam {beam.id} has long span ({beam_length:.2f}m) with fewer than 2 detected supporting columns.")

    # Slabs supported by Columns and Bearing Walls
    for slab in ceilings + floors:
        s_bounds = _get_entity_bounds(world, slab)
        if not s_bounds:
            continue
        s_min, s_max = s_bounds

        for col in columns:
            c_bounds = _get_entity_bounds(world, col)
            if not c_bounds:
                continue
            c_min, c_max = c_bounds
            if abs(c_max[1] - s_min[1]) <= 0.15:
                # Column touches slab
                col.relationships.append(Relationship(
                    kind=RelationshipKind.SUPPORTS,
                    target_id=slab.id,
                    confidence=0.80,
                    provenance=Provenance.INFERRED,
                    metadata={"load_path": "slab_support"},
                ))
                relationships_added += 1

    # 4. Compute structural bays from column grid
    bays: List[StructuralBay] = []
    max_span = 0.0

    if len(columns) >= 2:
        col_centers = [(col.id, _get_entity_center(world, col)) for col in columns]
        valid_cols = [(cid, c) for cid, c in col_centers if c is not None]

        # Calculate pairwise horizontal spans
        for i, (cid1, c1) in enumerate(valid_cols):
            for j, (cid2, c2) in enumerate(valid_cols):
                if j <= i:
                    continue
                dx = abs(c1[0] - c2[0])
                dz = abs(c1[2] - c2[2])
                dist_xz = math.sqrt(dx * dx + dz * dz)
                if dist_xz > max_span:
                    max_span = dist_xz

                # If columns are roughly orthogonal neighbors (within 3m to 10m)
                if 2.5 <= dist_xz <= 12.0 and (dx <= 0.3 or dz <= 0.3):
                    bays.append(StructuralBay(
                        bay_id=f"bay-{cid1[:6]}-{cid2[:6]}",
                        column_ids=[cid1, cid2],
                        span_x_m=dx,
                        span_z_m=dz,
                        area_m2=dx * dz if dx > 0 and dz > 0 else dist_xz * 0.5,
                    ))

    report = StructuralAnalysisReport(
        column_count=len(columns),
        beam_count=len(beams),
        bearing_wall_count=len(bearing_walls),
        slab_count=len(floors) + len(ceilings),
        bays=bays,
        relationships_added=relationships_added,
        max_unsupported_span_m=max_span,
        warnings=warnings,
    )

    # Attach to WorldIR metadata
    if world.metadata is None:
        world.metadata = {}
    world.metadata["structural_analysis"] = report.to_dict()

    return report
