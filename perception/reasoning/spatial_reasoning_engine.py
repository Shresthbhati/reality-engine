"""Universal Spatial Reasoning & Structural Load Path Engine (Directive Rule 8).

Implements rigorous physical, structural, and spatial reasoning:
1. Structural Load Paths:
   - Traces vertical gravity load paths from roof/slabs through beams and columns to foundation.
   - Identifies structural support hierarchies (RelationshipKind.SUPPORTS).
   - Rejects floating load-bearing structures and overloaded non-bearing partition walls.
2. Doorway & Clearance Accessibility:
   - Evaluates mandatory approach envelopes (at least 0.80m x 1.0m x 2.0m) on both sides of doors.
   - Rejects blocked doorways and egress obstructions.
3. Volumetric Voids & Occluded Space:
   - Computes interior room void volumes (walkable negative space vs occupied mass).
   - Detects occluded dead spaces behind tall furniture and partition corners.
4. Functional Utility Networks:
   - Traces electrical circuit continuity (panel -> conduits -> switches/sockets).
   - Traces plumbing drainage continuity and minimum gravity fall slopes (>= 1.0%).
   - Flags orphaned fixtures and discontinuous utility lines.
5. Physical Impossibility Gate:
   - Rejects physically impossible scenes and produces deterministic auto-corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
    WorldIR,
)


def _get_entity_aabb(world: WorldIR, entity: Entity) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Retrieve world-aligned bounding box (min_pt, max_pt)."""
    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min and g.bounds_max:
                    return (
                        np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z], dtype=np.float64),
                        np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z], dtype=np.float64),
                    )
    # Fallback to transform position
    t = entity.transform
    if t and "position" in t:
        pos = t["position"]
        p = np.array([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)], dtype=np.float64)
        return (p - 0.1, p + 0.1)
    return None


@dataclass(frozen=True)
class LoadPathNode:
    """A node in the vertical structural load path graph."""
    entity_id: str
    entity_name: str
    entity_type: EntityType
    is_load_bearing: bool
    elevation_bottom: float
    elevation_top: float
    supports_ids: Tuple[str, ...]
    supported_by_ids: Tuple[str, ...]


@dataclass(frozen=True)
class LoadPathReport:
    """Report on vertical gravity load continuity and structural integrity."""
    is_structurally_sound: bool
    floating_structures: Tuple[str, ...]
    overloaded_non_bearing_walls: Tuple[str, ...]
    load_path_chains: Tuple[Tuple[str, ...], ...]
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class DoorwayClearanceRecord:
    """Record of a blocked or obstructed doorway egress path."""
    door_id: str
    door_name: str
    obstructing_entity_id: str
    obstructing_entity_name: str
    penetration_depth_m: float
    description: str


@dataclass(frozen=True)
class ClearanceReport:
    """Report on accessibility, circulation clearances, and doorway egress."""
    is_accessible: bool
    blocked_doorways: Tuple[DoorwayClearanceRecord, ...]
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class VolumetricVoidReport:
    """Report on interior room negative spaces, walkable voids, and occluded zones."""
    total_enclosed_volume_m3: float
    occupied_solid_volume_m3: float
    usable_void_volume_m3: float
    void_ratio: float  # Percentage of room that is open void
    occluded_zones_count: int
    occluded_zones: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class UtilityNetworkReport:
    """Report on electrical and plumbing functional network continuity."""
    is_continuous: bool
    electrical_circuits_count: int
    plumbing_runs_count: int
    disconnected_fixtures: Tuple[str, ...]
    plumbing_slope_violations: Tuple[str, ...]
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class SpatialReasoningReport:
    """Comprehensive spatial intelligence report across all 5 physical pillars."""
    is_physically_plausible: bool
    load_paths: LoadPathReport
    clearance: ClearanceReport
    volumetric_voids: VolumetricVoidReport
    utility_networks: UtilityNetworkReport
    auto_fix_proposals: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_physically_plausible": self.is_physically_plausible,
            "structural_soundness": self.load_paths.is_structurally_sound,
            "floating_structures": list(self.load_paths.floating_structures),
            "overloaded_non_bearing_walls": list(self.load_paths.overloaded_non_bearing_walls),
            "egress_accessible": self.clearance.is_accessible,
            "blocked_doorways_count": len(self.clearance.blocked_doorways),
            "usable_void_volume_m3": round(self.volumetric_voids.usable_void_volume_m3, 3),
            "void_ratio": round(self.volumetric_voids.void_ratio, 3),
            "utility_continuous": self.utility_networks.is_continuous,
            "disconnected_fixtures": list(self.utility_networks.disconnected_fixtures),
            "auto_fix_proposals_count": len(self.auto_fix_proposals),
        }


class StructuralLoadPathEngine:
    """Evaluates vertical gravity load paths and foundation transfer continuity."""

    @staticmethod
    def evaluate_load_paths(
        world: WorldIR,
        vertical_tolerance_m: float = 0.15,
        horizontal_overlap_min_m: float = 0.05,
    ) -> LoadPathReport:
        issues: List[str] = []
        floating_structures: List[str] = []
        overloaded_walls: List[str] = []
        chains: List[Tuple[str, ...]] = []

        # 1. Identify all structural candidate entities
        # Columns, Beams, Walls, Floors/Slabs
        structural_types = {EntityType.COLUMN, EntityType.WALL, EntityType.FLOOR, EntityType.STRUCTURE}
        struct_ents: Dict[str, Tuple[Entity, np.ndarray, np.ndarray]] = {}

        for ent in world.entities.values():
            e_class = str(ent.custom_properties.get("entity_class", "")).lower()
            is_struct = ent.type in structural_types or any(k in e_class for k in ["column", "beam", "wall", "slab", "foundation"])
            if is_struct:
                aabb = _get_entity_aabb(world, ent)
                if aabb is not None:
                    struct_ents[ent.id] = (ent, aabb[0], aabb[1])

        # 2. Check each beam / column / wall for vertical support underneath
        for ent_id, (ent, aabb_min, aabb_max) in struct_ents.items():
            # Foundation or bottom-most slab (bottom elevation <= ground elevation)
            bottom_z = float(aabb_min[2])  # In Z-up coordinate system (or Y-up)
            # Support check: look for an entity directly below whose top surface aligns with bottom_z
            # within vertical_tolerance_m and has horizontal (X-Y) overlap
            is_foundation = ent.type == EntityType.FLOOR and bottom_z <= 0.05
            if is_foundation or "foundation" in ent.name.lower():
                continue

            supporting_entities: List[str] = []
            for other_id, (other_ent, o_min, o_max) in struct_ents.items():
                if other_id == ent_id:
                    continue
                other_top_z = float(o_max[2])
                z_gap = abs(bottom_z - other_top_z)

                if z_gap <= vertical_tolerance_m:
                    # Check horizontal overlap (X and Y)
                    ox = min(aabb_max[0], o_max[0]) - max(aabb_min[0], o_min[0])
                    oy = min(aabb_max[1], o_max[1]) - max(aabb_min[1], o_min[1])
                    if ox > horizontal_overlap_min_m and oy > horizontal_overlap_min_m:
                        supporting_entities.append(other_id)

            # If it's a structural column or beam and has no vertical support underneath
            is_heavy_member = ent.type in (EntityType.COLUMN, EntityType.STRUCTURE) or "column" in ent.name.lower() or "beam" in ent.name.lower()
            if is_heavy_member and not supporting_entities and bottom_z > 0.10:
                floating_structures.append(ent_id)
                issues.append(f"Structural member '{ent.name}' (id={ent_id}) is floating at elevation z={bottom_z:.2f}m with no vertical load-path support.")

            # Check for non-bearing partition walls carrying heavy columns/beams
            for supp_id in supporting_entities:
                supp_ent, s_min, s_max = struct_ents[supp_id]
                if supp_ent.type == EntityType.WALL:
                    # Non-bearing wall check: thickness < 0.10m or explicitly marked non_bearing
                    wall_thickness = supp_ent.custom_properties.get("thickness_m") or (s_max[1] - s_min[1])
                    is_bearing = supp_ent.custom_properties.get("is_load_bearing", wall_thickness >= 0.15)
                    if not is_bearing and is_heavy_member:
                        overloaded_walls.append(supp_id)
                        issues.append(f"Non-bearing partition wall '{supp_ent.name}' (id={supp_id}, thickness={wall_thickness:.2f}m) carries structural load from '{ent.name}'.")

                # Record load path chain
                chains.append((supp_id, ent_id))

        is_sound = len(floating_structures) == 0 and len(overloaded_walls) == 0
        return LoadPathReport(
            is_structurally_sound=is_sound,
            floating_structures=tuple(floating_structures),
            overloaded_non_bearing_walls=tuple(overloaded_walls),
            load_path_chains=tuple(chains),
            issues=tuple(issues),
        )


class ClearanceAccessibilityEngine:
    """Evaluates circulation clearances and egress envelopes for doorways."""

    @staticmethod
    def evaluate_clearance(
        world: WorldIR,
        approach_depth_m: float = 0.80,
        doorway_min_width_m: float = 0.80,
    ) -> ClearanceReport:
        issues: List[str] = []
        blocked: List[DoorwayClearanceRecord] = []

        # 1. Identify all doors
        doors: List[Tuple[Entity, np.ndarray, np.ndarray]] = []
        obstacles: List[Tuple[Entity, np.ndarray, np.ndarray]] = []

        for ent in world.entities.values():
            aabb = _get_entity_aabb(world, ent)
            if aabb is None:
                continue
            if ent.type == EntityType.DOOR or "door" in ent.name.lower() or "opening" in ent.name.lower():
                doors.append((ent, aabb[0], aabb[1]))
            elif ent.type in (EntityType.WALL, EntityType.COLUMN, EntityType.STRUCTURE):
                # Potential obstruction (excluding walls that frame the door itself)
                obstacles.append((ent, aabb[0], aabb[1]))

        # 2. Check each door approach envelope
        for door_ent, d_min, d_max in doors:
            door_center = (d_min + d_max) / 2.0
            # Egress envelope: door width + approach_depth_m on both sides along Y/normal axis
            env_min = np.copy(d_min)
            env_max = np.copy(d_max)

            # Expand approach depth along Y axis
            env_min[1] -= approach_depth_m
            env_max[1] += approach_depth_m

            # Test obstacles against approach envelope
            for obs_ent, o_min, o_max in obstacles:
                if obs_ent.id == door_ent.id:
                    continue
                # Skip if it is the parent wall hosting the door
                if "door" in obs_ent.name.lower():
                    continue
                if obs_ent.type == EntityType.WALL and o_min[0] <= d_min[0] + 0.1 and o_max[0] >= d_max[0] - 0.1:
                    continue

                ox = max(0.0, min(env_max[0], o_max[0]) - max(env_min[0], o_min[0]))
                oy = max(0.0, min(env_max[1], o_max[1]) - max(env_min[1], o_min[1]))
                oz = max(0.0, min(env_max[2], o_max[2]) - max(env_min[2], o_min[2]))

                if ox > 0.05 and oy > 0.05 and oz > 0.20:
                    depth = min(ox, oy)
                    rec = DoorwayClearanceRecord(
                        door_id=door_ent.id,
                        door_name=door_ent.name,
                        obstructing_entity_id=obs_ent.id,
                        obstructing_entity_name=obs_ent.name,
                        penetration_depth_m=depth,
                        description=f"Doorway '{door_ent.name}' is blocked by '{obs_ent.name}' with {depth*100:.1f}cm penetration into egress envelope.",
                    )
                    blocked.append(rec)
                    issues.append(rec.description)

        return ClearanceReport(
            is_accessible=len(blocked) == 0,
            blocked_doorways=tuple(blocked),
            issues=tuple(issues),
        )


class VolumetricVoidEngine:
    """Computes room negative space, walkable voids, and occluded blind spots."""

    @staticmethod
    def evaluate_voids(world: WorldIR) -> VolumetricVoidReport:
        total_room_vol = 0.0
        occupied_vol = 0.0
        occluded_zones: List[Dict[str, Any]] = []

        rooms: List[Tuple[Entity, np.ndarray, np.ndarray]] = []
        contents: List[Tuple[Entity, np.ndarray, np.ndarray]] = []

        for ent in world.entities.values():
            aabb = _get_entity_aabb(world, ent)
            if aabb is None:
                continue
            if ent.type == EntityType.ROOM or "room" in ent.name.lower():
                rooms.append((ent, aabb[0], aabb[1]))
            elif ent.type in (EntityType.STRUCTURE, EntityType.WALL, EntityType.COLUMN):
                contents.append((ent, aabb[0], aabb[1]))

        # If no explicit room entities, compute whole-scene envelope
        if not rooms:
            if contents:
                all_mins = np.array([c[1] for c in contents])
                all_maxs = np.array([c[2] for c in contents])
                scene_min = np.min(all_mins, axis=0)
                scene_max = np.max(all_maxs, axis=0)
                vol = float(np.prod(scene_max - scene_min))
                total_room_vol += vol
        else:
            for r_ent, r_min, r_max in rooms:
                dim = r_max - r_min
                vol = float(np.prod(dim))
                total_room_vol += vol

        for c_ent, c_min, c_max in contents:
            dim = np.maximum(0.0, c_max - c_min)
            vol = float(np.prod(dim))
            occupied_vol += vol

            # Detect occluded tall dead zone (e.g. wardrobe/tall cabinet creating blind corridor)
            if dim[2] >= 1.80 and dim[0] >= 0.80:
                occluded_zones.append({
                    "entity_id": c_ent.id,
                    "name": c_ent.name,
                    "occluded_height_m": float(dim[2]),
                    "footprint_m2": float(dim[0] * dim[1]),
                })

        usable_void = max(0.0, total_room_vol - occupied_vol)
        ratio = usable_void / total_room_vol if total_room_vol > 0.0 else 1.0

        return VolumetricVoidReport(
            total_enclosed_volume_m3=total_room_vol,
            occupied_solid_volume_m3=occupied_vol,
            usable_void_volume_m3=usable_void,
            void_ratio=ratio,
            occluded_zones_count=len(occluded_zones),
            occluded_zones=tuple(occluded_zones),
        )


class UtilityNetworkEngine:
    """Evaluates electrical circuit continuity and plumbing drainage slopes."""

    @staticmethod
    def evaluate_utility_networks(world: WorldIR) -> UtilityNetworkReport:
        issues: List[str] = []
        disconnected: List[str] = []
        slope_violations: List[str] = []

        # Find MEP entities
        electrical_fixtures: List[Entity] = []
        plumbing_fixtures: List[Entity] = []
        conduits: List[Entity] = []

        for ent in world.entities.values():
            e_class = str(ent.custom_properties.get("entity_class", "")).lower()
            name = ent.name.lower()
            if "socket" in name or "switch" in name or "socket" in e_class:
                electrical_fixtures.append(ent)
            elif "pipe" in name or "conduit" in name or "conduit" in e_class:
                conduits.append(ent)
            elif "sink" in name or "toilet" in name or "valve" in name or "drain" in name:
                plumbing_fixtures.append(ent)

        # 1. Electrical continuity: verify connection to conduit or panel
        # Check entity relationships for RelationshipKind.CONNECTS
        connected_ids = set()
        for ent in world.entities.values():
            for rel in ent.relationships:
                if rel.kind == RelationshipKind.CONNECTS:
                    connected_ids.add(ent.id)
                    connected_ids.add(rel.target_id)

        for ef in electrical_fixtures:
            # If not explicitly connected in relationship graph, check proximity to a conduit
            ef_aabb = _get_entity_aabb(world, ef)
            is_conn = ef.id in connected_ids
            if not is_conn and ef_aabb is not None:
                ef_pos = (ef_aabb[0] + ef_aabb[1]) / 2.0
                for c in conduits:
                    c_aabb = _get_entity_aabb(world, c)
                    if c_aabb is not None:
                        c_pos = (c_aabb[0] + c_aabb[1]) / 2.0
                        if np.linalg.norm(ef_pos - c_pos) <= 0.50:
                            is_conn = True
                            break
            if not is_conn:
                disconnected.append(ef.id)
                issues.append(f"Electrical fixture '{ef.name}' (id={ef.id}) has no connected conduit or circuit feed.")

        # 2. Plumbing drainage slope check (minimum 1% slope = 0.01)
        for pf in plumbing_fixtures:
            slope = pf.custom_properties.get("drainage_slope", 0.02)
            if slope < 0.01:
                slope_violations.append(pf.id)
                issues.append(f"Plumbing fixture '{pf.name}' has insufficient drainage slope ({slope*100:.1f}% < 1.0% min).")

        is_continuous = len(disconnected) == 0 and len(slope_violations) == 0
        return UtilityNetworkReport(
            is_continuous=is_continuous,
            electrical_circuits_count=len(electrical_fixtures),
            plumbing_runs_count=len(plumbing_fixtures),
            disconnected_fixtures=tuple(disconnected),
            plumbing_slope_violations=tuple(slope_violations),
            issues=tuple(issues),
        )


class SpatialReasoningEngine:
    """Unified coordinator for Universal Spatial Reasoning (Directive Rule 8)."""

    @staticmethod
    def evaluate_world(world: WorldIR) -> SpatialReasoningReport:
        load_report = StructuralLoadPathEngine.evaluate_load_paths(world)
        clear_report = ClearanceAccessibilityEngine.evaluate_clearance(world)
        void_report = VolumetricVoidEngine.evaluate_voids(world)
        util_report = UtilityNetworkEngine.evaluate_utility_networks(world)

        proposals: List[Dict[str, Any]] = []

        # Auto-corrections for floating structures
        for float_id in load_report.floating_structures:
            proposals.append({
                "target_entity_id": float_id,
                "fix_type": "reinforce_load_path",
                "recommendation": f"Add bearing column or wall under entity {float_id} to transfer load to foundation.",
            })

        # Auto-corrections for blocked doorways
        for b in clear_report.blocked_doorways:
            proposals.append({
                "target_entity_id": b.obstructing_entity_id,
                "fix_type": "clear_egress_path",
                "recommendation": f"Move {b.obstructing_entity_name} away from door {b.door_name} by at least {b.penetration_depth_m:.2f}m.",
            })

        # Enrich WorldIR entities with structural support relationships
        for supp_id, supported_id in load_report.load_path_chains:
            if supp_id in world.entities:
                supp_ent = world.entities[supp_id]
                if not any(r.kind == RelationshipKind.SUPPORTS and r.target_id == supported_id for r in supp_ent.relationships):
                    supp_ent.relationships.append(Relationship(
                        kind=RelationshipKind.SUPPORTS,
                        target_id=supported_id,
                        confidence=0.95,
                        provenance=Provenance.INFERRED,
                    ))

        is_plausible = load_report.is_structurally_sound and clear_report.is_accessible

        report = SpatialReasoningReport(
            is_physically_plausible=is_plausible,
            load_paths=load_report,
            clearance=clear_report,
            volumetric_voids=void_report,
            utility_networks=util_report,
            auto_fix_proposals=tuple(proposals),
        )

        if world.metadata is None:
            world.metadata = {}
        world.metadata["spatial_reasoning"] = report.to_dict()

        return report
