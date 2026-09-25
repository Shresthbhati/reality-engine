"""Architectural promotion + relationship graph (directive sections
25-28): candidate components -> canonical WorldIR entities, and
real-position-derived relationships.

Promotion reuses the promote_planes pattern (detection/classification
stay pure; promotion is the write path that mutates the world):
  - The candidate's REAL fit parameters are recorded in
    custom_properties (radius/axis/rms/span -- measured facts, never
    invented ones).
  - Evidence traceability (directive section 27): source observation
    evidence ids ride along, so entity -> observations -> images is
    queryable without touching the provenance graph (which remains the
    artifact-level chain of custody).
  - Confidence tiers (directive section 28) flow into the entity:
    entity.confidence = effective (prior-adjusted) confidence;
    the tier name is recorded; provenance is INFERRED -- a fitted
    component is an inference from evidence, never an observation
    itself.
  - Unaccepted observations are REFUSED promotion (ValueError): an
    unaccepted hypothesis is not an entity.

Relationship graph (directive sections 8-9, 25): edges are derived
from REAL measured geometry only --
  - adjacent_to: same-class components within a distance band;
  - part_of / contains / supports are NOT invented here: a hierarchy
    (building -> facade -> columns) requires evidence for the
    container (a detected building/facade component), so an arbitrary
    or lone component set produces NO invented hierarchy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Provenance,
    Relationship,
    RelationshipKind,
)
from perception.architecture.components import (
    CandidateComponent,
    ComponentObservation,
    ConfidenceTier,
)
from perception.architecture.parametric import CircleFit, CylinderFit, SphereFit
from perception.architecture.beams import BeamFit
from perception.architecture.columns import ColumnFit
from perception.architecture.roofs import RoofFit
from perception.architecture.stairs import StaircaseFit
from perception.architecture.windows import WindowFit

#: Adjacency band for same-class structural neighbors (colonnades,
#: window rows). A column 3 m from its neighbor is adjacent; 50 m away
#: it is not. Deliberately conservative; not tuned to any dataset.
ADJACENT_MAX_DISTANCE_M = 8.0
#: ...but not so close that they are the SAME component (resolution
#: handled upstream; here the floor just avoids self/dupe edges).
ADJACENT_MIN_DISTANCE_M = 0.05


_ARCH_CLASS_TO_ENTITY_TYPE: Dict[str, EntityType] = {
    "column": EntityType.COLUMN,
    "beam": EntityType.BEAM,
    "dome": EntityType.DOME,
    "arch": EntityType.ARCH,
    "wall": EntityType.WALL,
    "floor": EntityType.FLOOR,
    "ceiling": EntityType.CEILING,
    "roof": EntityType.ROOF,
    "door": EntityType.DOOR,
    "window": EntityType.WINDOW,
    "stairs": EntityType.STAIRS,
    "building": EntityType.BUILDING,
    "facade": EntityType.WALL,
    "corridor": EntityType.CORRIDOR,
    "level": EntityType.LEVEL,
    "room": EntityType.ROOM,
}


def _store_entity(world, entity) -> None:
    """Write an entity into world.entities, whichever container the
    caller supplied: an EntityRegistry (integrity-checked .add) or a
    plain dict (item assignment, the older test convention)."""
    entities = world.entities
    if hasattr(entities, "add"):
        entities.add(entity)
    else:
        entities[entity.id] = entity


def _fit_properties(fit) -> dict:
    """Measured fit facts for custom_properties -- whatever the fit
    actually measured, nothing else."""
    if isinstance(fit, CylinderFit):
        return {
            "fit_kind": "cylinder",
            "radius_m": fit.radius_m,
            "axis": list(fit.axis),
            "axis_point": list(fit.axis_point),
            "height_min_m": fit.height_min_m,
            "height_max_m": fit.height_max_m,
            "rms_residual_m": fit.rms_residual_m,
            "max_residual_m": fit.max_residual_m,
            "n_points": fit.n_points,
        }
    if isinstance(fit, SphereFit):
        return {
            "fit_kind": "sphere",
            "center": list(fit.center),
            "radius_m": fit.radius_m,
            "rms_residual_m": fit.rms_residual_m,
            "max_residual_m": fit.max_residual_m,
            "n_points": fit.n_points,
        }
    if isinstance(fit, CircleFit):
        return {
            "fit_kind": "circle",
            "center": list(fit.center),
            "radius_m": fit.radius_m,
            "plane_normal": list(fit.plane_normal),
            "angular_span_rad": fit.angular_span_rad,
            "extrusion_depth_m": fit.extrusion_depth_m,
            "rms_residual_m": fit.rms_residual_m,
            "n_points": fit.n_points,
        }
    if isinstance(fit, ColumnFit):
        return {
            "fit_kind": "column",
            "radius_m": fit.radius_m,
            "axis": list(fit.axis),
            "height_m": fit.height_m,
            "axis_up_dot": fit.axis_up_dot,
            "rms_residual_m": fit.cylinder.rms_residual_m,
            "max_residual_m": fit.cylinder.max_residual_m,
            "n_points": fit.n_points,
        }
    if isinstance(fit, BeamFit):
        return {
            "fit_kind": "beam",
            "axis": list(fit.axis),
            "length_m": fit.length_m,
            "height_m": fit.height_m,
            "width_m": fit.width_m,
            "aspect_ratio": fit.aspect_ratio,
            "rms_residual_m": fit.rms_residual_m,
            "n_points": fit.n_points,
        }
    if isinstance(fit, WindowFit):
        return {
            "fit_kind": "window",
            "wall_plane_id": fit.wall_plane_id,
            "width_m": fit.width_m,
            "height_m": fit.height_m,
            "sill_height_m": fit.sill_height_m,
            "bounds_min": list(fit.bounds_min),
            "bounds_max": list(fit.bounds_max),
            "n_points": fit.n_points,
        }
    if isinstance(fit, RoofFit):
        return {
            "fit_kind": "roof",
            "plane_id": fit.plane_id,
            "width_m": fit.width_m,
            "depth_m": fit.depth_m,
            "height_m": fit.height_m,
            "up_dot": fit.up_dot,
            "n_planes": fit.n_planes,
        }
    if isinstance(fit, StaircaseFit):
        return {
            "fit_kind": "stairs",
            "n_steps": fit.n_steps,
            "rise_m": fit.rise_m,
            "going_m": fit.going_m,
            "span_m": fit.span_m,
            "rms_residual_m": fit.rms_residual_m,
            "n_points": fit.n_points,
        }
    raise ValueError(f"unsupported fit type {type(fit).__name__}")


def promote_component_to_entity(
    component, world: WorldIR, entity_id: str, entity_name: str = ""
) -> Entity:
    """Promote one candidate component (or single observation) into a
    canonical WorldIR Entity. Refuses unaccepted components.

    Mutates `world` (appends the Entity) exactly like
    evidence/promote_planes.py -- promotion is the write path.
    """
    arch_class = getattr(component, "arch_class", None)
    fit = getattr(component, "fit", None)
    accepted = getattr(component, "accepted", None)
    if accepted is None:
        # CandidateComponent: acceptance lives on its observations.
        accepted = all(o.accepted for o in component.source_observations)
        if not accepted:
            raise ValueError(
                f"component {component.component_id} has unaccepted source "
                f"observations -- refusing promotion"
            )
        accepted = True
    if not accepted:
        obs = component
        raise ValueError(
            f"observation of class {obs.arch_class} was not accepted "
            f"({obs.rejection_reason}) -- refusing promotion"
        )
    if fit is None:
        raise ValueError("component carries no fit -- refusing promotion")
    entity_type = _ARCH_CLASS_TO_ENTITY_TYPE.get(arch_class)
    if entity_type is None:
        raise ValueError(
            f"arch class {arch_class!r} has no WorldIR mapping -- "
            f"register it in promotion._ARCH_CLASS_TO_ENTITY_TYPE"
        )

    evidence_ids = (
        component.evidence_ids
        if hasattr(component, "evidence_ids")
        else ()
    )
    obs_count = (
        component.observation_count
        if hasattr(component, "observation_count")
        else 1
    )
    segment_ids = (
        [o.segment_id for o in component.source_observations]
        if hasattr(component, "source_observations")
        else [component.segment_id]
    )
    tier = (
        ConfidenceTier.from_confidence(component.effective_confidence())
        if hasattr(component, "effective_confidence")
        else ConfidenceTier.from_confidence(component.confidence)
    )

    props = {"arch_class": arch_class}
    props.update(_fit_properties(fit))
    props["evidence_ids"] = list(evidence_ids)
    props["observation_count"] = obs_count
    props["segment_ids"] = segment_ids
    props["confidence_tier"] = tier.value

    entity = Entity(
        id=entity_id,
        type=entity_type,
        name=entity_name or f"{arch_class} {entity_id.rsplit('-', 1)[-1]}",
        custom_properties=props,
        confidence=component.effective_confidence()
        if hasattr(component, "effective_confidence")
        else component.confidence,
        provenance=Provenance.INFERRED,
    )
    _store_entity(world, entity)
    return entity


def build_architectural_graph(
    candidates: Sequence[CandidateComponent], world: WorldIR,
    adjacent_max_distance_m: float = ADJACENT_MAX_DISTANCE_M,
) -> List[str]:
    """Promote all candidates and derive relationships from real
    geometry. Returns promoted entity ids, in candidate order.

    Edges derived (directive section 25), all measured:
      - same-class adjacency within the band (colonnades, window rows)
    Edges deliberately NOT invented: part_of/supports/contains require
    evidence FOR the container component; an arbitrary or lone set
    gets no hierarchy.
    """
    entity_ids: List[str] = []
    by_class: Dict[str, List[Tuple[CandidateComponent, Entity]]] = {}
    for cand in candidates:
        eid = f"ent-{cand.arch_class}-{len(entity_ids):04d}"
        entity = promote_component_to_entity(cand, world, eid)
        entity_ids.append(eid)
        by_class.setdefault(cand.arch_class, []).append((cand, entity))

    # Same-class adjacency from real positions.
    for class_name, pairs in sorted(by_class.items()):
        for i, (cand_a, ent_a) in enumerate(pairs):
            for cand_b, ent_b in pairs[i + 1:]:
                d = _dist(cand_a.position, cand_b.position)
                if ADJACENT_MIN_DISTANCE_M < d <= adjacent_max_distance_m:
                    conf = max(
                        0.0, 1.0 - d / adjacent_max_distance_m
                    )
                    ent_a.relationships.append(Relationship(
                        kind=RelationshipKind.ADJACENT_TO,
                        target_id=ent_b.id,
                        confidence=conf,
                        provenance=Provenance.ESTIMATED,
                        metadata={"distance_m": d},
                    ))
                    ent_b.relationships.append(Relationship(
                        kind=RelationshipKind.ADJACENT_TO,
                        target_id=ent_a.id,
                        confidence=conf,
                        provenance=Provenance.ESTIMATED,
                        metadata={"distance_m": d},
                    ))
    return entity_ids


def _dist(a, b) -> float:
    return (
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ) ** 0.5


def promote_interior_graph_to_world(
    building_graph,
    rooms: Sequence[object],
    corridors: Optional[Sequence[object]],
    world: WorldIR,
    stairs: Optional[Sequence[object]] = None,
) -> Dict[str, str]:
    """Promote a BuildingGraph, its rooms, corridors, openings, and stairs
    into canonical WorldIR entities and structural relationships."""
    from world_ir.schema_v1 import Geometry, GeometryType, Vector3

    created_ids: Dict[str, str] = {}
    if building_graph is None:
        return created_ids

    # 1. Building entity
    bld_id = f"bld-{building_graph.building_id}"
    bmin = building_graph.envelope_bounds_min
    bmax = building_graph.envelope_bounds_max
    geom_id = f"geom-{bld_id}"
    world.geometries[geom_id] = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(bmin[0], bmin[1], bmin[2]),
        bounds_max=Vector3(bmax[0], bmax[1], bmax[2]),
    )
    bld_entity = Entity(
        id=bld_id,
        type=EntityType.BUILDING,
        name=f"Building {building_graph.building_id}",
        geometry_ids=[geom_id],
        custom_properties={
            "storey_count": len(building_graph.storeys),
            "envelope_bounds_min": list(bmin),
            "envelope_bounds_max": list(bmax),
        },
        provenance=Provenance.INFERRED,
        confidence=1.0,
    )
    world.entities[bld_id] = bld_entity
    created_ids["building"] = bld_id

    # Storey mapping: storey_id -> level_entity_id
    storey_to_level: Dict[str, str] = {}
    space_to_level: Dict[str, str] = {}

    # 2. Level entities (one per storey)
    for idx, storey in enumerate(building_graph.storeys, start=1):
        lvl_id = f"lvl-{idx:02d}"
        storey_to_level[storey.storey_id] = lvl_id
        for rid in storey.room_ids:
            space_to_level[rid] = lvl_id
        for cid in storey.corridor_ids:
            space_to_level[cid] = lvl_id

        lvl_entity = Entity(
            id=lvl_id,
            type=EntityType.LEVEL,
            name=f"Level {idx - 1}",
            custom_properties={
                "floor_height_m": storey.floor_height_m,
                "room_ids": list(storey.room_ids),
                "corridor_ids": list(storey.corridor_ids),
                "stair_ids": list(storey.stair_ids),
                "storey_id": storey.storey_id,
            },
            provenance=Provenance.INFERRED,
            confidence=1.0,
            relationships=[
                Relationship(kind=RelationshipKind.PART_OF, target_id=bld_id)
            ],
        )
        world.entities[lvl_id] = lvl_entity
        created_ids[storey.storey_id] = lvl_id

    # 3. Room entities
    for r in rooms:
        rid = f"room-{r.room_id}"
        r_bmin, r_bmax = r.bounds_min, r.bounds_max
        r_geom_id = f"geom-{rid}"
        world.geometries[r_geom_id] = Geometry(
            id=r_geom_id,
            type=GeometryType.BOX,
            bounds_min=Vector3(r_bmin[0], r_bmin[1], r_bmin[2]),
            bounds_max=Vector3(r_bmax[0], r_bmax[1], r_bmax[2]),
        )
        lvl_id = space_to_level.get(r.room_id)
        rels = []
        if lvl_id:
            rels.append(Relationship(kind=RelationshipKind.PART_OF, target_id=lvl_id))
        for be in r.boundary_element_ids:
            if be in world.entities:
                rels.append(Relationship(kind=RelationshipKind.CONTAINS, target_id=be))
        for adj in r.adjacent_room_ids:
            adj_id = f"room-{adj}"
            rels.append(Relationship(kind=RelationshipKind.ADJACENT_TO, target_id=adj_id))

        r_entity = Entity(
            id=rid,
            type=EntityType.ROOM,
            name=f"Room {r.room_id.rsplit('-', 1)[-1]}",
            geometry_ids=[r_geom_id],
            custom_properties={
                "dimensions_m": dict(r.dimensions_m),
                "floor_area_m2": r.floor_area_m2,
                "height_m": r.dimensions_m.get("z", 2.4),
                "boundary_element_ids": list(r.boundary_element_ids),
                "adjacent_room_ids": list(r.adjacent_room_ids),
                "corridor_ids": list(getattr(r, "corridor_ids", ())),
                "status": getattr(r, "status", "detected"),
                "confidence": getattr(r, "confidence", 1.0),
                "boundary_completeness": getattr(r, "boundary_completeness", 1.0),
                "ceiling_evidence": getattr(r, "ceiling_evidence", "measured"),
                "notes": list(getattr(r, "notes", ())),
            },
            relationships=rels,
            provenance=Provenance.INFERRED,
            confidence=getattr(r, "confidence", 1.0),
        )
        world.entities[rid] = r_entity
        created_ids[r.room_id] = rid

    # 4. Corridor entities
    for c in (corridors or []):
        cid = f"corridor-{getattr(c, 'corridor_id', '001')}"
        c_bmin = getattr(c, "bounds_min", (0, 0, 0))
        c_bmax = getattr(c, "bounds_max", (1, 1, 1))
        c_geom_id = f"geom-{cid}"
        world.geometries[c_geom_id] = Geometry(
            id=c_geom_id,
            type=GeometryType.BOX,
            bounds_min=Vector3(c_bmin[0], c_bmin[1], c_bmin[2]),
            bounds_max=Vector3(c_bmax[0], c_bmax[1], c_bmax[2]),
        )
        lvl_id = space_to_level.get(getattr(c, "corridor_id", ""))
        c_rels = []
        if lvl_id:
            c_rels.append(Relationship(kind=RelationshipKind.PART_OF, target_id=lvl_id))
        for r_id in getattr(c, "connected_room_ids", ()):
            target_rid = f"room-{r_id}"
            c_rels.append(Relationship(kind=RelationshipKind.CONNECTS, target_id=target_rid))
            c_rels.append(Relationship(kind=RelationshipKind.ADJACENT_TO, target_id=target_rid))

        c_entity = Entity(
            id=cid,
            type=EntityType.CORRIDOR,
            name=f"Corridor {getattr(c, 'corridor_id', '').rsplit('-', 1)[-1]}",
            geometry_ids=[c_geom_id],
            custom_properties={
                "length_m": getattr(c, "length_m", 0.0),
                "width_m": getattr(c, "width_m", 0.0),
                "height_m": getattr(c, "height_m", 2.4),
                "floor_area_m2": getattr(c, "floor_area_m2", 0.0),
                "longitudinal_axis": list(getattr(c, "longitudinal_axis", (1, 0, 0))),
                "connected_room_ids": list(getattr(c, "connected_room_ids", ())),
                "boundary_element_ids": list(getattr(c, "boundary_element_ids", ())),
                "status": getattr(c, "status", "detected"),
                "confidence": getattr(c, "confidence", 1.0),
                "notes": list(getattr(c, "notes", ())),
            },
            relationships=c_rels,
            provenance=Provenance.INFERRED,
            confidence=getattr(c, "confidence", 1.0),
        )
        world.entities[cid] = c_entity
        created_ids[getattr(c, "corridor_id", "")] = cid

    # 5. Openings (Doorways and Windows)
    all_spaces = list(rooms) + list(corridors or [])
    op_counter = 0
    for sp in all_spaces:
        sp_id = getattr(sp, "room_id", getattr(sp, "corridor_id", ""))
        target_space_entity_id = created_ids.get(sp_id, sp_id)
        for op in getattr(sp, "openings", ()):
            op_dict = op.to_dict() if hasattr(op, "to_dict") else dict(op)
            op_kind = op_dict.get("kind", "doorway")
            op_entity_type = EntityType.DOOR if op_kind == "doorway" else EntityType.WINDOW
            ent_op_id = f"op-{op_kind}-{op_counter:04d}"
            op_counter += 1

            host_wall = op_dict.get("wall_element_id")
            op_rels = []
            if host_wall and host_wall in world.entities:
                op_rels.append(Relationship(kind=RelationshipKind.PART_OF, target_id=host_wall))
            if target_space_entity_id and target_space_entity_id in world.entities:
                op_rels.append(Relationship(kind=RelationshipKind.CONNECTS, target_id=target_space_entity_id))

            op_entity = Entity(
                id=ent_op_id,
                type=op_entity_type,
                name=f"{op_kind.capitalize()} {op_counter}",
                custom_properties={
                    "opening_kind": op_kind,
                    "width_m": op_dict.get("width_m", 0.9),
                    "height_m": op_dict.get("height_m", 2.1),
                    "sill_height_m": op_dict.get("sill_height_m", 0.0),
                    "host_wall_id": host_wall,
                    "connected_space_ids": [target_space_entity_id],
                    "confidence": op_dict.get("confidence", 1.0),
                },
                relationships=op_rels,
                provenance=Provenance.INFERRED,
                confidence=op_dict.get("confidence", 1.0),
            )
            world.entities[ent_op_id] = op_entity
            created_ids[ent_op_id] = ent_op_id

    # 6. Stairs
    for st in (stairs or []):
        st_id = f"stair-{getattr(st, 'stair_id', getattr(st, 'id', '001'))}"
        st_dict = st.to_dict() if hasattr(st, "to_dict") else dict(st)
        pos = getattr(st, "position", (0, 0, 0))
        st_geom_id = f"geom-{st_id}"
        world.geometries[st_geom_id] = Geometry(
            id=st_geom_id,
            type=GeometryType.POINTCLOUD,
            bounds_min=Vector3(pos[0] - 1.0, pos[1] - 1.0, pos[2] - 1.0),
            bounds_max=Vector3(pos[0] + 1.0, pos[1] + 1.0, pos[2] + 1.0),
        )
        st_rels = []
        for s in building_graph.storeys:
            if getattr(st, "stair_id", getattr(st, "id", "stair-001")) in s.stair_ids:
                lvl_id = storey_to_level.get(s.storey_id)
                if lvl_id:
                    st_rels.append(Relationship(kind=RelationshipKind.CONNECTS, target_id=lvl_id))

        st_entity = Entity(
            id=st_id,
            type=EntityType.STAIRS,
            name=f"Stairs {st_id.rsplit('-', 1)[-1]}",
            geometry_ids=[st_geom_id],
            custom_properties={
                "n_steps": getattr(st, "n_steps", 10),
                "rise_m": getattr(st, "rise_m", 0.17),
                "going_m": getattr(st, "going_m", 0.28),
                "span_m": getattr(st, "span_m", 2.8),
                "confidence": getattr(st, "confidence", 1.0),
            },
            relationships=st_rels,
            provenance=Provenance.INFERRED,
            confidence=getattr(st, "confidence", 1.0),
        )
        world.entities[st_id] = st_entity
        created_ids[st_id] = st_id

    return created_ids

