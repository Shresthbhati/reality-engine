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
