"""Promote a MergedObjectCandidate into a WorldIR Entity (closes the
gap this audit cycle identified as the highest-value bottleneck: the
object-perception pipeline built over the last several sessions --
`perception/instances/lifting.py` -> `object_resolution.py` ->
`measurement.py` -- stopped at library data structures. Nothing wrote
an `ObjectHypothesis3D`/`MergedObjectCandidate` into WorldIR, unlike
structural elements, where `evidence/promote_planes.py` and
`evidence/promote_rooms.py` already close that exact loop for walls/
floors/ceilings/rooms. This module is the same loop for objects.

Mirrors `promote_planes.py`'s shape deliberately (promotion is the only
write path; detection/merging/measurement stay pure upstream of this):

  - geometry: a BOX Geometry from the candidate's real union AABB
    (`bounds_min`/`bounds_max` -- the same real, non-fabricated data
    every other exporter/compiler consumer in this repo already reads),
    provenance INFERRED (derived from 2D detection + depth + multi-view
    merging, never a direct observation);
  - entity: `EntityType.UNKNOWN` -- the WorldIR ontology
    (`world_ir/schema_v1.py`) has no furniture/object-category values
    (WALL/FLOOR/CEILING/etc. exist for structural elements, nothing for
    "chair"/"table"/"lamp"). Guessing an EntityType for the campaign's
    "ontology mapping" phase would be exactly the kind of speculative,
    unrequested categorization the project conventions warn against;
    the honest move is UNKNOWN type + the real detector/segmenter label
    preserved verbatim in `semantic_labels` and `name`, so nothing is
    lost and nothing is invented;
  - measurements: `perception.instances.measurement.measure_dimensions()`
    written onto `entity.custom_properties` (same convention
    `promote_planes.py`/`promote_rooms.py` already use for extent/area/
    height);
  - an Observation recording exactly which frames/regions/hypotheses
    fed this entity, so "why does the engine believe this is a chair
    here?" is answerable from the WorldIR alone (same pattern as every
    other promotion module in this repo).

Refuses (raises) a candidate with zero contributing hypotheses -- there
is no honest entity to promote from no evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Measurement, Observation, Vector3, WorldIR

from perception.instances.measurement import measure_dimensions
from perception.instances.object_resolution import MergedObjectCandidate


class ObjectPromotionError(ValueError):
    pass


@dataclass(frozen=True)
class ObjectPromotionResult:
    entity: Entity
    geometry: Geometry
    measurements: Dict[str, Measurement]


def promote_object_to_entity(
    candidate: MergedObjectCandidate,
    world: WorldIR,
    entity_id: str,
) -> ObjectPromotionResult:
    """Promote one merged object candidate into `world`. Mutates `world`
    (adds Entity + Geometry), matching every other promotion module's
    write-path convention in this repo."""
    if candidate.observation_count == 0:
        raise ObjectPromotionError(
            f"candidate {candidate.candidate_id!r} has zero contributing hypotheses -- "
            "refusing to promote an entity with no supporting evidence"
        )

    geometry = Geometry(
        id=f"geom-{entity_id}",
        type=GeometryType.BOX,
        bounds_min=Vector3(candidate.bounds_min.x, candidate.bounds_min.y, candidate.bounds_min.z),
        bounds_max=Vector3(candidate.bounds_max.x, candidate.bounds_max.y, candidate.bounds_max.z),
        provenance=Provenance.INFERRED,
        confidence=candidate.confidence,
        observations=[Observation(
            id=f"obs-object-geom-{entity_id}",
            sensor_type="object_lifting_and_resolution",
            confidence=candidate.confidence,
            metadata={
                "label": candidate.label,
                "observation_count": candidate.observation_count,
                "evidence_ids": list(candidate.evidence_ids),
                "region_ids": [h.region_id for h in candidate.source_hypotheses],
            },
        )],
    )
    world.geometries[geometry.id] = geometry

    entity = Entity(
        id=entity_id,
        name=candidate.label,
        # Ontology has no object/furniture categories yet -- UNKNOWN is
        # honest; the real label survives below, never discarded.
        type=EntityType.UNKNOWN,
        transform={"position": {
            "x": candidate.position.x, "y": candidate.position.y, "z": candidate.position.z,
        }},
        geometry_ids=[geometry.id],
        semantic_labels=[candidate.label],
        provenance=Provenance.INFERRED,
        confidence=candidate.confidence,
    )

    measurements = measure_dimensions(candidate)
    for key, measurement in measurements.items():
        entity.custom_properties[key] = measurement.value

    entity.observations.append(Observation(
        id=f"obs-object-promotion-{entity_id}",
        sensor_type="object_promotion",
        confidence=candidate.confidence,
        metadata={
            "label": candidate.label,
            "observation_count": candidate.observation_count,
            "evidence_ids": list(candidate.evidence_ids),
            **{key: m.value for key, m in measurements.items()},
        },
    ))

    world.entities[entity_id] = entity
    return ObjectPromotionResult(entity=entity, geometry=geometry, measurements=measurements)
