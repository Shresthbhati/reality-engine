"""Inspector: read-only query facade over a WorldIR for the Studio (§17/§35).

Not a new data model -- everything here already exists on WorldIR
(entities/materials/geometries/observations/causal_relations/branches).
Inspector answers the concrete questions §35 lists: inspect entities,
evidence, measurements, materials, structural relationships, uncertainty,
events, and causal chains -- by resolving the id-references WorldIR
stores (entity.material_ids -> world.materials, etc).

No mutation methods here on purpose: an inspection tool that can also
write would blur the "AI/tools query, engine owns truth" boundary (§7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from provenance import Provenance
from world_ir import CausalRelation, Entity, Geometry, Material, TemporalEvent, WorldIR


@dataclass(frozen=True)
class EntitySummary:
    """Everything about one entity, resolved from its id-references."""
    entity: Entity
    materials: List[Material]
    geometries: List[Geometry]
    relationships: List[Dict[str, Any]]  # {kind, target_id, target_name}


class Inspector:
    """Read-only query surface over a single WorldIR."""

    def __init__(self, world: WorldIR):
        self.world = world

    # ---- entities ----

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.world.entities.get(entity_id)

    def inspect_entity(self, entity_id: str) -> Optional[EntitySummary]:
        """Full resolved view of one entity: its materials, geometries,
        and relationships with target names filled in."""
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return None

        materials = [self.world.materials[mid] for mid in entity.material_ids if mid in self.world.materials]
        geometries = [self.world.geometries[gid] for gid in entity.geometry_ids if gid in self.world.geometries]

        relationships = []
        for rel in entity.relationships:
            target = self.world.entities.get(rel.target_id)
            relationships.append({
                "kind": rel.kind.value if hasattr(rel.kind, "value") else rel.kind,
                "target_id": rel.target_id,
                "target_name": target.name if target else None,
            })

        return EntitySummary(entity=entity, materials=materials, geometries=geometries, relationships=relationships)

    def list_entities(self) -> List[Entity]:
        return list(self.world.entities.values())

    # ---- evidence / measurements / uncertainty ----

    def get_evidence(self, entity_id: str) -> List[Any]:
        """Observations backing this entity (spec: 'show evidence supporting inferred properties')."""
        entity = self.world.entities.get(entity_id)
        return list(entity.observations) if entity else []

    def get_measurements(self, entity_id: str) -> Dict[str, Any]:
        """All Measurement-valued PhysicalProperties across the entity's materials, keyed by material id then field."""
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return {}

        result: Dict[str, Any] = {}
        for mid in entity.material_ids:
            material = self.world.materials.get(mid)
            if material is None:
                continue
            props = material.properties
            measurements = {}
            for field_name in (
                "density", "mass", "volume", "young_modulus", "thermal_conductivity",
                "specific_heat", "ignition_temperature", "tensile_strength", "shear_strength",
            ):
                value = getattr(props, field_name)
                if value is not None:
                    measurements[field_name] = value
            if measurements:
                result[mid] = measurements
        return result

    def get_confidence(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Uncertainty/confidence summary: entity-level plus its lowest-confidence material."""
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return None

        material_confidences = [
            self.world.materials[mid].confidence
            for mid in entity.material_ids
            if mid in self.world.materials
        ]

        return {
            "entity_provenance": entity.provenance.value,
            "entity_confidence": entity.confidence,
            "entity_uncertainty": entity.uncertainty.to_dict(),
            "min_material_confidence": min(material_confidences) if material_confidences else None,
        }

    # ---- materials ----

    def get_materials(self, entity_id: str) -> List[Material]:
        entity = self.world.entities.get(entity_id)
        if entity is None:
            return []
        return [self.world.materials[mid] for mid in entity.material_ids if mid in self.world.materials]

    # ---- structural relationships ----

    def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        summary = self.inspect_entity(entity_id)
        return summary.relationships if summary else []

    def find_supporting(self, entity_id: str) -> List[str]:
        """Entity ids with a SUPPORTS or RESTS_ON relationship targeting entity_id."""
        supporters = []
        for entity in self.world.entities.values():
            for rel in entity.relationships:
                kind = rel.kind.value if hasattr(rel.kind, "value") else rel.kind
                if rel.target_id == entity_id and kind in ("supports", "rests_on"):
                    supporters.append(entity.id)
        return supporters

    # ---- measurement between entities ----

    def measure_distance(self, entity_id_a: str, entity_id_b: str) -> Optional[float]:
        """Euclidean distance between two entities' transform origins, if both have transforms."""
        a = self.world.entities.get(entity_id_a)
        b = self.world.entities.get(entity_id_b)
        if a is None or b is None or a.transform is None or b.transform is None:
            return None

        pa = a.transform.get("position") or a.transform.get("translation")
        pb = b.transform.get("position") or b.transform.get("translation")
        if pa is None or pb is None:
            return None

        return sum((pa[k] - pb[k]) ** 2 for k in ("x", "y", "z")) ** 0.5

    # ---- provenance / quality queries ----

    def query_by_provenance(self, provenance: Provenance) -> List[Entity]:
        return [e for e in self.world.entities.values() if e.provenance == provenance]

    def low_confidence_entities(self, threshold: float = 0.5) -> List[Entity]:
        return [e for e in self.world.entities.values() if e.confidence < threshold]

    # ---- events / causality ----

    def get_events(self, entity_id: Optional[str] = None) -> List[TemporalEvent]:
        events = list(self.world.temporal_events.values())
        if entity_id is not None:
            events = [e for e in events if e.entity_id == entity_id]
        return sorted(events, key=lambda e: e.timestamp)

    def get_causal_chain(self, event_id: str) -> List[str]:
        """Forward causal chain: event_id and every effect reachable from it."""
        chain = [event_id]
        to_visit = [r.effect_id for r in self.world.causal_relations if r.cause_id == event_id]
        seen = set(chain)
        while to_visit:
            current = to_visit.pop(0)
            if current in seen:
                continue
            seen.add(current)
            chain.append(current)
            to_visit.extend(r.effect_id for r in self.world.causal_relations if r.cause_id == current)
        return chain

    def why(self, event_id: str) -> List[CausalRelation]:
        """Direct causes of event_id -- answers 'what caused this?' (§27)."""
        return [r for r in self.world.causal_relations if r.effect_id == event_id]

    # ---- branches / scenarios ----

    def list_branches(self) -> List[Any]:
        return list(self.world.branches.values())

    def list_scenarios(self) -> List[Any]:
        return list(self.world.scenarios.values())

    # ---- dashboard summary ----

    def summary(self) -> Dict[str, int]:
        return {
            "entities": len(self.world.entities),
            "materials": len(self.world.materials),
            "geometries": len(self.world.geometries),
            "surfaces": len(self.world.surfaces),
            "temporal_events": len(self.world.temporal_events),
            "causal_relations": len(self.world.causal_relations),
            "branches": len(self.world.branches),
            "scenarios": len(self.world.scenarios),
        }
