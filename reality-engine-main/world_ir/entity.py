"""Entity system (spec sec 6 WORLD IR: entity schema).

The entity schema in the spec names fields for subsystems that don't
exist yet (geometry, material, physics, destruction...). Rather than
guess their shape now, this holds them as opaque dicts under
`extra_fields` so later subsystems can define their own typed payloads
without a schema migration here -- the fields that ARE implemented at
this layer (id, type, transform, relationships, provenance, uncertainty)
are first-class and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from provenance import Provenance, Uncertainty
from .coordinates import Transform


class DuplicateEntityError(ValueError):
    pass


class UnknownEntityError(ValueError):
    pass


class DanglingRelationshipError(ValueError):
    pass


@dataclass
class Relationship:
    kind: str
    target_id: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target_id": self.target_id}

    @staticmethod
    def from_dict(data: dict) -> "Relationship":
        return Relationship(kind=data["kind"], target_id=data["target_id"])


@dataclass
class Entity:
    id: str
    type: str
    transform: Optional[Transform] = None
    semantic_labels: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    provenance: Provenance = Provenance.UNKNOWN
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    extra_fields: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "transform": self.transform.to_dict() if self.transform else None,
            "semantic_labels": list(self.semantic_labels),
            "relationships": [r.to_dict() for r in self.relationships],
            "provenance": self.provenance.value,
            "uncertainty": self.uncertainty.to_dict(),
            "extra_fields": self.extra_fields,
        }

    @staticmethod
    def from_dict(data: dict) -> "Entity":
        return Entity(
            id=data["id"],
            type=data["type"],
            transform=Transform.from_dict(data["transform"]) if data.get("transform") else None,
            semantic_labels=list(data.get("semantic_labels", [])),
            relationships=[Relationship.from_dict(r) for r in data.get("relationships", [])],
            provenance=Provenance(data.get("provenance", Provenance.UNKNOWN.value)),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
            extra_fields=data.get("extra_fields", {}),
        )


class EntityRegistry:
    """Owns entities and enforces relationship integrity: no entity may
    reference a relationship target that doesn't exist in the registry.
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}

    def add(self, entity: Entity) -> None:
        if entity.id in self._entities:
            raise DuplicateEntityError(f"entity '{entity.id}' already registered")
        for rel in entity.relationships:
            if rel.target_id not in self._entities and rel.target_id != entity.id:
                raise DanglingRelationshipError(
                    f"entity '{entity.id}' has relationship '{rel.kind}' -> "
                    f"'{rel.target_id}' which does not exist"
                )
        self._entities[entity.id] = entity

    def get(self, entity_id: str) -> Entity:
        try:
            return self._entities[entity_id]
        except KeyError:
            raise UnknownEntityError(f"no entity '{entity_id}'") from None

    def remove(self, entity_id: str) -> None:
        if entity_id not in self._entities:
            raise UnknownEntityError(f"no entity '{entity_id}'")
        referencing = [
            e.id for e in self._entities.values()
            for r in e.relationships if r.target_id == entity_id
        ]
        if referencing:
            raise DanglingRelationshipError(
                f"cannot remove '{entity_id}': referenced by {referencing}"
            )
        del self._entities[entity_id]

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entities

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self):
        return iter(self._entities.values())

    def by_type(self, type_: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.type == type_]

    def validate_relationships(self) -> None:
        """Re-check integrity of every relationship in the registry --
        useful after bulk loads (e.g. deserialization) that bypass add()."""
        for entity in self._entities.values():
            for rel in entity.relationships:
                if rel.target_id not in self._entities:
                    raise DanglingRelationshipError(
                        f"entity '{entity.id}' has relationship '{rel.kind}' -> "
                        f"'{rel.target_id}' which does not exist"
                    )

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._entities.values()]

    @staticmethod
    def from_list(data: list[dict]) -> "EntityRegistry":
        registry = EntityRegistry()
        for entity_data in data:
            registry._entities[entity_data["id"]] = Entity.from_dict(entity_data)
        registry.validate_relationships()
        return registry
