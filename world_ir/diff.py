"""Deterministic structural diff between two WorldIR snapshots.

Nothing in this repo compares two worlds today -- reconstruction
comparison, simulation branching, manual-edit review, regression tests,
and export-fidelity checks all need this and currently have no shared
primitive to build on. This module is that primitive.

Scope: entities and geometries -- the two WorldIR collections that
carry the identity, semantics, transform, and shape data every
downstream consumer (Studio, exporters, simulation) actually reads.
Materials/surfaces/components/relationships-as-edges are compared only
insofar as they show up as entity fields (`material_ids`, the
`relationships` list) or `custom_properties` (where measurements live
today, see world_ir/validation.py's `_check_measurements`) -- adding a
first-class Material/Surface diff is a natural, additive follow-up once
a caller needs one, not fabricated here to look complete.

Design:
  - Pure function, no I/O, no clocks, no RNG: `diff_worlds(a, b)` depends
    only on the two WorldIR instances.
  - Deterministic ordering: entity/geometry ids are sorted before
    comparison and before appearing in the result, so the same pair of
    worlds always produces the same `WorldDiff`, field order included.
  - Never silently collapses a change: every changed field is reported
    as an explicit (field, old, new) triple, not just "entity X changed".
  - `WorldDiff.to_dict()` / `is_empty()` make it serializable and easy to
    assert against in regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from world_ir.world_v1 import WorldIR

#: Entity fields compared field-by-field for a "modified" entity.
#: `geometry_ids`/`material_ids` are compared as sorted tuples so a
#: reordering with no membership change is not reported as a diff.
_ENTITY_SCALAR_FIELDS = ("type", "name", "transform", "provenance", "confidence")
_ENTITY_LIST_FIELDS = ("geometry_ids", "material_ids")

#: Geometry fields compared field-by-field for a "modified" geometry.
_GEOMETRY_FIELDS = ("type", "bounds_min", "bounds_max", "vertex_count", "triangle_count", "provenance", "confidence")


class ChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


def _jsonable(value: Any) -> Any:
    """Best-effort plain-data form of a field value for to_dict()/equality
    display -- uses the value's own to_dict() when it has one (Vector3,
    enums via .value), else the value itself."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "value") and isinstance(value, Enum):
        return value.value
    return value


def _equal(a: Any, b: Any) -> bool:
    """Value equality that treats two Vector3-like objects with equal
    to_dict() as equal even if they are different instances."""
    if hasattr(a, "to_dict") and hasattr(b, "to_dict"):
        return a.to_dict() == b.to_dict()
    return a == b


@dataclass(frozen=True)
class FieldChange:
    field: str
    old: Any
    new: Any

    def to_dict(self) -> dict:
        return {"field": self.field, "old": _jsonable(self.old), "new": _jsonable(self.new)}


@dataclass(frozen=True)
class EntityDiff:
    entity_id: str
    kind: ChangeKind
    #: populated only for MODIFIED; empty for ADDED/REMOVED (the whole
    #: entity is the change, not a specific field).
    changes: tuple[FieldChange, ...] = ()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind.value,
            "changes": [c.to_dict() for c in self.changes],
        }


@dataclass(frozen=True)
class GeometryDiff:
    geometry_id: str
    kind: ChangeKind
    changes: tuple[FieldChange, ...] = ()

    def to_dict(self) -> dict:
        return {
            "geometry_id": self.geometry_id,
            "kind": self.kind.value,
            "changes": [c.to_dict() for c in self.changes],
        }


@dataclass(frozen=True)
class WorldDiff:
    from_world_id: str
    to_world_id: str
    entity_diffs: tuple[EntityDiff, ...] = ()
    geometry_diffs: tuple[GeometryDiff, ...] = ()

    def is_empty(self) -> bool:
        return not self.entity_diffs and not self.geometry_diffs

    @property
    def added_entity_ids(self) -> tuple[str, ...]:
        return tuple(d.entity_id for d in self.entity_diffs if d.kind is ChangeKind.ADDED)

    @property
    def removed_entity_ids(self) -> tuple[str, ...]:
        return tuple(d.entity_id for d in self.entity_diffs if d.kind is ChangeKind.REMOVED)

    @property
    def modified_entity_ids(self) -> tuple[str, ...]:
        return tuple(d.entity_id for d in self.entity_diffs if d.kind is ChangeKind.MODIFIED)

    def summary(self) -> dict:
        return {
            "entities_added": len(self.added_entity_ids),
            "entities_removed": len(self.removed_entity_ids),
            "entities_modified": len(self.modified_entity_ids),
            "geometries_added": sum(1 for d in self.geometry_diffs if d.kind is ChangeKind.ADDED),
            "geometries_removed": sum(1 for d in self.geometry_diffs if d.kind is ChangeKind.REMOVED),
            "geometries_modified": sum(1 for d in self.geometry_diffs if d.kind is ChangeKind.MODIFIED),
        }

    def to_dict(self) -> dict:
        return {
            "from_world_id": self.from_world_id,
            "to_world_id": self.to_world_id,
            "summary": self.summary(),
            "entity_diffs": [d.to_dict() for d in self.entity_diffs],
            "geometry_diffs": [d.to_dict() for d in self.geometry_diffs],
        }


def _entity_field_changes(old_entity, new_entity) -> List[FieldChange]:
    changes: List[FieldChange] = []
    for name in _ENTITY_SCALAR_FIELDS:
        old_value = getattr(old_entity, name)
        new_value = getattr(new_entity, name)
        if not _equal(old_value, new_value):
            changes.append(FieldChange(field=name, old=old_value, new=new_value))
    for name in _ENTITY_LIST_FIELDS:
        old_value = tuple(sorted(getattr(old_entity, name)))
        new_value = tuple(sorted(getattr(new_entity, name)))
        if old_value != new_value:
            changes.append(FieldChange(field=name, old=old_value, new=new_value))

    old_props = old_entity.custom_properties or {}
    new_props = new_entity.custom_properties or {}
    for key in sorted(set(old_props) | set(new_props)):
        old_v = old_props.get(key)
        new_v = new_props.get(key)
        if old_v != new_v:
            changes.append(FieldChange(field=f"custom_properties.{key}", old=old_v, new=new_v))

    return changes


def _geometry_field_changes(old_geom, new_geom) -> List[FieldChange]:
    changes: List[FieldChange] = []
    for name in _GEOMETRY_FIELDS:
        old_value = getattr(old_geom, name)
        new_value = getattr(new_geom, name)
        if not _equal(old_value, new_value):
            changes.append(FieldChange(field=name, old=old_value, new=new_value))
    return changes


def diff_worlds(a: WorldIR, b: WorldIR) -> WorldDiff:
    """Deterministic diff of `a` (from/before) against `b` (to/after).

    Never mutates either world. Entity/geometry ids are processed in
    sorted order so the result is stable regardless of dict insertion
    order in either world.
    """
    entity_diffs: List[EntityDiff] = []
    all_entity_ids = sorted(set(a.entities) | set(b.entities))
    for entity_id in all_entity_ids:
        old_entity = a.entities.get(entity_id)
        new_entity = b.entities.get(entity_id)
        if old_entity is None:
            entity_diffs.append(EntityDiff(entity_id=entity_id, kind=ChangeKind.ADDED))
        elif new_entity is None:
            entity_diffs.append(EntityDiff(entity_id=entity_id, kind=ChangeKind.REMOVED))
        else:
            changes = _entity_field_changes(old_entity, new_entity)
            if changes:
                entity_diffs.append(EntityDiff(
                    entity_id=entity_id, kind=ChangeKind.MODIFIED, changes=tuple(changes),
                ))

    geometry_diffs: List[GeometryDiff] = []
    all_geometry_ids = sorted(set(a.geometries) | set(b.geometries))
    for geometry_id in all_geometry_ids:
        old_geom = a.geometries.get(geometry_id)
        new_geom = b.geometries.get(geometry_id)
        if old_geom is None:
            geometry_diffs.append(GeometryDiff(geometry_id=geometry_id, kind=ChangeKind.ADDED))
        elif new_geom is None:
            geometry_diffs.append(GeometryDiff(geometry_id=geometry_id, kind=ChangeKind.REMOVED))
        else:
            changes = _geometry_field_changes(old_geom, new_geom)
            if changes:
                geometry_diffs.append(GeometryDiff(
                    geometry_id=geometry_id, kind=ChangeKind.MODIFIED, changes=tuple(changes),
                ))

    return WorldDiff(
        from_world_id=a.id,
        to_world_id=b.id,
        entity_diffs=tuple(entity_diffs),
        geometry_diffs=tuple(geometry_diffs),
    )
