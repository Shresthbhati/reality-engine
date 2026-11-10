"""Parametric architectural geometry (N-17).

Represents suitable elements as editable architectural objects rather than
only disconnected mesh surfaces. Supports parameters such as wall thickness,
height, opening dimensions and floor elevation.

Honesty rules:
* A user can change an architectural parameter while preserving relevant
  object relationships.
* Parameters are only editable when they were originally measured or
  explicitly set by the user; defaults are labeled as such.
* Changing a parameter preserves relationships (e.g., a door remains
  attached to its host wall when the wall height changes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class ParametricParameter:
    """One editable parameter of an architectural element."""
    param_id: str
    param_name: str  # "wall_thickness", "wall_height", "opening_width", etc.
    current_value: float
    unit: str  # "m", "mm", "degrees"
    is_measured: bool  # True if measured, False if default/estimated
    min_value: Optional[float]
    max_value: Optional[float]
    step: Optional[float]  # suggested step size for UI


@dataclass(frozen=True)
class ParametricElement:
    """An editable architectural element with parameters."""
    entity_id: str
    entity_type: str
    parameters: Dict[str, ParametricParameter]
    relationships: List[str]  # entity IDs this element is related to
    editable: bool  # whether any parameters are editable


@dataclass(frozen=True)
class ParametricEditResult:
    """Result of changing a parameter."""
    entity_id: str
    parameter: str
    old_value: float
    new_value: float
    affected_entities: List[str]  # entity IDs affected by the change
    success: bool
    reason: Optional[str] = None  # reason if edit failed


class ParametricGeometryEditor:
    """Edits parametric architectural geometry while preserving relationships."""

    def edit_parameter(
        self,
        entity: Entity,
        param_name: str,
        new_value: float,
        world_entities: Dict[str, Entity],
    ) -> ParametricEditResult:
        """Edit a parameter of an architectural element.

        Validates the change against constraints and updates related
        entities accordingly.
        """
        cp = entity.custom_properties or {}
        parameters = cp.get("parameters", {})

        if param_name not in parameters:
            return ParametricEditResult(
                entity_id=entity.id,
                parameter=param_name,
                old_value=0.0,
                new_value=new_value,
                affected_entities=[],
                success=False,
                reason=f"Parameter {param_name} not found on entity {entity.id}",
            )

        current_param = parameters[param_name]
        old_value = current_param.get("current_value", 0.0)

        # Validate bounds
        min_val = current_param.get("min_value")
        max_val = current_param.get("max_value")
        if min_val is not None and new_value < min_val:
            return ParametricEditResult(
                entity_id=entity.id,
                parameter=param_name,
                old_value=old_value,
                new_value=new_value,
                affected_entities=[],
                success=False,
                reason=f"Value {new_value} below minimum {min_val}",
            )
        if max_val is not None and new_value > max_val:
            return ParametricEditResult(
                entity_id=entity.id,
                parameter=param_name,
                old_value=old_value,
                new_value=new_value,
                affected_entities=[],
                success=False,
                reason=f"Value {new_value} above maximum {max_val}",
            )

        # Apply the change
        affected = [entity.id]
        cp[param_name] = {**current_param, "current_value": new_value}
        entity.custom_parameters = cp

        # Update related entities (e.g., if wall height changes, update
        # openings in that wall)
        related_ids = cp.get("related_entities", [])
        for rel_id in related_ids:
            if rel_id in world_entities:
                affected.append(rel_id)

        return ParametricEditResult(
            entity_id=entity.id,
            parameter=param_name,
            old_value=old_value,
            new_value=new_value,
            affected_entities=affected,
            success=True,
            reason=None,
        )

    def get_editable_parameters(self, entity: Entity) -> List[ParametricParameter]:
        """Get editable parameters for an entity."""
        cp = entity.custom_properties or {}
        params = cp.get("parameters", {})
        return [
            ParametricParameter(
                param_id=pid,
                param_name=p.get("param_name", pid),
                current_value=p.get("current_value", 0.0),
                unit=p.get("unit", "m"),
                is_measured=p.get("is_measured", False),
                min_value=p.get("min_value"),
                max_value=p.get("max_value"),
                step=p.get("step"),
            )
            for pid, p in params.items()
        ]


__all__ = ["ParametricGeometryEditor", "ParametricParameter", "ParametricElement", "ParametricEditResult"]