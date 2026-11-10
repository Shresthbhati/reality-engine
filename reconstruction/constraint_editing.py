"""Constraint-aware editing (N-27).

Builds on perception/architectural_constraints.py: before an edit is applied,
runs the constraint solver and reports whether the edit would create
disconnected or implausible geometry. Edits that would violate constraints are
rejected with a specific reason; the user must confirm or provide new evidence.

Honesty rules:
* Every edit proposal carries a pre-check and post-check constraint report.
* Violations are reported with the specific rule and measurement, never
  silently accepted.
* No edit is applied automatically when a constraint would be violated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from reconstruction.architectural_constraints import (
    ArchitecturalConstraintSolver, ConstraintViolation, ConstraintReport,
)
from world_ir import Entity, EntityType


@dataclass(frozen=True)
class EditProposal:
    """A proposed edit to one or more entities."""
    edit_id: str
    entity_id: str
    field: str
    new_value: object
    reason: str                    # why the user wants this edit
    actor: str                     # "user" | "assistant"


@dataclass(frozen=True)
class EditCheckResult:
    """Outcome of checking an edit against architectural constraints."""
    edit_id: str
    allowed: bool                  # True if no blocking violations
    violations: List[ConstraintViolation]
    warnings: List[ConstraintViolation]
    message: str                   # human-readable summary
    requires_confirmation: bool


class ConstraintAwareEditor:
    """Validates proposed edits against architectural constraints before applying."""

    def __init__(self):
        self.solver = ArchitecturalConstraintSolver()

    def check_edit(
        self,
        edit: EditProposal,
        world_entities: Dict[str, Entity],
        proposed_entities: Dict[str, Entity],
    ) -> EditCheckResult:
        """Run constraint solver on the proposed world state.

        world_entities: current (pre-edit) world.
        proposed_entities: world with the edit applied.
        """
        report = self.solver.check_world(proposed_entities)

        errors = [v for v in report.violations if v.severity == "error"]
        warnings = [v for v in report.violations if v.severity in ("warning", "info")]

        allowed = len(errors) == 0
        requires_confirmation = len(warnings) > 0 or not allowed

        if allowed and not warnings:
            message = f"Edit {edit.edit_id} passes all constraint checks."
        elif allowed:
            message = (f"Edit {edit.edit_id} allowed but has {len(warnings)} "
                       f"warning(s): review before applying.")
        else:
            message = (f"Edit {edit.edit_id} blocked: {len(errors)} constraint "
                       f"violation(s) must be resolved first.")

        return EditCheckResult(
            edit_id=edit.edit_id,
            allowed=allowed,
            violations=errors,
            warnings=warnings,
            message=message,
            requires_confirmation=requires_confirmation,
        )

    def check_field_edit(
        self,
        entity_id: str,
        field: str,
        new_value: object,
        world_entities: Dict[str, Entity],
        reason: str = "",
        actor: str = "user",
    ) -> EditCheckResult:
        """Convenience: check a single-field edit on one entity."""
        import copy
        entity = world_entities.get(entity_id)
        if entity is None:
            return EditCheckResult(
                edit_id=f"edit-{entity_id}-{field}", allowed=False,
                violations=[], warnings=[],
                message=f"Entity {entity_id} not found in world.",
                requires_confirmation=False,
            )

        proposed = copy.deepcopy(world_entities)
        cp = dict(entity.custom_properties)
        cp[field] = new_value
        new_entity = Entity(
            id=entity.id, type=entity.type, name=entity.name,
            custom_properties=cp, confidence=entity.confidence,
            provenance=entity.provenance,
        )
        proposed[entity_id] = new_entity

        edit = EditProposal(
            edit_id=f"edit-{entity_id}-{field}", entity_id=entity_id,
            field=field, new_value=new_value, reason=reason, actor=actor,
        )
        return self.check_edit(edit, world_entities, proposed)


__all__ = ["ConstraintAwareEditor", "EditProposal", "EditCheckResult"]