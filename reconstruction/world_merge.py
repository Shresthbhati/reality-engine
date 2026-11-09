"""Three-way world merge (plan item E.22).

Compares the original world, a new capture, and the current edited world.
Separates physical changes from user edits and reconstruction differences.
A new capture cannot silently overwrite a user's manual corrections.

Honesty rules:
* Every difference carries both sides' values and a classification:
  physical_change | user_edit | reconstruction_difference | unchanged.
* Manual edits are protected: a new capture cannot override an entity whose
  ``user_edited`` flag is set without explicit confirmation.
* Conflicts are reported, never auto-resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class EntityDiff:
    """One entity's difference across the three worlds."""
    entity_id: str
    entity_type: str
    name: Optional[str]
    classification: str  # "physical_change" | "user_edit" | "reconstruction_difference" | "unchanged" | "added" | "removed"
    original_value: Optional[dict]
    capture_value: Optional[dict]
    edited_value: Optional[dict]
    message: str
    protected: bool  # True if user_edited and must not be overwritten


@dataclass(frozen=True)
class ThreeWayMergeReport:
    """Result of comparing original, capture, and edited worlds."""
    diffs: List[EntityDiff]
    conflicts: List[EntityDiff]            # diffs requiring user decision
    safe_to_apply: List[EntityDiff]        # diffs that can be applied without review
    summary: Dict[str, int]
    provisional: bool = True


class ThreeWayWorldMerge:
    """Compares three world versions and classifies every entity difference."""

    def merge(
        self,
        original: Dict[str, Entity],
        capture: Dict[str, Entity],
        edited: Dict[str, Entity],
    ) -> ThreeWayMergeReport:
        """Classify differences between original, capture, and edited worlds.

        Returns a report with diffs classified as physical_change, user_edit,
        reconstruction_difference, unchanged, added, or removed.
        """
        all_ids = set(original) | set(capture) | set(edited)
        diffs: List[EntityDiff] = []

        for eid in sorted(all_ids):
            o = original.get(eid)
            c = capture.get(eid)
            ed = edited.get(eid)

            if o is None and c is None and ed is not None:
                diffs.append(EntityDiff(
                    entity_id=eid, entity_type=ed.type.value if ed.type else "unknown",
                    name=ed.name, classification="added",
                    original_value=None, capture_value=None,
                    edited_value=self._entity_snapshot(ed),
                    message=f"Entity '{ed.name or eid}' exists only in the edited world.",
                    protected=self._is_user_edited(ed),
                ))
                continue
            if o is not None and c is None and ed is None:
                diffs.append(EntityDiff(
                    entity_id=eid, entity_type=o.type.value if o.type else "unknown",
                    name=o.name, classification="removed",
                    original_value=self._entity_snapshot(o), capture_value=None,
                    edited_value=None,
                    message=f"Entity '{o.name or eid}' existed in the original but is gone.",
                    protected=False,
                ))
                continue
            if o is not None and c is not None and ed is not None:
                diff = self._compare_three(eid, o, c, ed)
                diffs.append(diff)
                continue
            # two-world combinations (e.g. original + edited, no capture)
            diffs.append(EntityDiff(
                entity_id=eid,
                entity_type=(o or c or ed).type.value if (o or c or ed) and (o or c or ed).type else "unknown",
                name=(o or c or ed).name if (o or c or ed) else None,
                classification="reconstruction_difference",
                original_value=self._entity_snapshot(o) if o else None,
                capture_value=self._entity_snapshot(c) if c else None,
                edited_value=self._entity_snapshot(ed) if ed else None,
                message=f"Entity '{eid}' differs between available world versions.",
                protected=self._is_user_edited(ed) if ed else False,
            ))

        conflicts = [d for d in diffs if d.classification in ("physical_change", "user_edit")
                     and d.protected]
        safe = [d for d in diffs if d.classification == "reconstruction_difference"
                and not d.protected]

        summary: Dict[str, int] = {}
        for d in diffs:
            summary[d.classification] = summary.get(d.classification, 0) + 1

        return ThreeWayMergeReport(
            diffs=diffs, conflicts=conflicts, safe_to_apply=safe,
            summary=summary, provisional=True,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _entity_snapshot(e: Entity) -> dict:
        cp = e.custom_properties
        return {
            "type": e.type.value if e.type else None,
            "name": e.name,
            "bounds_min": cp.get("bounds_min"),
            "bounds_max": cp.get("bounds_max"),
            "confidence": e.confidence,
            "provenance": str(e.provenance),
            "custom_properties": {k: v for k, v in cp.items()
                                  if k not in ("bounds_min", "bounds_max")},
        }

    @staticmethod
    def _is_user_edited(e: Optional[Entity]) -> bool:
        if e is None:
            return False
        return bool(e.custom_properties.get("user_edited"))

    @staticmethod
    def _compare_three(
        eid: str, original: Entity, capture: Entity, edited: Entity
    ) -> EntityDiff:
        osnap = ThreeWayWorldMerge._entity_snapshot(original)
        csnap = ThreeWayWorldMerge._entity_snapshot(capture)
        esnap = ThreeWayWorldMerge._entity_snapshot(edited)

        # ---- edited != original AND edited == capture: physical change ----
        if esnap != osnap and esnap == csnap:
            classification = "physical_change"
            message = (f"Entity '{original.name or eid}' changed in the capture "
                       f"and the edit matches: physical change.")
        # ---- edited != capture AND edited == original: user reverted -------
        elif esnap != csnap and esnap == osnap:
            classification = "user_edit"
            message = (f"Entity '{original.name or eid}' was edited back to the "
                       f"original: user edit.")
        # ---- edited differs from both: conflict ----------------------------
        elif esnap != osnap and esnap != csnap:
            classification = "user_edit" if ThreeWayWorldMerge._is_user_edited(edited) else "reconstruction_difference"
            message = (f"Entity '{original.name or eid}' differs in all three "
                       f"versions: review required.")
        # ---- all equal -----------------------------------------------------
        else:
            classification = "unchanged"
            message = f"Entity '{original.name or eid}' is unchanged."

        return EntityDiff(
            entity_id=eid,
            entity_type=original.type.value if original.type else "unknown",
            name=original.name,
            classification=classification,
            original_value=osnap,
            capture_value=csnap,
            edited_value=esnap,
            message=message,
            protected=ThreeWayWorldMerge._is_user_edited(edited),
        )


__all__ = ["ThreeWayWorldMerge", "EntityDiff", "ThreeWayMergeReport"]