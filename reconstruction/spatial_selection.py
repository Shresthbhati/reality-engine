"""Unified spatial selection (N-26).

Allows users to select an object from the viewport, object tree, search results
or natural-language query and keeps the selection synchronized everywhere.

Honesty rules:
* A selection references a real entity id; selecting a non-existent object is
  reported as "not found", not silently ignored.
* Selection state is explicit: selected / hovered / cleared, never ambiguous.
* Multiple selection is supported with an explicit mode (single vs multi).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from world_ir import Entity


class SelectionMode(Enum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass(frozen=True)
class Selection:
    """Current selection state."""
    mode: SelectionMode
    selected_ids: List[str]          # ordered, most-recently-selected first
    hovered_id: Optional[str]
    source: str                      # "viewport" | "object_tree" | "search" | "natural_language"
    timestamp: str

    def is_selected(self, entity_id: str) -> bool:
        return entity_id in self.selected_ids

    def primary(self) -> Optional[str]:
        return self.selected_ids[0] if self.selected_ids else None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionChange:
    """A record of what changed in a selection update."""
    added: List[str]
    removed: List[str]
    hovered_changed: Optional[str]   # new hovered id or None
    source: str


class SpatialSelectionManager:
    """Manages spatial selection state across all interfaces."""

    def __init__(self, mode: SelectionMode = SelectionMode.SINGLE):
        self.mode = mode
        self._selected: List[str] = []
        self._hovered: Optional[str] = None
        self._source: str = "viewport"
        self._timestamp: str = ""

    def select(
        self,
        entity_id: str,
        *,
        source: str = "viewport",
        multi: Optional[bool] = None,
        timestamp: Optional[str] = None,
    ) -> SelectionChange:
        """Select one entity."""
        import time
        now = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        added: List[str] = []
        removed: List[str] = []

        use_multi = self.mode == SelectionMode.MULTI if multi is None else multi

        if use_multi and entity_id in self._selected:
            # toggle off
            self._selected.remove(entity_id)
            removed.append(entity_id)
        else:
            if not use_multi:
                removed.extend(self._selected)
                self._selected = []
            if entity_id not in self._selected:
                self._selected.insert(0, entity_id)
                added.append(entity_id)

        self._source = source
        self._timestamp = now

        return SelectionChange(
            added=added, removed=removed,
            hovered_changed=None, source=source,
        )

    def deselect(self, entity_id: str, *, source: str = "viewport") -> SelectionChange:
        """Remove one entity from the selection."""
        import time
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        removed: List[str] = []
        if entity_id in self._selected:
            self._selected.remove(entity_id)
            removed.append(entity_id)
        if self._hovered == entity_id:
            self._hovered = None
        self._source = source
        self._timestamp = now
        return SelectionChange(
            added=[], removed=removed,
            hovered_changed=None, source=source,
        )

    def clear(self, *, source: str = "viewport") -> SelectionChange:
        """Clear all selection."""
        import time
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        removed = list(self._selected)
        self._selected = []
        hovered = self._hovered
        self._hovered = None
        self._source = source
        self._timestamp = now
        return SelectionChange(
            added=[], removed=removed,
            hovered_changed=hovered, source=source,
        )

    def hover(self, entity_id: Optional[str], *, source: str = "viewport") -> Optional[str]:
        """Set the hovered entity (None clears hover). Returns the previous hovered id."""
        previous = self._hovered
        self._hovered = entity_id
        self._source = source
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return previous

    def current(self) -> Selection:
        """Return the current selection snapshot."""
        return Selection(
            mode=self.mode,
            selected_ids=list(self._selected),
            hovered_id=self._hovered,
            source=self._source,
            timestamp=self._timestamp,
        )

    def resolve(self, entity_id: str, entities: Dict[str, Entity]) -> Optional[Entity]:
        """Look up a selected entity id in the world, returning None if not found."""
        return entities.get(entity_id)

    def selected_entities(self, entities: Dict[str, Entity]) -> List[Entity]:
        """Return the currently selected entities that exist in the world."""
        return [entities[eid] for eid in self._selected if eid in entities]


__all__ = ["SpatialSelectionManager", "Selection", "SelectionChange", "SelectionMode"]