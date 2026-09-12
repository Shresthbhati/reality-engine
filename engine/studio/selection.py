"""Selection: Studio selection state (spec §27 - Studio foundation).

Pure state, not a WorldIR mutation -- selecting an entity never changes the
entity itself, only what the Studio currently has active/highlighted. Order
is preserved (an outliner shift-click builds an ordered multi-selection),
and `active` is always the most recently selected id so single-selection
tools (the property inspector) have an unambiguous target.
"""

from __future__ import annotations

from typing import List, Optional


class Selection:
    def __init__(self):
        self._ordered: List[str] = []

    @property
    def active(self) -> Optional[str]:
        return self._ordered[-1] if self._ordered else None

    @property
    def ids(self) -> List[str]:
        return list(self._ordered)

    def is_selected(self, entity_id: str) -> bool:
        return entity_id in self._ordered

    def select(self, entity_id: str, *, additive: bool = False) -> None:
        """Select one entity. additive=False (default) replaces the selection
        (a plain click); additive=True adds to it (a ctrl/shift-click) without
        duplicating an already-selected id."""
        if not additive:
            self._ordered = [entity_id]
            return
        if entity_id in self._ordered:
            self._ordered.remove(entity_id)
        self._ordered.append(entity_id)

    def deselect(self, entity_id: str) -> None:
        if entity_id in self._ordered:
            self._ordered.remove(entity_id)

    def toggle(self, entity_id: str) -> None:
        if entity_id in self._ordered:
            self.deselect(entity_id)
        else:
            self.select(entity_id, additive=True)

    def clear(self) -> None:
        self._ordered = []
