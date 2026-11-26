"""Durable, append-only event store for Reality Engine's event system.

Programme 8 (Event & Incremental Processing) requires event persistence.
``events.bus.EventBus`` is an in-memory ordered log; this module persists
those events to disk as JSON Lines and reloads them -- a fresh process over
the same root sees every event, and replay is reproducible because event ids
are deterministic (``events.bus`` derives them from seed/tick/sequence).

Design
------
* Append-only: events are never rewritten or deleted here. Integrity and
  retention are handled elsewhere; this layer only records.
* Idempotent append: appending an ``event_id`` already present is a no-op
  (returns ``False``), so at-least-once delivery cannot create duplicates.
* Crash-safe reads: blank or malformed trailing lines (e.g. a partial last
  write) are skipped on read rather than raising.
* Ordered: events are returned in append order. Replay over the same
  sequence is deterministic.

No third-party dependencies. One file per store: ``<root>/events.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .event import Event

__all__ = ["EventStore"]

_FILENAME = "events.jsonl"


class EventStore:
    """Append-only JSONL event log on disk."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / _FILENAME
        self._seen: set[str] = {e.event_id for e in self.read_all()}

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._seen)

    def append(self, event: Event) -> bool:
        """Append one event. Returns False (no-op) if event_id already stored."""
        if event.event_id in self._seen:
            return False
        line = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._seen.add(event.event_id)
        return True

    def extend(self, events: Iterable[Event]) -> int:
        """Append many; returns count newly added."""
        added = 0
        for e in events:
            if self.append(e):
                added += 1
        return added

    def _iter_lines(self) -> Iterator[dict]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue  # skip partial/corrupt trailing line
                if isinstance(obj, dict):
                    yield obj

    def read_all(self) -> list[Event]:
        """All events in append order. Malformed lines are skipped."""
        out: list[Event] = []
        for obj in self._iter_lines():
            try:
                out.append(Event.from_dict(obj))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def read_since(self, event_id: str | None) -> list[Event]:
        """Events appended strictly after ``event_id`` (all if None/unknown)."""
        events = self.read_all()
        if event_id is None:
            return events
        for i, e in enumerate(events):
            if e.event_id == event_id:
                return events[i + 1:]
        return events

    def read_types(self, event_type: str) -> list[Event]:
        return [e for e in self.read_all() if e.type == event_type]
