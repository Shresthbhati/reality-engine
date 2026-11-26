"""Event replay and idempotent consumption for Reality Engine's event system.

Programme 8 requires event replay, idempotent consumers, and rebuilding
state from the event log. These helpers operate on any ordered iterable of
``events.event.Event`` (e.g. ``EventBus.events`` or ``EventStore.read_all()``)
and compose with the existing ``EventBus`` -- no new event schema.

* ``replay``: left-fold events through a reducer to rebuild derived state.
  Replaying the same ordered events always yields the same state.
* ``IdempotentConsumer``: tracks processed ``event_id``s with a durable
  checkpoint so redelivery (at-least-once) does not re-apply side effects;
  survives process restart.
* ``replay_into_bus``: restore persisted events into a fresh ``EventBus``
  preserving original ids/order (uses additive ``EventBus.restore``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .event import Event

__all__ = ["replay", "IdempotentConsumer", "replay_into_bus", "Reducer", "Handler"]

Reducer = Callable[[Any, Event], Any]
Handler = Callable[[Event], None]


def replay(events: Iterable[Event], reducer: Reducer, initial: Any = None) -> Any:
    """Fold ``events`` (in order) through ``reducer(state, event) -> state``."""
    state = initial
    for event in events:
        state = reducer(state, event)
    return state


class IdempotentConsumer:
    """Applies each event at most once, durably.

    Processed ``event_id``s are kept in memory and mirrored to a JSON
    checkpoint file (atomic replace) so a restarted process does not
    re-apply already-handled events.
    """

    def __init__(self, checkpoint_path: str | os.PathLike[str], *, consumer_id: str = "default") -> None:
        self._path = Path(checkpoint_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.consumer_id = consumer_id
        self._processed: set[str] = set()
        self._last_id: str | None = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._processed = {str(x) for x in data.get("processed", [])}
            self._last_id = data.get("last_id")

    def _save(self) -> None:
        payload = json.dumps({"processed": sorted(self._processed), "last_id": self._last_id}, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".ckpt-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def has_processed(self, event_id: str) -> bool:
        return event_id in self._processed

    @property
    def last_processed_id(self) -> str | None:
        return self._last_id

    def consume(self, events: Iterable[Event], handler: Handler) -> int:
        """Call ``handler`` once per not-yet-processed event. Returns count applied.

        An event is marked processed only after ``handler`` returns without
        raising; a raising handler leaves it unprocessed for retry.
        """
        applied = 0
        for event in events:
            if event.event_id in self._processed:
                continue
            handler(event)  # if this raises, event is NOT marked processed
            self._processed.add(event.event_id)
            self._last_id = event.event_id
            self._save()
            applied += 1
        return applied

    def reset(self) -> None:
        self._processed.clear()
        self._last_id = None
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass


def replay_into_bus(events: Iterable[Event], bus: Any) -> int:
    """Restore persisted events into a fresh ``EventBus`` preserving ids/order."""
    count = 0
    for event in events:
        bus.restore(event)
        count += 1
    return count
