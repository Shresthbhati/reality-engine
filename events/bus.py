"""EventBus: an append-only, deterministically ordered event log with
type-scoped subscriptions.

Event ids are derived from (seed, tick, sequence) rather than a random
UUID -- spec sec 90 determinism requires replay to be reproducible, and
a random id would make two runs of the same scenario diverge in their
event log even when every physical quantity matched.
"""

from __future__ import annotations

from typing import Callable, Optional

from .event import Event

WILDCARD = "*"

Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self._sequence = 0
        self._events: list[Event] = []
        self._subscribers: dict[str, list[Handler]] = {}

    def _next_id(self, tick: int) -> str:
        event_id = f"evt-{self.seed}-{tick:08d}-{self._sequence:04d}"
        self._sequence += 1
        return event_id

    def emit(self, type: str, timestamp: float, tick: int, **fields) -> Event:
        event = Event(
            event_id=self._next_id(tick),
            type=type,
            timestamp=timestamp,
            tick=tick,
            deterministic_seed=fields.pop("deterministic_seed", self.seed),
            **fields,
        )
        self._events.append(event)
        for handler in self._subscribers.get(type, ()):
            handler(event)
        for handler in self._subscribers.get(WILDCARD, ()):
            handler(event)
        return event

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def events_of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.type == event_type]

    def clear(self) -> None:
        self._events.clear()
        self._sequence = 0

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._events]
