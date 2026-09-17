"""Event system: deterministic publish-subscribe messaging.

Invariants:
  - All events dispatched in order
  - Handlers called synchronously in registration order
  - Event data is immutable after creation
  - No handler may recursively publish events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class EventPriority(int, Enum):
    """Event handler execution priority."""
    CRITICAL = 0   # First
    HIGH = 1
    NORMAL = 2
    LOW = 3
    DEFERRED = 4   # Last


@dataclass(frozen=True)
class Event:
    """Base event: all events inherit from this.

    Immutable after creation.
    """
    event_type: str
    tick: int
    timestamp: float
    data: dict = field(default_factory=dict)  # Immutable dict


class EventHandler:
    """Wrapper for an event handler with priority."""

    def __init__(
        self,
        handler: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL,
    ):
        self.handler = handler
        self.priority = priority

    def __call__(self, event: Event) -> None:
        self.handler(event)


class EventBus:
    """Deterministic event publishing and subscription.

    Events are published atomically; handlers are called in priority order,
    then registration order. No handler may publish events (would create
    nondeterminism risk).

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        # event_type -> list of handlers (sorted by priority, then registration)
        self._handlers: dict[str, list[EventHandler]] = {}
        self._publishing: bool = False
        self._pending_events: list[Event] = []

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> EventHandler:
        """Subscribe to an event type.

        Handlers are called in priority order (lower = earlier), then
        registration order within same priority.

        Args:
            event_type: Type of event to subscribe to
            handler: Callable to receive events
            priority: Handler priority

        Returns:
            EventHandler for later unsubscribe
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        event_handler = EventHandler(handler, priority)
        handlers = self._handlers[event_type]

        # Insert in priority order
        insert_pos = 0
        for i, existing in enumerate(handlers):
            if existing.priority > priority:
                insert_pos = i
                break
            insert_pos = i + 1

        handlers.insert(insert_pos, event_handler)
        return event_handler

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Raises:
            ValueError: If handler not found
        """
        if event_type not in self._handlers:
            raise ValueError(f"No handlers for event type '{event_type}'")

        handlers = self._handlers[event_type]
        try:
            handlers.remove(handler)
        except ValueError:
            raise ValueError(f"Handler not found for event type '{event_type}'")

        if not handlers:
            del self._handlers[event_type]

    def publish(
        self,
        event_type: str,
        tick: int,
        timestamp: float,
        data: Optional[dict] = None,
    ) -> None:
        """Publish an event.

        Handlers are called immediately, in priority order. If a handler
        tries to publish an event, raises RuntimeError.

        Args:
            event_type: Type of event
            tick: Simulation tick
            timestamp: World time (seconds)
            data: Event-specific data (becomes immutable)

        Raises:
            RuntimeError: If called while publishing another event
        """
        if self._publishing:
            raise RuntimeError("Cannot publish event while publishing another event")

        if data is None:
            data = {}

        event = Event(event_type=event_type, tick=tick, timestamp=timestamp, data=data)

        self._publishing = True
        try:
            if event_type in self._handlers:
                for handler in self._handlers[event_type]:
                    handler(event)
        finally:
            self._publishing = False

    def handler_count(self, event_type: str) -> int:
        """Get number of handlers for an event type."""
        return len(self._handlers.get(event_type, []))

    def all_event_types(self) -> list[str]:
        """Get all event types with active handlers."""
        return list(self._handlers.keys())

    def clear(self) -> None:
        """Clear all handlers (for testing)."""
        self._handlers.clear()
        self._publishing = False
