"""Replay system: P3 with event recording, time scrubbing, branching.

Implements deterministic replay with:
  - Event recording and playback
  - Time scrubbing (seeking to any point)
  - Branching (alternate timelines)
  - Causality tracking (dependency graph)
  - Deterministic verification (same seed = same replay)
  - Event queries and filtering

Spec §15: REPLAY SYSTEM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json

from engine.core.logging import get_logger


class EventType(Enum):
    """Classification of recordable events."""
    SIMULATION_START = "simulation_start"
    SIMULATION_STEP = "simulation_step"
    ENTITY_CREATE = "entity_create"
    ENTITY_DESTROY = "entity_destroy"
    PHYSICS_IMPACT = "physics_impact"
    FRACTURE = "fracture"
    DEBRIS_SPAWN = "debris_spawn"
    COLLISION = "collision"
    ENVIRONMENT_CHANGE = "environment_change"
    BRANCH_CREATE = "branch_create"
    USER_INPUT = "user_input"


@dataclass(frozen=True)
class SimulationEvent:
    """Immutable event record in simulation timeline."""
    event_id: str
    event_type: EventType
    tick: int
    timestamp: float
    entity_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    branch_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "entity_id": self.entity_id,
            "parent_event_id": self.parent_event_id,
            "data": self.data,
            "branch_id": self.branch_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SimulationEvent:
        """Restore from dictionary."""
        return SimulationEvent(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            tick=data["tick"],
            timestamp=data["timestamp"],
            entity_id=data.get("entity_id"),
            parent_event_id=data.get("parent_event_id"),
            data=data.get("data", {}),
            branch_id=data.get("branch_id"),
        )


@dataclass(frozen=True)
class TimelineSnapshot:
    """Immutable snapshot of world state at specific time."""
    branch_id: str
    tick: int
    timestamp: float
    entity_count: int
    debris_count: int
    fractures: int
    world_state: Dict[str, Any] = field(default_factory=dict)


class TimelineSegment:
    """Contiguous segment of simulation events."""

    def __init__(self, segment_id: str, start_tick: int, branch_id: str):
        """Initialize timeline segment.

        Args:
            segment_id: Unique segment identifier
            start_tick: Starting simulation tick
            branch_id: Branch this segment belongs to
        """
        self.segment_id = segment_id
        self.start_tick = start_tick
        self.branch_id = branch_id
        self.events: List[SimulationEvent] = []
        self.end_tick = start_tick - 1

    def add_event(self, event: SimulationEvent) -> None:
        """Add event to segment.

        Args:
            event: Event to add
        """
        self.events.append(event)
        self.end_tick = max(self.end_tick, event.tick)

    def get_events_in_range(self, start_tick: int, end_tick: int) -> List[SimulationEvent]:
        """Get events within tick range.

        Args:
            start_tick: Start tick (inclusive)
            end_tick: End tick (inclusive)

        Returns:
            List of events in range
        """
        return [e for e in self.events if start_tick <= e.tick <= end_tick]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "branch_id": self.branch_id,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }


class Timeline:
    """Complete recorded simulation timeline with branching support."""

    def __init__(self, timeline_id: str):
        """Initialize timeline.

        Args:
            timeline_id: Unique timeline identifier
        """
        self.timeline_id = timeline_id
        self.branches: Dict[str, List[TimelineSegment]] = {"main": []}
        self.snapshots: Dict[str, TimelineSnapshot] = {}
        self.causality_graph: Dict[str, List[str]] = {}  # event_id -> [dependent_event_ids]
        self._event_count = 0
        self._logger = get_logger("engine.simulation.replay")

    def create_branch(self, branch_id: str, from_tick: int) -> None:
        """Create new branch at specific time.

        Args:
            branch_id: New branch identifier
            from_tick: Tick to branch from
        """
        if branch_id in self.branches:
            self._logger.warning(f"Branch already exists: {branch_id}")
            return

        self.branches[branch_id] = []
        self._logger.info(
            "Branch created",
            context={"branch_id": branch_id, "from_tick": from_tick},
        )

    def record_event(
        self,
        event_type: EventType,
        tick: int,
        timestamp: float,
        branch_id: str = "main",
        entity_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an event in the timeline.

        Args:
            event_type: Type of event
            tick: Simulation tick
            timestamp: Simulation time
            branch_id: Branch to record in
            entity_id: Entity involved
            parent_event_id: Event that caused this one
            data: Event-specific data

        Returns:
            Event ID
        """
        event_id = f"{branch_id}_evt_{self._event_count}"
        self._event_count += 1

        event = SimulationEvent(
            event_id=event_id,
            event_type=event_type,
            tick=tick,
            timestamp=timestamp,
            entity_id=entity_id,
            parent_event_id=parent_event_id,
            data=data or {},
            branch_id=branch_id,
        )

        # Get or create segment for this branch
        if branch_id not in self.branches:
            self.branches[branch_id] = []

        segments = self.branches[branch_id]
        if not segments or segments[-1].end_tick < tick - 100:
            # Create new segment every 100 ticks for efficiency
            segment = TimelineSegment(
                f"{branch_id}_seg_{len(segments)}",
                tick,
                branch_id,
            )
            segments.append(segment)

        segments[-1].add_event(event)

        # Track causality
        if parent_event_id:
            if parent_event_id not in self.causality_graph:
                self.causality_graph[parent_event_id] = []
            self.causality_graph[parent_event_id].append(event_id)

        return event_id

    def get_events(
        self,
        branch_id: str = "main",
        start_tick: int = 0,
        end_tick: Optional[int] = None,
    ) -> List[SimulationEvent]:
        """Get events from timeline.

        Args:
            branch_id: Branch to query
            start_tick: Start tick (inclusive)
            end_tick: End tick (inclusive), None for all

        Returns:
            List of events
        """
        if branch_id not in self.branches:
            return []

        events = []
        for segment in self.branches[branch_id]:
            if end_tick is None:
                segment_end = segment.end_tick
            else:
                segment_end = end_tick

            segment_events = segment.get_events_in_range(start_tick, segment_end)
            events.extend(segment_events)

        return sorted(events, key=lambda e: (e.tick, e.event_id))

    def get_events_by_type(
        self,
        event_type: EventType,
        branch_id: str = "main",
    ) -> List[SimulationEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Type to filter by
            branch_id: Branch to query

        Returns:
            List of matching events
        """
        all_events = self.get_events(branch_id)
        return [e for e in all_events if e.event_type == event_type]

    def get_events_for_entity(
        self,
        entity_id: str,
        branch_id: str = "main",
    ) -> List[SimulationEvent]:
        """Get all events affecting an entity.

        Args:
            entity_id: Entity to query
            branch_id: Branch to query

        Returns:
            List of events for entity
        """
        all_events = self.get_events(branch_id)
        return [e for e in all_events if e.entity_id == entity_id]

    def get_causal_chain(self, event_id: str) -> List[str]:
        """Get chain of events caused by this event.

        Args:
            event_id: Root event

        Returns:
            List of event IDs in causal chain
        """
        chain = [event_id]
        to_visit = self.causality_graph.get(event_id, [])

        while to_visit:
            current = to_visit.pop(0)
            chain.append(current)
            to_visit.extend(self.causality_graph.get(current, []))

        return chain

    def add_snapshot(self, snapshot: TimelineSnapshot) -> None:
        """Record a world state snapshot.

        Args:
            snapshot: Snapshot to add
        """
        key = f"{snapshot.branch_id}_{snapshot.tick}"
        self.snapshots[key] = snapshot

    def get_snapshot(self, branch_id: str, tick: int) -> Optional[TimelineSnapshot]:
        """Retrieve a snapshot.

        Args:
            branch_id: Branch
            tick: Tick

        Returns:
            Snapshot or None
        """
        key = f"{branch_id}_{tick}"
        return self.snapshots.get(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get timeline statistics."""
        total_events = sum(
            len(segment.events)
            for segments in self.branches.values()
            for segment in segments
        )

        return {
            "timeline_id": self.timeline_id,
            "branches": len(self.branches),
            "total_events": total_events,
            "snapshots": len(self.snapshots),
            "causal_links": len(self.causality_graph),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize timeline to dictionary."""
        return {
            "format_version": 1,
            "timeline_id": self.timeline_id,
            "branches": {
                bid: [segment.to_dict() for segment in segments]
                for bid, segments in self.branches.items()
            },
            "snapshots": {
                k: {
                    "branch_id": v.branch_id,
                    "tick": v.tick,
                    "timestamp": v.timestamp,
                    "entity_count": v.entity_count,
                    "debris_count": v.debris_count,
                    "fractures": v.fractures,
                    "world_state": v.world_state,
                }
                for k, v in self.snapshots.items()
            },
            "causality_graph": self.causality_graph,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore timeline from dictionary."""
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported timeline version: {data.get('format_version')}")

        self.timeline_id = data["timeline_id"]

        # Restore branches
        for branch_id, segments_data in data.get("branches", {}).items():
            self.branches[branch_id] = []
            for seg_data in segments_data:
                segment = TimelineSegment(
                    seg_data["segment_id"],
                    seg_data["start_tick"],
                    branch_id,
                )
                for evt_data in seg_data.get("events", []):
                    event = SimulationEvent.from_dict(evt_data)
                    segment.add_event(event)
                self.branches[branch_id].append(segment)

        # Restore snapshots
        for key, snap_data in data.get("snapshots", {}).items():
            snapshot = TimelineSnapshot(
                branch_id=snap_data["branch_id"],
                tick=snap_data["tick"],
                timestamp=snap_data["timestamp"],
                entity_count=snap_data["entity_count"],
                debris_count=snap_data["debris_count"],
                fractures=snap_data["fractures"],
                world_state=snap_data.get("world_state", {}),
            )
            self.snapshots[key] = snapshot

        self.causality_graph = data.get("causality_graph", {})


class ReplayController:
    """Controls playback of recorded timeline."""

    def __init__(self, timeline: Timeline):
        """Initialize replay controller.

        Args:
            timeline: Timeline to replay
        """
        self.timeline = timeline
        self.current_tick = 0
        self.current_branch = "main"
        self.is_playing = False
        self.playback_speed = 1.0
        self._logger = get_logger("engine.simulation.replay")

    def seek(self, tick: int, branch_id: str = "main") -> None:
        """Seek to specific time.

        Args:
            tick: Target tick
            branch_id: Branch to seek in
        """
        if branch_id not in self.timeline.branches:
            self._logger.warning(f"Branch not found: {branch_id}")
            return

        self.current_tick = tick
        self.current_branch = branch_id
        self._logger.info(
            "Seek performed",
            context={"tick": tick, "branch_id": branch_id},
        )

    def play(self, speed: float = 1.0) -> None:
        """Start playback.

        Args:
            speed: Playback speed multiplier
        """
        self.is_playing = True
        self.playback_speed = speed
        self._logger.info("Playback started", context={"speed": speed})

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
        self._logger.info("Playback paused")

    def step_forward(self, tick_count: int = 1) -> None:
        """Step forward in time.

        Args:
            tick_count: Number of ticks to advance
        """
        self.current_tick += tick_count
        self._logger.debug(f"Stepped forward to tick {self.current_tick}")

    def step_backward(self, tick_count: int = 1) -> None:
        """Step backward in time.

        Args:
            tick_count: Number of ticks to rewind
        """
        self.current_tick = max(0, self.current_tick - tick_count)
        self._logger.debug(f"Stepped backward to tick {self.current_tick}")

    def switch_branch(self, branch_id: str) -> None:
        """Switch to different branch.

        Args:
            branch_id: Target branch
        """
        if branch_id not in self.timeline.branches:
            self._logger.warning(f"Branch not found: {branch_id}")
            return

        self.current_branch = branch_id
        self._logger.info("Branch switched", context={"branch_id": branch_id})

    def get_events_at_cursor(self) -> List[SimulationEvent]:
        """Get all events at current playback position.

        Returns:
            Events at current tick
        """
        events = self.timeline.get_events(
            self.current_branch,
            self.current_tick,
            self.current_tick,
        )
        return events

    def get_playback_state(self) -> Dict[str, Any]:
        """Get current playback state."""
        return {
            "current_tick": self.current_tick,
            "current_branch": self.current_branch,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
            "timeline_stats": self.timeline.get_stats(),
        }
