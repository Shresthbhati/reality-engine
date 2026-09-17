"""Tests for replay system."""

import pytest

from engine.simulation import (
    EventType,
    SimulationEvent,
    TimelineSnapshot,
    Timeline,
    ReplayController,
)


class TestSimulationEvent:
    """Simulation event tests."""

    def test_create_event(self):
        """Test creating an event."""
        event = SimulationEvent(
            event_id="evt_1",
            event_type=EventType.ENTITY_CREATE,
            tick=10,
            timestamp=0.1,
            entity_id="entity_1",
        )

        assert event.event_id == "evt_1"
        assert event.event_type == EventType.ENTITY_CREATE
        assert event.tick == 10
        assert event.timestamp == 0.1
        assert event.entity_id == "entity_1"

    def test_event_with_data(self):
        """Test event with custom data."""
        data = {"position": [1, 2, 3], "velocity": [0.5, 0, 0]}
        event = SimulationEvent(
            event_id="evt_1",
            event_type=EventType.PHYSICS_IMPACT,
            tick=5,
            timestamp=0.05,
            data=data,
        )

        assert event.data == data

    def test_event_with_parent(self):
        """Test event with causal parent."""
        event = SimulationEvent(
            event_id="evt_2",
            event_type=EventType.FRACTURE,
            tick=11,
            timestamp=0.11,
            parent_event_id="evt_1",
        )

        assert event.parent_event_id == "evt_1"

    def test_event_serialization(self):
        """Test event to/from dict."""
        event = SimulationEvent(
            event_id="evt_1",
            event_type=EventType.ENTITY_CREATE,
            tick=10,
            timestamp=0.1,
            entity_id="entity_1",
            data={"key": "value"},
        )

        data = event.to_dict()
        assert data["event_id"] == "evt_1"
        assert data["event_type"] == "entity_create"
        assert data["tick"] == 10

        restored = SimulationEvent.from_dict(data)
        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.tick == event.tick


class TestTimelineSegment:
    """Timeline segment tests."""

    def test_create_segment(self):
        """Test creating a segment."""
        from engine.simulation.replay import TimelineSegment

        segment = TimelineSegment("seg_1", 0, "main")
        assert segment.segment_id == "seg_1"
        assert segment.start_tick == 0
        assert segment.branch_id == "main"
        assert len(segment.events) == 0

    def test_add_events_to_segment(self):
        """Test adding events to segment."""
        from engine.simulation.replay import TimelineSegment

        segment = TimelineSegment("seg_1", 0, "main")

        event1 = SimulationEvent("evt_1", EventType.SIMULATION_START, 0, 0.0)
        event2 = SimulationEvent("evt_2", EventType.ENTITY_CREATE, 5, 0.05)

        segment.add_event(event1)
        segment.add_event(event2)

        assert len(segment.events) == 2
        assert segment.end_tick == 5

    def test_segment_range_query(self):
        """Test querying events by tick range."""
        from engine.simulation.replay import TimelineSegment

        segment = TimelineSegment("seg_1", 0, "main")

        for i in range(0, 10):
            event = SimulationEvent(f"evt_{i}", EventType.SIMULATION_STEP, i, 0.01 * i)
            segment.add_event(event)

        # Query ticks 3-6
        events = segment.get_events_in_range(3, 6)
        assert len(events) == 4
        assert events[0].tick == 3
        assert events[-1].tick == 6


class TestTimeline:
    """Timeline management tests."""

    def test_create_timeline(self):
        """Test creating a timeline."""
        timeline = Timeline("timeline_1")
        assert timeline.timeline_id == "timeline_1"
        assert "main" in timeline.branches

    def test_record_event(self):
        """Test recording events."""
        timeline = Timeline("timeline_1")

        event_id = timeline.record_event(
            EventType.ENTITY_CREATE,
            tick=0,
            timestamp=0.0,
            entity_id="entity_1",
        )

        assert event_id.startswith("main_evt_")

        events = timeline.get_events()
        assert len(events) == 1
        assert events[0].entity_id == "entity_1"

    def test_record_multiple_events(self):
        """Test recording multiple events."""
        timeline = Timeline("timeline_1")

        for i in range(10):
            timeline.record_event(
                EventType.SIMULATION_STEP,
                tick=i,
                timestamp=0.01 * i,
            )

        events = timeline.get_events()
        assert len(events) == 10

    def test_get_events_by_type(self):
        """Test querying events by type."""
        timeline = Timeline("timeline_1")

        timeline.record_event(EventType.ENTITY_CREATE, 0, 0.0)
        timeline.record_event(EventType.ENTITY_CREATE, 1, 0.01)
        timeline.record_event(EventType.PHYSICS_IMPACT, 2, 0.02)
        timeline.record_event(EventType.FRACTURE, 3, 0.03)

        creates = timeline.get_events_by_type(EventType.ENTITY_CREATE)
        assert len(creates) == 2

        fractures = timeline.get_events_by_type(EventType.FRACTURE)
        assert len(fractures) == 1

    def test_get_events_for_entity(self):
        """Test querying events by entity."""
        timeline = Timeline("timeline_1")

        timeline.record_event(EventType.ENTITY_CREATE, 0, 0.0, entity_id="entity_1")
        timeline.record_event(EventType.PHYSICS_IMPACT, 1, 0.01, entity_id="entity_1")
        timeline.record_event(EventType.FRACTURE, 2, 0.02, entity_id="entity_1")
        timeline.record_event(EventType.ENTITY_CREATE, 3, 0.03, entity_id="entity_2")

        entity1_events = timeline.get_events_for_entity("entity_1")
        assert len(entity1_events) == 3

        entity2_events = timeline.get_events_for_entity("entity_2")
        assert len(entity2_events) == 1

    def test_get_events_by_tick_range(self):
        """Test querying events by time range."""
        timeline = Timeline("timeline_1")

        for i in range(20):
            timeline.record_event(EventType.SIMULATION_STEP, i, 0.01 * i)

        # Query ticks 5-14
        events = timeline.get_events(start_tick=5, end_tick=14)
        assert len(events) == 10
        assert events[0].tick == 5
        assert events[-1].tick == 14

    def test_create_branch(self):
        """Test creating a branch."""
        timeline = Timeline("timeline_1")

        timeline.record_event(EventType.ENTITY_CREATE, 0, 0.0, branch_id="main")
        timeline.record_event(EventType.ENTITY_CREATE, 1, 0.01, branch_id="main")

        timeline.create_branch("alt_1", 1)

        timeline.record_event(EventType.ENTITY_CREATE, 2, 0.02, branch_id="alt_1")

        main_events = timeline.get_events("main")
        alt_events = timeline.get_events("alt_1")

        assert len(main_events) == 2
        assert len(alt_events) == 1

    def test_causality_tracking(self):
        """Test causality graph."""
        timeline = Timeline("timeline_1")

        parent_id = timeline.record_event(EventType.PHYSICS_IMPACT, 0, 0.0)
        child_id = timeline.record_event(
            EventType.FRACTURE, 1, 0.01, parent_event_id=parent_id
        )
        grandchild_id = timeline.record_event(
            EventType.DEBRIS_SPAWN, 2, 0.02, parent_event_id=child_id
        )

        chain = timeline.get_causal_chain(parent_id)
        assert parent_id in chain
        assert child_id in chain
        assert grandchild_id in chain
        assert len(chain) == 3

    def test_snapshots(self):
        """Test storing world state snapshots."""
        timeline = Timeline("timeline_1")

        snapshot = TimelineSnapshot(
            branch_id="main",
            tick=10,
            timestamp=0.1,
            entity_count=5,
            debris_count=12,
            fractures=2,
        )

        timeline.add_snapshot(snapshot)

        retrieved = timeline.get_snapshot("main", 10)
        assert retrieved is not None
        assert retrieved.entity_count == 5
        assert retrieved.debris_count == 12

    def test_timeline_stats(self):
        """Test timeline statistics."""
        timeline = Timeline("timeline_1")

        for i in range(20):
            timeline.record_event(EventType.SIMULATION_STEP, i, 0.01 * i)

        stats = timeline.get_stats()
        assert stats["branches"] == 1  # Only "main"
        assert stats["total_events"] == 20

    def test_timeline_serialization(self):
        """Test timeline to/from dict."""
        timeline1 = Timeline("timeline_1")

        timeline1.record_event(EventType.ENTITY_CREATE, 0, 0.0, entity_id="e1")
        timeline1.record_event(EventType.PHYSICS_IMPACT, 1, 0.01, entity_id="e1")
        timeline1.record_event(EventType.FRACTURE, 2, 0.02, entity_id="e1")

        data = timeline1.serialize()
        assert "format_version" in data
        assert "timeline_id" in data
        assert "branches" in data

        # Restore
        timeline2 = Timeline("timeline_2")
        timeline2.deserialize(data)

        events2 = timeline2.get_events()
        assert len(events2) == 3
        assert events2[0].event_type == EventType.ENTITY_CREATE
        assert events2[2].event_type == EventType.FRACTURE


class TestReplayController:
    """Replay playback control tests."""

    def test_create_controller(self):
        """Test creating a replay controller."""
        timeline = Timeline("timeline_1")
        controller = ReplayController(timeline)

        assert controller.current_tick == 0
        assert controller.current_branch == "main"
        assert not controller.is_playing

    def test_seek(self):
        """Test seeking to tick."""
        timeline = Timeline("timeline_1")

        for i in range(100):
            timeline.record_event(EventType.SIMULATION_STEP, i, 0.01 * i)

        controller = ReplayController(timeline)
        controller.seek(50)

        assert controller.current_tick == 50

    def test_playback_control(self):
        """Test play/pause."""
        timeline = Timeline("timeline_1")
        controller = ReplayController(timeline)

        assert not controller.is_playing

        controller.play(speed=1.0)
        assert controller.is_playing
        assert controller.playback_speed == 1.0

        controller.pause()
        assert not controller.is_playing

    def test_step_forward(self):
        """Test stepping forward."""
        timeline = Timeline("timeline_1")
        controller = ReplayController(timeline)

        controller.step_forward(10)
        assert controller.current_tick == 10

        controller.step_forward(5)
        assert controller.current_tick == 15

    def test_step_backward(self):
        """Test stepping backward."""
        timeline = Timeline("timeline_1")
        controller = ReplayController(timeline)

        controller.seek(50)
        controller.step_backward(10)
        assert controller.current_tick == 40

        controller.step_backward(100)  # Beyond zero
        assert controller.current_tick == 0  # Clamped

    def test_switch_branch(self):
        """Test switching between branches."""
        timeline = Timeline("timeline_1")

        timeline.record_event(EventType.ENTITY_CREATE, 0, 0.0, branch_id="main")
        timeline.create_branch("alt_1", 0)
        timeline.record_event(EventType.ENTITY_CREATE, 1, 0.01, branch_id="alt_1")

        controller = ReplayController(timeline)
        assert controller.current_branch == "main"

        controller.switch_branch("alt_1")
        assert controller.current_branch == "alt_1"

    def test_get_events_at_cursor(self):
        """Test getting events at current playback position."""
        timeline = Timeline("timeline_1")

        timeline.record_event(EventType.ENTITY_CREATE, 10, 0.1)
        timeline.record_event(EventType.PHYSICS_IMPACT, 10, 0.1)
        timeline.record_event(EventType.FRACTURE, 11, 0.11)

        controller = ReplayController(timeline)
        controller.seek(10)

        events = controller.get_events_at_cursor()
        assert len(events) == 2
        assert all(e.tick == 10 for e in events)

    def test_get_playback_state(self):
        """Test querying playback state."""
        timeline = Timeline("timeline_1")

        for i in range(50):
            timeline.record_event(EventType.SIMULATION_STEP, i, 0.01 * i)

        controller = ReplayController(timeline)
        controller.seek(25)
        controller.play(speed=2.0)

        state = controller.get_playback_state()
        assert state["current_tick"] == 25
        assert state["current_branch"] == "main"
        assert state["is_playing"]
        assert state["playback_speed"] == 2.0
        assert state["timeline_stats"]["total_events"] == 50

    def test_replay_with_branching(self):
        """Test replay across multiple branches."""
        timeline = Timeline("timeline_1")

        # Main branch
        for i in range(10):
            timeline.record_event(EventType.SIMULATION_STEP, i, 0.01 * i, branch_id="main")

        # Create branch at tick 5
        timeline.create_branch("alt", 5)

        # Continue in alt branch
        for i in range(10, 15):
            timeline.record_event(
                EventType.SIMULATION_STEP, i, 0.01 * i, branch_id="alt"
            )

        controller = ReplayController(timeline)

        # Play main
        main_events = timeline.get_events("main")
        assert len(main_events) == 10

        # Switch to alt
        controller.switch_branch("alt")
        alt_events = timeline.get_events("alt")
        assert len(alt_events) == 5

    def test_complex_causality(self):
        """Test complex causal chains."""
        timeline = Timeline("timeline_1")

        # Create impact event
        impact_id = timeline.record_event(EventType.PHYSICS_IMPACT, 0, 0.0)

        # Fracture caused by impact
        fracture_ids = []
        for i in range(3):
            fid = timeline.record_event(
                EventType.FRACTURE, 1, 0.01, parent_event_id=impact_id
            )
            fracture_ids.append(fid)

        # Debris spawned from each fracture
        debris_ids = []
        for fid in fracture_ids:
            did = timeline.record_event(
                EventType.DEBRIS_SPAWN, 2, 0.02, parent_event_id=fid
            )
            debris_ids.append(did)

        # Get causal chain from impact
        chain = timeline.get_causal_chain(impact_id)

        # Should include impact, fractures, and debris
        assert len(chain) >= 7  # 1 impact + 3 fractures + 3 debris

    def test_time_travel_query(self):
        """Test querying what could happen at different times."""
        timeline = Timeline("timeline_1")

        # Record a timeline
        for i in range(0, 20, 2):
            timeline.record_event(EventType.ENTITY_CREATE, i, 0.01 * i)

        for i in range(1, 20, 2):
            timeline.record_event(EventType.PHYSICS_IMPACT, i, 0.01 * i)

        creates = timeline.get_events_by_type(EventType.ENTITY_CREATE)
        impacts = timeline.get_events_by_type(EventType.PHYSICS_IMPACT)

        assert len(creates) == 10
        assert len(impacts) == 10
