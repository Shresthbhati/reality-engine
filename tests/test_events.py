from events import Event, EventBus
from events.bus import WILDCARD


def test_emit_appends_and_returns_event():
    bus = EventBus(seed=1)
    event = bus.emit("ContactEvent", timestamp=1.5, tick=15, source_refs=("a",), target_refs=("b",))
    assert event.type == "ContactEvent"
    assert bus.events == [event]


def test_event_ids_are_deterministic_given_same_seed():
    bus_a = EventBus(seed=42)
    bus_b = EventBus(seed=42)
    ea = bus_a.emit("ContactEvent", timestamp=0.1, tick=1)
    eb = bus_b.emit("ContactEvent", timestamp=0.1, tick=1)
    assert ea.event_id == eb.event_id


def test_event_ids_unique_within_a_bus():
    bus = EventBus(seed=1)
    e1 = bus.emit("ContactEvent", timestamp=0.1, tick=1)
    e2 = bus.emit("ContactEvent", timestamp=0.1, tick=1)
    assert e1.event_id != e2.event_id


def test_subscribers_receive_matching_type_only():
    bus = EventBus()
    received = []
    bus.subscribe("ContactEvent", lambda e: received.append(e))
    bus.emit("ImpactEvent", timestamp=0.0, tick=0)
    assert received == []
    e = bus.emit("ContactEvent", timestamp=0.0, tick=0)
    assert received == [e]


def test_wildcard_subscriber_receives_everything():
    bus = EventBus()
    received = []
    bus.subscribe(WILDCARD, lambda e: received.append(e.type))
    bus.emit("ContactEvent", timestamp=0.0, tick=0)
    bus.emit("ImpactEvent", timestamp=0.0, tick=0)
    assert received == ["ContactEvent", "ImpactEvent"]


def test_events_of_type_filters():
    bus = EventBus()
    bus.emit("ContactEvent", timestamp=0.0, tick=0)
    bus.emit("ImpactEvent", timestamp=0.0, tick=0)
    bus.emit("ContactEvent", timestamp=0.1, tick=1)
    assert len(bus.events_of_type("ContactEvent")) == 2
    assert len(bus.events_of_type("ImpactEvent")) == 1


def test_clear_resets_log_and_sequence():
    bus = EventBus(seed=5)
    e1 = bus.emit("ContactEvent", timestamp=0.0, tick=0)
    bus.clear()
    e2 = bus.emit("ContactEvent", timestamp=0.0, tick=0)
    assert bus.events == [e2]
    assert e1.event_id == e2.event_id  # sequence counter reset, same deterministic id


def test_default_deterministic_seed_matches_bus_seed():
    bus = EventBus(seed=7)
    e = bus.emit("ContactEvent", timestamp=0.0, tick=0)
    assert e.deterministic_seed == 7


def test_roundtrip_to_list_and_back():
    bus = EventBus(seed=1)
    bus.emit(
        "ContactEvent", timestamp=1.0, tick=10,
        source_refs=("a",), target_refs=("b",), parameters={"penetration": 0.01},
        severity="warning", confidence=0.8,
    )
    restored = [Event.from_dict(d) for d in bus.to_list()]
    assert restored == bus.events
