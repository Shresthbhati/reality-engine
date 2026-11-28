"""Tests for events.store.EventStore + events.replay (Programme 8)."""

from __future__ import annotations

from events.bus import EventBus
from events.event import Event
from events.replay import IdempotentConsumer, replay, replay_into_bus
from events.store import EventStore


def _bus_with_events(n=5):
    bus = EventBus(seed=7)
    for i in range(n):
        bus.emit("EntityCreatedEvent", timestamp=float(i), tick=i, source_refs=(f"e{i}",))
    return bus


def test_append_and_read_all_round_trip(tmp_path):
    store = EventStore(tmp_path / "log")
    bus = _bus_with_events(5)
    added = store.extend(bus.events)
    assert added == 5
    back = store.read_all()
    assert [e.to_dict() for e in back] == [e.to_dict() for e in bus.events]


def test_idempotent_append_no_duplicates(tmp_path):
    store = EventStore(tmp_path / "log")
    e = Event(event_id="x", type="T", timestamp=0.0, tick=0)
    assert store.append(e) is True
    assert store.append(e) is False
    assert len(store.read_all()) == 1
    assert len(store) == 1


def test_fresh_instance_over_same_root_reads_all(tmp_path):
    root = tmp_path / "log"
    s1 = EventStore(root)
    s1.extend(_bus_with_events(4).events)
    s2 = EventStore(root)  # simulate process restart
    assert len(s2.read_all()) == 4
    assert len(s2) == 4


def test_read_since_and_types(tmp_path):
    store = EventStore(tmp_path / "log")
    evs = _bus_with_events(5).events
    store.extend(evs)
    after = store.read_since(evs[1].event_id)
    assert [e.event_id for e in after] == [e.event_id for e in evs[2:]]
    assert store.read_since("unknown-id") == evs
    assert len(store.read_types("EntityCreatedEvent")) == 5
    assert store.read_types("Nope") == []


def test_malformed_trailing_line_skipped(tmp_path):
    store = EventStore(tmp_path / "log")
    store.append(Event(event_id="ok", type="T", timestamp=0.0, tick=0))
    with (tmp_path / "log" / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"partial": ')  # corrupt trailing line, no newline
    evs = EventStore(tmp_path / "log").read_all()
    assert len(evs) == 1 and evs[0].event_id == "ok"


def test_replay_reducer_rebuilds_state():
    bus = _bus_with_events(5)
    count = replay(bus.events, lambda s, e: (s or 0) + 1, initial=0)
    assert count == 5
    refs = replay(bus.events, lambda s, e: (s or []) + list(e.source_refs), initial=[])
    assert refs == [f"e{i}" for i in range(5)]


def test_replay_is_deterministic(tmp_path):
    store = EventStore(tmp_path / "log")
    store.extend(_bus_with_events(6).events)
    a = replay(store.read_all(), lambda s, e: (s or 0) + 1, 0)
    b = replay(EventStore(tmp_path / "log").read_all(), lambda s, e: (s or 0) + 1, 0)
    assert a == b == 6


def test_idempotent_consumer_applies_once_and_survives_restart(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    evs = _bus_with_events(4).events
    applied: list[str] = []

    c1 = IdempotentConsumer(ckpt)
    assert c1.consume(evs, applied.append) == 4
    # redeliver same events to same instance -> nothing applied
    assert c1.consume(evs, applied.append) == 0
    assert len(applied) == 4

    # fresh consumer over same checkpoint (restart) -> nothing re-applied
    applied2: list[str] = []
    c2 = IdempotentConsumer(ckpt)
    assert c2.consume(evs, applied2.append) == 0
    assert applied2 == []
    assert c2.last_processed_id == evs[-1].event_id
    assert c2.has_processed(evs[0].event_id)


def test_idempotent_consumer_does_not_mark_on_handler_error(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    evs = _bus_with_events(1).events
    c = IdempotentConsumer(ckpt)

    def boom(_e):
        raise RuntimeError("handler failed")

    try:
        c.consume(evs, boom)
    except RuntimeError:
        pass
    assert not c.has_processed(evs[0].event_id)  # left for retry
    assert c.consume(evs, lambda e: None) == 1   # retry succeeds


def test_replay_into_bus_preserves_ids_and_order(tmp_path):
    store = EventStore(tmp_path / "log")
    orig = _bus_with_events(5).events
    store.extend(orig)
    fresh = EventBus(seed=999)  # different seed; restore must not re-id
    seen: list[str] = []
    fresh.subscribe("*", lambda e: seen.append(e.event_id))
    n = replay_into_bus(store.read_all(), fresh)
    assert n == 5
    assert [e.event_id for e in fresh.events] == [e.event_id for e in orig]
    assert seen == [e.event_id for e in orig]  # dispatched in order
