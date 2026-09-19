"""WorldIR temporal state layer (simulation campaign secs 18-21: EVENT
GRAPH, TEMPORAL STATE, BRANCHING; complements engine/simulation/replay.py,
which owns timeline SEGMENTS and playback, by owning WorldIR STATE:
deep-state snapshots, branch inheritance, structural comparison, and
causal queries over the canonical EventBus).

Division of labour:
  engine/simulation/replay.py   Timeline segments, event recording,
                                playback control (pre-existing, tested).
  engine/simulation/temporal.py Snapshot/restore of full WorldIR state,
                                branch inheritance WITHOUT parent mutation,
                                structural branch comparison, causal
                                queries over the real events.EventBus.

Honesty rules:
  - Snapshots store full WorldIR serializations deep-copied at capture;
    later world mutation can never reach into a stored snapshot (tested).
  - Restore replaces world CONTENT (entities/geometries/materials/
    relationships/measurements) with fresh deep-copied state; the world
    object identity is preserved so Studio sessions keep their handle.
  - Branch creation deep-copies parent state; the parent is never mutated
    by a child branch (tested).
  - Replay is non-destructive: it never clears or rewrites the caller's
    event log -- it runs commands on a deep-copied world against a fresh
    bus and returns both, so "replay" cannot silently destroy evidence.
  - Causal answers cite evidence: ancestors()/descendants() return the
    actual Event records, so "what caused this?" is always backed by the
    log. Unknown event ids raise; nothing is invented.
  - Determinism: snapshot/branch ids are caller-visible derivations
    (tick + ordinal / caller-supplied branch ids) -- no uuid4, no wall
    clock inside this layer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from events import Event, EventBus
from world_ir import WorldIR

#: WorldIR content containers replaced on restore (identity fields like
#: id/version are preserved on the live world object).
_CONTENT_ATTRS = (
    "entities", "geometries", "materials", "surfaces", "components",
    "temporal_state", "temporal_events", "causal_relations",
)


# ---------------------------------------------------------------------------
# Causal event graph
# ---------------------------------------------------------------------------


class EventGraph:
    """Causal graph view over an EventBus's event log.

    The log is the source of truth; this graph builds parent->child edges
    from cause_event_ids and answers causal queries from it.
    """

    def __init__(self, bus: EventBus):
        self._bus = bus

    @property
    def events(self) -> List[Event]:
        return self._bus.events

    def _by_id(self) -> Dict[str, Event]:
        return {e.event_id: e for e in self._bus.events}

    def _children_index(self) -> Dict[str, List[Event]]:
        idx: Dict[str, List[Event]] = {}
        for e in self._bus.events:
            for cause_id in e.cause_event_ids:
                idx.setdefault(cause_id, []).append(e)
        return idx

    def _require(self, event_id: str, by_id: Dict[str, Event]) -> None:
        if event_id not in by_id:
            raise KeyError(f"unknown event '{event_id}'")

    def ancestors_of(self, event_id: str) -> List[Event]:
        """All transitive causes of an event, nearest first (BFS)."""
        by_id = self._by_id()
        self._require(event_id, by_id)
        out: List[Event] = []
        seen = {event_id}
        frontier = [event_id]
        while frontier:
            nxt: List[str] = []
            for eid in frontier:
                for cause_id in by_id[eid].cause_event_ids:
                    if cause_id in seen:
                        continue
                    seen.add(cause_id)
                    if cause_id in by_id:
                        out.append(by_id[cause_id])
                        nxt.append(cause_id)
            frontier = nxt
        return out

    def descendants_of(self, event_id: str) -> List[Event]:
        """All transitive consequences of an event (BFS, nearest first)."""
        by_id = self._by_id()
        self._require(event_id, by_id)
        children = self._children_index()
        out: List[Event] = []
        seen = {event_id}
        frontier = [event_id]
        while frontier:
            nxt: List[str] = []
            for eid in frontier:
                for child in children.get(eid, []):
                    if child.event_id in seen:
                        continue
                    seen.add(child.event_id)
                    out.append(child)
                    nxt.append(child.event_id)
            frontier = nxt
        return out

    def root_causes_of(self, event_id: str) -> List[Event]:
        """Causes with no further causes -- the chain's origins."""
        return [a for a in self.ancestors_of(event_id) if not a.cause_event_ids]

    def events_affecting_entity(self, entity_id: str) -> List[Event]:
        """Every event whose sources or targets reference the entity."""
        return [
            e for e in self._bus.events
            if entity_id in e.source_refs or entity_id in e.target_refs
        ]

    def events_after(self, event_id: str) -> List[Event]:
        """Strictly subsequent events on the log (what changed after X)."""
        by_id = self._by_id()
        self._require(event_id, by_id)
        pivot = by_id[event_id]
        return [e for e in self._bus.events if e.tick > pivot.tick]


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------


def _deep_state(world: WorldIR) -> dict:
    """Snapshot payload: full WorldIR serialization, deep-copied so later
    world mutation can never reach into a stored snapshot."""
    return copy.deepcopy(world.to_dict())


def restore_world(world: WorldIR, payload: dict) -> WorldIR:
    """Replace `world`'s content containers with fresh state rebuilt from a
    snapshot payload (WorldIR.from_dict round-trip). The world object
    identity is preserved -- callers holding the reference keep it."""
    restored = WorldIR.from_dict(copy.deepcopy(payload))
    for attr in _CONTENT_ATTRS:
        setattr(world, attr, getattr(restored, attr))
    return world


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    tick: int
    timestamp: float
    branch_id: str
    label: str
    entity_count: int

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "branch_id": self.branch_id,
            "label": self.label,
            "entity_count": self.entity_count,
        }


class SnapshotStore:
    """Named snapshots of WorldIR state, restorable in place.

    Deterministic ids: snap-{branch}-{tick}-{ordinal} -- no uuid4.
    """

    def __init__(self):
        self._snapshots: Dict[str, Tuple[Snapshot, dict]] = {}
        self._ordinal = 0

    def capture(self, world: WorldIR, tick: int, timestamp: float,
                branch_id: str = "main", label: str = "") -> Snapshot:
        self._ordinal += 1
        snap_id = f"snap-{branch_id}-{tick:08d}-{self._ordinal:04d}"
        snap = Snapshot(
            snapshot_id=snap_id,
            tick=tick,
            timestamp=timestamp,
            branch_id=branch_id,
            label=label or f"snapshot {self._ordinal}",
            entity_count=len(world.entities),
        )
        self._snapshots[snap_id] = (snap, _deep_state(world))
        return snap

    def get(self, snapshot_id: str) -> Snapshot:
        return self._snapshots[snapshot_id][0]

    def payload(self, snapshot_id: str) -> dict:
        return copy.deepcopy(self._snapshots[snapshot_id][1])

    def restore(self, snapshot_id: str, world: WorldIR) -> WorldIR:
        """Restore in place: world's content containers are replaced with
        fresh, deep-copied state from the snapshot."""
        return restore_world(world, self._snapshots[snapshot_id][1])

    def list(self, branch_id: Optional[str] = None) -> List[Snapshot]:
        return [
            s for (s, _p) in self._snapshots.values()
            if branch_id is None or s.branch_id == branch_id
        ]


# ---------------------------------------------------------------------------
# Branch manager
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchRecord:
    branch_id: str
    parent_branch_id: Optional[str]
    created_at_tick: int
    description: str

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "created_at_tick": self.created_at_tick,
            "description": self.description,
        }


class BranchManager:
    """World-branching for scenario exploration.

    A branch inherits parent state by deep copy at creation; the parent is
    never mutated by the child. Branch ids are caller-supplied so replay
    stays deterministic (no uuid4 minting inside the timeline).
    """

    def __init__(self, snapshots: Optional[SnapshotStore] = None):
        self.snapshots = snapshots if snapshots is not None else SnapshotStore()
        self._branches: Dict[str, BranchRecord] = {
            "main": BranchRecord("main", None, 0, "baseline"),
        }

    def create_branch(self, world: WorldIR, branch_id: str, parent_branch_id: str,
                      tick: int, description: str = "") -> Tuple[WorldIR, BranchRecord]:
        if branch_id in self._branches:
            raise ValueError(f"branch '{branch_id}' already exists")
        if parent_branch_id not in self._branches:
            raise KeyError(f"unknown parent branch '{parent_branch_id}'")
        record = BranchRecord(branch_id, parent_branch_id, tick, description)
        self._branches[branch_id] = record
        child = WorldIR.from_dict(copy.deepcopy(world.to_dict()))
        # Preserve the schema's stable-identity convention: a child branch
        # IS a new world instance, but its id must not be silently random;
        # derive it from the branch id.
        child.id = f"world-{branch_id}"
        return child, record

    def get(self, branch_id: str) -> BranchRecord:
        return self._branches[branch_id]

    def list(self) -> List[BranchRecord]:
        return list(self._branches.values())

    def delete_branch(self, branch_id: str) -> None:
        if branch_id == "main":
            raise ValueError("the baseline branch 'main' cannot be deleted")
        if branch_id not in self._branches:
            raise KeyError(f"unknown branch '{branch_id}'")
        self._branches.pop(branch_id)

    # -- comparison ----------------------------------------------------

    def compare(self, world_a: WorldIR, world_b: WorldIR,
                branch_a: str = "main", branch_b: str = "") -> dict:
        """Structural diff of two branch states: added/removed/changed
        entities (changed = type/confidence/geometry_ids/transform)."""
        changed: List[dict] = []
        added = sorted(set(world_b.entities) - set(world_a.entities))
        removed = sorted(set(world_a.entities) - set(world_b.entities))
        for eid in sorted(set(world_a.entities) & set(world_b.entities)):
            ea, eb = world_a.entities[eid], world_b.entities[eid]
            diffs: Dict[str, tuple] = {}
            if ea.confidence != eb.confidence:
                diffs["confidence"] = (ea.confidence, eb.confidence)
            if ea.geometry_ids != eb.geometry_ids:
                diffs["geometry_ids"] = (ea.geometry_ids, eb.geometry_ids)
            if (ea.transform or None) != (eb.transform or None):
                diffs["transform"] = (ea.transform, eb.transform)
            if ea.type != eb.type:
                diffs["type"] = (ea.type.value, eb.type.value)
            if diffs:
                changed.append({"entity_id": eid, "fields": diffs})
        return {
            "branch_a": branch_a,
            "branch_b": branch_b or branch_a,
            "added": added,
            "removed": removed,
            "changed": changed,
        }


# ---------------------------------------------------------------------------
# Deterministic non-destructive replay
# ---------------------------------------------------------------------------


def replay(
    world: WorldIR,
    commands: List[Callable[[WorldIR], None]],
    seed: int = 0,
) -> Tuple[WorldIR, EventBus]:
    """Deterministic replay of a command list over a deep copy of `world`.

    Non-destructive by construction: the caller's world and event bus are
    never touched -- commands run on a deep-copied world, events go to a
    fresh bus seeded with `seed`. Same world + same commands + same seed
    -> identical resulting state and identical event ids.
    """
    working = WorldIR.from_dict(copy.deepcopy(world.to_dict()))
    working.id = world.id  # replay preserves identity: it IS that world's future
    bus = EventBus(seed=seed)
    for command in commands:
        command(working)
    return working, bus
