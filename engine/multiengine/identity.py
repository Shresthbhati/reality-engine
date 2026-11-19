"""Stable entity identity across incremental evidence.

``association.entity_key`` is a hash of the member set, so it changes every time a new photograph adds a member -- fine as a
per-run label, wrong as an identity. ``IdentityRegistry`` remembers which observations formed which entity and keeps an entity's id
when later evidence extends it:

* unchanged / extended: the entity keeps its id (members were added);
* merged: when new evidence links two earlier entities, the one with MORE members (ties: smaller id) keeps its id; the other id is
  RETIRED with ``merged_into`` -- never reused, never silently forgotten;
* split: when one earlier entity falls apart, the part with the largest overlap (ties: smaller first member) keeps the id; the rest get
  new ids and the event says which entity they came from;
* created: a new id, derived from the member set at creation (deterministic) -- never random, never a counter.

Deterministic (the result does not depend on input order), append-only in events, and persisted atomically. This module does NOT decide
who is the same object (that is association's job, which also enforces that one image never contributes two members to one entity); it
only keeps names stable once that decision has been made.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from engine.multiengine.association import AssociatedEntity, entity_key


@dataclass(frozen=True)
class IdentityEvent:
    kind: str  # created | unchanged | extended | merged | split | changed
    stable_id: str
    detail: str
    members: tuple[str, ...]
    related: tuple[str, ...] = ()  # ids merged away / the ids a split part came from

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "stable_id": self.stable_id, "detail": self.detail, "members": list(self.members),
                "related": list(self.related)}


class IdentityRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._entities: dict[str, frozenset[str]] = {}
        self._retired: dict[str, str] = {}
        self._events: list[dict[str, Any]] = []
        if self._path is not None and self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entities = {k: frozenset(v) for k, v in data["entities"].items()}
            self._retired = dict(data["retired"])
            self._events = list(data["events"])

    def clone(self) -> IdentityRegistry:
        """An in-memory copy that never writes to disk: used to evaluate TRIAL states without recording identity events for them."""
        c = IdentityRegistry(None)
        c._entities, c._retired, c._events = dict(self._entities), dict(self._retired), list(self._events)
        return c

    # ---- reads
    def members(self, stable_id: str) -> frozenset[str]:
        return self._entities[stable_id]

    @property
    def retired(self) -> dict[str, str]:
        return dict(self._retired)

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def ids(self) -> list[str]:
        return sorted(self._entities)

    # ---- resolve
    def resolve(self, entities: Sequence[AssociatedEntity]) -> tuple[list[AssociatedEntity], list[IdentityEvent]]:
        """Give every entity its stable id; returns the re-keyed entities and the events this call produced."""
        obs_owner = {m: sid for sid, ms in self._entities.items() for m in ms}
        ordered = sorted(entities, key=lambda e: min(e.members))
        overlap: list[Counter] = [Counter(obs_owner[m] for m in e.members if m in obs_owner) for e in ordered]

        # each earlier id is claimed by the new entity overlapping it most (ties: the one with the smaller first member)
        claims: dict[str, int] = {}
        for sid in sorted(self._entities):
            hits = [(overlap[i][sid], i) for i in range(len(ordered)) if overlap[i].get(sid)]
            if hits:
                top = max(h for h, _ in hits)
                claims[sid] = min((i for h, i in hits if h == top), key=lambda i: min(ordered[i].members))
        won: dict[int, list[str]] = {}
        for sid, i in claims.items():
            won.setdefault(i, []).append(sid)

        new_entities: dict[str, frozenset[str]] = {}
        events: list[IdentityEvent] = []
        out: list[AssociatedEntity] = []
        used: set[str] = set()
        for i, e in enumerate(ordered):
            members = frozenset(e.members)
            mine = sorted(won.get(i, []), key=lambda s: (-len(self._entities[s]), s))
            if mine:
                sid, absorbed = mine[0], mine[1:]
                prior = self._entities[sid]
                lost = tuple(sorted(s for s in overlap[i] if s not in mine))  # earlier ids this entity overlaps but did not win
                if members == prior and not absorbed:
                    kind, detail = "unchanged", "same observations as before"
                elif absorbed:
                    kind, detail = "merged", f"new evidence linked {', '.join(absorbed)} into {sid}"
                    for s in absorbed:
                        self._retired[s] = sid
                elif members >= prior:
                    kind, detail = "extended", f"{len(members - prior)} observation(s) added"
                else:
                    kind, detail = "changed", f"{len(prior - members)} observation(s) left, {len(members - prior)} joined"
                events.append(IdentityEvent(kind, sid, detail, tuple(sorted(members)), tuple(absorbed) or lost))
            else:
                sid = entity_key(members)
                while sid in self._entities or sid in self._retired or sid in used:
                    sid += "_2"
                came = tuple(sorted(overlap[i]))  # earlier ids this entity overlaps but won none of: it was split off from them
                events.append(IdentityEvent("split" if came else "created", sid,
                                            f"split from {', '.join(came)}" if came else "first seen",
                                            tuple(sorted(members)), came))
            used.add(sid)
            new_entities[sid] = members
            out.append(replace(e, entity_key=sid))
        # earlier entities that no new entity claimed are kept as they were (this call saw none of their evidence)
        for sid, ms in self._entities.items():
            if sid not in new_entities and sid not in self._retired:
                new_entities[sid] = ms
        self._entities = new_entities
        self._events.extend(ev.to_dict() for ev in events)
        self._save()
        return out, events

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        data = {"entities": {k: sorted(v) for k, v in sorted(self._entities.items())},
                "retired": dict(sorted(self._retired.items())), "events": self._events}
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)
