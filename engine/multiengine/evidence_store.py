"""Append-only evidence store for raw engine observations.

There is deliberately NO update and NO delete. Consensus and refinement add new observations and *relations* (``refines``,
``conflicts_with``, ``supports``) that point at the originals; the originals stay exactly as the engine produced them.

Persistence is one JSON object per line. On load every observation's content hash is re-computed; a line that is corrupt or
whose hash does not match raises ``EvidenceStoreError`` naming the line -- evidence is never silently dropped or repaired.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Set, Tuple

from engine.multiengine.observation import EngineObservation, content_id, id_core


class EvidenceStoreError(RuntimeError):
    pass


RELATION_KINDS = frozenset({"refines", "conflicts_with", "supports", "same_entity", "derived_from"})


class EvidenceStore:
    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._obs: Dict[str, EngineObservation] = {}
        self._order: List[str] = []
        self._relations: Set[Tuple[str, str, str]] = set()
        self._path = Path(path) if path is not None else None
        if self._path is not None and self._path.exists():
            self._load()

    # ------------------------------------------------------------------ writes
    def add(self, obs: EngineObservation) -> bool:
        """Store an observation. Returns True if it was new, False if the identical observation was already stored.
        An observation whose id does not match its own content is refused (it would break traceability)."""
        expected = content_id(id_core(obs))
        if obs.observation_id != expected:
            raise EvidenceStoreError(f"observation {obs.observation_id!r} does not match its content hash {expected!r}")
        with self._lock:
            if obs.observation_id in self._obs:
                return False
            self._obs[obs.observation_id] = obs
            self._order.append(obs.observation_id)
            self._append({"type": "observation", **obs.to_dict()})
            for kind, other in obs.relations:
                self._link_unlocked(obs.observation_id, other, kind, persist=True)
            return True

    def add_many(self, observations: Iterable[EngineObservation]) -> int:
        return sum(1 for o in observations if self.add(o))

    def link(self, source_id: str, target_id: str, kind: str) -> bool:
        """Record that two stored observations are related. Neither observation is touched."""
        with self._lock:
            return self._link_unlocked(source_id, target_id, kind, persist=True)

    def _link_unlocked(self, a: str, b: str, kind: str, *, persist: bool) -> bool:
        if kind not in RELATION_KINDS:
            raise EvidenceStoreError(f"unknown relation kind {kind!r}; expected one of {sorted(RELATION_KINDS)}")
        if a not in self._obs:
            raise EvidenceStoreError(f"cannot link unknown observation {a!r}")
        if b not in self._obs:
            raise EvidenceStoreError(f"cannot link to unknown observation {b!r}")
        key = (a, b, kind)
        if key in self._relations:
            return False
        self._relations.add(key)
        if persist:
            self._append({"type": "relation", "kind": kind, "from": a, "to": b})
        return True

    # ------------------------------------------------------------------ reads
    def get(self, observation_id: str) -> EngineObservation:
        with self._lock:
            try:
                return self._obs[observation_id]
            except KeyError:
                raise EvidenceStoreError(f"unknown observation {observation_id!r}") from None

    def __len__(self) -> int:
        return len(self._obs)

    def query(self, *, engine_id: str | None = None, task: str | None = None, quantity: str | None = None,
              source_id: str | None = None, frame: int | None = None,
              coordinate_system: str | None = None) -> List[EngineObservation]:
        with self._lock:
            out = []
            for oid in self._order:
                o = self._obs[oid]
                if engine_id is not None and o.engine_id != engine_id:
                    continue
                if task is not None and o.task != task:
                    continue
                if quantity is not None and o.quantity != quantity:
                    continue
                if source_id is not None and o.source_id != source_id:
                    continue
                if frame is not None and o.frame != frame:
                    continue
                if coordinate_system is not None and o.coordinate_system != coordinate_system:
                    continue
                out.append(o)
            return out

    def relations_of(self, observation_id: str) -> List[Tuple[str, str, str]]:
        """(from, to, kind) for every relation touching the observation."""
        with self._lock:
            return sorted(r for r in self._relations if observation_id in (r[0], r[1]))

    def lineage(self, observation_id: str) -> List[str]:
        """Ids this observation ``refines`` / is ``derived_from``, transitively (cycle-safe), nearest first."""
        with self._lock:
            seen: List[str] = []
            frontier = [observation_id]
            while frontier:
                cur = frontier.pop(0)
                for a, b, kind in sorted(self._relations):
                    if a == cur and kind in ("refines", "derived_from") and b not in seen and b != observation_id:
                        seen.append(b)
                        frontier.append(b)
            return seen

    def engines_for(self, quantity: str, source_id: str, frame: int | None = None) -> Dict[str, List[str]]:
        """engine_id -> observation ids that report ``quantity`` for the source: who said what about the same thing."""
        out: Dict[str, List[str]] = {}
        for o in self.query(quantity=quantity, source_id=source_id, frame=frame):
            out.setdefault(o.engine_id, []).append(o.observation_id)
        return out

    # ------------------------------------------------------------------ persistence
    def _append(self, record: dict) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self._path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def _load(self) -> None:
        assert self._path is not None
        with open(self._path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        relations: List[Tuple[int, dict]] = []
        for n, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceStoreError(f"{self._path}:{n}: corrupt evidence line ({exc})") from exc
            if rec.get("type") == "observation":
                try:
                    obs = EngineObservation.from_dict(rec)
                except (KeyError, TypeError) as exc:
                    raise EvidenceStoreError(f"{self._path}:{n}: malformed observation ({exc})") from exc
                if obs.observation_id != content_id(id_core(obs)):
                    raise EvidenceStoreError(f"{self._path}:{n}: observation {obs.observation_id} fails its content-hash check")
                if obs.observation_id not in self._obs:
                    self._obs[obs.observation_id] = obs
                    self._order.append(obs.observation_id)
            elif rec.get("type") == "relation":
                relations.append((n, rec))
            else:
                raise EvidenceStoreError(f"{self._path}:{n}: unknown record type {rec.get('type')!r}")
        for n, rec in relations:
            try:
                self._link_unlocked(rec["from"], rec["to"], rec["kind"], persist=False)
            except (KeyError, EvidenceStoreError) as exc:
                raise EvidenceStoreError(f"{self._path}:{n}: bad relation ({exc})") from exc
