"""Event record (V11 spec sec 930 EVENT SYSTEM field list: event_id,
type, timestamp, branch_id, actor_id, source_refs, target_refs,
parameters, cause_event_ids, severity, confidence, deterministic_seed,
resulting_state_refs).

`tick` is added beyond the spec's field list -- this engine's
determinism model (engine/core/clock.py) is built on discrete ticks, and
carrying it lets an event be replayed against an exact simulation step,
not just an approximate timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Event:
    event_id: str
    type: str
    timestamp: float
    tick: int
    source_refs: tuple[str, ...] = ()
    target_refs: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    cause_event_ids: tuple[str, ...] = ()
    severity: str = "info"
    confidence: float = 1.0
    branch_id: Optional[str] = None
    actor_id: Optional[str] = None
    deterministic_seed: Optional[int] = None
    resulting_state_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "tick": self.tick,
            "source_refs": list(self.source_refs),
            "target_refs": list(self.target_refs),
            "parameters": self.parameters,
            "cause_event_ids": list(self.cause_event_ids),
            "severity": self.severity,
            "confidence": self.confidence,
            "branch_id": self.branch_id,
            "actor_id": self.actor_id,
            "deterministic_seed": self.deterministic_seed,
            "resulting_state_refs": list(self.resulting_state_refs),
        }

    @staticmethod
    def from_dict(data: dict) -> "Event":
        return Event(
            event_id=data["event_id"],
            type=data["type"],
            timestamp=data["timestamp"],
            tick=data["tick"],
            source_refs=tuple(data.get("source_refs", ())),
            target_refs=tuple(data.get("target_refs", ())),
            parameters=data.get("parameters", {}),
            cause_event_ids=tuple(data.get("cause_event_ids", ())),
            severity=data.get("severity", "info"),
            confidence=data.get("confidence", 1.0),
            branch_id=data.get("branch_id"),
            actor_id=data.get("actor_id"),
            deterministic_seed=data.get("deterministic_seed"),
            resulting_state_refs=tuple(data.get("resulting_state_refs", ())),
        )
