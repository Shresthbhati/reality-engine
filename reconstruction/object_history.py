"""Object-level change history (N-23).

Tracks changes to individual walls, doors, windows, fixtures, furniture and
site objects across world versions. Records what changed, when, and what
evidence supports each change.

Honesty rules:
* Every change record carries before/after values, a timestamp, and the
  evidence/source that motivated it.
* A change with no supporting evidence is labelled "unknown", not invented.
* History is append-only; records are never rewritten.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity


@dataclass(frozen=True)
class ChangeRecord:
    """One change to one object between two world versions."""
    object_id: str
    object_type: str
    object_name: Optional[str]
    changed_at: str                # ISO-8601 UTC
    field: str                     # e.g. "bounds_max", "thickness_m", "semantic_labels"
    before: Optional[object]
    after: Optional[object]
    change_kind: str               # "updated" | "added" | "removed" | "provenance_changed"
    evidence_ids: List[str]        # source photos/sensor observations
    evidence_note: Optional[str]   # why this change is believed correct
    actor: Optional[str]           # "user" | "reconstruction" | "assistant" | "import"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectChangeHistory:
    """Complete change history for one object."""
    object_id: str
    object_type: str
    object_name: Optional[str]
    records: List[ChangeRecord]
    first_seen: Optional[str]
    last_seen: Optional[str]
    total_changes: int

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "object_name": self.object_name,
            "records": [r.to_dict() for r in self.records],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_changes": self.total_changes,
        }


class ChangeHistoryTracker:
    """Appends change records as worlds evolve, keyed by object id."""

    def __init__(self):
        self._records: Dict[str, List[ChangeRecord]] = {}
        self._first_seen: Dict[str, str] = {}

    def record_changes(
        self,
        object_id: str,
        object_type: str,
        object_name: Optional[str],
        before: Optional[Entity],
        after: Optional[Entity],
        *,
        change_kind: str = "updated",
        evidence_ids: Optional[Sequence[str]] = None,
        evidence_note: Optional[str] = None,
        actor: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[ChangeRecord]:
        """Record field-level changes between before/after snapshots."""
        now = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        records: List[ChangeRecord] = []

        if before is None and after is not None:
            records.append(ChangeRecord(
                object_id=object_id, object_type=object_type,
                object_name=after.name, changed_at=now, field="exists",
                before=None, after=True, change_kind="added",
                evidence_ids=list(evidence_ids or []),
                evidence_note=evidence_note or "Object first appears in this version.",
                actor=actor,
            ))
        elif before is not None and after is None:
            records.append(ChangeRecord(
                object_id=object_id, object_type=object_type,
                object_name=before.name, changed_at=now, field="exists",
                before=True, after=None, change_kind="removed",
                evidence_ids=list(evidence_ids or []),
                evidence_note=evidence_note or "Object no longer present.",
                actor=actor,
            ))
        elif before is not None and after is not None:
            before_cp = before.custom_properties or {}
            after_cp = after.custom_properties or {}
            keys = set(before_cp) | set(after_cp)
            for key in sorted(keys):
                b = before_cp.get(key)
                a = after_cp.get(key)
                if b != a:
                    records.append(ChangeRecord(
                        object_id=object_id, object_type=object_type,
                        object_name=after.name, changed_at=now, field=key,
                        before=b, after=a, change_kind="updated",
                        evidence_ids=list(evidence_ids or []),
                        evidence_note=evidence_note,
                        actor=actor,
                    ))
            if before.provenance != after.provenance:
                records.append(ChangeRecord(
                    object_id=object_id, object_type=object_type,
                    object_name=after.name, changed_at=now, field="provenance",
                    before=str(before.provenance), after=str(after.provenance),
                    change_kind="provenance_changed",
                    evidence_ids=list(evidence_ids or []),
                    evidence_note=evidence_note,
                    actor=actor,
                ))
            if before.confidence != after.confidence:
                records.append(ChangeRecord(
                    object_id=object_id, object_type=object_type,
                    object_name=after.name, changed_at=now, field="confidence",
                    before=before.confidence, after=after.confidence,
                    change_kind="updated",
                    evidence_ids=list(evidence_ids or []),
                    evidence_note=evidence_note,
                    actor=actor,
                ))

        if records:
            self._records.setdefault(object_id, []).extend(records)
            if object_id not in self._first_seen:
                self._first_seen[object_id] = now

        return records

    def history_for(self, object_id: str) -> Optional[ObjectChangeHistory]:
        records = self._records.get(object_id)
        if records is None and object_id not in self._first_seen:
            return None
        recs = list(records or [])
        return ObjectChangeHistory(
            object_id=object_id,
            object_type=recs[0].object_type if recs else "unknown",
            object_name=recs[0].object_name if recs else None,
            records=recs,
            first_seen=self._first_seen.get(object_id),
            last_seen=recs[-1].changed_at if recs else None,
            total_changes=len(recs),
        )

    def all_histories(self) -> List[ObjectChangeHistory]:
        ids = set(self._records) | set(self._first_seen)
        out: List[ObjectChangeHistory] = []
        for oid in sorted(ids):
            h = self.history_for(oid)
            if h is not None:
                out.append(h)
        return out


__all__ = ["ChangeHistoryTracker", "ChangeRecord", "ObjectChangeHistory"]