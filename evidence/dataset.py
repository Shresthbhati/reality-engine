"""Dataset (spec: §3 Dataset Builder) and session merging (spec: §5 Merge Sessions).

A Dataset is a named collection of Sessions plus derived artifacts (merges)
-- not a folder of files. It never owns evidence directly; it references
Sessions by id, so Session identity and immutability are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.core.logging import get_logger
from .session import Session

_logger = get_logger("evidence.dataset")


class UnknownSessionError(ValueError):
    pass


class DuplicateSessionError(ValueError):
    pass


class Dataset:
    """A named collection of Sessions and their derived merges."""

    def __init__(self, dataset_id: str, name: str = ""):
        self.id = dataset_id
        self.name = name
        self._sessions: Dict[str, Session] = {}
        self._session_order: List[str] = []
        self.merges: Dict[str, "MergedContext"] = {}

    def add_session(self, session: Session) -> None:
        if session.id in self._sessions:
            raise DuplicateSessionError(f"session '{session.id}' already in dataset '{self.id}'")
        self._sessions[session.id] = session
        self._session_order.append(session.id)
        _logger.info("Session added to dataset", context={"dataset_id": self.id, "session_id": session.id})

    def get_session(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise UnknownSessionError(f"no session '{session_id}' in dataset '{self.id}'") from None

    def list_sessions(self) -> List[Session]:
        return [self._sessions[sid] for sid in self._session_order]

    def merge(self, merge_id: str, session_ids: List[str]) -> "MergedContext":
        """Merge >=2 sessions into a derived, read-only MergedContext.

        Registration/feature-matching/scale-reconciliation (the "real"
        geometric half of §5) needs an actual CV backend this repo does
        not have yet -- deferred and recorded honestly (see
        MergedContext.registration_status), not faked. What IS real here:
        provenance is never destroyed (source sessions are untouched and
        still independently queryable), and conflict detection runs on
        the metadata that's actually available (coordinate frame
        agreement, temporal overlap).
        """
        if len(session_ids) < 2:
            raise ValueError("merge requires at least 2 sessions")
        sessions = [self.get_session(sid) for sid in session_ids]  # raises UnknownSessionError if any missing

        frames = {s.coordinate_frame for s in sessions}
        conflicts = []
        if len(frames) > 1:
            conflicts.append(f"coordinate_frame mismatch: {sorted(f.value for f in frames)}")

        times = [s.created_at for s in sessions if s.created_at is not None]
        overlap = None
        if len(times) == len(sessions) and times:
            overlap = {"earliest": min(times), "latest": max(times)}

        total_evidence = sum(len(s.all_evidence()) for s in sessions)

        merged = MergedContext(
            id=merge_id,
            source_session_ids=list(session_ids),
            conflicts=conflicts,
            temporal_overlap=overlap,
            evidence_count=total_evidence,
            registration_status="not_performed",  # honest: no geometric registration backend wired yet
        )
        self.merges[merge_id] = merged
        for s in sessions:
            s.record_processing("merged", detail={"merge_id": merge_id})
        _logger.info(
            "Sessions merged",
            context={"dataset_id": self.id, "merge_id": merge_id, "sources": session_ids, "conflicts": conflicts},
        )
        return merged

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "id": self.id,
            "name": self.name,
            "session_order": list(self._session_order),
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
            "merges": {mid: m.to_dict() for mid, m in self.merges.items()},
        }

    def deserialize(self, data: dict) -> None:
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported Dataset format version: {data.get('format_version')}")
        self.id = data["id"]
        self.name = data.get("name", "")
        self._sessions = {sid: Session.from_dict(v) for sid, v in data.get("sessions", {}).items()}
        self._session_order = list(data.get("session_order", list(self._sessions.keys())))
        self.merges = {mid: MergedContext.from_dict(v) for mid, v in data.get("merges", {}).items()}

    @staticmethod
    def from_dict(data: dict) -> "Dataset":
        d = Dataset(dataset_id=data["id"])
        d.deserialize(data)
        return d


@dataclass(frozen=True)
class MergedContext:
    """Derived artifact of merging sessions. Source sessions are untouched
    and remain independently valid -- this never deletes or mutates them."""
    id: str
    source_session_ids: List[str]
    conflicts: List[str] = field(default_factory=list)
    temporal_overlap: Optional[dict] = None
    evidence_count: int = 0
    registration_status: str = "not_performed"

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_session_ids": self.source_session_ids,
            "conflicts": self.conflicts,
            "temporal_overlap": self.temporal_overlap,
            "evidence_count": self.evidence_count,
            "registration_status": self.registration_status,
        }

    @staticmethod
    def from_dict(data: dict) -> "MergedContext":
        return MergedContext(
            id=data["id"],
            source_session_ids=list(data.get("source_session_ids", [])),
            conflicts=list(data.get("conflicts", [])),
            temporal_overlap=data.get("temporal_overlap"),
            evidence_count=data.get("evidence_count", 0),
            registration_status=data.get("registration_status", "not_performed"),
        )
