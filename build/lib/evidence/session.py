"""Evidence/Session model (spec: capture layer, dataset builder, session model).

A Session is the immutable evidence unit real-world capture flows into.
Everything downstream (merge, reconstruction, WorldIR) must trace back to
one or more Sessions rather than opaque files -- this is the single
foundational piece the rest of the reconstruction pipeline depends on,
so it comes before any of that pipeline is built.

Design choices, explicit rather than assumed:
  - EvidenceItem is frozen and append-only on Session: once evidence is
    added it is never mutated or removed (spec: "do not mutate original
    evidence"). There is no remove_evidence().
  - captured_at/created_at are None unless the caller supplies them --
    no wall-clock default. WorldIR's created_at/modified_at already hit
    this exact bug (nondeterministic default read at construction time);
    not repeating it here.
  - Session.status is a small state machine (OPEN -> PROCESSED ->
    ARCHIVED), matching the WorldLifecycle pattern already used in
    engine/world/lifecycle.py.
  - serialize()/deserialize() follow this codebase's established shape:
    format_version: 1, ValueError on mismatch, deserialize(self, data) as
    an instance method (not a classmethod -- that was a real fix-round
    finding on the rain/water branches, applying the lesson here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from provenance import Provenance, Uncertainty
from world_ir.coordinates import Frame
from engine.core.logging import get_logger

_logger = get_logger("evidence.session")


class EvidenceKind(str, Enum):
    """spec: capture layer evidence types (photo, video, point cloud, ...)."""
    PHOTO = "photo"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    DEPTH = "depth"
    LIDAR = "lidar"
    GPS_TRACK = "gps_track"
    IMU = "imu"
    MANUAL_MEASUREMENT = "manual_measurement"
    OTHER = "other"


class SessionStatus(str, Enum):
    OPEN = "open"          # still accepting evidence
    PROCESSED = "processed"  # derived artifacts exist; evidence still append-only
    ARCHIVED = "archived"    # read-only, no further evidence accepted


class DuplicateEvidenceError(ValueError):
    pass


class UnknownEvidenceError(ValueError):
    pass


class SessionClosedError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceItem:
    """One immutable piece of captured evidence. Never mutated after creation."""
    id: str
    kind: EvidenceKind
    source_uri: str  # where the raw evidence lives (file://, s3://, etc.) -- never copied/rewritten here
    captured_at: Optional[float] = None  # explicit; None means "not recorded", not "now"
    sha256: Optional[str] = None  # content hash, when available, to detect tampering/duplication
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = Provenance.OBSERVED
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "source_uri": self.source_uri,
            "captured_at": self.captured_at,
            "sha256": self.sha256,
            "metadata": self.metadata,
            "provenance": self.provenance.value,
            "uncertainty": self.uncertainty.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "EvidenceItem":
        return EvidenceItem(
            id=data["id"],
            kind=EvidenceKind(data["kind"]),
            source_uri=data["source_uri"],
            captured_at=data.get("captured_at"),
            sha256=data.get("sha256"),
            metadata=data.get("metadata", {}),
            provenance=Provenance(data.get("provenance", Provenance.OBSERVED.value)),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
        )


@dataclass
class ProcessingRecord:
    """One entry in a Session's processing history (spec: "processing history")."""
    operation: str
    at_tick: Optional[int] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"operation": self.operation, "at_tick": self.at_tick, "detail": self.detail}

    @staticmethod
    def from_dict(data: dict) -> "ProcessingRecord":
        return ProcessingRecord(operation=data["operation"], at_tick=data.get("at_tick"), detail=data.get("detail", {}))


class Session:
    """Immutable-evidence, append-only capture session.

    Evidence is added via add_evidence() and never removed or mutated --
    this is the core "do not mutate original evidence" invariant. Session
    metadata (name, status, coordinate_frame) may change; the evidence
    list itself only grows.
    """

    def __init__(
        self,
        session_id: str,
        name: str = "",
        coordinate_frame: Frame = Frame.SESSION_LOCAL,
        created_at: Optional[float] = None,
    ):
        self.id = session_id
        self.name = name
        self.coordinate_frame = coordinate_frame
        self.created_at = created_at
        self.status = SessionStatus.OPEN
        self._evidence: Dict[str, EvidenceItem] = {}
        self._evidence_order: List[str] = []  # insertion order, for deterministic iteration
        self.processing_history: List[ProcessingRecord] = []

    def add_evidence(self, item: EvidenceItem) -> None:
        if self.status == SessionStatus.ARCHIVED:
            raise SessionClosedError(f"session '{self.id}' is archived; cannot accept new evidence")
        if item.id in self._evidence:
            raise DuplicateEvidenceError(f"evidence '{item.id}' already exists in session '{self.id}'")
        self._evidence[item.id] = item
        self._evidence_order.append(item.id)
        _logger.info(
            "Evidence added to session",
            context={"session_id": self.id, "evidence_id": item.id, "kind": item.kind.value},
        )

    def get_evidence(self, evidence_id: str) -> EvidenceItem:
        try:
            return self._evidence[evidence_id]
        except KeyError:
            raise UnknownEvidenceError(f"no evidence '{evidence_id}' in session '{self.id}'") from None

    def all_evidence(self) -> List[EvidenceItem]:
        """Evidence in insertion (capture) order -- deterministic, not dict iteration order."""
        return [self._evidence[eid] for eid in self._evidence_order]

    def evidence_by_kind(self, kind: EvidenceKind) -> List[EvidenceItem]:
        return [e for e in self.all_evidence() if e.kind == kind]

    def record_processing(self, operation: str, at_tick: Optional[int] = None, detail: Optional[Dict[str, Any]] = None) -> None:
        self.processing_history.append(ProcessingRecord(operation=operation, at_tick=at_tick, detail=detail or {}))
        if operation == "archive":
            self.status = SessionStatus.ARCHIVED
        elif self.status == SessionStatus.OPEN and operation not in ("create",):
            self.status = SessionStatus.PROCESSED

    def archive(self) -> None:
        self.record_processing("archive")

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "id": self.id,
            "name": self.name,
            "coordinate_frame": self.coordinate_frame.value,
            "created_at": self.created_at,
            "status": self.status.value,
            "evidence_order": list(self._evidence_order),
            "evidence": {eid: e.to_dict() for eid, e in self._evidence.items()},
            "processing_history": [p.to_dict() for p in self.processing_history],
        }

    def deserialize(self, data: dict) -> None:
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported Session format version: {data.get('format_version')}")
        self.id = data["id"]
        self.name = data.get("name", "")
        self.coordinate_frame = Frame(data.get("coordinate_frame", Frame.SESSION_LOCAL.value))
        self.created_at = data.get("created_at")
        self.status = SessionStatus(data.get("status", SessionStatus.OPEN.value))
        self._evidence = {eid: EvidenceItem.from_dict(v) for eid, v in data.get("evidence", {}).items()}
        self._evidence_order = list(data.get("evidence_order", list(self._evidence.keys())))
        self.processing_history = [ProcessingRecord.from_dict(p) for p in data.get("processing_history", [])]

    @staticmethod
    def from_dict(data: dict) -> "Session":
        s = Session(session_id=data["id"])
        s.deserialize(data)
        return s
