"""Evidence Matrix v2: the run / checkpoint / artifact / calibration context around every raw observation.

``EvidenceStore`` already keeps the immutable raw observation. What it did not record is *which run, architecture and checkpoint
produced it*, which camera it came from, where large raw outputs live, and the separate, explicitly-calibrated confidence. This
module adds exactly that as an append-only SIDECAR next to the store; it never edits or re-hashes an observation.

Large raw outputs go to the repository's existing content-addressed store (``world_ir.artifact_store``, ``artifact://<sha256>``);
nothing here re-implements blob storage.

Rules enforced here:
* A context field that is unknown stays ``None`` -- nothing is invented (no default architecture, no default checkpoint).
* ``calibrated_confidence`` may only be set together with a ``calibration_basis`` naming what it was calibrated against; the raw
  ``EngineObservation.confidence`` is never overwritten or reinterpreted.
* An observation carries ``artifact_value(uri)`` instead of an inline array; ``verify()`` re-hashes every referenced blob.
* Status (raw -> validated -> rejected) is an append-only history. A rejected row stays in the matrix: disagreement is never hidden.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.observation import EngineObservation
from world_ir.artifact_store import (
    ArtifactStore,
    FileArtifactStore,
    MemoryArtifactStore,
)

MATRIX_SCHEMA_VERSION = 2
STATUSES = ("raw", "validated", "rejected")
ARTIFACT_KEY = "$artifact"


class MatrixError(RuntimeError):
    pass


def artifact_value(uri: str, **meta: Any) -> dict[str, Any]:
    """What an observation's ``predicted`` carries in place of a large inline array."""
    return {ARTIFACT_KEY: uri, **meta}


def _artifact_uris_in(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        if ARTIFACT_KEY in value:
            yield str(value[ARTIFACT_KEY])
        for v in value.values():
            yield from _artifact_uris_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _artifact_uris_in(v)


# ------------------------------------------------------------------------------------------------ context record
@dataclass(frozen=True)
class EvidenceContext:
    observation_id: str
    run_id: str
    architecture_id: str | None = None
    checkpoint_id: str | None = None
    camera_ref: str | None = None
    world_transform: str | None = None  # id/description of the frame-graph edge used to express the value in the world frame
    artifact_refs: Mapping[str, str] = field(default_factory=dict)  # role -> artifact:// uri (masks, depth maps, ...)
    calibrated_confidence: float | None = None
    calibration_basis: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise MatrixError("an evidence context must name the run that produced the observation")
        if self.calibrated_confidence is not None:
            if not (0.0 <= float(self.calibrated_confidence) <= 1.0):
                raise MatrixError(f"calibrated_confidence {self.calibrated_confidence!r} is outside [0, 1]")
            if not self.calibration_basis:
                raise MatrixError(
                    "calibrated_confidence needs calibration_basis: what the confidence was calibrated against"
                )
        elif self.calibration_basis:
            raise MatrixError("calibration_basis was given without a calibrated_confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "architecture_id": self.architecture_id,
            "checkpoint_id": self.checkpoint_id,
            "camera_ref": self.camera_ref,
            "world_transform": self.world_transform,
            "artifact_refs": dict(sorted(self.artifact_refs.items())),
            "calibrated_confidence": self.calibrated_confidence,
            "calibration_basis": self.calibration_basis,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvidenceContext:
        return cls(
            observation_id=d["observation_id"],
            run_id=d["run_id"],
            architecture_id=d.get("architecture_id"),
            checkpoint_id=d.get("checkpoint_id"),
            camera_ref=d.get("camera_ref"),
            world_transform=d.get("world_transform"),
            artifact_refs=dict(d.get("artifact_refs") or {}),
            calibrated_confidence=d.get("calibrated_confidence"),
            calibration_basis=d.get("calibration_basis"),
        )


@dataclass(frozen=True)
class StatusEvent:
    observation_id: str
    status: str
    reason: str
    by: str


# ------------------------------------------------------------------------------------------------ the matrix
class EvidenceMatrix:
    """observation store + context sidecar + artifact store. ``root=None`` is fully in-memory."""

    def __init__(self, root: Path | None = None, *, artifacts: ArtifactStore | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self.store = EvidenceStore(self._root / "observations.jsonl" if self._root else None)
        self.artifacts: ArtifactStore = artifacts or (
            FileArtifactStore(self._root / "artifacts") if self._root else MemoryArtifactStore()
        )
        self._ctx: dict[str, EvidenceContext] = {}
        self._history: dict[str, list[StatusEvent]] = {}
        self._lock = threading.RLock()
        self._log = self._root / "context.jsonl" if self._root else None
        if self._log is not None and self._log.exists():
            self._load()

    def _has_uri(self, uri: str) -> bool:
        try:
            return self.artifacts.has(self.artifacts.digest_of(uri))
        except ValueError:
            return False

    # ---- writes
    def add(self, obs: EngineObservation, context: EvidenceContext) -> bool:
        """Store the raw observation and its context. Re-adding identical evidence is a no-op; a DIFFERENT context for an
        observation that already has one is refused (context is part of the evidence, it is not edited)."""
        if context.observation_id != obs.observation_id:
            raise MatrixError("context does not belong to this observation")
        with self._lock:
            old = self._ctx.get(obs.observation_id)
            if old is not None:
                if old != context:
                    raise MatrixError(
                        f"observation {obs.observation_id} already has a different context; it cannot be rewritten"
                    )
                return False
            for uri in {*context.artifact_refs.values(), *_artifact_uris_in(obs.predicted)}:
                if not self._has_uri(uri):
                    raise MatrixError(f"observation references artifact {uri!r} that is not in the artifact store")
            self.store.add(obs)
            self._ctx[obs.observation_id] = context
            self._history[obs.observation_id] = []
            self._append({"type": "context", **context.to_dict()})
            return True

    def set_status(self, observation_id: str, status: str, *, reason: str, by: str) -> None:
        if status not in STATUSES:
            raise MatrixError(f"unknown status {status!r}; expected one of {STATUSES}")
        if not reason or not by:
            raise MatrixError("a status change must state its reason and who made it")
        with self._lock:
            if observation_id not in self._ctx:
                raise MatrixError(f"no evidence row for observation {observation_id!r}")
            self._history[observation_id].append(StatusEvent(observation_id, status, reason, by))
            self._append(
                {"type": "status", "observation_id": observation_id, "status": status, "reason": reason, "by": by}
            )

    # ---- reads
    def status(self, observation_id: str) -> str:
        hist = self._history.get(observation_id)
        if hist is None:
            raise MatrixError(f"no evidence row for observation {observation_id!r}")
        return hist[-1].status if hist else "raw"

    def status_history(self, observation_id: str) -> list[StatusEvent]:
        return list(self._history[observation_id])

    def context(self, observation_id: str) -> EvidenceContext:
        try:
            return self._ctx[observation_id]
        except KeyError:
            raise MatrixError(f"no evidence row for observation {observation_id!r}") from None

    def row(self, observation_id: str) -> dict[str, Any]:
        """One matrix row: the untouched raw observation next to its context. Raw and calibrated confidence stay separate."""
        obs = self.store.get(observation_id)
        ctx = self.context(observation_id)
        return {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "observation": obs.to_dict(),
            "context": ctx.to_dict(),
            "raw_confidence": obs.confidence,
            "calibrated_confidence": ctx.calibrated_confidence,
            "status": self.status(observation_id),
            "status_history": [
                {"status": e.status, "reason": e.reason, "by": e.by} for e in self._history[observation_id]
            ],
        }

    def rows(self) -> list[dict[str, Any]]:
        return [self.row(i) for i in sorted(self._ctx)]

    def __len__(self) -> int:
        return len(self._ctx)

    # ---- integrity
    def verify(self) -> list[str]:
        """Every problem found, never raising on the first: observations with no context, missing or corrupt artifacts."""
        problems: list[str] = []
        for oid in sorted(o.observation_id for o in self.store.query()):
            if oid not in self._ctx:
                problems.append(f"observation {oid} has no evidence context")
        for oid, ctx in sorted(self._ctx.items()):
            uris = {*ctx.artifact_refs.values(), *_artifact_uris_in(self.store.get(oid).predicted)}
            for uri in sorted(uris):
                try:
                    data = self.artifacts.get(uri)
                except (KeyError, FileNotFoundError, ValueError):
                    problems.append(f"observation {oid}: artifact {uri!r} is missing")
                    continue
                if hashlib.sha256(data).hexdigest() != self.artifacts.digest_of(uri):
                    problems.append(f"observation {oid}: artifact {uri!r} fails its content hash (corrupted or tampered)")
        return problems

    # ---- persistence
    def _append(self, record: dict[str, Any]) -> None:
        if self._log is None:
            return
        self._log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self._log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def _load(self) -> None:
        assert self._log is not None
        for n, line in enumerate(self._log.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                kind = rec["type"]
                if kind == "context":
                    ctx = EvidenceContext.from_dict(rec)
                    self.store.get(ctx.observation_id)  # raises if the observation is not in the store
                    self._ctx[ctx.observation_id] = ctx
                    self._history.setdefault(ctx.observation_id, [])
                elif kind == "status":
                    if rec["status"] not in STATUSES or rec["observation_id"] not in self._ctx:
                        raise MatrixError("status record for an unknown row or status")
                    self._history[rec["observation_id"]].append(
                        StatusEvent(rec["observation_id"], rec["status"], rec["reason"], rec["by"])
                    )
                else:
                    raise MatrixError(f"unknown record type {kind!r}")
            except (KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                raise MatrixError(f"{self._log}:{n}: bad evidence-matrix record ({exc})") from exc
