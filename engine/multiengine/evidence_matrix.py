"""Evidence Matrix: Append-only multi-engine evidence store with run/checkpoint context,
artifact offloading, tamper verification, and universal reconstruction support.

Supports both:
1. Sonnet's Evidence Matrix v2 sidecar (EvidenceContext, EvidenceMatrix, artifact_value).
2. Gemini's Shared Evidence Matrix for Universal Perception (EvidenceMatrixEntry, EvidenceMatrixStore).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
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


class EvidenceMatrixError(RuntimeError):
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


LEGACY_UNSPECIFIED = "unspecified (legacy record)"


@dataclass(frozen=True)
class CalibrationRef:
    """What a calibrated confidence was calibrated WITH: the dataset, the method and its version. All three are required.
    A legacy record that only carried a free-text basis is kept readable but marked incomplete (``complete`` is False)."""

    dataset: str
    method: str
    version: str

    def __post_init__(self) -> None:
        if not (self.dataset and self.method and self.version):
            raise MatrixError("a calibration must name its dataset, method and version")

    @property
    def complete(self) -> bool:
        return LEGACY_UNSPECIFIED not in (self.method, self.version)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "method": self.method, "version": self.version, "complete": self.complete}


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
    calibration_basis: str | None = None  # legacy free text; new records give ``calibration`` instead
    validation: str | None = None  # outcome of the validation actually applied; None = none was applied (never assumed passed)
    calibration: CalibrationRef | None = None  # dataset + method + version behind ``calibrated_confidence``

    def __post_init__(self) -> None:
        if not self.run_id:
            raise MatrixError("an evidence context must name the run that produced the observation")
        if self.calibrated_confidence is not None:
            if not (0.0 <= float(self.calibrated_confidence) <= 1.0):
                raise MatrixError(f"calibrated_confidence {self.calibrated_confidence!r} is outside [0, 1]")
            if self.calibration is None:
                if not self.calibration_basis:
                    raise MatrixError(
                        "calibrated_confidence needs calibration=CalibrationRef(dataset, method, version) "
                        "(or, for a legacy record, a calibration_basis)"
                    )
                # a legacy free-text basis stays readable; it is marked incomplete rather than pretending to be complete
                object.__setattr__(
                    self, "calibration", CalibrationRef(self.calibration_basis, LEGACY_UNSPECIFIED, LEGACY_UNSPECIFIED)
                )
        elif self.calibration_basis or self.calibration is not None:
            raise MatrixError("a calibration was given without a calibrated_confidence")

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
            "calibration": None if self.calibration is None else self.calibration.to_dict(),
            "validation": self.validation,
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
            validation=d.get("validation"),
            calibration=CalibrationRef(c["dataset"], c["method"], c["version"]) if (c := d.get("calibration")) else None,
        )


@dataclass(frozen=True)
class RunRecord:
    """One engine execution, kept whether it succeeded, failed or was unavailable: a failure is evidence too."""

    run_id: str
    task: str
    engine_id: str
    model_version: str
    status: str  # EngineRunStatus value: succeeded | failed | unavailable | cancelled
    architecture_id: str | None = None
    checkpoint_id: str | None = None
    device: str | None = None
    error_kind: str | None = None
    error: str = ""
    wall_s: float | None = None
    peak_vram_mb: float | None = None
    attempts: int = 1
    fallback: Mapping[str, Any] | None = None
    sources: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    output_artifact: str | None = None  # artifact:// uri of the run's complete raw output bundle

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "task": self.task, "engine_id": self.engine_id, "model_version": self.model_version,
            "status": self.status, "architecture_id": self.architecture_id, "checkpoint_id": self.checkpoint_id,
            "device": self.device, "error_kind": self.error_kind, "error": self.error, "wall_s": self.wall_s,
            "peak_vram_mb": self.peak_vram_mb, "attempts": self.attempts,
            "fallback": dict(self.fallback) if self.fallback else None, "sources": list(self.sources),
            "observation_ids": list(self.observation_ids), "output_artifact": self.output_artifact,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunRecord:
        return cls(
            run_id=d["run_id"], task=d["task"], engine_id=d["engine_id"], model_version=d["model_version"],
            status=d["status"], architecture_id=d.get("architecture_id"), checkpoint_id=d.get("checkpoint_id"),
            device=d.get("device"), error_kind=d.get("error_kind"), error=d.get("error", ""), wall_s=d.get("wall_s"),
            peak_vram_mb=d.get("peak_vram_mb"), attempts=d.get("attempts", 1), fallback=d.get("fallback"),
            sources=tuple(d.get("sources") or ()), observation_ids=tuple(d.get("observation_ids") or ()),
            output_artifact=d.get("output_artifact"),
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
        self._runs: list[RunRecord] = []
        self._lock = threading.RLock()
        self._log = self._root / "context.jsonl" if self._root else None
        if self._log is not None and self._log.exists():
            self._load()

    def _has_uri(self, uri: str) -> bool:
        try:
            return self.artifacts.has(self.artifacts.digest_of(uri))
        except ValueError:
            return False

    def has_artifact(self, uri: str) -> bool:
        return self._has_uri(uri)

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

    def record_run(self, run: RunRecord) -> None:
        """Append one engine execution (successful or not). Runs are a log: the same engine may legitimately run again."""
        if not run.run_id or not run.engine_id:
            raise MatrixError("a run record needs a run id and an engine id")
        with self._lock:
            if run.output_artifact is not None and not self._has_uri(run.output_artifact):
                raise MatrixError(f"run output artifact {run.output_artifact!r} is not in the artifact store")
            self._runs.append(run)
            self._append({"type": "run", **run.to_dict()})

    def runs(self, run_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._runs if run_id is None or r.run_id == run_id]

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
        for r in self._runs:
            if r.output_artifact is None:
                continue
            try:
                data = self.artifacts.get(r.output_artifact)
            except (KeyError, FileNotFoundError, ValueError):
                problems.append(f"run {r.run_id}/{r.engine_id}: output artifact {r.output_artifact!r} is missing")
                continue
            if hashlib.sha256(data).hexdigest() != self.artifacts.digest_of(r.output_artifact):
                problems.append(f"run {r.run_id}/{r.engine_id}: output artifact fails its content hash")
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
                elif kind == "run":
                    self._runs.append(RunRecord.from_dict(rec))
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


# ------------------------------------------------------------------------------------------------ Universal Evidence Matrix Store
def _compute_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class EvidenceMatrixEntry:
    """An immutable record of an engine prediction or observation."""

    evidence_id: str
    model_id: str
    architecture_id: str
    checkpoint_id: str
    run_id: str
    source_ids: list[str]
    input_parameters: dict[str, Any]
    prediction_type: str  # e.g., "point_cloud", "depth_map", "bounding_box_3d", "room_layout"
    raw_output_reference: str  # file path / URI to offloaded artifact
    coordinate_frame: str  # "camera_local", "world_canonical", "enu", etc.
    camera_reference: dict[str, Any] | None = None
    world_transform: list[list[float]] | None = None  # 4x4 matrix
    entity_association: str | None = None
    confidence: float = 1.0
    uncertainty: dict[str, Any] = field(default_factory=dict)
    calibration_info: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "completed"  # "completed", "partial", "failed"
    timestamps: dict[str, float] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamps:
            self.timestamps = {"created_at": time.time()}
        if not self.content_hash:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 hash across immutable core fields."""
        payload = {
            "evidence_id": self.evidence_id,
            "model_id": self.model_id,
            "architecture_id": self.architecture_id,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "source_ids": sorted(self.source_ids),
            "input_parameters": self.input_parameters,
            "prediction_type": self.prediction_type,
            "raw_output_reference": self.raw_output_reference,
            "coordinate_frame": self.coordinate_frame,
            "camera_reference": self.camera_reference,
            "world_transform": self.world_transform,
            "entity_association": self.entity_association,
            "confidence": round(float(self.confidence), 6),
            "uncertainty": self.uncertainty,
            "execution_status": self.execution_status,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _compute_sha256_bytes(encoded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceMatrixEntry:
        return cls(**data)


class EvidenceMatrixStore:
    """Thread-safe append-only store for EvidenceMatrixEntry items with large artifact offloading."""

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.artifacts_dir = self.storage_root / "evidence_artifacts"
        self.index_path = self.storage_root / "evidence_matrix.jsonl"
        self._lock = threading.RLock()
        self._entries: dict[str, EvidenceMatrixEntry] = {}
        self._artifact_hashes: dict[str, str] = {}  # artifact_rel_path -> sha256

        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        with open(self.index_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    entry_type = raw.get("_matrix_record_type", "entry")
                    if entry_type == "entry":
                        raw.pop("_matrix_record_type", None)
                        entry = EvidenceMatrixEntry.from_dict(raw)
                        expected_hash = entry.compute_hash()
                        if entry.content_hash != expected_hash:
                            raise EvidenceMatrixError(
                                f"Corrupt entry at line {line_no}: hash {entry.content_hash} "
                                f"does not match computed {expected_hash}"
                            )
                        self._entries[entry.evidence_id] = entry
                    elif entry_type == "artifact_hash":
                        self._artifact_hashes[raw["rel_path"]] = raw["sha256"]
                except Exception as e:
                    raise EvidenceMatrixError(f"Failed to read index at line {line_no}: {e}") from e

    def save_artifact(self, evidence_id: str, filename: str, data: bytes | str | Path) -> str:
        """Saves a binary or text artifact offloaded from metadata. Returns relative URI/path."""
        evidence_artifact_dir = self.artifacts_dir / evidence_id
        evidence_artifact_dir.mkdir(parents=True, exist_ok=True)
        target_path = evidence_artifact_dir / filename

        with self._lock:
            if isinstance(data, bytes):
                target_path.write_bytes(data)
                sha = _compute_sha256_bytes(data)
            elif isinstance(data, str):
                encoded = data.encode("utf-8")
                target_path.write_bytes(encoded)
                sha = _compute_sha256_bytes(encoded)
            elif isinstance(data, Path):
                if not data.exists():
                    raise EvidenceMatrixError(f"Artifact source file does not exist: {data}")
                content = data.read_bytes()
                target_path.write_bytes(content)
                sha = _compute_sha256_bytes(content)
            else:
                raise TypeError(f"Unsupported artifact data type: {type(data)}")

            rel_path = f"evidence_artifacts/{evidence_id}/{filename}"
            self._artifact_hashes[rel_path] = sha

            # Persist artifact hash record
            record = {
                "_matrix_record_type": "artifact_hash",
                "rel_path": rel_path,
                "sha256": sha,
                "saved_at": time.time(),
            }
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            return rel_path

    def get_artifact(self, rel_path: str) -> bytes:
        """Reads offloaded artifact bytes, validating against recorded sha256."""
        target_path = self.storage_root / rel_path
        if not target_path.exists():
            raise EvidenceMatrixError(f"Artifact not found on disk: {rel_path}")

        data = target_path.read_bytes()
        recorded_sha = self._artifact_hashes.get(rel_path)
        if recorded_sha:
            computed_sha = _compute_sha256_bytes(data)
            if computed_sha != recorded_sha:
                raise EvidenceMatrixError(
                    f"Artifact corrupted: {rel_path} has hash {computed_sha}, expected {recorded_sha}"
                )
        return data

    def add_entry(self, entry: EvidenceMatrixEntry) -> bool:
        """Appends a new EvidenceMatrixEntry. Enforces immutability and content hash match."""
        computed_hash = entry.compute_hash()
        if entry.content_hash and entry.content_hash != computed_hash:
            raise EvidenceMatrixError(
                f"Entry content_hash mismatch: expected {computed_hash}, got {entry.content_hash}"
            )
        entry.content_hash = computed_hash

        with self._lock:
            if entry.evidence_id in self._entries:
                raise EvidenceMatrixError(f"Entry {entry.evidence_id} already exists (immutable store)")

            self._entries[entry.evidence_id] = entry
            payload = {"_matrix_record_type": "entry", **entry.to_dict()}
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            return True

    def get_entry(self, evidence_id: str) -> EvidenceMatrixEntry:
        with self._lock:
            if evidence_id not in self._entries:
                raise EvidenceMatrixError(f"Unknown evidence_id: {evidence_id}")
            return self._entries[evidence_id]

    def query(
        self,
        *,
        model_id: str | None = None,
        prediction_type: str | None = None,
        run_id: str | None = None,
        source_id: str | None = None,
        entity_association: str | None = None,
    ) -> list[EvidenceMatrixEntry]:
        """Queries stored entries matching filters."""
        with self._lock:
            results = []
            for entry in self._entries.values():
                if model_id is not None and entry.model_id != model_id:
                    continue
                if prediction_type is not None and entry.prediction_type != prediction_type:
                    continue
                if run_id is not None and entry.run_id != run_id:
                    continue
                if source_id is not None and source_id not in entry.source_ids:
                    continue
                if entity_association is not None and entry.entity_association != entity_association:
                    continue
                results.append(entry)
            return results

    def verify_immutable(self) -> tuple[bool, list[str]]:
        """Verifies content hashes for every entry and artifact. Returns (all_valid, errors)."""
        errors: list[str] = []
        with self._lock:
            for eid, entry in self._entries.items():
                computed = entry.compute_hash()
                if entry.content_hash != computed:
                    errors.append(f"Entry {eid} hash violation: {entry.content_hash} vs {computed}")

            for rel_path, expected_sha in self._artifact_hashes.items():
                full_path = self.storage_root / rel_path
                if not full_path.exists():
                    errors.append(f"Missing artifact file: {rel_path}")
                    continue
                actual_sha = _compute_sha256_file(full_path)
                if actual_sha != expected_sha:
                    errors.append(f"Artifact {rel_path} corrupted: actual {actual_sha} != expected {expected_sha}")

        return len(errors) == 0, errors

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
