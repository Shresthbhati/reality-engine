"""Shared Evidence Matrix (engine/multiengine/evidence_matrix.py).

Provides an append-only, tamper-verifiable Shared Evidence Matrix for multi-engine
observations, predictions, and geometric reconstructions.

Key design principles:
1. Append-Only: No updates, no deletes. Revisions or refinements are recorded as new
   linked entries.
2. Large Artifact Offloading: Point clouds, depth maps, meshes, and high-dimensional
   tensors are offloaded to dedicated filesystem artifacts under
   `storage_root / "evidence_artifacts" / evidence_id / filename` and referenced by URI/path.
   Large blobs are never inlined into JSON metadata.
3. Immutability & Verifiability: Every entry and offloaded artifact has an immutable
   SHA-256 hash. `verify_immutable()` verifies the integrity of every entry and file on disk.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


class EvidenceMatrixError(RuntimeError):
    pass


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
        errors: List[str] = []
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
