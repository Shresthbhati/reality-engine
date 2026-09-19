"""Canonical Source model (multi-source ingestion milestone, spec §8-9).

DISTINCT LAYERS, never collapsed (spec §3):

    SOURCE      what the user supplied (this module)
    EVIDENCE    what the engine extracts/derives from a source
                (evidence/packages.py EvidenceAsset, evidence/session.py
                EvidenceItem)
    WORLDIR     the compiled interpretation of the world
                (world_ir/)

A Source carries identity (deterministic, content-derived id), format,
size, content hash, timestamps (caller-supplied or decoded from the
media itself -- never a wall clock), device/camera info when the media
declares it, explicit lifecycle status, and the ids of the evidence
assets derived from it. Derived bytes are NOT copied here; they live in
the EvidencePackage and are referenced by id.

Design invariants, matching the repo's established rules:

  - Deterministic identity. `source_id = src-<sha256[:16]>` over the
    file's bytes (or a sorted-tree hash for a directory). The same
    content always yields the same source id -- across sessions,
    across machines. This is what makes duplicate detection and
    incremental ingestion possible: adding the same file twice returns
    the SAME source, never a silent copy.
  - No clocks. imported_at/acquired_at are None unless the caller or
    the media supplies them (EXIF, container headers). Never "now".
  - Composite sources. A multi-sensor capture (phone scan, drone run)
    is ONE source with typed components (images / video / imu / gps /
    calibration / depth), each with its own hash. Component data
    models are deliberately NOT special: PHONE_CAPTURE and
    DRONE_CAPTURE differ only in metadata, not structure.
  - Explicit lifecycle. SourceStatus transitions DISCOVERED ->
    VALIDATING -> INGESTING -> INGESTED | PARTIAL | FAILED |
    UNSUPPORTED are recorded on the source; failures are structured
    (errors list), never silently swallowed.
  - Frozen dataclasses + to_dict/from_dict (format_version: 1,
    ValueError on mismatch) -- the established serialization shape.

LLM-free, stdlib-only, no clocks, no RNG.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "SourceType",
    "SourceStatus",
    "SourceComponent",
    "Source",
    "SourceRegistry",
    "DuplicateSourceError",
    "UnknownSourceError",
    "UnsupportedSourceError",
    "hash_file",
    "hash_tree",
    "classify_source",
    "IMAGE_EXTS",
    "VIDEO_EXTS",
    "POINT_CLOUD_EXTS",
]


class SourceType(str, Enum):
    """Extensible taxonomy. Composite captures keep general component
    structure -- phone/drone are metadata, not special data models."""
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"
    PHONE_CAPTURE = "phone_capture"
    DRONE_CAPTURE = "drone_capture"
    POINT_CLOUD = "point_cloud"
    RGBD_SEQUENCE = "rgbd_sequence"
    SENSOR_LOG = "sensor_log"
    DATASET = "dataset"
    COMPOSITE_CAPTURE = "composite_capture"
    OTHER = "other"


class SourceStatus(str, Enum):
    """Explicit lifecycle. A failed ingest is never classified as success."""
    DISCOVERED = "discovered"
    VALIDATING = "validating"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    PARTIAL = "partial"          # some components ingested, some failed
    FAILED = "failed"
    UNSUPPORTED = "unsupported"  # no adapter for this format/content


class DuplicateSourceError(ValueError):
    pass


class UnknownSourceError(ValueError):
    pass


class UnsupportedSourceError(ValueError):
    """No adapter can handle this source -- recorded, never faked."""


# ---------------------------------------------------------------------+
# Content addressing                                                    |
# ---------------------------------------------------------------------+

def hash_file(path: str, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a file's bytes, streamed (deterministic)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: str) -> str:
    """Deterministic content hash of a directory tree.

    Hashes each file's bytes, then folds relpath -> digest pairs in
    sorted order. Moving the ROOT does not change the digest; renaming
    files inside does (it changed the capture's own structure). No
    mtimes -- they don't survive copies.
    """
    paths: List[str] = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            paths.append(rel)
    paths.sort()
    digest = hashlib.sha256()
    digest.update(f"tree:{len(paths)}".encode("utf-8"))
    for rel in paths:
        file_digest = hash_file(os.path.join(root, rel))
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def source_id_for_digest(digest: str) -> str:
    """Content-derived, stable source id. The full sha256 is always
    carried alongside in Source.sha256."""
    return f"src-{digest[:16]}"


# ---------------------------------------------------------------------+
# Classification: path/extension -> SourceType                          |
# ---------------------------------------------------------------------+

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
POINT_CLOUD_EXTS = {".las", ".laz", ".ply", ".pcd", ".e57"}
SENSOR_LOG_EXTS = {".csv"}


def classify_source(path: str) -> SourceType:
    """Path-shape + extension classification. Directories are DATASET;
    the caller refines to COMPOSITE_CAPTURE when a manifest.json exists."""
    if os.path.isdir(path):
        return SourceType.DATASET
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return SourceType.IMAGE
    if ext in VIDEO_EXTS:
        return SourceType.VIDEO
    if ext in POINT_CLOUD_EXTS:
        return SourceType.POINT_CLOUD
    if ext in SENSOR_LOG_EXTS:
        return SourceType.SENSOR_LOG
    return SourceType.OTHER


# ---------------------------------------------------------------------+
# Source records                                                        |
# ---------------------------------------------------------------------+

@dataclass(frozen=True)
class SourceComponent:
    """One typed part of a (possibly composite) source. Its own hash, so
    component-level provenance survives even when the capture is
    reorganized on disk."""
    role: str  # "image" | "video" | "imu" | "gps" | "calibration" | "depth" | "point_cloud" | "telemetry"
    path: str
    sha256: str
    size_bytes: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict) -> "SourceComponent":
        return SourceComponent(
            role=data["role"],
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=int(data.get("size_bytes", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Source:
    """What the user supplied. Immutable once created; lifecycle
    transitions produce a new frozen instance via the registry."""
    source_id: str
    source_type: SourceType
    original_name: str
    path: str                 # where the user's bytes live (reference, not a copy)
    format: str               # extension ("mp4", "las", "") -- explicit, may be ""
    size_bytes: int = 0
    sha256: str = ""          # full sha256 (file) or tree hash (dir)
    acquired_at: Optional[float] = None   # decoded from media (EXIF etc.) or None -- never a wall clock
    imported_at: Optional[float] = None   # caller-supplied or None
    device: str = ""          # make/model when media declares it
    platform: str = ""        # "phone" | "drone" | "dslr" | "lidar" | ...
    session_id: str = ""
    status: SourceStatus = SourceStatus.DISCOVERED
    components: Tuple[SourceComponent, ...] = ()
    asset_ids: Tuple[str, ...] = ()       # evidence assets derived from this source
    parent_source_id: str = ""
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "original_name": self.original_name,
            "path": self.path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "acquired_at": self.acquired_at,
            "imported_at": self.imported_at,
            "device": self.device,
            "platform": self.platform,
            "session_id": self.session_id,
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "asset_ids": list(self.asset_ids),
            "parent_source_id": self.parent_source_id,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }



    @staticmethod
    def from_dict(data: dict) -> "Source":
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported Source format version: {data.get('format_version')}")
        return Source(
            source_id=data["source_id"],
            source_type=SourceType(data.get("source_type", SourceType.OTHER.value)),
            original_name=data.get("original_name", ""),
            path=data.get("path", ""),
            format=data.get("format", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=data.get("sha256", ""),
            acquired_at=data.get("acquired_at"),
            imported_at=data.get("imported_at"),
            device=data.get("device", ""),
            platform=data.get("platform", ""),
            session_id=data.get("session_id", ""),
            status=SourceStatus(data.get("status", SourceStatus.DISCOVERED.value)),
            components=tuple(SourceComponent.from_dict(c) for c in data.get("components", [])),
            asset_ids=tuple(data.get("asset_ids", [])),
            parent_source_id=data.get("parent_source_id", ""),
            warnings=tuple(data.get("warnings", [])),
            errors=tuple(data.get("errors", [])),
            metadata=dict(data.get("metadata", {})),
        )


class SourceRegistry:

    """Append-only registry of sources for one session, keyed by content
    hash. Re-registering the same bytes is NOT a duplicate row -- it
    resolves to the existing source (the caller gets `created=False`),
    which is the whole point of content-derived identity."""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._sources: Dict[str, Source] = {}      # source_id -> Source
        self._by_hash: Dict[str, str] = {}         # sha256 -> source_id

    def register(
        self,
        path: str,
        *,
        source_type: SourceType,
        sha256: str,
        size_bytes: int = 0,
        platform: str = "",
        device: str = "",
        acquired_at: Optional[float] = None,
        imported_at: Optional[float] = None,
        components: Tuple[SourceComponent, ...] = (),
        parent_source_id: str = "",
        metadata: Optional[Dict[str, object]] = None,
        format_ext: str = "",
    ) -> Tuple[Source, bool]:
        """Register or resolve a source. Returns (source, created).
        `created=False` means these exact bytes were already registered:
        the existing source is returned unchanged."""
        existing_id = self._by_hash.get(sha256)
        if existing_id is not None:
            return self._sources[existing_id], False
        source = Source(
            source_id=source_id_for_digest(sha256),
            source_type=source_type,
            original_name=os.path.basename(path.rstrip("\\/")) or path,
            path=path,
            format=format_ext or os.path.splitext(path)[1].lower().lstrip("."),
            size_bytes=size_bytes,
            sha256=sha256,
            acquired_at=acquired_at,
            imported_at=imported_at,
            device=device,
            platform=platform,
            session_id=self.session_id,
            status=SourceStatus.DISCOVERED,
            components=components,
            parent_source_id=parent_source_id,
            metadata=dict(metadata or {}),
        )
        self._sources[source.source_id] = source
        self._by_hash[sha256] = source.source_id
        return source, True

    def get(self, source_id: str) -> Source:
        try:
            return self._sources[source_id]
        except KeyError:
            raise UnknownSourceError(
                f"no source '{source_id}' in session '{self.session_id}'"
            ) from None

    def update(self, source_id: str, **changes) -> Source:
        """Replace one or more fields (status, asset_ids, ...), returning
        the new frozen instance. Identity is unchanged."""
        current = self.get(source_id)
        updated = replace(current, **changes)
        self._sources[source_id] = updated
        return updated

    def all_sources(self) -> List[Source]:
        """In registration (insertion) order -- deterministic."""
        return list(self._sources.values())

    def __len__(self) -> int:
        return len(self._sources)

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "session_id": self.session_id,
            "source_order": [s.source_id for s in self.all_sources()],
            "sources": {s.source_id: s.to_dict() for s in self.all_sources()},
        }

    def deserialize(self, data: dict) -> None:
        if data.get("format_version") != 1:
            raise ValueError(
                f"Unsupported SourceRegistry format version: {data.get('format_version')}"
            )
        self.session_id = data.get("session_id", self.session_id)
        self._sources = {sid: Source.from_dict(v) for sid, v in data.get("sources", {}).items()}
        self._by_hash = {s.sha256: s.source_id for s in self._sources.values()}

    @staticmethod
    def from_dict(data: dict) -> "SourceRegistry":
        registry = SourceRegistry(session_id=data.get("session_id", ""))
        registry.deserialize(data)
        return registry
