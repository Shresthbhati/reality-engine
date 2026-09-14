"""Multi-source ingestion session (milestone: unified source ingestion /
multi-source sessions -- SOURCE is what the user supplied, distinct from
the EVIDENCE it produces).

A ``MultiSourceSession`` lets a caller build one logical session out of
many sources added over time (a phone video today, a drone folder
tomorrow, a LAS scan next week) without re-importing what was already
ingested. It is a thin orchestration layer over the existing, already
deterministic evidence stack (``evidence.importers`` +
``evidence.packages.DeterministicPackageBuilder``/``EvidencePackage``) --
no parallel evidence model, no new binary storage, no LLM.

Design:

  - SOURCE vs EVIDENCE: a ``SourceRecord`` tracks what was handed to the
    session (a path, its content identity, and what happened when it was
    ingested); the resulting ``EvidenceAsset``s live in the session's one
    accumulated ``EvidencePackage`` exactly as they already do today.
    ``SourceRecord.asset_ids`` is the provenance link between the two.
  - Source-level dedup: a source is identified by a content hash (file
    bytes, or a hash over the sorted (relative-path, file-hash) pairs
    for a directory) so re-adding the same file/folder is a no-op that
    reports the existing record (``SourceStatus.ALREADY_INGESTED``)
    rather than re-ingesting or duplicating package content.
  - True incremental ingestion: adding a new source does not replay
    previously ingested sources. The builder used for one ``add_source``
    call is seeded with the assets/sources already in the session's
    package (so asset ids for OLD content never change) and only the new
    path is walked/decoded.
  - Honest failure: an unsupported extension or a decoder that refuses a
    corrupt file does not raise out of ``add_source`` and does not touch
    the package -- it is recorded as a ``SourceRecord`` with
    ``SourceStatus.FAILED``/``UNSUPPORTED`` and an ``error`` message, so
    a multi-source import loop can keep going. Nothing is silently
    dropped: `add_source` always returns the record it created.
  - Deterministic, serializable: ``to_dict()``/``from_dict()`` follow the
    repo's established shape (format_version: 1, ValueError on
    mismatch), embedding the full ``EvidencePackage`` so a session
    directory needs exactly one JSON file to resume from.

LLM-free; no clocks beyond what the underlying importers already accept
as caller-supplied.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from evidence.frames import IFrameSelectionStrategy
from evidence.importers import (
    FolderImportReport,
    ImporterCapabilityError,
    _KNOWN_EXTS,
    _LAS_EXTS,
    _PHOTO_EXTS,
    _VIDEO_EXTS,
    import_file,
    import_folder,
)
from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidencePackage,
    EvidenceSource,
)

__all__ = [
    "SourceStatus",
    "SourceType",
    "SourceRecord",
    "MultiSourceSession",
]


class SourceStatus(str, Enum):
    INGESTED = "ingested"                  # new content, successfully imported
    ALREADY_INGESTED = "already_ingested"  # identical content already in this session
    UNSUPPORTED = "unsupported"            # single-file extension with no importer
    FAILED = "failed"                      # importer refused (corrupt / missing decoder)


class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    DATASET = "dataset"    # a directory of mixed evidence
    UNKNOWN = "unknown"    # unrecognized single-file extension


def _source_type_of(path: str) -> SourceType:
    if os.path.isdir(path):
        return SourceType.DATASET
    extension = os.path.splitext(path)[1].lower()
    if extension in _VIDEO_EXTS:
        return SourceType.VIDEO
    if extension in _PHOTO_EXTS:
        return SourceType.IMAGE
    if extension in _LAS_EXTS:
        return SourceType.POINT_CLOUD
    return SourceType.UNKNOWN


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _content_hash_of_path(path: str) -> str:
    """Stable identity for a source, independent of where it lives on
    disk: a single file hashes its own bytes; a directory hashes the
    sorted set of (relative path, file sha256) pairs, so copying the
    exact same tree to a new location still dedupes."""
    if os.path.isfile(path):
        return _sha256_file(path)
    entries: List[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            entries.append(f"{rel}:{_sha256_file(full)}")
    entries.sort()
    return hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()


@dataclass
class SourceRecord:
    """One entry in a session's provenance ledger: what was handed to
    `add_source`, and what came of it. Never mutated after creation --
    a session's source list only grows, matching evidence/session.py's
    append-only evidence discipline."""

    source_id: str
    original_path: str
    source_type: SourceType
    content_hash: str
    status: SourceStatus
    asset_ids: List[str] = field(default_factory=list)
    unhandled_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "original_path": self.original_path,
            "source_type": self.source_type.value,
            "content_hash": self.content_hash,
            "status": self.status.value,
            "asset_ids": list(self.asset_ids),
            "unhandled_paths": list(self.unhandled_paths),
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: dict) -> "SourceRecord":
        return SourceRecord(
            source_id=data["source_id"],
            original_path=data["original_path"],
            source_type=SourceType(data["source_type"]),
            content_hash=data["content_hash"],
            status=SourceStatus(data["status"]),
            asset_ids=list(data.get("asset_ids", [])),
            unhandled_paths=list(data.get("unhandled_paths", [])),
            error=data.get("error"),
        )


class MultiSourceSession:
    """One logical capture session accumulating evidence from many
    sources over time, backed by a single deterministic EvidencePackage.
    """

    FORMAT_VERSION = 1

    def __init__(self, session_id: str, name: str = "", seed: Optional[str] = None):
        self.session_id = session_id
        self.name = name
        self.seed = seed or session_id
        self._sources: Dict[str, SourceRecord] = {}
        self._source_order: List[str] = []
        self._package = EvidencePackage(package_id=f"pkg-{self.seed}", seed=self.seed)

    @property
    def package(self) -> EvidencePackage:
        return self._package

    def sources(self) -> List[SourceRecord]:
        """Sources in the order they were added -- deterministic."""
        return [self._sources[sid] for sid in self._source_order]

    def evidence_summary(self) -> dict:
        """Real, computed session-level readiness -- reuses the
        reconstruction orchestrator's own gate (MIN_IMAGE_EVIDENCE, etc.)
        rather than a second guess at what "enough evidence" means."""
        from reconstruction.orchestrator import EvidenceValidationError, validate_evidence

        assets = self._package.all_assets()
        asset_counts: Dict[str, int] = {}
        gps_asset_count = 0
        for asset in assets:
            asset_counts[asset.kind.value] = asset_counts.get(asset.kind.value, 0) + 1
            if "latitude_deg" in asset.sensor_metadata:
                gps_asset_count += 1

        readiness_issues: List[str] = []
        try:
            validate_evidence(self._package.to_evidence_items())
            ready = True
        except EvidenceValidationError as exc:
            ready = False
            readiness_issues = list(exc.issues)

        return {
            "source_count": len(self._source_order),
            "asset_counts": asset_counts,
            "gps_asset_count": gps_asset_count,
            "ready_for_reconstruction": ready,
            "readiness_issues": readiness_issues,
        }

    def source_for_content(self, content_hash: str) -> Optional[SourceRecord]:
        for record in self._sources.values():
            if record.content_hash == content_hash:
                return record
        return None

    def _builder_seeded_from_package(self) -> DeterministicPackageBuilder:
        """A builder pre-loaded with everything already in the session's
        package, so a new add_source call continues asset numbering
        instead of replaying old sources (spec: incremental ingestion)."""
        builder = DeterministicPackageBuilder(seed=self.seed)
        for source in self._package.sources.values():
            builder.register_source(source)
        for asset in self._package.all_assets():
            builder._assets.append(asset)  # noqa: SLF001 -- same module family, intentional resume
            builder._seen_content[asset.sha256] = asset.id
        return builder

    def add_source(
        self,
        path: str,
        *,
        source: Optional[EvidenceSource] = None,
        frame_strategy: Optional[IFrameSelectionStrategy] = None,
    ) -> SourceRecord:
        """Ingest one file or directory into the session.

        Returns the SourceRecord for this path -- always, even on
        failure/unsupported. A source whose content hash already exists
        in this session is a no-op (ALREADY_INGESTED, existing record
        returned unchanged); the package is only touched on genuinely
        new content.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        content_hash = _content_hash_of_path(path)
        existing = self.source_for_content(content_hash)
        if existing is not None:
            return existing

        source_id = f"src-{len(self._source_order):04d}-{content_hash[:12]}"
        source_type = _source_type_of(path)
        use_source = source or EvidenceSource(
            source_id=f"disk:{os.path.basename(os.path.normpath(path))}",
            platform="filesystem",
            device="local disk",
        )

        record: SourceRecord
        builder = self._builder_seeded_from_package()
        builder.register_source(use_source)
        try:
            if os.path.isdir(path):
                report = import_folder(builder, path, source=use_source, frame_strategy=frame_strategy)
            else:
                report = FolderImportReport()
                if os.path.splitext(path)[1].lower() not in _KNOWN_EXTS:
                    raise ValueError(f"unknown evidence extension for {path!r}")
                import_file(builder, path, source=use_source, report=report, frame_strategy=frame_strategy)
        except ValueError as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.UNSUPPORTED, error=str(exc),
            )
        except (CorruptEvidenceError, ImporterCapabilityError) as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.FAILED, error=str(exc),
            )
        else:
            self._package = builder.build(package_id="")
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.INGESTED,
                asset_ids=[a.asset_id for a in report.imported],
                unhandled_paths=list(report.unhandled_paths),
            )

        self._sources[source_id] = record
        self._source_order.append(source_id)
        return record

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "session_id": self.session_id,
            "name": self.name,
            "seed": self.seed,
            "source_order": list(self._source_order),
            "sources": {sid: self._sources[sid].to_dict() for sid in self._source_order},
            "package": self._package.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "MultiSourceSession":
        if data.get("format_version") != MultiSourceSession.FORMAT_VERSION:
            raise ValueError(f"Unsupported MultiSourceSession format version: {data.get('format_version')}")
        session = MultiSourceSession(
            session_id=data["session_id"], name=data.get("name", ""), seed=data.get("seed"),
        )
        session._package = EvidencePackage.from_dict(data["package"])
        session._sources = {sid: SourceRecord.from_dict(v) for sid, v in data.get("sources", {}).items()}
        session._source_order = list(data.get("source_order", list(session._sources.keys())))
        return session
