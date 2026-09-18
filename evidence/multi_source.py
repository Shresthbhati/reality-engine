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
    "CaptureComponent",
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
    DATASET = "dataset"                        # a directory of mixed, non-composite evidence
    PHONE_CAPTURE = "phone_capture"             # composite: caller declared capture_type="phone"
    DRONE_CAPTURE = "drone_capture"             # composite: caller declared capture_type="drone"
    COMPOSITE_CAPTURE = "composite_capture"     # composite: capture_type not declared
    UNKNOWN = "unknown"    # unrecognized single-file extension


class CaptureComponent(str, Enum):
    """One synchronized component of a composite acquisition (spec §10:
    a phone/drone capture may bundle RGB, video, depth, IMU, GPS,
    calibration, and telemetry as related, not merged, evidence)."""

    RGB = "rgb"
    VIDEO = "video"
    DEPTH = "depth"
    IMU = "imu"
    GPS = "gps"
    CALIBRATION = "calibration"
    TELEMETRY = "telemetry"


#: Direct-child subdirectory names (case-insensitive) recognized as
#: composite-capture components. "images"/"photos" are accepted aliases
#: for RGB so a capture package can use whichever the source device
#: convention prefers.
_COMPONENT_DIR_NAMES: Dict[str, CaptureComponent] = {
    "rgb": CaptureComponent.RGB,
    "images": CaptureComponent.RGB,
    "photos": CaptureComponent.RGB,
    "video": CaptureComponent.VIDEO,
    "depth": CaptureComponent.DEPTH,
    "imu": CaptureComponent.IMU,
    "gps": CaptureComponent.GPS,
    "calibration": CaptureComponent.CALIBRATION,
    "telemetry": CaptureComponent.TELEMETRY,
}

#: Components this repo can decode into real evidence today (via the
#: existing photo/video importers). Everything else (DEPTH/IMU/GPS/
#: CALIBRATION/TELEMETRY) has no parser here yet, so it is recorded as a
#: file manifest only -- never fabricated as parsed sensor values.
_VISUAL_COMPONENTS = frozenset({CaptureComponent.RGB, CaptureComponent.VIDEO})
_SIDECAR_COMPONENTS = frozenset(
    {CaptureComponent.DEPTH, CaptureComponent.IMU, CaptureComponent.GPS,
     CaptureComponent.CALIBRATION, CaptureComponent.TELEMETRY}
)


def _detect_composite_components(path: str) -> Dict[CaptureComponent, List[str]]:
    """Scan `path`'s DIRECT child subdirectories for known composite-
    capture component names. Returns {component: [sorted relative
    paths]} for every matching, non-empty subdirectory; a subdirectory
    that exists but holds zero files is not reported (nothing to
    preserve). Paths are relative to `path`, POSIX-separated."""
    found: Dict[CaptureComponent, List[str]] = {}
    if not os.path.isdir(path):
        return found
    for entry in sorted(os.listdir(path)):
        component = _COMPONENT_DIR_NAMES.get(entry.lower())
        if component is None:
            continue
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        files: List[str] = []
        for root, _dirs, names in os.walk(subdir):
            for name in names:
                full = os.path.join(root, name)
                files.append(os.path.relpath(full, path).replace(os.sep, "/"))
        if files:
            files.sort()
            found[component] = files
    return found


def _is_composite_capture(components: Dict[CaptureComponent, List[str]]) -> bool:
    """A composite capture needs at least one visual component (rgb/
    video) PLUS at least one sidecar component (depth/imu/gps/
    calibration/telemetry). A folder with only rgb/video is an ordinary
    photo/video collection and must keep classifying as DATASET --
    composite detection must never change existing behavior for it."""
    has_visual = any(c in components for c in _VISUAL_COMPONENTS)
    has_sidecar = any(c in components for c in _SIDECAR_COMPONENTS)
    return has_visual and has_sidecar


def _source_type_of(path: str, capture_type: Optional[str] = None) -> SourceType:
    if os.path.isdir(path):
        components = _detect_composite_components(path)
        if _is_composite_capture(components):
            if capture_type == "phone":
                return SourceType.PHONE_CAPTURE
            if capture_type == "drone":
                return SourceType.DRONE_CAPTURE
            return SourceType.COMPOSITE_CAPTURE
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
    components: Dict[str, List[str]] = field(default_factory=dict)
    registration: dict = field(default_factory=lambda: {"status": "unknown", "transform": None})

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
            "components": {k: list(v) for k, v in self.components.items()},
            "registration": dict(self.registration),
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
            components={k: list(v) for k, v in data.get("components", {}).items()},
            registration=dict(data.get("registration", {"status": "unknown", "transform": None})),
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

    def sensor_streams(self, source_id: str, component: str):
        """Parse a source's recorded sidecar files for one component
        ("imu"/"gps"/"telemetry") into real evidence.sensors.SensorStream
        objects -- NOT just the file-manifest ``SourceRecord.components``
        already gives you. Parses on demand (not cached, not stored on
        the record/serialized) rather than eagerly during add_source(),
        so a directory holding non-JSONL sidecar files (a vendor's raw
        binary log, say) never fails an otherwise-successful import;
        the caller who actually wants the parsed samples asks for them
        and gets an honest SensorParseError only then.

        Returns [] if the source has no recorded files for that
        component (nothing to parse, not an error). Raises KeyError if
        source_id is unknown, ValueError if `component` has no
        registered parser (see evidence.sensors.PARSERS_BY_COMPONENT --
        "calibration" is not registered here; use calibrations()).
        """
        from evidence.sensors import parse_component_streams

        record = self._sources[source_id]
        relative_paths = record.components.get(component, [])
        if not relative_paths:
            return []
        absolute_paths = self._absolute_component_paths(record, relative_paths)
        return parse_component_streams(absolute_paths, component)

    def calibrations(self, source_id: str):
        """Parse a source's recorded "calibration" sidecar files into
        real evidence.sensors.CalibrationRecord objects. Separate from
        sensor_streams() because a calibration file parses to ONE
        record per file, not a SensorStream of time-ordered samples.
        Returns [] if the source has no recorded calibration files.
        """
        from evidence.sensors import parse_component_calibrations

        record = self._sources[source_id]
        relative_paths = record.components.get(CaptureComponent.CALIBRATION.value, [])
        if not relative_paths:
            return []
        absolute_paths = self._absolute_component_paths(record, relative_paths)
        return parse_component_calibrations(absolute_paths)

    def depth_maps(self, source_id: str):
        """Parse a source's recorded "depth" sidecar files into real
        perception.depth.interface.DepthMap objects (JSON descriptor +
        16-bit grayscale PNG -- see evidence.sensors' module-level
        docs for the exact format and what it does NOT support).
        Separate from sensor_streams()/calibrations() for the same
        reason calibration is: one DepthMap per file, not a sample
        stream. Returns [] if the source has no recorded depth files.
        """
        from evidence.sensors import parse_component_depth_maps

        record = self._sources[source_id]
        relative_paths = record.components.get(CaptureComponent.DEPTH.value, [])
        if not relative_paths:
            return []
        absolute_paths = self._absolute_component_paths(record, relative_paths)
        return parse_component_depth_maps(absolute_paths)

    @staticmethod
    def _absolute_component_paths(record: "SourceRecord", relative_paths: List[str]) -> List[str]:
        return [
            os.path.join(record.original_path, rel.replace("/", os.sep))
            for rel in relative_paths
        ]

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
        # Detected once here and threaded into every SourceRecord branch
        # below -- previously _source_type_of() computed this same dict
        # internally (to decide DATASET vs *_CAPTURE) and discarded it,
        # so SourceRecord.components stayed {} forever despite the field
        # existing on the dataclass: sidecar files (imu/gps/depth/...)
        # were detected for classification purposes but never actually
        # recorded anywhere a caller could resolve them from, i.e.
        # exactly "recognizing imu/ is not the same as ingesting it".
        detected_components = (
            {c.value: list(p) for c, p in _detect_composite_components(path).items()}
            if os.path.isdir(path) else {}
        )
        # Identity: the default EvidenceSource uses the SAME source_id as
        # the SourceRecord it belongs to, not a separate basename-derived
        # id -- every EvidenceAsset produced from this add_source() call
        # embeds this EvidenceSource, so its source.source_id must be
        # joinable back to SourceRecord.source_id without a second lookup
        # table. A basename-derived id ("disk:<basename>") was both
        # disconnected from the record's own id AND collision-prone
        # (two different folders both named "photos" would produce the
        # same id); source_id is unique per add_source call by
        # construction (index + content hash) and IS the record's own
        # identity. Callers who pass an explicit `source` keep full
        # control of its id -- this default only applies when none was
        # given.
        use_source = source or EvidenceSource(
            source_id=source_id,
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
                components=detected_components,
            )
        except (CorruptEvidenceError, ImporterCapabilityError) as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.FAILED, error=str(exc),
                components=detected_components,
            )
        else:
            self._package = builder.build(package_id="")
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.INGESTED,
                asset_ids=[a.asset_id for a in report.imported],
                unhandled_paths=list(report.unhandled_paths),
                components=detected_components,
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
