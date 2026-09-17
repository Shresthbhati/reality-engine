"""Multi-source session store + incremental source ingestion.

This is the milestone's integration layer. It binds together, without
replacing any of them:

    evidence/session.py   Session (append-only EvidenceItems)
    evidence/packages.py  EvidencePackage + DeterministicPackageBuilder
    evidence/importers.py the real disk-to-evidence path (photo/video/LAS,
                          frame extraction with per-frame provenance)
    evidence/sources.py   the canonical Source model + registry

WHAT IT ADDS (and what did not exist before):

  - A persistent, on-disk session workspace:
        <root>/sessions/<session_id>/
            session.json    the append-only Session
            sources.json    the session's SourceRegistry
            package.json    the session's EvidencePackage (all sources)
            payloads/       content-addressed raw source bytes
  - Canonical Source records for everything the user supplies, with
    deterministic content-derived ids, explicit lifecycle status, and
    the evidence-asset ids derived from each source (provenance chain:
    asset -> source -> original bytes).
  - Incremental ingestion: add a source now, ingest it now or later,
    add more sources later. Previously ingested sources are never
    re-imported; re-adding the same bytes resolves to the same source
    (content hash identity) and returns "already ingested".
  - Composite capture packages: a directory with a manifest.json
    describing components (images / video / imu / gps / calibration /
    depth) is ONE source with typed components, ingested as a unit.
  - Sensor log ingestion (CSV): IMU (accel/gyro columns) and GNSS
    (lat/lon columns) become real EvidenceAssets with measured
    completeness metadata -- honest counts, never invented samples.
  - PLY/PCD point cloud header parsing (point count, data format)
    alongside the existing LAS/LASZ path.

STORAGE RULES:

  - Raw source bytes are stored content-addressed (sha256) under
    payloads/ -- git-object-store sharding, digests validated as
    64-hex before ever touching a path (the ArtifactStore security
    rule; traversal can never occur because only self-computed
    digests form paths).
  - Derived frame bytes are NOT duplicated into payloads/ -- they are
    re-derivable from the stored video via the recorded sampling
    strategy, and the package records their hashes. (Recorded
    limitation: re-deriving requires OpenCV.)
  - No wall clocks anywhere: timestamps come from media metadata or
    are None. Serialization format_version: 1 everywhere.

LLM-free; LLMs are never part of the runtime (standing repo rule).
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evidence.frames import IFrameSelectionStrategy, UniformTimeSamplingStrategy
from evidence.importers import (
    ImporterCapabilityError,
    import_file,
    import_folder,
)
from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidencePackage,
    EvidenceSource,
)
from evidence.session import EvidenceKind, Session
from evidence.sources import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    Source,
    SourceComponent,
    SourceRegistry,
    SourceStatus,
    SourceType,
    UnknownSourceError,
    UnsupportedSourceError,
    classify_source,
    hash_file,
    hash_tree,
    source_id_for_digest,
)

__all__ = [
    "SessionWorkspace",
    "SessionHandle",
    "AddSourceResult",
    "IngestResult",
    "SourcePayloadStore",
    "DuplicateSessionError",
    "UnknownSessionError",
    "parse_sensor_csv",
    "parse_ply_header",
    "parse_pcd_header",
    "COMPOSITE_MANIFEST_NAME",
]



COMPOSITE_MANIFEST_NAME = "manifest.json"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: manifest component keys -> source-component roles. Anything else in
#: a manifest is recorded as a warning, not silently dropped.
_MANIFEST_ROLES = {
    "images": "image",
    "image": "image",
    "video": "video",
    "videos": "video",
    "imu": "imu",
    "gps": "gps",
    "gnss": "gps",
    "calibration": "calibration",
    "depth": "depth",
    "point_cloud": "point_cloud",
    "lidar": "point_cloud",
    "telemetry": "telemetry",
}



# ---------------------------------------------------------------------+
# Format probing: sensor CSVs and point-cloud headers (stdlib only)     |
# ---------------------------------------------------------------------+

_TIMESTAMP_COLS = ("timestamp", "time", "ts", "t", "t_us", "time_us")
_ACCEL_COLS = ("ax", "ay", "az")
_GYRO_COLS = ("gx", "gy", "gz")
_MAG_COLS = ("mx", "my", "mz")


def parse_sensor_csv(path: str) -> Tuple[EvidenceKind, Dict[str, object]]:
    """Parse a sensor log CSV into (evidence kind, measured summary).

    Column detection is by conventional names (case-insensitive). The
    summary contains REAL measured counts and completeness -- a row
    whose numeric fields fail to parse counts against completeness, it
    is never silently dropped or fabricated. Raises ValueError when
    nothing recognizable is present (callers record that honestly)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_fields = [(c or "").strip().lower() for c in (reader.fieldnames or [])]
        if not raw_fields:
            raise ValueError(f"{path}: sensor CSV has no header row")
        rows = list(reader)

    def _has(prefixes: Tuple[str, ...]) -> bool:
        return any(c in raw_fields for c in prefixes)

    def _first(colnames: Tuple[str, ...]) -> Optional[str]:
        for c in colnames:
            if c in raw_fields:
                return c
        return None

    has_accel = _has(_ACCEL_COLS)
    has_gyro = _has(_GYRO_COLS)
    has_mag = _has(_MAG_COLS)
    has_latlon = _has(("lat", "latitude")) and _has(("lon", "lng", "longitude"))
    ts_col = _first(_TIMESTAMP_COLS)

    if has_accel or has_gyro:
        kind = EvidenceKind.IMU
    elif has_latlon:
        kind = EvidenceKind.GPS_TRACK
    else:
        raise ValueError(
            f"{path}: no recognizable sensor columns (need accel/gyro "
            f"ax..az/gx..gz or lat/lon); found {raw_fields}"
        )

    numeric_cols = [
        c for c in raw_fields
        if c in _ACCEL_COLS + _GYRO_COLS + _MAG_COLS
        + ("lat", "latitude", "lon", "lng", "longitude", "alt", "altitude")
    ]
    good = 0
    first_ts = None
    last_ts = None
    for row in rows:
        try:
            ok = all(
                float(row[c]) == float(row[c])
                for c in numeric_cols if row.get(c) not in (None, "")
            )
            if ts_col is not None and row.get(ts_col):
                ts = float(row[ts_col])
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        except (TypeError, ValueError):
            ok = False
        if ok and numeric_cols:
            good += 1

    summary: Dict[str, object] = {
        "format": "csv",
        "row_count": len(rows),
        "columns": raw_fields,
        "has_timestamps": ts_col is not None,
        "has_accel": has_accel,
        "has_gyro": has_gyro,
        "has_mag": has_mag,
        "has_latlon": has_latlon,
        "numeric_complete_rows": good,
        "completeness": round(good / len(rows), 6) if rows else 0.0,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }
    if ts_col is None:
        summary["warning"] = "no timestamp column; temporal alignment with imagery is UNKNOWN"
    return kind, summary


def parse_ply_header(path: str) -> Dict[str, object]:
    """Parse a PLY header: data format + element counts. Header-only --
    the payload bytes are stored untouched; big geometry is never
    converted into a fake in-memory representation."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if not head.startswith(b"ply"):
        raise CorruptEvidenceError(f"{path}: not a PLY file (missing 'ply' magic)")
    end = head.find(b"end_header")
    header_text = head[: end if end != -1 else len(head)].decode("ascii", errors="replace")
    fmt = None
    vertex_count = None
    elements: Dict[str, int] = {}
    for line in header_text.splitlines():
        parts = line.strip().split()
        if parts[:1] == ["format"] and len(parts) >= 3:
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"] and len(parts) >= 3:
            vertex_count = int(parts[2])
        elif parts[:1] == ["element"] and len(parts) >= 3:
            elements[parts[1]] = int(parts[2])
    if fmt is None:
        raise CorruptEvidenceError(f"{path}: PLY header declares no data format")
    if fmt == "ascii" and end == -1:
        raise CorruptEvidenceError(f"{path}: PLY ascii file has no end_header")
    meta: Dict[str, object] = {
        "format": "ply",
        "ply_format": fmt,
        "elements": {k: int(v) for k, v in sorted(elements.items())},
    }
    if vertex_count is not None:
        meta["point_count"] = vertex_count
    return meta


def parse_pcd_header(path: str) -> Dict[str, object]:
    """Parse a PCD header (# .PCD text header, per the PCD spec)."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if not head.startswith(b"# .PCD"):
        raise CorruptEvidenceError(f"{path}: not a PCD file (missing '# .PCD' magic)")
    text = head.decode("ascii", errors="replace")
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            fields[parts[0].upper()] = " ".join(parts[1:])
    meta: Dict[str, object] = {"format": "pcd"}
    if "FIELDS" in fields:
        meta["fields"] = fields["FIELDS"].split()
    if "POINTS" in fields:
        try:
            meta["point_count"] = int(fields["POINTS"])
        except ValueError:
            pass
    if "DATA" in fields:
        meta["data_mode"] = fields["DATA"].lower()
    return meta


# ---------------------------------------------------------------------+
# Content-addressed raw payload store                                    |
# ---------------------------------------------------------------------+

class SourcePayloadStore:
    """Content-addressed store for RAW SOURCE bytes (photos, videos,
    point clouds as captured). Git-object-store sharding: payloads/ab/
    <sha256>.bin. Only self-computed 64-hex digests ever form paths --
    a digest that is not exactly 64 lowercase hex characters is
    refused before any filesystem join (the artifact-store security
    rule, applied here from day one)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    @staticmethod
    def _validate_digest(digest: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
            raise ValueError(f"invalid payload digest: {digest!r}")
        return digest

    def has(self, digest: str) -> bool:
        return os.path.isfile(self._path_for(digest))

    def path_for(self, digest: str) -> str:
        return self._path_for(digest)

    def _path_for(self, digest: str) -> str:
        digest = self._validate_digest(digest)
        return os.path.join(self.root, digest[:2], f"{digest}.bin")

    def put(self, data: bytes, digest: str) -> str:
        """Store bytes under their (already computed) digest. Returns
        the stored path. Idempotent."""
        path = self._path_for(digest)
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return path

    def get(self, digest: str) -> bytes:
        with open(self._path_for(digest), "rb") as f:
            return f.read()


# ---------------------------------------------------------------------+
# Result records                                                         |
# ---------------------------------------------------------------------+

@dataclass
class AddSourceResult:
    """Outcome of `add source`. `created=False` means the exact same
    bytes were already registered in this session -- the existing
    source is returned and nothing is duplicated."""
    source_id: str
    created: bool
    source_type: SourceType
    sha256: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "created": self.created,
            "source_type": self.source_type.value,
            "sha256": self.sha256,
            "note": self.note,
        }


@dataclass
class IngestResult:
    """Outcome of ingesting one source into the session's evidence.
    status is an explicit lifecycle state -- FAILED/PARTIAL are honest
    outcomes, never disguised as INGESTED."""
    source_id: str
    status: SourceStatus
    asset_ids: List[str] = field(default_factory=list)
    duplicate_asset_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    already_ingested: bool = False

    @property
    def ok(self) -> bool:
        return self.status in (SourceStatus.INGESTED, SourceStatus.PARTIAL)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "status": self.status.value,
            "asset_ids": list(self.asset_ids),
            "duplicate_asset_ids": list(self.duplicate_asset_ids),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "already_ingested": self.already_ingested,
        }

# ---------------------------------------------------------------------+
# SessionHandle: one session's sources + evidence, persisted on disk     |
# ---------------------------------------------------------------------+

class SessionHandle:
    """A loaded session: its append-only Session, its SourceRegistry,
    and its single accumulating EvidencePackage. All persistence is
    explicit (`save()`); every ingested source appends new assets
    incrementally and never re-imports previous sources."""

    def __init__(self, workspace: "SessionWorkspace", directory: str):
        self._workspace = workspace
        self.dir = directory
        self.session = Session(session_id=os.path.basename(directory))
        self.sources = SourceRegistry(session_id=self.session.id)
        self.package = EvidencePackage(package_id=f"pkg-{self.session.id}", seed=self.session.id)
        self._payloads = SourcePayloadStore(os.path.join(directory, "payloads"))
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        session_path = os.path.join(self.dir, "session.json")
        if os.path.isfile(session_path):
            self.session.deserialize(json.loads(_read_text(session_path)))
        sources_path = os.path.join(self.dir, "sources.json")
        if os.path.isfile(sources_path):
            self.sources.deserialize(json.loads(_read_text(sources_path)))
        package_path = os.path.join(self.dir, "package.json")
        if os.path.isfile(package_path):
            self.package = EvidencePackage.from_dict(json.loads(_read_text(package_path)))

    def save(self) -> None:
        _write_text(os.path.join(self.dir, "session.json"), json.dumps(self.session.to_dict(), indent=2, sort_keys=True))
        _write_text(os.path.join(self.dir, "sources.json"), json.dumps(self.sources.to_dict(), indent=2, sort_keys=True))
        _write_text(os.path.join(self.dir, "package.json"), json.dumps(self.package.to_dict(), indent=2, sort_keys=True))

    # ---- add source ----

    def add_source(
        self,
        path: str,
        *,
        source_type: Optional[SourceType] = None,
        platform: str = "",
        device: str = "",
        imported_at: Optional[float] = None,
    ) -> AddSourceResult:
        """Register a source with this session. Hashes content,
        resolves duplicates by content hash (re-adding the same bytes
        returns created=False and touches nothing), stores the raw
        bytes content-addressed, and refines composite captures
        (directories with a manifest.json) into typed components."""
        if not os.path.exists(path):
            raise ValueError(f"source path does not exist: {path!r}")
        if os.path.isdir(path):
            digest = hash_tree(path)
            size = _tree_size(path)
        else:
            digest = hash_file(path)
            size = os.path.getsize(path)
        source, created = self.sources.register(
            path,
            source_type=source_type or classify_source(path),
            sha256=digest,
            size_bytes=size,
            platform=platform,
            device=device,
            imported_at=imported_at,
        )
        if not created:
            return AddSourceResult(
                source_id=source.source_id,
                created=False,
                source_type=source.source_type,
                sha256=digest,
                note=f"already registered in session '{self.session.id}' "
                     f"(content hash {digest[:16]}); source identity preserved, nothing duplicated",
            )
        # Refine: composite capture directories carry a manifest.json.
        if source.source_type is SourceType.DATASET and os.path.isdir(path):
            if os.path.isfile(os.path.join(path, COMPOSITE_MANIFEST_NAME)):
                source = self._attach_composite_manifest(source, path)
        # Persist raw bytes content-addressed (file) or per component (dir).
        try:
            if os.path.isfile(path):
                self._payloads.put(_read_bytes(path), digest)
                source = self.sources.update(source.source_id, metadata={**source.metadata, "stored_digest": digest})
            else:
                stored = dict(source.metadata)
                for component in source.components:
                    component_path = component.path
                    if os.path.isfile(component_path):
                        stored[f"stored_digest:{component.sha256}"] = component.sha256
                        self._payloads.put(_read_bytes(component_path), component.sha256)
                source = self.sources.update(source.source_id, metadata=stored)
        except OSError as exc:
            source = self.sources.update(
                source.source_id, warnings=source.warnings + (f"payload storage failed: {exc}",)
            )
        self.save()
        return AddSourceResult(source_id=source.source_id, created=True, source_type=source.source_type, sha256=digest)

    # ---- ingest ----

    def ingest_source(
        self,
        source_id: str,
        *,
        frame_strategy: Optional[IFrameSelectionStrategy] = None,
        frame_min_bytes: int = 2048,
    ) -> IngestResult:
        """Ingest one registered source into the session's evidence.

        Idempotent: a source already INGESTED is not re-imported
        (result carries already_ingested=True). Re-ingesting a
        FAILED/PARTIAL source retries it. Every derived asset keeps
        provenance back to this source (EvidenceSource.source_id ==
        the source's id; video frames additionally carry the video
        asset id and frame timestamp). Failures set an explicit
        FAILED/PARTIAL status with structured errors -- never success."""
        source = self.sources.get(source_id)
        if source.status is SourceStatus.INGESTED:
            return IngestResult(
                source_id=source_id,
                status=SourceStatus.INGESTED,
                asset_ids=list(source.asset_ids),
                already_ingested=True,
            )
        self.sources.update(source_id, status=SourceStatus.INGESTING)
        try:
            result = self._dispatch_ingest(source, frame_strategy, frame_min_bytes)
        except (CorruptEvidenceError, ImporterCapabilityError, UnsupportedSourceError, ValueError, OSError) as exc:
            self.sources.update(source_id, status=SourceStatus.FAILED, errors=(str(exc),))
            self.save()
            return IngestResult(source_id=source_id, status=SourceStatus.FAILED, errors=[str(exc)])
        except Exception as exc:  # decoder crashes (e.g. cv2 on a corrupt container) are honest failures too
            message = f"{type(exc).__name__}: {exc}"
            self.sources.update(source_id, status=SourceStatus.FAILED, errors=(message,))
            self.save()
            return IngestResult(source_id=source_id, status=SourceStatus.FAILED, errors=[message])
        asset_ids = result["asset_ids"]
        duplicates = result["duplicate_asset_ids"]
        warnings = list(source.warnings) + list(result["warnings"])
        errors = list(result["errors"])
        # Sync media-declared metadata (EXIF etc.) back onto the Source:
        # the asset layers decode it during ingestion; the source record
        # keeps the acquisition/device identity once it is actually known.
        sync_changes: Dict[str, object] = {}
        extra_metadata: Dict[str, object] = {}
        for asset_id in asset_ids:
            asset = self.package.assets.get(asset_id)
            if asset is None:
                continue
            if source.acquired_at is None and "acquired_at" not in sync_changes and asset.acquired_at is not None:
                sync_changes["acquired_at"] = asset.acquired_at
            make = asset.sensor_metadata.get("make")
            model = asset.sensor_metadata.get("model")
            if not source.device and make:
                sync_changes["device"] = " ".join(str(x) for x in (make, model) if x)
            for key in ("latitude_deg", "longitude_deg", "altitude_m"):
                if key in asset.sensor_metadata:
                    extra_metadata[key] = asset.sensor_metadata[key]
        if extra_metadata:
            sync_changes["metadata"] = {**source.metadata, **extra_metadata}
        if sync_changes:
            source = self.sources.update(source_id, **sync_changes)
        if errors and asset_ids:
            status = SourceStatus.PARTIAL
        elif errors:
            status = SourceStatus.FAILED
        else:
            status = SourceStatus.INGESTED
        existing_assets = set(source.asset_ids)
        all_ids = list(existing_assets) + [a for a in asset_ids if a not in existing_assets]
        self.sources.update(
            source_id,
            status=status,
            asset_ids=tuple(all_ids),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )
        self._append_session_evidence(all_ids)
        self.session.record_processing(
            "ingest_source",
            detail={"source_id": source_id, "status": status.value, "asset_count": len(all_ids)},
        )
        self.save()
        return IngestResult(
            source_id=source_id,
            status=status,
            asset_ids=all_ids,
            duplicate_asset_ids=duplicates,
            warnings=warnings,
            errors=errors,
        )

    def _dispatch_ingest(
        self,
        source: Source,
        frame_strategy: Optional[IFrameSelectionStrategy],
        frame_min_bytes: int,
    ) -> Dict[str, List[str]]:
        """Route one source to the right ingestion path, returning
        {"asset_ids", "duplicate_asset_ids", "warnings", "errors"}."""
        warnings: List[str] = []
        errors: List[str] = []
        builder = DeterministicPackageBuilder(seed=f"{self.session.id}:{source.source_id}")
        evidence_source = EvidenceSource(
            source_id=source.source_id,
            platform=source.platform or "filesystem",
            device=source.device or "unknown device",
        )
        strategy = frame_strategy or UniformTimeSamplingStrategy(target_count=8)

        if source.components:
            return self._ingest_composite(builder, evidence_source, source, strategy, frame_min_bytes)

        path = source.path
        if os.path.isdir(path):
            report = import_folder(
                builder, path, source=evidence_source,
                frame_strategy=strategy, frame_min_bytes=frame_min_bytes,
            )
            for unhandled in report.unhandled_paths:
                warnings.append(f"unhandled path skipped: {unhandled}")
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                self._stage_sensor_log(builder, evidence_source, path, warnings)
            elif ext == ".ply":
                self._stage_pointcloud(builder, evidence_source, path, parse_ply_header)
            elif ext == ".pcd":
                self._stage_pointcloud(builder, evidence_source, path, parse_pcd_header)
            else:
                import_file(
                    builder, path, source=evidence_source,
                    frame_strategy=strategy, frame_min_bytes=frame_min_bytes,
                )

        fragment = builder.build()
        merge = self.package.merge_from(fragment)
        added = merge["added"]
        # Honest video validation: a video source that yields no
        # decodable frames is a FAILED ingest, not a silently
        # "successful" video-container-only import.
        if source.source_type is SourceType.VIDEO and added:
            fragment_kinds = {fragment.assets[a].kind for a in added}
            if EvidenceKind.PHOTO not in fragment_kinds:
                errors.append(
                    "no frames decodable from video container -- the bytes "
                    "are stored, but no reconstruction-usable imagery exists"
                )
        return {
            "asset_ids": added,
            "duplicate_asset_ids": merge["duplicates"],
            "warnings": warnings,
            "errors": errors,
        }

    def _stage_sensor_log(
        self,
        builder: DeterministicPackageBuilder,
        evidence_source: EvidenceSource,
        path: str,
        warnings: List[str],
    ) -> str:
        kind, summary = parse_sensor_csv(path)
        if summary.get("warning"):
            warnings.append(str(summary["warning"]))
        payload = _read_bytes(path)
        return builder.add_payload(
            payload,
            kind=kind,
            source=evidence_source,
            source_uri=path,
            acquired_at=None,  # log-internal timestamps are metadata, not acquisition time
            sensor_metadata=summary,
            quality={"measured": 1.0, "completeness": float(summary["completeness"])},
        )

    def _stage_pointcloud(self, builder, evidence_source, path, header_parser) -> str:
        meta = header_parser(path)
        payload = _read_bytes(path)
        return builder.add_payload(
            payload,
            kind=EvidenceKind.POINT_CLOUD,
            source=evidence_source,
            source_uri=path,
            sensor_metadata=meta,
            quality={"measured": 0.0},  # header-only validation; geometric quality is a later stage's job
        )

    def _stage_calibration(
        self,
        builder: DeterministicPackageBuilder,
        evidence_source: EvidenceSource,
        path: str,
        warnings: List[str],
    ) -> str:
        text = _read_text(path)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CorruptEvidenceError(f"{path}: calibration file is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise CorruptEvidenceError(f"{path}: calibration file must be a JSON object")
        payload = _read_bytes(path)
        return builder.add_payload(
            payload,
            kind=EvidenceKind.OTHER,
            source=evidence_source,
            source_uri=path,
            sensor_metadata={"calibration": data},
            quality={"measured": 0.0},
        )

    def _ingest_composite(
        self,
        builder: DeterministicPackageBuilder,
        evidence_source: EvidenceSource,
        source: Source,
        strategy: IFrameSelectionStrategy,
        frame_min_bytes: int,
    ) -> Dict[str, List[str]]:
        """Ingest a composite capture component by component, in
        deterministic role order. Component failures are recorded and
        the remaining components still ingest (PARTIAL status, never
        silent loss)."""
        warnings: List[str] = []
        errors: List[str] = []
        for component in sorted(source.components, key=lambda c: (c.role, c.path)):
            path = component.path
            try:
                if not os.path.isfile(path):
                    errors.append(f"component '{component.role}' missing on disk: {path}")
                    continue
                ext = os.path.splitext(path)[1].lower()
                if component.role == "image":
                    import_file(builder, path, source=evidence_source, frame_min_bytes=frame_min_bytes)
                elif component.role == "video":
                    import_file(builder, path, source=evidence_source, frame_strategy=strategy, frame_min_bytes=frame_min_bytes)
                elif component.role == "point_cloud":
                    if ext == ".ply":
                        self._stage_pointcloud(builder, evidence_source, path, parse_ply_header)
                    elif ext == ".pcd":
                        self._stage_pointcloud(builder, evidence_source, path, parse_pcd_header)
                    else:
                        import_file(builder, path, source=evidence_source, frame_min_bytes=frame_min_bytes)
                elif component.role in ("imu", "gps"):
                    self._stage_sensor_log(builder, evidence_source, path, warnings)
                elif component.role == "calibration":
                    self._stage_calibration(builder, evidence_source, path, warnings)
                else:
                    errors.append(f"component '{component.role}' has no ingestion path yet: {path}")
            except (CorruptEvidenceError, ImporterCapabilityError, UnsupportedSourceError, OSError, ValueError) as exc:
                errors.append(f"component '{component.role}' ({path}): {exc}")
        fragment = builder.build()
        merge = self.package.merge_from(fragment)
        return {
            "asset_ids": merge["added"],
            "duplicate_asset_ids": merge["duplicates"],
            "warnings": warnings,
            "errors": errors,
        }

    def _append_session_evidence(self, asset_ids: List[str]) -> None:
        """Mirror newly added package assets into the append-only
        Session as EvidenceItems (the shape reconstruction consumes).
        Already-present ids are skipped (idempotent re-ingest)."""
        for asset_id in asset_ids:
            asset = self.package.assets.get(asset_id)
            if asset is None or asset_id in {
                item.id for item in self.session.all_evidence()
            }:
                continue
            self.session.add_evidence(asset.to_evidence_item())

    def _attach_composite_manifest(self, source: Source, root: str) -> Source:
        """Read a composite capture's manifest.json and turn the source
        into a COMPOSITE_CAPTURE with typed components. Manifest format:
            {"format_version": 1,
             "platform": "drone", "device": "...",
             "components": {"images": "images/", "video": "video/f.mp4",
                            "imu": "imu/imu.csv", "gps": "gps/gps.csv",
                            "calibration": "calibration/cam.json"}}
        Unknown keys are warnings; missing files are recorded on the
        components (ingest reports them as errors -- explicit, not silent)."""
        manifest_path = os.path.join(root, COMPOSITE_MANIFEST_NAME)
        try:
            manifest = json.loads(_read_text(manifest_path))
        except json.JSONDecodeError as exc:
            return self.sources.update(
                source.source_id, warnings=source.warnings + (f"manifest.json unparseable: {exc}",)
            )
        if not isinstance(manifest, dict):
            return self.sources.update(
                source.source_id, warnings=source.warnings + ("manifest.json is not an object",)
            )
        components: List[SourceComponent] = []
        warnings = list(source.warnings)
        declared = manifest.get("components", {})
        if not isinstance(declared, dict):
            declared = {}
        for key in sorted(declared):
            role = _MANIFEST_ROLES.get(key.lower())
            if role is None:
                warnings.append(f"manifest declares unknown component type {key!r}")
                continue
            value = declared[key]
            paths: List[str] = []
            if isinstance(value, str):
                paths = [value]
            elif isinstance(value, list):
                paths = [str(v) for v in value]
            target_dir = None
            if isinstance(value, str) and os.path.isdir(os.path.join(root, value)):
                target_dir = os.path.join(root, value)
            if target_dir is not None:
                paths = []
                for base, _dirs, files in os.walk(target_dir):
                    for name in files:
                        paths.append(os.path.relpath(os.path.join(base, name), root).replace("\\", "/"))
                paths = sorted(set(paths))
            for rel in paths:
                full = os.path.join(root, rel)
                if not os.path.isfile(full):
                    components.append(SourceComponent(role=role, path=full, sha256="", metadata={"missing": True}))
                    continue
                components.append(SourceComponent(
                    role=role,
                    path=full,
                    sha256=hash_file(full),
                    size_bytes=os.path.getsize(full),
                ))
        source_type = manifest.get("platform") or source.platform
        stype = SourceType.COMPOSITE_CAPTURE
        if (source_type or "").lower() in ("phone", "phone_capture", "smartphone"):
            stype = SourceType.PHONE_CAPTURE
        elif (source_type or "").lower() in ("drone", "uav", "drone_capture"):
            stype = SourceType.DRONE_CAPTURE
        return self.sources.update(
            source.source_id,
            source_type=stype,
            platform=source_type or source.platform,
            device=manifest.get("device", source.device),
            components=tuple(components),
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------+
# SessionWorkspace: the on-disk multi-source session root                |
# ---------------------------------------------------------------------+

class SessionWorkspace:
    """A workspace directory holding any number of multi-source
    sessions:

        <root>/sessions/<session_id>/{session,sources,package}.json + payloads/

    This is the backend the CLI (`reality session ...`, `reality
    source ...`) and the Studio viewer drive. Everything persists to
    plain, inspectable JSON; raw source bytes persist content-addressed
    under payloads/."""

    SESSIONS_DIR = "sessions"

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, self.SESSIONS_DIR), exist_ok=True)

    def _session_dir(self, session_id: str) -> str:
        if not _SESSION_ID_RE.match(session_id or ""):
            raise ValueError(
                f"invalid session id {session_id!r}: must match "
                f"[A-Za-z0-9][A-Za-z0-9_.-]* (filesystem-safe, no traversal)"
            )
        return os.path.join(self.root, self.SESSIONS_DIR, session_id)

    def create_session(self, name: str, session_id: str = "", coordinate_frame=None) -> SessionHandle:
        """Create a new session. `name` is human-facing; the id defaults
        to a filesystem-safe slug of it. Duplicate ids are refused (the
        session is the unit of accumulation -- silently reopening one
        would mix two captures' evidence)."""
        sid = _slugify(session_id or name) if not session_id else session_id
        if not _SESSION_ID_RE.match(sid):
            raise ValueError(
                f"invalid session id {sid!r}: must match "
                f"[A-Za-z0-9][A-Za-z0-9_.-]* (filesystem-safe, no traversal)"
            )
        sdir = self._session_dir(sid)
        if os.path.exists(sdir):
            raise DuplicateSessionError(
                f"session '{sid}' already exists in workspace '{self.root}' -- "
                f"open it instead of creating it again"
            )
        os.makedirs(sdir)
        handle = SessionHandle(self, sdir)
        handle.session.name = name
        if coordinate_frame is not None:
            handle.session.coordinate_frame = coordinate_frame
        handle.save()
        return handle

    def open_session(self, session_id: str) -> SessionHandle:
        sdir = self._session_dir(session_id)
        if not os.path.isdir(sdir):
            raise UnknownSessionError(f"no session '{session_id}' in workspace '{self.root}'")
        return SessionHandle(self, sdir)

    def list_sessions(self) -> List[Dict[str, object]]:
        sessions_root = os.path.join(self.root, self.SESSIONS_DIR)
        out: List[Dict[str, object]] = []
        for name in sorted(os.listdir(sessions_root)):
            sdir = os.path.join(sessions_root, name)
            if not os.path.isdir(sdir):
                continue
            entry: Dict[str, object] = {"id": name}
            session_path = os.path.join(sdir, "session.json")
            if os.path.isfile(session_path):
                try:
                    data = json.loads(_read_text(session_path))
                    entry["name"] = data.get("name", "")
                    entry["status"] = data.get("status", "")
                    entry["evidence_count"] = len(data.get("evidence", {}))
                except (json.JSONDecodeError, OSError) as exc:
                    entry["error"] = str(exc)
            sources_path = os.path.join(sdir, "sources.json")
            if os.path.isfile(sources_path):
                try:
                    data = json.loads(_read_text(sources_path))
                    entry["source_count"] = len(data.get("sources", {}))
                except (json.JSONDecodeError, OSError):
                    pass
            out.append(entry)
        return out


# ---------------------------------------------------------------------+
# Module helpers                                                         |
# ---------------------------------------------------------------------+

def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-.")
    if not slug or not _SESSION_ID_RE.match(slug):
        raise ValueError(f"cannot derive a valid session id from {text!r}")
    return slug


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _tree_size(root: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


class DuplicateSessionError(ValueError):
    pass


class UnknownSessionError(ValueError):
    pass
