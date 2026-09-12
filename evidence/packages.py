"""Evidence packages (spec sec 6 EVIDENCE INGESTION -- the structured
layer above the append-only Session).

evidence/session.py answers "what was captured" (immutable items, one
session). This module answers "what does the engine hold, where did it
come from, what happened to it, and can I trust it": a versioned,
hash-addressed EvidencePackage of EvidenceAssets (each with sensor /
acquisition / quality metadata, provenance, and its own processing
history), EvidenceReferences that bind assets to WorldIR Observations,
ObservationSets that group reference sets into fused quantities, and a
DeterministicPackageBuilder that hashes/validates payloads and detects
duplicates before anything enters the store.

Design invariants (each traceable to a spec rule or a repo convention):

  - Deterministic IDs. Asset ids are content-derived (sha256 of the
    payload bytes) and sequence-stamped, so the same bytes through the
    same builder produce the same id -- and a re-import of the same
    payload into a rebuilt-from-scratch package produces the same id.
    Package id is derived from the sorted asset ids (rebuild-stable).
    No uuid4 anywhere in this module; no clocks (timestamps are
    caller-supplied acquisition stamps or None, never "now").
  - Raw evidence is immutable. Assets are frozen dataclasses; the
    package stores assets in a private dict; there is no remove_asset.
    Processing history lives on the asset as a list of ProcessingRecord
    (reusing evidence/session.py's type -- one vocabulary, not two).
  - Corruption handling is real, not a boolean: payloads are validated
    for minimum size and magic-byte container signatures where the
    format has them (JPEG/PNG/EXIV/TS/LAS/laz/E57/PLY/PCD/ZIP). Corrupt
    payloads raise CorruptEvidenceError at build time -- evidence that
    cannot be read honestly cannot be ingested honestly.
  - Duplicate detection is by content hash: the same bytes with a
    different filename is the same evidence; DuplicateEvidenceError
    (session.py's type, same meaning) is raised with the existing
    asset's id so the caller can de-duplicate by reference.
  - Provenance vocabulary is the existing provenance.Provenance enum;
    default OBSERVED for captured bytes. is_canonical() semantics are
    preserved (GENERATED/UNKNOWN/CONFLICT assets are non-canonical).
  - Everything round-trips through to_dict()/from_dict() deterministically
    (sorted keys, fixed order, format_version: 1, ValueError on
    mismatch -- the established repo serialization shape).

LLM-free, stdlib-only, no clocks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from provenance import Provenance, Uncertainty
from world_ir.coordinates import Frame

from evidence.session import (
    DuplicateEvidenceError,
    EvidenceItem,
    EvidenceKind,
    ProcessingRecord,
)

__all__ = [
    "CorruptEvidenceError",
    "EvidenceAsset",
    "EvidencePackage",
    "EvidenceReference",
    "EvidenceSource",
    "ObservationSet",
    "DeterministicPackageBuilder",
]



class CorruptEvidenceError(ValueError):
    """Payload failed structural validation (size/magic-byte check)."""


class UnknownAssetError(ValueError):
    """Processing history was requested for an asset the package doesn't hold."""


# ----------------------------------------------------------------------
# Magic-byte signatures: the honest minimum of file validation without
# pulling in decoders. A JPEG that doesn't start with FF D8 FF is not a
# JPEG; refusing it here is cheaper than a confusing failure three
# subsystems downstream.
# ----------------------------------------------------------------------

_MAGIC_CHECKS: Tuple[Tuple[str, bytes], ...] = (
    (".jpg", b"\xff\xd8\xff"),
    (".jpeg", b"\xff\xd8\xff"),
    (".png", b"\x89PNG\r\n\x1a\n"),
    (".tif", b"II*\x00"),
    (".tiff", b"MM\x00*"),
    (".exiv", b"Exiv2"),
    (".exif", b"II*\x00"),
    (".ts", b"\x47"),
    (".mj2", b"\x00\x00\x00\x18ftypmj2"),
    (".las", b"LASF"),
    (".laz", b"LASF"),
    (".e57", b"ASTM-E57"),
    (".ply", b"ply"),
    (".pcd", b"# .PCD"),
)

#: Every asset must carry at least this many payload bytes -- a
#: sub-16-byte "photo" is corruption by definition.
MIN_PAYLOAD_BYTES = 16


def _validate_payload(payload: bytes, extension: str) -> None:
    """Raise CorruptEvidenceError when the payload cannot honestly be
    the format its extension claims."""
    if len(payload) < MIN_PAYLOAD_BYTES:
        raise CorruptEvidenceError(
            f"payload too small to be real evidence ({len(payload)} bytes < {MIN_PAYLOAD_BYTES})"
        )
    needle = None
    for ext, magic in _MAGIC_CHECKS:
        if ext == extension:
            needle = magic
            break
    if needle is not None and not payload.startswith(needle):
        raise CorruptEvidenceError(
            f"payload with extension {extension} does not start with its "
            f"expected magic bytes {needle!r} -- corrupt or mislabeled"
        )


def _to_hex_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence_id(package_seed: str, index: int, digest: str) -> str:
    """Stable, content-derived asset id: package seed + sequence + hash.
    Re-importing the same bytes into the same-seeded rebuilt package
    yields the same id; different bytes never collide on the hash part."""
    return f"ev-{package_seed}-{index:04d}-{digest[:16]}"


# ----------------------------------------------------------------------
# Core records
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSource:
    """WHERE the evidence came from: device/sensor identity, operator,
    capture rig, and the platform that produced it. Stable identity for
    "which camera took this" without assuming any specific hardware."""

    source_id: str
    platform: str  # "phone" | "drone" | "terrestrial_laser_scanner" | ...
    device: str    # make/model or rig name
    operator: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "platform": self.platform,
            "device": self.device,
            "operator": self.operator,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> "EvidenceSource":
        return EvidenceSource(
            source_id=data["source_id"],
            platform=data["platform"],
            device=data["device"],
            operator=data.get("operator", ""),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class EvidenceAsset:
    """One immutable, hash-addressed piece of captured evidence with its
    full acquisition/sensor/quality context and processing history.

    The payload bytes are hashed (sha256) and the digest is part of the
    asset id; the bytes themselves live outside this module (on disk /
    object store -- source_uri points at them). quality is a dict of
    measured metrics (blur score, exposure histogram stats, gps fix
    type...), never a fabricated number.
    """

    id: str
    kind: EvidenceKind
    source_uri: str
    sha256: str
    source: EvidenceSource
    acquired_at: Optional[float]          # caller-supplied capture stamp; None = unrecorded
    coordinate_frame: Frame
    sensor_metadata: Dict[str, object]    # focal length, IMU bias, GPS fix, ...
    quality: Dict[str, float]             # measured quality metrics
    provenance: Provenance
    uncertainty: Uncertainty
    processing_history: Tuple[ProcessingRecord, ...]

    def content_key(self) -> str:
        """The dedup key: the payload's own hash."""
        return self.sha256

    def is_canonical(self) -> bool:
        return self.provenance not in (Provenance.GENERATED, Provenance.UNKNOWN, Provenance.CONFLICT)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "source_uri": self.source_uri,
            "sha256": self.sha256,
            "source": self.source.to_dict(),
            "acquired_at": self.acquired_at,
            "coordinate_frame": self.coordinate_frame.value,
            "sensor_metadata": self.sensor_metadata,
            "quality": self.quality,
            "provenance": self.provenance.value,
            "uncertainty": self.uncertainty.to_dict(),
            "processing_history": [p.to_dict() for p in self.processing_history],
        }

    @staticmethod
    def from_dict(data: dict) -> "EvidenceAsset":
        return EvidenceAsset(
            id=data["id"],
            kind=EvidenceKind(data["kind"]),
            source_uri=data["source_uri"],
            sha256=data["sha256"],
            source=EvidenceSource.from_dict(data["source"]),
            acquired_at=data.get("acquired_at"),
            coordinate_frame=Frame(data.get("coordinate_frame", Frame.SESSION_LOCAL.value)),
            sensor_metadata=dict(data.get("sensor_metadata", {})),
            quality=dict(data.get("quality", {})),
            provenance=Provenance(data.get("provenance", Provenance.OBSERVED.value)),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
            processing_history=tuple(
                ProcessingRecord.from_dict(p) for p in data.get("processing_history", [])
            ),
        )


@dataclass(frozen=True)
class EvidenceReference:
    """The binding between one evidence asset and one downstream
    consumption of it (a WorldIR Observation, a fused estimate, a
    reconstruction input). Answers "which captured bytes support this
    claim" -- the unit the provenance panel and the fusion engine cite."""

    reference_id: str
    asset_id: str
    observation_id: str
    role: str  # "primary" | "support" | "measurement_source" | "calibration" | ...

    def to_dict(self) -> dict:
        return {
            "reference_id": self.reference_id,
            "asset_id": self.asset_id,
            "observation_id": self.observation_id,
            "role": self.role,
        }

    @staticmethod
    def from_dict(data: dict) -> "EvidenceReference":
        return EvidenceReference(
            reference_id=data["reference_id"],
            asset_id=data["asset_id"],
            observation_id=data["observation_id"],
            role=data["role"],
        )


@dataclass(frozen=True)
class ObservationSet:
    """A named group of EvidenceReferences feeding ONE downstream
    quantity -- the structured caller interface for the fusion engine:
    fuse_quantity() consumes one ObservationSet's claims; the set keeps
    the raw evidence chain visible regardless of the fusion outcome."""

    set_id: str
    quantity: str
    references: Tuple[EvidenceReference, ...]

    def to_dict(self) -> dict:
        return {
            "set_id": self.set_id,
            "quantity": self.quantity,
            "references": [r.to_dict() for r in self.references],
        }

    @staticmethod
    def from_dict(data: dict) -> "ObservationSet":
        return ObservationSet(
            set_id=data["set_id"],
            quantity=data["quantity"],
            references=tuple(EvidenceReference.from_dict(r) for r in data["references"]),
        )


class EvidencePackage:
    """Versioned, deterministic store of EvidenceAssets + ObservationSets
    with content-hash duplicate detection. Append-only on assets."""

    FORMAT_VERSION = 1

    def __init__(self, package_id: str, seed: str = "pkg"):
        self.package_id = package_id
        self.seed = seed
        self.assets: Dict[str, EvidenceAsset] = {}
        self._content_index: Dict[str, str] = {}   # sha256 -> asset id
        self._order: List[str] = []                # insertion order (deterministic)
        self.observation_sets: Dict[str, ObservationSet] = {}
        self.sources: Dict[str, EvidenceSource] = {}

    # ---- assets ----

    def register_source(self, source: EvidenceSource) -> None:
        self.sources[source.source_id] = source

    def add_asset(self, asset: EvidenceAsset) -> str:
        """Register an already-built asset. Returns the existing asset's
        id when the content hash is already present (duplicate by
        content). Raises DuplicateEvidenceError semantics via
        _content_index -- callers who want the existing id can use
        add_asset's return value; raising happens in the builder where
        policy lives."""
        existing = self._content_index.get(asset.sha256)
        if existing is not None:
            return existing
        self.assets[asset.id] = asset
        self._content_index[asset.sha256] = asset.id
        self._order.append(asset.id)
        return asset.id

    def has_content(self, sha256: str) -> bool:
        return sha256 in self._content_index

    def asset_for_content(self, sha256: str) -> Optional[str]:
        return self._content_index.get(sha256)

    def all_assets(self) -> List[EvidenceAsset]:
        return [self.assets[a] for a in self._order]

    def record_processing(self, asset_id: str, record: ProcessingRecord) -> None:
        """Append one processing entry to an asset's history (the ONLY
        mutation assets support -- history grows, evidence does not
        change)."""
        if asset_id not in self.assets:
            raise UnknownAssetError(f"no asset {asset_id}")
        asset = self.assets[asset_id]
        new_history = asset.processing_history + (record,)
        replacement = EvidenceAsset(
            id=asset.id, kind=asset.kind, source_uri=asset.source_uri,
            sha256=asset.sha256, source=asset.source,
            acquired_at=asset.acquired_at, coordinate_frame=asset.coordinate_frame,
            sensor_metadata=asset.sensor_metadata, quality=asset.quality,
            provenance=asset.provenance, uncertainty=asset.uncertainty,
            processing_history=new_history,
        )
        self.assets[asset_id] = replacement

    def add_observation_set(self, observation_set: ObservationSet) -> None:
        if observation_set.set_id in self.observation_sets:
            raise ValueError(f"observation set {observation_set.set_id} already exists")
        self.observation_sets[observation_set.set_id] = observation_set

    def to_evidence_items(self) -> List[EvidenceItem]:
        """Bridge to the existing session layer: every asset becomes the
        EvidenceItem shape reconstruction backends already consume."""
        return [
            EvidenceItem(
                id=a.id, kind=a.kind, source_uri=a.source_uri,
                captured_at=a.acquired_at, sha256=a.sha256,
                metadata=dict(a.sensor_metadata),
                provenance=a.provenance, uncertainty=a.uncertainty,
            )
            for a in self.all_assets()
        ]

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "package_id": self.package_id,
            "seed": self.seed,
            "sources": {k: v.to_dict() for k, v in sorted(self.sources.items())},
            "asset_order": list(self._order),
            "assets": {k: self.assets[k].to_dict() for k in sorted(self.assets.keys())},
            "observation_sets": {
                k: v.to_dict() for k, v in sorted(self.observation_sets.items())
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "EvidencePackage":
        if data.get("format_version") != EvidencePackage.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported EvidencePackage format version: {data.get('format_version')}"
            )
        package = EvidencePackage(package_id=data["package_id"], seed=data.get("seed", "pkg"))
        package.sources = {
            k: EvidenceSource.from_dict(v) for k, v in data.get("sources", {}).items()
        }
        for asset_data in data.get("assets", {}).values():
            asset = EvidenceAsset.from_dict(asset_data)
            package.assets[asset.id] = asset
            package._content_index[asset.sha256] = asset.id
        package._order = list(data.get("asset_order", list(package.assets.keys())))
        package.observation_sets = {
            k: ObservationSet.from_dict(v) for k, v in data.get("observation_sets", {}).items()
        }
        return package

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvidencePackage) and self.to_dict() == other.to_dict()


# ----------------------------------------------------------------------
# Deterministic builder: the only place payloads are hashed, validated,
# and deduplicated -- policy lives here, storage stays passive.
# ----------------------------------------------------------------------

class DeterministicPackageBuilder:
    """Builds an EvidencePackage from payloads with deterministic ids,
    content validation, and duplicate detection.

    seed controls id stability: rebuilding from the same payloads with
    the same seed reproduces the same asset ids and the same package id.
    """

    def __init__(self, seed: str = "pkg"):
        self.seed = seed
        self._assets: List[EvidenceAsset] = []
        self._seen_content: Dict[str, str] = {}
        self._sources: Dict[str, EvidenceSource] = {}

    def register_source(self, source: EvidenceSource) -> "DeterministicPackageBuilder":
        self._sources[source.source_id] = source
        return self

    def add_payload(
        self,
        payload: bytes,
        *,
        kind: EvidenceKind,
        source: EvidenceSource,
        source_uri: str,
        acquired_at: Optional[float] = None,
        coordinate_frame: Frame = Frame.SESSION_LOCAL,
        sensor_metadata: Optional[Dict[str, object]] = None,
        quality: Optional[Dict[str, float]] = None,
        provenance: Provenance = Provenance.OBSERVED,
        uncertainty: Optional[Uncertainty] = None,
        processing_history: Tuple[ProcessingRecord, ...] = (),
    ) -> str:
        """Hash, validate, and stage one payload. Returns the asset id.

        Raises DuplicateEvidenceError when the exact bytes were already
        staged (the error names the existing asset id); raises
        CorruptEvidenceError when the payload fails its structural
        checks. Timestamps are caller-supplied -- no wall clock.
        """
        digest = _to_hex_digest(payload)
        if digest in self._seen_content:
            raise DuplicateEvidenceError(
                f"duplicate payload (sha256 {digest[:16]}...) already staged as "
                f"{self._seen_content[digest]} -- de-duplicate by reference"
            )
        _validate_payload(payload, _extension_of(source_uri))
        index = len(self._assets)
        asset_id = _evidence_id(self.seed, index, digest)
        asset = EvidenceAsset(
            id=asset_id,
            kind=kind,
            source_uri=source_uri,
            sha256=digest,
            source=source,
            acquired_at=acquired_at,
            coordinate_frame=coordinate_frame,
            sensor_metadata=dict(sensor_metadata or {}),
            quality=dict(quality or {}),
            provenance=provenance,
            uncertainty=uncertainty or Uncertainty(),
            processing_history=processing_history,
        )
        self._assets.append(asset)
        self._seen_content[digest] = asset_id
        return asset_id

    def build(self, package_id: str = "") -> EvidencePackage:
        """Freeze the staged assets into a package. The package id
        defaults to a content-derived, rebuild-stable digest of the
        sorted asset ids."""
        package = EvidencePackage(package_id=package_id, seed=self.seed)
        for source in self._sources.values():
            package.register_source(source)
        for asset in self._assets:
            package.add_asset(asset)
        if not package_id:
            content_digest = _to_hex_digest(
                "|".join(sorted(a.sha256 for a in self._assets)).encode("utf-8")
            )
            package.package_id = f"pkg-{self.seed}-{content_digest[:16]}"
        return package


def _extension_of(source_uri: str) -> str:
    marker = source_uri.rfind(".")
    if marker == -1:
        return ""
    return source_uri[marker:].lower()

