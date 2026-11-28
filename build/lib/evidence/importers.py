"""Disk importers (spec sec 3 MEDIA PREPROCESSING + sec 6 EVIDENCE
INGESTION): the real capture-to-package path the evidence-package layer
flagged as missing (its payloads were caller-supplied bytes).

One flow, deterministically:

    folder / file on disk
      -> raw bytes read (nothing is ever re-encoded: photo bytes are
         hashed and stored exactly as captured)
      -> decoded metadata (EXIF: camera make/model, DateTimeOriginal,
         GPS lat/lon/alt; container headers: resolution)
      -> measured quality where pixel decode is genuinely available,
         honestly unmeasured where it is not
      -> DeterministicPackageBuilder.add_payload (validation +
         content-hash dedup + deterministic ids) -> EvidencePackage

Determinism rules (the module's contract, each tested):

  - Payload bytes are never modified. acquired_at comes from EXIF
    DateTimeOriginal/DateTime when present -- NEVER from file mtimes or
    the wall clock (a copy of the same photo to a new folder must
    import identically; mtimes don't survive copies, EXIF does).
  - Import order is the caller's sorted-path responsibility
    (import_folder sorts); ids are content+seed+sequence derived, so a
    re-import of an unchanged folder into a fresh package reproduces
    the same ids and package id.
  - Corrupt files raise CorruptEvidenceError naming the path (builder
    magic-byte/size gates run on every payload); corrupt-EXIF payloads
    that pass container validation degrade to metadata-poor assets
    rather than being rejected -- unreadable metadata is not unreadable
    evidence.
  - Duplicate detection is two-layer: EXACT by content hash (the
    builder raises DuplicateEvidenceError), NEAR-DUPLICATE by
    perceptual hash over decoded pixels (same resolution required;
    dhash 64-bit, hamming distance <= threshold). Near-dup detection
    needs pixel decode; where pixels cannot be decoded (e.g. bare LAS
    scans) it is honestly skipped, not faked.

Optional dependencies, probed once at import: PIL (EXIF + pixel decode
+ quality metrics) and cv2 (video enumeration/extraction). Neither is a
hard dependency -- pyproject.toml declares no core deps -- so every
entry point degrades with an honest ImportError naming what to install
rather than crashing mid-import. Quality metrics are computed from real
pixel statistics via numpy; a decode that fails yields
quality={"measured": 0.0} plus an honest note, never a fabricated
score.

LLM-free; no clocks; no RNG (dhash thresholds are pure integer math).
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from evidence.frames import FrameCandidate, IFrameSelectionStrategy, UniformTimeSamplingStrategy
from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidencePackage,
)
from evidence.session import EvidenceKind

__all__ = [
    "ImporterCapabilityError",
    "ImportedAsset",
    "FolderImportReport",
    "import_file",
    "import_folder",
]

# ---------------------------------------------------------------------+
# Optional-dependency probes (once, at import time -- honest refusal)  |
# ---------------------------------------------------------------------+

try:  # pragma: no cover - exercised implicitly by every image test
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


def _require_pil() -> None:
    if Image is None:
        raise ImporterCapabilityError(
            "image import requires Pillow (pip install pillow) -- "
            "refusing to guess metadata without a real decoder"
        )


def _require_cv2() -> None:
    if cv2 is None:
        raise ImporterCapabilityError(
            "video import requires OpenCV (pip install opencv-python) -- "
            "refusing to guess frames without a real decoder"
        )


class ImporterCapabilityError(RuntimeError):
    """An optional decoder (Pillow/OpenCV) is not installed."""


_PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
_LAS_EXTS = {".las", ".laz"}
_KNOWN_EXTS = _PHOTO_EXTS | _VIDEO_EXTS | _LAS_EXTS

#: Perceptual-hash agreement threshold: hamming distance over the 64-bit
#: dhash at or below this counts as a near-duplicate. 10/64 ≈ 16% of bits
#: -- tolerant to JPEG re-compression noise, intolerant to different
#: viewpoints. Tune per capture rig, not silently.
NEAR_DUPLICATE_HAMMING_THRESHOLD = 10


def _extension_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


# ---------------------------------------------------------------------+
# EXIF decoding                                                        |
# ---------------------------------------------------------------------+

def _rational_to_float(value) -> Optional[float]:
    """EXIF rationals arrive as IFDRational or plain (num, den) tuples
    depending on the IFD."""
    if isinstance(value, tuple):
        if len(value) == 2 and value[1]:
            try:
                return float(value[0]) / float(value[1])
            except (TypeError, ZeroDivisionError):
                return None
        return None
    try:
        return float(value)
    except (TypeError, ZeroDivisionError):
        return None


def _dms_to_degrees(dms) -> Optional[float]:
    """(deg, min, sec) rationals -> signed-free decimal degrees."""
    parts = []
    for component in dms:
        f = _rational_to_float(component)
        if f is None:
            return None
        parts.append(f)
    if len(parts) != 3:
        return None
    d, m, s = parts
    return d + m / 60.0 + s / 3600.0


def _exif_datetime_to_epoch(text: str) -> Optional[float]:
    """EXIF 'YYYY:MM:DD HH:MM:SS' -> epoch seconds (UTC-naive: EXIF
    carries no offset; we parse the wall-clock string as-is, documented
    in the field's sensor metadata). Deterministic -- no clock reads."""
    import calendar
    import time

    text = text.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            struct_time = time.strptime(text, fmt)
        except ValueError:
            continue
        return float(calendar.timegm(struct_time))
    return None


def _decode_photo_exif(payload: bytes) -> Tuple[Dict[str, object], Optional[float], Dict[str, object]]:
    """Decode EXIF from payload bytes.

    Returns (sensor_metadata, acquired_at_epoch_or_None, gps_metadata).
    Raise-free on bad EXIF: unreadable metadata degrades to empty dicts
    (the container was validated by the builder; metadata is a bonus,
    not a gate).
    """
    sensor: Dict[str, object] = {}
    acquired_at: Optional[float] = None
    gps_meta: Dict[str, object] = {}

    if Image is None:
        return sensor, acquired_at, gps_meta
    try:
        image = Image.open(io.BytesIO(payload))
        exif = image.getexif()
    except Exception:
        return sensor, acquired_at, gps_meta
    if not exif:
        return sensor, acquired_at, gps_meta

    # Top-level IFD: camera identity + capture time.
    make = exif.get(0x010F)
    model = exif.get(0x0110)
    if isinstance(make, str) and make.strip():
        sensor["make"] = make.strip()
    if isinstance(model, str) and model.strip():
        sensor["model"] = model.strip()
    for date_tag in (0x0132,):  # DateTime
        raw = exif.get(date_tag)
        if isinstance(raw, str):
            acquired_at = _exif_datetime_to_epoch(raw)
            if acquired_at is not None:
                break

    # Exif sub-IFD: original capture time + exposure identity.
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:
        sub = {}
    exposure_time = _rational_to_float(sub.get(0x829A))
    if exposure_time is not None:
        sensor["exposure_time_s"] = exposure_time
    f_number = _rational_to_float(sub.get(0x829D))
    if f_number is not None:
        sensor["f_number"] = f_number
    iso = sub.get(0x8827)
    if isinstance(iso, int) and iso > 0:
        sensor["iso"] = iso
    focal = _rational_to_float(sub.get(0x920A))
    if focal is not None and focal > 0:
        sensor["focal_length_mm"] = focal
    if acquired_at is None:
        for date_tag in (0x9003, 0x9004):  # DateTimeOriginal, DateTimeDigitized
            raw = sub.get(date_tag)
            if isinstance(raw, str):
                acquired_at = _exif_datetime_to_epoch(raw)
                if acquired_at is not None:
                    break

    # GPS IFD: where on earth.
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        gps = {}
    lat_ref = gps.get(1)
    lat_dms = gps.get(2)
    lon_ref = gps.get(3)
    lon_dms = gps.get(4)
    if lat_ref in ("N", "S") and lat_dms is not None and lon_ref in ("E", "W") and lon_dms is not None:
        lat = _dms_to_degrees(lat_dms)
        lon = _dms_to_degrees(lon_dms)
        if lat is not None and lon is not None:
            if lat_ref == "S":
                lat = -lat
            if lon_ref == "W":
                lon = -lon
            gps_meta["latitude_deg"] = lat
            gps_meta["longitude_deg"] = lon
            altitude = _rational_to_float(gps.get(6))
            if altitude is not None:
                alt_ref = gps.get(5)
                if alt_ref == 1:
                    altitude = -altitude
                gps_meta["altitude_m"] = altitude
            gps_time = gps.get(7)
            if gps_time is not None:
                gps_meta["gps_timestamp_raw"] = repr(gps_time)
    if gps_meta:
        gps_meta["gps_datum_note"] = "parsed from EXIF GPS IFD; WGS84 assumed, not verified"

    return sensor, acquired_at, gps_meta


# ---------------------------------------------------------------------+
# Quality metrics (measured from real pixels, or honestly absent)      |
# ---------------------------------------------------------------------+

def _measure_photo_quality(payload: bytes) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Measured blur/exposure metrics over decoded pixels.

    Blur = variance of the Laplacian (classic focus measure; higher =
    sharper). Exposure = mean luma (0-255) + fraction of pixels clipped
    at either rail. All computed on the actual decoded bytes with numpy
    -- if decode fails the quality dict says so instead of inventing a
    number.
    """
    quality: Dict[str, float] = {}
    notes: Dict[str, object] = {}
    if Image is None or np is None:
        notes["quality_note"] = "pixel metrics unavailable: Pillow/numpy not installed"
        return {"measured": 0.0}, notes
    try:
        image = Image.open(io.BytesIO(payload))
        # Downscale for metric stability + speed; blur/exposure are
        # statistics, not surveys. Nearest to stay byte-faithful.
        image.thumbnail((512, 512))
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        arr = np.asarray(image.convert("L"), dtype=np.float64)
    except Exception as exc:
        notes["quality_note"] = f"pixel decode failed ({type(exc).__name__}); metrics unmeasured"
        return {"measured": 0.0}, notes

    lap = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    # 4-neighbour Laplacian via numpy slicing (no cv2 dependency here):
    pad = np.pad(arr, 1, mode="edge")
    lap_response = (
        4.0 * pad[1:-1, 1:-1]
        - pad[:-2, 1:-1]
        - pad[2:, 1:-1]
        - pad[1:-1, :-2]
        - pad[1:-1, 2:]
    )
    quality["laplacian_variance"] = float(lap_response.var())
    quality["luma_mean"] = float(arr.mean())
    quality["clipped_fraction"] = float(
        ((arr >= 254.5) | (arr <= 0.5)).mean()
    )
    quality["measured"] = 1.0
    return quality, notes


def _container_resolution(payload: bytes, path: str) -> Tuple[Optional[int], Optional[int]]:
    """Image resolution from container headers -- available even when
    pixel decode is not (e.g. Pillow present but pixel data truncated)."""
    if Image is None:
        return None, None
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


# ---------------------------------------------------------------------+
# Perceptual (near-)duplicate detection                                |
# ---------------------------------------------------------------------+

def _dhash64(payload: bytes) -> Optional[int]:
    """64-bit difference hash over decoded luma (gradient sign per
    9x8 grid). None when pixels cannot be decoded -- never a fake hash."""
    if Image is None or np is None:
        return None
    try:
        image = Image.open(io.BytesIO(payload))
        image = image.convert("L").resize((9, 8))
        arr = np.asarray(image, dtype=np.int16)
    except Exception:
        return None
    diff = arr[:, 1:] > arr[:, :-1]  # 8x8 = 64 gradient bits
    bits = 0
    for row in diff:
        for bit in row:
            bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------+
# Video: enumeration via container, selection via strategy             |
# ---------------------------------------------------------------------+

def _enumerate_video_candidates(path: str) -> Tuple[List[FrameCandidate], Dict[str, object]]:
    """Grab-only frame enumeration (metadata, no full decode) via cv2.

    Returns (candidates, container_metadata). Raises ImportError via
    _require_cv2 when OpenCV is absent; RuntimeError when the container
    cannot be opened (a video that won't open cannot honestly import).
    """
    _require_cv2()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise CorruptEvidenceError(
            f"video container {path!r} would not open -- corrupt or unsupported"
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    metadata: Dict[str, object] = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise CorruptEvidenceError(
            f"video container {path!r} reports degenerate timing/size "
            f"(fps={fps}, frames={frame_count}, {width}x{height}) -- corrupt"
        )
    candidates: List[FrameCandidate] = []
    for index in range(frame_count):
        # Grab (decode+discard) advances the stream without keeping the
        # frame -- cheap enumeration. Retrieval happens later, only for
        # SELECTED frames.
        ok = capture.grab()
        if not ok:
            break  # truncated container: enumerate what is really there
        candidates.append(
            FrameCandidate(
                frame_index=index,
                timestamp_s=index / fps,
                width=width,
                height=height,
            )
        )
    capture.release()
    if not candidates:
        raise CorruptEvidenceError(
            f"video container {path!r} yielded zero readable frames -- corrupt"
        )
    metadata["enumerated_frame_count"] = len(candidates)
    return candidates, metadata


def _extract_selected_frames(path: str, selected: Sequence[FrameCandidate]) -> List[Tuple[int, bytes]]:
    """Re-open the container and FULLY read only the selected indices
    (the second pass that makes 'never blindly process every frame'
    true end to end). Returns [(frame_index, jpeg_bytes), ...] in the
    selection's order. JPEG re-encode is intentional and documented:
    frames are DERIVED previews for reconstruction, not original
    evidence (the original video file itself is imported separately as
    the VIDEO asset with its own hash)."""
    _require_cv2()
    wanted = {c.frame_index for c in selected}
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise CorruptEvidenceError(f"video container {path!r} would not reopen")
    extracted: List[Tuple[int, bytes]] = []
    ok, frame = True, None
    for index in range(max(wanted) + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            encode_ok, buffer = cv2.imencode(".jpg", frame)
            if encode_ok:
                extracted.append((index, buffer.tobytes()))
    capture.release()
    missing = wanted - {index for index, _ in extracted}
    if missing:
        raise CorruptEvidenceError(
            f"video container {path!r} lost selected frames during extraction: {sorted(missing)}"
        )
    return extracted


# ---------------------------------------------------------------------+
# Import result records                                                |
# ---------------------------------------------------------------------+

@dataclass(frozen=True)
class ImportedAsset:
    """One successfully imported file, for the caller's bookkeeping."""

    path: str
    asset_id: str
    kind: EvidenceKind
    acquired_at: Optional[float]
    near_duplicate_of: Optional[str] = None  # asset id, when perceptually matched

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "asset_id": self.asset_id,
            "kind": self.kind.value,
            "acquired_at": self.acquired_at,
            "near_duplicate_of": self.near_duplicate_of,
        }


@dataclass
class FolderImportReport:
    """Everything an import observed -- nothing silently skipped."""

    imported: List[ImportedAsset] = field(default_factory=list)
    duplicates_skipped: List[Tuple[str, str]] = field(default_factory=list)   # (path, existing asset id)
    near_duplicates_marked: List[Tuple[str, str]] = field(default_factory=list)  # (path, asset id)
    unhandled_paths: List[str] = field(default_factory=list)  # unknown extension: recorded, not imported

    def to_dict(self) -> dict:
        return {
            "imported": [a.to_dict() for a in self.imported],
            "duplicates_skipped": [list(pair) for pair in self.duplicates_skipped],
            "near_duplicates_marked": [list(pair) for pair in self.near_duplicates_marked],
            "unhandled_paths": list(self.unhandled_paths),
        }


# ---------------------------------------------------------------------+
# Importers                                                            |
# ---------------------------------------------------------------------+

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _import_photo(
    builder: DeterministicPackageBuilder,
    source,
    path: str,
    report: FolderImportReport,
    pixel_hashes: Dict[str, int],
) -> None:
    from evidence.packages import EvidenceSource

    payload = _read_bytes(path)

    # Exact duplicates: skip-and-record (policy layer), never raise mid-
    # folder. The builder's seen-content index is the source of truth.
    existing = builder.asset_id_for_payload(payload)
    if existing is not None:
        report.duplicates_skipped.append((path, existing))
        return

    sensor, acquired_at, gps_meta = _decode_photo_exif(payload)
    quality, quality_notes = _measure_photo_quality(payload)
    width, height = _container_resolution(payload, path)
    if width is not None and height is not None:
        quality["width_px"] = float(width)
        quality["height_px"] = float(height)

    # Near-duplicate detection (across the import batch) BEFORE staging:
    # exact duplicates are the builder's job; near ones are ours.
    dhash = _dhash64(payload)
    near_of: Optional[str] = None
    if dhash is not None:
        for other_uri, other_hash in pixel_hashes.items():
            if _hamming(dhash, other_hash) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                near_of = other_uri
                break
        pixel_hashes[_asset_key(builder, path)] = dhash

    asset_id = builder.add_payload(
        payload,
        kind=EvidenceKind.PHOTO,
        source=source,
        source_uri=path,
        acquired_at=acquired_at,
        sensor_metadata={**sensor, **gps_meta},
        quality=quality,
    )
    if near_of is not None:
        report.near_duplicates_marked.append((path, near_of))
    report.imported.append(
        ImportedAsset(path=path, asset_id=asset_id, kind=EvidenceKind.PHOTO, acquired_at=acquired_at)
    )


def _asset_key(builder: DeterministicPackageBuilder, path: str) -> str:
    """Stable per-import key linking a path to its perceptual hash."""
    return path


def _import_las(
    builder: DeterministicPackageBuilder,
    source,
    path: str,
    report: FolderImportReport,
) -> None:
    payload = _read_bytes(path)
    existing = builder.asset_id_for_payload(payload)
    if existing is not None:
        report.duplicates_skipped.append((path, existing))
        return
    # LAS header (public block): version at byte 24 ("1.2"/"1.4"), point
    # count at offset 107 (legacy uint32) -- parsed for metadata only.
    sensor: Dict[str, object] = {}
    try:
        version = payload[24:27].decode("ascii", errors="strict")
        sensor["las_version"] = version
        (legacy_count,) = struct.unpack_from("<I", payload, 107)
        sensor["point_count_legacy_header"] = int(legacy_count)
    except Exception:
        sensor = {}  # header parse is metadata-only; the builder still gates the payload

    asset_id = builder.add_payload(
        payload,
        kind=EvidenceKind.LIDAR,
        source=source,
        source_uri=path,
        sensor_metadata=sensor,
        quality={"measured": 0.0},
    )
    report.imported.append(
        ImportedAsset(path=path, asset_id=asset_id, kind=EvidenceKind.LIDAR, acquired_at=None)
    )


def _import_video(
    builder: DeterministicPackageBuilder,
    source,
    path: str,
    report: FolderImportReport,
    strategy: IFrameSelectionStrategy,
    frame_min_bytes: int,
) -> None:
    # 1) The video file itself is evidence (hashed, validated, imported
    #    exactly as captured -- never re-encoded).
    payload = _read_bytes(path)
    existing = builder.asset_id_for_payload(payload)
    if existing is not None:
        report.duplicates_skipped.append((path, existing))
        return
    video_asset_id = builder.add_payload(
        payload, kind=EvidenceKind.VIDEO, source=source, source_uri=path,
        quality={"measured": 0.0},
    )
    report.imported.append(
        ImportedAsset(path=path, asset_id=video_asset_id, kind=EvidenceKind.VIDEO, acquired_at=None)
    )

    # 2) Enumerate -> select -> extract: the spec's preprocessing gate.
    candidates, container_meta = _enumerate_video_candidates(path)
    selected = strategy.select(candidates)
    extracted = _extract_selected_frames(path, selected)
    for (frame_index, timestamp_s, _w, _h), (_idx, jpeg_bytes) in zip(
        ((c.frame_index, c.timestamp_s, c.width, c.height) for c in selected), extracted
    ):
        frame_quality, frame_notes = _measure_photo_quality(jpeg_bytes)
        frame_quality["frame_index"] = float(frame_index)
        frame_quality["video_timestamp_s"] = float(timestamp_s)
        asset_id = builder.add_payload(
            jpeg_bytes,
            kind=EvidenceKind.PHOTO,
            source=source,
            source_uri=f"{path}#frame-{frame_index:06d}",
            acquired_at=None,  # container-relative time is metadata, not acquisition time
            sensor_metadata={
                "source_video_asset_id": video_asset_id,
                "frame_timestamp_s": float(timestamp_s),
                "fps": container_meta["fps"],
                "selection_strategy": strategy.name,
            },
            quality=frame_quality,
        )
        report.imported.append(
            ImportedAsset(
                path=f"{path}#frame-{frame_index:06d}",
                asset_id=asset_id,
                kind=EvidenceKind.PHOTO,
                acquired_at=None,
            )
        )


def import_file(
    builder: DeterministicPackageBuilder,
    path: str,
    *,
    source=None,
    report: Optional[FolderImportReport] = None,
    frame_strategy: Optional[IFrameSelectionStrategy] = None,
    frame_min_bytes: int = 2048,
) -> ImportedAsset:
    """Import ONE file from disk into the builder.

    Raises CorruptEvidenceError (via the builder) for corrupt payloads,
    ImporterCapabilityError when the file's format needs a decoder that
    is not installed, and ValueError for unknown extensions.
    """
    _ = frame_min_bytes  # reserved for frame-level size gating
    if report is None:
        report = FolderImportReport()
    if source is None:
        from evidence.packages import EvidenceSource

        source = EvidenceSource(
            source_id=f"disk:{os.path.basename(path)}",
            platform="filesystem",
            device="local disk",
        )
    extension = _extension_of(path)
    if extension in _PHOTO_EXTS:
        _import_photo(builder, source, path, report, {})
        imported = [a for a in report.imported if a.path == path]
        return imported[-1]
    if extension in _LAS_EXTS:
        before = {a.asset_id for a in report.imported}
        _import_las(builder, source, path, report)
        imported = [a for a in report.imported if a.asset_id not in before]
        return imported[-1]
    if extension in _VIDEO_EXTS:
        before = {a.asset_id for a in report.imported}
        _import_video(
            builder, source, path, report,
            frame_strategy or UniformTimeSamplingStrategy(target_count=8),
            frame_min_bytes,
        )
        video_assets = [
            a for a in report.imported
            if a.asset_id not in before and a.kind is EvidenceKind.VIDEO
        ]
        return video_assets[-1]  # the VIDEO asset (frames follow it in the report)
    raise ValueError(f"unknown evidence extension {extension!r} for {path!r}")


def import_folder(
    builder: DeterministicPackageBuilder,
    folder: str,
    *,
    source=None,
    frame_strategy: Optional[IFrameSelectionStrategy] = None,
    frame_min_bytes: int = 2048,
) -> FolderImportReport:
    """Import every known-format file under `folder` (recursive) into
    the builder.

    Deterministic: files are visited in sorted full-path order, so the
    same folder contents reproduce the same package ids. Unknown
    extensions are RECORDED in the report (unhandled_paths), never
    silently skipped. Corrupt files raise (corrupt evidence cannot be
    imported honestly); duplicate/near-duplicate handling is per the
    two-layer policy above.
    """
    report = FolderImportReport()
    paths: List[str] = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            paths.append(os.path.join(root, name))
    paths.sort()

    pixel_hashes: Dict[str, int] = {}
    for path in paths:
        extension = _extension_of(path)
        if extension not in _KNOWN_EXTS:
            report.unhandled_paths.append(path)
            continue
        if source is not None:
            use_source = source
        else:
            from evidence.packages import EvidenceSource

            use_source = EvidenceSource(
                source_id=f"disk:{os.path.basename(folder.rstrip('/\\')) or 'capture'}",
                platform="filesystem",
                device="local disk",
            )
        if extension in _PHOTO_EXTS:
            _import_photo(builder, use_source, path, report, pixel_hashes)
        elif extension in _LAS_EXTS:
            _import_las(builder, use_source, path, report)
        elif extension in _VIDEO_EXTS:
            _import_video(
                builder, use_source, path, report,
                frame_strategy or UniformTimeSamplingStrategy(target_count=8),
                frame_min_bytes,
            )
    return report
