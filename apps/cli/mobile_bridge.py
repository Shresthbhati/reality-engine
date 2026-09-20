"""Mobile field bridge -- the authoritative Python side of the
``re.mobile-session-bundle/v1`` handoff contract (docs/mobile/BACKEND_CONTRACT.md).

The phone app (``frontend/src/apps/mobile``) exports a self-contained
bundle: the session record, per-frame metadata with on-device triage
(ADVISORY), and every frame's actual pixels (base64) keyed by
``sha256(bytes)``. This module is the engine-side reader for that
artifact. It verifies content identity instead of trusting the file,
lands verified pixels into a real ``MultiSourceSession`` through the
existing ``add_source`` path (no parallel evidence model), and derives
follow-up capture tasks (``re.mobile-capture-task/v1``) from measured
bundle state only.

Honesty rules (mirroring the CLI and the mobile app):

  - A frame whose payload does not hash to its declared ``contentSha256``
    is an integrity failure: reported, never ingested, never silently
    dropped -- the operator sees exactly which frames did not make it.
  - Task derivation reads only what the bundle actually measured
    (verdicts, reasons, telemetry, task outcomes). When no frame carries
    heading/GPS telemetry the coverage state is ``UNAVAILABLE`` and tasks
    say so instead of guessing viewing directions.
  - No fabricated telemetry anywhere: absent sensors produce tasks that
    request a capture that would measure them, never invented values.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from evidence.multi_source import MultiSourceSession
from evidence.packages import ProcessingRecord

__all__ = [
    "MOBILE_BUNDLE_SCHEMA",
    "MOBILE_TASK_SCHEMA",
    "MobileBundle",
    "MobileFrame",
    "MobileBundleError",
    "MobileBundleIntegrityError",
    "load_mobile_bundle",
    "ingest_mobile_bundle_to_dir",
    "derive_capture_tasks",
    "write_capture_tasks",
    "parse_exported_at",
    "compare_coverage_snapshots",
    "merge_capture_tasks",
]

MOBILE_BUNDLE_SCHEMA = "re.mobile-session-bundle/v1"
MOBILE_TASK_SCHEMA = "re.mobile-capture-task/v1"

# Compass octants + bounds (degrees) -- mirrors frontend/src/apps/mobile/coverage.ts.
# "N" wraps: 337.5..360 and 0..22.5.
_OCTANT_BOUNDS: List[Tuple[str, float, float]] = [
    ("N", 337.5, 22.5),
    ("NE", 22.5, 67.5),
    ("E", 67.5, 112.5),
    ("SE", 112.5, 157.5),
    ("S", 157.5, 202.5),
    ("SW", 202.5, 247.5),
    ("W", 247.5, 292.5),
    ("NW", 292.5, 337.5),
]

_MIME_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "video/mp4": "mp4",
}

# Verbatim mobile triage reason codes (frontend/src/apps/mobile/types.ts
# QualityReason). Only these may travel in a task's typed `reasons` field;
# anything else stays in `reasonCodes` so the phone never sees a value its
# quality model cannot represent.
_QUALITY_REASONS = frozenset({
    "sharpness_ok",
    "sharpness_low",
    "exposure_ok",
    "overexposed",
    "underexposed",
    "duplicate_of_previous",
    "similar_to_previous",
    "no_reference_available",
})


class MobileBundleError(ValueError):
    """Structural problem: the file is not a readable mobile bundle at all."""


class MobileBundleIntegrityError(MobileBundleError):
    """Whole-bundle integrity problem (missing session id, no frames
    section). Per-frame hash mismatches never raise: they are reported in
    ``MobileBundle.integrity_failures`` and excluded from ``frames``."""


@dataclass
class MobileFrame:
    """One verified frame: metadata verbatim + payload bytes."""

    frame_id: str
    content_sha256: str
    payload: bytes
    meta: dict = field(default_factory=dict)

    @property
    def mime(self) -> str:
        return str(self.meta.get("mime") or "image/jpeg")

    @property
    def verdict(self) -> str:
        return str(self.meta.get("verdict") or "UNASSESSED")

    @property
    def reasons(self) -> List[str]:
        value = self.meta.get("reasons")
        return [str(r) for r in value] if isinstance(value, list) else []

    @property
    def telemetry(self) -> dict:
        value = self.meta.get("telemetry")
        return value if isinstance(value, dict) else {}

    @property
    def task_id(self) -> str | None:
        value = self.meta.get("taskId")
        return str(value) if value else None


@dataclass
class MobileBundle:
    bundle_id: str
    session_id: str
    session_name: str
    exported_at: str
    device: str
    frames: List[MobileFrame] = field(default_factory=list)
    skipped_frames: List[dict] = field(default_factory=list)
    task_outcomes: List[dict] = field(default_factory=list)
    integrity_failures: List[dict] = field(default_factory=list)


def _octant_of(heading_deg: float) -> str:
    heading = ((heading_deg % 360.0) + 360.0) % 360.0
    if heading >= 337.5 or heading < 22.5:
        return "N"
    for name, lo, hi in _OCTANT_BOUNDS:
        if name == "N":
            continue
        if lo <= heading < hi:
            return name
    return "N"


def _frame_id_for(sha_hex: str) -> str:
    return f"frame:{sha_hex[:16]}"


# --------------------------------------------------------------------------
# Reading + verification
# --------------------------------------------------------------------------


def _decode_payload(entry: dict, frame_id: str) -> bytes:
    payload_b64 = entry.get("payloadBase64")
    if not isinstance(payload_b64, str) or not payload_b64:
        raise MobileBundleIntegrityError(f"frame {frame_id}: no payloadBase64 in bundle entry")
    try:
        return base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MobileBundleIntegrityError(
            f"frame {frame_id}: payload is not valid base64 ({exc})"
        ) from exc


def load_mobile_bundle(path: str) -> MobileBundle:
    """Parse + verify a bundle file.

    Every frame payload is re-hashed on read: identity is sha256(bytes),
    never the file's word. Frames that fail verification are collected in
    ``integrity_failures`` and excluded from ``frames`` -- nothing is
    silently dropped, the caller sees the exact count.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise MobileBundleError(f"bundle file not found: {path}")
    try:
        root = json.loads(raw_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MobileBundleError(f"bundle is not valid JSON: {exc}") from exc
    if not isinstance(root, dict):
        raise MobileBundleError("bundle root must be a JSON object")
    if root.get("schema") != MOBILE_BUNDLE_SCHEMA:
        raise MobileBundleError(
            f"unsupported bundle schema {root.get('schema')!r} (expected {MOBILE_BUNDLE_SCHEMA!r})"
        )
    session = root.get("session")
    if not isinstance(session, dict) or not session.get("sessionId"):
        raise MobileBundleIntegrityError("bundle is missing a session object with a sessionId")
    entries = root.get("frames")
    if not isinstance(entries, list):
        raise MobileBundleIntegrityError("bundle is missing a 'frames' array")

    bundle = MobileBundle(
        bundle_id=str(root.get("bundleId") or ""),
        session_id=str(session["sessionId"]),
        session_name=str(session.get("name") or ""),
        exported_at=str(root.get("exportedAt") or ""),
        device=str(root.get("device") or ""),
    )

    for entry in entries:
        if not isinstance(entry, dict):
            bundle.integrity_failures.append(
                {"frameId": "", "reason": "frame entry is not an object"}
            )
            continue
        frame_id = str(entry.get("frameId") or "")
        declared_sha = str(entry.get("contentSha256") or "")
        try:
            payload = _decode_payload(entry, frame_id or "(unnamed)")
        except MobileBundleIntegrityError as exc:
            bundle.integrity_failures.append({"frameId": frame_id, "reason": str(exc)})
            continue
        actual_sha = hashlib.sha256(payload).hexdigest()
        if declared_sha and actual_sha != declared_sha:
            bundle.integrity_failures.append({
                "frameId": frame_id,
                "reason": (
                    f"payload sha256 {actual_sha[:16]} does not match "
                    f"declared {declared_sha[:16]}"
                ),
            })
            continue
        expected_id = _frame_id_for(actual_sha)
        if frame_id and frame_id != expected_id:
            bundle.integrity_failures.append({
                "frameId": frame_id,
                "reason": f"frame id does not match content: expected {expected_id}",
            })
            continue
        meta = {k: v for k, v in entry.items() if k != "payloadBase64"}
        bundle.frames.append(
            MobileFrame(
                frame_id=frame_id or expected_id,
                content_sha256=declared_sha or actual_sha,
                payload=payload,
                meta=meta,
            )
        )

    skipped = root.get("skippedFrames")
    if isinstance(skipped, list):
        bundle.skipped_frames = [s for s in skipped if isinstance(s, dict)]
    outcomes = root.get("taskOutcomes")
    if isinstance(outcomes, list):
        bundle.task_outcomes = [o for o in outcomes if isinstance(o, dict)]
    return bundle


def _extension_for(mime: str) -> str:
    return _MIME_EXTENSIONS.get(mime, "bin")


def write_verified_frames(
    bundle: MobileBundle, session_dir: Path
) -> Tuple[Path, List[str], List[str]]:
    """Write each verified frame's bytes under
    ``<session_dir>/mobile/rgb/<sha256>.<ext>`` (content-addressed).

    Returns ``(rgb_dir, files_written, files_already_present)``. An
    existing file whose bytes do not match its content address is disk
    corruption: the engine refuses to overwrite existing evidence.
    """
    rgb_dir = Path(session_dir) / "mobile" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    already: List[str] = []
    for frame in bundle.frames:
        target = rgb_dir / f"{frame.content_sha256}.{_extension_for(frame.mime)}"
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() != frame.content_sha256:
                raise MobileBundleIntegrityError(
                    f"frame {frame.frame_id}: on-disk file {target.name} does not match "
                    f"its content address -- refusing to overwrite existing evidence"
                )
            already.append(target.name)
            continue
        target.write_bytes(frame.payload)
        written.append(target.name)
    return rgb_dir, written, already


def _sha_to_asset_id(session: MultiSourceSession) -> Dict[str, str]:
    """Map both full sha256 hex AND the 16-char frame-id prefix to assets,
    so `frame:<sha16>` ids in bundles/outcomes resolve to real assets."""
    mapping: Dict[str, str] = {}
    for asset in session.package.all_assets():
        mapping[asset.sha256] = asset.id
        mapping[asset.sha256[:16]] = asset.id
    return mapping



def ingest_mobile_bundle_to_dir(
    bundle: MobileBundle, session: MultiSourceSession, session_dir: Path
) -> dict:
    """Full ingest: verify -> write pixels -> add_source -> record outcomes.

    Returns a report; integrity failures are listed, never swallowed.
    """
    rgb_dir, files_written, files_already = write_verified_frames(bundle, session_dir)

    source_status = "no_frames_to_ingest"
    source_id = ""
    if files_written or files_already:
        record = session.add_source(str(rgb_dir), capture_type="phone")
        source_id = record.source_id
        source_status = record.status.value

    sha_to_asset = _sha_to_asset_id(session)

    # Advisory triage: the mobile's measured verdict is attached to the
    # asset it belongs to, explicitly advisory -- the engine re-derives
    # quality; these records never assert engine-side truth.
    advisory_recorded = 0
    for frame in bundle.frames:
        asset_id = sha_to_asset.get(frame.content_sha256)
        if not asset_id:
            continue
        session.package.record_processing(
            asset_id,
            ProcessingRecord(
                operation="mobile_triage_advisory",
                detail={
                    "verdict": frame.verdict,
                    "reasons": frame.reasons,
                    "telemetry": frame.telemetry,
                    "task_id": frame.task_id,
                    "frame_id": frame.frame_id,
                    "origin": "mobile_bundle_advisory",
                },
            ),
        )
        advisory_recorded += 1


    # Task outcomes: which engine-issued capture requests the phone serviced.
    outcomes_recorded = 0
    for outcome in bundle.task_outcomes:
        task_id = str(outcome.get("taskId") or "")
        if not task_id:
            continue
        frame_ids = [str(fid) for fid in (outcome.get("frameIds") or [])]
        serviced_assets: List[str] = []
        for fid in frame_ids:
            asset_id = sha_to_asset.get(fid.replace("frame:", ""))
            if asset_id:
                serviced_assets.append(asset_id)
        anchor = serviced_assets[0] if serviced_assets else f"task:{task_id}"
        session.package.record_processing(
            anchor,
            ProcessingRecord(
                operation="mobile_task_outcome",
                detail={
                    "task_id": task_id,
                    "status": str(outcome.get("status") or "OPEN"),
                    "frame_ids": frame_ids,
                    "completed_at": outcome.get("completedAt"),
                    "asset_ids": serviced_assets,
                },
            ),
        )
        outcomes_recorded += 1

    return {
        "bundle_id": bundle.bundle_id,
        "session_id": bundle.session_id,
        "frames_verified": len(bundle.frames),
        "frames_integrity_failed": len(bundle.integrity_failures),
        "integrity_failures": bundle.integrity_failures,
        "frames_skipped_in_bundle": len(bundle.skipped_frames),
        "files_written": len(files_written),
        "files_already_present": len(files_already),
        "source_id": source_id,
        "source_status": source_status,
        "advisory_triage_recorded": advisory_recorded,
        "task_outcomes_recorded": outcomes_recorded,
    }



# --------------------------------------------------------------------------
# Capture-task derivation (desktop -> mobile)
# --------------------------------------------------------------------------


def _heading_octants(frames: List[MobileFrame]) -> Dict[str, int]:
    """Count USEFUL frames per compass octant from *reported* headings only."""
    counts: Dict[str, int] = {}
    for frame in frames:
        if frame.verdict != "USEFUL":
            continue
        heading = frame.telemetry.get("headingDeg")
        if not isinstance(heading, (int, float)):
            continue
        name = _octant_of(float(heading))
        counts[name] = counts.get(name, 0) + 1
    return counts


def _coverage_state(frames: List[MobileFrame]) -> dict:
    """Coverage analysis from real telemetry only. Never guesses."""
    heading_frames = [
        f for f in frames if isinstance(f.telemetry.get("headingDeg"), (int, float))
    ]
    gps_frames = [f for f in frames if isinstance(f.telemetry.get("geolocation"), dict)]
    useful = [f for f in frames if f.verdict == "USEFUL"]
    octants = _heading_octants(frames)
    covered = sorted(octants)
    if heading_frames and gps_frames:
        state = "AVAILABLE"
    elif heading_frames or gps_frames:
        state = "PARTIAL"
    else:
        state = "UNAVAILABLE"
    return {
        "state": state,
        "usefulFrames": len(useful),
        "headingFrames": len(heading_frames),
        "gpsFrames": len(gps_frames),
        "coveredOctants": covered,
        "coveredOctantCounts": {k: octants[k] for k in covered},
        "totalOctants": 8,
    }


def _gps_bounds(frames: List[MobileFrame]) -> dict | None:
    lats: List[float] = []
    lons: List[float] = []
    for frame in frames:
        geo = frame.telemetry.get("geolocation")
        if isinstance(geo, dict):
            lat, lon = geo.get("latitude"), geo.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                lats.append(float(lat))
                lons.append(float(lon))
    if not lats:
        return None
    return {
        "minLat": min(lats),
        "maxLat": max(lats),
        "minLon": min(lons),
        "maxLon": max(lons),
    }



def _task_id(bundle_id: str, kind: str, key: str) -> str:
    digest = hashlib.sha256(
        f"{MOBILE_TASK_SCHEMA}|{bundle_id}|{kind}|{key}".encode()
    ).hexdigest()
    return f"task-{digest[:12]}"


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 3)


def derive_capture_tasks(bundle: MobileBundle, max_tasks: int = 12) -> dict:
    """Derive follow-up capture tasks from *measured* bundle state.

    Task kinds (all derived from real evidence, in priority order):

    1. ``reframe`` -- a frame the phone REJECTED for measurable pixel
       defects (motion blur, exposure). Re-shooting the same viewpoint is
       the highest-value next action: the operator already tried.
    2. ``coverage_gap`` -- a viewing direction with zero USEFUL frames,
       computed from REAL compass telemetry. When no frame carries a
       heading, the gap list is empty and the analysis says UNAVAILABLE.
    3. ``telemetry`` -- only when the device reported no heading and no
       GPS at all: a request to capture with sensors enabled. This asks
       for real measurement; it never fabricates a value.

    Task ids are content-derived and stable, so regenerating from the
    same bundle yields the same ids and the phone dedupes them.
    """
    analysis = _coverage_state(bundle.frames)
    outcomes_by_task: Dict[str, dict] = {}
    for outcome in bundle.task_outcomes:
        task_id = str(outcome.get("taskId") or "")
        if task_id:
            outcomes_by_task[task_id] = outcome
    serviced_target_frames = {
        str(fid)
        for outcome in bundle.task_outcomes
        if str(outcome.get("status") or "") == "DONE"
        for fid in (outcome.get("frameIds") or [])
    }

    tasks: List[dict] = []


    # 1. Replacement captures for measurable, rejected pixels.
    for frame in sorted(bundle.frames, key=lambda f: f.frame_id):
        if frame.verdict != "REJECTED" or frame.frame_id in serviced_target_frames:
            continue
        reasons = frame.reasons or ["unmeasured"]
        key = f"reframe|{frame.frame_id}"
        tasks.append({
            "taskId": _task_id(bundle.bundle_id, "reframe", key),
            "kind": "reframe",
            "sessionId": bundle.session_id,
            "createdAt": bundle.exported_at,
            "createdBy": "reality-engine",

            "region": {
                "type": "reframe",
                "frameId": frame.frame_id,
                "label": "same viewpoint as the rejected frame",
            },
            "desiredViewpoint": {
                "description": (
                    "Hold the device steadier and re-capture the same viewpoint; "
                    "wait for the blur to clear between shots."
                ),
                "sourceFrameId": frame.frame_id,
                "poseAvailable": False,
            },
            "evidenceType": "photo",
            "priority": "high",
            "reasons": [r for r in reasons if r in _QUALITY_REASONS],
            "reasonCodes": [f"mobile_triage:{r}" for r in reasons],
            "reason": (
                f"On-device triage rejected {frame.frame_id} for: " + ", ".join(reasons)
            ),
            "expectedCoverageContribution": (
                f"Replaces rejected evidence {frame.frame_id}. No direction coverage is "
                f"gained until a USEFUL frame comes back from this viewpoint."
            ),
            "guidance": (
                "Re-shoot the rejected viewpoint: hold steady, check exposure, "
                "then capture again."
            ),
            "targetFrameIds": [frame.frame_id],
            "status": "OPEN",
        })


    # 2. Coverage gaps from real compass telemetry.
    if analysis["state"] in ("AVAILABLE", "PARTIAL") and analysis["headingFrames"] > 0:
        covered = set(analysis["coveredOctants"])
        useful_total = analysis["usefulFrames"]
        covered_fraction = len(covered) / analysis["totalOctants"]
        for name, lo, _hi in _OCTANT_BOUNDS:
            if name in covered:
                continue
            key = f"coverage|{name}"
            center = (lo + 22.5) % 360.0  # octant center heading
            priority = "high" if useful_total < 8 or covered_fraction < 0.5 else "medium"
            tasks.append({
                "taskId": _task_id(bundle.bundle_id, "coverage_gap", key),
                "kind": "coverage_gap",
                "sessionId": bundle.session_id,
                "createdAt": bundle.exported_at,
                "createdBy": "reality-engine",
                "region": {
                    "type": "viewing_direction",
                    "octant": name,
                    "headingDeg": center,
                    "label": f"the {name}-facing side of the scene",
                },
                "desiredViewpoint": {
                    "description": (
                        f"Face {name} (compass heading about {center:.0f} degrees) and "
                        f"capture the scene from that side."
                    ),
                    "headingDeg": center,
                    "poseAvailable": False,
                },
                "evidenceType": "photo",
                "priority": priority,
                "reasons": [],
                "reasonCodes": [f"coverage_gap:{name}"],
                "reason": (
                    f"Zero USEFUL frames carry a compass heading in the {name} octant."
                ),
                "expectedCoverageContribution": (
                    f"One USEFUL frame facing {name} raises measured direction coverage "
                    f"from {len(covered)}/8 to at most {len(covered) + 1}/8 octants."
                ),
                "guidance": (
                    f"Capture the {name} side: move so the scene fills the frame facing {name}."
                ),
                "targetFrameIds": [],
                "status": "OPEN",
            })


    # 3. Telemetry request -- only when the device reported nothing.
    if analysis["state"] == "UNAVAILABLE":
        tasks.append({
            "taskId": _task_id(bundle.bundle_id, "telemetry", "sensors_off"),
            "kind": "telemetry",
            "sessionId": bundle.session_id,
            "createdAt": bundle.exported_at,
            "createdBy": "reality-engine",
            "region": {"type": "scene", "label": "whole capture site"},
            "desiredViewpoint": None,
            "evidenceType": "telemetry",
            "priority": "medium",
            "reasons": [],
            "reasonCodes": [
                "telemetry_unavailable:heading",
                "telemetry_unavailable:geolocation",
            ],
            "reason": (
                "No frame carried compass heading or geolocation, so direction and "
                "position coverage are UNAVAILABLE rather than guessed."
            ),
            "expectedCoverageContribution": (
                "Does not add pixels directly; it makes direction and position coverage "
                "computable from real measurements on the next handoff."
            ),
            "guidance": (
                "No frame carried compass or GPS telemetry, so coverage is unknown. "
                "Enable location/compass for this capture site and take a few frames; "
                "the next handoff can then be assessed for real coverage gaps."
            ),
            "targetFrameIds": [],
            "status": "OPEN",
        })

    # Supersede tasks the phone already completed or cancelled.
    tasks = [
        t
        for t in tasks
        if outcomes_by_task.get(t["taskId"], {}).get("status") not in ("DONE", "CANCELLED")
    ]

    tasks.sort(key=lambda t: (_priority_rank(str(t["priority"])), str(t["taskId"])))
    tasks = tasks[: max_tasks]

    return {
        "schema": MOBILE_TASK_SCHEMA,
        "bundleId": bundle.bundle_id,
        "sessionId": bundle.session_id,
        "generatedAt": bundle.exported_at,
        "coverageAnalysis": analysis,
        "gpsBounds": _gps_bounds(bundle.frames),
        "tasks": tasks,
    }


def parse_exported_at(value: str) -> float | None:
    """Epoch seconds for an ISO-8601 bundle timestamp, or ``None``.

    ``None`` means *unknown age* -- callers must treat it as uncomparable,
    never as "old" or "now". The phone writes ``Z``-suffixed UTC strings
    (``frontend/src/apps/mobile/bundle.ts``); offset forms and naive
    strings are accepted too (naive is read as UTC, which is what the
    contract requires).
    """
    text = (value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def compare_coverage_snapshots(incoming: dict, incumbent: dict) -> dict:
    """Decide which measured coverage snapshot stays authoritative.

    ``incoming`` is the payload just derived from the operator's bundle;
    ``incumbent`` is the task payload already on disk. A measurement never
    regresses: a bundle exported *before* the persisted snapshot does not
    overwrite it, and one exported *after* it does. Both inputs are read
    as ``generatedAt`` (bundle ``exportedAt``) + ``bundleId``; the decision
    is reported, never inferred silently.
    """
    incoming_at = parse_exported_at(str(incoming.get("generatedAt") or ""))
    incumbent_at = parse_exported_at(str(incumbent.get("generatedAt") or ""))
    has_incumbent_snapshot = "coverageAnalysis" in incumbent
    if incoming_at is None or incumbent_at is None or not has_incumbent_snapshot:
        comparison = "unknown"
        selected = "incoming"
    elif incoming_at > incumbent_at:
        comparison = "newer"
        selected = "incoming"
    elif incoming_at < incumbent_at:
        comparison = "older"
        selected = "incumbent"
    else:
        comparison = "same"
        selected = "incoming"
    return {
        "selected": selected,
        "comparison": comparison,
        "incomingBundleId": str(incoming.get("bundleId") or ""),
        "incomingExportedAt": str(incoming.get("generatedAt") or ""),
        "incumbentBundleId": str(incumbent.get("bundleId") or ""),
        "incumbentExportedAt": str(incumbent.get("generatedAt") or ""),
    }


def _gap_octant(task: dict) -> str | None:
    """The compass octant a ``coverage_gap`` task asks the operator to shoot."""
    region = task.get("region")
    if isinstance(region, dict):
        octant = region.get("octant")
        if isinstance(octant, str) and octant:
            return octant
    codes = task.get("reasonCodes")
    if isinstance(codes, list):
        for code in codes:
            text = str(code)
            if text.startswith("coverage_gap:"):
                return text.split(":", 1)[1]
    return None


def _retract_resolved_tasks(tasks: List[dict], analysis: dict, chosen: dict) -> List[str]:
    """Close OPEN tasks the chosen snapshot proves are already serviced.

    Only *measured* state retracts a task: an octant listed in
    ``coveredOctants`` (USEFUL frames carrying real compass headings)
    closes its own gap request, and a non-UNAVAILABLE telemetry state
    closes the "sensors were off" request. Nothing is closed on
    assumption, and nothing is deleted -- superseded tasks keep their
    history plus the bundle that proved them serviced.
    """
    superseded: List[str] = []
    covered = {str(o) for o in analysis.get("coveredOctants") or []}
    telemetry_measured = str(analysis.get("state") or "") in ("AVAILABLE", "PARTIAL")
    for task in tasks:
        if str(task.get("status") or "") != "OPEN":
            continue
        kind = str(task.get("kind") or "")
        octant = _gap_octant(task) if kind == "coverage_gap" else None
        if kind == "coverage_gap" and not (octant and octant in covered):
            continue
        if kind == "telemetry" and not telemetry_measured:
            continue
        if kind not in ("coverage_gap", "telemetry"):
            continue
        reason = (
            f"USEFUL frames with measured headings now cover the {octant} octant, "
            f"so this gap request is serviced."
            if kind == "coverage_gap"
            else (
                f"Frames now carry real telemetry (coverage state "
                f"{analysis.get('state')}), so this sensor request is serviced."
            )
        )
        task["status"] = "SUPERSEDED"
        task["supersededBy"] = {
            "bundleId": chosen.get("bundleId") or "",
            "exportedAt": chosen.get("exportedAt") or "",
            "reason": reason,
        }
        if task.get("taskId"):
            superseded.append(str(task["taskId"]))
    return superseded


def merge_capture_tasks(payload: dict, existing: dict) -> Tuple[dict, dict]:
    """Merge a freshly derived task payload into the persisted task file.

    Rules (docs/mobile/BACKEND_CONTRACT.md):

    1. **Task set is a union.** Tasks already handed to the phone survive,
       so a handoff is never retracted by a re-run; ids the new bundle
       re-derives replace their older copies in place.
    2. **A newer bundle analysis wins, a stale one never regresses it.**
       The persisted ``coverageAnalysis`` / ``gpsBounds`` snapshot is
       replaced by the incoming one only when the incoming bundle is at
       least as new.
    3. **Resolved requests are retracted, not deleted.** Tasks the chosen
       snapshot proves serviced become ``SUPERSEDED``, carrying the bundle
       that proved it -- append-only evidence semantics.

    Returns ``(merged_payload, report)``.
    """
    incoming_tasks = [dict(t) for t in payload.get("tasks") or [] if isinstance(t, dict)]
    if not isinstance(existing, dict) or ("tasks" not in existing and "coverageAnalysis" not in existing):
        report = {
            "tasks_added": len(incoming_tasks),
            "tasks_retained": 0,
            "tasks_superseded": [],
            "snapshot": compare_coverage_snapshots(payload, {}),
            "selectedBundleId": str(payload.get("bundleId") or ""),
            "staleBundle": False,
        }
        merged = dict(payload)
        merged["coverageSnapshot"] = report["snapshot"]
        merged["tasks"] = incoming_tasks
        return merged, report

    decision = compare_coverage_snapshots(payload, existing)
    use_incoming_snapshot = decision["selected"] == "incoming"
    kept_analysis = (
        payload.get("coverageAnalysis") if use_incoming_snapshot else existing.get("coverageAnalysis")
    )
    if not isinstance(kept_analysis, dict):
        kept_analysis = {}

    # Accumulate GPS bounds across sessions/bundles so indoor or sensor-off runs do not erase prior site GPS.
    incoming_bounds = payload.get("gpsBounds")
    incumbent_bounds = existing.get("gpsBounds")
    if incoming_bounds and incumbent_bounds:
        kept_bounds = {
            "minLat": min(incoming_bounds["minLat"], incumbent_bounds["minLat"]),
            "maxLat": max(incoming_bounds["maxLat"], incumbent_bounds["maxLat"]),
            "minLon": min(incoming_bounds["minLon"], incumbent_bounds["minLon"]),
            "maxLon": max(incoming_bounds["maxLon"], incumbent_bounds["maxLon"]),
        }
    elif incoming_bounds:
        kept_bounds = incoming_bounds
    else:
        kept_bounds = incumbent_bounds

    chosen = {
        "bundleId": decision[f"{'incoming' if use_incoming_snapshot else 'incumbent'}BundleId"],
        "exportedAt": decision[
            f"{'incoming' if use_incoming_snapshot else 'incumbent'}ExportedAt"
        ],
    }

    by_id: Dict[str, dict] = {}
    order: List[str] = []
    for task in existing.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("taskId") or "")
        if not task_id:
            continue
        by_id[task_id] = dict(task)
        order.append(task_id)

    added = 0
    for task in incoming_tasks:
        task_id = str(task.get("taskId") or "")
        if not task_id:
            continue
        if task_id not in by_id:
            order.append(task_id)
            by_id[task_id] = dict(task)
            added += 1
        elif use_incoming_snapshot:
            # Only newer/authoritative bundles update existing task definitions in place
            # while preserving client completion state if already present
            existing_task = by_id[task_id]
            incoming_task = dict(task)
            if existing_task.get("status") in ("DONE", "CANCELLED") and incoming_task.get("status") == "OPEN":
                incoming_task["status"] = existing_task["status"]
                if "completedAt" in existing_task:
                    incoming_task["completedAt"] = existing_task["completedAt"]
                if "completedByFrameIds" in existing_task:
                    incoming_task["completedByFrameIds"] = existing_task["completedByFrameIds"]
            by_id[task_id] = incoming_task

    merged_tasks = [by_id[task_id] for task_id in order]
    superseded = _retract_resolved_tasks(merged_tasks, kept_analysis, chosen)

    merged = dict(existing)
    merged.update(
        {
            "schema": payload.get("schema") or existing.get("schema"),
            "bundleId": payload.get("bundleId") if use_incoming_snapshot else existing.get("bundleId", payload.get("bundleId")),
            "sessionId": payload.get("sessionId") or existing.get("sessionId"),
            "generatedAt": payload.get("generatedAt") if use_incoming_snapshot else existing.get("generatedAt", payload.get("generatedAt")),
            "coverageAnalysis": kept_analysis,
            "gpsBounds": kept_bounds,
            "coverageSnapshot": decision,
            "tasks": merged_tasks,
        }
    )
    return merged, {
        "tasks_added": added,
        "tasks_retained": len(order) - added,
        "tasks_superseded": superseded,
        "snapshot": decision,
        "selectedBundleId": chosen["bundleId"],
        "staleBundle": decision["comparison"] == "older",
    }


def write_capture_tasks(payload: dict, output_path: str) -> str:
    """Persist a capture-task file (deterministic JSON, sorted keys)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


