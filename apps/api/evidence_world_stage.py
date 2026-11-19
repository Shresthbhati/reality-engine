"""Job-side glue for the evidence-to-world flow.

Runs INSIDE the production reconstruction job (``apps/api/jobs.py``), opt-in via ``RE_EVIDENCE_WORLD=1``. It never changes the job's
grade: the outcome is recorded in the job report under ``stages.evidence_world`` and the durable evidence lives under the tenant's
directory, so a failure here is visible but cannot turn a healthy reconstruction into a failed one. An operator cancel does stop the
job (the flow checks the cancel flag between engines; whatever finished stays persisted).

Tenant isolation: the evidence directory is ``<WORLDSTORE_ROOT>/evidence-world/<team>/<world>/<job>/``. Every id is validated
against a strict pattern before it becomes a path segment, and sources are only the server-resolved evidence files of the job --
never strings supplied by a client.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ENV_FLAG = "RE_EVIDENCE_WORLD"
SNAPSHOT_REL = "snapshot.json"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class EvidenceWorldError(ValueError):
    pass


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


#: Production enablement gate. ``enabled()`` stays opt-in (default OFF); this table says what must hold before the DEFAULT may be turned on.
#: ``met`` is only True where a test in this repository executes the claim; everything else is honestly unmet.
ENABLEMENT_CRITERIA: tuple[tuple[str, bool, str], ...] = (
    ("tenant_isolation", True, "tests/test_evidence_world_api.py, tests/test_evidence_world_job.py"),
    ("job_authorization", True, "tests/test_job_authorization.py"),
    ("cancellation_and_failure_isolation", True, "tests/test_evidence_world_job.py"),
    ("real_multi_photo_job_run", False, "no recorded run of the stage on a real multi-photo job yet"),
    ("detector_accuracy_benchmark", False, "no benchmark of the production detectors on labelled interior imagery"),
    ("confidence_calibration", False, "observation calibration is LEGACY_UNSPECIFIED; no calibration dataset"),
    ("metric_3d_observation_contract", False, "BLOCKED: no shared metric-3D schema from the Gemini side; depth maps not persisted"),
    ("quality_gate_thresholds", False, "no benchmark-established thresholds; gates stay UNVERIFIED"),
)


def enablement_gate() -> dict[str, Any]:
    """Whether the stage may be on BY DEFAULT. ``ready`` only when every criterion is met; the env flag is unaffected."""
    unmet = [c for c, met, _ in ENABLEMENT_CRITERIA if not met]
    return {"ready": not unmet, "unmet": unmet,
            "criteria": [{"id": c, "met": met, "evidence": ev} for c, met, ev in ENABLEMENT_CRITERIA]}


def _segment(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value) or ".." in value:
        raise EvidenceWorldError(f"unsafe {what} {value!r} for a storage path")
    return value


def world_root(team_id: str | None, world_id: str, *, base: Path | None = None) -> Path:
    """Tenant-scoped, traversal-safe directory holding all of one world's evidence runs."""
    if base is None:
        from apps.api import worldstore_service

        base = worldstore_service.worldstore_root()
    return Path(base) / "evidence-world" / _segment(team_id or "default", "team id") / _segment(world_id, "world id")


def evidence_root(team_id: str | None, world_id: str, job_id: str, *, base: Path | None = None) -> Path:
    return world_root(team_id, world_id, base=base) / _segment(job_id, "job id")


#: Starting values, NOT tuned against any dataset (the repository's own epipolar constant carries the same caveat). They are recorded in
#: every snapshot under ``scoring`` so a reader sees exactly what gated the associations.
APPEARANCE_MIN = 0.2


def cameras_from_poses(poses, id_to_path: Mapping[str, str], intrinsics: Mapping[str, Any] | None, image_size) -> dict[str, Any]:
    """Registered cameras keyed by the SAME source path the engines saw. ``poses`` are the job's ``(evidence_id, position,
    camera-to-world quaternion wxyz)``. Needs real intrinsics and one common image size; otherwise ``{}`` (no geometric scoring)."""
    if not intrinsics or not image_size or len(image_size) != 2:
        return {}
    from engine.math import Quat, Vec3
    from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

    try:
        k = CameraIntrinsics(fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]), cx=float(intrinsics["cx"]),
                             cy=float(intrinsics["cy"]), width=int(image_size[0]), height=int(image_size[1]))
    except (KeyError, TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for eid, pos, rot in poses:
        path = id_to_path.get(eid)
        if path is not None:
            out[path] = PinholeCamera(k, CameraExtrinsics(Vec3(*pos), Quat(*rot)))
    return out


def _default_flow_inputs():
    from engine.multiengine.adapters import create_adapters_for_registry
    from engine.multiengine.executor import TorchResourceProbe
    from engine.multiengine.registry import maximum_fidelity_registry

    reg = maximum_fidelity_registry()
    return reg, create_adapters_for_registry(reg), TorchResourceProbe()


def run_for_job(
    *,
    team_id: str | None,
    world_id: str,
    job_id: str,
    sources: Sequence[str],
    should_cancel: Callable[[], bool] | None = None,
    base: Path | None = None,
    flow_inputs: Callable[[], tuple[Any, Any, Any]] | None = None,
    cameras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the flow on the job's evidence files and return the compact facts for the job report.

    Never raises for a flow failure (the facts say ``failed`` with the error); raises ``EvidenceWorldError`` only for an unsafe id.
    ``status == "cancelled"`` means the operator cancelled and the caller should stop the job.
    """
    from engine.multiengine.contract import Task
    from engine.multiengine.evidence_flow import StageSpec, run_evidence_flow

    root = evidence_root(team_id, world_id, job_id, base=base)
    real = [str(s) for s in sources if not str(s).startswith("virtual:") and os.path.isfile(str(s))]
    skipped = len(sources) - len(real)
    if not real:
        return {"status": "skipped", "reason": "no readable evidence image files", "skipped_sources": skipped, "run_id": job_id}
    try:
        registry, adapters, probe = (flow_inputs or _default_flow_inputs)()
        scorer, scoring_note = None, None
        usable = {p: c for p, c in (cameras or {}).items() if p in real}
        if len(usable) >= 2:  # a cross-view score needs registered cameras for at least two of the images
            from PIL import Image

            from engine.multiengine.pair_scoring import (
                ScorerConfig,
                DEFAULT_EPIPOLAR_TOLERANCE_PX,
                appearance_from_images,
                make_pair_scorer,
            )

            cfg = ScorerConfig(epipolar_tolerance_px=DEFAULT_EPIPOLAR_TOLERANCE_PX, appearance_min=APPEARANCE_MIN)
            scorer = make_pair_scorer(usable, cfg, appearance=appearance_from_images(Image.open))
            scoring_note = {"method": "epipolar + colour histogram", "epipolar_tolerance_px": cfg.epipolar_tolerance_px,
                            "appearance_min": cfg.appearance_min, "cameras": len(usable),
                            "caveat": "thresholds are untuned starting values; epipolar geometry is necessary, not sufficient"}
        from engine.multiengine.identity import IdentityRegistry

        res = run_evidence_flow(
            root, job_id, real, registry=registry, adapters=adapters, stages=[StageSpec(Task.DETECTION)],
            should_cancel=should_cancel, probe=probe, pair_scorer=scorer, scoring_note=scoring_note,
            identity_registry=IdentityRegistry(world_root(team_id, world_id, base=base) / "identities.json"),
        )
    except Exception as exc:  # noqa: BLE001 -- the optional stage must never fail the job; the failure is recorded
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "run_id": job_id, "skipped_sources": skipped}
    snap = res.snapshot
    return {
        "status": res.status, "run_id": job_id, "counts": snap["counts"], "limitations": snap["limitations"],
        "skipped_sources": skipped, "snapshot": SNAPSHOT_REL, "scoring": snap.get("scoring"),
        "identity_events": [e["kind"] for e in snap.get("identity_events", [])],
    }
