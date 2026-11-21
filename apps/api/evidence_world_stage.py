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

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ENV_FLAG = "RE_EVIDENCE_WORLD"
ENV_DEPTH_ENGINES = "RE_EVIDENCE_WORLD_DEPTH_ENGINES"
SNAPSHOT_REL = "snapshot.json"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class EvidenceWorldError(ValueError):
    pass


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


#: Production enablement gate. ``enabled()`` stays opt-in (default OFF); this table says what must hold before the DEFAULT may be turned on.
#: ``met`` is only True where a test in this repository executes the claim; everything else is honestly unmet.
#: (id, claimed met, evidence test files, note). A claimed-met criterion is only counted when EVERY evidence file exists in the checkout:
#: deleting the test demotes the claim. File existence is a floor, not proof the tests pass -- CI running them is the proof.
ENABLEMENT_CRITERIA: tuple[tuple[str, bool, tuple[str, ...], str], ...] = (
    ("tenant_isolation", True, ("tests/test_evidence_world_api.py", "tests/test_evidence_world_job.py"), ""),
    ("job_authorization", True, ("tests/test_job_authorization.py",), ""),
    ("cancellation_and_failure_semantics", True, ("tests/test_evidence_world_job.py", "tests/test_evidence_world_failure_semantics.py"), ""),
    ("metric3d_contract_published", True, ("tests/test_metric3d_contract.py",), "contract + ingest exist; see the producers criterion"),
    ("covariance_propagation", True, ("tests/test_canonical_covariance_propagation.py", "tests/test_worldir_bridge_uncertainty_roundtrip.py"), ""),
    ("metric3d_producers_emit_contract", True, ("tests/test_metric3d_producers.py",),
     "depth + sparse-lift producers emit metric3d.v1 with recoverable raw arrays (real-model run on South Building recorded in the ledger); "
     "wired into the stage only behind RE_EVIDENCE_WORLD_DEPTH_ENGINES, and no depth producer yet feeds fusion"),
    ("metric_anchor_integrity", True, ("tests/test_metric_anchor_integrity.py",), "verified on synthetic scenes only; no physical anchor applied to real data"),
    ("real_multi_photo_job_run", False, (), "one outdoor run produced 0 detections; no furnished-interior multi-photo run exists"),
    ("detector_accuracy_benchmark", False, (), "no benchmark of the production detectors on labelled interior imagery"),
    ("confidence_calibration", False, (), "observation calibration is LEGACY_UNSPECIFIED; no calibration dataset"),
    ("quality_gate_thresholds", False, (), "no benchmark-established thresholds; gates stay UNVERIFIED"),
)


def enablement_gate(repo_root: Path | None = None) -> dict[str, Any]:
    """Whether the stage may be on BY DEFAULT. ``ready`` only when every criterion is met; the env flag is unaffected."""
    root = repo_root or Path(__file__).resolve().parents[2]
    rows = []
    for cid, claimed, files, note in ENABLEMENT_CRITERIA:
        missing = [f for f in files if not (root / f).is_file()]
        met = claimed and bool(files) and not missing
        rows.append({"id": cid, "met": met, "evidence": list(files), "missing_evidence": missing,
                     "note": note or ("evidence file(s) missing" if claimed and missing else "")})
    unmet = [r["id"] for r in rows if not r["met"]]
    return {"ready": not unmet, "unmet": unmet, "criteria": rows}


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
    depth_engines: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the flow on the job's evidence files and return the compact facts for the job report.

    ``depth_engines`` (default: the ``RE_EVIDENCE_WORLD_DEPTH_ENGINES`` env, comma separated; empty = none) runs those depth
    adapters ONE AT A TIME after the detection flow and persists their ``metric3d.v1`` records + raw depth arrays in the same
    run's evidence matrix. An engine that is unknown, fails or is rejected is reported in ``depth`` -- never skipped silently.

    Never raises for a flow failure (the facts say ``failed`` with the error); raises ``EvidenceWorldError`` only for an unsafe id.
    ``status == "cancelled"`` means the operator cancelled and the caller should stop the job.
    """
    from engine.multiengine.contract import Task
    from engine.multiengine.evidence_flow import StageSpec, run_evidence_flow

    root = evidence_root(team_id, world_id, job_id, base=base)
    recorded = root / SNAPSHOT_REL
    if recorded.is_file():  # a job's evidence is executed ONCE: re-running would append a second copy of every observation
        try:
            snap = json.loads(recorded.read_text(encoding="utf-8"))
            return {**_facts(snap, job_id, 0), "duplicate_execution": True}
        except (OSError, ValueError, KeyError):
            return {"status": "failed", "run_id": job_id, "duplicate_execution": True,
                    "error": "this job already has evidence on disk but its snapshot is unreadable; it was not re-run"}
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
                DEFAULT_EPIPOLAR_TOLERANCE_PX,
                ScorerConfig,
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

    wanted = list(depth_engines) if depth_engines is not None else [
        e.strip() for e in os.environ.get(ENV_DEPTH_ENGINES, "").split(",") if e.strip()
    ]
    if not wanted and usable and ("depth_anything" in adapters or "midas" in adapters):
        wanted = ["depth_anything"] if "depth_anything" in adapters else ["midas"]

    depth_reports = []
    if wanted and res.status != "cancelled":
        depth_reports = _run_depth(root, job_id, real, wanted, adapters)

    # 3D spatial lifting if registered cameras and depth maps are available
    if usable and res.status != "cancelled":
        from engine.multiengine.evidence_matrix import EvidenceMatrix
        matrix = EvidenceMatrix(root)
        depth_maps = _extract_depth_maps(root, real)
        if depth_maps:
            lifted_n = _lift_detections_for_cameras(matrix, usable, depth_maps, job_id)
            if lifted_n > 0:
                from engine.multiengine.evidence_flow import derive_entities
                from engine.multiengine.worldir_bridge import convert
                from world.ontology.universal_object_ontology import get_universal_ontology
                onto = get_universal_ontology()
                identity_reg = IdentityRegistry(world_root(team_id, world_id, base=base) / "identities.json")
                derived = derive_entities(
                    matrix, active=None, registry=registry, onto=onto, pair_scorer=scorer,
                    identity=identity_reg, record=True,
                )
                res.bridge = convert(derived.canonicals, matrix, ontology=onto)
                res.canonicals = derived.canonicals
                res.snapshot["counts"]["lifted_entities"] = lifted_n

    facts = _facts(res.snapshot, job_id, skipped, status=res.status)
    if depth_reports:
        facts["depth"] = depth_reports
    facts["world_ir"] = persist_world(root, job_id, res.bridge)
    if facts["status"] == "complete" and facts["world_ir"]["status"] != "persisted":
        facts["status"] = "partial"  # a flow that finished but whose canonical world was not persisted and verified is not a success
    return facts


def _extract_depth_maps(root: Path, sources: Sequence[str]) -> dict[str, Any]:
    import io

    import numpy as np

    from engine.multiengine.evidence_matrix import EvidenceMatrix

    matrix = EvidenceMatrix(root)
    maps: dict[str, np.ndarray] = {}
    for o in matrix.store.query(task="metric3d"):
        if o.source_id not in sources or o.source_id in maps:
            continue
        pred = o.predicted or {}
        raw_ref = pred.get("raw_artifact_ref")
        if raw_ref:
            try:
                raw_bytes = matrix.artifacts.get(raw_ref)
                arr = np.load(io.BytesIO(raw_bytes))
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    maps[o.source_id] = arr
            except Exception:
                pass
    return maps


def _lift_detections_for_cameras(
    matrix: Any,
    usable_cameras: Mapping[str, Any],
    depth_maps: Mapping[str, Any],
    job_id: str,
) -> int:
    """Lifts 2D bounding boxes into 3D using calibrated cameras and available depth maps."""
    from engine.multiengine.metric_lifting import Metric3DLifter
    from engine.multiengine.observation import make_observation
    from world.ontology.universal_object_ontology import get_universal_ontology

    lifter = Metric3DLifter(get_universal_ontology())
    bbox_obs = [o for o in matrix.store.query(quantity="bbox") if matrix.status(o.observation_id) != "rejected"]
    lifted_count = 0

    for o in bbox_obs:
        src = o.source_id
        if src not in usable_cameras or src not in depth_maps:
            continue
        cam = usable_cameras[src]
        depth_arr = depth_maps[src]
        box = o.image_region
        if not box or len(box) < 4:
            continue
        lbl = str(o.predicted.get("label", "object"))

        is_synthetic = "synthetic" in src.lower() or "synthetic" in str(getattr(cam, "custom_properties", {})).lower()
        is_physical = bool(getattr(cam, "is_physically_metric", False))
        scale_source = str(getattr(cam, "scale_source", ""))

        if is_physical and scale_source:
            scale_state = "PHYSICALLY_ANCHORED"
            scale_units = "METRES"
            is_metric = True
        elif is_synthetic:
            scale_state = "SYNTHETIC_SCALE_ALIGNED"
            scale_units = "SYNTHETIC_METRES"
            is_metric = False
        else:
            scale_state = "SFM_GAUGE"
            scale_units = "COLMAP_SFM_UNITS"
            is_metric = False

        lifted = lifter.lift_detection(
            entity_id=f"lift-{o.observation_id}",
            class_name=lbl,
            box_2d=box,
            depth_map=depth_arr,
            camera=cam,
            target_frame="world",
            is_metric_depth=is_metric,
            scale_state=scale_state,
            scale_units=scale_units,
            scale_source=scale_source,
            confidence=o.confidence,
            source_id=src,
        )
        if lifted is None:
            lifted = lifter.lift_detection(
                entity_id=f"lift-{o.observation_id}",
                class_name=lbl,
                box_2d=box,
                depth_map=depth_arr,
                camera=cam,
                target_frame="world",
                is_metric_depth=False,
                scale_state="RELATIVE_GAUGE",
                scale_units="RELATIVE_UNITS",
                confidence=o.confidence,
                source_id=src,
            )
        if lifted is not None:
            drafts = lifter.to_observation_drafts(lifted)
            for d in drafts:
                eng_obs = make_observation(
                    d,
                    engine_id="metric_3d_lifter",
                    model_version="1.0",
                    execution_config={"job_id": job_id, "method": "pca_horizontal_quantile"},
                )
                matrix.store.add(eng_obs)
                matrix.store.link(eng_obs.observation_id, o.observation_id, "lifted_from")
            lifted_count += 1

    return lifted_count


WORLDSTORE_REL = "worldstore"


def persist_world(root: Path, job_id: str, bridge: Any) -> dict[str, Any]:
    """Evidence -> WorldIR -> validated -> immutable WorldStore version -> verified -> reloaded and compared.

    ``status`` is ``persisted`` ONLY when all of that held. Otherwise it is ``not_persisted`` (nothing to persist, e.g. no entities)
    or ``failed`` (invalid world, store error, verification or reload mismatch), always with the reason. The version id is derived
    from the world's content, so persisting the same world twice is idempotent. Nothing is invented: entities, geometry and
    relationships are exactly what the bridge produced from the evidence."""
    import hashlib

    try:
        from world_ir.validation import validate_world_ir
        from world_ir.world_v1 import WorldIR
        from worldstore.store import WorldStore

        if bridge is None or not getattr(bridge, "entities", None):
            return {"status": "not_persisted", "reason": "the evidence produced no canonical entities, so there is no world to persist",
                    "entities": 0, "geometries": 0}
        world = WorldIR(id=f"evw-{job_id}", name=f"evidence world {job_id}")
        bridge.apply_to(world)
        counts = {"entities": len(world.entities), "geometries": len(world.geometries)}
        errors = [str(i) for i in validate_world_ir(world).errors]
        if errors:
            return {"status": "failed", "reason": "the world built from the evidence failed validation and was NOT persisted",
                    "validation_errors": errors[:20], **counts}
        payload = json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        store = WorldStore(root / WORLDSTORE_REL)
        vid = f"v-{digest[:16]}"
        try:
            store.save_version(world, parent=None, version_id=vid)
            reused = False
        except Exception as exc:  # noqa: BLE001 -- an identical, already-saved version is fine; anything else is a real failure
            if "already exists" not in str(exc):
                raise
            reused = True
        problems = store.verify_version(vid)
        if problems:
            return {"status": "failed", "reason": "the saved version failed integrity verification", "verification": problems[:10],
                    "version_id": vid, **counts}
        back = store.load_version(vid)
        if back.to_dict() != world.to_dict():
            return {"status": "failed", "reason": "the reloaded world differs from the one that was saved", "version_id": vid, **counts}
        return {"status": "persisted", "version_id": vid, "sha256": digest, "store": WORLDSTORE_REL, "reused_existing_version": reused,
                "reload_verified": True, "validation": "valid", **counts}
    except Exception as exc:  # noqa: BLE001 -- recorded, never a silent success and never a failed job
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def readiness(root: Path) -> dict[str, Any]:
    """What the frontend may honestly say about one run, derived ONLY from what is persisted on disk (snapshot, evidence matrix,
    persisted world). Nothing is estimated, seeded or defaulted: an unreadable or missing piece is reported as such."""
    out: dict[str, Any] = {"physical_accuracy": "NOT_MEASURED"}
    try:
        snap = json.loads((root / SNAPSHOT_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {**out, "stage_status": "unreadable", "reason": f"snapshot unavailable: {exc}", "export_ready": False}
    out["stage_status"] = snap.get("status")
    out["evidence_received"] = {"sources": len(snap.get("sources", []))}
    out["stages"] = [{"task": s.get("task"), "status": s.get("status"), "engines": s.get("engines", {})} for s in snap.get("stages", [])]
    out["missing_evidence"] = {"capture_requests": len(snap.get("capture_requests", [])), "limitations": list(snap.get("limitations", []))}
    out["refinement"] = [{"stop_reason": h.get("stop_reason"), "new_inference": h.get("new_inference"),
                          "refinement_claim": h.get("refinement_claim")} for h in snap.get("refinement_history", [])]
    # scale status: what the metric-3D records actually say, never a default
    scale: dict[str, int] = {}
    try:
        from engine.multiengine.evidence_matrix import EvidenceMatrix

        m = EvidenceMatrix(root)
        for o in m.store.query(task="metric3d"):
            s = str((o.predicted or {}).get("scale_state", "unknown"))
            scale[s] = scale.get(s, 0) + 1
        out["rejected_predictions"] = len(m.rejected())
    except Exception as exc:  # noqa: BLE001
        out["scale_error"] = f"{type(exc).__name__}: {exc}"
    out["scale"] = {"records_by_scale_state": scale,
                    "physical_metric": "METRIC_ANCHORED" in {k.upper() for k in scale} or "metric_anchored" in scale,
                    "status": "physically_anchored" if "metric_anchored" in scale else ("unanchored" if scale else "no_depth_evidence")}
    # entity / geometry counts and export readiness from the PERSISTED world
    out["world"] = {"status": "not_persisted"}
    out["export_ready"] = False
    try:
        from worldstore.store import WorldStore

        store_root = root / WORLDSTORE_REL
        versions = sorted((store_root / "versions").glob("*.json")) if (store_root / "versions").is_dir() else []
        if versions:
            vid = versions[-1].stem
            store = WorldStore(store_root)
            problems = store.verify_version(vid)
            w = store.load_version(vid)
            placed = [e for e in w.entities.values() if (e.transform or {}).get("position")]
            out["world"] = {"status": "persisted" if not problems else "failed_verification", "version_id": vid,
                            "entities": len(w.entities), "geometries": len(w.geometries), "entities_with_position": len(placed),
                            "verification": problems[:5]}
            out["export_ready"] = bool(placed) and not problems
            if not placed:
                out["export_not_ready_reason"] = "no entity has a measured position yet; exporters would place nothing"
    except Exception as exc:  # noqa: BLE001
        out["world"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return out


def _run_depth(root: Path, job_id: str, sources: Sequence[str], engines: Sequence[str], adapters: Mapping[str, Any]) -> list[dict[str, Any]]:
    from engine.multiengine.evidence_matrix import EvidenceMatrix
    from engine.multiengine.metric3d_producers import (
        checkpoint_sha256,
        hf_checkpoint_file,
        produce_depth,
    )

    matrix = EvidenceMatrix(root)
    reports: list[dict[str, Any]] = []
    for eid in engines:  # sequential: produce_depth unloads each model before the next one loads
        adapter = adapters.get(eid)
        if adapter is None:
            reports.append({"engine_id": eid, "status": "failed", "error": "engine is not available in this deployment", "accepted": [], "rejected": []})
            continue
        ckpt = getattr(adapter, "checkpoint", None)
        path = hf_checkpoint_file(ckpt) if isinstance(ckpt, str) and "/" in ckpt else None
        rep = produce_depth(adapter, sources, matrix=matrix, run_id=job_id, checkpoint_id=str(ckpt or eid), checkpoint_hash=checkpoint_sha256(path))
        rep["accepted"], rep["rejected"] = len(rep["accepted"]), list(rep["rejected"])
        reports.append(rep)
    (root / "depth_report.json").write_text(json.dumps(reports, indent=1, default=str), encoding="utf-8")
    return reports


def _facts(snap: Mapping[str, Any], job_id: str, skipped: int, *, status: str | None = None) -> dict[str, Any]:
    return {
        "status": status or snap["status"], "run_id": job_id, "counts": snap["counts"], "limitations": snap["limitations"],
        "skipped_sources": skipped, "snapshot": SNAPSHOT_REL, "scoring": snap.get("scoring"),
        "identity_events": [e["kind"] for e in snap.get("identity_events", [])],
    }
