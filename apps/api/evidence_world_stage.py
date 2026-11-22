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
ENV_DETECTION = "RE_EVIDENCE_WORLD_DETECTION"  # "pipeline" (default for real engines: tiling + sub-part crops + SAM masks) | "legacy"
ENV_LIFT = "RE_EVIDENCE_WORLD_LIFT"  # "spatial" (default: spatial_intelligence package) | "legacy" (box lifter + 3D-IoU association)
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
     "wired into the stage only behind RE_EVIDENCE_WORLD_DEPTH_ENGINES"),
    ("depth_feeds_fusion", True, ("tests/test_lift_stage_e2e.py",),
     "aligned relative depth + registered cameras + sparse SfM points -> position/dimensions observations that join the detection's entity "
     "(SfM gauge units, never metres); proven on a SYNTHETIC scene only -- no real-photo accuracy result exists"),
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
            out[str(path)] = PinholeCamera(k, CameraExtrinsics(Vec3(*pos), Quat(*rot)))  # str: the job's paths may be Path objects
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
    points3d: Sequence[Sequence[float]] | None = None,
    observations: Mapping[str, Mapping[str, Any]] | None = None,
    detection: str | None = None,
    pipeline_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the flow on the job's evidence files and return the compact facts for the job report.

    ``detection`` ("pipeline" | "legacy"; default: the ``RE_EVIDENCE_WORLD_DETECTION`` env, else "pipeline" for the real engines and
    "legacy" when the caller supplies its own ``flow_inputs``) selects how detections are produced. "pipeline" runs the
    ``detection_pipeline`` engine (tiled inference, sub-part crops, class filtering, SAM masks over the REAL registered detector);
    if it cannot be installed the run falls back to the legacy whole-image detectors and says so (status ``partial``).

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
        mode = (detection or os.environ.get(ENV_DETECTION) or ("legacy" if flow_inputs is not None else "pipeline")).strip().lower()
        det_stage, det_note = StageSpec(Task.DETECTION), None
        if mode == "pipeline":
            from engine.multiengine.adapters import pipeline_detection

            try:
                det_stage = StageSpec(Task.DETECTION, only=(pipeline_detection.install(registry, adapters, **dict(pipeline_options or {})),))
            except Exception as exc:  # noqa: BLE001 -- explicit: the run continues on the legacy real detectors and is marked partial
                det_note = f"detection pipeline not installed ({type(exc).__name__}: {exc}); whole-image detectors were used instead"
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

        wanted = list(depth_engines) if depth_engines is not None else [
            e.strip() for e in os.environ.get(ENV_DEPTH_ENGINES, "").split(",") if e.strip()
        ]

        if wanted:  # depth is requested, so detections may carry lifted 3D boxes: associate across views by 3D overlap first
            from engine.multiengine.pair_scoring_3d import MIN_IOU, make_geometry_pair_scorer

            scorer = make_geometry_pair_scorer(scorer)  # pairs without 3D boxes still go to the epipolar scorer (if any)
            scoring_note = {**(scoring_note or {}), "geometry": {"method": "3D box IoU of lifted boxes", "min_iou": MIN_IOU,
                                                                  "caveat": "untuned threshold; same-class objects overlapping in space would merge"}}

        scene_state: dict[str, Any] = {}
        use_spatial = bool(wanted) and os.environ.get(ENV_LIFT, "spatial").strip().lower() != "legacy"
        relations_fn = None
        surfaces_fn = None
        if use_spatial:  # the spatial_intelligence package decides identity across views and supplies relationships
            from engine.multiengine.spatial_stage import make_scene_pair_scorer, make_scene_relations_fn

            scorer = make_scene_pair_scorer(scene_state, scorer)
            relations_fn = make_scene_relations_fn(scene_state)

            def surfaces_fn(matrix: Any):
                from engine.multiengine.surfaces import build_surface_entities

                return build_surface_entities(scene_state.get("aligned") or {}, matrix, job_id,
                                              depth_observations=scene_state.get("aligned_depth_observations"))
            scoring_note = {**(scoring_note or {}), "spatial_intelligence": {"method": "multi-view ray triangulation", "fallback": "3D box IoU"}}

        def _depth_and_lift(matrix: Any) -> dict[str, Any]:
            """Depth (sequential, each model unloaded before the next) then lifting, on the flow's own matrix so refinement and
            entity derivation see the result. Lifting needs depth + registered cameras + the job's sparse SfM points."""
            from engine.multiengine.lift_stage import run_lift_stage

            out: dict[str, Any] = {"status": "skipped", "reason": "no depth engines requested (RE_EVIDENCE_WORLD_DEPTH_ENGINES)"}
            if not wanted:
                return out
            depth = _run_depth(root, job_id, real, wanted, adapters, matrix=matrix)
            if not usable:
                return {**out, "reason": "no registered cameras for these images; depth was produced but nothing can be lifted", "depth": depth}
            if use_spatial:
                from engine.multiengine.spatial_stage import run_spatial_stage

                try:
                    return {**run_spatial_stage(matrix, cameras=usable, points3d=points3d or [], run_id=job_id, sources=real,
                                                observations=observations, state=scene_state), "depth": depth}
                except Exception as exc:  # noqa: BLE001 -- never silent: the fallback is recorded in the facts
                    scene_state.clear()
                    fallback = run_lift_stage(matrix, cameras=usable, points3d=points3d or [], run_id=job_id, sources=real, observations=observations)
                    return {**fallback, "depth": depth, "spatial_error": f"{type(exc).__name__}: {exc}", "engine": "legacy box lifter (spatial stage failed)"}
            lift = run_lift_stage(matrix, cameras=usable, points3d=points3d or [], run_id=job_id, sources=real, observations=observations)
            return {**lift, "depth": depth, "engine": "legacy box lifter"}

        res = run_evidence_flow(
            root, job_id, real, registry=registry, adapters=adapters, stages=[det_stage],
            should_cancel=should_cancel, probe=probe, pair_scorer=scorer, scoring_note=scoring_note,
            identity_registry=IdentityRegistry(world_root(team_id, world_id, base=base) / "identities.json"),
            after_stages=_depth_and_lift, relations_fn=relations_fn, surfaces_fn=surfaces_fn,
        )
    except Exception as exc:  # noqa: BLE001 -- the optional stage must never fail the job; the failure is recorded
        import traceback

        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "run_id": job_id, "skipped_sources": skipped,
                "where": [ln.strip() for ln in traceback.format_exc().splitlines() if ln.lstrip().startswith("File ")][-4:]}
    facts = _facts(res.snapshot, job_id, skipped, status=res.status)
    facts["detection_pipeline"] = _detection_facts(mode, det_stage, det_note, adapters, res.snapshot)
    if facts["detection_pipeline"].get("problems") and facts["status"] == "complete":
        facts["status"] = "partial"  # a missing pipeline / missing masks is never reported as a finished detection stage
    facts["world_ir"] = persist_world(root, job_id, res.bridge)
    if facts["status"] == "complete" and facts["world_ir"]["status"] != "persisted":
        facts["status"] = "partial"  # a flow that finished but whose canonical world was not persisted and verified is not a success
    after = res.snapshot.get("after_stages") or {}
    if after.get("depth") is not None:
        facts["depth"] = after["depth"]
    if after:
        facts["lifting"] = {k: v for k, v in after.items() if k != "depth"}
    if res.snapshot.get("surfaces") is not None:
        facts["surfaces"] = res.snapshot["surfaces"]
    try:  # reproducibility record: a failure to write it is reported, never fatal and never silent
        from engine.multiengine import run_manifest

        man = run_manifest.build(root, job_id, real, facts)
        run_manifest.write(root, man)
        facts["manifest"] = {"file": run_manifest.MANIFEST_FILE, "content_sha256": man["content_sha256"]}
    except Exception as exc:  # noqa: BLE001
        facts["manifest"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return facts


WORLDSTORE_REL = "worldstore"


_CYCLE_RE = re.compile(r"cross_entity_cyclic_support: cyclic vertical support relationship detected between (\S+) and (\S+)")
_DUP_RE = re.compile(r"invariant_6_duplicate_entity: entities (\S+) and (\S+) are the same class and overlap (\d+)%")


def _withhold_unresolved_duplicates(world: Any, max_rounds: int = 20) -> list[dict[str, Any]]:
    """Same-class entities that overlap in space with no merge and no DIFFERENT_ENTITY decision are an UNRESOLVED identity: the
    evidence does not say whether they are one object seen twice (which the association layer could not merge, e.g. because they share
    an image) or two objects. Claiming either would be a guess, and refusing to persist the whole world would hide all the resolved
    geometry. So the entity with LESS evidence (fewer observations; ties by id) of each such pair is withheld from the canonical world
    and recorded in ``world.metadata["withheld_entities"]`` with the pair and overlap; it stays in the run's snapshot and Evidence
    Matrix. Any other validation error is left alone for the caller to report."""
    from world_ir.validation import validate_world_ir

    withheld: list[dict[str, Any]] = []
    for _ in range(max_rounds):
        errs = [str(i) for i in validate_world_ir(world).errors]
        cycles = [m.groups() for m in (_CYCLE_RE.search(e) for e in errs) if m]
        for a, b in cycles:  # a vertical-support cycle is physically impossible: drop the support claims between the pair, keep both entities
            from world_ir.schema_v1 import RelationshipKind as _RK

            for x, y in ((a, b), (b, a)):
                if x in world.entities:
                    world.entities[x].relationships = [r for r in world.entities[x].relationships
                                                       if not (r.target_id == y and r.kind in (_RK.SUPPORTED_BY, _RK.SUPPORTS))]
            world.metadata.setdefault("withheld_relationships", []).append(
                {"entities": [a, b], "reason": "vertical support relationships between them formed a cycle (impossible); claims dropped, entities kept"})
        pairs = [m.groups() for m in (_DUP_RE.search(e) for e in errs) if m]
        if not pairs and not cycles:
            break
        dropped: set[str] = set()
        for a, b, pct in pairs:
            if a in dropped or b in dropped or a not in world.entities or b not in world.entities:
                continue
            loser, keeper = sorted((a, b), key=lambda k: (len(world.entities[k].observations), k))
            dropped.add(loser)
            withheld.append({"entity_id": loser, "kept_entity_id": keeper, "overlap_iou_pct": int(pct),
                             "reason": "same class, overlapping in space, not merged and not declared different: identity unresolved"})
        for eid in dropped:
            ent = world.entities.pop(eid)
            for gid in ent.geometry_ids:
                world.geometries.pop(gid, None)
        for ent in world.entities.values():  # no relationship may point at a withheld entity
            ent.relationships = [r for r in ent.relationships if r.target_id not in dropped]
    if withheld:
        world.metadata["withheld_entities"] = withheld
    return withheld


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
        withheld = _withhold_unresolved_duplicates(world)
        counts = {"entities": len(world.entities), "geometries": len(world.geometries), "withheld_entities": len(withheld)}
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


COMMITTED_VERSION_FILE = "committed_version.json"


def record_committed_version(team_id: str | None, world_id: str, job_id: str, version_id: str, *, base: Path | None = None) -> bool:
    """Record, beside the run's evidence, the committed world version the SAME job adopted (one COLMAP model, one canonical frame, one
    scale state). Returns False when the run has no evidence directory. It is the proof ``propose`` needs that the evidence world and
    the committed world share a frame."""
    root = evidence_root(team_id, world_id, job_id, base=base)
    if not root.is_dir():
        return False
    tmp = root / (COMMITTED_VERSION_FILE + ".tmp")
    tmp.write_text(json.dumps({"job_id": job_id, "committed_version_id": version_id}), encoding="utf-8")
    os.replace(tmp, root / COMMITTED_VERSION_FILE)
    return True


def committed_version(root: Path) -> str | None:
    try:
        return json.loads((root / COMMITTED_VERSION_FILE).read_text(encoding="utf-8")).get("committed_version_id")
    except (OSError, ValueError):
        return None


def head_version(root: Path) -> str | None:
    """The newest persisted version of a run's world, by the store's own sequence (a correction adds a version; file names carry no
    order). ``None`` when nothing is persisted."""
    from worldstore.store import WorldStore

    store_root = root / WORLDSTORE_REL
    if not (store_root / "versions").is_dir():
        return None
    versions = WorldStore(store_root).list_versions()
    return versions[-1].version_id if versions else None


def _detection_readiness(m: Any) -> dict[str, Any]:
    """The detection stage as the PERSISTED evidence shows it (one ``detection_pipeline_summary`` per image). No summary means the
    pipeline did not run for this run; coverage is None (never 0) when no box exists."""
    sums = list(m.store.query(quantity="detection_pipeline_summary"))
    if not sums:
        return {"status": "not_run", "reason": "no detection-pipeline evidence: any detections came from whole-image detectors"}
    stats = [s.predicted.get("execution_stats") or {} for s in sums]
    segs = [s.predicted.get("segmentation") or {} for s in sums]
    boxes, masks = sum(int(x.get("refined_box_count", 0)) for x in stats), sum(int(g.get("masks", 0)) for g in segs)
    return {"status": "ran", "engine": sums[0].engine_id, "inner_detector": sums[0].predicted.get("inner_detector"), "images": len(sums),
            "images_with_detections": sum(1 for x in stats if int(x.get("refined_box_count", 0)) > 0),
            "tiles_processed": sum(int(x.get("tiles_processed", 0)) for x in stats), "subparts_detected": sum(int(x.get("subparts_detected", 0)) for x in stats),
            "raw_candidates": sum(int(x.get("raw_detection_count", 0)) for x in stats), "refined_boxes": boxes, "masks": masks,
            "mask_coverage": round(masks / boxes, 4) if boxes else None,
            "segmentation": "ran" if all(g.get("status") == "ran" for g in segs) else "unavailable",
            "segmentation_reason": next((g.get("reason") for g in segs if g.get("status") != "ran"), None)}


def _coverage(w: Any, placed: Sequence[Any], sized: Sequence[Any]) -> dict[str, Any]:
    """Separately measured coverage of the PERSISTED world: every value is a ``{numerator, denominator}`` pair (never a bare score) so a
    reader sees what was counted. ``None`` denominators mean nothing of that kind exists, which is not the same as 0 covered."""
    from provenance import Provenance
    from world_ir.validation import validate_world_ir

    def frac(num: int, den: int) -> dict[str, Any]:
        return {"covered": num, "of": den, "fraction": round(num / den, 4) if den else None}

    ents = list(w.entities.values())
    objects = [e for e in ents if not str(e.id).startswith("surf-")]

    def images(e: Any) -> set[str]:
        """Source photographs an entity is linked to: explicit image references plus the camera observations it carries (surfaces
        cite the photographs their depth came from through observations, not image references)."""
        refs = {r.uri for r in (getattr(e, "external_references", None) or ()) if r.kind == "image"}
        obs = {o.data_uri or o.frame_id for o in (getattr(e, "observations", None) or ()) if o.sensor_type == "camera" and (o.data_uri or o.frame_id)}
        return obs or refs  # ONE identifier family per entity (a path and a frame id for the same photo must not count twice)

    # competing hypotheses the fusion PRESERVED instead of merging (class / position / dimension disagreements between independent engines)
    conflicted = {e.id: (e.custom_properties.get("evidence") or {}).get("competing_hypotheses") for e in ents}
    conflicted = {k: v for k, v in conflicted.items() if v}
    rep = validate_world_ir(w)
    topo = [i for i in rep.issues if i.code.startswith("topology_")]
    return {
        "spatial_positioning": frac(len(placed), len(ents)),
        "geometry_dimensions": frac(len(sized), len(ents)),
        "multiview_association": frac(sum(1 for e in objects if len(images(e)) >= 2), len(objects)),
        "source_traceability": frac(sum(1 for e in ents if images(e)), len(ents)),
        "provenance_declared": frac(sum(1 for e in ents if e.provenance != Provenance.UNKNOWN), len(ents)),
        "unresolved_conflicts": {"entities": len(conflicted), "hypotheses": sum(len(h) for h in conflicted.values())},
        "topology": {"errors": sum(1 for i in topo if i.severity.value == "error"), "warnings": sum(1 for i in topo if i.severity.value == "warning"),
                     "world_validation_errors": len(rep.errors)},
    }


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
        out["detection"] = _detection_readiness(m)
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
        vid = head_version(root)
        if vid:
            store = WorldStore(store_root)
            problems = store.verify_version(vid)
            w = store.load_version(vid)
            placed = [e for e in w.entities.values() if (e.transform or {}).get("position")]
            out["world"] = {"status": "persisted" if not problems else "failed_verification", "version_id": vid,
                            "entities": len(w.entities), "geometries": len(w.geometries), "entities_with_position": len(placed),
                            "verification": problems[:5]}
            out["export_ready"] = bool(placed) and not problems
            sized = [e for e in placed if any(w.geometries.get(g) is not None for g in e.geometry_ids)]
            states = [str((e.transform or {}).get("scale_state") or "undeclared") for e in placed]
            n = len(w.entities)
            # a world is only "complete" when every entity is positioned AND sized AND the stage itself completed; anything less is
            # reported as partial/empty with the numbers, never as a finished reconstruction
            withheld = list((w.metadata or {}).get("withheld_entities") or [])
            world_state = ("empty" if n == 0 else "partial" if (len(placed) < n or len(sized) < n or withheld or out.get("stage_status") != "complete")
                           else "complete")
            n_surf = sum(1 for k in w.entities if str(k).startswith("surf-"))
            out["world"].update({"object_entities": n - n_surf, "surface_entities": n_surf,
                                 "entities_with_geometry": len(sized), "state": world_state, "withheld_entities": withheld,
                                 "scale_states": {s: states.count(s) for s in sorted(set(states))}})
            out["coverage"] = _coverage(w, placed, sized)
            out["metric_export_ready"] = bool(placed) and not problems and all(s == "metric_anchored" for s in states)
            if placed and not out["metric_export_ready"]:
                out["export_scale_warning"] = ("positions/dimensions are not physically anchored metres (scale states: "
                                               f"{sorted(set(states))}); an export carries unitless/gauge values, not a metric model")
            if not placed:
                out["export_not_ready_reason"] = "no entity has a measured position yet; exporters would place nothing"
    except Exception as exc:  # noqa: BLE001
        out["world"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    try:  # the run's recorded inputs (photograph hashes), manifest and persisted world, re-checked against the disk
        from engine.multiengine import run_manifest

        if (root / run_manifest.MANIFEST_FILE).is_file():
            v = run_manifest.verify(root)
            out["integrity"] = {"ok": bool(v["ok"]), "problems": list(v["problems"])[:5]}
    except Exception as exc:  # noqa: BLE001 -- an unverifiable run is reported as such (integrity stays absent -> not measured)
        out["integrity_error"] = f"{type(exc).__name__}: {exc}"
    from engine.multiengine.quality_gate import assess

    out["quality"] = assess(out)  # READY / PARTIAL / BLOCKED / FAILED, from the measurements above only
    return out


def _depth_with_cache(adapter: Any, sources: Sequence[str], *, matrix: Any, run_id: str, checkpoint_id: str, checkpoint_hash: str | None, cache: Any) -> dict[str, Any]:
    """``produce_depth`` for the images that are not cached; cached images are ingested from the cache WITHOUT loading the model (an
    all-cached engine is never loaded). Caching needs the checkpoint hash: without it every image is a miss and the report says why."""
    from engine.multiengine.depth_cache import file_sha256
    from engine.multiengine.metric3d import Metric3DObservation, ingest
    from engine.multiengine.metric3d_producers import depth_records_from_drafts, produce_depth
    from engine.multiengine.observation import ObservationDraft

    spec = adapter.spec
    keys = {src: cache.key(spec.engine_id, checkpoint_hash, file_sha256(src)) for src in sources}
    hits = {src: cache.get(k) for src, k in keys.items() if k is not None}
    hits = {src: a for src, a in hits.items() if a is not None}
    misses = [src for src in sources if src not in hits]
    info = {"hits": len(hits), "misses": len(misses),
            "disabled_reason": None if checkpoint_hash else "checkpoint hash unknown: depth maps are only cached when the exact weights are known"}
    if hits:
        drafts = [ObservationDraft(task="depth", quantity="depth_map", source_id=s, coordinate_system=f"cam:{s}",
                                   predicted={"array": a, "scale_state": "RELATIVE_UNSCALED", "from_cache": True}) for s, a in hits.items()]
        recs = depth_records_from_drafts(drafts, matrix=matrix, engine_id=spec.engine_id, architecture_id=getattr(spec, "name", spec.engine_id),
                                         checkpoint_id=checkpoint_id, checkpoint_hash=checkpoint_hash)
        cached = ingest(matrix, recs, run_id=run_id, model_version=getattr(spec, "version", "unknown"))
    else:
        cached = {"accepted": [], "rejected": []}
    if misses:
        rep = produce_depth(adapter, misses, matrix=matrix, run_id=run_id, checkpoint_id=checkpoint_id, checkpoint_hash=checkpoint_hash)
        import io

        import numpy as np

        for src in misses:  # remember what the model just produced, keyed by the exact weights + image bytes
            k = keys.get(src)
            if k is None:
                continue
            for o in matrix.store.query(quantity="depth_relative", source_id=src):
                if o.engine_id == spec.engine_id and matrix.status(o.observation_id) != "rejected":
                    rec = Metric3DObservation.from_observation(o)
                    if rec.raw_artifact_ref:
                        cache.put(k, np.load(io.BytesIO(matrix.artifacts.get(rec.raw_artifact_ref)), allow_pickle=False))
    else:
        rep = {"engine_id": spec.engine_id, "status": "succeeded", "error": None, "sources": [], "records": 0, "accepted": [], "rejected": [], "wall_s": 0.0,
               "peak_vram_mb": None, "peak_ram_mb": None, "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_hash, "model_loaded": False}
    rep["accepted"] = len(rep["accepted"]) + len(cached["accepted"])
    rep["rejected"] = list(rep["rejected"]) + list(cached["rejected"])
    rep["cache"] = info
    return rep


def _run_depth(root: Path, job_id: str, sources: Sequence[str], engines: Sequence[str], adapters: Mapping[str, Any],
               matrix: Any | None = None) -> list[dict[str, Any]]:
    from engine.multiengine.depth_cache import DepthCache
    from engine.multiengine.evidence_matrix import EvidenceMatrix
    from engine.multiengine.metric3d_producers import checkpoint_sha256, hf_checkpoint_file

    matrix = matrix if matrix is not None else EvidenceMatrix(root)
    reports: list[dict[str, Any]] = []
    for eid in engines:  # sequential: produce_depth unloads each model before the next one loads
        adapter = adapters.get(eid)
        if adapter is None:
            reports.append({"engine_id": eid, "status": "failed", "error": "engine is not available in this deployment", "accepted": [], "rejected": []})
            continue
        ckpt = getattr(adapter, "checkpoint", None)
        path = hf_checkpoint_file(ckpt) if isinstance(ckpt, str) and "/" in ckpt else None
        ckpt_hash = checkpoint_sha256(path)
        reports.append(_depth_with_cache(adapter, sources, matrix=matrix, run_id=job_id, checkpoint_id=str(ckpt or eid), checkpoint_hash=ckpt_hash,
                                         cache=DepthCache(root.parent / "depth-cache")))
    (root / "depth_report.json").write_text(json.dumps(reports, indent=1, default=str), encoding="utf-8")
    return reports


def _detection_facts(mode: str, stage: Any, note: str | None, adapters: Mapping[str, Any], snap: Mapping[str, Any]) -> dict[str, Any]:
    """What the detection stage actually did: which engine ran, whether tiling / sub-part crops / segmentation executed, and every
    reason it is not a complete detection (``problems``). Read from the adapter that executed, never inferred."""
    if note is not None or mode != "pipeline":
        return {"mode": "legacy", "status": "not_run", "reason": note or "whole-image detectors by request", "problems": [note] if note else []}
    from engine.multiengine.adapters.pipeline_detection import ENGINE_ID

    eng_status = next((s["engines"].get(ENGINE_ID) for s in snap.get("stages", []) if s.get("task") == "detection"), None)
    adapter = adapters.get(ENGINE_ID)
    out: dict[str, Any] = {"mode": "pipeline", "status": eng_status or "unknown", "problems": []}
    if eng_status != "succeeded" or adapter is None or not adapter.last_facts:
        out["problems"].append(f"the detection pipeline did not succeed (engine status: {eng_status})")
        return out
    f = adapter.last_facts
    out.update({k: v for k, v in f.items() if k != "images"})
    out["per_image"] = [{k: i.get(k) for k in ("source", "image_size", "tiles_processed", "subparts_detected", "raw_detection_count",
                                                "filtered_detection_count", "refined_box_count", "masks", "seconds")} for i in f["images"]]
    seg = f["segmentation"]
    if seg["status"] != "ran":
        out["problems"].append(f"segmentation did not run: {seg['reason']}")
    elif seg["sam_failures"]:
        out["problems"].append(f"{seg['sam_failures']} box(es) could not be segmented and carry no mask")
    return out


def _facts(snap: Mapping[str, Any], job_id: str, skipped: int, *, status: str | None = None) -> dict[str, Any]:
    return {
        "status": status or snap["status"], "run_id": job_id, "counts": snap["counts"], "limitations": snap["limitations"],
        "skipped_sources": skipped, "snapshot": SNAPSHOT_REL, "scoring": snap.get("scoring"),
        "identity_events": [e["kind"] for e in snap.get("identity_events", [])],
    }
