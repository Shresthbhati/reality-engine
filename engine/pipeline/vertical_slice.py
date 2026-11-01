"""Vertical slice: REAL IMAGES -> WorldIR in one deterministic call.

The capture-to-world path the campaigns define as the product's core:

    photos on disk
      -> evidence items (IDs = filenames, trusted intrinsics attached)
      -> ReconstructionOrchestrator (real COLMAP SfM backend selection)
      -> metric scale anchoring (measured camera baseline; METRIC or
         honest RELATIVE -- never invented)
      -> frame canonicalization (dominant plane -> +Y; provenance kept)
      -> [optional] depth stage: MiDaS relative depth per registered
         view, metricized against the sparse cloud (documented
         approximation), unprojected to a dense cloud, plausibility-
         filtered to the camera envelope
      -> compile_reconstruction_to_world (planes -> classification ->
         promotion -> rooms -> validation gate)
      -> WorldIR with the scale state recorded in world.metadata

Every stage is a real module seam that already had its own tests; this
module wires them and carries the honest artifacts forward:

  - registration_status "partial" is preserved (never upgraded)
  - camera poses are camera CENTERS with camera-to-world rotations
    (the backend's parse contract, regression-tested)
  - scale anchoring uses an operator-measured baseline; when the
    reference pair failed registration the anchoring raises
    ScaleAnchoringError and the pipeline surfaces it -- the caller then
    gets an honest RELATIVE world instead of a fake-metric one
  - the depth stage is skipped with a recorded note when its optional
    dependencies are unavailable -- never silently degraded
  - the world's validation gate runs inside the compiler

Semantic perception (optional stage 3.5): a real detector/segmenter
(Mask R-CNN) produces per-view instance masks; masks + metricized depth
+ cameras lift into 3D object hypotheses; hypotheses merge across views
and promote into the compiled world as traceable entities (P0.6-P0.8,
P0.14 convergence). Skipped honestly when the model or depth maps are
unavailable -- never fabricated.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.compiler.world_compiler import (
    CompileDiagnostics,
    CompileOptions,
    compile_reconstruction_to_world,
)
from world_ir.artifact_store import ArtifactStore
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.scale import (
    ScaleAnchoringError,
    ScaleReference,
    ScaleState,
    anchor_metric_scale,
    unscaled,
)

__all__ = [
    "VerticalSliceError",
    "VerticalSliceOptions",
    "VerticalSliceResult",
    "vertical_slice",
]


class VerticalSliceError(ValueError):
    """The capture-to-world path cannot honestly proceed."""


@dataclass(frozen=True)
class VerticalSliceOptions:
    """Pipeline parameters; all deterministic (no clocks, ambient RNG)."""

    seed: int = 42
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: Operator-measured baselines, tried in order; the first whose pair
    #: both registered becomes the metric anchor. Empty -> RELATIVE.
    measured_baselines: Sequence[ScaleReference] = ()
    #: COLMAP binary lookup name / path.
    colmap_binary: str = "colmap"
    #: Trusted camera intrinsics (fx, fy, cx, cy). When set, every
    #: evidence item carries them and COLMAP refinement is pinned.
    intrinsics: Optional[Tuple[float, float, float, float]] = None
    image_size: Tuple[int, int] = (1280, 960)
    #: Depth stage: model type passed to MiDaSDepthBackend. None disables
    #: the stage entirely.
    depth_model: Optional[str] = "DPT_Hybrid"
    depth_stride: int = 16
    #: Perception stage: detector name (currently only "maskrcnn"). None
    #: disables object perception entirely.
    perception_model: Optional[str] = "maskrcnn"
    #: Minimum detector score for a mask to attempt lifting.
    detection_score_threshold: float = 0.5
    #: Cross-view merge distance for object hypotheses (meters).
    object_merge_distance_m: float = 0.5
    #: how per-view object hypotheses become entities: "association" (gated Hungarian, no transitive merges, AMBIGUOUS
    #: kept apart; perception/instances/association.py) or "union_find" (the older label+distance resolver)
    object_resolver: str = "association"
    #: Content-addressed store for real geometry artifacts: the surface
    #: mesh (stage 3.6) and, when given, the promoted planes' inlier
    #: point payloads. None keeps legacy behavior (no real payloads).
    artifact_store: Optional[ArtifactStore] = None
    #: Surface reconstruction stage: fused metric points -> triangle
    #: mesh artifact -> WorldIR geometry (P0.11-P0.13). Disabled with
    #: False; skipped honestly when its dependencies are unavailable.
    mesh_enabled: bool = True
    mesh_voxel_size_m: float = 0.02
    mesh_poisson_depth: int = 10

    #: Dense-MVS stage (P6-01): run COLMAP patch_match_stereo +
    #: stereo_fusion over the sparse model and ingest fused.ply as a
    #: canonical WorldIR POINTCLOUD. OFF by default -- a heavyweight
    #: GPU stage; enabling it is an explicit operator decision. The
    #: three paths below must name the real capture layout (the sparse
    #: model directory COLMAP's mapper wrote).
    dense_mvs_enabled: bool = False
    dense_mvs_image_dir: Optional[Path] = None
    dense_mvs_sparse_model_dir: Optional[Path] = None
    dense_mvs_workspace: Optional[Path] = None
    dense_mvs_use_gpu: Optional[bool] = None
    #: Dense cost limits (None = COLMAP's own defaults: full resolution, up to 20 source views per image). The job
    #: runner sets a fast tier (apps.api.jobs._dense_quality_options); REALITY_DENSE_QUALITY=full restores the defaults.
    dense_max_image_size: Optional[int] = None
    dense_num_src_images: Optional[int] = None
    dense_geom_consistency: bool = True

    #: Detail stage (P7-06): measured quality -> detail discovery ->
    #: ROI work orders -> local refinement, recorded in world metadata
    #: under "detail". Disabled with False; skipped honestly when its
    #: inputs cannot be measured (no cameras) or dependencies are
    #: unavailable.
    detail_enabled: bool = True
    #: None (default) derives the voxel from the world's scale: a metre when the scale is metric, a fraction of the
    #: robust scene extent when it is relative (perception.detail.voxel). A number is used as given.
    detail_voxel_size_m: Optional[float] = None
    #: Reconstruction backend override (tests inject a deterministic
    #: backend; production leaves None for the real COLMAP backend).
    reconstruction_backend: Optional[object] = None
    #: Persistent per-world COLMAP workspace (reconstruction.colmap_session.ColmapSession).
    #: When set, the real COLMAP backend registers new photographs into the previous sparse
    #: model with image_registrator instead of re-solving the whole scene. The CALLER owns
    #: commit()/discard(): the state only advances when the resulting version is adopted.
    colmap_session: Optional[object] = None
    #: The world's current HEAD as a world_delta snapshot (and its conflicts). With a colmap_session and
    #: a HEAD, the backend judges its incremental and full candidates as WORLDS against it
    #: (engine.pipeline.candidate_selection) instead of by registered-camera count.
    head_snapshot: Optional[dict] = None
    head_conflicts: Optional[list] = None
    #: HEAD's recorded canonical-frame record (``world.metadata["frame"]``: rotation raw -> canonical). A candidate that
    #: demonstrably shares HEAD's raw frame reuses it instead of re-estimating 'up' (reconstruction.frame.canonicalize_like_head),
    #: so the world's visible frame does not wander between versions.
    head_frame: Optional[dict] = None
    #: The working unit HEAD's RELATIVE world already uses (``scale.provisional_unit_k``); reused while this solve agrees
    #: with it so successive versions of one world keep one unit (reconstruction.scene_scale).
    head_unit_k: Optional[float] = None
    #: Escalate to dense MVS (Level 3) when the sparse model's MEASURED properties justify it
    #: (engine.pipeline.dense_gate), using the persistent COLMAP session's own images and sparse model; the dense
    #: cloud is then JUDGED against the sparse world and kept only if better (engine.pipeline.dense_judge). The
    #: library default stays off (a GPU-heavy stage a bare caller must ask for); the job runner turns it on through
    #: dense_gate.dense_auto_enabled(), so the product path escalates on evidence. Thresholds are EXPERIMENTAL.
    dense_auto: bool = False

    #: Plan F3: CPU plane-sweep dense geometry (reconstruction.backend.plane_sweep). Off by default because it is a
    #: per-pixel multi-view sweep; when on, it runs only in a METRIC world with registered cameras + trusted
    #: intrinsics, and only accepts pixels with a clear photometric winner (a blank wall yields no points, never a
    #: filled one). Points are added with track_id "plane_sweep:" so fusion treats them as measured dense geometry.
    plane_sweep_enabled: bool = False
    plane_sweep_max_views: int = 4
    plane_sweep_stride: int = 2
    plane_sweep_planes: int = 48


def _refine_unit(result, unit_k: float, unit_facts):
    """RELATIVE worlds only (unit_facts set): once the frame is canonical, refine the working unit so the cameras sit at a
    typical handheld eye height above the detected floor -- a far better prior than viewing distance (measured: the
    viewing-distance unit was ~2.8x too large on real photos, so walls fell under the 'room scale' thresholds). Still a
    prior, never a measurement: the world stays relative. ``base_k`` (pre-refinement) is what the next version reuses."""
    if not unit_facts:
        return result, unit_k, unit_facts
    from perception.geometry.room_box import NOMINAL_EYE_HEIGHT, eye_height_factor
    from reconstruction.scene_scale import rescale

    f = eye_height_factor(result)
    if f is None or not (0.2 < f < 5.0):          # an implausible factor means the 'floor' was not one: keep the unit
        return result, unit_k, dict(unit_facts, base_k=unit_k, eye_height_refined=False)
    return rescale(result, f), unit_k * f, dict(
        unit_facts, base_k=unit_k, unit_k=unit_k * f, eye_height_refined=True, eye_height_factor=f,
        basis=unit_facts["basis"] + f"; refined so camera height above the floor is {NOMINAL_EYE_HEIGHT:g} units "
                                    "(a typical handheld eye height) -- still NOT metres")


def _canonicalize(result, options):
    """Stage 2.5 for both the arbiter's candidate snapshots and the final run: one rule, so they cannot disagree."""
    from reconstruction.frame import canonicalize_like_head

    return canonicalize_like_head(
        result, options.seed, (options.head_frame or {}).get("rotation"), (options.head_snapshot or {}).get("cameras"))


def _candidate_snapshot(result, evidence_items, options) -> dict:
    """A reconstruction candidate as the world snapshot the adoption gate compares: the same scale ->
    frame -> WorldIR compile stages as the real path, without the optional depth/perception/mesh/dense
    stages (identical for both candidates, so the comparison stays symmetric)."""
    from engine.pipeline import world_delta

    scaled = None
    for ref in options.measured_baselines:
        try:
            scaled = anchor_metric_scale(result, ref)
            break
        except ScaleAnchoringError:
            continue
    if scaled is not None:
        result = scaled.result
    else:                          # the same working unit the real run uses, or the two candidates would be judged apart
        from reconstruction.scene_scale import working_unit

        result, _uf = working_unit(unscaled(result).result, options.head_unit_k)
        result, _ = _canonicalize(result, options)
        result, _, _ = _refine_unit(result, _uf["unit_k"] if _uf else 1.0, _uf)
        scaled = True
    if scaled is not True:
        result, _ = _canonicalize(result, options)
    world, _ = compile_reconstruction_to_world(
        result, CompileOptions(seed=options.seed, up=options.up, artifact_store=options.artifact_store))
    registered = [p.evidence_id for p in result.camera_poses]
    prior = (options.head_snapshot or {}).get("input_ids") or []
    level = 2 if len(registered) >= 3 else 1
    if level >= 2 and any(e.type.value in {"room", "storey", "corridor", "level"} for e in world.entities.values()):
        level = 4
    if prior and level >= 2:
        level = 5
    poses = [(p.evidence_id, tuple(float(v) for v in p.position), tuple(float(v) for v in p.rotation))
             for p in result.camera_poses]
    return world_delta.snapshot(
        registered_ids=registered, input_ids=[e.id for e in evidence_items], level=level,
        model_state="PARTIAL", points=len(result.points), camera_poses=poses, world=world,
        artifact_store=options.artifact_store)


@dataclass(frozen=True)
class VerticalSliceResult:
    """Observed outputs of one capture-to-world run."""

    world: object  # WorldIR; kept untyped to avoid an import cycle
    world_id: str
    compile: CompileDiagnostics
    cameras_registered: int
    cameras_input: int
    points_total: int
    registration_status: str
    scale_state: str
    meters_per_unit: Optional[float]
    scale_note: str
    #: Real reconstructed geometry in meters (sparse + dense points), for
    #: artifact persistence (PLY export) and viewers. Tuple of (x, y, z).
    points: Tuple[Tuple[float, float, float], ...] = ()
    #: Registered camera poses in meters as (evidence_id, position, rotation)
    #: with camera-to-world (w, x, y, z) quaternion, for artifact persistence
    #: and camera-frustum visualization.
    camera_poses: Tuple[Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float]], ...] = ()
    #: Per-stage facts for reports and tests.
    stage_facts: Dict[str, object] = field(default_factory=dict)

    def summary_text(self) -> str:
        lines = [
            f"vertical slice: {self.cameras_registered}/{self.cameras_input} "
            f"cameras registered ({self.registration_status}), "
            f"{self.points_total} points, scale={self.scale_state}",
            f"  world {self.world_id}: {len(self.world.entities)} entities, "
            f"{self.compile.measurements_count} measurements, "
            f"{self.compile.relationships_count} relationships",
            f"  scale: {self.scale_note}",
        ]
        return "\n".join(lines)


def vertical_slice(
    evidence_items: Sequence[EvidenceItem],
    options: Optional[VerticalSliceOptions] = None,
) -> VerticalSliceResult:
    """Run photos -> SfM -> scale -> WorldIR. Raises VerticalSliceError
    (wrapping the underlying stage error) when a stage refuses honestly;
    never returns a fabricated world."""
    options = options or VerticalSliceOptions()

    import time as _time

    clock: Dict[str, float] = {}      # seconds per stage, written to the report so slowness is never invisible
    _mark = [_time.perf_counter()]

    def lap(name: str) -> None:
        now = _time.perf_counter()
        clock[name] = round(now - _mark[0], 2)
        _mark[0] = now

    if len(evidence_items) < 2:
        raise VerticalSliceError(
            f"vertical slice needs >= 2 image evidence items, got {len(evidence_items)}"
        )

    if options.intrinsics is not None:
        evidence_items = [
            dataclasses.replace(
                item,
                metadata={
                    **(item.metadata or {}),
                    "intrinsics": {
                        "fx": options.intrinsics[0],
                        "fy": options.intrinsics[1],
                        "cx": options.intrinsics[2],
                        "cy": options.intrinsics[3],
                    },
                },
            )
            for item in evidence_items
        ]

    # ---- stage 1: reconstruction (real COLMAP through the orchestrator) ----
    from reconstruction.orchestrator import ReconstructionOrchestrator

    if options.reconstruction_backend is not None:
        backend = options.reconstruction_backend
    else:
        from reconstruction.backend.colmap_backend import ColmapReconstructionBackend

        backend = ColmapReconstructionBackend(colmap_binary=options.colmap_binary,
                                              session=options.colmap_session)
        if options.colmap_session is not None and options.head_snapshot is not None:
            from engine.pipeline.candidate_selection import WorldArbiter

            backend.candidate_arbiter = WorldArbiter(
                options.head_snapshot, options.head_conflicts,
                lambda res: _candidate_snapshot(res, evidence_items, options))
    orchestrator = ReconstructionOrchestrator(backends=[backend])
    try:
        run = orchestrator.run(list(evidence_items))
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim below
        raise VerticalSliceError(f"reconstruction stage failed: {exc}") from exc

    lap("sparse_reconstruction")
    result = run.result
    if result.registration_status == "failed" or not result.points:
        raise VerticalSliceError(
            f"reconstruction produced nothing usable "
            f"(status={result.registration_status!r}, "
            f"{len(result.points)} points) -- not compiling"
        )

    # ---- stage 2: metric scale anchoring (honest RELATIVE fallback) ----
    scaled = None
    unit_k, unit_facts = 1.0, None     # the RELATIVE working unit (reconstruction.scene_scale); 1.0 = solver units
    scale_error: Optional[str] = None
    device_depth_facts = None
    for ref in options.measured_baselines:
        try:
            scaled = anchor_metric_scale(result, ref)
            break
        except ScaleAnchoringError:
            continue
    if scaled is None:
        # Plan F2: an explicit tape measurement always wins, but a phone LiDAR / structured-light capture already
        # carries metres, so when no baseline registration succeeded the device depth anchors the world. A refused
        # estimate (spread too wide, too few samples, undeclared depth scale) leaves it RELATIVE and is reported.
        estimate, device_depth_facts = _device_depth_scale(result, evidence_items, options)
        if estimate is not None and estimate.accepted:
            from reconstruction.depth_scale import scale_from_depth

            scaled = scale_from_depth(result, estimate)
    if scaled is not None:
        # Adopt the scaled reconstruction itself -- poses, points, and
        # every downstream stage (depth unprojection, perception
        # lifting, meshing, exports) must consume METERS, not the
        # model-unit geometry this branch started with. (Bug fixed
        # 2026-09-14: `result` was never reassigned here, so the world
        # claimed meters while downstream geometry stayed in model
        # units -- masked while tests' model units happened to equal
        # meters.)
        result = scaled.result
        meters_per_unit = scaled.meters_per_unit
        scale_state = scaled.state
        scale_note = scaled.diagnostics.note
    else:
        rel = unscaled(result)
        # COLMAP's scale is arbitrary but every structure threshold downstream is in metres: give a RELATIVE result a
        # working unit so walls and floors are findable (still "relative" -- no metric claim, see scene_scale).
        from reconstruction.scene_scale import working_unit

        result, unit_facts = working_unit(rel.result, options.head_unit_k)
        unit_k = unit_facts["unit_k"] if unit_facts else 1.0
        meters_per_unit = None
        scale_state = rel.state
        scale_note = rel.diagnostics.note + (f" | working unit: {unit_facts['basis']}" if unit_facts else "")
        if options.measured_baselines:
            scale_error = (
                "no measured baseline had both endpoints registered; world "
                "stays RELATIVE (unit-less model distances, not meters)"
            )
            scale_note = scale_note + " | " + scale_error

    # ---- stage 2.5: frame canonicalization (dominant plane -> +Y) ----
    from reconstruction.frame import FrameCanonicalizationError

    try:
        result, frame_record = _canonicalize(result, options)
    except FrameCanonicalizationError as exc:
        raise VerticalSliceError(f"frame canonicalization stage failed: {exc}") from exc
    result, unit_k, unit_facts = _refine_unit(result, unit_k, unit_facts)
    if unit_facts:
        scale_note = scale_note + f" | {unit_facts['basis']}"

    lap("scale_and_frame")
    # ---- stage 2.8: depth -> dense metric points (optional, honest skip) ----
    depth_facts, metric_depth_maps = _depth_stage(result, evidence_items, options, scale_state)

    # ---- stage 2.9: single-image layout evidence (plan F4; projected, lower-confidence wall support) ----
    layout_edges_ab, layout_facts = _layout_evidence_stage(result, evidence_items, options)

    lap("depth_and_layout")
    # ---- stage 3: compile to validated WorldIR ----
    compile_options = CompileOptions(
        seed=options.seed, up=options.up, artifact_store=options.artifact_store,
        layout_edges_ab=layout_edges_ab,
    )
    try:
        world, diagnostics = compile_reconstruction_to_world(result, compile_options)
    except Exception as exc:  # noqa: BLE001
        raise VerticalSliceError(f"world compile stage failed: {exc}") from exc

    lap("compile")
    # ---- stage 3.5: semantic perception -> object entities (optional) ----
    perception_facts = _perception_stage(
        result, world, evidence_items, metric_depth_maps, options
    )

    # ---- stage 3.6: dense geometry -> surface mesh (optional) ----
    mesh_facts = _mesh_stage(result, world, options, scale_state)

    lap("perception_and_mesh")
    # ---- stage 3.65: dense MVS -> fused.ply -> WorldIR pointcloud (P6-01) ----
    # Runs BEFORE fusion so dense points can participate in cross-source fusion
    dense_options, dense_gate_facts, dense_tmp = options, None, None
    if options.dense_auto and not options.dense_mvs_enabled:
        dense_options, dense_gate_facts, dense_tmp = _auto_dense_options(result, len(evidence_items), options)
    try:
        dense_mvs_facts = _dense_mvs_stage(
            result, world, dense_options, scale_state, meters_per_unit,
            frame_rotation=frame_record.rotation,
            judge=dense_gate_facts is not None,          # an AUTOMATIC dense candidate must earn its place
            unit_k=unit_k,
        )
    finally:
        if dense_tmp is not None:
            import shutil

            shutil.rmtree(dense_tmp, ignore_errors=True)       # the fused cloud was ingested into the artifact store

    lap("dense_mvs")
    # ---- stage 3.655: structure from the DENSE cloud (adopted only when it finds strictly more) ----
    structure_facts = None
    if (dense_mvs_facts or {}).get("status") == "ran" and (perception_facts or {}).get("status") in (None, "skipped", "disabled") \
            and (mesh_facts or {}).get("status") in (None, "skipped", "disabled"):
        world, diagnostics, structure_facts = _dense_structure_pass(
            result, world, diagnostics, compile_options, dense_mvs_facts)
    lap("dense_structure")
    # ---- stage 3.66: CPU plane-sweep dense geometry (plan F3) ----
    plane_sweep_facts = _plane_sweep_stage(result, evidence_items, options, scale_state)

    # ---- stage 3.7: plural-source fusion -> WorldIR pointcloud (P6-02) ----
    fusion_facts = _fusion_stage(result, world, options, scale_state)

    lap("plane_sweep_and_fusion")
    # ---- stage 3.8: detail discovery + ROI refinement (P7-06, optional) ----
    detail_facts = _detail_stage(result, world, options, scale_state)
    lap("detail")

    # ---- stage 4: record the scale state in canonical metadata ----
    world.metadata["scale"] = {
        "state": scale_state,
        "meters_per_unit": meters_per_unit,
        "note": scale_note,
        # RELATIVE worlds only: the working unit (displayed units per solver unit) the structure was found in. Never a
        # metric claim; read back by the next version (head_unit_k) and by the scale-reference endpoint.
        "provisional_unit_k": unit_k if unit_facts else None,
        "provisional_unit_base_k": (unit_facts or {}).get("base_k") or (unit_facts or {}).get("unit_k"),
        "provisional_unit": unit_facts,
    }
    # Plan D1: the compile stamped dimensions before the scale existed (RELATIVE caps at RECONSTRUCTED/model_units).
    # Re-stamp now that the unit is known; idempotent, and a tape-verified value is preserved.
    try:
        from world_ir.measure import stamp_world_dimensions
        world.metadata["dimension_stamp"] = stamp_world_dimensions(world)
    except Exception as exc:  # noqa: BLE001 -- never lose the world over a labelling pass
        world.metadata["dimension_stamp"] = {"error": f"{type(exc).__name__}: {exc}"}
    world.metadata["reconstruction"] = {
        "backend": run.diagnostics.backend_name,
        "registration_status": result.registration_status,
        "cameras_registered": len(result.camera_poses),
        "cameras_input": len(evidence_items),
        "points": len(result.points),
    }
    world.metadata["frame"] = frame_record.to_dict()
    world.metadata["layout_evidence"] = layout_facts
    if depth_facts is not None:
        world.metadata["depth"] = depth_facts
    world.metadata["perception"] = perception_facts
    world.metadata["mesh"] = mesh_facts
    world.metadata["fusion"] = {
        k: v for k, v in fusion_facts.items() if k != "fused"
    }

    return VerticalSliceResult(
        world=world,
        world_id=world.id,
        compile=diagnostics,
        cameras_registered=len(result.camera_poses),
        cameras_input=len(evidence_items),
        points_total=len(result.points),
        registration_status=result.registration_status,
        scale_state=scale_state,
        meters_per_unit=meters_per_unit,
        scale_note=scale_note,
        points=tuple(
            (float(p.position[0]), float(p.position[1]), float(p.position[2]))
            for p in result.points
        ),
        camera_poses=tuple(
            (
                p.evidence_id,
                (float(p.position[0]), float(p.position[1]), float(p.position[2])),
                (float(p.rotation[0]), float(p.rotation[1]), float(p.rotation[2]), float(p.rotation[3])),
            )
            for p in result.camera_poses
        ),
        stage_facts={
            "backend": run.diagnostics.backend_name,
            "duration_s": run.diagnostics.duration_s,
            "timings_s": clock,
            "scale_error_note": scale_error,
            # Plan F2: how a device-depth scale was estimated (or why it was refused); None when no device depth exists
            "device_depth_scale": device_depth_facts,
            "layout_evidence": layout_facts,
            "depth": depth_facts,
            "perception": perception_facts,
            "mesh": mesh_facts,
            "detail": detail_facts,
            "dense_mvs": dense_mvs_facts,
            "dense_structure": structure_facts,
            "plane_sweep": plane_sweep_facts,
            "dense_gate": dense_gate_facts,
            # how the sparse model was actually produced (full / incremental / reused, counts, timings);
            # empty for backends without a persistent session
            "colmap_session": dict(getattr(backend, "last_run_info", None) or {}) or None,
        },
    )


def _structure_rank(world, diagnostics) -> Tuple[int, int]:
    """(closed rooms, promoted wall/floor/ceiling entities): what a compile actually understood, rooms first."""
    # CAPPED per kind: a room has one floor, one ceiling and about four walls, so 70 'floor' fragments from furniture
    # tops must not outrank a clean floor plus walls (that rewarded noise, the opposite of understanding the room).
    n = {k: sum(1 for e in world.entities.values() if e.type.value == k) for k in ("wall", "floor", "ceiling")}
    return (int(diagnostics.rooms_detected), min(n["wall"], 4) + min(n["floor"], 1) + min(n["ceiling"], 1))


def _dense_structure_pass(result, world, diagnostics, compile_options, dense_facts, voxel: float = 0.05,
                          cap: int = 12000):
    """Plain walls carry few sparse features, so the sparse compile often finds floors but no closed room. The dense
    cloud covers them: recompile on sparse + a voxel-thinned dense cloud and adopt that world ONLY when it understood
    strictly more (more closed rooms, then more promoted structure). Otherwise the sparse world stands. The dense
    POINTCLOUD geometry the dense stage attached is carried over. Never raises: a failure keeps the sparse world."""
    import numpy as np

    from reconstruction.backend.interface import ReconstructionResult

    try:
        dense = [p for p in result.points if (p.track_id or "").startswith("dense_mvs:")]
        sparse = [p for p in result.points if not (p.track_id or "").startswith("dense_mvs:")]
        if not dense:
            return world, diagnostics, {"status": "skipped", "note": "no dense points"}
        keys = np.floor(np.asarray([p.position for p in dense], dtype=float) / voxel).astype(np.int64)
        idx = np.sort(np.unique(keys, axis=0, return_index=True)[1])
        from perception.geometry.room_box import core_mask

        idx = idx[core_mask(np.asarray([dense[i].position for i in idx], dtype=float))]   # floaters out
        if len(idx) > cap:
            idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
        cand = ReconstructionResult(points=sparse + [dense[i] for i in idx], camera_poses=result.camera_poses,
                                    registration_status=result.registration_status)
        w2, d2 = compile_reconstruction_to_world(cand, compile_options)
        tried = {"ransac": _structure_rank(w2, d2)}
        # the room prior (floor/ceiling/walls from histograms) is a second candidate; the best-understood world wins
        w3, d3 = compile_reconstruction_to_world(cand, dataclasses.replace(compile_options, room_prior=True))
        tried["room_prior"] = _structure_rank(w3, d3)
        # the room prior is the principled candidate (clean floor + outer walls); RANSAC only wins by closing a room
        if tried["room_prior"][1] > 0 and tried["room_prior"][0] >= tried["ransac"][0]:
            w2, d2 = w3, d3
    except Exception as exc:  # noqa: BLE001 -- an enrichment pass must never lose the sparse world
        return world, diagnostics, {"status": "failed", "note": f"{type(exc).__name__}: {exc}"[:300]}
    before, after = _structure_rank(world, diagnostics), _structure_rank(w2, d2)
    facts = {"sparse_rank": before, "dense_rank": after, "candidates": tried, "points_used": len(sparse) + len(idx)}
    if after > before:
        gid = dense_facts.get("geometry_id")
        if gid in world.geometries:
            w2.geometries[gid] = world.geometries[gid]
        return w2, d2, {"status": "adopted", **facts,
                        "note": "the dense cloud understood more structure (rank = closed rooms, promoted structure)"}
    return world, diagnostics, {"status": "kept_sparse", **facts}


def _auto_dense_options(result, n_input: int, options):
    """Decide from MEASURED sparse-model properties whether dense MVS is justified; if so, return options that
    point the existing dense stage at the session workspace's own images and sparse model (a scratch directory
    holds the dense output so the persistent COLMAP state never carries depth maps). Returns
    (options, gate_facts, scratch_dir_or_None). Never guesses: without a usable session workspace the gate says so."""
    import dataclasses
    import tempfile
    from pathlib import Path

    from engine.pipeline.dense_gate import dense_readiness
    from engine.pipeline.guidance import coverage_degrees

    poses = [(p.evidence_id, tuple(float(v) for v in p.position), tuple(float(v) for v in p.rotation))
             for p in result.camera_poses]
    xs = sorted(float(p.position[0]) for p in result.points)
    zs = sorted(float(p.position[2]) for p in result.points)
    center = (xs[len(xs) // 2], zs[len(zs) // 2]) if xs else None
    cov = coverage_degrees(poses, center) if (poses and center) else None
    coverage = cov["degrees"] if cov else None
    facts = dense_readiness(result, n_input, coverage)
    session = options.colmap_session
    ws = Path(session.staging) if session is not None and getattr(session, "staging", None) else None
    model = ws / "sparse" / "0" if ws is not None else None
    if not facts["escalate"]:
        return options, dict(facts, decision="sparse only: dense not justified by the measured evidence"), None
    if model is None or not (ws / "images").is_dir() or not model.is_dir():
        facts.update(escalate=False, reasons=["no persistent COLMAP workspace with images and a sparse model"])
        return options, dict(facts, decision="sparse only: dense inputs unavailable"), None
    from reconstruction.dense_pipeline import dense_mvs_available

    if not dense_mvs_available(options.colmap_binary):
        facts.update(escalate=False, reasons=["dense compute unavailable: COLMAP lacks patch_match_stereo"])
        return options, dict(facts, decision="sparse only: dense compute unavailable"), None
    scratch = Path(tempfile.mkdtemp(prefix="re_dense_"))
    return (dataclasses.replace(options, dense_mvs_enabled=True, dense_mvs_image_dir=ws / "images",
                                dense_mvs_sparse_model_dir=model, dense_mvs_workspace=scratch),
            dict(facts, decision="escalated to dense MVS: every measured criterion passed"), scratch)


#: Detail voxel size for a world with NO metric anchor, as a fraction of the scene's bounding-box diagonal.
#: Measured on the real South Building scene (docs/engineering/DETAIL_CALIBRATION.md): ROI count has a cliff
#: between 2% and 4% of the extent, so 1.5% stays on the detail side of it. EXPERIMENTAL (one dataset).
RELATIVE_DETAIL_VOXEL_FRACTION = 0.015


def _detail_voxel(result, options, scale_state):
    """(voxel_size, basis). An absolute size only means something in meters; in a relative-scale world it is
    derived from the scene's own extent instead of silently assuming 1 unit = 1 m."""
    if scale_state != "relative" or len(result.points) < 2:
        return options.detail_voxel_size_m, "options.detail_voxel_size_m (metric / unspecified scale)"
    lo, hi = _cloud_bounds(result.points)
    diag = sum((h - l) ** 2 for l, h in zip(lo, hi)) ** 0.5
    if not diag > 0:
        return options.detail_voxel_size_m, "options.detail_voxel_size_m (degenerate extent)"
    return (RELATIVE_DETAIL_VOXEL_FRACTION * diag,
            f"{RELATIVE_DETAIL_VOXEL_FRACTION:.1%} of the scene extent (relative-scale world: model units, not meters)")


def _detail_stage(result, world, options, scale_state: str = "unknown"):
    """Optional P7-06 detail chain: measured evidence quality -> detail
    discovery -> ROI work orders -> local refinement (tier-driven,
    honest refusals). Records its facts in world metadata under
    "detail" and returns the facts dict (None = fully skipped).

    Follows the slice's stage pattern: a skip is visible (status
    "skipped" + reason), never silent; the stage never raises -- the
    detail layer is an enhancement and must not lose the sparse world.
    The canonical PinholeCameras built for scale/perception stages are
    reused for the quality measurement (GSD needs real focal lengths;
    the assessor refuses to guess intrinsics). Success records the
    facts into world.metadata["detail"] here (the stage owns its
    metadata key, like the perception stage owns its entities).
    """
    if not options.detail_enabled:
        return {
            "status": "skipped",
            "note": "disabled (detail_enabled=False)",
        }
    if options.intrinsics is None or not result.camera_poses:
        # GSD is unmeasurable without calibrated cameras; the assessor
        # refuses to guess intrinsics (the same gate the sidecar depth
        # stage applies). Visible skip, never fabricated budgets.
        return {
            "status": "skipped",
            "note": "no trusted intrinsics (options.intrinsics) -- evidence "
                    "quality (GSD) is unmeasurable; detail chain skipped",
        }

    from reconstruction.calibration.camera import (
        CameraIntrinsics,
        camera_from_pose,
    )

    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )
    cameras = [
        camera_from_pose(intrinsics, pose)
        for pose in result.camera_poses
    ]

    try:
        from perception.detail.pipeline import run_detail_pipeline
    except ImportError as exc:
        return {
            "status": "skipped",
            "note": f"optional detail dependencies unavailable: {exc}",
        }

    voxel, voxel_basis = _detail_voxel(result, options, scale_state)
    try:
        report = run_detail_pipeline(
            result,
            cameras,
            voxel_size=voxel,
            up=options.up,
            build_world_ir=True,
            world=world,
            scale_state=scale_state,
        )
    except ValueError as exc:
        # Assessor/budget contract refusals are honest, visible skips.
        return {
            "status": "skipped",
            "note": f"detail chain refused: {exc}",
        }

    facts = {
        "status": "ran",
        "quality": report.quality.to_dict(),
        "summary": dict(report.summary),
        "n_candidates": len(report.candidates),
        "voxel_size": voxel,
        "voxel_basis": voxel_basis,
        "budget_units": "model-unit mm/px: GSD level bands are NOT metric" if scale_state == "relative" else "metric",
        "rois": [r.to_dict() for r in report.rois],
        "outcomes": [o.to_dict() for o in report.outcomes],
        "note": (
            "discovery budgets derive from the measured evidence-quality "
            "report; refinement consumes resolvable evidence only -- "
            "refusals are recorded, never retried with invented geometry"
        ),
    }
    world.metadata["detail"] = facts
    return facts


def _depth_stage(result, evidence_items, options, scale_state: str):
    """Optional MiDaS depth -> metric dense points appended to the result.

    Returns (facts_dict, metric_depth_maps). The maps are also returned so
    the perception stage can lift object masks against the SAME metricized
    maps -- one owner of depth state, never re-inferred. Empty list when
    the stage is disabled/unavailable (a skip is visible, never silent).
    The stage itself never raises: depth is an enhancement, and its
    failure must not lose the sparse reconstruction world.

    RGB-D priority (P1.4): when the capture carries real device depth
    sidecars (16-bit PNG per Decision 020), THOSE are unprojected
    directly -- device-metric, no per-view alignment approximation --
    and the MiDaS relative-depth path is not run for matched frames.
    Sidecar depth is coherent with the reconstruction only when the
    world is METRIC-anchored (device meters must not be mixed into a
    unit-less model frame); otherwise the sidecar path reports an
    honest skip and MiDaS handling applies as before.
    """
    # ---- RGB-D sidecar path first (real sensor depth beats estimation) ----
    sidecar_facts = _sidecar_depth_stage(result, evidence_items, options, scale_state)
    if sidecar_facts is not None:
        return sidecar_facts

    if options.depth_model is None:
        return {"status": "skipped", "note": "disabled (depth_model=None)"}, []

    try:
        from perception.depth.midAS_backend import MiDaSDepthBackend
        from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose
        from reconstruction.depth_to_points import depth_map_to_points, metricize_all
    except ImportError as exc:
        return {
            "status": "skipped",
            "note": f"optional perception dependencies unavailable: {exc}",
        }, []
    if options.intrinsics is None:
        return {
            "status": "skipped",
            "note": "no trusted intrinsics (options.intrinsics) -- unprojection "
            "needs a camera model; refusing to guess one",
        }, []

    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )

    try:
        backend = MiDaSDepthBackend(model_type=options.depth_model)
        depth_maps = backend.estimate_depth(list(evidence_items))
    except Exception as exc:  # noqa: BLE001 -- model/dep failures are a skip, not a crash
        return {
            "status": "skipped",
            "note": f"depth backend failed: {exc}",
            "model": options.depth_model,
        }, []

    metric_maps, alignments, failed = metricize_all(depth_maps, result, intrinsics)

    dense: List = []
    for dm in metric_maps:
        camera = None
        for pose in result.camera_poses:
            if pose.evidence_id == dm.evidence_id:
                camera = camera_from_pose(intrinsics, pose)
                break
        if camera is None:
            continue  # metricize_all guarantees a pose; belt and braces
        dense.extend(depth_map_to_points(dm, camera, stride=options.depth_stride,
                                         image_rgb=_photo_rgb_or_none(evidence_items, dm.evidence_id)))

    result.points.extend(dense)
    residuals = sorted(a.residual_median_m for a in alignments)
    median_residual = residuals[len(residuals) // 2] if residuals else None
    facts = {
        "status": "ran",
        "model": options.depth_model,
        "maps": len(depth_maps),
        "metricized": len(metric_maps),
        "metricize_failed": failed,
        "dense_points": len(dense),
        "dense_points_with_colour": sum(1 for p in dense if p.color is not None),
        "stride": options.depth_stride,
        "residual_median_m": median_residual,
        "alignments": [a.to_dict() for a in alignments],
        "note": (
            "relative monocular depth metricized per-view against the SfM "
            "sparse cloud (documented approximation, see "
            "reconstruction.depth_to_points)"
        ),
    }
    return facts, metric_maps


def _perception_stage(result, world, evidence_items, metric_depth_maps, options):
    """Optional semantic perception -> object entities in the world.

    Real detector (Mask R-CNN) -> per-view instance masks -> lifted via the
    metricized depth maps + registered cameras into 3D hypotheses -> merged
    across views -> promoted into `world` as traceable entities. Skipped
    honestly (status recorded, world untouched) when disabled, when the
    model is unavailable, or when no metric depth survived -- object
    entities require metric lifting, and fabricating them would violate
    the pipeline's contract. Never raises: perception is an enhancement;
    its failure must not lose the structural world.
    """
    skip = {"status": "skipped", "objects": 0}
    if options.perception_model is None:
        return {**skip, "note": "disabled (perception_model=None)"}
    if options.intrinsics is None:
        return {**skip, "note": "no trusted intrinsics -- lifting needs a camera model"}
    if not metric_depth_maps:
        return {
            **skip,
            "note": "no metricized depth maps survived -- object lifting needs "
            "metric depth; refusing to lift masks against relative depth",
        }

    try:
        from perception.detection.maskrcnn_backend import MaskRCNNDetector
        from perception.instances.lifting import lift_region_to_3d
        from perception.instances.object_resolution import merge_hypotheses
        from evidence.promote_objects import promote_object_to_entity
        from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose
    except ImportError as exc:
        return {**skip, "note": f"optional perception dependencies unavailable: {exc}"}

    try:
        detector = MaskRCNNDetector(score_threshold=options.detection_score_threshold)
    except Exception as exc:  # noqa: BLE001 -- model unavailability is a skip
        return {**skip, "note": f"detector unavailable: {exc}", "model": options.perception_model}

    # Registered-camera lookup: only poses COLMAP actually registered can
    # lift anything (unregistered views have no pose -- lifting there would
    # invent geometry).
    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    depth_by_id = {dm.evidence_id: dm for dm in metric_depth_maps}

    try:
        seg_results = detector.segment(list(evidence_items))
    except Exception as exc:  # noqa: BLE001
        return {**skip, "note": f"segmentation failed: {exc}"}

    hypotheses = []
    masks_total = 0
    for seg in seg_results:
        depth = depth_by_id.get(seg.evidence_id)
        if depth is None:
            continue  # view's depth never metricized -> cannot lift honestly
        camera = None
        for pose in result.camera_poses:
            if pose.evidence_id == seg.evidence_id:
                camera = camera_from_pose(intrinsics, pose)
                break
        if camera is None:
            continue  # unregistered view
        for region in seg.regions:
            masks_total += 1
            hyp = lift_region_to_3d(region, depth, camera)
            if hyp is not None:
                hypotheses.append(hyp)

    resolution = {"resolver": options.object_resolver}
    if options.object_resolver == "union_find":
        candidates = merge_hypotheses(hypotheses, options.object_merge_distance_m)
    elif options.object_resolver == "association":
        from collections import Counter

        from perception.instances.association import resolve_hypotheses
        from world_ir.observations import DepthSource

        cam_pos = {p.evidence_id: tuple(float(v) for v in p.position) for p in result.camera_poses}
        # depth is SfM-aligned (metric-by-alignment, approximate), so MVS-grade uncertainty, not metric-sensor grade
        candidates, assoc = resolve_hypotheses(hypotheses, DepthSource.MVS, cam_pos, model="mask_rcnn@coco", license="BSD-3-Clause")
        resolution["decisions"] = dict(Counter(d.state.value for d in assoc.decisions))
        resolution["ambiguous_entities"] = sum(1 for e in assoc.entities if e.ambiguous_with)
    else:
        raise VerticalSliceError(f"unknown object_resolver {options.object_resolver!r}: expected 'association' or 'union_find'")

    promoted = 0
    for i, cand in enumerate(candidates):
        promote_object_to_entity(
            cand, world, entity_id=f"entity-object-{i:03d}"
        )
        promoted += 1

    # Final gate: the world must still be valid after mutation.
    from world_ir.validation import validate_world_ir
    report = validate_world_ir(world)
    if report.errors:
        raise VerticalSliceError(
            f"perception stage produced an invalid world: {report.errors[:3]}"
        )

    return {
        "status": "ran",
        "model": options.perception_model,
        "views_segmented": len(seg_results),
        "masks_considered": masks_total,
        "hypotheses_lifted": len(hypotheses),
        "candidates_merged": len(candidates),
        "entities_promoted": promoted,
        "resolution": resolution,
        "note": (
            "COCO Mask R-CNN masks lifted via SfM-aligned depth (metric-by-"
            "alignment, approximate); merged by the gated association resolver (or label+proximity if "
            "object_resolver='union_find'); provenance "
            "INFERRED with per-entity evidence ids"
        ),
    }


def _quat_matrix(quat):
    """Camera-to-world rotation from a (w, x, y, z) quaternion."""
    import numpy as np

    w, x, y, z = (float(v) for v in quat)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _plane_sweep_stage(result, evidence_items, options, scale_state: str):
    """Stage 3.66 (plan F3): CPU plane-sweep dense geometry for textured surfaces.

    The COLMAP dense stage needs a GPU and a COLMAP build with dense support. This stage is the no-CUDA fallback: the
    sparse cloud supplies the depth range (never a guessed default), every registered photo is a reference view, and a
    pixel is kept only where one candidate depth clearly wins the photometric match against the other photos. A plain
    painted wall therefore contributes nothing rather than a fabricated surface; the result is appended to
    ``result.points`` (track ids ``plane_sweep:``) so the normal fusion stage sees it as measured geometry.
    """
    if not options.plane_sweep_enabled:
        return {"status": "disabled", "note": "CPU plane sweep is opt-in (set plane_sweep_enabled=True)"}
    if scale_state != ScaleState.METRIC:
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": f"world scale is {scale_state!r}; a metric depth range cannot be derived from model units"}
    if options.intrinsics is None or not options.image_size:
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": "no trusted intrinsics/image size -- plane sweep cannot warp the views"}

    import numpy as np

    from reconstruction.calibration.camera import CameraIntrinsics

    poses = sorted(result.camera_poses, key=lambda p: p.evidence_id)
    if len(poses) < 2:
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": f"plane sweep needs >=2 registered cameras (have {len(poses)})"}

    # Depth range from the sparse cloud itself: camera-frame z of every registered point. The 2nd/98th percentiles
    # keep a single stray point from spanning the sweep over an absurd range.
    zs = []
    for pose in poses:
        rel = np.asarray([p.position for p in result.points], float) - np.asarray(pose.position, float)
        if not len(rel):
            continue
        z = (rel @ _quat_matrix(pose.rotation))[:, 2]
        zs.extend(z[z > 1e-3].tolist())
    if len(zs) < 10:
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": "too few points in front of the cameras to bound a depth range"}
    lo = max(0.05, float(np.percentile(zs, 2)) * 0.9)
    hi = float(np.percentile(zs, 98)) * 1.1
    if not (hi > lo):
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": f"degenerate depth range from the sparse cloud ({lo:.3f}..{hi:.3f} m)"}

    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    K = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    views, unreadable = [], []
    for pose in poses:
        rgb = _photo_rgb_or_none(evidence_items, pose.evidence_id)
        if rgb is None:
            unreadable.append(pose.evidence_id)
            continue
        views.append({"id": pose.evidence_id, "K": K, "R": tuple(pose.rotation), "C": tuple(pose.position),
                      "image": np.asarray(rgb)})
    cap = max(2, int(options.plane_sweep_max_views))
    views = views[:cap]
    if len(views) < 2:
        return {"status": "skipped", "backend": "plane_sweep_cpu",
                "note": f"only {len(views)} registered photo(s) could be read; unreadable: {unreadable[:5]}"}

    from reconstruction.backend import plane_sweep
    from reconstruction.backend.interface import ReconstructedPoint

    added, per_view = 0, {}
    for ref in views:
        neighbours = [v for v in views if v["id"] != ref["id"]][:cap]
        report = plane_sweep.sweep_view(
            ref, neighbours, lo, hi, planes=int(options.plane_sweep_planes),
            stride=max(1, int(options.plane_sweep_stride)),
        )
        sources = [ref["id"]] + [v["id"] for v in neighbours]
        for k, (xyz, rgb) in enumerate(zip(report["points_world"], report["colors"])):
            result.points.append(ReconstructedPoint(
                position=tuple(float(v) for v in xyz),
                track_id=f"plane_sweep:{ref['id']}:{k}",
                source_evidence_ids=list(sources),
                color=tuple(int(v) for v in rgb),
            ))
        added += len(report["points_world"])
        per_view[ref["id"]] = len(report["points_world"])

    from reconstruction.backend.dense_registry import select_dense_backends
    return {"status": "ran" if added else "refused", "backend": "plane_sweep_cpu", "license": "Apache-2.0 (OpenCV)",
            "runnable_dense_backends": list(select_dense_backends()),
            "views": len(views), "points_added": added, "points_per_view": per_view,
            "depth_range_m": [round(lo, 3), round(hi, 3)], "stride": int(options.plane_sweep_stride),
            "planes": int(options.plane_sweep_planes), "unreadable": unreadable[:5],
            "note": ("dense points accepted only where one candidate depth clearly won the photometric match" if added
                     else "no pixel had a clear depth winner (blank / occluded surfaces); no geometry invented")}


def _fusion_stage(result, world, options, scale_state: str):
    """Stage 3.7 (P6-02 tail): plural-source fusion -> WorldIR
    POINTCLOUD geometry.

    The pipeline's fused cloud already IS plural evidence (sparse
    triangulation + depth unprojection + dense MVS, distinguishable by track_id),
    so the consumer associates and fuses them here, then writes the
    result back as a pointcloud geometry artifact attached to the
    world -- the '-> WorldIR geometry' tail the ledger named as the
    missing integration.

    Gates (honest skips, same discipline as _mesh_stage): RELATIVE
    scale would make unit-less lies; too-few points make
    fusion statistical noise; no artifact_store makes the geometry
    untraceable.
    """
    if options.artifact_store is None:
        return {"status": "skipped", "note": "no artifact_store configured"}
    if scale_state != "metric":
        return {
            "status": "skipped",
            "note": "world is not METRIC-scale -- fusing unit-less "
            "ranges would silently claim meters",
        }
    sparse_points = [
        p for p in result.points
        if p.track_id and not p.track_id.startswith("depth-") and not p.track_id.startswith("dense_mvs:")
    ]
    depth_points = [
        p for p in result.points
        if p.track_id and p.track_id.startswith("depth-")
    ]
    dense_mvs_points = [
        p for p in result.points
        if p.track_id and p.track_id.startswith("dense_mvs:")
    ]
    # Need at least two sources with enough points for meaningful fusion
    sources_with_points = [
        (len(sparse_points), "sparse"),
        (len(depth_points), "depth"),
        (len(dense_mvs_points), "dense_mvs"),
    ]
    sources_with_points = [(n, name) for n, name in sources_with_points if n >= 10]
    if len(sources_with_points) < 2:
        return {
            "status": "skipped",
            "note": f"plural fusion needs >=2 sources with >=10 points each "
            f"(sparse={len(sparse_points)}, depth={len(depth_points)}, dense_mvs={len(dense_mvs_points)}) -- "
            "fewer would make association statistical noise",
        }

    try:
        from reconstruction.fusion.consumer import fuse_pipeline_points
    except ImportError as exc:
        return {"status": "skipped", "note": f"fusion deps unavailable: {exc}"}

    # Association tolerance: the depth points' own uncertainty scale,
    # declared once here (0.05 m default matches the consumer's
    # precision default) -- not inferred from the data per-point.
    facts = fuse_pipeline_points(
        sparse_points=sparse_points,
        depth_points=depth_points,
        dense_mvs_points=dense_mvs_points,
        association_tolerance_m=0.05,
    )

    from provenance import Provenance as _Provenance
    from world_ir import Geometry as _Geometry, GeometryType as _GeometryType
    from world_ir import Observation as _Observation, Vector3 as _Vector3

    fused_estimates = facts["fused"]
    n_pts = len(result.points)
    (mnx, mny, mnz), (mxx, mxy, mxz) = _cloud_bounds(result.points)
    payload = repr({
        "n_points": n_pts,
        "fused_estimates": [
            {"quantity": e.quantity, "value": e.value, "unit": e.unit.value,
             "provenance": e.provenance.value, "precision": e.precision}
            for e in fused_estimates
        ],
    }).encode("utf-8")
    data_uri, data_hash = options.artifact_store.put(payload)
    geometry = _Geometry(
        id="geom-fused-pointcloud",
        type=_GeometryType.POINTCLOUD,
        lod_level=0,
        vertex_count=n_pts,
        data_uri=data_uri,
        data_hash=data_hash,
        bounds_min=_Vector3(x=mnx, y=mny, z=mnz),
        bounds_max=_Vector3(x=mxx, y=mxy, z=mxz),
        provenance=_Provenance.ESTIMATED,
        confidence=0.6,
        observations=[_Observation(
            id="obs-fused-pointcloud",
            sensor_type="multi_source_fusion",
            confidence=0.6,
            metadata={
                "n_sparse": facts["n_sparse"],
                "n_depth": facts["n_depth"],
                "n_dense_mvs": facts["n_dense_mvs"],
                "n_associated": facts["n_associated"],
                "n_conflicts": facts["n_conflicts"],
                "n_passthrough_sparse": facts["n_passthrough_sparse"],
                "n_passthrough_depth": facts["n_passthrough_depth"],
                "n_passthrough_dense_mvs": facts["n_passthrough_dense_mvs"],
                "association_tolerance_m": facts["association_tolerance_m"],
                "note": "per-point range fusion of sparse+depth+dense_mvs claims; "
                "conflicts recorded, never winner-picked",
            },
        )],
    )
    world.geometries[geometry.id] = geometry
    return {
        "status": "ran",
        "n_sparse": facts["n_sparse"],
        "n_depth": facts["n_depth"],
        "n_dense_mvs": facts["n_dense_mvs"],
        "n_associated": facts["n_associated"],
        "n_conflicts": facts["n_conflicts"],
        "n_passthrough_sparse": facts["n_passthrough_sparse"],
        "n_passthrough_depth": facts["n_passthrough_depth"],
        "n_passthrough_dense_mvs": facts["n_passthrough_dense_mvs"],
        "artifact_uri": data_uri,
        "artifact_sha256": data_hash,
        "geometry_id": geometry.id,
    }


def _dense_mvs_stage(result, world, options, scale_state: str, meters_per_unit,
                     frame_rotation=None, judge: bool = False, unit_k: float = 1.0):
    """Stage 3.65 (P6-01): COLMAP dense MVS as a first-class stage.

    The sparse model that stage 1 just produced is fed to COLMAP's dense
    stereo chain (image_undistorter -> patch_match_stereo ->
    stereo_fusion) and the resulting fused.ply is ingested through the
    canonical chain (dense_ingest -> PointCloudData -> content-addressed
    artifact -> WorldIR POINTCLOUD geometry). This is what turns the
    externally verified GPU dense run (2026-09-15) into an engine
    capability instead of a manual CLI ritual.

    Gates (honest skips, same discipline as _fusion_stage/_mesh_stage):
    - OFF by default: a dense run is heavyweight and GPU-bound, so
      enabling it is an explicit operator decision and "disabled" is
      recorded rather than silently absent;
    - the capture paths (images, sparse model, workspace) must be named:
      a dense run cannot be guessed into existence from a result object;
    - the binary must PROBE as dense-capable (patch_match_stereo listed
      in its own help); a COLMAP on PATH is not evidence of that;
    - scale: fused.ply is in COLMAP's own SfM scale. It is rescaled by
      meters_per_unit ONLY when the pipeline established a MEASURED
      metric anchor; otherwise scale_factor=1.0 is used and recorded,
      which the ingest observation names as "not meters" -- never a
      scale lie.

    The dense points are ALSO parsed into ReconstructedPoint objects and
    appended to result.points with track_id prefix "dense_mvs:" so they
    can participate in cross-source fusion (stage 3.7).

    Frame: fused.ply is in COLMAP's raw SfM frame, while ``result`` has already been scaled and rotated into the
    canonical frame; ``frame_rotation`` puts the dense cloud through the same transform (scale, then rotation).

    ``judge=True`` (the automatic, evidence-gated path) makes the dense cloud a CANDIDATE: it is judged against the
    sparse structure (engine.pipeline.dense_judge) BEFORE anything is written, and a rejected or equivalent cloud
    leaves the world and ``result`` exactly as the sparse pipeline produced them. An explicit operator opt-in
    (``dense_mvs_enabled=True``) is not judged.
    """
    if not options.dense_mvs_enabled:
        return {
            "status": "disabled",
            "note": "dense MVS is off by default (heavyweight GPU stage); "
            "set dense_mvs_enabled=True and name the capture paths to run it",
        }
    if options.artifact_store is None:
        return {"status": "skipped", "note": "no artifact_store configured"}

    unset = [
        name
        for name, value in (
            ("dense_mvs_image_dir", options.dense_mvs_image_dir),
            ("dense_mvs_sparse_model_dir", options.dense_mvs_sparse_model_dir),
            ("dense_mvs_workspace", options.dense_mvs_workspace),
        )
        if value is None
    ]
    if unset:
        return {
            "status": "skipped",
            "note": "dense MVS enabled but capture paths unset: "
            + ", ".join(unset),
        }

    try:
        from reconstruction.dense_pipeline import (
            DenseMVSUnavailableError,
            DenseMVSRunError,
            dense_mvs_available,
            run_dense_mvs,
        )
        from reconstruction.dense_ingest import ingest_fused_ply
    except ImportError as exc:  # pragma: no cover - packaging failure
        return {"status": "skipped", "note": f"dense MVS deps unavailable: {exc}"}

    if not dense_mvs_available(options.colmap_binary):
        return {
            "status": "skipped",
            "note": f"COLMAP binary {options.colmap_binary!r} unavailable or "
            "lacks patch_match_stereo (capability probe failed) -- a "
            "synthetic stand-in would be a fake reconstruction",
        }

    source_evidence_ids = [
        eid
        for eid in (getattr(p, "evidence_id", None) for p in result.camera_poses)
        if eid
    ]
    if len(source_evidence_ids) < 2:
        return {
            "status": "skipped",
            "note": "dense MVS needs >=2 registered cameras to rectify and "
            f"stereo-match against (have {len(source_evidence_ids)})",
        }
    try:
        run = run_dense_mvs(
            image_dir=options.dense_mvs_image_dir,
            sparse_model_dir=options.dense_mvs_sparse_model_dir,
            workspace=options.dense_mvs_workspace,
            binary=options.colmap_binary,
            use_gpu=options.dense_mvs_use_gpu,
            max_image_size=options.dense_max_image_size,
            num_src_images=options.dense_num_src_images,
            geom_consistency=options.dense_geom_consistency,
        )
    except (DenseMVSUnavailableError, DenseMVSRunError) as exc:
        return {"status": "failed", "note": f"{type(exc).__name__}: {exc}"}

    # Metric anchor: only a MEASURED metric world may rescale into meters
    # (this is the ingest's own scale-honesty contract).
    if scale_state == "metric" and meters_per_unit:
        scale_factor = float(meters_per_unit)
        scale_note = "rescaled by the pipeline's measured metric anchor"
    else:
        # a RELATIVE world's sparse points were put in the working unit, so the dense cloud must follow them
        scale_factor = float(unit_k)
        scale_note = (
            f"world scale is {scale_state}: the cloud follows the sparse model's working unit "
            f"(x{scale_factor:.6g} of COLMAP's own SfM scale), recorded and never claimed as meters"
        )

    judged = None
    if judge:
        try:
            judged = _judge_dense_candidate(result, run.fused_ply_path, source_evidence_ids, scale_factor,
                                            frame_rotation)
        except Exception as exc:  # noqa: BLE001 -- an unjudgeable candidate is not a justified one
            return {"status": "failed", "note": f"dense candidate could not be judged: {type(exc).__name__}: {exc}"}
        if judged["verdict"] != "accepted":
            return {"status": judged["verdict"], "judge": judged, "note": judged["why"],
                    "n_fused_points": run.n_fused_points}

    try:
        with open(run.fused_ply_path, "rb") as fh:
            fused_bytes = fh.read()
        geometry = ingest_fused_ply(
            fused_bytes,
            artifact_store=options.artifact_store,
            source_evidence_ids=source_evidence_ids,
            scale_factor=scale_factor,
            rotation=frame_rotation,
        )
    except (OSError, ValueError) as exc:
        return {
            "status": "failed",
            "note": f"ingesting {run.fused_ply_path} failed: "
            f"{type(exc).__name__}: {exc}",
        }

    # Parse fused.ply into ReconstructedPoint objects for cross-source fusion
    # Track IDs are prefixed with "dense_mvs:" to distinguish from sparse/depth points
    try:
        from reconstruction.backend.dense_output import parse_fused_ply
        dense_points = parse_fused_ply(
            fused_bytes,
            source_evidence_ids=source_evidence_ids,
        )
        # Prefix track_ids to identify dense MVS points in fusion
        dense_points = _place_points(dense_points, scale_factor, frame_rotation, track_prefix="dense_mvs:")
        result.points.extend(dense_points)
    except Exception:
        # Parsing failure is logged but doesn't fail the stage - geometry is still ingested
        pass

    # The run's own facts (timings, filters, device) are recorded on the
    # geometry's observation so the dense claim stays auditable next to
    # the artifact it produced.
    stage_durations = {k: round(v, 3) for k, v in run.stage_durations_s.items()}
    if geometry.observations:
        geometry.observations[0].metadata["geom_consistency"] = run.geom_consistency
        geometry.observations[0].metadata["use_gpu"] = run.use_gpu
        geometry.observations[0].metadata["stage_durations_s"] = stage_durations
        geometry.observations[0].metadata["scale_state"] = scale_state
    world.geometries[geometry.id] = geometry

    return {
        "status": "ran",
        "n_fused_points": run.n_fused_points,
        "geom_consistency": run.geom_consistency,
        "use_gpu": run.use_gpu,
        "stage_durations_s": stage_durations,
        "scale_factor": scale_factor,
        "scale_note": scale_note,
        "fused_ply_path": run.fused_ply_path,
        "artifact_uri": geometry.data_uri,
        "artifact_sha256": geometry.data_hash,
        "geometry_id": geometry.id,
        "dense_points_added": len(dense_points) if 'dense_points' in locals() else 0,
        "judge": judged,
    }


def _photo_rgb_or_none(evidence_items, evidence_id: str):
    """The view's photo as RGB for sampling dense-point colour, or None when it cannot be read.
    None means those points carry color=None ("colour not measured"); the surface-colour stage then
    reports it instead of inventing a grey."""
    from reconstruction.depth_to_points import DepthToPointsError, load_image_rgb

    item = next((e for e in evidence_items if e.id == evidence_id), None)
    if item is None or not getattr(item, "source_uri", None):
        return None
    try:
        return load_image_rgb(item.source_uri)
    except DepthToPointsError:
        return None


def _place_points(points, scale_factor: float, rotation, track_prefix: Optional[str] = None):
    """Copy ``points`` (COLMAP raw frame) into the world frame: scale, then the canonical-frame rotation.
    ReconstructedPoint is frozen, so re-identifying (``track_prefix``) happens on the copy."""
    import numpy as np

    if not points:
        return list(points)
    xyz = np.asarray([p.position for p in points], dtype=float) * float(scale_factor)
    if rotation is not None:
        xyz = xyz @ np.asarray(rotation, dtype=float).T
    return [dataclasses.replace(p, position=tuple(float(v) for v in q),
                                **({"track_id": f"{track_prefix}{i}"} if track_prefix else {}))
            for i, (p, q) in enumerate(zip(points, xyz))]


def _judge_dense_candidate(result, fused_ply_path, source_evidence_ids, scale_factor: float, rotation):
    """Judge the fused cloud on disk against the sparse SfM structure in ``result`` (same frame and scale)."""
    from engine.pipeline.dense_judge import judge_dense
    from reconstruction.backend.dense_output import parse_fused_ply

    with open(fused_ply_path, "rb") as fh:
        dense = _place_points(parse_fused_ply(fh.read(), source_evidence_ids=list(source_evidence_ids)),
                              scale_factor, rotation)
    sparse = [p.position for p in result.points
              if not (p.track_id or "").startswith(("depth-", "dense_mvs:"))]
    return judge_dense(sparse, [p.position for p in dense])


def _cloud_bounds(points):
    xs = [float(p.position[0]) for p in points]
    ys = [float(p.position[1]) for p in points]
    zs = [float(p.position[2]) for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _mesh_stage(result, world, options, scale_state: str):
    """Stage 3.6: fused metric points -> oriented -> Poisson mesh ->
    real MESH artifact -> WorldIR geometry + entity (P0.11-P0.13).

    `scale_state` is passed in from the scale-anchoring stage (the single
    owner of that fact) -- reading it from world.metadata here would be a
    temporal lie: stage 4 writes that block after this stage runs.

    The cloud is the SAME fused cloud the compiler consumed (sparse +
    metricized depth points): one owner of geometry state. Skips
    honestly -- with an explicit status and reason -- when disabled,
    when the world ended up RELATIVE (meshing unit-less points would
    silently claim meters), when there are too few fused points, or
    when COLMAP/its poisson_mesher is unavailable. Never fabricates a
    fallback mesh.

    Gate order (deliberate, environment-independence matters):
    disabled? -> artifact_store? -> metric? -> POINT COUNT -> meshing
    deps importable? -> COLMAP/poisson_mesher capable? -> execution.
    The point-count gate runs BEFORE the COLMAP capability probe: a
    too-small cloud is a fact about the evidence alone, and checking it
    first means the skip reason is the same on every machine regardless
    of whether COLMAP happens to be installed. (Previously the COLMAP
    probe ran first, so a machine WITH COLMAP installed silently masked
    this ordering -- it fell through to the same "too few" skip either
    way -- while a machine WITHOUT COLMAP would report the wrong reason.)
    """
    if not options.mesh_enabled:
        return {"status": "skipped", "note": "disabled (mesh_enabled=False)"}
    if options.artifact_store is None:
        return {
            "status": "skipped",
            "note": "no artifact_store configured -- a mesh without a "
            "persistent artifact would be untraceable",
        }
    if scale_state != "metric":  # ScaleState.METRIC.value
        return {
            "status": "skipped",
            "note": "world is not METRIC-scale -- meshing unit-less points "
            "would silently claim meters",
        }

    # Input quantity/quality gate BEFORE preprocess viability and backend
    # capability: a too-small cloud is a fact about the evidence, knowable
    # immediately, and independent of whether meshing deps or COLMAP happen
    # to be installed on this machine. Checking backend capability first
    # made the diagnostic (and observed test behavior) depend on which
    # machine ran it -- an environment WITH COLMAP silently masked this
    # ordering bug because it fell through to the same "too few" skip
    # either way; an environment WITHOUT COLMAP would report the wrong
    # reason ("COLMAP unavailable") even when the real, environment-
    # independent reason is "not enough points."
    points = [tuple(float(c) for c in p.position) for p in result.points]
    if len(points) < 100:
        return {
            "status": "skipped",
            "note": f"only {len(points)} fused points -- too few for "
            "surface reconstruction",
        }

    try:
        import numpy as _np  # noqa: F401 -- preprocess imports it; fail fast
        from reconstruction.meshing.preprocess import (
            camera_envelope_filter,
            estimate_oriented_normals,
            statistical_outlier_filter,
            voxel_downsample,
        )
        from reconstruction.meshing.surface import (
            MeshingError,
            MeshingUnavailableError,
            poisson_mesher_available,
            reconstruct_surface,
        )
    except ImportError as exc:
        return {
            "status": "skipped",
            "note": f"meshing dependencies unavailable: {exc}",
        }

    colmap_binary = options.colmap_binary
    if not poisson_mesher_available(colmap_binary):
        return {
            "status": "skipped",
            "note": f"COLMAP binary {colmap_binary!r} unavailable or lacks "
            "poisson_mesher (capability probe failed)",
        }

    centers = [
        tuple(float(c) for c in pose.position) for pose in result.camera_poses
    ]

    try:
        downsampled, _ = voxel_downsample(points, options.mesh_voxel_size_m)
        kept, outlier_facts = statistical_outlier_filter(downsampled)
        # The triangulated sparse points define the plausible scene
        # envelope; depth-derived points beyond it are unprojection
        # artifacts (observed: extents of kilometers from depth noise).
        sparse_points = [
            tuple(float(c) for c in p.position)
            for p in result.points
            if p.track_id and not p.track_id.startswith("depth-")
        ]
        if sparse_points:
            kept, envelope_facts = camera_envelope_filter(
                kept, sparse_points, centers
            )
            outlier_facts["camera_envelope"] = envelope_facts
        if len(kept) < 100:
            return {
                "status": "skipped",
                "note": f"only {len(kept)} points after filtering -- too few",
            }
        normals = estimate_oriented_normals(kept, centers)
        mesh = reconstruct_surface(
            kept,
            normals,
            colmap_binary=colmap_binary,
            depth=options.mesh_poisson_depth,
        )
    except (MeshingUnavailableError, MeshingError, ValueError) as exc:
        return {"status": "failed", "note": f"{type(exc).__name__}: {exc}"}

    from provenance import Provenance as _Provenance
    from world_ir import Entity as _Entity, EntityType as _EntityType
    from world_ir import Geometry as _Geometry, GeometryType as _GeometryType
    from world_ir import Observation as _Observation, Vector3 as _Vector3
    from reconstruction.meshing.mesh import mesh_summary
    from reconstruction.meshing.validation import validate_mesh as _validate_mesh

    # Measure, don't assume: the quality report is computed at production
    # time and recorded on the geometry as metadata. No expected extent is
    # declared at this seam (the operator hasn't stated one), so the
    # bad-scale check is honestly absent from the report -- not guessed.
    _mesh_quality = _validate_mesh(mesh)

    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds()
    data_uri, data_hash = options.artifact_store.put(mesh.to_bytes())
    geometry = _Geometry(
        id="geom-mesh-room",
        type=_GeometryType.MESH,
        lod_level=0,
        vertex_count=len(mesh.vertices),
        triangle_count=len(mesh.faces),
        data_uri=data_uri,
        data_hash=data_hash,
        bounds_min=_Vector3(x=mnx, y=mny, z=mnz),
        bounds_max=_Vector3(x=mxx, y=mxy, z=mxz),
        provenance=_Provenance.RECONSTRUCTED,
        confidence=0.6,
        quality_metrics=_mesh_quality.to_dict(),
        observations=[_Observation(
            id="obs-mesh-room",
            sensor_type="surface_reconstruction",
            confidence=0.6,
            metadata={
                **mesh_summary(mesh),
                "outlier_filter": outlier_facts,
                "input_points_fused": len(points),
                "input_points_meshed": len(kept),
                "voxel_size_m": options.mesh_voxel_size_m,
                "poisson_depth": options.mesh_poisson_depth,
                "colmap_binary": colmap_binary,
                "note": "screened Poisson over camera-oriented depth-fused "
                "cloud (metric-by-alignment); trimmed surface, typically "
                "not watertight",
            },
        )],
    )
    world.geometries[geometry.id] = geometry

    cx, cy, cz = (
        (mnx + mxx) / 2.0,
        (mny + mxy) / 2.0,
        (mnz + mxz) / 2.0,
    )
    entity = _Entity(
        id="entity-mesh-room",
        type=_EntityType.STRUCTURE,
        name="Reconstructed surface mesh",
        transform={"position": {"x": cx, "y": cy, "z": cz}},
        geometry_ids=[geometry.id],
        semantic_labels=["mesh"],
        provenance=_Provenance.RECONSTRUCTED,
        confidence=0.6,
    )
    world.entities[entity.id] = entity

    from world_ir.validation import validate_world_ir
    report = validate_world_ir(world)
    if report.errors:
        # Roll the mesh back out rather than emit an invalid world.
        del world.entities[entity.id]
        del world.geometries[geometry.id]
        return {
            "status": "failed",
            "note": f"mesh produced an invalid world: {report.errors[:3]}",
        }

    return {
        "status": "ran",
        **mesh_summary(mesh),
        "artifact_uri": data_uri,
        "artifact_sha256": data_hash,
        "outlier_filter": outlier_facts,
    }

def _capture_dataset_dirs(evidence_items):
    """Capture roots inferred from evidence source URIs (``.../images/x.jpg`` -> ``...``).

    The dataset loader records ``file:///.../images/x.jpg``, so the dataset root is two levels up from each image.
    url2pathname (not bare urlparse.path) is what makes Windows file URIs work: file:///C:/... -> a drive path.
    """
    dataset_dirs = set()
    for item in evidence_items:
        uri = item.source_uri or ""
        if not uri.startswith("file:"):
            continue
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        images_dir = Path(url2pathname(urlparse(uri).path)).parent
        if images_dir.name.lower() in {"images", "photos", "rgb"}:
            dataset_dirs.add(images_dir.parent)
    return sorted(dataset_dirs)


def _depth_sidecar_frames(evidence_items):
    """Locate + parse the capture's depth sidecars once. Returns (frames_by_evidence, sidecar_count, errors)."""
    dataset_dirs = _capture_dataset_dirs(evidence_items)
    if not dataset_dirs:
        return {}, 0, []

    from evidence.depth_frames import (
        DepthFrameError,
        DepthScaleUnavailable,
        DepthSidecarManifest,
        parse_depth_frames,
    )

    frames_by_evidence = {}
    sidecar_count = 0
    parse_errors = []
    for ds_dir in dataset_dirs:
        depth_dir = ds_dir / "depth"
        if not depth_dir.is_dir():
            continue
        depth_files = sorted(
            str(p) for p in depth_dir.glob("*.png") if not p.name.startswith(".")
        )
        if not depth_files:
            continue
        sidecar_count += len(depth_files)
        manifest = None
        manifest_path = depth_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = DepthSidecarManifest.load(str(manifest_path))
            except DepthFrameError as exc:
                parse_errors.append(f"manifest: {exc}")
                continue
        if manifest is None:
            parse_errors.append(
                f"{manifest_path}: no depth/manifest.json -- depth_scale is "
                "required (raw 16-bit values are never assumed to be meters)"
            )
            continue
        try:
            frames = parse_depth_frames(depth_files, manifest=manifest)
        except (DepthFrameError, DepthScaleUnavailable, OSError) as exc:
            parse_errors.append(str(exc))
            continue
        for frame in frames:
            frames_by_evidence[frame.frame_id] = frame
    return frames_by_evidence, sidecar_count, parse_errors


def _device_depth_scale(result, evidence_items, options):
    """Plan F2: metres-per-model-unit from the capture's own metric depth frames.

    A phone depth frame is already in metres, so the ratio between its depth and the reconstruction's model depth at the
    same pixels is the scale. The sparse points must project into a frame, the frame is resampled to the photo's camera
    model when the sensor stores it at lower resolution (the depth frame covers the same field of view), and the spread
    gate in ``estimate_scale_from_depth`` decides whether it is one rigid scene. Returns (estimate or None, facts).
    """
    frames, sidecar_count, errors = _depth_sidecar_frames(evidence_items)
    if sidecar_count == 0:
        return None, None
    facts = {"source": "sidecar", "frames_parsed": len(frames), "frames_used": 0,
             "resampled_to_image_size": 0, "errors": errors}
    if not frames:
        return None, {**facts, "status": "failed", "note": "depth sidecars exist but none parsed -- see errors"}
    if options.intrinsics is None:
        return None, {**facts, "status": "skipped",
                      "note": "no trusted intrinsics -- device depth cannot be projected onto the model"}
    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    import numpy as np

    from reconstruction.calibration.camera import CameraIntrinsics
    from reconstruction.depth_scale import estimate_scale_from_depth

    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    depth_by_id = {}
    for fid, frame in sorted(frames.items()):
        raw = np.asarray(frame.raw, dtype=np.float32) * float(frame.depth_scale)
        raw[raw <= 0.0] = 0.0
        if (frame.width, frame.height) != (width, height):
            try:
                import cv2

                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
                facts["resampled_to_image_size"] += 1
            except ImportError:
                errors.append(f"{fid}: depth is {frame.width}x{frame.height}, image is {width}x{height}, "
                              "and OpenCV is unavailable to resample it -- frame skipped")
                continue
        depth_by_id[fid] = raw
    if not depth_by_id:
        return None, {**facts, "status": "skipped", "note": "no depth frame matches the photo camera model"}
    estimate = estimate_scale_from_depth(result.camera_poses, result.points, depth_by_id, intrinsics)
    return estimate, {**facts, "status": "accepted" if estimate.accepted else "refused",
                      "frames_used": len(estimate.used_views), "estimate": estimate.to_dict()}


def _layout_evidence_stage(result, evidence_items, options):
    """Stage 2.9 (plan F4): project optional single-image layout sidecars onto the floor plane.

    Returns (edges or None, facts). No learned estimator is bundled, so a capture without ``layout/<stem>.json``
    reports the honest UNAVAILABLE status; supplied evidence is projected through the registered pose and only ever
    supports walls nobody's points saw.
    """
    from perception.architecture.layout_evidence import (
        edges_from_hypotheses,
        estimator_status,
        load_layout_sidecars,
    )

    dirs = _capture_dataset_dirs(evidence_items)
    hypotheses, errors = load_layout_sidecars(dirs, default_size=options.image_size)
    if not hypotheses:
        facts = estimator_status()
        facts.update({"hypotheses": 0, "errors": errors[:10],
                      "note": "" if not dirs else "no layout/<stem>.json sidecars in the capture"})
        return None, facts
    if options.intrinsics is None:
        facts = estimator_status()
        facts.update({"status": "skipped", "hypotheses": len(hypotheses), "errors": errors[:10],
                      "note": "no trusted intrinsics -- a boundary line cannot be projected to the floor plane"})
        return None, facts

    # Floor plane from the canonical cloud (+Y up): the 2nd percentile is a robust floor estimate; a capture without
    # floor points simply shifts the projected edges, which then fail the support band and change nothing.
    import numpy as np

    ys = [float(p.position[1]) for p in result.points]
    if len(ys) < 20:
        facts = estimator_status()
        facts.update({"status": "skipped", "hypotheses": len(hypotheses),
                      "note": f"only {len(ys)} points -- no floor plane can be estimated"})
        return None, facts
    floor_y = float(np.percentile(ys, 2))
    edges, facts = edges_from_hypotheses(result, hypotheses, options.intrinsics, floor_y)
    facts["errors"] = (errors + list(facts.get("errors") or []))[:10]
    return (edges or None), facts


def _sidecar_depth_stage(result, evidence_items, options, scale_state: str):
    """RGB-D priority leg of the depth stage (P1.4).

    Parse the capture's 16-bit PNG depth sidecars (Decision 020) and
    unproject them directly as device-metric depth. Returns None when
    the run carries no sidecar depth at all (the caller then falls
    through to the MiDaS path unchanged); returns a facts dict (with
    no dense points added) when sidecars exist but cannot honestly be
    used, so the skip is visible in the report.

    Rules:
      - Device meters never mix into a RELATIVE/UNKNOWN world: the
        scale gate refuses with the reason rather than inventing a
        conversion.
      - Frames match evidence items by stem (frame_0001.png <->
        img_0001); unmatched frames and frames without a registered
        camera are counted, not silently dropped.
      - Unprojection runs through the same PinholeCamera path as the
        MiDaS leg, producing 'depth-'-prefixed track ids so the mesh
        stage's envelope filter treats sidecar and estimated depth
        identically.
      - Never raises: a DepthFrameError/SensorParseError from one bad
        sidecar is recorded as a failed frame; the stage stays an
        enhancement that cannot lose the sparse world.
    """
    frames_by_evidence, sidecar_count, parse_errors = _depth_sidecar_frames(evidence_items)

    if sidecar_count == 0:
        return None

    if not frames_by_evidence:
        return {
            "status": "failed",
            "source": "sidecar",
            "sidecar_files": sidecar_count,
            "frames_parsed": 0,
            "dense_points": 0,
            "errors": parse_errors,
            "note": "depth sidecars exist but none parsed -- see errors",
        }, []

    if options.intrinsics is None:
        return {
            "status": "skipped",
            "source": "sidecar",
            "sidecar_files": sidecar_count,
            "note": "no trusted intrinsics (options.intrinsics) -- sidecar "
            "unprojection needs a camera model; refusing to guess one",
        }, []

    fx, fy, cx, cy = options.intrinsics
    width, height = options.image_size
    from perception.depth.interface import DepthMap
    from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose
    from reconstruction.depth_to_points import depth_map_to_points

    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )

    # Scale coherence gate: sidecar depth is in meters. Unprojecting it
    # into a unit-less model frame would invent scale (exactly what the
    # depth_to_points contract forbids), so it only runs when the run
    # is METRIC-anchored. The reconstruction result has been rescaled
    # in place since the metric-branch fix (result = scaled.result).
    if scale_state != ScaleState.METRIC:
        return {
            "status": "skipped",
            "source": "sidecar",
            "sidecar_files": sidecar_count,
            "frames_parsed": len(frames_by_evidence),
            "note": (
                f"world scale is {scale_state!r} -- device-metric sidecar "
                "depth cannot be unprojected into a non-metric frame "
                "without inventing scale; anchor metric scale first"
            ),
        }, []

    poses_by_id = {pose.evidence_id: pose for pose in result.camera_poses}
    unmatched_frames = sorted(
        fid for fid in frames_by_evidence if fid not in poses_by_id
    )
    matched = [
        (fid, frames_by_evidence[fid])
        for fid in sorted(frames_by_evidence)
        if fid in poses_by_id
    ]

    # Convert DepthFrame (raw ints + scale) to DepthMap (metric floats).
    # DepthMap's values are list-of-rows of floats; invalid pixels are
    # 0.0 and skipped downstream by the existing non-positive guard.
    metric_maps = []
    for fid, frame in matched:
        if (frame.width, frame.height) != (width, height):
            parse_errors.append(
                f"{frame.source_path}: dimensions {frame.width}x{frame.height} "
                f"do not match image_size {width}x{height} -- skipped"
            )
            continue
        values = [
            [
                (float(v) * frame.depth_scale) if v != frame.invalid_value else 0.0
                for v in row
            ]
            for row in frame.raw
        ]
        metric_maps.append(
            DepthMap(
                evidence_id=fid,
                width=frame.width,
                height=frame.height,
                values=values,
                unit="meters",
                uncertainty=frame.uncertainty,
            )
        )

    dense: List = []
    failed_views = 0
    for dm in metric_maps:
        camera = camera_from_pose(intrinsics, poses_by_id[dm.evidence_id])
        try:
            dense.extend(depth_map_to_points(dm, camera, stride=options.depth_stride,
                                             image_rgb=_photo_rgb_or_none(evidence_items, dm.evidence_id)))
        except Exception as exc:  # noqa: BLE001 -- per-view isolation
            failed_views += 1
            parse_errors.append(f"{dm.evidence_id}: unprojection failed: {exc}")

    result.points.extend(dense)
    return {
        "status": "ran",
        "source": "sidecar",
        "sidecar_files": sidecar_count,
        "frames_parsed": len(frames_by_evidence),
        "frames_matched": len(metric_maps),
        "frames_unmatched": len(unmatched_frames),
        "unmatched_frame_ids": unmatched_frames,
        "frames_dimension_mismatch": sum(1 for e in parse_errors if "do not match image_size" in e),
        "failed_views": failed_views,
        "dense_points": len(dense),
        "dense_points_with_colour": sum(1 for p in dense if p.color is not None),
        "stride": options.depth_stride,
        "errors": parse_errors,
        "note": (
            "device-metric depth from 16-bit PNG sidecars unprojected "
            "directly (Decision 020); no per-view scale alignment needed"
        ),
    }, metric_maps
