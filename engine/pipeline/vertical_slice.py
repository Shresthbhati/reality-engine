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
    #: Reconstruction backend override (tests inject a deterministic
    #: backend; production leaves None for the real COLMAP backend).
    reconstruction_backend: Optional[object] = None


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

        backend = ColmapReconstructionBackend(colmap_binary=options.colmap_binary)
    orchestrator = ReconstructionOrchestrator(backends=[backend])
    try:
        run = orchestrator.run(list(evidence_items))
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim below
        raise VerticalSliceError(f"reconstruction stage failed: {exc}") from exc

    result = run.result
    if result.registration_status == "failed" or not result.points:
        raise VerticalSliceError(
            f"reconstruction produced nothing usable "
            f"(status={result.registration_status!r}, "
            f"{len(result.points)} points) -- not compiling"
        )

    # ---- stage 2: metric scale anchoring (honest RELATIVE fallback) ----
    scaled = None
    scale_error: Optional[str] = None
    for ref in options.measured_baselines:
        try:
            scaled = anchor_metric_scale(result, ref)
            break
        except ScaleAnchoringError:
            continue
    if scaled is not None:
        meters_per_unit = scaled.meters_per_unit
        scale_state = scaled.state
        scale_note = scaled.diagnostics.note
    else:
        rel = unscaled(result)
        result = rel.result
        meters_per_unit = None
        scale_state = rel.state
        scale_note = rel.diagnostics.note
        if options.measured_baselines:
            scale_error = (
                "no measured baseline had both endpoints registered; world "
                "stays RELATIVE (unit-less model distances, not meters)"
            )
            scale_note = scale_note + " | " + scale_error

    # ---- stage 2.5: frame canonicalization (dominant plane -> +Y) ----
    from reconstruction.frame import FrameCanonicalizationError, canonicalize_frame

    try:
        result, frame_record = canonicalize_frame(result, seed=options.seed)
    except FrameCanonicalizationError as exc:
        raise VerticalSliceError(f"frame canonicalization stage failed: {exc}") from exc

    # ---- stage 2.8: depth -> dense metric points (optional, honest skip) ----
    depth_facts, metric_depth_maps = _depth_stage(result, evidence_items, options)

    # ---- stage 3: compile to validated WorldIR ----
    compile_options = CompileOptions(
        seed=options.seed, up=options.up, artifact_store=options.artifact_store
    )
    try:
        world, diagnostics = compile_reconstruction_to_world(result, compile_options)
    except Exception as exc:  # noqa: BLE001
        raise VerticalSliceError(f"world compile stage failed: {exc}") from exc

    # ---- stage 3.5: semantic perception -> object entities (optional) ----
    perception_facts = _perception_stage(
        result, world, evidence_items, metric_depth_maps, options
    )

    # ---- stage 3.6: dense geometry -> surface mesh (optional) ----
    mesh_facts = _mesh_stage(result, world, options, scale_state)

    # ---- stage 4: record the scale state in canonical metadata ----
    world.metadata["scale"] = {
        "state": scale_state,
        "meters_per_unit": meters_per_unit,
        "note": scale_note,
    }
    world.metadata["reconstruction"] = {
        "backend": run.diagnostics.backend_name,
        "registration_status": result.registration_status,
        "cameras_registered": len(result.camera_poses),
        "cameras_input": len(evidence_items),
        "points": len(result.points),
    }
    world.metadata["frame"] = frame_record.to_dict()
    if depth_facts is not None:
        world.metadata["depth"] = depth_facts
    world.metadata["perception"] = perception_facts
    world.metadata["mesh"] = mesh_facts

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
            "scale_error_note": scale_error,
            "depth": depth_facts,
            "perception": perception_facts,
            "mesh": mesh_facts,
        },
    )


def _depth_stage(result, evidence_items, options):
    """Optional MiDaS depth -> metric dense points appended to the result.

    Returns (facts_dict, metric_depth_maps). The maps are also returned so
    the perception stage can lift object masks against the SAME metricized
    maps -- one owner of depth state, never re-inferred. Empty list when
    the stage is disabled/unavailable (a skip is visible, never silent).
    The stage itself never raises: depth is an enhancement, and its
    failure must not lose the sparse reconstruction world.
    """
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
        dense.extend(depth_map_to_points(dm, camera, stride=options.depth_stride))

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

    candidates = merge_hypotheses(hypotheses, options.object_merge_distance_m)

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
        "note": (
            "COCO Mask R-CNN masks lifted via SfM-aligned depth (metric-by-"
            "alignment, approximate); merged by label+proximity; provenance "
            "INFERRED with per-entity evidence ids"
        ),
    }


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
    silently claim meters), or when COLMAP/its poisson_mesher is
    unavailable. Never fabricates a fallback mesh.
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

    points = [tuple(float(c) for c in p.position) for p in result.points]
    if len(points) < 100:
        return {
            "status": "skipped",
            "note": f"only {len(points)} fused points -- too few for "
            "surface reconstruction",
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
