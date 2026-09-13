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

Not in this module (honest gaps, next steps): semantic
detection/segmentation, 2D->3D object lifting, multi-sensor fusion.
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
    depth_facts = _depth_stage(result, evidence_items, options)

    # ---- stage 3: compile to validated WorldIR ----
    compile_options = CompileOptions(seed=options.seed, up=options.up)
    try:
        world, diagnostics = compile_reconstruction_to_world(result, compile_options)
    except Exception as exc:  # noqa: BLE001
        raise VerticalSliceError(f"world compile stage failed: {exc}") from exc

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
        stage_facts={
            "backend": run.diagnostics.backend_name,
            "duration_s": run.diagnostics.duration_s,
            "scale_error_note": scale_error,
            "depth": depth_facts,
        },
    )


def _depth_stage(result, evidence_items, options):
    """Optional MiDaS depth -> metric dense points appended to the result.

    Returns a facts dict for metadata, or None with a note recorded in
    it when the stage is disabled or its optional deps are missing --
    a skip is visible, never silent. The stage itself never raises:
    depth is an enhancement, and its failure must not lose the sparse
    reconstruction world.
    """
    if options.depth_model is None:
        return {"status": "skipped", "note": "disabled (depth_model=None)"}

    try:
        from perception.depth.midAS_backend import MiDaSDepthBackend
        from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose
        from reconstruction.depth_to_points import depth_map_to_points, metricize_all
    except ImportError as exc:
        return {
            "status": "skipped",
            "note": f"optional perception dependencies unavailable: {exc}",
        }
    if options.intrinsics is None:
        return {
            "status": "skipped",
            "note": "no trusted intrinsics (options.intrinsics) -- unprojection "
            "needs a camera model; refusing to guess one",
        }

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
        }

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
    return {
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
