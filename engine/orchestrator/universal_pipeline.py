"""Universal Pipeline Orchestrator (Universal Intelligence Expansion Epic E01-1).

Autonomous, single-call pipeline executing the full real-world spatial reconstruction chain:
1. Universal Ingestion & Media Intelligence (EXIF/GPS, keyframe extraction, pHash/dHash dedup)
2. Hardware-Aware Resource Adaptation (CUDA/MPS/CPU profiling)
3. Multi-View Feature Matching & Camera Registration (LightGlue / COLMAP)
4. Metric Scale Anchoring (EXIF intrinsics, known physical entity priors, GNSS)
5. Architectural & Semantic Compilation (Planes -> Classification -> Promotion -> Rooms)
6. Spatial Reasoning & Bilateral Relationship Inference (ATTACHED_TO, RESTS_ON, CONTAINS)
7. Material & Appearance Classification
8. Validation Gate & Consistency Verification
9. Optional Immutable WorldStore Persistence & Universal Export

Principle: RESULTS OVER TECHNOLOGIES.
The user supplies evidence and receives a verified, editable 3D world with zero required
technical configuration.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import os
import shutil
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from engine.compiler.building_envelope import reconstruct_building_envelope
from engine.core.resource_governor import (
    GovernedExecution,
    compute_governor_policy,
    probe_system_resources,
)
from evidence.cad_importer import import_cad_floorplan_file, promote_cad_floorplan
from perception.reasoning.physical_plausibility import validate_physical_plausibility
from perception.style.analyzer import analyze_world_design_style
from planning.completeness import evaluate_scene_completeness
from perception.architecture.shared_walls import deduplicate_shared_walls
from reconstruction.scale.georeferencing import georeference_world_from_gnss
from engine.compiler.world_compiler import (
    CompileDiagnostics,
    CompileOptions,
    compile_reconstruction_to_world,
)
from engine.core.device import select_device
from evidence.importers import import_folder
from evidence.packages import DeterministicPackageBuilder, EvidencePackage
from evidence.session import EvidenceItem
from perception.materials.material_hypotheses import hypothesize_materials
from perception.spatial_reasoning import (
    SpatialInferenceResult,
    infer_spatial_relationships,
)
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.interface import (
    IReconstructionBackend,
    ReconstructionResult,
)
from reconstruction.batch import MemoryBudget
from reconstruction.orchestrator import ReconstructionOrchestrator
from reconstruction.scale import (
    MetricScaleFusionEngine,
    ScaleAnchoringError,
    ScaleDiagnostics,
    ScaledReconstruction,
    ScaleReference,
    ScaleState,
    anchor_metric_scale,
    unscaled,
)
from world_ir.artifact_store import ArtifactStore, MemoryArtifactStore
from world_ir.schema_v1 import EntityType
from worldstore.store import WorldStore
from world_ir.validation import WorldValidationReport, validate_world_ir
from world_ir.world_v1 import WorldIR


class PipelineMode(str, Enum):
    """Autonomous execution mode."""
    AUTO = "auto"          # Autonomous selection based on evidence & hardware
    FAST = "fast"          # Rapid sparse reconstruction for preview
    ACCURATE = "accurate"  # Full multi-view dense fusion & architectural reasoning
    CIVIL = "civil"        # Large-scale outdoor site & infrastructure mode


@dataclass(frozen=True)
class UniversalPipelineOptions:
    """Options for universal pipeline execution."""
    mode: PipelineMode = PipelineMode.AUTO
    session_name: str = "auto-session"
    work_dir: Optional[Path] = None
    scale_reference: Optional[ScaleReference] = None
    seed: int = 42
    enable_spatial_reasoning: bool = True
    enable_materials: bool = True
    enable_fine_grained_perception: bool = True
    enable_validation_gate: bool = True
    max_images: Optional[int] = None
    backends: Optional[Sequence[Any]] = None
    compile_options: Optional[CompileOptions] = None


@dataclass
class UniversalPipelineResult:
    """Container for the output of an autonomous reconstruction run."""
    world: WorldIR
    validation_report: WorldValidationReport
    compile_diagnostics: CompileDiagnostics
    spatial_inference: Optional[SpatialInferenceResult] = None
    evidence_package: Optional[EvidencePackage] = None
    reconstruction_result: Optional[ReconstructionResult] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    scale_state: ScaleState = ScaleState.RELATIVE

    def save_to_store(
        self,
        store_or_path: Union[WorldStore, str, Path],
        parent: Optional[str] = None,
        version_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> str:
        """Persist the reconstructed world atomically into WorldStore."""
        if isinstance(store_or_path, (str, Path)):
            store = WorldStore(Path(store_or_path))
        else:
            store = store_or_path
        stored = store.save_version(self.world, parent=parent, version_id=version_id)
        return stored.version_id

    def export(self, format: str, artifact_store: Optional[ArtifactStore] = None):
        """Export the world to any supported format (IFC, glTF, USD, OBJ, DXF, SVG, PDF)."""
        from sdk import reality
        return reality.export(self.world, format, artifact_store=artifact_store)

    def summary(self) -> str:
        """Human-readable execution summary."""
        entities = len(self.world.entities)
        rooms = sum(1 for e in self.world.entities.values() if e.type == EntityType.ROOM)
        walls = sum(1 for e in self.world.entities.values() if e.type == EntityType.WALL)
        edges = self.spatial_inference.total_edges_created if self.spatial_inference else 0
        runtime = self.telemetry.get("total_runtime_s", 0.0)
        device = self.telemetry.get("device", "cpu")
        valid = self.validation_report.is_valid
        return (
            f"UniversalPipelineResult: {entities} entities ({rooms} rooms, {walls} walls), "
            f"{edges} spatial relationships inferred. Valid={valid}. "
            f"Runtime={runtime:.2f}s on {device}."
        )


class UniversalPipeline:
    """Universal intelligent pipeline coordinator."""

    @classmethod
    def reconstruct(
        cls,
        media: Union[str, Path, Sequence[Union[str, Path, EvidenceItem]], EvidencePackage],
        options: Optional[UniversalPipelineOptions] = None,
    ) -> UniversalPipelineResult:
        """Single-call universal spatial reconstruction."""
        opts = options or UniversalPipelineOptions()
        telemetry: Dict[str, Any] = {"stages": {}}
        started_all = time.perf_counter()
        tracemalloc.start()

        # Step 1: Hardware resource profiling
        hw_device = select_device()
        telemetry["device"] = hw_device.device
        telemetry["cuda_available"] = hw_device.is_gpu
        gov_policy = compute_governor_policy(target_budget_s=30.0)
        telemetry["governor_policy"] = gov_policy.to_dict()

        # Step 2: Working directory setup
        temp_dir_obj = None
        if opts.work_dir is not None:
            work_dir = Path(opts.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="reality_pipeline_")
            work_dir = Path(temp_dir_obj.name)

        try:
            # Step 3: Media Ingestion
            t0 = time.perf_counter()
            evidence_package, evidence_items = cls._ingest_media(media, opts, work_dir)
            telemetry["stages"]["ingest"] = {
                "runtime_s": round(time.perf_counter() - t0, 3),
                "items_count": len(evidence_items),
            }

            if len(evidence_items) < 2:
                raise ValueError(
                    f"Insufficient evidence for reconstruction: need at least 2 items, got {len(evidence_items)}"
                )

            # Step 4: Reconstruction (SfM / Feature Matching / Camera Registration)
            t0 = time.perf_counter()
            recon_result = cls._run_reconstruction(evidence_items, opts, work_dir, hw_device)
            residual = None
            if recon_result.registration_stats:
                residual = recon_result.registration_stats.get("mean_reprojection_error_px")
            elif hasattr(recon_result, "residual_mean_px"):
                residual = getattr(recon_result, "residual_mean_px")
            telemetry["stages"]["reconstruction"] = {
                "runtime_s": round(time.perf_counter() - t0, 3),
                "points_count": len(recon_result.points),
                "cameras_registered": len(recon_result.camera_poses),
                "residual_mean_px": residual,
            }

            # Step 5: Metric Scale Anchoring
            t0 = time.perf_counter()
            scaled_recon, scale_state = cls._anchor_scale(recon_result, opts, evidence_items)
            telemetry["stages"]["scale_anchoring"] = {
                "runtime_s": round(time.perf_counter() - t0, 3),
                "state": scale_state.value if hasattr(scale_state, "value") else str(scale_state),
            }

            # Step 6: Architectural Compilation & Room Boundary Solving
            t0 = time.perf_counter()
            compile_opts = opts.compile_options or CompileOptions(seed=opts.seed)
            world, diagnostics = compile_reconstruction_to_world(scaled_recon.result, options=compile_opts)
            telemetry["stages"]["architectural_compiler"] = {
                "runtime_s": round(time.perf_counter() - t0, 3),
                "planes_detected": diagnostics.planes_total,
                "rooms_detected": diagnostics.rooms_detected,
                "entities_promoted": len(world.entities),
            }

            # Record scale state in world metadata
            world.metadata["scale"] = {
                "state": scale_state.value if hasattr(scale_state, "value") else str(scale_state),
                "mode": opts.mode.value,
            }

            # Building envelope & storeys grouping
            envelope_res = reconstruct_building_envelope(world)
            telemetry["stages"]["building_envelope"] = {
                "storeys_created": envelope_res.total_storeys,
                "total_gfa_m2": round(envelope_res.gross_floor_area_m2, 2),
            }

            # Step 6.5: Fine-Grained Perception & Multi-View Component Recognition (Rule 5 & 6)
            if getattr(opts, "enable_fine_grained_perception", True):
                t0 = time.perf_counter()
                from perception.fine_grained import FineGrainedPerceptionEngine
                fine_engine = FineGrainedPerceptionEngine()
                world, perc_summary = fine_engine.process_observations_and_promote(
                    raw_observations=[],
                    world=world,
                )
                telemetry["stages"]["fine_grained_perception"] = perc_summary.to_dict()

            # Step 7: Material Appearance Hypotheses
            if opts.enable_materials:
                t0 = time.perf_counter()
                cls._enrich_materials(world)
                telemetry["stages"]["materials"] = {
                    "runtime_s": round(time.perf_counter() - t0, 3),
                }

            # Step 8: Spatial Reasoning (Bilateral Relations: ATTACHED_TO, RESTS_ON, CONTAINS)
            spatial_inference = None
            if opts.enable_spatial_reasoning:
                t0 = time.perf_counter()
                spatial_inference = infer_spatial_relationships(world)
                telemetry["stages"]["spatial_reasoning"] = {
                    "runtime_s": round(time.perf_counter() - t0, 3),
                    "edges_created": spatial_inference.total_edges_created,
                }

            # Step 8.1: Design Style Analysis
            style_analysis = analyze_world_design_style(world)
            telemetry["stages"]["design_style"] = {
                "primary_style": style_analysis.primary_style.value,
                "confidence": round(style_analysis.confidence, 3),
            }

            # Step 8.2: Physical Plausibility & Collision Analysis
            plausibility_report = validate_physical_plausibility(world)
            telemetry["stages"]["physical_plausibility"] = {
                "is_plausible": plausibility_report.is_plausible,
                "collisions_count": len(plausibility_report.collisions),
                "floating_count": len(plausibility_report.floating_entities),
                "auto_fix_proposals": len(plausibility_report.fix_proposals),
            }

            # Step 8.3: Completeness Evaluation
            completeness_report = evaluate_scene_completeness(world)
            telemetry["stages"]["completeness"] = {
                "overall_completeness_pct": round(completeness_report.overall_completeness_pct, 1),
                "rooms_evaluated": len(completeness_report.room_reports),
            }

            # Step 8.4: Shared Wall Deduplication & Room Topology Consolidation
            shared_wall_report = deduplicate_shared_walls(world)
            telemetry["stages"]["shared_walls"] = {
                "consolidated_wall_count": shared_wall_report.consolidated_wall_count,
            }

            # Step 8.5: Georeferencing from GNSS if available
            recon_inner = getattr(scaled_recon, 'result', scaled_recon)
            if recon_inner and getattr(recon_inner, 'camera_poses', None) and evidence_items:
                cam_poses = [(p.evidence_id, np.array(p.position)) for p in recon_inner.camera_poses]
                gnss_recs = [
                    (
                        it.id,
                        float(it.metadata["latitude_deg"]),
                        float(it.metadata["longitude_deg"]),
                        float(it.metadata.get("altitude_m", 0.0)),
                    )
                    for it in evidence_items
                    if it.metadata and "latitude_deg" in it.metadata and "longitude_deg" in it.metadata
                ]
                if len(gnss_recs) >= 3:
                    try:
                        geo_res = georeference_world_from_gnss(world, cam_poses, gnss_recs)
                        telemetry["stages"]["georeferencing"] = {
                            "success": geo_res.success,
                            "rmse_total_3d_m": geo_res.diagnostics.rmse_total_3d_m,
                            "inlier_count": len(geo_res.inlier_indices),
                        }
                    except Exception as e:
                        telemetry["stages"]["georeferencing"] = {"error": str(e)}

            # Step 9: Validation Gate
            t0 = time.perf_counter()
            val_report = validate_world_ir(world)
            telemetry["stages"]["validation"] = {
                "runtime_s": round(time.perf_counter() - t0, 3),
                "is_valid": val_report.is_valid,
                "issues_count": len(val_report.issues),
            }

            total_elapsed = time.perf_counter() - started_all
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            telemetry["total_runtime_s"] = round(total_elapsed, 3)
            telemetry["peak_memory_mb"] = round(peak / (1024 * 1024), 2)

            return UniversalPipelineResult(
                world=world,
                validation_report=val_report,
                compile_diagnostics=diagnostics,
                spatial_inference=spatial_inference,
                evidence_package=evidence_package,
                reconstruction_result=scaled_recon,
                telemetry=telemetry,
                scale_state=scale_state,
            )

        finally:
            if temp_dir_obj is not None:
                try:
                    temp_dir_obj.cleanup()
                except Exception:
                    pass

    @classmethod
    def _ingest_media(
        cls,
        media: Union[str, Path, Sequence[Union[str, Path, EvidenceItem]], EvidencePackage],
        opts: UniversalPipelineOptions,
        work_dir: Path,
    ) -> tuple[Optional[EvidencePackage], List[EvidenceItem]]:
        """Ingest raw media into verified EvidenceItems and an optional EvidencePackage."""
        if isinstance(media, EvidencePackage):
            items = media.to_evidence_items()
            if opts.max_images:
                items = items[: opts.max_images]
            return media, items

        builder = DeterministicPackageBuilder(seed=f"pipeline-{opts.seed}")

        # Check if media is a directory path
        if isinstance(media, (str, Path)):
            p = Path(media)
            if p.is_dir():
                import_folder(builder, str(p), on_error="record")
                pkg = builder.build()
                items = pkg.to_evidence_items()
                if opts.max_images:
                    items = items[: opts.max_images]
                return pkg, items
            elif p.is_file():
                # Single file or video container
                from evidence.importers import import_file
                import_file(builder, str(p))
                pkg = builder.build()
                items = pkg.to_evidence_items()
                return pkg, items

        # Sequence of items or paths
        if isinstance(media, Sequence):
            if all(isinstance(x, EvidenceItem) for x in media):
                items = list(media)
                if opts.max_images:
                    items = items[: opts.max_images]
                return None, items
            
            # List of paths
            from evidence.importers import import_file
            for item in media:
                p = Path(item)
                if p.is_file():
                    import_file(builder, str(p))
                elif p.is_dir():
                    import_folder(builder, str(p), on_error="record")
            pkg = builder.build()
            items = pkg.to_evidence_items()
            if opts.max_images:
                items = items[: opts.max_images]
            return pkg, items

        raise ValueError(f"Unsupported media input type: {type(media)}")

    @classmethod
    def _run_reconstruction(
        cls,
        evidence_items: List[EvidenceItem],
        opts: UniversalPipelineOptions,
        work_dir: Path,
        hw_device,
    ) -> ReconstructionResult:
        """Select and run the optimal reconstruction backend."""
        if opts.backends:
            backends = list(opts.backends)
        else:
            # Production candidates only. Test doubles (FakeReconstructionBackend) must be injected
            # explicitly via opts.backends; if COLMAP fails the orchestrator raises rather than
            # letting a stand-in report a result.
            # ponytail: ColmapReconstructionBackend has no `quality` option (the old kwarg raised
            # TypeError on every default run); mode->dense_mvs mapping needs a measured decision.
            backends = [ColmapReconstructionBackend(use_gpu=hw_device.is_gpu)]

        orchestrator = ReconstructionOrchestrator(backends)
        recon_run = orchestrator.run(evidence_items)
        return recon_run.result

    @classmethod
    def _anchor_scale(
        cls,
        recon: ReconstructionResult,
        opts: UniversalPipelineOptions,
        evidence_items: Optional[List[EvidenceItem]] = None,
    ) -> tuple[ScaledReconstruction, ScaleState]:
        """Anchor metric scale via multi-way fusion engine if evidence exists; preserve honest RELATIVE otherwise."""
        engine = MetricScaleFusionEngine(target_error_max_pct=1.5)
        references = [opts.scale_reference] if opts.scale_reference else None

        # Remap filenames or paths to item IDs if needed
        if references and evidence_items:
            name_to_id = {}
            for item in evidence_items:
                name_to_id[item.id] = item.id
                if item.source_uri:
                    name_to_id[Path(item.source_uri).name] = item.id
                    name_to_id[str(item.source_uri)] = item.id
            remapped_refs = []
            for ref in references:
                if hasattr(ref, "evidence_id_a") and hasattr(ref, "evidence_id_b"):
                    id_a = name_to_id.get(ref.evidence_id_a, ref.evidence_id_a)
                    id_b = name_to_id.get(ref.evidence_id_b, ref.evidence_id_b)
                    if id_a != ref.evidence_id_a or id_b != ref.evidence_id_b:
                        ref = dataclasses.replace(ref, evidence_id_a=id_a, evidence_id_b=id_b)
                remapped_refs.append(ref)
            references = remapped_refs

        scaled_result, diag = engine.fuse(
            recon,
            world=None,
            evidence_items=evidence_items,
            references=references,
        )

        if diag.consensus_reached and diag.fused_meters_per_unit is not None:
            scale_diag = ScaleDiagnostics(
                state=ScaleState.METRIC,
                meters_per_unit=diag.fused_meters_per_unit,
                ratio_count=diag.inlier_count,
                ratio_spread=diag.ratio_spread,
                note=diag.notes[0] if diag.notes else "metric scale fused",
                basis=diag.primary_basis,
                dimension_state="MEASURED",
            )
            return ScaledReconstruction(
                result=scaled_result,
                state=ScaleState.METRIC,
                meters_per_unit=diag.fused_meters_per_unit,
                diagnostics=scale_diag,
            ), ScaleState.METRIC

        return unscaled(recon), ScaleState.RELATIVE

    @classmethod
    def _enrich_materials(cls, world: WorldIR) -> None:
        """Enrich world entities with material appearance hypotheses."""
        for ent in world.entities.values():
            props = ent.custom_properties or {}
            color = props.get("color_rgb") or props.get("albedo_rgb")
            kind_str = ent.type.value if hasattr(ent.type, "value") else str(ent.type)
            if color:
                hypos = hypothesize_materials(color, kind=kind_str)
                if hypos:
                    if "hypotheses" not in props:
                        props["hypotheses"] = {}
                    props["hypotheses"]["material"] = hypos
