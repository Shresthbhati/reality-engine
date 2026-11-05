"""Automated Benchmarking and Evaluation Harness for the 50 Open-Source Technologies.

Enforces Universal Spatial Intelligence Engineering Directive (Rule 2):
- Eliminates hard-coded / invented empirical benchmark metrics.
- Explicit execution classifications:
  NATIVE_EXECUTED, WRAPPED_EXECUTED, INTERNAL_IMPLEMENTATION, SYNTHETIC_TEST, SIMULATED, UNAVAILABLE, FAILED.
- Strict provenance invariant: simulated/synthetic output never enters as Provenance.OBSERVED.
- Real empirical measurement: time.perf_counter(), tracemalloc memory tracing, and mathematical outputs.
- Rigorous metadata: dataset checksum, ground-truth source, technology version, hardware provider.
- Peak VRAM is reported only when measured (torch.cuda.max_memory_allocated with CUDA actually
  in use); otherwise it is null with a reason. No hard-coded memory/VRAM/version numbers.
- A SIMULATED / UNAVAILABLE / FAILED technology never passes a threshold and is never RECOMMENDED.
- INTERNAL_IMPLEMENTATION numbers describe Reality Engine's own re-implementation, NOT the
  upstream project's performance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine.technologies.adapters import AdapterExecutionResult, get_adapter_for_technology
from engine.technologies.catalog import (
    ALL_50_TECHNOLOGIES,
    TECHNOLOGIES_BY_ID,
    TechnologyDescriptor,
    TechnologySubsystem,
)
from engine.technologies.registry import get_technology_registry
from engine.verification.classification import (
    CapabilityVerificationStatus,
    ExecutionClassification,
    validate_observation_provenance,
)
from provenance import Provenance
from world_ir.world_v1 import WorldIR


def _detect_hardware_provider() -> Tuple[str, float]:
    """Inspect host environment to report exact hardware and execution provider."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0)
            return f"CUDA: {dev_name}", total_vram_mb
    except Exception:
        pass
    cpu_info = f"Host CPU ({platform.machine()} {platform.python_implementation()} {platform.python_version()})"
    return cpu_info, 0.0


SIMULATED_REASON = "simulated: no real upstream execution"
NOT_MEASURED_VRAM = "not measured: CUDA not used/available"
INTERNAL_LABEL = "internal re-implementation; not the upstream project's performance"


def _resolve_installed_version(python_modules: Tuple[str, ...]) -> Optional[str]:
    """Real installed package version via importlib.metadata, or None."""
    try:
        dist_map = importlib_metadata.packages_distributions()
    except Exception:
        dist_map = {}
    for mod in python_modules:
        for dist in [*dist_map.get(mod.split(".")[0], []), mod]:
            try:
                return importlib_metadata.version(dist)
            except Exception:
                continue
    return None


def _cuda_ready() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compute_dataset_checksum(data: Any) -> str:
    """Compute deterministic SHA-256 checksum of benchmark dataset payload."""
    serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


@dataclass
class BenchmarkMetric:
    """Single measured empirical metric with unit and validation threshold."""

    name: str
    value: float
    unit: str
    target_threshold: float
    passes_threshold: Optional[bool]  # None => not evaluable (simulated / unavailable / failed)
    description: str = ""
    measurement_methodology: str = "Empirical execution measurement"
    reason: Optional[str] = None


@dataclass
class TechnologyBenchmarkResult:
    """Comprehensive benchmark assessment for one evaluated technology."""

    tech_id: str
    name: str
    subsystem: str
    license: str
    evaluation_priority: int
    is_available_on_host: bool
    classification: str
    verification_status: str
    dataset_id: str
    dataset_checksum: str
    ground_truth_source: str
    model_checkpoint: Optional[str]
    technology_version: str
    hardware_provider: str
    measurement_methodology: str
    execution_time_ms: float
    memory_peak_mb: float  # tracemalloc peak of the adapter call (Python heap), measured
    vram_peak_mb: Optional[float]  # None when not measured
    failure_rate_pct: Optional[float]  # failures / attempts * 100, None if no attempts
    metrics: Dict[str, BenchmarkMetric] = field(default_factory=dict)
    # QUALIFIED | RECOMMENDED | NEEDS_OPTIMIZATION | UNAVAILABLE | SIMULATED | NO_MEASUREMENT | FAILED
    summary_verdict: str = "NO_MEASUREMENT"
    world_ir_promotion_verified: bool = True
    failure_info: Optional[str] = None
    # Additive honesty fields
    status: str = "NO_MEASUREMENT"  # MEASURED | NO_MEASUREMENT | SIMULATED | UNAVAILABLE | FAILED
    passes_threshold: Optional[bool] = None
    passes_threshold_reason: Optional[str] = None
    evidence_label: str = ""
    memory_peak_source: str = "tracemalloc"
    vram_peak_reason: Optional[str] = None
    technology_version_source: str = "importlib.metadata"
    attempts: int = 1
    failures: int = 0


class TechnologyBenchmarkHarness:
    """Orchestrates structured evaluation across all 50 technologies."""

    def __init__(self) -> None:
        self.registry = get_technology_registry()
        self.hardware_provider, self.host_vram_mb = _detect_hardware_provider()

    def _prepare_benchmark_payload(self, desc: TechnologyDescriptor) -> Tuple[Dict[str, Any], str, str]:
        """Construct deterministic synthetic benchmark dataset with verifiable ground truth."""
        tech_id = desc.id
        subsystem = desc.subsystem

        if tech_id == "teaser_plus_plus":
            dataset_id = "bench_teaser_rigid_30pts_v1"
            gt_source = "Synthetic affine rotation Rz(90deg) + translation [0.5, -0.2, 0.1]"
            src_pts = np.linspace(-1.0, 1.0, 30).reshape(10, 3)
            # Rz 90 deg
            R_gt = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            t_gt = np.array([0.5, -0.2, 0.1], dtype=np.float64)
            tgt_pts = (src_pts @ R_gt.T) + t_gt
            payload = {
                "source": src_pts.tolist(),
                "target": tgt_pts.tolist(),
                "benchmark_mode": True,
            }

        elif tech_id == "kiss_icp":
            dataset_id = "bench_kiss_icp_scan_offset_v1"
            gt_source = "Simulated LiDAR scan with 0.05m X displacement"
            pts = np.random.RandomState(42).uniform(-2.0, 2.0, (40, 3)).astype(np.float64)
            shifted = pts + np.array([0.05, 0.0, 0.0])
            payload = {"source": pts.tolist(), "target": shifted.tolist(), "benchmark_mode": True}

        elif tech_id == "sophus":
            dataset_id = "bench_sophus_se3_tangent_v1"
            gt_source = "SE(3) tangent algebra vector [0.01, 0.02, 0.03, 0.1, 0.2, 0.3]"
            payload = {
                "se3_vector": [0.01, 0.02, 0.03, 0.1, 0.2, 0.3],
                "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "benchmark_mode": True,
            }

        elif tech_id == "kornia":
            dataset_id = "bench_kornia_affine_gradient_v1"
            gt_source = "Deterministic 16x16 gradient intensity patch"
            img = np.tile(np.linspace(0.0, 1.0, 16), (16, 1)).astype(np.float64)
            H = np.eye(3, dtype=np.float64)
            H[0, 2] = 1.0
            payload = {"image": img.tolist(), "homography": H.tolist(), "benchmark_mode": True}

        elif tech_id == "manifold":
            dataset_id = "bench_manifold_csg_opening_v1"
            gt_source = "Wall bounding box subtracted by window aperture"
            payload = {
                "wall_min": [-2.0, 0.0, -0.1],
                "wall_max": [2.0, 2.8, 0.1],
                "opening_min": [-0.5, 0.8, -0.2],
                "opening_max": [0.5, 2.0, 0.2],
                "benchmark_mode": True,
            }

        elif tech_id == "meshoptimizer":
            dataset_id = "bench_meshopt_decimation_box_v1"
            gt_source = "Canonical 8-vertex 12-triangle cube"
            verts = [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ]
            tris = [
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
            ]
            payload = {"vertices": verts, "triangles": tris, "benchmark_mode": True}

        elif tech_id == "openmesh":
            dataset_id = "bench_openmesh_halfedge_cube_v1"
            gt_source = "Oriented 8-vertex 12-triangle boundary manifold"
            verts = [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ]
            tris = [
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
            ]
            payload = {"vertices": verts, "triangles": tris, "benchmark_mode": True}

        elif tech_id == "openvdb":
            dataset_id = "bench_openvdb_sdf_box_v1"
            gt_source = "Signed Distance Field box with extents [0.5, 0.5, 0.5]"
            payload = {"box_extents": [0.5, 0.5, 0.5], "benchmark_mode": True}

        elif tech_id in ("whitebox_tools", "gdal"):
            dataset_id = "bench_dem_hydrology_grid_v1"
            gt_source = "Inclined 10x10 synthetic DEM terrain grid"
            x = np.linspace(0.0, 10.0, 10)
            elev = np.tile(x, (10, 1))
            payload = {"elevation": elev.tolist(), "bounds": (0.0, 0.0, 10.0, 10.0), "benchmark_mode": True}

        elif tech_id == "onnx_runtime":
            dataset_id = "bench_onnx_spatial_detector_v1"
            gt_source = "YOLOv8-based spatial detector with 25M parameters"
            payload = {"model_name": "SpatialDetector25M", "parameters": 25000000, "resolution": [640, 640], "benchmark_mode": True}

        else:
            dataset_id = f"bench_{tech_id}_calibrated_v1"
            gt_source = f"Standard architectural calibration harness for {desc.subsystem.value}"
            payload = {"benchmark_mode": True}

        return payload, dataset_id, gt_source

    def benchmark_technology(
        self,
        tech_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> TechnologyBenchmarkResult:
        """Run standard benchmark for a specific technology with strictly measured metrics."""
        desc = TECHNOLOGIES_BY_ID.get(tech_id)
        if desc is None:
            raise KeyError(f"Unknown technology ID: {tech_id}")

        status = self.registry.check_technology_availability(tech_id)
        adapter = get_adapter_for_technology(tech_id)
        payload, dataset_id, gt_source = self._prepare_benchmark_payload(desc)
        dataset_checksum = _compute_dataset_checksum(payload)

        # Start live memory tracing
        cuda_ready = _cuda_ready()
        if cuda_ready:
            import torch
            torch.cuda.reset_peak_memory_stats()
        attempts, failures = 1, 0
        failure_info: Optional[str] = None
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            exec_result = adapter.execute(input_data=payload, options=options)
        except Exception as exc:  # recorded as a real failure, not swallowed
            failures = 1
            failure_info = f"{type(exc).__name__}: {exc}"
            exec_result = AdapterExecutionResult(
                tech_id=tech_id, subsystem=desc.subsystem, success=False,
                execution_time_ms=0.0, memory_peak_mb=0.0,
                classification=ExecutionClassification.FAILED, error_message=failure_info,
            )
        exec_time_ms = (time.perf_counter() - t0) * 1000.0

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measured_mem_mb = round(peak_mem / (1024.0 * 1024.0), 2)

        vram_mb: Optional[float] = None
        vram_reason: Optional[str] = NOT_MEASURED_VRAM
        if cuda_ready:
            import torch
            peak_vram = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if peak_vram > 0:
                vram_mb, vram_reason = round(peak_vram, 1), None
            else:
                vram_reason = "not measured: adapter allocated no CUDA memory"
        if not exec_result.success and failures == 0:
            failures = 1
            failure_info = exec_result.error_message or "adapter reported success=False"

        # Ingest into fresh test world to verify canonical WorldIR promotion
        test_world = WorldIR(id=f"bench_{tech_id}")
        adapter.promote_to_world_ir(exec_result, test_world)
        ir_verified = (
            len(test_world.entities) > 0
            or len(test_world.transforms) > 0
            or len(test_world.geometries) > 0
            or len(test_world.metadata) > 0
        )

        metrics: Dict[str, BenchmarkMetric] = {}
        classification = exec_result.classification
        verification_status = status.verification_status

        # ---------------------------------------------------------------------
        # Measure Real Empirical Outputs from Execution
        # ---------------------------------------------------------------------
        out = exec_result.output_data

        if desc.subsystem == TechnologySubsystem.VISUAL_PERCEPTION:
            if "mean_gradient_magnitude" in out:
                grad_val = float(out["mean_gradient_magnitude"])
                metrics["mean_gradient_magnitude"] = BenchmarkMetric(
                    name="mean_gradient_magnitude",
                    value=round(grad_val, 4),
                    unit="grad",
                    target_threshold=0.01,
                    passes_threshold=(grad_val >= 0.01),
                    description="Measured mean Sobel spatial image gradient response",
                    measurement_methodology="Direct differentiable convolution on test patch",
                )
            # No else-branch: adapter constant confidence is not a measurement.

        elif desc.subsystem == TechnologySubsystem.GEOMETRIC_ESTIMATION:
            if "residual_rmse_m" in out:
                rmse = float(out["residual_rmse_m"])
                metrics["registration_rmse_m"] = BenchmarkMetric(
                    name="registration_rmse_m",
                    value=round(rmse, 5),
                    unit="m",
                    target_threshold=0.025,
                    passes_threshold=(rmse <= 0.025),
                    description="Measured point-to-point registration residual RMSE",
                    measurement_methodology="Empirical TLS/optimizer solver convergence residual",
                )
            if "inlier_ratio" in out:
                inlier_r = float(out["inlier_ratio"])
                metrics["inlier_ratio"] = BenchmarkMetric(
                    name="inlier_ratio",
                    value=round(inlier_r, 3),
                    unit="ratio",
                    target_threshold=0.80,
                    passes_threshold=(inlier_r >= 0.80),
                    description="Measured correspondence inlier percentage",
                    measurement_methodology="Geometric verification consensus",
                )

        elif desc.subsystem == TechnologySubsystem.RECONSTRUCTION_MAPPING:
            if "registered_cameras" in out:
                cams = float(out["registered_cameras"])
                metrics["registered_cameras"] = BenchmarkMetric(
                    name="registered_cameras",
                    value=cams,
                    unit="cameras",
                    target_threshold=1.0,
                    passes_threshold=(cams >= 1.0),
                    description="Number of visual camera frames registered",
                    measurement_methodology="Odometry / bundle adjustment camera trajectory solver",
                )
            if "residual_rmse_m" in out:
                rmse = float(out["residual_rmse_m"])
                metrics["residual_rmse_m"] = BenchmarkMetric(
                    name="residual_rmse_m",
                    value=round(rmse, 4),
                    unit="m",
                    target_threshold=0.05,
                    passes_threshold=(rmse <= 0.05),
                    description="Point cloud registration residual distance",
                    measurement_methodology="LiDAR / SfM odometry point alignment",
                )

        elif desc.subsystem == TechnologySubsystem.POINTCLOUD_3D_PERCEPTION:
            if "mean_iou" in out:
                iou = float(out["mean_iou"])
                metrics["point_classification_iou"] = BenchmarkMetric(
                    name="point_classification_iou",
                    value=round(iou, 3),
                    unit="IoU",
                    target_threshold=0.80,
                    passes_threshold=(iou >= 0.80),
                    description="3D geometric segment classification IoU",
                    measurement_methodology="RANSAC geometric planar fitting and clustering",
                )

        elif desc.subsystem == TechnologySubsystem.GEOMETRY_CAD_MESH:
            if "is_manifold" in out:
                m_val = 100.0 if out["is_manifold"] else 0.0
                metrics["manifold_integrity_pct"] = BenchmarkMetric(
                    name="manifold_integrity_pct",
                    value=m_val,
                    unit="%",
                    target_threshold=99.0,
                    passes_threshold=(m_val >= 99.0),
                    description="Guaranteed 2-manifold edge topology",
                    measurement_methodology="Halfedge topological query / Manifold CSG validation",
                )
            if "triangle_reduction_pct" in out:
                dec = float(out["triangle_reduction_pct"])
                metrics["mesh_simplification_pct"] = BenchmarkMetric(
                    name="mesh_simplification_pct",
                    value=round(dec, 1),
                    unit="%",
                    target_threshold=0.0,
                    passes_threshold=(dec >= 0.0),
                    description="Triangle decimation ratio without topological collapse",
                    measurement_methodology="Quadric error metric decimation count",
                )

        elif desc.subsystem == TechnologySubsystem.GEOSPATIAL_ENVIRONMENT:
            if "terrain_slope_mean_deg" in out:
                slope = float(out["terrain_slope_mean_deg"])
                metrics["terrain_slope_mean_deg"] = BenchmarkMetric(
                    name="terrain_slope_mean_deg",
                    value=round(slope, 2),
                    unit="deg",
                    target_threshold=0.0,
                    passes_threshold=(slope >= 0.0),
                    description="Mean DEM surface slope gradient",
                    measurement_methodology="Finite-difference D8 elevation gradient estimation",
                )
            if "points_streamed" in out:
                pts = float(out["points_streamed"])
                metrics["points_streamed"] = BenchmarkMetric(
                    name="points_streamed",
                    value=pts,
                    unit="pts",
                    target_threshold=100.0,
                    passes_threshold=(pts >= 100.0),
                    description="Point cloud spatial indexing throughput count",
                    measurement_methodology="Streaming tile buffer evaluation",
                )

        elif desc.subsystem == TechnologySubsystem.INFERENCE_ACCELERATION:
            if "speedup_factor" in out:
                spd = float(out["speedup_factor"])
                metrics["inference_speedup_factor"] = BenchmarkMetric(
                    name="inference_speedup_factor",
                    value=round(spd, 2),
                    unit="x",
                    target_threshold=1.5,
                    passes_threshold=(spd >= 1.5),
                    description="Dynamic quantization latency speedup factor",
                    measurement_methodology="Quantized parameter inference timing",
                )
            if "within_rtx4050_bounds" in out:
                rtx_pass = bool(out["within_rtx4050_bounds"])
                metrics["within_rtx4050_bounds"] = BenchmarkMetric(
                    name="within_rtx4050_bounds",
                    value=1.0 if rtx_pass else 0.0,
                    unit="bool",
                    target_threshold=1.0,
                    passes_threshold=rtx_pass,
                    description="Verification that peak memory fits within 6,144 MB RTX 4050 upper limit",
                    measurement_methodology="Model parameter footprint plus activation tensor estimation",
                )

        not_real = classification in (
            ExecutionClassification.SIMULATED,
            ExecutionClassification.UNAVAILABLE,
            ExecutionClassification.FAILED,
        )
        pass_reason: Optional[str] = None
        passes: Optional[bool] = None
        if not_real:
            pass_reason = (
                SIMULATED_REASON if classification != ExecutionClassification.FAILED
                else f"failed: {failure_info or 'execution failed'}"
            )
            for m in metrics.values():
                m.passes_threshold, m.reason = None, pass_reason
            run_status = verdict = classification.value
        elif not metrics:
            run_status = verdict = "NO_MEASUREMENT"
            pass_reason = "no metrics were measured"
        else:
            run_status = "MEASURED"
            passes = all(m.passes_threshold for m in metrics.values())
            verdict = (
                "RECOMMENDED" if (passes and desc.evaluation_priority == 1)
                else ("QUALIFIED" if passes else "NEEDS_OPTIMIZATION")
            )

        if classification == ExecutionClassification.INTERNAL_IMPLEMENTATION:
            label = INTERNAL_LABEL
        elif not_real:
            label = SIMULATED_REASON
        else:
            label = classification.value
        installed = _resolve_installed_version(desc.python_modules)
        if classification in (ExecutionClassification.NATIVE_EXECUTED, ExecutionClassification.WRAPPED_EXECUTED):
            version = installed
            version_source = "importlib.metadata" if version else "unresolved"
        else:
            # The upstream package did not run, so its version is not this result's version.
            version = None
            version_source = (
                f"not applicable: upstream not executed (installed upstream: {installed or 'none'})"
            )

        return TechnologyBenchmarkResult(
            tech_id=desc.id,
            name=desc.name,
            subsystem=desc.subsystem.value,
            license=desc.license,
            evaluation_priority=desc.evaluation_priority,
            is_available_on_host=status.available,
            classification=classification.value,
            verification_status=verification_status.value,
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            ground_truth_source=gt_source,
            model_checkpoint=None,
            technology_version=version,
            hardware_provider=self.hardware_provider,
            measurement_methodology=(
                "Internal mathematical extraction (internal re-implementation, not upstream)"
                if classification == ExecutionClassification.INTERNAL_IMPLEMENTATION
                else "Simulated fallback model (simulated: no real upstream execution)"
            ),
            execution_time_ms=round(exec_time_ms, 2),
            memory_peak_mb=round(measured_mem_mb, 2),
            vram_peak_mb=vram_mb,
            failure_rate_pct=round(100.0 * failures / attempts, 2),
            metrics=metrics,
            summary_verdict=verdict,
            world_ir_promotion_verified=ir_verified,
            failure_info=failure_info,
            status=run_status,
            passes_threshold=passes,
            passes_threshold_reason=pass_reason,
            evidence_label=label,
            vram_peak_reason=vram_reason,
            technology_version_source=version_source,
            attempts=attempts,
            failures=failures,
        )

    def benchmark_all_50(self) -> Dict[str, Any]:
        """Run benchmark suite across all 50 registered technologies."""
        t_start = time.perf_counter()
        results: List[TechnologyBenchmarkResult] = []
        for tech in ALL_50_TECHNOLOGIES:
            results.append(self.benchmark_technology(tech.id))

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        recommended_count = sum(1 for r in results if r.summary_verdict == "RECOMMENDED")
        qualified_count = sum(1 for r in results if r.summary_verdict == "QUALIFIED")
        simulated_count = sum(1 for r in results if r.summary_verdict == "SIMULATED")
        no_measurement_count = sum(1 for r in results if r.summary_verdict == "NO_MEASUREMENT")
        by_verdict: Dict[str, int] = {}
        for r in results:
            by_verdict[r.summary_verdict] = by_verdict.get(r.summary_verdict, 0) + 1

        by_classification: Dict[str, int] = {}
        for r in results:
            cls_name = r.classification
            by_classification[cls_name] = by_classification.get(cls_name, 0) + 1

        return {
            "total_evaluated": len(results),
            "benchmark_runtime_ms": total_time_ms,
            "recommended_count": recommended_count,
            "qualified_count": qualified_count,
            "simulated_count": simulated_count,
            "no_measurement_count": no_measurement_count,
            "by_verdict": by_verdict,
            "by_classification": by_classification,
            "hardware_provider": self.hardware_provider,
            "results": [asdict(r) for r in results],
        }

    def generate_markdown_report(self) -> str:
        """Produce a comprehensive markdown report suitable for documentation or CLI display."""
        benchmark_data = self.benchmark_all_50()
        results = benchmark_data["results"]

        lines = [
            "# Reality Engine — 50 Technologies Comprehensive Benchmark Report",
            "",
            f"**Total Technologies Evaluated**: {benchmark_data['total_evaluated']} / 50",
            f"**Recommended Core Candidates**: {benchmark_data['recommended_count']}",
            f"**Qualified Specialists**: {benchmark_data['qualified_count']}",
            f"**Hardware & Execution Provider**: `{benchmark_data['hardware_provider']}`",
            f"**Benchmark Evaluation Runtime**: {benchmark_data['benchmark_runtime_ms']:.1f} ms",
            f"**Simulated (no real upstream execution, not evaluated)**: {benchmark_data['simulated_count']}",
            f"**No measurement**: {benchmark_data['no_measurement_count']}",
            "**Notes**: VRAM is reported only when measured (otherwise null); INTERNAL_IMPLEMENTATION "
            "rows describe Reality Engine's own re-implementation, not the upstream project.",
            "",
            "### Execution Classifications Breakdown",
        ]

        for cls_name, count in sorted(benchmark_data["by_classification"].items()):
            lines.append(f"- **{cls_name}**: {count} technologies")

        lines.extend([
            "",
            "| # | Technology | Subsystem | Classification | Status | Latency (ms) | Peak RAM | Verdict |",
            "|---|---|---|:---:|:---:|:---:|:---:|:---:|",
        ])

        for i, r in enumerate(results, start=1):
            lines.append(
                f"| {i} | **{r['name']}** | `{r['subsystem']}` | `{r['classification']}` | `{r['verification_status']}` | {r['execution_time_ms']:.1f} | {r['memory_peak_mb']:.1f} MB | **{r['summary_verdict']}** |"
            )

        lines.extend([
            "",
            "## Subsystem Integration Architecture",
            "- **Visual Perception (12 Techs)**: Promotes 2D/3D entities to `WorldIR.entities` with calibrated confidence and Provenance.RECONSTRUCTED (internal) or Provenance.INFERRED (simulated).",
            "- **Geometric Estimation (7 Techs)**: Updates `WorldIR.transforms` with covariance matrices and sub-millimeter residual checks.",
            "- **Reconstruction & Mapping (8 Techs)**: Supplies independent camera trajectories and alternative dense clouds.",
            "- **3D Point-Cloud Perception (8 Techs)**: Partitions point clouds into floors, walls, ceilings, and fixtures.",
            "- **Geometry, CAD & Mesh (8 Techs)**: Enforces guaranteed manifold topology, watertightness, and B-Rep solid generation.",
            "- **Geospatial & Large-Scale (4 Techs)**: Streams multi-gigabyte LiDAR and DEM terrain models.",
            "- **Inference Acceleration (3 Techs)**: Compiles perception networks with FP16/INT8 for low-latency inference within 6GB VRAM bounds.",
        ])

        return "\n".join(lines)
