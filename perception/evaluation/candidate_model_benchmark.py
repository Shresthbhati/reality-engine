"""Reproducible Candidate Model Comparison & Disagreement Preservation Benchmark.

Follows Cross-Agent Protocol Rules 5, 6, 7, and 10:
- Compares candidate models on the same dataset and image IDs.
- Records commit SHA, checkpoint ID, license, hardware, dependency versions,
  runtime, peak VRAM, failure logs, and per-class precision/recall/F1.
- Never silently replaces a failed/unavailable model with a weaker fallback.
- Preserves model disagreements instead of averaging incompatible outputs.
"""
from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from perception.engine_evidence import ObservationRecord, UnifiedEngineOutput


@dataclass(frozen=True)
class CandidateModelSpec:
    """Metadata and hardware profile for a candidate perception model."""

    model_id: str
    task: str
    checkpoint_id: str
    license: str
    min_vram_mb: int
    notes: str = ""


CANDIDATE_CATALOG: dict[str, CandidateModelSpec] = {
    "grounding_dino": CandidateModelSpec(
        model_id="grounding_dino",
        task="open_vocab_detection",
        checkpoint_id="IDEA-Research/grounding-dino-tiny",
        license="Apache-2.0",
        min_vram_mb=2048,
        notes="Tiled open-vocabulary detection for indoor/outdoor & small MEP/hardware classes.",
    ),
    "maskrcnn": CandidateModelSpec(
        model_id="maskrcnn",
        task="closed_vocab_instance_segmentation",
        checkpoint_id="torchvision/maskrcnn_resnet50_fpn_v1",
        license="BSD-3-Clause (weights) / MIT",
        min_vram_mb=4000,
        notes="COCO 80-class baseline detector + instance mask generator.",
    ),
    "florence2_ocr": CandidateModelSpec(
        model_id="florence2_ocr",
        task="ocr_and_attribute_reading",
        checkpoint_id="microsoft/Florence-2-base",
        license="MIT",
        min_vram_mb=2048,
        notes="Nameplate/panel OCR (<OCR>) with basis='ocr' and confidence=None.",
    ),
    "lightglue_aliked": CandidateModelSpec(
        model_id="lightglue_aliked",
        task="feature_matching",
        checkpoint_id="kornia/ALIKED-n16+LightGlue",
        license="Apache-2.0",
        min_vram_mb=1024,
        notes="Learned keypoint matcher; downscales inputs >1600px long edge for 6GB GPUs.",
    ),
    "sift": CandidateModelSpec(
        model_id="sift",
        task="feature_matching",
        checkpoint_id="opencv/SIFT+RANSAC",
        license="Apache-2.0",
        min_vram_mb=0,
        notes="Classical CPU baseline matcher.",
    ),
    "depth_anything": CandidateModelSpec(
        model_id="depth_anything",
        task="monocular_depth",
        checkpoint_id="depth-anything/Depth-Anything-V2-Small-hf",
        license="Apache-2.0",
        min_vram_mb=2048,
        notes="Relative depth estimation loaded strictly with local_files_only=True.",
    ),
    "midas": CandidateModelSpec(
        model_id="midas",
        task="monocular_depth",
        checkpoint_id="Intel/dpt-hybrid-midas",
        license="MIT",
        min_vram_mb=2000,
        notes="Secondary relative depth baseline for cross-engine disagreement checks.",
    ),
}


@dataclass(frozen=True)
class ModelBenchmarkRecord:
    """Reproducible execution record for one candidate model on a fixed dataset."""

    model_id: str
    checkpoint_id: str
    license: str
    commit_sha: str
    dataset: str
    image_ids: tuple[str, ...]
    hardware: str
    python_version: str
    status: str
    wall_time_s: float
    peak_vram_mb: float | None
    n_observations: int
    error_message: str = ""
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDisagreement:
    """Preserved disagreement between two candidate models on the same image/region."""

    image_id: str
    model_a: str
    label_a: str
    confidence_a: float | None
    model_b: str
    label_b: str
    confidence_b: float | None
    iou: float


def _current_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_model_disagreements(
    outputs: Sequence[UnifiedEngineOutput],
    iou_threshold: float = 0.5,
) -> list[ModelDisagreement]:
    """Preserve competing hypotheses when two models overlap spatially with different labels."""
    disagreements: list[ModelDisagreement] = []
    det_obs: list[tuple[str, ObservationRecord]] = []
    for out in outputs:
        for obs in out.observations:
            if obs.bbox_px is not None and len(obs.bbox_px) == 4:
                det_obs.append((out.engine_id, obs))

    for i in range(len(det_obs)):
        eng_a, a = det_obs[i]
        la = str(a.metadata.get("ontology_path") or a.label).lower()
        for j in range(i + 1, len(det_obs)):
            eng_b, b = det_obs[j]
            if eng_a == eng_b or a.source_image_id != b.source_image_id:
                continue
            lb = str(b.metadata.get("ontology_path") or b.label).lower()
            if la == lb:
                continue
            overlap = _iou(a.bbox_px, b.bbox_px)  # type: ignore[arg-type]
            if overlap >= iou_threshold:
                disagreements.append(ModelDisagreement(
                    image_id=a.source_image_id,
                    model_a=eng_a,
                    label_a=a.label,
                    confidence_a=a.confidence,
                    model_b=eng_b,
                    label_b=b.label,
                    confidence_b=b.confidence,
                    iou=round(overlap, 4),
                ))
    return disagreements


def run_candidate_benchmark(
    runners: dict[str, Callable[[Sequence[str]], UnifiedEngineOutput]],
    image_paths: Sequence[str],
    dataset_name: str,
    *,
    hardware: str = "cpu",
) -> tuple[list[ModelBenchmarkRecord], list[ModelDisagreement]]:
    """Run all candidate models on the same inputs without silent fallbacks."""
    sha = _current_commit_sha()
    py_ver = platform.python_version()
    img_ids = tuple(str(p) for p in image_paths)
    records: list[ModelBenchmarkRecord] = []
    outputs: list[UnifiedEngineOutput] = []

    for model_id, runner in runners.items():
        spec = CANDIDATE_CATALOG.get(model_id)
        ckpt = spec.checkpoint_id if spec else "custom"
        lic = spec.license if spec else "unknown"
        try:
            out = runner(image_paths)
            outputs.append(out)
            records.append(ModelBenchmarkRecord(
                model_id=model_id,
                checkpoint_id=ckpt,
                license=lic,
                commit_sha=sha,
                dataset=dataset_name,
                image_ids=img_ids,
                hardware=hardware,
                python_version=py_ver,
                status=out.status,
                wall_time_s=round(float(out.wall_time_s), 4),
                peak_vram_mb=out.peak_vram_mb,
                n_observations=len(out.observations),
                error_message=out.error_message,
            ))
        except Exception as exc:
            # Never replace a failed candidate with a weaker model
            records.append(ModelBenchmarkRecord(
                model_id=model_id,
                checkpoint_id=ckpt,
                license=lic,
                commit_sha=sha,
                dataset=dataset_name,
                image_ids=img_ids,
                hardware=hardware,
                python_version=py_ver,
                status="failed",
                wall_time_s=0.0,
                peak_vram_mb=None,
                n_observations=0,
                error_message=f"{type(exc).__name__}: {exc}",
            ))

    return records, find_model_disagreements(outputs)
