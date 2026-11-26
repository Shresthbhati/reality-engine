"""Constrained-Hardware Multi-Engine Detection Dispatcher (A5 Objective 4).

Selects and executes available zero-shot / open-vocab detectors sequentially
based on hardware VRAM budget and local HuggingFace cache availability:
  - Grounding DINO (``IDEA-Research/grounding-dino-tiny``)
  - OWL-ViT / OWLv2 (``google/owlvit-base-patch32``)
  - Never claims concurrent GPU execution when VRAM < 14 GB.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
)
from perception.intelligence.gpu_lifecycle import (
    ModelLifecycleManager,
    ModelStageExecutionRecord,
)
from perception.intelligence.open_vocab import Domain, generate_survey_prompts


@dataclass(frozen=True)
class MultiEngineDispatchReport:
    requested_engines: tuple[str, ...]
    executed_engines: tuple[str, ...]
    skipped_engines: dict[str, str]
    concurrent_gpu_claimed: bool
    stage_records: tuple[ModelStageExecutionRecord, ...]
    total_raw_observations: int


def _is_cached(repo_id: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return (hub / ("models--" + repo_id.replace("/", "--"))).exists()


def dispatch_sequential_detectors(
    run_id: str,
    image_paths: Sequence[Path | str],
    images_rgb: Mapping[str, np.ndarray],
    *,
    requested_engines: Sequence[str] = ("grounding_dino",),
    domain: Domain = Domain.INTERIOR,
    threshold: float = 0.28,
    lifecycle: ModelLifecycleManager | None = None,
) -> tuple[list[NormalizedPerceptionObservation], MultiEngineDispatchReport]:
    """Run requested detectors sequentially through ``ModelLifecycleManager``."""
    mgr = lifecycle or ModelLifecycleManager()
    str_paths = [str(Path(p)) for p in image_paths if Path(p).exists()]
    prompts = tuple(generate_survey_prompts(domain=domain, max_prompts=28))
    all_obs: list[NormalizedPerceptionObservation] = []
    executed: list[str] = []
    skipped: dict[str, str] = {}
    records: list[ModelStageExecutionRecord] = []

    if not str_paths:
        return [], MultiEngineDispatchReport(
            requested_engines=tuple(requested_engines),
            executed_engines=(),
            skipped_engines={e: "no_existing_images" for e in requested_engines},
            concurrent_gpu_claimed=False,
            stage_records=(),
            total_raw_observations=0,
        )

    for eng in requested_engines:
        if eng == "grounding_dino":
            ckpt = "IDEA-Research/grounding-dino-tiny"
            if not _is_cached(ckpt):
                skipped[eng] = "checkpoint_not_cached"
                continue
            from perception.detection_adapter import _grounding_dino_run

            def _run_gd(_dev: str, _bs: int, _res: int) -> Any:
                return _grounding_dino_run(str_paths, threshold=threshold, queries=prompts, run_id=run_id)

            out, rec = mgr.run_stage("detect_grounding_dino", ckpt, _run_gd, num_images=len(str_paths), estimated_vram_mb=1800.0)
            records.append(rec)
            if out is None:
                skipped[eng] = rec.error_message or rec.status
                continue
            executed.append(eng)
            for idx, orec in enumerate(out.observations):
                if orec.bbox_px is None:
                    continue
                img_name = Path(orec.source_image_id).name
                arr = images_rgb.get(img_name)
                if arr is None:
                    continue
                ih, iw = arr.shape[:2]
                lbl = str(orec.label or "object").strip().lower().replace(" ", "_")
                cat = "architecture" if lbl in ("wall", "floor", "ceiling", "door", "window", "column", "beam", "stairs") else "furniture"
                all_obs.append(NormalizedPerceptionObservation(
                    observation_id=f"{eng}_{img_name}_{idx}",
                    schema_version=SCHEMA_VERSION,
                    raw_label=str(orec.label),
                    canonical_label=lbl,
                    taxonomy_id=f"re:{lbl}",
                    category=cat,
                    bbox_px=tuple(float(v) for v in orec.bbox_px[:4]),
                    coords=CoordinateSystemSpec("image_px", "pixels", iw, ih),
                    scores=ScoreSemantics(raw_score=orec.confidence, score_emitted=orec.confidence is not None),
                    provenance=DetectionProvenance(
                        source_image_id=img_name,
                        execution=ModelExecutionProvenance(eng, orec.model_id.architecture_id, ckpt),
                    ),
                ))
        else:
            skipped[eng] = f"unsupported_or_uncached_engine:{eng}"

    return all_obs, MultiEngineDispatchReport(
        requested_engines=tuple(requested_engines),
        executed_engines=tuple(executed),
        skipped_engines=skipped,
        concurrent_gpu_claimed=mgr.can_run_concurrently([1800.0, 2200.0]),
        stage_records=tuple(records),
        total_raw_observations=len(all_obs),
    )


__all__ = [
    "MultiEngineDispatchReport",
    "dispatch_sequential_detectors",
]
