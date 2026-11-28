"""A5 Real-Photo & Multi-View Experiment Runner (Experiments A, B, C).

Provides reproducible entry points for:
  - Experiment A: ``south_building`` real photographs (comparing A5 vs A4)
  - Experiment B: Real indoor photographs (or explicit blocker report if unavailable)
  - Experiment C: Multi-view indoor scene (repeated objects, masks, cross-view identity)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from perception.contracts.claude_integration_v2 import ClaudeReconstructionFeedback
from perception.intelligence.data_model import NormalizedPerceptionObservation
from perception.intelligence.open_vocab import Domain
from perception.intelligence.production_pipeline_a5 import (
    A5ProductionResult,
    run_a5_production_pipeline,
)


def run_experiment_a_south_building(
    image_paths: Sequence[Path | str],
    *,
    work_dir: Path | str | None = None,
    claude_feedback: ClaudeReconstructionFeedback | None = None,
) -> dict[str, Any]:
    """Experiment A: run A5 pipeline on south_building real photos and report A4 vs A5 gains."""
    res: A5ProductionResult = run_a5_production_pipeline(
        "experiment_a_south_building",
        image_paths=image_paths,
        run_live_detector=True,
        run_live_segmenter=True,
        domain=Domain.EXTERIOR,
        detection_threshold=0.30,
        claude_feedback=claude_feedback,
        work_dir=work_dir,
        persist_to_world=work_dir is not None,
    )
    return {
        "experiment": "A_south_building",
        "images": [Path(p).name for p in image_paths],
        "summary": res.to_summary_dict(),
        "is_labelled": False,
        "accuracy_claim_allowed": False,
    }


def run_experiment_b_indoor(
    image_paths: Sequence[Path | str],
    *,
    work_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Experiment B: run A5 pipeline on indoor photos or report explicit dataset blocker."""
    existing = [Path(p) for p in image_paths if Path(p).exists()]
    if not existing:
        return {
            "experiment": "B_indoor_real",
            "status": "blocked_no_local_indoor_photos",
            "blocker": "No licensed real indoor photographs present at supplied paths.",
            "accuracy_claim_allowed": False,
        }
    res = run_a5_production_pipeline(
        "experiment_b_indoor_real",
        image_paths=existing,
        run_live_detector=True,
        run_live_segmenter=True,
        domain=Domain.INTERIOR,
        detection_threshold=0.28,
        work_dir=work_dir,
        persist_to_world=work_dir is not None,
    )
    return {
        "experiment": "B_indoor_real",
        "status": "executed",
        "images": [p.name for p in existing],
        "summary": res.to_summary_dict(),
        "is_labelled": False,
        "accuracy_claim_allowed": False,
    }


def run_experiment_c_multiview_indoor(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    precomputed_masks: Mapping[str, np.ndarray] | None = None,
    claude_feedback: ClaudeReconstructionFeedback | None = None,
    work_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Experiment C: evaluate multi-view indoor scene (repeated objects, masks, cross-view identity)."""
    res = run_a5_production_pipeline(
        "experiment_c_multiview_indoor",
        images_rgb=images_rgb,
        observations=observations,
        precomputed_masks=precomputed_masks,
        run_live_detector=False,
        run_live_segmenter=precomputed_masks is not None,
        claude_feedback=claude_feedback,
        work_dir=work_dir,
        persist_to_world=work_dir is not None,
    )
    return {
        "experiment": "C_multiview_indoor",
        "summary": res.to_summary_dict(),
        "pairwise_outcomes": [
            {
                "a": d.obs_a_id,
                "b": d.obs_b_id,
                "label": d.canonical_label,
                "score": d.similarity_score,
                "outcome": d.outcome,
            }
            for d in res.identity_v2.pairwise_decisions
        ],
    }


__all__ = [
    "run_experiment_a_south_building",
    "run_experiment_b_indoor",
    "run_experiment_c_multiview_indoor",
]
