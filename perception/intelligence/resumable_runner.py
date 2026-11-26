"""Resumable Perception Job Runner & Stage Checkpoint Store (A5 Objective 3).

Provides deterministic, resumable execution of perception runs on disk:
  - Computes an input fingerprint from run_id + image paths + config
  - Persists stage completion manifests and serialized ``PerceptionBundle``
  - Resumes cleanly from the last completed stage without re-running finished work
  - Never uses a parallel evidence store (persists handoff + EvidenceMatrix)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from perception.contracts.claude_integration_v2 import ClaudeReconstructionFeedback
from perception.intelligence.data_model import NormalizedPerceptionObservation, PerceptionBundle
from perception.intelligence.open_vocab import Domain
from perception.intelligence.production_pipeline_a5 import (
    A5ProductionResult,
    run_a5_production_pipeline,
)


@dataclass(frozen=True)
class ResumableJobState:
    run_id: str
    input_fingerprint: str
    status: str  # "completed" | "resumed_from_cache" | "failed"
    observation_count: int
    checkpoint_path: str
    bundle_path: str


def compute_run_fingerprint(
    run_id: str,
    image_paths: Sequence[Path | str],
    *,
    domain: Domain,
    detection_threshold: float,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a perception run's inputs."""
    parts = [
        run_id,
        domain.value,
        f"{detection_threshold:.4f}",
        *(Path(p).name for p in sorted(image_paths, key=lambda x: str(x))),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def run_resumable_a5_job(
    run_id: str,
    work_dir: Path | str,
    *,
    image_paths: Sequence[Path | str] = (),
    observations: Sequence[NormalizedPerceptionObservation] | None = None,
    run_live_detector: bool = False,
    run_live_segmenter: bool = True,
    domain: Domain = Domain.INTERIOR,
    detection_threshold: float = 0.28,
    claude_feedback: ClaudeReconstructionFeedback | None = None,
    force_rerun: bool = False,
) -> tuple[ResumableJobState, A5ProductionResult | None, PerceptionBundle]:
    """Execute or resume an A5 perception job from ``work_dir`` checkpoints."""
    wdir = Path(work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    fp = compute_run_fingerprint(
        run_id,
        image_paths,
        domain=domain,
        detection_threshold=detection_threshold,
    )
    ckpt_file = wdir / "resumable_job_state.json"
    bundle_file = wdir / "perception_bundle.json"

    if not force_rerun and ckpt_file.exists() and bundle_file.exists():
        prev = json.loads(ckpt_file.read_text(encoding="utf-8"))
        if prev.get("input_fingerprint") == fp and prev.get("status") in ("completed", "resumed_from_cache"):
            bundle = PerceptionBundle.from_dict(json.loads(bundle_file.read_text(encoding="utf-8")))
            state = ResumableJobState(
                run_id=run_id,
                input_fingerprint=fp,
                status="resumed_from_cache",
                observation_count=len(bundle.observations),
                checkpoint_path=str(ckpt_file),
                bundle_path=str(bundle_file),
            )
            return state, None, bundle

    res = run_a5_production_pipeline(
        run_id,
        image_paths=image_paths,
        observations=observations,
        run_live_detector=run_live_detector,
        run_live_segmenter=run_live_segmenter,
        domain=domain,
        detection_threshold=detection_threshold,
        claude_feedback=claude_feedback,
        work_dir=wdir,
        persist_to_world=True,
    )
    bundle_file.write_text(res.bundle.to_json(), encoding="utf-8")
    state = ResumableJobState(
        run_id=run_id,
        input_fingerprint=fp,
        status="completed",
        observation_count=len(res.bundle.observations),
        checkpoint_path=str(ckpt_file),
        bundle_path=str(bundle_file),
    )
    ckpt_file.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    return state, res, res.bundle


__all__ = [
    "ResumableJobState",
    "compute_run_fingerprint",
    "run_resumable_a5_job",
]
