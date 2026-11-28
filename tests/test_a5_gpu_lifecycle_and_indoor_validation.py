"""A5 Tests (11, 13, 14 + Indoor Validation): multi-engine disagreement, OOM recovery, timeout, and indoor datasets."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from benchmarks.perception.evaluator import EvaluationMode
from benchmarks.perception.indoor_validation_runner import (
    prepare_six_room_measured_capture_pathway,
    run_indoor_dataset_validation,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.datasets.indoor_registry import normalize_dataset_sample
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
)
from perception.intelligence.deduplication_v2 import resolve_duplicate_detections_v2
from perception.intelligence.dinov2_embeddings import extract_dinov2_embeddings
from perception.intelligence.fusion_v2 import fuse_observations_v2
from perception.intelligence.gpu_lifecycle import (
    HardwareProfile,
    ModelLifecycleManager,
    select_adaptive_batch_and_resolution,
)
from perception.intelligence.multi_engine_dispatch import dispatch_sequential_detectors


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.85,
    engine_id: str = "grounding_dino",
    category: str = "furniture",
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec("image_px", "pixels", 800, 600),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(engine_id, engine_id, f"{engine_id}_v1"),
        ),
    )


def test_11_multi_engine_disagreement_preserves_competing_hypotheses() -> None:
    obs = [
        _obs("o_chair", "img_1", "chair", (100, 200, 250, 420), confidence=0.84, engine_id="grounding_dino"),
        _obs("o_cab", "img_1", "cabinet", (102, 202, 248, 418), confidence=0.78, engine_id="owlv2"),
    ]
    dedup = resolve_duplicate_detections_v2(obs)
    assert len(dedup.competing_hypotheses) == 1
    assert dedup.competing_hypotheses[0].kept_label == "chair"
    assert dedup.competing_hypotheses[0].competing_label == "cabinet"

    decisions = fuse_observations_v2(obs)
    assert len(decisions) == 1
    assert len(decisions[0].detector_disagreements) >= 1


def test_13_gpu_memory_exhaustion_triggers_cpu_oom_fallback_and_blocks_concurrent_claims() -> None:
    mgr = ModelLifecycleManager()
    # Constrained 6 GB consumer GPU must NEVER claim concurrent multi-model execution
    constrained_hw = HardwareProfile(
        cuda_available=True,
        device_name="RTX 4050 Laptop",
        total_vram_mb=6140.0,
        free_vram_mb=4200.0,
        safe_vram_ceiling_mb=5540.0,
        supports_concurrent_models=False,
    )
    mgr.hw = constrained_hw
    mgr.vram_ceiling_mb = 5540.0
    assert mgr.can_run_concurrently([3000.0, 3000.0]) is False

    batch, res, dev = select_adaptive_batch_and_resolution(8, 3000.0, constrained_hw)
    assert dev == "cuda"
    assert batch <= 2
    assert res <= 1024

    # Simulate CUDA OOM on first call, succeeding on CPU fallback
    calls: list[str] = []

    def _oom_then_cpu(device: str, _bs: int, _res: int) -> str:
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "ok_on_cpu"

    out, rec = mgr.run_stage(
        "test_oom_stage",
        "mock_model",
        _oom_then_cpu,
        estimated_vram_mb=1000.0,
        allow_cpu_fallback_on_oom=True,
    )
    assert rec.status in ("oom_recovered_cpu", "succeeded")
    if rec.status == "oom_recovered_cpu":
        assert out == "ok_on_cpu"
        assert rec.device_used == "cpu"


def test_14_model_timeout_surfaces_explicit_timed_out_status() -> None:
    mgr = ModelLifecycleManager()

    def _slow_stage(_dev: str, _bs: int, _res: int) -> str:
        time.sleep(0.35)
        return "late"

    out, rec = mgr.run_stage(
        "slow_detector",
        "mock_slow",
        _slow_stage,
        timeout_s=0.05,
    )
    assert out is None
    assert rec.status == "timed_out"
    assert rec.error_message is not None and "timed out" in rec.error_message


def test_indoor_validation_runner_and_six_room_capture_pathway(tmp_path: Path) -> None:
    real_sample = normalize_dataset_sample(
        {
            "sample_id": "nyu_01",
            "scene_id": "kitchen_1",
            "image_path": "nyu/01.jpg",
            "image_width": 640,
            "image_height": 480,
            "boxes": [
                {"label": "wall", "bbox_xyxy": [0, 0, 640, 350]},
                {"label": "chair", "bbox_xyxy": [120, 220, 260, 440]},
            ],
        },
        dataset_id="nyu_depth_v2",
    )
    preds = {
        "nyu_01": [
            DetectionBox(box_id="p1", class_name="wall", confidence=0.90, box_2d=(0, 0, 640, 350), source_engine="grounding_dino"),
            DetectionBox(box_id="p2", class_name="chair", confidence=0.88, box_2d=(120, 220, 260, 440), source_engine="grounding_dino"),
        ]
    }
    rep_real = run_indoor_dataset_validation(
        "nyu_depth_v2",
        [real_sample],
        preds,
        duplicate_count=1,
        total_raw_detections=3,
        masked_detections=2,
    )
    assert rep_real.evaluation_mode is EvaluationMode.LABELLED_REAL
    assert rep_real.accuracy_claim_allowed is True
    assert rep_real.architectural_ap50 is not None
    assert rep_real.furniture_ap50 is not None

    syn_sample = normalize_dataset_sample(
        {
            "sample_id": "s3d_01",
            "scene_id": "s3d_r1",
            "image_path": "s3d/01.png",
            "image_width": 640,
            "image_height": 480,
            "boxes": [{"label": "chair", "bbox_xyxy": [120, 220, 260, 440]}],
        },
        dataset_id="structured3d",
    )
    rep_syn = run_indoor_dataset_validation("structured3d", [syn_sample], {"s3d_01": preds["nyu_01"][1:]})
    assert rep_syn.evaluation_mode is EvaluationMode.SYNTHETIC
    assert rep_syn.accuracy_claim_allowed is False

    # Six-room measured capture pathway: 1 verified, 1 unlabelled, 4 blocked
    r1 = tmp_path / "room_01"
    r1.mkdir()
    (r1 / "view_01.jpg").write_bytes(b"fake_jpg")
    (r1 / "ground_truth.json").write_text(json.dumps({"is_human_reviewed": True}), encoding="utf-8")
    (r1 / "measured_dimensions.json").write_text(json.dumps({"width_m": 4.2}), encoding="utf-8")

    r2 = tmp_path / "room_02"
    r2.mkdir()
    (r2 / "view_01.jpg").write_bytes(b"fake_jpg")

    six = prepare_six_room_measured_capture_pathway(tmp_path)
    assert six["total_rooms"] == 6
    assert six["verified_rooms"] == 1
    assert six["blocked_rooms"] == 4
    assert six["ready_for_full_six_room_benchmark"] is False


def test_dinov2_embeddings_and_sequential_dispatch_handles_missing_and_real_inputs() -> None:
    obs = [_obs("o1", "img_1", "chair", (40, 40, 140, 140))]
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    img[50:130, 50:130] = 200
    out_obs, d_rep = extract_dinov2_embeddings(obs, {"img_1": img}, local_files_only=True)
    assert d_rep.status in ("succeeded", "unavailable")
    if d_rep.status == "succeeded":
        assert d_rep.observations_embedded == 1
        assert len(out_obs[0].appearance.embedding_vector) > 0

    _, disp_rep = dispatch_sequential_detectors(
        "disp_empty",
        ["does_not_exist.jpg"],
        {},
        requested_engines=("grounding_dino",),
    )
    assert disp_rep.concurrent_gpu_claimed is False
    assert disp_rep.total_raw_observations == 0
