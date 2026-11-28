"""Tests for candidate model benchmark, disagreement preservation, class-aware NMS, and OCR (PR-F)."""
from __future__ import annotations

from perception.detection.ocr import match_attribute
from perception.detection.ontology_grounding import nms
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.evaluation.candidate_model_benchmark import (
    run_candidate_benchmark,
)


def _det_obs(engine_id: str, label: str, box: tuple[float, float, float, float], conf: float) -> ObservationRecord:
    return ObservationRecord(
        evidence_id=f"{engine_id}:{label}",
        source_image_id="img0.jpg",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(
            engine_id="detection",
            architecture_id=engine_id,
            checkpoint_id="test",
            license="Apache-2.0",
            hardware_class="cpu",
        ),
        label=label,
        category="object",
        bbox_px=box,
        confidence=conf,
        calibration_quality="unverified",
    )


def test_candidate_benchmark_preserves_failures_and_disagreements() -> None:
    def run_gd(_paths):
        return UnifiedEngineOutput(
            engine_id="grounding_dino",
            run_id="b1",
            observations=(_det_obs("grounding_dino", "door", (10.0, 10.0, 110.0, 210.0), 0.90),),
            status="succeeded",
            wall_time_s=0.4,
            peak_vram_mb=1800.0,
        )

    def run_mrcnn(_paths):
        return UnifiedEngineOutput(
            engine_id="maskrcnn",
            run_id="b1",
            observations=(_det_obs("maskrcnn", "cabinet", (10.0, 10.0, 110.0, 210.0), 0.75),),
            status="succeeded",
            wall_time_s=0.6,
            peak_vram_mb=2400.0,
        )

    def run_broken(_paths):
        raise RuntimeError("checkpoint missing offline")

    records, disagreements = run_candidate_benchmark(
        {"grounding_dino": run_gd, "maskrcnn": run_mrcnn, "depth_anything": run_broken},
        ["img0.jpg"],
        "test_indoor",
    )
    by_id = {r.model_id: r for r in records}
    assert by_id["grounding_dino"].status == "succeeded"
    assert by_id["depth_anything"].status == "failed"
    assert "checkpoint missing offline" in by_id["depth_anything"].error_message
    assert len(disagreements) == 1
    assert disagreements[0].label_a == "door"
    assert disagreements[0].label_b == "cabinet"
    assert disagreements[0].iou == 1.0


def test_class_aware_nms_preserves_distinct_overlapping_classes() -> None:
    boxes = [
        (10.0, 10.0, 50.0, 50.0),
        (12.0, 12.0, 48.0, 48.0),
        (11.0, 11.0, 49.0, 49.0),
    ]
    scores = [0.95, 0.85, 0.70]
    labels = ["switch", "socket", "switch"]
    kept = nms(boxes, scores, iou_threshold=0.5, labels=labels)
    # Index 0 (switch, 0.95) and 1 (socket, 0.85) kept; index 2 (switch, 0.70) suppressed by index 0
    assert kept == [0, 1]


def test_ocr_match_attribute_whole_word_and_ambiguity() -> None:
    allowed = {"exit": ["emergency exit"], "entrance": ["entry"], "unknown": []}
    res_ok = match_attribute("EMERGENCY EXIT ONLY", allowed)
    assert res_ok["value"] == "exit"
    assert res_ok["basis"] == "ocr"
    assert res_ok["p"] is None

    res_ambig = match_attribute("EXIT AND ENTRY", allowed)
    assert res_ambig["value"] is None
    assert len(res_ambig["matches"]) == 2
