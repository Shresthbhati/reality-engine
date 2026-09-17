"""Regression tests for the mapping-spine perception convergence.

Covers the seams this campaign added:
  - perception/detection/interface.py      (Detection contract)
  - perception/model_registry.py           (checkpoint facts, hash gate)
  - engine/pipeline/vertical_slice.py      (perception stage wiring)

Detector-backed tests use a tiny in-test fake detector so they run
deterministically without torch; the REAL model path is exercised in
`test_maskrcnn_real_model.py` (marked, requires torchvision + weights).
Everything here is deterministic: no clocks, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.math import Vec3
from engine.pipeline.vertical_slice import VerticalSliceOptions
from perception.detection.interface import Detection, DetectionResult
from perception.instances.object_resolution import merge_hypotheses
from perception.model_registry import (
    ModelSpec,
    get as registry_get,
    record_pretrained,
    verify_checkpoint,
)
from perception.segmentation.interface import SegmentedRegion, SegmentationResult


# ------------------------------------------------------------------
# Detection contract
# ------------------------------------------------------------------


class TestDetectionContract:
    def test_valid_detection_roundtrips(self):
        d = Detection(
            detection_id="det-1", evidence_id="IMG_0000", label="chair",
            box=(10.0, 20.0, 110.0, 220.0), score=0.87, model_name="maskrcnn",
        )
        assert d.to_dict()["label"] == "chair"
        assert d.to_dict()["box"] == [10.0, 20.0, 110.0, 220.0]

    def test_score_must_be_a_probability(self):
        with pytest.raises(ValueError):
            Detection("d", "e", "chair", (1, 2, 3, 4), 1.5, "m")
        with pytest.raises(ValueError):
            Detection("d", "e", "chair", (1, 2, 3, 4), -0.1, "m")

    def test_degenerate_boxes_rejected(self):
        with pytest.raises(ValueError):
            Detection("d", "e", "chair", (5, 5, 5, 5), 0.9, "m")
        with pytest.raises(ValueError):
            Detection("d", "e", "chair", (10, 5, 1, 9), 0.9, "m")


# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------


class TestModelRegistry:
    def test_record_and_get(self, tmp_path):
        ck = tmp_path / "fake_weights.pt"
        ck.write_bytes(b"weights-bytes")
        spec = record_pretrained(
            name="test-model", task="detection", source="test",
            checkpoint=ck, sha256=None,
        )
        assert registry_get("test-model") is spec
        d = spec.to_dict()
        assert d["present"] is True
        assert d["checkpoint"] == str(ck)

    def test_hash_verification(self, tmp_path):
        import hashlib
        ck = tmp_path / "w.pt"
        payload = b"deterministic-payload"
        ck.write_bytes(payload)
        good = hashlib.sha256(payload).hexdigest()
        assert verify_checkpoint(ck, good) is True
        assert verify_checkpoint(ck, "0" * 64) is False

    def test_missing_checkpoint_reports_not_present(self, tmp_path):
        spec = ModelSpec(
            name="absent", task="t", source="s",
            checkpoint=tmp_path / "nope.pt",
        )
        assert spec.to_dict()["present"] is False


# ------------------------------------------------------------------
# Cross-view merge (P0.10 baseline behavior, reused by the stage)
# ------------------------------------------------------------------


class TestMergeBasics:
    def _hyp(self, rid, label, x, confidence=0.9):
        from provenance import Provenance, Uncertainty
        from perception.instances.lifting import ObjectHypothesis3D
        return ObjectHypothesis3D(
            region_id=rid, evidence_id=f"IMG_{rid}", label=label,
            position=Vec3(x, 0.0, 0.0),
            bounds_min=Vec3(x - 0.2, -0.2, -0.2),
            bounds_max=Vec3(x + 0.2, 0.2, 0.2),
            point_count=100, mask_pixel_count=400,
            confidence=confidence,
            uncertainty=Uncertainty(), provenance=Provenance.INFERRED,
        )

    def test_same_object_three_views_one_candidate(self):
        merged = merge_hypotheses([
            self._hyp("a", "chair", 1.0),
            self._hyp("b", "chair", 1.15),
            self._hyp("c", "chair", 1.3),
            self._hyp("d", "table", 1.2),  # different class, never merges
        ])
        chairs = [c for c in merged if c.label == "chair"]
        assert len(chairs) == 1
        assert chairs[0].observation_count == 3

    def test_distant_same_label_stay_separate(self):
        merged = merge_hypotheses([
            self._hyp("a", "chair", 0.0),
            self._hyp("b", "chair", 5.0),
        ])
        assert len(merged) == 2


# ------------------------------------------------------------------
# Pipeline perception stage (fake detector -- no torch)
# ------------------------------------------------------------------


class _FakeDetector:
    """Deterministic stand-in producing one square mask per view centered
    where the fixture camera looks. Lets the full stage wiring (segment ->
    lift -> merge -> promote -> validate) run without torch."""

    calls = 0

    def __init__(self, **kwargs):
        _FakeDetector.calls += 1

    def segment(self, evidence):
        n = 64
        results = []
        for item in evidence:
            mask = [[False] * n for _ in range(n)]
            for r in range(20, 44):
                for c in range(20, 44):
                    mask[r][c] = True
            results.append(SegmentationResult(
                evidence_id=item.evidence_id,
                regions=[SegmentedRegion(
                    region_id=f"seg-{item.evidence_id}-000",
                    evidence_id=item.evidence_id,
                    label="chair",
                    mask=mask,
                    confidence=0.9,
                )],
                model_name="fake",
            ))
        return results


@pytest.fixture
def synthetic_registered_result():
    """A minimal registered ReconstructionResult in metric meters: two
    cameras on +Z looking at -Z, one wall of sparse points at z=-2, plus
    matching synthetic depth maps so lifting has something metric."""
    from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructionResult
    from reconstruction.calibration.camera import CameraIntrinsics
    from perception.depth.interface import DepthMap

    intr = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0, width=64, height=64)
    poses = [
        ReconstructedCameraPose(
            evidence_id="IMG_A", position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
        ReconstructedCameraPose(
            evidence_id="IMG_B", position=(0.5, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
    ]
    points = [(x, y, -2.0) for x in (-0.5, 0.0, 0.5) for y in (0.0, 0.5)]
    result = ReconstructionResult(
        points=points, camera_poses=poses, registration_status="success"
    )

    # A metric depth map consistent with a wall at z=-2 seen from z=0:
    # depth = 2.0 everywhere.
    maps = []
    for ev in ("IMG_A", "IMG_B"):
        maps.append(DepthMap(
            evidence_id=ev, width=64, height=64,
            values=[[2.0] * 64 for _ in range(64)],
            unit="meters",
        ))
    return result, intr, maps


def test_perception_stage_runs_fake_detector(
    synthetic_registered_result, monkeypatch
):
    from engine.pipeline import vertical_slice as vs

    result, intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0),
        image_size=(64, 64),
        perception_model="fake",
    )

    class _Evidence:
        def __init__(self, eid):
            self.evidence_id = eid
            self.source_uri = f"file://{eid}.jpg"

    evidence = [_Evidence("IMG_A"), _Evidence("IMG_B")]

    # Patch the detector import inside the stage's lazy import block.
    import perception.detection.maskrcnn_backend as mb
    monkeypatch.setattr(mb, "MaskRCNNDetector", _FakeDetector)

    # Call the stage directly (the real entry `vertical_slice` needs COLMAP;
    # the stage function is the unit under test here).
    world = _empty_world()
    facts = vs._perception_stage(result, world, evidence, maps, options)

    assert facts["status"] == "ran"
    assert facts["masks_considered"] == 2
    assert facts["hypotheses_lifted"] == 2
    assert facts["candidates_merged"] == 1  # same wall position -> one object
    assert facts["entities_promoted"] == 1
    # The world now carries a traceable object entity.
    ent = [e for e in world.entities.values() if e.id.startswith("entity-object")]
    assert len(ent) == 1
    assert ent[0].semantic_labels == ["chair"]
    assert ent[0].observations, "promoted entity must carry provenance observations"


def test_perception_stage_skips_without_metric_depth(
    synthetic_registered_result,
):
    from engine.pipeline import vertical_slice as vs

    result, intr, _ = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0),
        image_size=(64, 64),
        perception_model="fake",
    )

    class _Evidence:
        def __init__(self, eid):
            self.evidence_id = eid
            self.source_uri = f"file://{eid}.jpg"

    world = _empty_world()
    facts = vs._perception_stage(
        result, world, [_Evidence("IMG_A")], [], options
    )
    assert facts["status"] == "skipped"
    assert "metric depth" in facts["note"]
    assert not [e for e in world.entities.values() if e.id.startswith("entity-object")]


def test_perception_stage_disabled(synthetic_registered_result):
    from engine.pipeline import vertical_slice as vs

    result, intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(perception_model=None)

    world = _empty_world()
    facts = vs._perception_stage(result, world, [], maps, options)
    assert facts["status"] == "skipped"
    assert facts["note"] == "disabled (perception_model=None)"


def _empty_world():
    from world_ir.world_v1 import WorldIR
    return WorldIR(name="test")
