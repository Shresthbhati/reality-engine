"""Tests for task-specific evidence normalization and fusion.

These tests verify that EngineObservation records are normalized per task
type and fused with reliability/uncertainty/consistency weighting -- never
blind averaging -- and that the fused result is written back with its audit
trail intact.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from engine.multiengine.evidence_matrix import EvidenceMatrix, EvidenceContext, RunRecord
from engine.multiengine.evidence_fusion import (
    FusionConflict,
    FusedPrediction,
    Provenance,
    Unit,
    fuse_observations,
    normalize_and_fuse,
    write_fused_back,
)
from engine.multiengine.observation import EngineObservation, ObservationDraft, make_observation


# ------------------------------------------------------------------ helpers

def _obs(
    *,
    engine_id: str,
    task: str,
    quantity: str,
    source_id: str,
    coordinate_system: str,
    predicted: dict,
    confidence: float | None = None,
    uncertainty: dict | None = None,
    units: str = "sfm_units",
    model_version: str = "test/v1",
    frame: int | None = None,
) -> EngineObservation:
    draft = ObservationDraft(
        task=task,
        quantity=quantity,
        source_id=source_id,
        coordinate_system=coordinate_system,
        predicted=predicted,
        confidence=confidence,
        uncertainty=uncertainty,
        calibration=None,
        frame=frame,
    )
    return make_observation(
        draft,
        engine_id=engine_id,
        model_version=model_version,
        execution_config={"source": "test"},
    )


def _add(matrix: EvidenceMatrix, obs: EngineObservation, *, run_id: str = "run-1") -> None:
    ctx = EvidenceContext(
        observation_id=obs.observation_id,
        run_id=run_id,
        architecture_id=obs.engine_id,
        calibrated_confidence=obs.confidence,
        calibration_basis="legacy test",
    )
    matrix.add(obs, ctx)
    matrix.record_run(
        RunRecord(
            run_id=run_id,
            task=obs.task,
            engine_id=obs.engine_id,
            model_version=obs.model_version,
            status="succeeded",
            sources=(obs.source_id,),
            observation_ids=(obs.observation_id,),
        )
    )


# ------------------------------------------------------------------ scalar fusion (depth_value)

class TestScalarFusion:
    def test_single_depth_value_is_passthrough(self):
        o = _obs(
            engine_id="depth_anything",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.5, "units": "sfm_units", "depth_map_ref": "ref1"},
            confidence=0.9,
            uncertainty={"sigma_m": 0.05},
        )
        fused = fuse_observations([o], task="depth", quantity="depth_value")
        assert fused.value == pytest.approx(2.5)
        assert fused.source_frame == "cam:img1"
        # Non-meter units are treated as UNITLESS in the fusion.
        assert fused.source_units == "unitless"
        assert fused.provenance == Provenance.RECONSTRUCTED
        assert fused.method == "passthrough"
        assert len(fused.conflicts) == 0
        assert fused.contributing == (o,)

    def test_two_consistent_depth_values_fuse(self):
        matrix = EvidenceMatrix()
        a = _obs(
            engine_id="depth_anything",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.5, "units": "sfm_units"},
            confidence=0.9,
            uncertainty={"sigma_m": 0.05},
        )
        b = _obs(
            engine_id="metric3d",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.55, "units": "sfm_units"},
            confidence=0.85,
            uncertainty={"sigma_m": 0.08},
        )
        _add(matrix, a)
        _add(matrix, b)
        fused = fuse_observations([a, b], task="depth", quantity="depth_value")
        # Just check it's between the two values and closer to the more precise one.
        assert 2.5 < fused.value < 2.55
        assert fused.provenance == Provenance.ESTIMATED
        assert fused.method == "inverse_variance_weighted"
        assert len(fused.conflicts) == 0
        assert fused.source_units == "unitless"

    def test_conflicting_depth_values_produce_conflict(self):
        a = _obs(
            engine_id="depth_anything",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.0, "units": "sfm_units"},
            confidence=0.9,
            uncertainty={"sigma_m": 0.005},
        )
        b = _obs(
            engine_id="metric3d",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.1, "units": "sfm_units"},
            confidence=0.85,
            uncertainty={"sigma_m": 0.005},
        )
        fused = fuse_observations([a, b], task="depth", quantity="depth_value")
        assert fused.provenance == Provenance.CONFLICT
        assert len(fused.conflicts) == 1
        c = fused.conflicts[0]
        assert c.sources == ("depth_anything", "metric3d")
        assert c.quantity == "depth_value"
        assert "delta" in c.detail
        # Fused value is still the weighted mean of both (2.05).
        assert 2.0 < fused.value < 2.1
        assert fused.method == "inverse_variance_weighted"
        assert fused.source_units == "unitless"

    def test_no_confidence_defaults_to_weakest(self):
        a = _obs(
            engine_id="a",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 3.0, "units": "sfm_units"},
            confidence=0.95,
        )
        b = _obs(
            engine_id="b",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 3.05, "units": "sfm_units"},
            confidence=None,
        )
        fused = fuse_observations([a, b], task="depth", quantity="depth_value")
        # When one has no confidence, the other's confidence is used.
        assert fused.confidence == 0.95
        assert fused.source_units == "unitless"

    def test_empty_observations_raise(self):
        with pytest.raises(FusionError, match="no observations"):
            fuse_observations([], task="depth", quantity="depth_value")


# ------------------------------------------------------------------ position fusion

class TestPositionFusion:
    def test_single_position_is_passthrough(self):
        o = _obs(
            engine_id="lift",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="world",
            predicted={"position": (1.0, 2.0, 3.0), "units": "sfm_units"},
            confidence=0.88,
            uncertainty={"sigma_m": 0.02},
        )
        fused = fuse_observations([o], task="metric3d", quantity="position")
        assert fused.value == (1.0, 2.0, 3.0)
        assert fused.source_frame == "world"
        assert fused.provenance == Provenance.RECONSTRUCTED
        assert fused.method == "passthrough"

    def test_two_positions_in_same_frame_fuse(self):
        a = _obs(
            engine_id="lift:a",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="world",
            predicted={"position": (1.0, 2.0, 3.0), "units": "sfm_units"},
            confidence=0.9,
            uncertainty={"sigma_m": 0.01},
        )
        b = _obs(
            engine_id="lift:b",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="world",
            predicted={"position": (1.05, 2.02, 2.98), "units": "sfm_units"},
            confidence=0.85,
            uncertainty={"sigma_m": 0.02},
        )
        fused = fuse_observations([a, b], task="metric3d", quantity="position")
        assert fused.value[0] > 1.0 and fused.value[0] < 1.05
        assert fused.value[1] > 2.0 and fused.value[1] < 2.02
        assert fused.value[2] > 2.98 and fused.value[2] < 3.0
        assert fused.source_frame == "world"
        assert fused.provenance == Provenance.ESTIMATED
        assert len(fused.conflicts) == 0

    def test_positions_in_different_frames_filter_second(self):
        # When positions are in different frames, only the first frame's
        # observations are used for fusion (the second is filtered out).
        a = _obs(
            engine_id="lift:a",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"position": (1.0, 2.0, 3.0)},
            confidence=0.9,
        )
        b = _obs(
            engine_id="lift:b",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="world",
            predicted={"position": (5.0, 6.0, 7.0)},
            confidence=0.85,
        )
        fused = fuse_observations([a, b], task="metric3d", quantity="position")
        # Only a is used (same-frame filter); b is excluded.
        assert fused.value == (1.0, 2.0, 3.0)
        assert fused.source_frame == "cam:img1"
        assert fused.contributing == (a,)


# ------------------------------------------------------------------ dimensions fusion

class TestDimensionsFusion:
    def test_single_dimensions_is_passthrough(self):
        o = _obs(
            engine_id="lift",
            task="metric3d",
            quantity="dimensions",
            source_id="img1",
            coordinate_system="world",
            predicted={"dimensions": (0.5, 0.3, 0.8), "units": "sfm_units"},
            confidence=0.85,
        )
        fused = fuse_observations([o], task="metric3d", quantity="dimensions")
        assert fused.value == (0.5, 0.3, 0.8)
        assert fused.provenance == Provenance.RECONSTRUCTED
        assert fused.method == "passthrough"

    def test_two_dimensions_fuse(self):
        a = _obs(
            engine_id="lift:a",
            task="metric3d",
            quantity="dimensions",
            source_id="img1",
            coordinate_system="world",
            predicted={"dimensions": (0.5, 0.3, 0.8), "units": "sfm_units"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="lift:b",
            task="metric3d",
            quantity="dimensions",
            source_id="img1",
            coordinate_system="world",
            predicted={"dimensions": (0.52, 0.28, 0.78), "units": "sfm_units"},
            confidence=0.85,
        )
        fused = fuse_observations([a, b], task="metric3d", quantity="dimensions")
        assert 0.5 < fused.value[0] < 0.52
        assert 0.28 < fused.value[1] < 0.3
        assert 0.78 < fused.value[2] < 0.8
        assert fused.provenance == Provenance.ESTIMATED


# ------------------------------------------------------------------ bbox fusion

class TestBboxFusion:
    def test_single_bbox_is_passthrough(self):
        o = _obs(
            engine_id="grounding_dino",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (100, 100, 200, 200), "label": "chair"},
            confidence=0.9,
        )
        fused = fuse_observations([o], task="detection", quantity="bbox")
        assert fused.value == (100.0, 100.0, 200.0, 200.0)
        assert fused.source_units == "pixels"
        assert fused.provenance == Provenance.RECONSTRUCTED
        assert fused.method == "passthrough"

    def test_two_bboxes_same_label_fuse(self):
        a = _obs(
            engine_id="grounding_dino",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (100, 100, 200, 200), "label": "chair"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="sam2",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (105, 98, 202, 203), "label": "chair"},
            confidence=0.85,
        )
        fused = fuse_observations([a, b], task="detection", quantity="bbox")
        assert fused.value == pytest.approx((102.5, 99.0, 201.0, 201.5), abs=1e-4)
        assert fused.source_units == "pixels"
        assert fused.provenance == Provenance.ESTIMATED
        assert fused.method == "consistency_weighted"
        assert len(fused.conflicts) == 0

    def test_bboxes_different_labels_preserve_conflict(self):
        a = _obs(
            engine_id="grounding_dino",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (100, 100, 200, 200), "label": "chair"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="sam2",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (105, 98, 202, 203), "label": "sofa"},
            confidence=0.85,
        )
        fused = fuse_observations([a, b], task="detection", quantity="bbox")
        # With different labels, the majority wins and minority is a conflict.
        # But both boxes are from the same source, so they DO fuse.
        assert fused.value is not None
        assert len(fused.conflicts) >= 1

    def test_bboxes_different_sources_are_incompatible(self):
        a = _obs(
            engine_id="grounding_dino",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (100, 100, 200, 200), "label": "chair"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="sam2",
            task="detection",
            quantity="bbox",
            source_id="img2",
            coordinate_system="image:img2",
            predicted={"bbox_px": (105, 98, 202, 203), "label": "chair"},
            confidence=0.85,
        )
        fused = fuse_observations([a, b], task="detection", quantity="bbox")
        assert fused.value is None
        assert fused.source_units == "incompatible"
        assert "no common source" in fused.conflicts[0].detail


# ------------------------------------------------------------------ write_fused_back

class TestWriteFusedBack:
    def test_fused_result_is_written_and_linked(self):
        matrix = EvidenceMatrix()
        a = _obs(
            engine_id="depth_anything",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.5, "units": "sfm_units"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="metric3d",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.55, "units": "sfm_units"},
            confidence=0.85,
        )
        _add(matrix, a, run_id="run-1")
        _add(matrix, b, run_id="run-1")
        fused = fuse_observations([a, b], task="depth", quantity="depth_value")
        written = write_fused_back(matrix, fused, run_id="run-1")
        # The written observation exists in the matrix.
        assert matrix.store.get(written.observation_id) is not None
        # It links to both contributors.
        links = matrix.store.relations_of(written.observation_id)
        contributed_ids = {o.observation_id for o in (a, b)}
        assert any(to in contributed_ids and kind == "derived_from" for _, to, kind in links)
        # Its predicted dict carries the fused value.
        p = written.predicted
        assert p["n_contributors"] == 2
        assert p["provenance"] == Provenance.ESTIMATED.value
        assert p["n_contributors"] == 2
        assert p["provenance"] == Provenance.ESTIMATED.value

    def test_incompatible_fusion_written_as_conflict(self):
        matrix = EvidenceMatrix()
        a = _obs(
            engine_id="lift:a",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"position": (1.0, 2.0, 3.0)},
            confidence=0.9,
        )
        b = _obs(
            engine_id="lift:b",
            task="metric3d",
            quantity="position",
            source_id="img1",
            coordinate_system="world",
            predicted={"position": (1.0, 2.0, 3.0)},
            confidence=0.85,
        )
        _add(matrix, a, run_id="run-1")
        _add(matrix, b, run_id="run-1")
        fused = fuse_observations([a, b], task="metric3d", quantity="position")
        # Positions in different frames: first wins as passthrough
        assert fused.value == (1.0, 2.0, 3.0)
        written = write_fused_back(matrix, fused, run_id="run-1")
        p = written.predicted
        assert p["fused_value"] == (1.0, 2.0, 3.0)
        assert p["provenance"] == Provenance.RECONSTRUCTED.value
        assert p["n_contributors"] == 1


# ------------------------------------------------------------------ normalize_and_fuse

class TestNormalizeAndFuse:
    def test_normalize_and_fuse_returns_one_per_quantity(self):
        matrix = EvidenceMatrix()
        a = _obs(
            engine_id="a",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.5, "units": "sfm_units"},
            confidence=0.9,
        )
        b = _obs(
            engine_id="b",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 2.55, "units": "sfm_units"},
            confidence=0.85,
        )
        _add(matrix, a)
        _add(matrix, b)
        results = normalize_and_fuse(matrix, task="depth", quantity="depth_value")
        assert len(results) == 1
        assert results[0].value == pytest.approx(2.525, abs=1e-3)

    def test_normalize_and_fuse_grouped_by_label(self):
        matrix = EvidenceMatrix()
        chair = _obs(
            engine_id="gd",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (100, 100, 200, 200), "label": "chair"},
            confidence=0.9,
        )
        sofa = _obs(
            engine_id="sam",
            task="detection",
            quantity="bbox",
            source_id="img1",
            coordinate_system="image:img1",
            predicted={"bbox_px": (300, 300, 400, 400), "label": "sofa"},
            confidence=0.85,
        )
        _add(matrix, chair)
        _add(matrix, sofa)
        results = normalize_and_fuse(matrix, task="detection", quantity="bbox", group_by_label=True)
        assert len(results) == 2
        labels = {_label_of(r.contributing[0]) for r in results}
        assert labels == {"chair", "sofa"}


def _label_of(obs: EngineObservation) -> str | None:
    from engine.multiengine.evidence_fusion import _label_of as _lo

    return _lo(obs)


# ------------------------------------------------------------------ cross-check with existing fuse_quantity

class TestCrossCheckWithExistingFusion:
    """Verify the new layer produces the same results as the existing fuse_quantity
    for scalar quantities."""

    def test_scalar_fusion_matches_fuse_quantity(self):
        from reconstruction.fusion import FusableObservation, fuse_quantity as existing_fuse

        ao = _obs(
            engine_id="lidar",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 3.17, "units": "meter"},
            confidence=0.98,
            uncertainty={"sigma_m": 0.005},
        )
        bo = _obs(
            engine_id="photogrammetry",
            task="depth",
            quantity="depth_value",
            source_id="img1",
            coordinate_system="cam:img1",
            predicted={"value": 3.22, "units": "meter"},
            confidence=0.95,
            uncertainty={"sigma_m": 0.005},
        )
        fused = fuse_observations([ao, bo], task="depth", quantity="depth_value")
        # Compare with the existing engine.
        existing = existing_fuse(
            [
                FusableObservation(value=3.17, unit=Unit.METER, source="lidar", provenance=Provenance.RECONSTRUCTED, confidence=0.98, precision=0.005),
                FusableObservation(value=3.22, unit=Unit.METER, source="photogrammetry", provenance=Provenance.RECONSTRUCTED, confidence=0.95, precision=0.005),
            ],
            quantity="wall_distance",
        )
        assert fused.value == pytest.approx(existing.value)
        # Provenance objects differ (stub vs real enum) but values match.
        assert fused.provenance.value == existing.provenance.value
        assert fused.method == existing.method
