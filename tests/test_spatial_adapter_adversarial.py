"""Adversarial tests for the perception-to-reconstruction adapter and multi-engine stage.

These tests deliberately try to break the system by feeding:
- Invalid scale states
- Missing observations
- Invalid camera poses
- Empty inputs
- Non-metric scale claims
- Corrupt structural observations
- Unavailable engines

All failures must be visible and recoverable. No silent success.
"""

from __future__ import annotations

import pytest

from perception.spatial_adapter import adapt_spatial_inputs_v1
from perception.spatial_inputs import (
    CameraInput,
    ScaleStateInput,
    SpatialInputsV1,
    StructuralInput,
)


# ---------------------------------------------------------------------------
# Scale invariants: never promote relative to metric
# ---------------------------------------------------------------------------

def test_relative_scale_stays_relative():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="relative_uncalibrated", unit="relative", is_metric=False),
        cameras=(), observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None
    assert any("not metric" in r["reason"] for r in rejected)


def test_sfm_gauge_stays_sfm_gauge():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="sfm_gauge", unit="sfm_gauge", is_metric=False),
        cameras=(), observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None
    assert any("not metric" in r["reason"] for r in rejected)


def test_unknown_scale_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="unknown", unit="unknown", is_metric=False),
        cameras=(), observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None


# ---------------------------------------------------------------------------
# Camera invariants: absent pose stays absent
# ---------------------------------------------------------------------------

def test_camera_without_position_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(CameraInput(camera_id="c1", image_id="i1", source_uri="x", position_m=None,
                            orientation_wxyz=None, intrinsics=None, pose_source="unknown"),),
        observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None or len(inputs.get("poses", [])) == 0
    assert any("camera" in r["reason"] for r in rejected)


def test_camera_with_nan_position_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(CameraInput(camera_id="c1", image_id="i1", source_uri="x", position_m=(float("nan"), 0.0, 0.0),
                            orientation_wxyz=(1.0, 0.0, 0.0, 0.0), intrinsics=None, pose_source="registered"),),
        observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None or len(inputs.get("poses", [])) == 0


# ---------------------------------------------------------------------------
# Structural observation invariants: never fabricate geometry
# ---------------------------------------------------------------------------

def test_wall_with_zero_sigma_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="wall_slab", label="wall", evidence_id="e",
                                                payload={"start_xyz_m": (0.0, 0.0, 0.0), "end_xyz_m": (5.0, 0.0, 0.0),
                                                         "measurement_uncertainty_m": 0.0}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert any("wall_slab payload lacks" in r["reason"] for r in rejected)
    assert inputs is None or len(inputs.get("observations", [])) == 0


def test_wall_with_nan_endpoints_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="wall_slab", label="wall", evidence_id="e",
                                                payload={"start_xyz_m": (float("nan"), 0.0, 0.0),
                                                         "end_xyz_m": (5.0, 0.0, 0.0),
                                                         "measurement_uncertainty_m": 0.05}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert any("wall_slab payload lacks" in r["reason"] for r in rejected)


def test_empty_payload_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="wall_slab", label="wall", evidence_id="e",
                                                payload={}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert any("wall_slab payload lacks" in r["reason"] for r in rejected)


def test_opening_not_converted():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="opening", label="door", evidence_id="e",
                                                payload={"opening_class": "door", "estimated_width_m": 0.9}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert any("opening" in r["reason"] for r in rejected)


def test_unknown_structural_kind_rejected():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="magic", label="x", evidence_id="e",
                                                payload={"foo": "bar"}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert any("unhandled structural kind" in r["reason"] for r in rejected)


# ---------------------------------------------------------------------------
# Empty input: no fabricated world
# ---------------------------------------------------------------------------

def test_empty_observations_produce_none():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None


def test_no_cameras_no_walls_produce_none():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(), structural_observations=(),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None


# ---------------------------------------------------------------------------
# Multi-observation: valid + invalid mixed
# ---------------------------------------------------------------------------

def test_mixed_valid_invalid_preserves_valid():
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(
            StructuralInput(observation_id="good", kind="wall_slab", label="wall", evidence_id="ev_good",
                            payload={"start_xyz_m": (0.0, 0.0, 0.0), "end_xyz_m": (3.0, 0.0, 0.0),
                                     "measurement_uncertainty_m": 0.05}),
            StructuralInput(observation_id="bad", kind="wall_slab", label="wall", evidence_id="ev_bad",
                            payload={"start_xyz_m": None, "end_xyz_m": None}),
        ),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is not None
    assert len(inputs["observations"]) == 1
    assert inputs["observations"][0]["evidence_id"] == "ev_good"
    assert len(rejected) == 1
    assert rejected[0]["observation_id"] == "bad"


# ---------------------------------------------------------------------------
# Confidence invariants: never fabricate confidence
# ---------------------------------------------------------------------------

def test_wall_label_confidence_is_none():
    """The adapter must not fabricate confidence values."""
    si = SpatialInputsV1(
        run_id="r", scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=(), observations=(),
        structural_observations=(StructuralInput(observation_id="o", kind="wall_slab", label="wall", evidence_id="e",
                                                payload={"start_xyz_m": (0.0, 0.0, 0.0), "end_xyz_m": (5.0, 0.0, 0.0),
                                                         "measurement_uncertainty_m": 0.05}),),
        cross_view_associations=(), relationships=(), room_hypotheses=(),
    )
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is not None
    assert inputs["observations"][0]["label_confidence"] is None
