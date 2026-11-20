"""Metric-lifting scale safety: nothing defaults to metres, and only an independently anchored scale is physically metric.
SYNTHETIC depth planes and a pinhole camera: these prove the acceptance rules, not any model's accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.adapters.metric_3d_lifter import Metric3DLifterAdapter
from engine.multiengine.contract import Device, EngineError, EngineErrorKind, EngineRequest
from engine.multiengine.metric_lifting import Metric3DLifter, classify_scale
from engine.multiengine.registry import maximum_fidelity_registry
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

BOX = (300.0, 220.0, 500.0, 380.0)


@pytest.fixture
def rig():
    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    depth = np.full((600, 800), 2.5, dtype=np.float32) + np.linspace(0, 0.01, 800, dtype=np.float32)  # a plane with a little relief
    meta = {"intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0, "width": 800, "height": 600},
            "extrinsics": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
            "is_canonical_frame": True, "source": "synthetic_rig", "method": "pinhole_canonical", "coordinate_frame": "world:canonical"}
    return cam, depth, meta


def _lift(rig, **kw):
    cam, depth, _ = rig
    lifter = Metric3DLifter()
    ent = lifter.lift_detection(entity_id="e1", class_name="window", box_2d=BOX, depth_map=depth, camera=cam, **kw)
    return ent, lifter.last_rejection_reason


@pytest.mark.parametrize("state,units,metric,source,cls", [
    ("PHYSICALLY_ANCHORED", "METRES", True, "anchor://survey-1", "physical_metric"),
    ("METRIC_SCALE_ALIGNED", "METRES", True, "anchor://survey-1", "physical_metric"),
    ("METRIC_TRAINED_MODEL", "UNSPECIFIED", True, "", "model_metric"),
    ("MODEL_CALIBRATED", "UNSPECIFIED", True, "", "model_metric"),
    ("METRIC_CALIBRATED", "UNSPECIFIED", True, "", "model_metric"),
    ("SFM_GAUGE", "UNSPECIFIED", False, "", "sfm_gauge"),
    ("SFM_ALIGNED_UNMETRIC", "COLMAP_SFM_UNITS", False, "", "sfm_gauge"),
])
def test_accepted_scale_classes(state, units, metric, source, cls):
    assert classify_scale(state, units, metric, source) == (cls, None)


@pytest.mark.parametrize("state,metric,source,reason", [
    ("UNDECLARED", True, "x", "UNDECLARED_SCALE"),
    ("", True, "x", "UNDECLARED_SCALE"),
    (None, True, "x", "UNDECLARED_SCALE"),
    ("UNKNOWN", True, "x", "UNDECLARED_SCALE"),
    ("RELATIVE_DEPTH", True, "x", "UNSCALED_RELATIVE_DEPTH"),
    ("MODEL_RELATIVE_DEPTH", True, "x", "UNSCALED_RELATIVE_DEPTH"),
    ("RELATIVE_UNSCALED", False, "", "UNSCALED_RELATIVE_DEPTH"),
    ("PHYSICALLY_ANCHORED", False, "anchor://x", "UNSCALED_RELATIVE_DEPTH"),  # claims anchored but not metric: contradictory
    ("PHYSICALLY_ANCHORED", True, "", "PHYSICAL_SCALE_WITHOUT_ANCHOR"),  # a metres claim without its evidence
    ("METRIC_SCALE_ALIGNED", True, "", "PHYSICAL_SCALE_WITHOUT_ANCHOR"),
    ("SOMETHING_ELSE", True, "x", "UNSUPPORTED_SCALE_STATE"),
])
def test_refused_scale_declarations(state, metric, source, reason):
    cls, why = classify_scale(state, "METRES", metric, source)
    assert cls == "" and why is not None and why.startswith(reason)


def test_omitting_every_scale_argument_is_refused_not_assumed_metric(rig):
    ent, why = _lift(rig)  # the old defaults would have produced a METRES entity here
    assert ent is None and why.startswith("UNDECLARED_SCALE")


def test_is_metric_depth_alone_is_not_a_scale_declaration(rig):
    ent, why = _lift(rig, is_metric_depth=True)
    assert ent is None and why.startswith("UNDECLARED_SCALE")


def test_an_anchored_lift_is_the_only_physically_metric_result(rig):
    ent, why = _lift(rig, is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="anchor://survey-1",
                     scale_units="METRES", depth_source="synthetic_test_fixture")
    assert why is None and ent.is_physically_metric is True and ent.scale_units == "METRES" and ent.scale_source == "anchor://survey-1"


def test_a_models_own_metres_claim_is_lifted_but_never_physically_metric(rig):
    ent, why = _lift(rig, is_metric_depth=True, scale_state="METRIC_TRAINED_MODEL", scale_units="METRES", depth_source="synthetic_test_fixture")
    assert why is None and ent.is_physically_metric is False and ent.scale_units == "MODEL_ESTIMATED_METRES_UNVERIFIED"


def test_sfm_gauge_lifts_stay_in_gauge_units_not_metres(rig):
    ent, why = _lift(rig, is_metric_depth=False, scale_state="SFM_ALIGNED_UNMETRIC", scale_units="COLMAP_SFM_UNITS",
                     depth_source="synthetic_test_fixture")
    assert why is None and ent.is_physically_metric is False and ent.scale_units == "COLMAP_SFM_UNITS"


def test_physical_metres_without_an_anchor_reference_are_refused(rig):
    ent, why = _lift(rig, is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_units="METRES")
    assert ent is None and why.startswith("PHYSICAL_SCALE_WITHOUT_ANCHOR")


# ------------------------------------------------------------------------------------------------ the adapter (the engine path)
def _adapter():
    a = Metric3DLifterAdapter(maximum_fidelity_registry().get("sam"))
    a.load(Device("cpu"))
    return a


def _req(rig, **payload):
    _, depth, meta = rig
    base = {"camera": meta, "depth_map": depth, "detections": [{"entity_id": "w1", "label": "window", "box_2d": BOX}]}
    base.update(payload)
    return EngineRequest(sources=("img.jpg",), payload=base)


def test_the_adapter_refuses_a_request_with_no_scale_declaration(rig):
    with pytest.raises(EngineError) as ei:
        _adapter().infer(_req(rig))
    assert ei.value.kind is EngineErrorKind.INVALID_INPUT and "scale_state" in str(ei.value)


def test_the_adapter_no_longer_treats_a_missing_is_metric_depth_as_true(rig):
    with pytest.raises(EngineError) as ei:  # declared model-metric but is_metric_depth omitted -> not metric -> rejected, loudly
        _adapter().infer(_req(rig, scale_state="METRIC_TRAINED_MODEL", depth_source="synthetic_test_fixture"))
    assert "UNSCALED_RELATIVE_DEPTH" in str(ei.value)


def test_the_adapter_lifts_with_the_declared_scale(rig):
    a = _adapter()
    drafts = a.infer(_req(rig, is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="anchor://survey-1",
                          scale_units="METRES", depth_source="synthetic_test_fixture"))
    assert drafts and a.rejections == []


def test_rejected_detections_are_kept_with_their_reason_not_dropped_silently(rig):
    a = _adapter()
    good = {"entity_id": "w1", "label": "window", "box_2d": BOX}
    tiny = {"entity_id": "w3", "label": "window", "box_2d": (0.0, 0.0, 2.0, 2.0)}  # too few depth points to lift
    drafts = a.infer(_req(rig, is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="anchor://s",
                          depth_source="synthetic_test_fixture", detections=[good, tiny]))
    assert drafts  # the good one lifted
    assert a.rejections and all(r["reason"] and r["entity_id"] == "w3" for r in a.rejections)
