"""Cross-route parity: the legacy reconstruction path (anchor_metric_scale) and the evidence-world path (apply_calibration) must scale equivalent
geometry identically and record equivalent provenance, because both now use ONE estimator record and ONE scaling function."""

from __future__ import annotations

import pytest

pytest.importorskip("trimesh")

from engine.multiengine.scale_anchors import ScaleCalibration, calibrate_baseline, scale_xyz  # noqa: E402
from engine.multiengine.world_scale import CAL_KEY, apply_calibration, world_calibration  # noqa: E402
from reconstruction.scale import KNOWN_LENGTH, PRIOR_ACCEPTED, ScaleReference, anchor_metric_scale  # noqa: E402
from tests.test_scale_suggestion import _result  # noqa: E402
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402


def _world_from(result) -> WorldIR:
    """The same geometry as ``result``, expressed as world entities (one per camera pose and point) in the gauge state."""
    w = WorldIR(id="w")
    for k, c in enumerate(result.camera_poses):
        x, y, z = c.position
        w.entities[f"cam-{c.evidence_id}"] = Entity(id=f"cam-{c.evidence_id}", type=EntityType.UNKNOWN, geometry_ids=[f"g-{k}"],
                                                    transform={"position": {"x": x, "y": y, "z": z}, "frame_id": "world", "scale_state": "sfm_gauge"})
        w.geometries[f"g-{k}"] = Geometry(id=f"g-{k}", type=GeometryType.BOX, bounds_min=Vector3(x - .1, y - .1, z - .1), bounds_max=Vector3(x + .1, y + .1, z + .1),
                                          quality_metrics={"scale_state": "sfm_gauge"})
    return w


@pytest.mark.parametrize("method,prior", [(KNOWN_LENGTH, False), (PRIOR_ACCEPTED, True)])
def test_the_same_reference_scales_the_same_geometry_and_records_the_same_provenance_through_both_paths(method, prior):
    result = _result()
    ref = ScaleReference("A", "C", 2.0, method, model_distance=0.8, unit_check_sep=1.0)
    legacy = anchor_metric_scale(result, ref)                      # the legacy reconstruction route
    cal = legacy.calibration
    assert isinstance(cal, ScaleCalibration) and cal.scale_m_per_gauge == pytest.approx(2.5)
    world, rec = apply_calibration(_world_from(result), cal, parent_version="v0")   # the evidence-world route, SAME calibration record
    for pose in legacy.result.camera_poses:  # equivalent geometry scales identically
        got = world.entities[f"cam-{pose.evidence_id}"].transform["position"]
        assert (got["x"], got["y"], got["z"]) == pytest.approx(pose.position)
    assert scale_xyz((1.0, 2.0, 3.0), 2.5) == (2.5, 5.0, 7.5)
    # provenance is equivalent: same scale, same basis, same MEASURED/ESTIMATED distinction, same status
    legacy_rec, world_rec = cal.to_dict(), world_calibration(world)
    assert world_rec == legacy_rec and world_rec["basis"] == method
    assert world_rec["dimension_state"] == ("ESTIMATED" if prior else "MEASURED") == legacy.diagnostics.dimension_state
    assert world_rec["status"] == ("estimated_prior" if prior else "unvalidated")
    assert rec["scale_m_per_gauge"] == pytest.approx(legacy.meters_per_unit)


def test_a_single_measured_baseline_is_unvalidated_and_a_prior_is_never_validated_or_metric_exportable():
    from apps.api.evidence_world_stage import readiness  # noqa: F401  (imported to prove the module loads with the shared reader)

    measured = calibrate_baseline(0.8, 2.0, method=KNOWN_LENGTH)
    prior = calibrate_baseline(0.8, 2.0, method=PRIOR_ACCEPTED, prior=True)
    assert measured.status == "unvalidated" and prior.status == "estimated_prior" and prior.applicable
    assert measured.scale_sigma is None and prior.scale_sigma is None  # no uncertainty is claimed for one reference
    assert all("single reference" in w for w in measured.warnings)


def test_a_legacy_world_and_an_evidence_world_expose_the_calibration_through_one_reader():
    cal = calibrate_baseline(0.8, 2.0, method=KNOWN_LENGTH, anchor_ids=("A", "C"))
    legacy_world = WorldIR(id="l")
    legacy_world.metadata["scale"] = {"state": "metric", "meters_per_unit": 2.5, "calibration": cal.to_dict()}
    ev_world, _ = apply_calibration(_world_from(_result()), cal, parent_version="v0")
    assert world_calibration(legacy_world) == world_calibration(ev_world) == cal.to_dict()
    assert CAL_KEY in ev_world.metadata and world_calibration(WorldIR(id="none")) is None
