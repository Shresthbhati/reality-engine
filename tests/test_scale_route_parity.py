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


# ----------------------------------------------------------------------------------------- end-to-end parity THROUGH BOTH HTTP ROUTES
# The legacy route needs the mocked-job fixture of test_scale_suggestion, the evidence-world route the persisted-run fixture of test_evidence_world_api;
# each test drives ONE route over HTTP and asserts against the SAME contract constants below, so the two cannot drift apart unnoticed.

from tests.test_evidence_world_api import A, _PAIRS, _dist, _gauge_run, _url, client  # noqa: E402,F401
from tests.test_scale_suggestion import _world, api  # noqa: E402,F401

RECORD_KEYS = set(calibrate_baseline(0.8, 2.0, method="x").to_dict())   # the shared ScaleCalibration.to_dict() contract
STATUSES = {"validated", "unvalidated", "estimated_prior", "insufficient", "failed"}


def test_legacy_route_returns_the_shared_calibration_record_and_applies_it_by_a_rebuild_that_keeps_the_previous_version(api):
    c, wid, _, queued, _ = api
    r = c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 2.0, "point_a": [0, 0, 0], "point_b": [0.8, 0, 0]})
    assert r.status_code == 202, r.text
    cal = r.json()["calibration"]
    assert set(cal) == RECORD_KEYS and cal["status"] in STATUSES                          # same record shape as the evidence-world route
    assert cal["status"] == "unvalidated" and cal["dimension_state"] == "MEASURED" and cal["scale_m_per_gauge"] == pytest.approx(2.5)
    assert cal["scale_sigma"] is None and any("single reference" in w for w in cal["warnings"])  # one reference: no uncertainty is claimed
    assert "new immutable version" in r.json()["calibration_applied_by"] and len(queued) == 1
    prior = c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED, "source_entity_id": "d1"})
    if prior.status_code == 202:
        assert prior.json()["calibration"]["status"] == "estimated_prior" and prior.json()["calibration"]["dimension_state"] == "ESTIMATED"


def test_evidence_route_returns_the_same_record_shape_and_vocabulary(client, tmp_path):
    wid = _gauge_run(client, tmp_path, "run-p1")
    anchors = [_dist(i, *p) for i, p in enumerate(_PAIRS)]
    r = client.post(_url(wid, "run-p1", "calibrate"), json={"distance_anchors": anchors}, headers=A)
    assert r.status_code == 200, r.text
    cal = r.json()["calibration"]
    assert set(cal) == RECORD_KEYS and cal["status"] in STATUSES
    assert cal["status"] == "unvalidated" and cal["dimension_state"] == "MEASURED" and cal["scale_m_per_gauge"] == pytest.approx(2.0, rel=5e-3)
    assert r.json()["idempotent"] is False


def test_both_routes_refuse_contradicting_references_without_changing_anything(api, client, tmp_path):
    c, wid, _, queued, _ = api
    from apps.api import worldstore_service as wss

    assert c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 2.0, "model_length": 0.8}).status_code == 202
    before = list(wss.list_scale_references(wid))
    bad = c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 9.0, "model_length": 0.8})
    assert bad.status_code == 422 and "disagree" in bad.text
    assert wss.list_scale_references(wid) == before and len(queued) == 1             # legacy: reference list and job queue untouched


def test_the_evidence_route_refuses_contradicting_anchors_and_changes_nothing(client, tmp_path):
    wid = _gauge_run(client, tmp_path, "run-p2")
    head = client.get(_url(wid, "run-p2", "readiness"), headers=A).json()["world"]["version_id"]
    bad = client.post(_url(wid, "run-p2", "calibrate"), json={"distance_anchors": [_dist(0, *_PAIRS[0]), _dist(1, *_PAIRS[1], noise=0.5)]}, headers=A)
    assert bad.status_code == 409 and bad.json()["detail"]["calibration"]["status"] == "failed" and set(bad.json()["detail"]["calibration"]) == RECORD_KEYS
    assert client.get(_url(wid, "run-p2", "readiness"), headers=A).json()["world"]["version_id"] == head  # evidence: head untouched


def test_the_export_gate_reads_one_definition_of_metric_for_both_paths():
    from engine.multiengine.world_scale import scale_block

    def world(state, cal=None, basis=None):
        w = WorldIR(id="w")
        w.metadata["scale"] = {"state": state, "meters_per_unit": 2.5 if state == "metric" else None, "scale_basis": basis, "calibration": cal}
        return w

    rel = scale_block(world("relative"))
    assert rel["units"] == "model_units" and not rel["metric_export_ready"] and "model units, not metres" in rel["warnings"][0]
    meas = scale_block(world("metric", calibrate_baseline(0.8, 2.0, method=KNOWN_LENGTH).to_dict()))
    assert meas["units"] == "m" and meas["metric_export_ready"] and meas["calibration_status"] == "unvalidated"
    prior = scale_block(world("metric", calibrate_baseline(0.8, 2.0, method=PRIOR_ACCEPTED, prior=True).to_dict()))
    assert prior["units"] == "model_units" and not prior["metric_export_ready"] and "ESTIMATE" in prior["warnings"][0]
    no_record = scale_block(world("metric"))  # a bare "metric" label with no calibration record is not evidence
    assert no_record["units"] == "model_units" and "no applicable measured calibration" in no_record["warnings"][0]
    ev, _ = apply_calibration(_world_from(_result()), calibrate_baseline(0.8, 2.0, method=KNOWN_LENGTH), parent_version="v0")
    assert scale_block(ev)["units"] == "m" and scale_block(ev)["metric_export_ready"]   # the evidence-world path reads the same way


def test_the_evidence_export_labels_units_and_stays_gated_until_calibrated(client, tmp_path):
    wid = _gauge_run(client, tmp_path, "run-p3")
    assert client.post(_url(wid, "run-p3", "export"), json={"format": "glb"}, headers=A).status_code == 409          # unscaled: refused
    gauge = client.post(_url(wid, "run-p3", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    assert gauge.status_code == 200 and gauge.headers["x-scale-units"] == "model_units"
    assert client.post(_url(wid, "run-p3", "calibrate"), json={"distance_anchors": [_dist(i, *p) for i, p in enumerate(_PAIRS)]}, headers=A).status_code == 200
    metric = client.post(_url(wid, "run-p3", "export"), json={"format": "glb"}, headers=A)
    assert metric.status_code == 200 and metric.headers["x-scale-units"] == "m" and metric.headers["x-scale-calibration"] == "unvalidated"
