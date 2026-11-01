"""D1: every dimension is {value, unit, state, basis, uncertainty_m}; a RELATIVE world never shows metres."""

from __future__ import annotations

import pytest

from provenance import Provenance
from world_ir import measure
from world_ir.measure import MeasureError, dimension_from_metadata, make_dimension
from world_ir.schema_v1 import Measurement

KEYS = {"value", "unit", "state", "basis", "uncertainty_m"}
RELATIVE = {"scale": {"state": "relative"}}
METRIC = {"scale": {"state": "metric", "scale_basis": "camera_baseline", "dimension_state": "MEASURED"}}
PRIOR = {"scale": {"state": "metric", "scale_basis": "prior_accepted_by_user", "dimension_state": "ESTIMATED"}}


def test_shape_and_measured_on_metric_baseline():
    d = dimension_from_metadata(2.5, METRIC, state="MEASURED", basis="camera baseline", uncertainty_m=0.02)
    assert set(d) == KEYS
    assert (d["unit"], d["state"], d["uncertainty_m"]) == ("m", "MEASURED", 0.02)


@pytest.mark.parametrize("meta", [RELATIVE, {}, None, {"scale": {"state": "nominal"}}])
@pytest.mark.parametrize("state", ["MEASURED", "RECONSTRUCTED", "ESTIMATED"])
def test_relative_world_never_shows_metres(meta, state):
    d = dimension_from_metadata(0.93, meta, state=state, basis="x")
    assert d["unit"] == "model_units"
    assert d["state"] != "MEASURED"  # no metric evidence in a model-unit world
    assert dimension_from_metadata(1.0, meta, state=state, basis="x", kind="area")["unit"] == "model_units2"


def test_accepted_prior_is_estimated_until_measured():
    for requested in ("MEASURED", "RECONSTRUCTED", "ESTIMATED"):
        d = dimension_from_metadata(2.05, PRIOR, state=requested, basis="door prior")
        assert (d["unit"], d["state"]) == ("m", "ESTIMATED")
    # a real measurement replaces the prior: dimension_state flips to MEASURED
    assert dimension_from_metadata(2.05, METRIC, state="MEASURED", basis="b")["state"] == "MEASURED"


def test_verified_needs_metric_and_beats_the_prior_cap():
    assert make_dimension(2.0, state="VERIFIED", basis="tape", scale_state="metric",
                          scale_dimension_state="ESTIMATED")["state"] == "VERIFIED"
    with pytest.raises(MeasureError):
        make_dimension(2.0, state="VERIFIED", basis="tape", scale_state="unknown")
    with pytest.raises(MeasureError):
        make_dimension(2.0, state="GUESSED", basis="x", scale_state="metric")
    with pytest.raises(MeasureError):
        make_dimension(float("nan"), state="MEASURED", basis="x", scale_state="metric")


def test_measurement_producer_maps_provenance_to_state():
    from perception.instances.measurement import to_dimension

    est = Measurement(value=1.0, unit="meter", precision=0.05, provenance=Provenance.ESTIMATED, confidence=0.5)
    assert to_dimension(est, METRIC)["state"] == "ESTIMATED"
    rec = Measurement(value=1.0, unit="meter", precision=0.05, provenance=Provenance.RECONSTRUCTED, confidence=0.9)
    assert to_dimension(rec, METRIC)["state"] == "RECONSTRUCTED"
    assert to_dimension(rec, RELATIVE)["unit"] == "model_units"  # the Measurement's "meter" is not trusted


# ---- API -----------------------------------------------------------------------------------------------------------
from tests.test_measurements import _post, client, world_id  # noqa: E402,F401


def test_api_returns_dimension_and_keeps_old_keys(client, world_id):
    row = _post(client, world_id).json()
    assert set(row["dimension"]) == KEYS
    assert row["dimension"]["unit"] == "m" and row["state"] == "MEASURED"
    for k in ("value", "unit", "uncertainty", "method", "scale_state", "points", "kind"):
        assert k in row
    assert row["value"] == pytest.approx(5.0)


def test_api_relative_world_is_model_units_and_reconstructed(client, world_id):
    row = _post(client, world_id, scale_state="unknown").json()
    assert row["dimension"]["unit"] == "model_units" and row["state"] == "RECONSTRUCTED"
    listed = client.get(f"/api/worlds/{world_id}/measurements").json()[0]
    assert listed["dimension"]["unit"] == "model_units"


def test_api_tape_verified_override(client, world_id):
    row = _post(client, world_id, verified_value=4.98).json()
    assert row["state"] == "VERIFIED" and row["dimension"]["value"] == pytest.approx(4.98)
    assert row["value"] == pytest.approx(5.0)  # the computed value is kept beside the tape value
    assert client.get(f"/api/worlds/{world_id}/measurements").json()[0]["state"] == "VERIFIED"
    assert _post(client, world_id, scale_state="unknown", verified_value=1.0).status_code == 422
    assert _post(client, world_id, verified_value=-1).status_code == 422
    assert _post(client, world_id, kind="area", points=[[0, 0, 0], [1, 0, 0], [1, 0, 1]],
                 verified_value=0.5).status_code == 422


def test_api_accepted_prior_world_yields_estimated(client, world_id, monkeypatch):
    import apps.api.routes_measurements as rm

    async def prior(db, wid):
        return "ESTIMATED"

    monkeypatch.setattr(rm, "_scale_dimension_state", prior)
    row = _post(client, world_id).json()
    assert row["state"] == "ESTIMATED" and row["dimension"]["unit"] == "m"
    assert client.get(f"/api/worlds/{world_id}/measurements").json()[0]["state"] == "ESTIMATED"
