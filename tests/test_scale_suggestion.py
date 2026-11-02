"""D2 / D2b: known-length scale, spread gate, and the typical-size scale SUGGESTION (user must accept).

Contract under test (Decision 023 amendment): a suggestion is computed only from a detected door with a
measured height; no door => no suggestion (never a default); nothing is applied until the user accepts;
an accepted prior is labelled ESTIMATED; rejecting leaves the world RELATIVE; a real measurement
replaces the prior.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.scale import (
    DOOR_HEIGHT_RANGE_M,
    KNOWN_LENGTH,
    PRIOR_ACCEPTED,
    RATIO_SPREAD_MAX,
    ScaleAnchoringError,
    ScaleReference,
    anchor_metric_scale,
    basis_from_note,
    consolidate_references,
    dimension_state_for_basis,
    gate_ratios,
    suggest_scale_from_world,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


# ------------------------------------------------------------------------------------------ helpers


def _result(scale: float = 1.0) -> ReconstructionResult:
    cams = [ReconstructedCameraPose(evidence_id=e, position=(x * scale, 0.0, 0.0), rotation=(0, 0, 0, 1),
                                    uncertainty=0.0) for e, x in (("A", 0.0), ("B", 0.5), ("C", 1.0))]
    pts = [ReconstructedPoint(position=(i * scale, 1.0, 2.0), track_id=f"t{i}", source_evidence_ids=["A", "B"],
                              uncertainty=0.0) for i in range(5)]
    return ReconstructionResult(points=pts, camera_poses=cams, registration_status="registered")


def _world(state: str | None = "relative", doors=(("d1", 0.93),), fitted=True) -> WorldIR:
    w = WorldIR()
    if state:
        w.metadata["scale"] = {"state": state, "meters_per_unit": None}
    for eid, h in doors:
        props = {"height_m": h, "sill_height_m": 0.0}
        if fitted:
            props.update(bounds_min=[0, 0, 0], bounds_max=[0.4, h, 0.1])
        w.entities[eid] = Entity(id=eid, type=EntityType.DOOR, custom_properties=props)
    return w


# ------------------------------------------------------------------------------------ the suggestion


def test_a_detected_door_yields_a_suggestion_with_range_and_implied_scale():
    s = suggest_scale_from_world(_world())
    assert s is not None and s.source_kind == "door" and s.source_entity_id == "d1"
    assert s.measured_model_units == pytest.approx(0.93)
    assert s.typical_range_m == DOOR_HEIGHT_RANGE_M == (2.0, 2.1)
    assert s.suggested_value_m == pytest.approx(2.05)
    assert s.implied_scale == pytest.approx(2.05 / 0.93)
    assert s.to_dict()["dimension_state_if_accepted"] == "ESTIMATED"


def test_no_door_means_no_suggestion_never_a_default():
    assert suggest_scale_from_world(_world(doors=())) is None


def test_a_door_without_a_fitted_height_is_not_evidence():
    # promotion.py writes a 2.1 default height for openings with no fit: that is not a measurement
    assert suggest_scale_from_world(_world(fitted=False)) is None


def test_windows_and_sill_raised_openings_are_not_doors():
    w = _world(doors=())
    w.entities["win"] = Entity(id="win", type=EntityType.WINDOW,
                               custom_properties={"height_m": 1.2, "bounds_min": [0] * 3, "bounds_max": [1] * 3})
    w.entities["hi"] = Entity(id="hi", type=EntityType.DOOR, custom_properties={
        "height_m": 1.0, "sill_height_m": 0.5, "bounds_min": [0] * 3, "bounds_max": [1] * 3})
    assert suggest_scale_from_world(w) is None


def test_doors_that_disagree_give_no_suggestion():
    assert suggest_scale_from_world(_world(doors=(("d1", 0.5), ("d2", 1.0)))) is None
    assert suggest_scale_from_world(_world(doors=(("d1", 0.93), ("d2", 0.95)))) is not None


def test_a_metric_world_gets_no_suggestion():
    assert suggest_scale_from_world(_world(state="metric")) is None


# ------------------------------------------------------------------------------- anchoring + gating


def test_known_length_anchors_with_the_same_machinery_and_is_measured():
    # the model is 0.8 units/metre^-1: an edge of 0.8 model units that is really 2 m long -> 2.5 m/unit
    ref = ScaleReference("A", "C", 2.0, KNOWN_LENGTH, model_distance=0.8, unit_check_sep=1.0)
    scaled = anchor_metric_scale(_result(), ref)
    assert scaled.state == "metric" and scaled.meters_per_unit == pytest.approx(2.5)
    assert scaled.diagnostics.basis == KNOWN_LENGTH and scaled.diagnostics.dimension_state == "MEASURED"
    assert scaled.result.camera_poses[2].position[0] == pytest.approx(2.5)
    assert basis_from_note(scaled.diagnostics.note) == KNOWN_LENGTH


def test_accepted_prior_is_estimated_not_measured():
    ref = ScaleReference("A", "C", 2.05, PRIOR_ACCEPTED, model_distance=0.93, unit_check_sep=1.0)
    scaled = anchor_metric_scale(_result(), ref)
    assert scaled.meters_per_unit == pytest.approx(2.05 / 0.93)
    assert scaled.diagnostics.dimension_state == "ESTIMATED" == dimension_state_for_basis(PRIOR_ACCEPTED)
    assert "ESTIMATED" in scaled.diagnostics.note and basis_from_note(scaled.diagnostics.note) == PRIOR_ACCEPTED


def test_a_model_whose_unit_changed_refuses_the_stale_length():
    ref = ScaleReference("A", "C", 2.0, KNOWN_LENGTH, model_distance=0.8, unit_check_sep=1.0)
    with pytest.raises(ScaleAnchoringError, match="model unit changed"):
        anchor_metric_scale(_result(scale=3.0), ref)       # a rebuild came back 3x larger


def test_model_space_references_need_a_positive_model_length():
    with pytest.raises(ScaleAnchoringError, match="model_distance"):
        ScaleReference("A", "C", 2.0, KNOWN_LENGTH)
    with pytest.raises(ScaleAnchoringError, match="model_distance"):
        ScaleReference("A", "C", 2.0, PRIOR_ACCEPTED, model_distance=0.0)
    with pytest.raises(ScaleAnchoringError, match="only valid"):
        ScaleReference("A", "C", 2.0, "tape", model_distance=1.0)


def test_spread_gate_accepts_agreeing_lengths_and_rejects_disagreeing_ones():
    med, spread = gate_ratios([2.50, 2.52, 2.48])
    assert med == pytest.approx(2.50) and spread < RATIO_SPREAD_MAX
    with pytest.raises(ScaleAnchoringError, match="disagree"):
        gate_ratios([2.5, 3.5])


def test_consolidation_orders_by_trust_and_drops_disagreeing_known_lengths():
    base = ScaleReference("A", "B", 1.2)
    k1 = ScaleReference("A", "C", 2.0, KNOWN_LENGTH, model_distance=0.8, unit_check_sep=1.0)
    k2 = ScaleReference("A", "C", 2.02, KNOWN_LENGTH, model_distance=0.8, unit_check_sep=1.0)
    bad = ScaleReference("A", "C", 9.0, KNOWN_LENGTH, model_distance=0.8, unit_check_sep=1.0)
    prior = ScaleReference("A", "C", 2.05, PRIOR_ACCEPTED, model_distance=0.93, unit_check_sep=1.0)

    out, notes = consolidate_references([prior, k1, k2, base])
    assert [r.method for r in out] == ["manual_measurement", KNOWN_LENGTH]       # prior replaced by real ones
    assert any("replaced by a real measurement" in n for n in notes)

    out, notes = consolidate_references([k1, bad])                                # spread gate fails
    assert out == [] and any("disagree" in n for n in notes)

    out, _ = consolidate_references([prior])                                      # prior stands alone
    assert out == [prior]


# ----------------------------------------------------------------------------------------- API paths


@pytest.fixture()
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.routes_worlds as rw

    state = {"world": _world()}
    monkeypatch.setattr(rw, "_load_world_or_409", lambda row: state["world"])
    queued = []

    class _Job:
        id = "job_test"

    import apps.api.jobs as jobs

    monkeypatch.setattr(jobs, "enqueue_job", lambda db, *a, **k: queued.append(a) or _Job())

    with TestClient(main_mod.app) as c:
        wid = c.post("/api/worlds", json={"name": "Scale"}).json()["id"]
        import asyncio

        from apps.api.models import Session as Sess
        from apps.api.models import World, WorldVersion

        async def seed():
            async with db_mod.get_sessionmaker()() as db:
                db.add(Sess(id="ses_1", name="s", world_id=wid))
                db.add(WorldVersion(id="v1", world_id=wid, report={"cameras": {"A": [0, 0, 0], "B": [0.5, 0, 0]}}))
                (await db.get(World, wid)).current_version_id = "v1"
                await db.commit()

        asyncio.run(seed())
        yield c, wid, state, queued, Path(tmp_path / "ws")


def test_suggestion_endpoint_reports_the_door_and_applies_nothing(api):
    c, wid, _, queued, _ = api
    r = c.get(f"/api/worlds/{wid}/scale/suggestion").json()
    assert r["scale_state"] == "relative" and r["suggestion"]["source_entity_id"] == "d1"
    assert r["suggestion"]["typical_range_m"] == [2.0, 2.1] and r["rejected"] is False
    assert queued == []                                                    # nothing was queued or stored


def test_no_door_detected_no_suggestion_over_the_api(api):
    c, wid, state, _, _ = api
    state["world"] = _world(doors=())
    r = c.get(f"/api/worlds/{wid}/scale/suggestion").json()
    assert r["suggestion"] is None and "no door" in r["reason"]


def test_accept_path_records_an_estimated_prior_and_queues_the_same_world(api):
    c, wid, _, queued, ws = api
    r = c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED, "source_entity_id": "d1"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["dimension_state_after"] == "ESTIMATED" and len(queued) == 1
    ref = body["reference"]
    assert ref["method"] == PRIOR_ACCEPTED and ref["distance_m"] == pytest.approx(2.05)
    assert ref["model_distance"] == pytest.approx(0.93) and ref["unit_check_sep"] == pytest.approx(0.5)
    from apps.api import worldstore_service as wss

    assert wss.list_scale_references(wid)[0]["method"] == PRIOR_ACCEPTED


def test_accept_refuses_a_size_outside_the_typical_range_and_a_stale_suggestion(api):
    c, wid, _, queued, _ = api
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED, "source_entity_id": "d1",
                                                    "distance_m": 3.0}).status_code == 422
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED,
                                                    "source_entity_id": "nope"}).status_code == 409
    assert queued == []


def test_accept_with_no_door_is_refused(api):
    c, wid, state, queued, _ = api
    state["world"] = _world(doors=())
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED,
                                                    "source_entity_id": "d1"}).status_code == 409
    assert queued == []


def test_reject_path_changes_nothing(api):
    c, wid, _, queued, _ = api
    assert c.post(f"/api/worlds/{wid}/scale/suggestion/reject", json={"source_entity_id": "d1"}).status_code == 200
    r = c.get(f"/api/worlds/{wid}/scale/suggestion").json()
    assert r["rejected"] is True and r["scale_state"] == "relative"
    from apps.api import worldstore_service as wss

    assert wss.list_scale_references(wid) == [] and queued == []         # still RELATIVE: no reference, no job


def test_known_length_api_contract_and_its_spread_gate(api):
    c, wid, _, queued, _ = api
    ok = c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 2.0,
                                                  "point_a": [0, 0, 0], "point_b": [0.8, 0, 0]})
    assert ok.status_code == 202, ok.text
    assert ok.json()["dimension_state_after"] == "MEASURED"
    assert ok.json()["reference"]["model_distance"] == pytest.approx(0.8)
    # a second length that implies a very different scale is refused, not averaged in
    bad = c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 9.0, "model_length": 0.8})
    assert bad.status_code == 422 and "disagree" in bad.text
    # malformed inputs
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": 2.0}).status_code == 422
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": KNOWN_LENGTH, "distance_m": -1.0,
                                                    "model_length": 0.8}).status_code == 422
    assert len(queued) == 1


def test_a_metric_world_never_takes_an_estimate(api):
    c, wid, state, queued, _ = api
    state["world"] = _world(state="metric")
    state["world"].metadata["scale"]["meters_per_unit"] = 2.0
    assert c.post(f"/api/worlds/{wid}/scale", json={"method": PRIOR_ACCEPTED,
                                                    "source_entity_id": "d1"}).status_code == 409
    assert queued == []
