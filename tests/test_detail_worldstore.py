"""Detail refinement is part of the World: it must survive WorldStore save -> reload with identical statements,
provenance and refusal facts, diff cleanly between versions, and leave earlier versions untouched.

Real chain (points -> quality -> discovery -> ROI -> refinement -> WorldIR) on the repo's measured-cylinder fixture;
the store is the real WorldStore on disk.
"""

from __future__ import annotations

import json

import pytest

from tests.test_detail_worldir import _empty_world, _roi_from_shell, _shell, _slice_options
from worldstore.store import WorldStore


def _detail_world():
    """A world whose detail entities came through the production slice stage (stage 3.8)."""
    from engine.pipeline.vertical_slice import _detail_stage
    from tests.test_detail_discovery import _report_cameras
    from tests.test_evidence_quality import _result

    world = _empty_world()
    facts = _detail_stage(_result(_shell(0.0, 0.0), _report_cameras()), world, _slice_options(), "metric")
    assert facts["status"] == "ran" and world.entities and world.geometries
    return world


def _canon(world):
    return json.dumps(world.to_dict(), sort_keys=True)


def test_refined_detail_survives_save_and_reload_identically_with_provenance(tmp_path):
    world = _detail_world()
    store = WorldStore(tmp_path / "ws")
    saved = store.save_version(world, parent=None)
    assert store.verify_version(saved.version_id) == []          # hashes clean on read-back
    loaded = store.load_version(saved.version_id)
    assert _canon(loaded) == _canon(world), "the reloaded world differs from the one that was saved"
    assert loaded.metadata["detail"] == world.metadata["detail"]
    for eid, ent in world.entities.items():
        back = loaded.entities[eid]
        assert back.provenance == ent.provenance and back.confidence == ent.confidence
        assert back.statement_state == ent.statement_state and back.geometry_ids == ent.geometry_ids
    for gid, geom in world.geometries.items():
        obs = geom.observations[0]
        back = loaded.geometries[gid].observations[0]
        # "what produced this": algorithm, measured residuals, ROI provenance all come back verbatim
        assert back.sensor_type == obs.sensor_type == "detail_refinement" and back.metadata == obs.metadata


def test_a_refused_roi_persists_as_a_fact_and_never_becomes_geometry(tmp_path):
    from perception.detail.refinement import RefinementOutcome
    from perception.detail.roi import RegionOfInterest
    from perception.detail.worldir import integrate_detail_outcomes

    _, _, rois, outcomes = _roi_from_shell()
    other = RegionOfInterest(
        roi_id="roi-9-9-9", parent_entity_id=None, bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), n_cells=1, n_points=2,
        detail_cells=("9-9-9",), point_ids=("ghost-a", "ghost-b"), budget=rois[0].budget, max_curvature=0.2,
        status="pending", provenance={"voxel_size": 1.0})
    refused = RefinementOutcome(
        roi_id="roi-9-9-9", status="refused",
        reason="evidence unresolvable: 2 of 2 point_ids missing (first: 'ghost-a')",
        refinement=None, fits_attempted=0, fits_succeeded=0, backends_run=())
    world = _empty_world()
    report = integrate_detail_outcomes(rois + [other], outcomes + [refused], world)
    world.metadata["detail"] = {"integration": report.to_dict()}   # how the slice stage records the report
    assert report.refusal_count == 1 and report.entity_ids

    store = WorldStore(tmp_path / "ws")
    v = store.save_version(world, parent=None)
    loaded = store.load_version(v.version_id)
    assert _canon(loaded) == _canon(world)
    refusals = loaded.metadata["detail"]["integration"]["refusals"]
    assert len(refusals) == 1 and "ghost-a" in refusals[0]["reason"]                  # the fact survived
    assert set(loaded.entities) == set(report.entity_ids) and set(loaded.geometries) == set(report.geometry_ids)


def test_diff_between_versions_names_the_detail_and_old_versions_stay_immutable(tmp_path):
    from world_ir.diff import ChangeKind, diff_worlds

    detail = _detail_world()
    base = _empty_world()
    base.name = "site"
    detail.name = "site"
    store = WorldStore(tmp_path / "ws")
    v1 = store.save_version(base, parent=None)
    v1_bytes = _canon(store.load_version(v1.version_id))
    v2 = store.save_version(detail, parent=v1.version_id)

    diff = diff_worlds(store.load_version(v1.version_id), store.load_version(v2.version_id))
    added_e = {d.entity_id for d in diff.entity_diffs if d.kind == ChangeKind.ADDED}
    added_g = {d.geometry_id for d in diff.geometry_diffs if d.kind == ChangeKind.ADDED}
    assert added_e == set(detail.entities) and added_g == set(detail.geometries)
    assert set(v2.changed_entity_ids) == added_e                      # the store recorded the same change set

    # "rollback" is reading the earlier version: it is byte-identical to what was first saved
    assert _canon(store.load_version(v1.version_id)) == v1_bytes
    with pytest.raises(Exception):
        store.save_version(base, parent=None, version_id=v1.version_id)   # versions are never overwritten


def test_running_the_same_chain_twice_saves_byte_identical_worlds(tmp_path):
    a, b = _detail_world(), _detail_world()
    for w in (a, b):                       # WorldIR() mints random world/branch ids; the DETAIL content is what must match
        w.id, w.main_branch_id = "w", "branch-main-w"
    store = WorldStore(tmp_path / "ws")
    va, vb = store.save_version(a, parent=None), store.save_version(b, parent=None)
    assert _canon(store.load_version(va.version_id)) == _canon(store.load_version(vb.version_id))
