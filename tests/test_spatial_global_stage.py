"""Pipeline integration of the global spatial optimization: input contract, structured refusals, scale rule, lineage,
dedup, the real stage runner, profile registration (default profile untouched), solver parity, and the production job path.
"""

from __future__ import annotations

import json
import math

import pytest

from benchmarks.spatial.synth import (
    PoseSim,
    SimConfig,
    rect_room,
    simulate_multiview,
    two_rooms_shared_wall,
)
from reconstruction.spatial.multiview import MultiViewConfig, optimize_multiview
from reconstruction.spatial.stage import (
    HASH_KEY,
    SCHEMA,
    SpatialInputError,
    input_hash,
    parse_inputs,
    run_spatial_global,
)
from tests.test_application_api import (
    client,  # noqa: F401  (pytest fixture used by the job-path tests)
)
from worldstore.store import WorldStore


def _inputs(sc=None, n=6, seed=101, scale="metric", **kw):
    local, prior, _ = simulate_multiview(sc or two_rooms_shared_wall(), SimConfig(n_images=n, seed=seed, p_truncate=0.0, **kw), PoseSim())
    return {"schema": SCHEMA, "scale_state": scale, "frame": "world", "storeys": {"L0": 0.0},
            "poses": [{"image_id": p.image_id, "theta": p.theta, "tx": p.tx, "ty": p.ty, "sigma_theta": p.sigma_theta, "sigma_t": p.sigma_t}
                      for p in prior.values()],
            "observations": [o.to_dict() for o in local]}


# --------------------------------------------------------------------------------------------- input contract

@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.update(schema="v0"), "BAD_SCHEMA"),
    (lambda d: d.pop("poses"), "MISSING_FIELD"),
    (lambda d: d["poses"][0].update(tx=float("nan")), "NON_FINITE"),
    (lambda d: d["poses"][0].update(sigma_t=-1.0), "NON_FINITE"),
    (lambda d: d["observations"][0].update(p0=[float("inf"), 0.0]), "NON_FINITE"),
    (lambda d: d["observations"].append(dict(d["observations"][0])), "DUPLICATE_EVIDENCE_ID"),
    (lambda d: d["poses"].pop(0), "NO_POSE_FOR_FRAME"),
    (lambda d: d["observations"][0].update(frame_id="world"), "FRAME_MISMATCH"),
])
def test_bad_inputs_are_refused_with_a_structured_code(mutate, code):
    d = _inputs()
    mutate(d)
    with pytest.raises(SpatialInputError) as e:
        parse_inputs(d)
    assert e.value.code == code
    out = run_spatial_global(d)
    assert out.status == "failed" and out.facts["error"]["code"] == code and out.world is None


def test_relative_scale_is_never_promoted_to_metres():
    out = run_spatial_global(_inputs(scale="relative"))
    assert out.status == "refused" and out.facts["error"]["code"] == "SCALE_NOT_METRIC" and out.world is None
    assert run_spatial_global(_inputs(scale="unknown")).status == "refused"


# ------------------------------------------------------------------------------------------- run + lineage

def test_run_produces_json_facts_world_and_version_lineage(tmp_path):
    store = WorldStore(tmp_path / "ws")
    a = run_spatial_global(_inputs(), store=store)
    assert a.status in ("succeeded", "degraded") and a.version_id
    json.dumps(a.facts)                                                          # JSON-serialisable end to end
    f = a.facts
    assert f["optimization"]["solver"] in ("dense", "sparse") and f["optimization"]["cost_after"] < f["optimization"]["cost_before"]
    assert f["validation"]["state"] in ("valid_complete", "valid_partial", "uncertain", "incomplete")
    assert set(f["entity_epistemic_status"]) == {"existence", "position", "dimensions", "orientation", "geometry", "topology", "semantic"}
    assert f["sigma_budget_entities"] > 0
    back = store.load_version(a.version_id)
    assert back.metadata[HASH_KEY] == input_hash(_inputs())                      # the input fingerprint rides with the world
    assert back.metadata["spatial_optimization"]["raw"] and back.metadata["spatial_facts"]["input_hash"] == f["input_hash"]

    same = run_spatial_global(_inputs(), store=store, parent=a.version_id)
    assert same.status == "unchanged" and same.version_id == a.version_id        # reprocessing safety: no duplicate version
    assert len(store.list_versions()) == 1

    before = json.dumps(store.load_version(a.version_id).to_dict(), sort_keys=True)
    more = run_spatial_global(_inputs(n=9), store=store, parent=a.version_id)
    assert more.version_id and more.version_id != a.version_id and store.ancestors(more.version_id) == [a.version_id]
    assert json.dumps(store.load_version(a.version_id).to_dict(), sort_keys=True) == before   # history never rewritten


def test_tentative_exclusions_and_outliers_stay_traceable():
    d = _inputs(rect_room(), n=6)
    d["observations"].append({**d["observations"][0], "evidence_id": "extra#lonely", "image_id": d["observations"][0]["image_id"],
                              "frame_id": d["observations"][0]["image_id"], "p0": [0.0, 9.0], "p1": [3.0, 9.0]})
    out = run_spatial_global(d)
    tent = {x["entity_id"]: x for x in out.facts["tentative_excluded"]}
    assert any("extra#lonely" in x["evidence_ids"] and x["views"] == 1 for x in tent.values())   # single view: excluded but recorded
    assert all(set(o) == {"evidence_id", "max_abs_z"} for o in out.facts["outliers"])


# ---------------------------------------------------------------------------------- real stage runner / profile

def _ctx(inputs=None, store=None, parent=None):
    from pathlib import Path

    from engine.pipeline.stages.base import StageContext
    st = {}
    if inputs is not None:
        st.update(spatial_inputs=inputs, spatial_store=store, spatial_parent=parent)
    return StageContext(job_id="job-1", workdir=Path("."), artifacts={}, params={}, state=st)


def test_default_profile_is_unchanged_and_spatial_profile_registers_the_stage_once():
    from engine.pipeline.stages.registry import (
        PIPELINE_PROFILES,
        PROFILE_NAME,
        SPATIAL_PROFILE_NAME,
        build_stages,
    )
    assert "spatial_global" not in PIPELINE_PROFILES[PROFILE_NAME]
    names = PIPELINE_PROFILES[SPATIAL_PROFILE_NAME]
    assert names.count("spatial_global") == 1 and names.index("spatial_global") < names.index("compile_validate")
    assert [n for n in names if n != "spatial_global"] == PIPELINE_PROFILES[PROFILE_NAME]
    stages = build_stages(SPATIAL_PROFILE_NAME)
    assert [s.spec.name for s in stages] == names and not next(s for s in stages if s.spec.name == "spatial_global").spec.required


def test_stage_runner_skips_without_inputs_never_claims_success():
    from engine.pipeline.stages.registry import build_stages
    from engine.pipeline.stages.runner import run_stages
    ctx = _ctx()
    stage = [s for s in build_stages("interior_survey_spatial") if s.spec.name == "spatial_global"]
    m = run_stages(stage, ctx, pipeline_profile="interior_survey_spatial", pipeline_version="t")
    r = m.stages[0]
    assert r.status == "skipped" and "no structural observations" in r.warnings[0]
    assert ctx.state["spatial_facts"]["status"] == "skipped"
    assert m.grade([s.spec for s in stage]) == "succeeded"                        # skipped neither upgrades nor downgrades


def test_stage_runner_runs_the_optimizer_and_reports_metrics(tmp_path):
    from engine.pipeline.stages.registry import build_stages
    from engine.pipeline.stages.runner import run_stages
    store = WorldStore(tmp_path / "ws")
    ctx = _ctx(_inputs(), store)
    stage = [s for s in build_stages("interior_survey_spatial") if s.spec.name == "spatial_global"]
    m = run_stages(stage, ctx, pipeline_profile="interior_survey_spatial", pipeline_version="t")
    r = m.stages[0]
    assert r.status in ("succeeded", "degraded") and "spatial_world" in r.outputs and r.metrics["cost_after"] < r.metrics["cost_before"]
    assert ctx.state["spatial_facts"]["version_id"] and len(store.list_versions()) == 1


def test_bad_inputs_fail_the_optional_stage_only_to_partial_with_the_code():
    from engine.pipeline.stages.registry import build_stages
    from engine.pipeline.stages.runner import run_stages
    d = _inputs(scale="relative")
    stage = [s for s in build_stages("interior_survey_spatial") if s.spec.name == "spatial_global"]
    ctx = _ctx(d)
    m = run_stages(stage, ctx, pipeline_profile="interior_survey_spatial", pipeline_version="t")
    assert m.stages[0].status == "failed" and "SCALE_NOT_METRIC" in m.stages[0].error
    assert m.grade([s.spec for s in stage]) == "partial"                          # optional: never FAILED


# ------------------------------------------------------------------------------------------------ solvers

def test_dense_and_sparse_solvers_agree_on_geometry_and_cost():
    local, prior, true = simulate_multiview(two_rooms_shared_wall(), SimConfig(n_images=20, seed=102), PoseSim())
    d = optimize_multiview(local, prior, MultiViewConfig(solver="dense"))
    s = optimize_multiview(local, prior, MultiViewConfig(solver="sparse"))
    assert d.solver == "dense" and s.solver == "sparse" and d.converged and s.converged
    assert s.cost_after == pytest.approx(d.cost_after, rel=1e-3)
    for k in d.poses:
        assert math.hypot(d.poses[k].tx - s.poses[k].tx, d.poses[k].ty - s.poses[k].ty) < 5e-3
        assert d.poses[k].cov == pytest.approx(s.poses[k].cov, abs=5e-4)           # posterior covariance agrees too
    assert s.jac_nnz < s.n_params * s.n_residuals / 5                            # the Jacobian really is sparse


def test_auto_solver_selects_by_view_count():
    for n, expect in ((6, "dense"), (24, "sparse")):
        local, prior, _ = simulate_multiview(rect_room(), SimConfig(n_images=n, seed=101), PoseSim())
        assert optimize_multiview(local, prior).solver == expect


# --------------------------------------------------------------------------------------- host helpers / job path

def test_host_flag_is_off_by_default_and_ids_cannot_escape_the_store(monkeypatch, tmp_path):
    from apps.api import spatial_global_stage as sg
    monkeypatch.delenv(sg.ENV_FLAG, raising=False)
    assert sg.enabled() is False
    monkeypatch.setenv(sg.ENV_FLAG, "1")
    assert sg.enabled() is True
    for bad in ("../x", "a/b", "", ".hidden", "a" * 200):
        with pytest.raises(sg.SpatialGlobalError):
            sg.store_dir(tmp_path, bad, "w")
        with pytest.raises(sg.SpatialGlobalError):
            sg.store_dir(tmp_path, "t", bad)
    assert sg.facts_for_report(None) == {} and sg.facts_for_report({"status": "skipped"}) == {"spatial_global": {"status": "skipped"}}
    monkeypatch.setenv(sg.ENV_SOLVER, "bogus")
    assert sg.solver() == "auto"


def test_production_job_path_with_flag_records_the_skipped_stage(client, tmp_path, monkeypatch):
    """Through the real API job: flag on -> the spatial profile runs, the stage is recorded as SKIPPED (no producer supplies
    spatial_inputs.v1 yet), the job still succeeds, and the report carries the reason. Nothing is optimized from absent data."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    monkeypatch.setenv("RE_SPATIAL_GLOBAL", "1")
    from tests.test_job_stage_runs_pipeline import STAGES, _first_run, _stages
    _, job = _first_run(client, tmp_path, monkeypatch)
    items = _stages(client, job)
    assert [i["name"] for i in items] == STAGES[:-2] + ["spatial_global"] + STAGES[-2:]
    sg = next(i for i in items if i["name"] == "spatial_global")
    assert sg["status"] == "skipped" and sg["error"] is None
    assert job["status"] == "succeeded"
    facts = (job.get("payload") or {}).get("spatial_global")
    assert facts, sorted(job)
    assert facts["status"] == "skipped" and "producer" in facts["reason"]


def test_production_job_path_runs_the_optimizer_when_a_producer_supplies_inputs(client, tmp_path, monkeypatch):
    """Real API job + a registered producer: the stage optimizes, the world is persisted as a spatial-global version under the
    tenant's store, the grade is unchanged, and the facts (validation state, solver, status counts) reach the job payload."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from apps.api import spatial_global_stage as sg
    monkeypatch.setenv("RE_SPATIAL_GLOBAL", "1")
    from tests.test_job_stage_runs_pipeline import _first_run, _stages
    sg.register_inputs_provider(lambda job: _inputs(n=8))
    try:
        _, job = _first_run(client, tmp_path, monkeypatch)
    finally:
        sg.register_inputs_provider(None)
    assert job["status"] == "succeeded"
    stage = next(i for i in _stages(client, job) if i["name"] == "spatial_global")
    assert stage["status"] in ("succeeded", "degraded") and stage["error"] is None
    facts = job["payload"]["spatial_global"]
    assert facts["optimization"]["cost_after"] < facts["optimization"]["cost_before"] and facts["version_id"]
    assert facts["validation"]["state"] and facts["entity_epistemic_status"]["position"]
    roots = list((tmp_path / "ws" / "spatial-global").glob("*/*"))
    assert len(roots) == 1 and len(WorldStore(roots[0]).list_versions()) == 1


def test_without_the_flag_the_default_job_has_no_spatial_stage(client, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    monkeypatch.delenv("RE_SPATIAL_GLOBAL", raising=False)
    from tests.test_job_stage_runs_pipeline import STAGES, _first_run, _stages
    _, job = _first_run(client, tmp_path, monkeypatch)
    assert [i["name"] for i in _stages(client, job)] == STAGES


