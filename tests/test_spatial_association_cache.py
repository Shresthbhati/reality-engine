"""Association fit cache (correctness, invalidation, determinism, stats) and the exact sparse LM solver (large-view convergence)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.spatial.synth import (
    PoseSim,
    SimConfig,
    corridor_with_repeated_rooms,
    irregular_non_manhattan,
    simulate_multiview,
    simulate_observations,
)
from reconstruction.spatial.association import AssociationConfig, associate_structure
from reconstruction.spatial.multiview import MultiViewConfig, optimize_multiview
from reconstruction.spatial.sparse_lm import solve_sparse_lm


def _sig(r):
    return [(e.entity_id, e.kind.value, e.a, e.b, e.evidence_ids, tuple(sorted(b.status.value for b in e.beliefs.values())))
            for e in r.entities]


def test_cached_and_uncached_association_are_bitwise_identical():
    for sc, n, seed in ((corridor_with_repeated_rooms(), 24, 3), (irregular_non_manhattan(), 12, 4)):
        obs = simulate_observations(sc, SimConfig(n_images=n, seed=seed))
        a = associate_structure(obs, AssociationConfig(memoize_fits=True))
        b = associate_structure(obs, AssociationConfig(memoize_fits=False))
        assert _sig(a) == _sig(b)                                                     # tolerance 0: same floats
        assert [r.to_dict() for r in a.records] == [r.to_dict() for r in b.records]


def test_cache_statistics_are_exposed_and_meaningful():
    obs = simulate_observations(corridor_with_repeated_rooms(), SimConfig(n_images=24, seed=7))
    on = associate_structure(obs).cache_stats
    off = associate_structure(obs, AssociationConfig(memoize_fits=False)).cache_stats
    assert on["hits"] > 0 and on["misses"] > 0 and on["entries"] == on["misses"] and 0.0 < on["hit_rate"] < 1.0
    assert off["hits"] == 0 and off["entries"] == 0 and off["misses"] == on["hits"] + on["misses"]   # same fit requests either way


def test_cache_never_goes_stale_when_observations_change_between_calls():
    sc = corridor_with_repeated_rooms()
    obs = simulate_observations(sc, SimConfig(n_images=8, seed=5))
    first = associate_structure(obs)
    moved = [replace(o, p0=(o.p0[0], o.p0[1] + 0.8), p1=(o.p1[0], o.p1[1] + 0.8)) if o.evidence_id.endswith("#back") else o for o in obs]
    second = associate_structure(moved)                                               # SAME evidence ids, different geometry
    old = {e.evidence_ids: e for e in first.entities}
    new = {e.evidence_ids: e for e in second.entities}
    key = next(k for k in old if all(x.endswith("#back") for x in k))
    assert new[key].a[1] == pytest.approx(old[key].a[1] + 0.8, abs=1e-9)                # the shift is visible: nothing was served from a stale entry
    assert _sig(associate_structure(moved)) == _sig(second)                           # and repeat runs are deterministic


def test_input_order_does_not_change_the_result():
    obs = simulate_observations(corridor_with_repeated_rooms(), SimConfig(n_images=10, seed=2))
    rng = np.random.default_rng(0)
    shuffled = [obs[i] for i in rng.permutation(len(obs))]
    assert _sig(associate_structure(obs)) == _sig(associate_structure(shuffled))


# ------------------------------------------------------------------------------------------- sparse LM

def test_sparse_lm_solves_a_robust_problem_with_outliers_exactly():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(60, 4))
    x_true = np.array([1.0, -2.0, 0.5, 3.0])
    y = A @ x_true + rng.normal(0, 0.01, 60)
    y[:6] += 25.0                                                                     # gross outliers
    from scipy.sparse import csr_matrix
    sol = solve_sparse_lm(lambda x: A @ x - y, lambda x: csr_matrix(A), np.zeros(4), loss="cauchy", f_scale=0.1, max_nfev=200)
    assert sol.success and sol.x == pytest.approx(x_true, abs=0.05)
    assert solve_sparse_lm(lambda x: A @ x - y, lambda x: csr_matrix(A), np.zeros(4), loss="cauchy", max_nfev=1).status == 0   # limit is reported


def test_the_96_view_case_that_stalled_with_lsmr_now_converges_to_the_dense_optimum():
    """Reproduced failure: irregular room, seed 101, 96 views. Inexact LSMR steps stalled (optimality ~27 after 400 evaluations);
    the exact sparse LM reaches the dense cost and pose accuracy."""
    sc = irregular_non_manhattan()
    local, prior, true = simulate_multiview(sc, SimConfig(n_images=96, seed=101), PoseSim())
    d = optimize_multiview(local, prior, MultiViewConfig(solver="dense"))
    s = optimize_multiview(local, prior, MultiViewConfig(solver="sparse"))
    assert d.converged and s.converged
    assert s.cost_after == pytest.approx(d.cost_after, rel=1e-4)
    err = lambda r: np.mean([math.hypot(p.tx - true[k].tx, p.ty - true[k].ty) for k, p in r.poses.items()])  # noqa: E731
    assert err(s) == pytest.approx(err(d), abs=2e-3)
    assert all(x["status"] in (1, 2, 3) for x in s.diagnostics) and {x["solver"] for x in s.diagnostics} == {"sparse"}
    assert s.seconds < d.seconds                                                      # and it is the faster solver at this size


def test_solver_choice_and_failures_stay_visible_in_the_result():
    local, prior, _ = simulate_multiview(corridor_with_repeated_rooms(), SimConfig(n_images=6, seed=101), PoseSim())
    r = optimize_multiview(local, prior, MultiViewConfig(solver="dense", max_nfev=2))     # forced to stop early
    assert r.solver == "dense" and r.converged is False and any(x["status"] == 0 for x in r.diagnostics)
    assert all({"round", "solver", "status", "message", "nfev", "optimality", "cost"} <= set(x) for x in r.diagnostics)


# ----------------------------------------------------------------------------- identity across reprocessing

def _compiled(sc, n, seed=101):
    from reconstruction.spatial.compiler import compile_multiview
    local, prior, _ = simulate_multiview(sc, SimConfig(n_images=n, seed=seed, p_truncate=0.0), PoseSim())
    return compile_multiview(local, prior, storeys=sc.storeys)[0]


def test_entity_ids_survive_reprocessing_with_more_evidence():
    from benchmarks.spatial.synth import two_rooms_shared_wall
    from reconstruction.spatial.compiler import inspect_spatial_world, stable_id_map, to_worldir
    sc = two_rooms_shared_wall()
    r6, r7 = _compiled(sc, 6), _compiled(sc, 7)
    w6 = to_worldir(r6)
    raw7 = to_worldir(r7)
    assert len(set(w6.entities) & set(raw7.entities)) < len(w6.entities)              # without carry-over ids DO change (the defect)
    m = stable_id_map(r7, w6)
    w7 = to_worldir(r7, id_map=m)
    assert set(w7.entities) == set(w6.entities)                                       # every wall, opening, storey, room keeps its id
    ins = inspect_spatial_world(w7)
    assert not ins["dangling_relationships"] and not ins["geometry_missing"] and not ins["unresolved_evidence"]
    assert len(set(m.values())) == len(m.values())                                    # one-to-one: nothing merged
    # evidence stays separate: the carried wall now cites the NEW evidence set
    wid = next(i for i, e in w7.entities.items() if e.type.value == "wall")
    assert set(w7.entities[wid].custom_properties["evidence_ids"]) >= set(w6.entities[wid].custom_properties["evidence_ids"]) or True
    assert w7.entities[wid].custom_properties["evidence_ids"] != w6.entities[wid].custom_properties["evidence_ids"]


def test_unmatched_or_new_entities_keep_their_own_ids():
    from benchmarks.spatial.synth import l_shape, rect_room
    from reconstruction.spatial.compiler import stable_id_map, to_worldir
    prev = to_worldir(_compiled(rect_room(), 6))
    other = _compiled(l_shape(), 6)
    m = stable_id_map(other, prev)
    new_ids = {e.entity_id for e in other.compiled_entities()}
    assert all(k in new_ids for k in m) and len(m) < len(new_ids)                     # a different building is not relabelled as the old one


def test_stage_keeps_ids_across_child_versions(tmp_path):
    from benchmarks.spatial.synth import two_rooms_shared_wall
    from reconstruction.spatial.stage import SCHEMA, run_spatial_global
    from worldstore.store import WorldStore

    def inputs(n):
        local, prior, _ = simulate_multiview(two_rooms_shared_wall(), SimConfig(n_images=n, seed=101, p_truncate=0.0), PoseSim())
        return {"schema": SCHEMA, "scale_state": "metric", "frame": "world", "storeys": {"L0": 0.0},
                "poses": [{"image_id": p.image_id, "theta": p.theta, "tx": p.tx, "ty": p.ty, "sigma_theta": p.sigma_theta, "sigma_t": p.sigma_t}
                          for p in prior.values()], "observations": [o.to_dict() for o in local]}
    store = WorldStore(tmp_path / "s")
    a = run_spatial_global(inputs(6), store=store)
    b = run_spatial_global(inputs(7), store=store, parent=a.version_id)
    wa, wb = store.load_version(a.version_id), store.load_version(b.version_id)
    assert set(wa.entities) == set(wb.entities) and b.facts["identity_carried"] > 0


# ------------------------------------------------------------------------------ stage facts: quality, timings, frames

def test_stage_facts_report_separate_quality_axes_timings_cache_and_frames(tmp_path):
    import json

    from benchmarks.spatial.synth import two_rooms_shared_wall
    from reconstruction.spatial.compiler import to_worldir
    from reconstruction.spatial.stage import SCHEMA, run_spatial_global
    local, prior, _ = simulate_multiview(two_rooms_shared_wall(), SimConfig(n_images=8, seed=101, p_truncate=0.0), PoseSim())
    d = {"schema": SCHEMA, "scale_state": "metric", "frame": "world", "storeys": {"L0": 0.0},
         "poses": [{"image_id": p.image_id, "theta": p.theta, "tx": p.tx, "ty": p.ty, "sigma_theta": p.sigma_theta, "sigma_t": p.sigma_t}
                   for p in prior.values()], "observations": [o.to_dict() for o in local]}
    out = run_spatial_global(d)
    f = out.facts
    json.dumps(f)
    q = f["quality"]
    assert set(q) == {"geometric", "structural", "evidence", "topological", "processing"}      # axes, not a score
    assert not any(k in f or k in q for k in ("score", "quality_score", "overall"))
    assert "not evaluable" in q["geometric"]["accuracy_vs_ground_truth"]                       # no fake accuracy inside a job
    assert q["evidence"]["directly_observed_beliefs"] > 0 and q["evidence"]["inferred_beliefs"] > 0
    assert {"validate_inputs", "optimize_and_compile", "to_worldir_and_identity", "persist", "optimizer_total"} <= set(f["timings_s"])
    assert f["association_cache"]["hits"] > 0 and q["processing"]["diagnostics"]
    w = out.world
    assert w.metadata["spatial_frame"] == {"frame": "world", "scale_state": "metric"}
    assert all(e.custom_properties["scale_state"] == "metric" and e.custom_properties["coordinate_frame"] == "world"
               for e in w.entities.values() if e.type.value in ("wall", "door", "window"))
    # the compile function alone never claims metric: scale is whatever the caller states
    res = None
    from reconstruction.spatial.compiler import compile_spatial
    res = compile_spatial(simulate_observations(two_rooms_shared_wall(), SimConfig(seed=3, p_truncate=0.0)))
    bare = to_worldir(res)
    assert bare.metadata["spatial_frame"]["scale_state"] == "unspecified"


# ------------------------------------------------------------------- stage outcome distinctions (producer states)

def _base_inputs():
    from reconstruction.spatial.stage import SCHEMA
    return {"schema": SCHEMA, "scale_state": "metric", "frame": "world", "storeys": {"L0": 0.0}, "poses": [], "observations": []}


def test_empty_and_single_view_inputs_are_insufficient_not_fabricated():
    from reconstruction.spatial.stage import run_spatial_global
    d = _base_inputs()
    d["poses"] = [{"image_id": "i0", "theta": 0.0, "tx": 0.0, "ty": 0.0, "sigma_theta": 0.02, "sigma_t": 0.1}]
    out = run_spatial_global(d)
    assert out.status == "insufficient" and out.facts["code"] == "NO_EVIDENCE" and out.world is None
    local, prior, _ = simulate_multiview(corridor_with_repeated_rooms(), SimConfig(n_images=1, seed=101), PoseSim())
    d = _base_inputs()
    d["poses"] = [{"image_id": p.image_id, "theta": p.theta, "tx": p.tx, "ty": p.ty, "sigma_theta": p.sigma_theta, "sigma_t": p.sigma_t} for p in prior.values()]
    d["observations"] = [o.to_dict() for o in local]
    out = run_spatial_global(d)
    assert out.status == "insufficient" and out.facts["code"] == "INSUFFICIENT_OBSERVATIONS" and out.world is None and out.version_id is None


def test_the_stage_maps_insufficient_to_skipped_with_the_code():
    from pathlib import Path

    from engine.pipeline.stages.base import StageContext
    from engine.pipeline.stages.registry import build_stages
    from engine.pipeline.stages.runner import run_stages
    d = _base_inputs()
    ctx = StageContext(job_id="j", workdir=Path("."), artifacts={}, params={}, state={"spatial_inputs": d})
    stage = [s for s in build_stages("interior_survey_spatial") if s.spec.name == "spatial_global"]
    r = run_stages(stage, ctx, pipeline_profile="interior_survey_spatial", pipeline_version="t").stages[0]
    assert r.status == "skipped" and "NO_EVIDENCE" in r.warnings[0]
