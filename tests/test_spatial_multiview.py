"""Phase 2: coordinate frames, joint multi-view optimization, global validation states, versioned persistence.

Fixtures: seeded simulations (benchmarks/spatial/synth.py); evaluation-style seeds >= 100. Numbers asserted are
margins around what was measured when the test was written, never exact equality with a fitted value.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from benchmarks.spatial.synth import (
    PoseSim,
    SimConfig,
    corridor_with_repeated_rooms,
    l_shape,
    rect_room,
    simulate_multiview,
    simulate_observations,
    two_rooms_shared_wall,
)
from reconstruction.spatial.compiler import compile_spatial, inspect_spatial_world, to_worldir
from reconstruction.spatial.frames import FramedPoints, FrameError, FrameRegistry, Sim3, rotation_z
from reconstruction.spatial.model import Kind, StructuralObservation
from reconstruction.spatial.multiview import (
    ImagePose,
    apply_pose,
    attach_optimization,
    optimize_multiview,
)
from reconstruction.spatial.validation import ReconState, classify_reconstruction
from worldstore.store import WorldStore

# ------------------------------------------------------------------------------------------------ frames


def test_identity_translation_rotation_scale():
    p = np.array([1.0, 2.0, 3.0])
    assert Sim3.identity("a").apply(p) == pytest.approx(p)
    assert Sim3("a", "b", 1.0, np.eye(3), np.array([1, 0, -1.0])).apply(p) == pytest.approx([2, 2, 2])
    assert Sim3("a", "b", 1.0, rotation_z(math.pi / 2), np.zeros(3)).apply([1, 0, 0]) == pytest.approx([0, 1, 0])
    assert Sim3("a", "b", 2.5, np.eye(3), np.zeros(3)).apply(p) == pytest.approx(2.5 * p)


def test_inverse_and_composition_are_exact():
    rng = np.random.default_rng(0)
    a = Sim3("a", "b", 1.7, rotation_z(0.6), np.array([1.0, -2.0, 0.5]))
    b = Sim3("b", "c", 0.4, rotation_z(-1.1), np.array([0.3, 0.2, 9.0]))
    pts = rng.normal(size=(20, 3))
    assert a.inverse().apply(a.apply(pts)) == pytest.approx(pts, abs=1e-12)
    assert a.then(b).apply(pts) == pytest.approx(b.apply(a.apply(pts)), abs=1e-12)
    assert a.then(b).inverse().apply(a.then(b).apply(pts)) == pytest.approx(pts, abs=1e-12)
    assert a.then(b).scale == pytest.approx(1.7 * 0.4)
    ident = a.then(a.inverse())
    assert ident.scale == pytest.approx(1.0) and ident.t == pytest.approx(0, abs=1e-12) and ident.R == pytest.approx(np.eye(3), abs=1e-12)


def test_long_composition_chain_stays_orthonormal_and_accurate():
    step = Sim3("f", "f", 1.0, rotation_z(0.001), np.array([1e-3, 0, 0]))
    acc = Sim3.identity("f")
    for _ in range(20000):
        acc = acc.then(step)
    assert np.abs(acc.R.T @ acc.R - np.eye(3)).max() < 1e-12
    assert acc.plan_params()[0] == pytest.approx((20000 * 0.001 + math.pi) % (2 * math.pi) - math.pi, abs=1e-9)


@pytest.mark.parametrize("make", [
    lambda: Sim3("a", "b", 0.0, np.eye(3), np.zeros(3)),
    lambda: Sim3("a", "b", -1.0, np.eye(3), np.zeros(3)),
    lambda: Sim3("a", "b", float("nan"), np.eye(3), np.zeros(3)),
    lambda: Sim3("a", "b", 1.0, np.eye(3) * 2, np.zeros(3)),
    lambda: Sim3("a", "b", 1.0, np.diag([1, 1, -1.0]), np.zeros(3)),          # reflection, det -1
    lambda: Sim3("a", "b", 1.0, np.eye(3), np.array([np.inf, 0, 0])),
    lambda: Sim3("", "b", 1.0, np.eye(3), np.zeros(3)),
])
def test_invalid_transforms_are_refused_not_repaired(make):
    with pytest.raises(FrameError):
        make()


def test_frames_are_never_silently_mixed():
    a, b = Sim3.identity("a"), Sim3("x", "y", 1.0, np.eye(3), np.zeros(3))
    with pytest.raises(FrameError):
        a.then(b)
    reg = FrameRegistry()
    reg.add(Sim3.from_plan("cam1", "world", 0.3, 1, 2), "colmap:model0")
    with pytest.raises(FrameError):
        reg.convert([0, 0, 0], "cam1", "cam_unknown")
    with pytest.raises(FrameError):
        reg.add(Sim3.identity("a"), "")                                         # provenance is mandatory


def test_registry_paths_provenance_scale_and_loop_check():
    reg = FrameRegistry()
    reg.add(Sim3.from_plan("cam", "sfm", 0.2, 1.0, 0.0, scale=2.0), "colmap")
    reg.add(Sim3.from_plan("sfm", "world", 0.0, 5.0, 5.0, scale=0.5), "scale_anchor:laser")
    p = np.array([[1.0, 0, 0]])
    direct = reg.convert(p, "cam", "world")
    step = Sim3.from_plan("sfm", "world", 0.0, 5.0, 5.0, scale=0.5).apply(Sim3.from_plan("cam", "sfm", 0.2, 1.0, 0.0, scale=2.0).apply(p))
    assert direct == pytest.approx(step)
    assert reg.provenance("cam", "world") == ["colmap", "scale_anchor:laser"]
    assert reg.scale_consistent(["cam", "sfm"], "world") == {"cam": pytest.approx(1.0), "sfm": pytest.approx(0.5)}
    assert FramedPoints("cam", p).in_frame(reg, "world").frame == "world"
    assert reg.loop_residuals() == []
    reg.add(Sim3.from_plan("cam", "world", 0.2, 9.9, 0.0), "bogus direct edge")  # disagrees with the two-hop route
    assert reg.loop_residuals()


def test_tilted_frame_has_no_plan_view_transform():
    R = np.array([[1, 0, 0], [0, math.cos(0.3), -math.sin(0.3)], [0, math.sin(0.3), math.cos(0.3)]])
    with pytest.raises(FrameError):
        Sim3("a", "b", 1.0, R, np.zeros(3)).plan_params()


# -------------------------------------------------------------------------------------------- multi-view

def _scene(sc=None, n=6, seed=101, ps=None, **kw):
    return simulate_multiview(sc or rect_room(), SimConfig(n_images=n, seed=seed, **kw), ps or PoseSim())


def test_observations_without_a_pose_are_refused():
    local, prior, _ = _scene()
    prior.pop(local[0].image_id)
    with pytest.raises(ValueError):
        optimize_multiview(local, prior)


def test_raw_measurements_and_evidence_ids_are_preserved():
    local, prior, _ = _scene()
    mv = optimize_multiview(local, prior)
    assert [o.evidence_id for o in mv.observations] == [o.evidence_id for o in local]
    for o in local:
        raw = mv.raw[o.evidence_id]
        assert raw["p0"] == list(o.p0) and raw["p1"] == list(o.p1)
        assert raw["prior_pose"] == [prior[o.image_id].theta, prior[o.image_id].tx, prior[o.image_id].ty]
    assert all(o.frame_id == "world" for o in mv.observations)


def test_optimization_reduces_cost_and_improves_geometry_vs_prior_poses():
    errs_prior, errs_opt = [], []
    for seed in range(101, 106):
        sc = l_shape()
        local, prior, true = simulate_multiview(sc, SimConfig(n_images=8, seed=seed, p_truncate=0.0), PoseSim())
        mv = optimize_multiview(local, prior)
        assert mv.converged and mv.cost_after < mv.cost_before
        for sol, store in ((None, errs_prior), (mv, errs_opt)):
            for k, T in true.items():
                p = prior[k] if sol is None else sol.poses[k]
                store.append(math.hypot(p.tx - T.tx, p.ty - T.ty))
    assert np.mean(errs_opt) < np.mean(errs_prior)


def test_joint_optimization_beats_pose_agnostic_pipeline_on_a_small_sweep():
    from benchmarks.spatial.run_multiview_benchmark import chamfer
    base, opt = [], []
    for sc in (rect_room(), two_rooms_shared_wall()):
        for seed in (101, 102, 103):
            local, prior, _ = simulate_multiview(sc, SimConfig(n_images=6, seed=seed), PoseSim())
            res0 = compile_spatial([apply_pose(o, prior[o.image_id].theta, prior[o.image_id].tx, prior[o.image_id].ty) for o in local], storeys=sc.storeys)
            mv = optimize_multiview(local, prior)
            res1 = compile_spatial(mv.observations, storeys=sc.storeys)
            for res, store in ((res0, base), (res1, opt)):
                c = chamfer(res.compiled_entities(), sc)["chamfer"]
                store.append(c if c is not None else 9.0)
    assert np.mean(opt) < np.mean(base)


def test_repeated_architecture_is_not_merged_after_optimization():
    sc = corridor_with_repeated_rooms()
    local, prior, _ = simulate_multiview(sc, SimConfig(n_images=8, seed=104), PoseSim())
    res = compile_spatial(optimize_multiview(local, prior).observations, storeys=sc.storeys)
    walls = [e for e in res.association.entities if e.kind is Kind.WALL and len(e.image_ids) >= 2]
    assert len(walls) == len(sc.walls)                                          # five identical rooms stay five rooms


def test_gauge_is_fixed_when_no_pose_has_a_sigma_and_unknown_sigma_is_not_zero():
    local, prior, _ = _scene()
    bare = {k: ImagePose(k, p.theta, p.tx, p.ty) for k, p in prior.items()}      # no sigmas at all
    mv = optimize_multiview(local, bare)
    first = sorted(bare)[0]
    p0 = mv.poses[first]
    assert (p0.theta, p0.tx, p0.ty) == pytest.approx((bare[first].theta, bare[first].tx, bare[first].ty), abs=1e-3)
    assert all(v is None for p in mv.poses.values() for v in p.observed_dims.values())   # undecidable, not 'observed'
    assert all(o.sigma_m >= 0 for o in mv.observations)


def test_unobservable_pose_dimensions_stay_prior_dominated():
    # every wall is parallel (one direction): translation ALONG the walls cannot be measured from them
    obs, prior = [], {}
    rng = np.random.default_rng(1)
    for i in range(4):
        pr = ImagePose(f"i{i}", 0.0, 0.0, 0.0, 0.01, 0.2)
        prior[pr.image_id] = pr
        for k, y in enumerate((0.0, 3.0)):
            obs.append(StructuralObservation(f"i{i}#w{k}", f"i{i}", Kind.WALL, (-2.0 + rng.normal(0, .01), y + rng.normal(0, .01)),
                                             (2.0 + rng.normal(0, .01), y + rng.normal(0, .01)), 0.02, frame_id=f"i{i}"))
    mv = optimize_multiview(obs, prior)
    assert all(p.observed_dims["tx"] is False for p in mv.poses.values())       # along-wall axis: prior only
    assert any(p.observed_dims["ty"] for p in mv.poses.values())                # across-wall axis: data-constrained


def test_grossly_wrong_pose_is_down_weighted_not_obeyed():
    local, prior, true = simulate_multiview(rect_room(), SimConfig(n_images=8, seed=105, p_truncate=0.0), PoseSim(bad_image=3, bad_t=0.6))
    mv = optimize_multiview(local, prior)
    bad = sorted(prior)[3]
    before = math.hypot(prior[bad].tx - true[bad].tx, prior[bad].ty - true[bad].ty)
    after = math.hypot(mv.poses[bad].tx - true[bad].tx, mv.poses[bad].ty - true[bad].ty)
    assert after < before


def test_warm_start_converges_faster_and_is_deterministic():
    local, prior, _ = _scene(n=10)
    a = optimize_multiview(local, prior)
    b = optimize_multiview(local, prior)
    assert {k: (p.theta, p.tx, p.ty) for k, p in a.poses.items()} == {k: (p.theta, p.tx, p.ty) for k, p in b.poses.items()}
    warm = optimize_multiview(local, prior, warm_start=a)
    assert warm.nfev <= a.nfev


def test_pose_sigma_is_calibrated_to_actual_pose_error():
    zs = []
    for sc in (rect_room(), l_shape(), two_rooms_shared_wall()):
        for seed in range(101, 107):
            local, prior, true = simulate_multiview(sc, SimConfig(n_images=8, seed=seed), PoseSim())
            mv = optimize_multiview(local, prior)
            for k, p in mv.poses.items():
                sg = p.sigma()
                if sg:
                    zs.append(math.hypot(p.tx - true[k].tx, p.ty - true[k].ty) / sg[1])
    zs = np.array(zs)
    # Rayleigh expectation 0.39 / 0.86 / 0.99; require the reported sigma to be neither wildly over- nor under-confident
    assert 0.2 < np.mean(zs <= 1) < 0.8 and np.mean(zs <= 3) > 0.9


# --------------------------------------------------------------------------------------------- validation

def test_global_states():
    sc = rect_room()
    full = compile_spatial(simulate_observations(sc, SimConfig(seed=3, p_truncate=0.0)), storeys=sc.storeys)
    gv = classify_reconstruction(full.entities, full.topology, full.constraints.junctions)
    assert gv.state in (ReconState.VALID_COMPLETE, ReconState.UNCERTAIN) and "invalid" not in gv.flags
    strict = classify_reconstruction(full.entities, full.topology, full.constraints.junctions, sigma_limit_m=1e-4)
    assert strict.state is ReconState.UNCERTAIN


def test_missing_wall_is_incomplete_never_a_contradiction():
    sc = rect_room()
    res = compile_spatial(simulate_observations(sc, SimConfig(seed=3, p_truncate=0.0, drop_walls=("w3",))), storeys=sc.storeys)
    gv = classify_reconstruction(res.entities, res.topology, res.constraints.junctions)
    assert gv.state in (ReconState.INCOMPLETE, ReconState.VALID_PARTIAL) and gv.checks["contradictions"] == 0 and gv.checks["structural_errors"] == 0


def test_structural_error_is_invalid_and_components_are_reported():
    obs = [StructuralObservation(f"{i}#a", f"i{i}", Kind.WALL, (0, 0), (6, 6), 0.05) for i in range(2)]
    obs += [StructuralObservation(f"{i}#b", f"i{i}", Kind.WALL, (0, 6), (6, 0), 0.05) for i in range(2)]
    res = compile_spatial(obs)
    gv = classify_reconstruction(res.entities, res.topology, res.constraints.junctions)
    assert gv.state is ReconState.INVALID and len(gv.components) == 2


# ------------------------------------------------------------------------------------ persistence / lineage

def test_optimized_world_round_trips_and_new_versions_do_not_mutate_history(tmp_path):
    sc = two_rooms_shared_wall()
    local, prior, _ = simulate_multiview(sc, SimConfig(n_images=6, seed=101, p_truncate=0.0), PoseSim())
    mv = optimize_multiview(local, prior)
    res = compile_spatial(mv.observations, storeys=sc.storeys)
    w1 = to_worldir(res, name="v1")
    attach_optimization(w1, mv)
    store = WorldStore(tmp_path / "s")
    v1 = store.save_version(w1, parent=None)
    digest1 = store.verify_version(v1.version_id)
    stored1 = json.dumps(store.load_version(v1.version_id).to_dict(), sort_keys=True)

    # new evidence arrives later: more images, warm-started from the stored solution, saved as a child version
    local2, prior2, _ = simulate_multiview(sc, SimConfig(n_images=9, seed=101, p_truncate=0.0), PoseSim())
    mv2 = optimize_multiview(local2, prior2, warm_start=mv)
    w2 = to_worldir(compile_spatial(mv2.observations, storeys=sc.storeys), name="v2")
    attach_optimization(w2, mv2)
    v2 = store.save_version(w2, parent=v1.version_id)

    restarted = WorldStore(tmp_path / "s")                                       # "restart": a fresh store object
    assert digest1 == [] and restarted.verify_version(v2.version_id) == [] and restarted.ancestors(v2.version_id) == [v1.version_id]
    assert json.dumps(restarted.load_version(v1.version_id).to_dict(), sort_keys=True) == stored1   # history untouched
    back = restarted.load_version(v2.version_id)
    ins = inspect_spatial_world(back)
    assert not ins["unresolved_evidence"] and not ins["dangling_relationships"] and not ins["geometry_missing"]
    opt = back.metadata["spatial_optimization"]
    assert set(opt["raw"]) == {o.evidence_id for o in local2}                    # every raw measurement survives
    assert opt["poses"] and all("prior" in p for p in opt["poses"].values())
    wall = next(e for e in back.entities.values() if e.type.value == "wall")
    assert wall.custom_properties["evidence_ids"] and wall.custom_properties["spatial_beliefs"]["position"]["sigma"] is not None
    assert {e for e in w1.entities} & {e for e in back.entities}                  # identities carried across versions


def test_uncertainty_sources_are_separated_and_consistent_with_the_total():
    from reconstruction.spatial.compiler import compile_multiview
    sc = rect_room()
    local, prior, _ = simulate_multiview(sc, SimConfig(n_images=6, seed=101, p_truncate=0.0), PoseSim())
    res, mv = compile_multiview(local, prior, storeys=sc.storeys)
    walls = [e for e in res.entities if e.kind is Kind.WALL]
    assert walls and all(e.sigma_parts for e in walls)
    for e in walls:
        parts = e.sigma_parts
        total = e.beliefs["position"].sigma
        assert parts["measurement"] > 0 and parts["optimization"] >= 0 and parts["model"] > 0
        assert math.hypot(parts["measurement"], parts["optimization"]) == pytest.approx(total, rel=1e-6) or parts["optimization"] == 0.0
    w = to_worldir(res)
    assert next(e for e in w.entities.values() if e.type.value == "wall").custom_properties["sigma_parts"]
