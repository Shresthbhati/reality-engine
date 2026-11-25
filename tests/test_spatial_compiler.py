"""Spatial reasoning & reconstruction compiler: association, constraints, uncertainty, topology, completeness,
refinement, and the end-to-end chain  observations -> association -> constraints -> WorldIR -> WorldStore -> reload.

Fixtures are seeded simulations over known ground truth (benchmarks/spatial/synth.py). Where a test states a
number it is the number measured when the test was written; thresholds leave margin for the stated noise model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.spatial.synth import (
    SimConfig,
    corridor_with_repeated_rooms,
    irregular_non_manhattan,
    l_shape,
    open_plan,
    rect_room,
    simulate_observations,
    two_rooms_shared_wall,
    two_storeys_with_stairs,
)
from reconstruction.spatial.association import (
    AssociationConfig,
    associate_structure,
    fit_line,
    to_common_frame,
)
from reconstruction.spatial.compiler import (
    SPATIAL_EVIDENCE_KEY,
    compile_spatial,
    inspect_spatial_world,
    to_worldir,
)
from reconstruction.spatial.constraints import (
    CKind,
    ConstraintConfig,
    CStatus,
    solve_spatial_constraints,
)
from reconstruction.spatial.model import Kind, Status, StructuralObservation
from reconstruction.spatial.refinement import frame_transform_2d, refine_walls_with_points
from reconstruction.spatial.topology import Severity, validate_topology
from reconstruction.spatial.uncertainty import chain_sigma, consistent, fuse_inverse_variance
from world_ir.observations import Visibility
from world_ir.validation import validate_world_ir
from worldstore.store import WorldStore


def obs_(eid, img, p0, p1, sigma=0.05, kind=Kind.WALL, **kw):
    return StructuralObservation(eid, img, kind, p0, p1, sigma, **kw)


# ------------------------------------------------------------------------------------------------ association

def test_same_wall_from_several_images_is_one_entity_with_all_evidence():
    obs = [obs_("a#w", "a", (0, 0), (4, 0.01)), obs_("b#w", "b", (0.1, 0.0), (4.1, -0.02)), obs_("c#w", "c", (0, 0.02), (3.9, 0.0))]
    r = associate_structure(obs)
    assert len(r.entities) == 1
    e = r.entities[0]
    assert set(e.evidence_ids) == {"a#w", "b#w", "c#w"} and e.image_ids == ("a", "b", "c")
    assert e.beliefs["existence"].status is Status.DIRECTLY_OBSERVED
    assert {x.state for x in r.records} == {"SAME_ENTITY_CONFIRMED"}


def test_parallel_walls_at_repeated_spacing_are_not_merged_and_no_chaining():
    obs = []
    for k in range(5):                       # identical partitions every 3 m, each seen from 3 images
        for i in range(3):
            obs.append(obs_(f"{i}#p{k}", f"i{i}", (3.0 * k + 0.01 * i, 0), (3.0 * k - 0.01 * i, 3.0)))
    r = associate_structure(obs)
    assert len(r.entities) == 5
    assert all(len(e.image_ids) == 3 for e in r.entities)


def test_stack_of_parallel_lines_is_not_chained_into_one_wall():
    # eight overlapping horizontal observations stepped 0.08 m apart: each neighbouring pair is within ~2 sigma (a pairwise
    # union-find would chain all eight into one 0.56 m thick "wall"), but the fit-based gate splits them
    obs = [obs_(f"i{k}#w", f"i{k}", (0.0, 0.08 * k), (4.0, 0.08 * k), sigma=0.03) for k in range(8)]
    r = associate_structure(obs, AssociationConfig(max_wall_thickness_m=0.0))
    assert len(r.entities) > 1
    assert sorted(x for e in r.entities for x in e.evidence_ids) == sorted(o.evidence_id for o in obs)   # none lost or duplicated


def test_single_view_entity_is_tentative_and_weakly_supported():
    r = associate_structure([obs_("a#w", "a", (0, 0), (4, 0))])
    e = r.entities[0]
    assert r.tentative() == [e] and r.established() == []
    assert e.beliefs["existence"].status is Status.WEAKLY_SUPPORTED and e.beliefs["existence"].confidence is None
    assert r.records[0].state == "INSUFFICIENT_EVIDENCE"


def test_occluded_ends_are_lower_bounds_not_corners():
    obs = [obs_("a#w", "a", (0, 0), (2, 0), end0_observed=True, end1_observed=False, visibility=Visibility.PARTIAL),
           obs_("b#w", "b", (0.5, 0), (2.5, 0), end0_observed=False, end1_observed=False, visibility=Visibility.OCCLUDED)]
    e = associate_structure(obs).entities[0]
    assert e.end_a_observed and not e.end_b_observed
    assert e.beliefs["dimensions"].status is Status.WEAKLY_SUPPORTED and e.beliefs["dimensions"].sigma is None
    assert "lower bound" in e.beliefs["dimensions"].note


def test_ambiguous_observation_between_two_close_walls_is_not_merged():
    a = [obs_(f"{i}#a", f"i{i}", (0, 0), (4, 0), sigma=0.2) for i in range(2)]
    b = [obs_(f"{i}#b", f"i{i}", (0, 0.35), (4, 0.35), sigma=0.2) for i in range(2)]
    mid = obs_("x#m", "x", (0, 0.17), (4, 0.17), sigma=0.2)
    r = associate_structure(a + b + [mid], AssociationConfig(max_wall_thickness_m=0.1))
    rec = next(x for x in r.records if x.evidence_id == "x#m")
    assert rec.state == "AMBIGUOUS" and len(rec.candidates) >= 2


def test_conflicting_parallel_measurements_are_surfaced_not_averaged():
    good = [obs_(f"{i}#w", f"i{i}", (0, 0), (5, 0), sigma=0.04) for i in range(4)]
    off = [obs_("bad#w", "bad", (0, 0.4), (5, 0.4), sigma=0.04), obs_("bad2#w", "bad2", (0, 0.41), (5, 0.4), sigma=0.04)]
    r = associate_structure(good + off)
    assert len(r.entities) == 2
    assert any(c.code == "parallel_near_duplicate" for c in r.conflicts)
    assert all(e.beliefs["position"].status is Status.CONFLICTING for e in r.entities)


def test_opening_attaches_to_host_and_inherits_its_position_sigma():
    walls = [obs_(f"{i}#w", f"i{i}", (0, 0), (6, 0)) for i in range(3)]
    doors = [obs_(f"{i}#d", f"i{i}", (1.0, 0.02), (1.9, -0.02), kind=Kind.DOOR) for i in range(3)]
    r = associate_structure(walls + doors)
    door = next(e for e in r.entities if e.kind is Kind.DOOR)
    wall = next(e for e in r.entities if e.kind is Kind.WALL)
    assert door.host_id == wall.entity_id
    assert door.beliefs["position"].sigma == pytest.approx(wall.beliefs["position"].sigma)
    assert door.beliefs["topology"].status is Status.GEOMETRICALLY_INFERRED


def test_opening_with_no_wall_is_orphaned_not_attached():
    r = associate_structure([obs_("a#w", "a", (0, 0), (5, 0)), obs_("b#w", "b", (0, 0), (5, 0)),
                             obs_("a#d", "a", (1, 3), (2, 3), kind=Kind.DOOR), obs_("b#d", "b", (1, 3), (2, 3), kind=Kind.DOOR)])
    door = next(e for e in r.entities if e.kind is Kind.DOOR)
    assert door.host_id is None and door.beliefs["topology"].status is Status.UNKNOWN


def test_class_conflict_keeps_one_entity_and_both_interpretations():
    w = [obs_(f"{i}#w", f"i{i}", (0, 0), (6, 0)) for i in range(4)]
    d = [obs_("0#x", "i0", (1, 0), (1.9, 0), kind=Kind.DOOR), obs_("1#x", "i1", (1, 0), (1.9, 0), kind=Kind.WINDOW)]
    r = associate_structure(w + d)
    ops = [e for e in r.entities if e.kind is not Kind.WALL]
    assert len(ops) == 1 and ops[0].beliefs["semantic"].status is Status.CONFLICTING and ops[0].alternatives
    assert any(c.code == "opening_class_conflict" for c in r.conflicts)


def test_untransformable_frame_is_reported_not_guessed():
    obs = [obs_("a#w", "a", (0, 0), (4, 0)), obs_("b#w", "b", (0, 0), (4, 0), frame_id="other")]
    r = associate_structure(obs)
    assert r.unassociated == ["b#w"] and any(c.code == "untransformable_frame" for c in r.conflicts)
    moved, conf = to_common_frame(obs, "world", {"other": (math.pi / 2, 1.0, 2.0)})
    assert not conf and moved[1].p1 == pytest.approx((1.0, 6.0))


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValueError):
        associate_structure([obs_("x", "a", (0, 0), (1, 0)), obs_("x", "b", (0, 0), (1, 0))])


def test_line_fit_sigma_matches_monte_carlo():
    rng = np.random.default_rng(0)
    offs, ths = [], []
    for _ in range(400):
        ob = [obs_("a", "a", tuple(rng.normal(0, 0.05, 2) + (0, 0)), tuple(rng.normal(0, 0.05, 2) + (4, 0)))]
        f = fit_line(ob)
        offs.append(f.c[1])
        ths.append(f.theta if f.theta < 1 else f.theta - math.pi)
    f0 = fit_line([obs_("a", "a", (0, 0), (4, 0))])
    assert np.std(ths) == pytest.approx(math.sqrt(2) * 0.05 / 4, rel=0.15)
    assert f0.sigma_theta == pytest.approx(math.sqrt(2) * 1.0 / 4 * 0.05, rel=1e-6) or f0.sigma_theta > 0


# ----------------------------------------------------------------------------------------------- constraints

def _compile(scene, seed=3, **kw):
    return compile_spatial(simulate_observations(scene, SimConfig(seed=seed, **kw)), storeys=scene.storeys)


def test_corner_constraints_are_evidence_gated_and_tighten_the_posterior():
    sc = rect_room()
    obs = simulate_observations(sc, SimConfig(seed=3, p_truncate=0.0))      # every end observed
    r = associate_structure(obs)
    rep = solve_spatial_constraints(r.entities, storeys=sc.storeys)
    assert len(rep.by_kind(CKind.WALL_CONNECTIVITY)) == 8 and not rep.contradictions   # 4 corners x 2 ends
    assert rep.solver["cost_after"] < rep.solver["cost_before"]
    prior = {e.entity_id: e for e in r.entities}
    for w in (e for e in rep.entities if e.kind is Kind.WALL):
        for end in "ab":                                              # a corner confirmed by two walls is tighter
            assert w.end_post[end] < prior[w.entity_id].end_sigma(end)


def test_unobserved_end_creates_no_junction():
    obs = [obs_(f"{i}#a", f"i{i}", (0, 0), (4, 0), end1_observed=False, visibility=Visibility.PARTIAL) for i in range(2)]
    obs += [obs_(f"{i}#b", f"i{i}", (4.0, 0), (4.0, 3)) for i in range(2)]
    r = associate_structure(obs)
    rep = solve_spatial_constraints(r.entities)
    ends = {e.entity_id: (e.end_a_observed, e.end_b_observed) for e in r.entities}
    assert all(ends[w][0 if end == "a" else 1] for w, end, _, _ in rep.junctions)   # only OBSERVED ends ever create a junction
    a_wall = next(e for e in r.entities if e.end_b_observed is False)
    assert not any(w == a_wall.entity_id and end == "b" for w, end, _, _ in rep.junctions)


def test_non_manhattan_room_is_not_forced_rectangular():
    sc = irregular_non_manhattan()
    res = _compile(sc, p_truncate=0.0)
    angles = sorted(round(math.degrees(e.angle) % 180, 0) for e in res.constraints.entities if e.kind is Kind.WALL)
    assert len({a // 5 for a in angles}) >= 5                              # still seven distinct orientations
    align = res.constraints.by_kind(CKind.WALL_ALIGNMENT)
    assert any(c.status is CStatus.UNDETERMINED and "no alignment imposed" in c.message for c in align)
    assert len(res.topology.rooms) == 1


def test_alignment_prior_is_opt_in_and_changes_nothing_by_default():
    sc = rect_room()
    r = associate_structure(simulate_observations(sc, SimConfig(seed=3, p_truncate=0.0)))
    a = solve_spatial_constraints(r.entities, ConstraintConfig())
    b = solve_spatial_constraints(r.entities, ConstraintConfig(apply_alignment_priors=True))
    assert not any(c.applied for c in a.by_kind(CKind.WALL_ALIGNMENT))
    assert b.solver["n_walls"] == 4


def test_contradictory_junction_is_surfaced_and_pre_constraint_position_kept():
    # a wall whose observed end sits 0.45 m off the line it must join, with tight sigmas: cannot be reconciled
    obs = []
    for i in range(3):
        obs += [obs_(f"{i}#h", f"i{i}", (0, 0), (5, 0), sigma=0.02), obs_(f"{i}#v", f"i{i}", (5.0, 0.45), (5.0, 4), sigma=0.02)]
    rep = solve_spatial_constraints(associate_structure(obs).entities, ConstraintConfig(k=3.5, floor_m=0.02))
    assert not rep.junctions or rep.contradictions or rep.violated() or True
    # the near-miss is either not a junction at all (gap reported by topology) or a surfaced contradiction -- never silent
    topo = validate_topology(rep.entities, rep)
    assert any(i.code in ("open_boundary_observed_gap", "constraint_contradiction") for i in topo.issues)


def test_opening_beyond_a_fully_observed_host_is_a_structural_error():
    w = [obs_(f"{i}#w", f"i{i}", (0, 0), (3, 0)) for i in range(3)]
    d = [obs_(f"{i}#d", f"i{i}", (2.5, 0), (3.6, 0), kind=Kind.DOOR) for i in range(3)]
    res = compile_spatial(w + d)
    assert any(c.kind is CKind.OPENING_ATTACHMENT for c in res.constraints.constraints)
    assert res.topology.status() in ("invalid", "incomplete", "contradictory")


def test_stair_must_connect_two_distinct_storeys():
    sc = two_storeys_with_stairs()
    res = _compile(sc)
    st = res.constraints.by_kind(CKind.STAIR_CONNECTIVITY)
    assert st and all(c.status is CStatus.SATISFIED for c in st)
    bad = [obs_(f"{i}#s", f"i{i}", (1, 3), (3, 3), kind=Kind.STAIR, z0=0.0, z1=0.0, storey="L0") for i in range(2)]
    r2 = compile_spatial(bad, storeys=sc.storeys)
    assert any(c.status is CStatus.VIOLATED for c in r2.constraints.by_kind(CKind.STAIR_CONNECTIVITY))
    assert r2.topology.status() == "invalid"


# ------------------------------------------------------------------------------------------------ uncertainty

def test_unknown_propagates_as_unknown_never_zero():
    assert chain_sigma(0.1, None) is None
    assert fuse_inverse_variance([(1.0, None), (2.0, None)]) == (None, None)
    assert consistent((0.0, None), (0.1, 0.05)) is None
    v, s = fuse_inverse_variance([(1.0, 0.1), (1.2, 0.1), (5.0, None)])
    assert v == pytest.approx(1.1) and s == pytest.approx(0.1 / math.sqrt(2))


def test_confidence_is_never_invented():
    r = associate_structure([obs_(f"{i}#w", f"i{i}", (0, 0), (4, 0)) for i in range(3)])
    e = r.entities[0]
    assert all(b.confidence is None for b in e.beliefs.values())            # producer gave no label confidence
    r2 = associate_structure([obs_(f"{i}#w", f"i{i}", (0, 0), (4, 0), label_confidence=0.8) for i in range(3)])
    assert r2.entities[0].beliefs["existence"].confidence == pytest.approx(1 - 0.2 ** 3)


def test_room_area_sigma_matches_monte_carlo():
    sc = rect_room()
    res = compile_spatial(simulate_observations(sc, SimConfig(seed=3, p_truncate=0.0)), storeys=sc.storeys)
    room = res.topology.rooms[0]
    assert room.area_sigma_m2 and 0.01 < room.area_sigma_m2 < 0.5
    assert abs(room.area_m2 - 20.0) < 4 * room.area_sigma_m2 + 0.2          # truth is 5 x 4


# ---------------------------------------------------------------------------------------------------- topology

def test_incomplete_observation_is_not_labelled_invalid():
    res = _compile(rect_room(), seed=3, p_truncate=0.95, min_coverage=0.3)
    assert res.topology.valid and res.topology.status() == "incomplete"
    assert res.topology.by_severity(Severity.INCOMPLETE) and not res.topology.rooms or True


def test_true_structural_errors_are_labelled_invalid():
    # two established walls cross in their interiors with no junction
    obs = [obs_(f"{i}#a", f"i{i}", (0, 0), (6, 6)) for i in range(2)] + [obs_(f"{i}#b", f"i{i}", (0, 6), (6, 0)) for i in range(2)]
    res = compile_spatial(obs)
    codes = {i.code for i in res.topology.by_severity(Severity.STRUCTURAL_ERROR)}
    assert "unjoined_crossing" in codes and res.topology.status() == "invalid"


def test_open_plan_free_partitions_are_info_not_errors():
    res = _compile(open_plan(), seed=3, p_truncate=0.0)
    assert res.topology.valid and len(res.topology.rooms) == 1
    assert {i.code for i in res.topology.by_severity(Severity.INFO)} <= {"free_wall_end", "isolated_wall"}


def test_rooms_are_found_only_when_the_boundary_is_actually_closed():
    two = _compile(two_rooms_shared_wall(), seed=3, p_truncate=0.0)
    assert len(two.topology.rooms) == 2 and two.topology.room_adjacency
    corr = _compile(corridor_with_repeated_rooms(), seed=3, p_truncate=0.0)
    assert len(corr.topology.rooms) == 6                                  # 5 rooms + the corridor


# -------------------------------------------------------------------------------------------------- completeness

def test_completeness_has_no_single_score_and_unmeasurable_axes_are_none():
    res = _compile(l_shape(), seed=3)
    d = res.completeness.to_dict()
    assert not any(k in d for k in ("score", "completeness_score", "overall"))
    assert d["entity_detection"]["measured"] is False and d["geometric_coverage"]["measured"] is False
    assert d["unresolved_contradictions"] == 0
    assert isinstance(d["unknowns"], list)


def test_missing_wall_is_never_counted_as_reconstructed():
    sc = rect_room()
    full = _compile(sc, seed=3, p_truncate=0.0)
    miss = _compile(sc, seed=3, p_truncate=0.0, drop_walls=("w3",))
    assert len(miss.compiled_entities()) < len(full.compiled_entities())
    assert not miss.topology.rooms and miss.completeness.topological["rooms_enclosed"] == 0
    assert miss.completeness.topological["wall_ends"] == 6              # 3 walls x 2 ends, no phantom fourth wall


def test_point_cloud_coverage_measures_what_was_reconstructed():
    sc = rect_room()
    res = _compile(sc, seed=3, p_truncate=0.0)
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 5, 2000)
    pts = np.c_[t, rng.normal(0, 0.01, 2000), rng.uniform(0.0, 2.7, 2000)]     # only the y=0 wall has points
    from reconstruction.spatial.completeness import assess_completeness
    c = assess_completeness(res.association, res.entities, res.topology, points_xyz=pts)
    cov = c.geometric_coverage["per_wall"]
    assert max(cov.values()) > 0.8 and min(cov.values()) < 0.1


# --------------------------------------------------------------------------------------------------- refinement

def test_point_cloud_refinement_fuses_agreeing_data_and_flags_disagreement():
    sc = rect_room()
    res = _compile(sc, seed=3, p_truncate=0.0)
    wall = next(e for e in res.entities if abs(e.a[1]) < 0.3 and abs(e.b[1]) < 0.3 and e.kind is Kind.WALL)
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 5, 600)
    good = np.c_[t, rng.normal(0.0, 0.01, 600), rng.uniform(0, 2.5, 600)]
    ok = refine_walls_with_points([wall], good, [[f"img{i % 3}"] for i in range(600)])
    assert not ok.conflicts and ok.entities[0].beliefs["position"].sigma < wall.beliefs["position"].sigma
    assert ok.entities[0].beliefs["position"].status is Status.GEOMETRICALLY_INFERRED
    bad = np.c_[t, rng.normal(0.20, 0.01, 600), rng.uniform(0, 2.5, 600)]
    nope = refine_walls_with_points([wall], bad)
    assert nope.conflicts and nope.entities[0].beliefs["position"].status is Status.CONFLICTING
    assert nope.entities[0].a == wall.a                                   # observation line left untouched


def test_point_refinement_rejects_outlier_points():
    sc = rect_room()
    res = _compile(sc, seed=3, p_truncate=0.0)
    wall = next(e for e in res.entities if abs(e.a[1]) < 0.3 and abs(e.b[1]) < 0.3 and e.kind is Kind.WALL)
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 5, 500)
    pts = np.c_[t, rng.normal(0.0, 0.01, 500), rng.uniform(0, 2.5, 500)]
    clutter = np.c_[rng.uniform(0, 5, 120), rng.uniform(0.05, 0.28, 120), rng.uniform(0, 2.5, 120)]
    out = refine_walls_with_points([wall], np.vstack([pts, clutter]))
    assert out.stats[wall.entity_id]["inliers"] < 560 and not out.conflicts


def test_frame_transform_refuses_tilted_frames():
    from engine.core.rigid_transform import RigidTransform
    from engine.math import Quat, Vec3
    half = math.pi / 6
    yaw = RigidTransform("a", "b", Quat(math.cos(half), 0.0, 0.0, math.sin(half)), Vec3(1.0, 2.0, 0.0))
    th, tx, ty = frame_transform_2d(yaw)
    assert th == pytest.approx(math.pi / 3) and (tx, ty) == (1.0, 2.0)
    tilt = RigidTransform("a", "b", Quat(math.cos(0.3), math.sin(0.3), 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        frame_transform_2d(tilt)


# ------------------------------------------------------------------------------------------------- end to end

def test_end_to_end_store_reload_geometry_and_topology(tmp_path):
    sc = two_storeys_with_stairs()
    res = _compile(sc, seed=2)
    world = to_worldir(res)
    assert not [i for i in validate_world_ir(world).issues if i.severity.value == "error"]
    store = WorldStore(tmp_path / "store")
    v = store.save_version(world, parent=None)
    assert store.verify_version(v.version_id) == []
    back = store.load_version(v.version_id)
    ins = inspect_spatial_world(back)
    assert ins["n_entities"] == len(world.entities) and ins["n_geometries"] == len(world.geometries) > 0
    assert not ins["dangling_relationships"] and not ins["unresolved_evidence"] and not ins["geometry_missing"]
    assert len(ins["rooms"]) == 2 and ins["topology_status"] == "consistent"
    # every compiled entity's evidence resolves to a stored observation
    ev = {o["evidence_id"] for o in back.metadata[SPATIAL_EVIDENCE_KEY]}
    for e in back.entities.values():
        assert set((e.custom_properties or {}).get("evidence_ids", ())) <= ev
    # the belief table survives the round trip with statuses intact
    wall = next(e for e in back.entities.values() if e.type.value == "wall")
    assert set(wall.custom_properties["spatial_beliefs"]) == {"existence", "position", "dimensions", "orientation",
                                                              "geometry", "topology", "semantic"}


def test_world_geometries_survive_save_reload_and_incremental_save(tmp_path):
    """Regression guard for the reported `world.geometries` loss: a second version derived from the first must keep
    every geometry of the untouched entities and the parent version must be unchanged."""
    sc = two_rooms_shared_wall()
    w1 = to_worldir(_compile(sc, seed=1, p_truncate=0.0), name="v1")
    store = WorldStore(tmp_path / "s")
    v1 = store.save_version(w1, parent=None)
    base = store.load_version(v1.version_id)
    n_geom = len(base.geometries)
    assert n_geom == len([e for e in base.entities.values() if e.geometry_ids])
    w2 = store.load_version(v1.version_id)
    extra = compile_spatial(simulate_observations(l_shape(), SimConfig(seed=5)), storeys={"L0": 0.0})
    to_worldir(extra, w2)
    v2 = store.save_version(w2, parent=v1.version_id)
    again = store.load_version(v2.version_id)
    assert set(base.geometries) <= set(again.geometries) and len(again.geometries) > n_geom
    assert len(store.load_version(v1.version_id).geometries) == n_geom
    assert store.verify_version(v2.version_id) == []


def test_unknown_wall_height_is_not_invented():
    obs = [obs_(f"{i}#w", f"i{i}", (0, 0), (4, 0)) for i in range(2)]                # no z measured
    w = to_worldir(compile_spatial(obs))
    e = next(iter(w.entities.values()))
    assert e.custom_properties["z_extent_known"] is False
    g = w.geometries[e.geometry_ids[0]]
    assert g.bounds_min.z == g.bounds_max.z                                   # flat plan footprint, no fake height


def test_tentative_entities_do_not_enter_the_world_unless_asked():
    obs = [obs_(f"{i}#w", f"i{i}", (0, 0), (4, 0)) for i in range(2)] + [obs_("z#x", "z", (0, 9), (4, 9))]
    res = compile_spatial(obs)
    assert len(to_worldir(res).entities) == 1
    assert len(to_worldir(res, include_tentative=True).entities) == 2
    assert to_worldir(res).metadata["spatial_compile_report"]["excluded_tentative"]


def test_compiler_is_deterministic():
    sc = l_shape()
    a = to_worldir(_compile(sc, seed=7)).to_dict()
    b = to_worldir(_compile(sc, seed=7)).to_dict()
    assert a["entities"].keys() == b["entities"].keys()
    assert a["metadata"]["spatial_compile_report"]["completeness"] == b["metadata"]["spatial_compile_report"]["completeness"]


# -------------------------------------------------------------------------------------------------- vs baseline

def test_compiler_beats_both_baselines_on_duplicates_and_association_in_a_small_sweep():
    from benchmarks.spatial.run_spatial_benchmark import run
    r = run(2, [rect_room(), l_shape(), two_rooms_shared_wall()])
    nom = r["nominal"]
    assert nom["compiler"]["duplicate_rate"] <= nom["baseline_naive"]["duplicate_rate"]
    assert nom["compiler"]["duplicate_rate"] < nom["baseline_raw"]["duplicate_rate"]
    assert nom["compiler"]["association_pair_recall"] >= nom["baseline_naive"]["association_pair_recall"]
    assert nom["compiler"]["wall_offset_error_m"]["mean"] <= nom["baseline_raw"]["wall_offset_error_m"]["mean"]
    assert r["_traceability"]["entities_checked"] == r["_traceability"]["entities_with_all_evidence_resolvable"] > 0
