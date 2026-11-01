"""Phase 1 association engine (perception/instances/association.py): adversarial scenarios from the plan's Appendix A
with ground-truth ids, plus the transitive-chaining failure the old union-find resolver has."""

from __future__ import annotations

import random

import pytest

from engine.math import Quat, Vec3
from perception.instances.association import (
    associate,
    identity_metrics,
    labels_compatible,
    observation_from_hypothesis,
    overlap_relation,
    surface_relation,
)
from perception.instances.lifting import ObjectHypothesis3D
from perception.instances.object_resolution import merge_hypotheses
from provenance import Uncertainty
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir.observations import AssociationState, DepthSource, Observation, sigma_for

S = AssociationState


def _obs(oid, image, label, x, y=0.0, z=0.0, sigma=0.05, depth=DepthSource.METRIC, normal=None) -> Observation:
    return Observation(oid, image, label, (x, y, z), sigma, depth, normal=normal)


def test_scenario_2_same_door_in_five_photos_is_one_entity_with_five_refs():
    obs = [_obs(f"d{i}", f"img{i}", "door", 2.0 + 0.01 * (i % 3), 1.0, 0.0) for i in range(5)]
    r = associate(obs)
    assert len(r.entities) == 1
    assert len(r.entities[0].observation_ids) == 5 and len(r.entities[0].image_ids) == 5
    assert [d.state for d in r.decisions].count(S.SAME_ENTITY_CONFIRMED) == 4


def test_scenario_4_two_similar_but_distinct_doors_stay_two_entities():
    obs = []
    for i in range(4):
        obs += [_obs(f"a{i}", f"img{i}", "door", 0.0 + 0.01 * i), _obs(f"b{i}", f"img{i}", "door", 3.0 - 0.01 * i)]
    r = associate(obs)
    assert len(r.entities) == 2
    truth = {o.observation_id: o.observation_id[0] for o in obs}
    m = identity_metrics(r.assignment, truth)
    assert m["wrong_merge_rate"] == 0 and m["wrong_split_rate"] == 0 and m["duplicate_entity_rate"] == 0


def test_scenario_5_two_overlapping_detections_in_one_image_never_merge():
    obs = []
    for i in range(3):
        obs += [_obs(f"p{i}", f"img{i}", "chair", 0.0 + 0.01 * i), _obs(f"q{i}", f"img{i}", "chair", 0.15 - 0.01 * i)]
    r = associate(obs)
    assert len(r.entities) == 2
    for e in r.entities:  # one detection per image per entity
        assert len(e.observation_ids) == len(e.image_ids) == 3
    assert identity_metrics(r.assignment, {o.observation_id: o.observation_id[0] for o in obs})["wrong_merge_rate"] == 0


def test_chain_of_nearby_distinct_objects_does_not_collapse_but_union_find_does():
    # cushions 0.4 m apart, one seen per image: each pair is within the old 0.5 m gate
    xs = [0.4 * i for i in range(4)]

    def hyp(i, x):
        return ObjectHypothesis3D(
            region_id=f"c{i}", evidence_id=f"img{i}", label="cushion", position=Vec3(x, 0, 0),
            bounds_min=Vec3(x - 0.1, -0.1, -0.1), bounds_max=Vec3(x + 0.1, 0.1, 0.1), point_count=50,
            mask_pixel_count=60, confidence=0.8, uncertainty=Uncertainty(confidence=0.8))

    assert len(merge_hypotheses([hyp(i, x) for i, x in enumerate(xs)])) == 1  # the documented old behaviour

    r = associate([_obs(f"c{i}", f"img{i}", "cushion", x, sigma=0.1) for i, x in enumerate(xs)])
    assert len(r.entities) >= 2
    pos = {f"c{i}": x for i, x in enumerate(xs)}
    for e in r.entities:  # no entity spans the whole chain
        member_x = [pos[o] for o in e.observation_ids]
        assert max(member_x) - min(member_x) <= 0.6


def test_equidistant_candidates_are_ambiguous_not_merged():
    obs = [_obs("a1", "imgA", "box", 0.0, sigma=0.2), _obs("a2", "imgA", "box", 0.5, sigma=0.2),
           _obs("b1", "imgB", "box", 0.25, sigma=0.2)]
    r = associate(obs)
    d = next(d for d in r.decisions if d.observation_id == "b1")
    assert d.state is S.AMBIGUOUS and len(d.candidates) == 2
    assert len(r.entities) == 3  # nothing was merged on a coin flip
    prov = next(e for e in r.entities if "b1" in e.observation_ids)
    assert set(prov.ambiguous_with) == {c[0] for c in d.candidates}


def test_relative_depth_widens_uncertainty_so_match_is_loose_where_metric_is_tight():
    sm, sr = sigma_for(DepthSource.METRIC, 3.0), sigma_for(DepthSource.RELATIVE, 3.0)
    assert sr > 10 * sm
    metric = associate([_obs("m1", "i1", "vase", 0.0, z=3.0, sigma=sm), _obs("m2", "i2", "vase", 0.3, z=3.0, sigma=sm)])
    relative = associate([_obs("r1", "i1", "vase", 0.0, z=3.0, sigma=sr, depth=DepthSource.RELATIVE),
                          _obs("r2", "i2", "vase", 0.3, z=3.0, sigma=sr, depth=DepthSource.RELATIVE)])
    assert len(metric.entities) == 2 and len(relative.entities) == 1


def test_label_hierarchy_allows_chair_armchair_but_not_chair_table():
    assert labels_compatible("chair", "armchair") and not labels_compatible("chair", "table")
    assert len(associate([_obs("x1", "i1", "chair", 0.0), _obs("x2", "i2", "armchair", 0.01)]).entities) == 1
    assert len(associate([_obs("y1", "i1", "chair", 0.0), _obs("y2", "i2", "table", 0.01)]).entities) == 2


def test_orientation_gate_separates_perpendicular_surfaces_at_the_same_spot():
    a = _obs("w1", "i1", "wall", 0.0, normal=(1, 0, 0))
    b = _obs("w2", "i2", "wall", 0.0, normal=(0, 0, 1))
    flipped = _obs("w3", "i3", "wall", 0.0, normal=(-1, 0, 0))  # same wall seen from behind
    assert len(associate([a, b]).entities) == 2
    assert len(associate([a, flipped]).entities) == 1


def test_scenario_1_16_surface_relation_same_wall_vs_parallel_walls():
    assert surface_relation(((1, 0, 0), 2.0), ((1, 0.02, 0), 2.03)) is S.SAME_SURFACE_CONFIRMED
    assert surface_relation(((1, 0, 0), 2.0), ((-1, 0, 0), -2.02)) is S.SAME_SURFACE_CONFIRMED  # seen from behind
    assert surface_relation(((1, 0, 0), 2.0), ((1, 0, 0), 6.0)) is S.DIFFERENT_ENTITY  # repeated wall 4 m away
    assert surface_relation(((1, 0, 0), 2.0), ((0, 0, 1), 2.0)) is S.DIFFERENT_ENTITY


def _camera() -> PinholeCamera:
    return PinholeCamera(
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=160.0, cy=120.0, width=320, height=240),
        extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def test_image_overlap_is_occlusion_not_intersection_unless_the_volumes_touch():
    cam = _camera()
    near = ((-0.2, -0.2, 1.8), (0.2, 0.2, 2.2))
    far = ((-0.5, -0.5, 4.5), (0.5, 0.5, 5.5))
    assert overlap_relation(near, far, cam) is S.IMAGE_OCCLUSION  # masks overlap in the image, 2.3 m apart in depth
    assert overlap_relation(near, far) is S.DIFFERENT_ENTITY  # without a camera there is no image overlap to explain
    floor = ((-2.0, -0.05, 1.0), (2.0, 0.0, 4.0))
    wall = ((-2.0, 0.0, 4.0), (2.0, 2.5, 4.05))
    assert overlap_relation(floor, wall, cam) is S.PHYSICAL_INTERSECTION  # wall meets floor
    assert overlap_relation(((-3, -0.2, 2), (-2.5, 0.2, 2.4)), ((2.5, -0.2, 2), (3, 0.2, 2.4)), cam) is S.DIFFERENT_ENTITY


def test_decisions_carry_reasons_and_serialise_and_order_does_not_change_the_grouping():
    obs = [_obs(f"o{i}", f"img{i}", "lamp", 1.0 + 0.01 * i) for i in range(4)] + [_obs("far", "img9", "lamp", 5.0)]
    base = associate(obs)
    assert all(d.reason for d in base.decisions)
    assert all(isinstance(d.to_dict()["state"], str) for d in base.decisions)
    shuffled = obs[:]
    random.Random(7).shuffle(shuffled)
    other = associate(shuffled)
    group = lambda r: sorted(sorted(e.observation_ids) for e in r.entities)
    assert group(base) == group(other)


def test_observation_from_hypothesis_uses_depth_source_and_range_for_sigma():
    h = ObjectHypothesis3D(
        region_id="r", evidence_id="e", label="chair", position=Vec3(0, 0, 4), bounds_min=Vec3(-0.1, -0.1, 3.9),
        bounds_max=Vec3(0.1, 0.1, 4.1), point_count=10, mask_pixel_count=10, confidence=0.7,
        uncertainty=Uncertainty(confidence=0.7))
    rel = observation_from_hypothesis(h, DepthSource.RELATIVE, model="midas@dpt", license="MIT")
    met = observation_from_hypothesis(h, DepthSource.METRIC)
    assert rel.sigma_m == pytest.approx(1.0) and met.sigma_m == pytest.approx(0.08)
    assert rel.model == "midas@dpt" and rel.license == "MIT" and rel.image_id == "e"


# --- wiring: resolve_hypotheses is a drop-in for merge_hypotheses ------------------------------------------------

def _hyp(region, evidence, label, x, y=0.0, z=3.0, conf=0.8):
    return ObjectHypothesis3D(
        region_id=region, evidence_id=evidence, label=label, position=Vec3(x, y, z),
        bounds_min=Vec3(x - 0.2, y - 0.2, z - 0.2), bounds_max=Vec3(x + 0.2, y + 0.2, z + 0.2), point_count=50,
        mask_pixel_count=60, confidence=conf, uncertainty=Uncertainty(confidence=conf))


def test_resolve_hypotheses_returns_the_same_candidate_type_with_every_source_kept():
    from perception.instances.association import resolve_hypotheses

    hyps = [_hyp(f"r{i}", f"ev{i}", "door", 2.0 + 0.01 * i) for i in range(5)]
    cands, result = resolve_hypotheses(hyps, DepthSource.METRIC)
    assert len(cands) == 1
    assert cands[0].observation_count == 5 and cands[0].evidence_ids == tuple(f"ev{i}" for i in range(5))
    assert cands[0].bounds_min.x <= 1.8 and cands[0].confidence >= 0.8
    assert len(result.decisions) == 5


def test_resolve_hypotheses_survives_region_ids_repeated_across_images():
    from perception.instances.association import resolve_hypotheses

    hyps = [_hyp("region-0", f"ev{i}", "lamp", 1.0) for i in range(3)]  # same region id in every image
    cands, _ = resolve_hypotheses(hyps, DepthSource.METRIC)
    assert len(cands) == 1 and cands[0].observation_count == 3


def test_crowded_row_of_chairs_measured_union_find_versus_association(capsys):
    """8 identical chairs 0.4 m apart, each seen in 6 noisy views (sigma 2 cm), metric depth. Union-find chains the
    row into one blob; the association engine keeps the chairs apart. Numbers are printed for the record."""
    from perception.instances.association import resolve_hypotheses

    rng = random.Random(1)
    hyps, truth = [], {}
    for v in range(6):
        for k in range(8):
            h = _hyp(f"v{v}-k{k}", f"view{v}", "chair", 0.4 * k + rng.gauss(0, 0.02), rng.gauss(0, 0.02), 3.0 + rng.gauss(0, 0.02))
            hyps.append(h)
            truth[f"view{v}|{h.region_id}"] = k

    def assignment_from(cands):
        return {f"{h.evidence_id}|{h.region_id}": c.candidate_id for c in cands for h in c.source_hypotheses}

    old = identity_metrics(assignment_from(merge_hypotheses(hyps)), truth)
    new_cands, _ = resolve_hypotheses(hyps, DepthSource.METRIC)
    new = identity_metrics(assignment_from(new_cands), truth)
    with capsys.disabled():
        print(f"\nunion_find : {old}\nassociation: {new}")
    assert old["wrong_merge_rate"] > 0.5 and old["duplicate_entity_rate"] < 0  # collapsed the row
    assert new["wrong_merge_rate"] <= 0.05
    assert abs(new["duplicate_entity_rate"]) <= 0.25


def test_pipeline_option_defaults_to_the_association_resolver():
    from engine.pipeline.vertical_slice import VerticalSliceOptions

    assert VerticalSliceOptions().object_resolver == "association"
