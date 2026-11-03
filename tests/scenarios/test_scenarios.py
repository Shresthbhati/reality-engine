"""Adversarial reconstruction scenarios A1-A20 (docs/blueprint/18_VALIDATION_AND_BENCHMARKS.md, task 2.2).

One test per scenario. A scenario is implemented ONLY where an existing synthetic generator / deterministic
fixture can exercise real engine code with a meaningful assertion; every other scenario is an explicit
``pytest.skip("not covered: needs ...")`` -- never a vacuous pass. Run with ``-rs`` to see the skip reasons.

Where a scenario is only partly exercised, the docstring says exactly which part.
"""

from __future__ import annotations

import math

import pytest

from engine.math import Quat, Vec3
from perception.instances.association import associate, identity_metrics, overlap_relation, surface_relation
from reconstruction.evaluation import evaluate_absolute_poses
from reconstruction.scene_identity import PairEvidence, group_images
from registration.cross_session import align_session
from world_ir.observations import AssociationState as S

from tests.test_association import _camera, _obs
from tests.test_pose_evaluation import GT, _apply_gauge, _axis_angle_quat, _poses_from_gt, _Pose, _quat_mul
from tests.test_scene_identity import A_PTS, B_PTS, CAMS_A, CAMS_B, make_scene


# ------------------------------------------------------------------ A1
def test_a01_same_wall_from_different_angles_is_one_surface():
    """Same plane seen from two sides / with noise -> one surface; a parallel wall 4 m away is different."""
    assert surface_relation(((1, 0, 0), 2.0), ((1, 0.02, 0), 2.03)) is S.SAME_SURFACE_CONFIRMED
    assert surface_relation(((1, 0, 0), 2.0), ((-1, 0, 0), -2.02)) is S.SAME_SURFACE_CONFIRMED
    assert surface_relation(((1, 0, 0), 2.0), ((1, 0, 0), 6.0)) is S.DIFFERENT_ENTITY
    obs = [
        _obs("w1", "i1", "wall", 0.0, normal=(1, 0, 0)),
        _obs("w2", "i2", "wall", 0.02, normal=(1, 0, 0)),
        _obs("w3", "i3", "wall", 0.01, normal=(-1, 0, 0)),  # seen from behind
    ]
    r = associate(obs)
    assert len(r.entities) == 1 and len(r.entities[0].image_ids) == 3


# ------------------------------------------------------------------ A2
def test_a02_same_door_in_five_images_is_one_entity_with_five_observations():
    obs = [_obs(f"d{i}", f"img{i}", "door", 2.0 + 0.01 * (i % 3), 1.0, 0.0) for i in range(5)]
    r = associate(obs)
    assert len(r.entities) == 1
    assert len(r.entities[0].observation_ids) == 5 and len(r.entities[0].image_ids) == 5


# ------------------------------------------------------------------ A3
def test_a03_window_partly_visible_in_three_images_is_partially_observed():
    pytest.skip("not covered: needs a per-entity PARTIALLY_OBSERVED coverage state computed from partial "
                "multi-view window observations (no synthetic partial-window fixture; association has no such state)")


# ------------------------------------------------------------------ A4
def test_a04_two_similar_but_distinct_doors_are_two_entities():
    obs = []
    for i in range(4):
        obs += [_obs(f"a{i}", f"img{i}", "door", 0.0 + 0.01 * i), _obs(f"b{i}", f"img{i}", "door", 3.0 - 0.01 * i)]
    r = associate(obs)
    assert len(r.entities) == 2
    m = identity_metrics(r.assignment, {o.observation_id: o.observation_id[0] for o in obs})
    assert m["wrong_merge_rate"] == 0 and m["wrong_split_rate"] == 0


# ------------------------------------------------------------------ A5
def test_a05_overlapping_furniture_in_image_space_is_separate_and_not_intersecting():
    cam = _camera()
    near = ((-0.2, -0.2, 1.8), (0.2, 0.2, 2.2))
    far = ((-0.5, -0.5, 4.5), (0.5, 0.5, 5.5))
    assert overlap_relation(near, far, cam) is S.IMAGE_OCCLUSION  # masks overlap, volumes 2.3 m apart
    obs = []
    for i in range(3):
        obs += [_obs(f"p{i}", f"img{i}", "chair", 0.0 + 0.01 * i), _obs(f"q{i}", f"img{i}", "chair", 0.15 - 0.01 * i)]
    assert len(associate(obs).entities) == 2


# ------------------------------------------------------------------ A6
def test_a06_furniture_occluding_a_wall_leaves_the_wall_continuous():
    """Furniture placed in front of a wall occludes a central span of the wall face.
    The wall remains a single continuous surface (not split into disjoint pieces);
    the occluded span is attributed OCCLUDED/INFERRED while unobstructed spans are OBSERVED."""
    from perception.architecture.adapters import evaluate_wall_continuity_under_occlusion

    # 5.0m wide wall with a 2.0m wide bookcase placed from 1.5m to 3.5m
    wall_len = 5.0
    bookcase_span = (1.5, 3.5)

    res = evaluate_wall_continuity_under_occlusion(wall_len, [bookcase_span], wall_id="wall-north")
    assert res.is_continuous is True
    assert len(res.spans) == 3
    assert res.spans[0].evidence_state == "OBSERVED"
    assert res.spans[0].start_m == 0.0 and res.spans[0].end_m == 1.5
    assert res.spans[1].evidence_state == "OCCLUDED"
    assert res.spans[1].start_m == 1.5 and res.spans[1].end_m == 3.5
    assert res.spans[2].evidence_state == "OBSERVED"
    assert res.spans[2].start_m == 3.5 and res.spans[2].end_m == 5.0
    assert res.observed_fraction == pytest.approx(0.60)
    assert res.overall_evidence_state == "PARTIALLY_OBSERVED"


# ------------------------------------------------------------------ A7
def test_a07_corner_seen_in_one_image_is_kept_at_low_confidence():
    pytest.skip("not covered: needs a single-image corner observation fed through a confidence model; "
                "association.associate() has no confidence output to assert on")


# ------------------------------------------------------------------ A8
def test_a08_opposite_viewing_directions_are_one_room_group():
    """Two cameras on opposite sides of one room seeing the same points group as ONE scene (not two)."""
    cams = {"E": ([3.0, 0.0, 0.5], [0.0, 0.0, 0.0]), "W": ([-3.0, 0.0, 0.5], [0.0, 0.0, 0.0])}
    g = group_images(list(cams), make_scene(A_PTS, cams))
    assert len(g.groups) == 1 and sorted(g.groups[0].members) == ["E", "W"] and g.unlinked == []


# ------------------------------------------------------------------ A9
def test_a09_second_session_of_same_room_attaches_to_the_first_frame():
    """A second capture, reconstructed in its own arbitrary frame, registers onto the first with the REAL
    rigid transform (not identity assumed)."""
    half = math.radians(30.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    shift = Vec3(3.0, -1.5, 0.2)
    cloud_a = [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]
    cloud_b = [Vec3(q.x + shift.x, q.y + shift.y, q.z + shift.z) for q in (quat.rotate(p) for p in cloud_a)]
    rep = align_session(cloud_b, cloud_a, from_session="s2", to_session="s1")
    assert rep.status == "accepted" and rep.method == "icp" and rep.rmse < 0.15
    t = rep.transform
    for pb, pa in zip(cloud_b, cloud_a):
        q = t.rotation.rotate(pb)
        assert math.isclose(q.x + t.translation.x, pa.x, abs_tol=0.15)
        assert math.isclose(q.y + t.translation.y, pa.y, abs_tol=0.15)


# ------------------------------------------------------------------ A10
def test_a10_new_image_with_big_viewpoint_change_registers_or_is_rejected_with_reason():
    pytest.skip("not covered: needs real wide-baseline photos through real COLMAP (slow journey: "
                "tests/integration/test_progressive_product_journey.py HARD window) -- too slow for this suite")


# ------------------------------------------------------------------ A11
def test_a11_no_shared_features_is_rejected_never_force_merged():
    """Disjoint structure: cross-session registration refuses with a reason; image grouping keeps them apart."""
    import numpy as np

    cloud_a = [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]
    far = [Vec3(p.x + 100.0, p.y + 100.0, p.z) for p in cloud_a]
    rep = align_session(far, cloud_a, from_session="s2", to_session="s1")
    assert rep.status == "refused" and rep.transform is None and rep.reason
    cams = {**CAMS_A, **CAMS_B}
    g = group_images(list(cams), make_scene(np.vstack([A_PTS, B_PTS]), cams))
    assert len(g.groups) == 2
    assert {frozenset(x.members) for x in g.groups} == {frozenset(CAMS_A), frozenset(CAMS_B)}


# ------------------------------------------------------------------ A12
def test_a12_missing_wall_added_later_leaves_the_rest_unchanged():
    pytest.skip("not covered: needs a two-pass synthetic room (wall withheld, then added) diffed through "
                "world_delta/WorldStore; no fixture withholds a single wall")


# ------------------------------------------------------------------ A13
def test_a13_mirror_creates_no_phantom_room():
    pytest.skip("not covered: needs a mirror-bearing capture (real photos or a reflective synthetic scene) and a "
                "reflection detector; none exists in synthetic/ or perception/")


# ------------------------------------------------------------------ A14
def test_a14_glass_is_an_opening_not_an_opaque_wall():
    """Glazed / transparent surface observations are classified as an Opening (window/glazing),
    never as an opaque solid wall."""
    from perception.architecture.adapters import classify_glazing_opening

    # Transparent glass window with high translucency
    kind, props = classify_glazing_opening("plane-glass-01", is_translucent=True, specular_sheen=0.85)
    assert kind == "window"
    assert props["glazing"] is True
    assert props["is_opaque"] is False
    assert props["evidence_state"] == "OBSERVED"

    # Opaque drywall
    kind_wall, props_wall = classify_glazing_opening("plane-drywall-01", is_translucent=False, specular_sheen=0.1)
    assert kind_wall == "wall"
    assert props_wall["glazing"] is False
    assert props_wall["is_opaque"] is True


# ------------------------------------------------------------------ A15
def test_a15_blank_white_walls_lower_confidence_without_invented_features():
    pytest.skip("not covered: needs textureless-wall photos reconstructed end to end (real COLMAP) and an "
                "assertion that no features/points are invented")


# ------------------------------------------------------------------ A16
def test_a16_repeated_patterns_do_not_cause_false_merges():
    """Look-alike scenes (many raw matches, no epipolar consistency) stay separate and the risk is recorded;
    a repeated wall 4 m away is a different surface."""
    ids = ["A0", "A1", "B0", "B1"]

    def verify(a, b):
        if a[0] == b[0]:
            return PairEvidence(a, b, 200, 190, 0.95)
        return PairEvidence(a, b, 120, 18, 0.15)

    g = group_images(ids, verify)
    assert len(g.groups) == 2 and g.uncertain_links
    assert all(set(x.members) in ({"A0", "A1"}, {"B0", "B1"}) for x in g.groups)
    assert surface_relation(((1, 0, 0), 2.0), ((1, 0, 0), 6.0)) is S.DIFFERENT_ENTITY


# ------------------------------------------------------------------ A17
def test_a17_blur_and_low_light_are_flagged_with_reasons_not_accepted_as_good():
    """Generated textured frames, then blurred / darkened: capture-quality flags them with reasons.
    (Evidence-level only -- the downstream 'no unjustified dimensions' half is not exercised.)"""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from reconstruction import capture_quality as cq
    from tests.test_capture_quality import _jpg, _scene

    sharp = _scene()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    dark = (sharp * 0.1).astype(np.uint8)
    res = cq.assess_capture([("sharp", _jpg(sharp)), ("blurred", _jpg(blurred)), ("dark", _jpg(dark))])["photos"]
    assert res["sharp"]["metrics"]["blur_laplacian_var"] > 10 * res["blurred"]["metrics"]["blur_laplacian_var"]
    assert res["blurred"]["verdict"] == cq.DROP and res["blurred"]["reasons"]
    assert res["dark"]["verdict"] in (cq.WARN, cq.DROP) and res["dark"]["reasons"]
    assert res["sharp"]["verdict"] != cq.DROP


# ------------------------------------------------------------------ A18
def test_a18_wrong_camera_pose_injected_is_caught_by_the_consistency_check():
    gauge = _axis_angle_quat((0.3, 1.0, -0.2), 1.234)
    good = _poses_from_gt(_apply_gauge(GT, gauge, 2.5, (100.0, -7.0, 3.0)))
    flip = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(160.0))
    bad = [_Pose(p.evidence_id, _quat_mul(flip, p.rotation) if p.evidence_id == "img03.jpg" else p.rotation,
                 p.position) for p in good]
    report = evaluate_absolute_poses(bad, GT)
    assert report["refused"] is False
    assert [o["camera"] for o in report["outliers"]] == ["img03.jpg"]
    assert report["rotation_deg_median"] < 1.0  # healthy majority still measures as truth
    assert not evaluate_absolute_poses(good, GT)["outliers"]  # control: no injected fault, no outlier


# ------------------------------------------------------------------ A19
def test_a19_adjacent_room_proposed_for_merge_is_refused():
    pytest.skip("not covered: needs a merge-proposal API for two adjacent rooms; room_join only connects rooms "
                "through openings, there is no room-merge refusal path to assert on")


# ------------------------------------------------------------------ A20
def test_a20_contradictory_geometry_is_conflicted_and_both_kept():
    pytest.skip("not covered: needs a contradictory-geometry fixture producing a CONFLICT record with both "
                "hypotheses retained (conflicts are only produced inside the job path, no synthetic driver)")
