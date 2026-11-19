"""Coordinate alignment and consensus, checked against synthetic scenes whose ground truth is known exactly.

Synthetic evidence proves the algorithms behave as specified on controlled cases (outliers, correlated sources, biased majorities,
degenerate geometry). It is NOT evidence about how any real engine performs on real photographs.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.alignment import (
    AlignmentError, DegenerateGeometry, Frame, FrameGraph, FrameMismatch, Sim3, estimate_similarity, flag_inconsistent_poses,
    pose_residuals, ransac_similarity, scale_mismatch,
)
from engine.multiengine.consensus import (
    ConsensusError, Estimate, LabelClaim, make_reprojection_check, reconcile_labels, reprojection_error, resolve_estimates,
)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot_xyz(ax, ay, az):
    def rx(a):
        a = np.radians(a)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def ry(a):
        a = np.radians(a)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    return rx(ax) @ ry(ay) @ rot_z(az)


TRUE = Sim3.from_arrays(2.5, rot_xyz(20, -35, 70), np.array([1.0, -2.0, 0.5]))


def cloud(n=60, seed=0):
    return np.random.default_rng(seed).uniform(-3, 3, size=(n, 3))


# ------------------------------------------------------------------------------------------------ similarity estimation

def test_similarity_is_recovered_exactly_without_noise_and_closely_with_noise():
    src = cloud()
    dst = TRUE.apply(src)
    est_ = estimate_similarity(src, dst)
    assert est_.scale == pytest.approx(TRUE.scale, rel=1e-9)
    assert np.allclose(est_.R, TRUE.R, atol=1e-9) and np.allclose(est_.t, TRUE.t, atol=1e-8)
    noisy = estimate_similarity(src, dst + np.random.default_rng(1).normal(0, 0.01, dst.shape))
    assert noisy.scale == pytest.approx(TRUE.scale, rel=5e-3)
    assert np.linalg.norm(noisy.t - TRUE.t) < 0.05


def test_rotation_is_proper_even_for_a_mirrored_correspondence_set():
    src = cloud()
    mirrored = estimate_similarity(src, src * np.array([1, 1, -1]))        # no proper rotation maps it exactly
    assert np.linalg.det(mirrored.R) == pytest.approx(1.0)


def test_degenerate_inputs_are_refused_not_invented():
    with pytest.raises(DegenerateGeometry):
        estimate_similarity(cloud(2), cloud(2))
    line = np.outer(np.linspace(0, 1, 10), [1.0, 2.0, 3.0])
    with pytest.raises(DegenerateGeometry):
        estimate_similarity(line, line * 2 + 1)
    with pytest.raises(AlignmentError):
        estimate_similarity(np.array([[0, 0, np.nan]] * 4), cloud(4))
    with pytest.raises(AlignmentError):
        estimate_similarity(cloud(5), cloud(6))


def test_sim3_inverse_and_composition_round_trip():
    p = cloud(10)
    assert np.allclose(TRUE.inverse().apply(TRUE.apply(p)), p, atol=1e-9)
    other = Sim3.from_arrays(0.5, rot_z(33), np.array([4.0, 0, 1]))
    assert np.allclose(TRUE.compose(other).apply(p), other.apply(TRUE.apply(p)), atol=1e-9)


# ------------------------------------------------------------------------------------------------ RANSAC

def test_ransac_recovers_the_transform_despite_thirty_percent_gross_outliers():
    src = cloud(100)
    dst = TRUE.apply(src) + np.random.default_rng(2).normal(0, 0.005, (100, 3))
    bad = np.random.default_rng(3).choice(100, 30, replace=False)
    dst[bad] += np.random.default_rng(4).uniform(2, 6, (30, 3))
    res = ransac_similarity(src, dst, inlier_threshold=0.05, seed=7)
    assert res.ok and res.n_inliers >= 68
    assert not res.inliers[bad].any(), "a corrupted correspondence was accepted as an inlier"
    assert res.transform.scale == pytest.approx(TRUE.scale, rel=5e-3) and res.rms_inlier_residual < 0.02
    again = ransac_similarity(src, dst, inlier_threshold=0.05, seed=7)
    assert np.array_equal(res.inliers, again.inliers)                 # deterministic for a fixed seed


def test_ransac_reports_registration_failure_for_unrelated_point_sets_and_requires_a_tolerance():
    res = ransac_similarity(cloud(40, 1), cloud(40, 2), inlier_threshold=0.01, min_inliers=20)
    assert res.status == "registration_failed" and res.transform is None and res.notes
    with pytest.raises(AlignmentError):
        ransac_similarity(cloud(10), cloud(10), inlier_threshold=0.0)
    tiny = ransac_similarity(cloud(2), cloud(2), inlier_threshold=0.1)
    assert tiny.status == "degenerate" and tiny.transform is None


def test_a_fitted_scale_that_should_be_metric_but_is_not_is_reported():
    src = cloud()
    res = ransac_similarity(src, TRUE.apply(src), inlier_threshold=0.01)
    assert scale_mismatch(res, expected=1.0, tolerance=0.05) is not None
    assert scale_mismatch(res, expected=2.5, tolerance=0.05) is None


# ------------------------------------------------------------------------------------------------ frame graph

def graph():
    g = FrameGraph()
    g.add_frame(Frame("colmap", "model", "relative"))
    g.add_frame(Frame("depth_model", "model", "relative"))
    g.add_frame(Frame("world", "world", "metric"))
    g.add_frame(Frame("lonely", "model", "metric"))
    return g


def test_values_in_frames_with_no_known_transform_cannot_be_compared():
    g = graph()
    with pytest.raises(FrameMismatch, match="no known transform"):
        g.require_comparable("colmap", "lonely")
    with pytest.raises(FrameMismatch, match="unknown coordinate frame"):
        g.require_comparable("colmap", "nope")
    with pytest.raises(FrameMismatch):
        g.transform_points(np.zeros((1, 3)), "colmap", "lonely")


def test_metric_comparison_needs_a_scale_anchored_path_a_relative_frame_is_not_enough():
    g = graph()
    g.add_transform("colmap", "world", TRUE, source="registration to a model, scale not measured")
    assert g.comparable("colmap", "world")[0]                                        # shape/direction comparison is fine
    ok, why = g.comparable("colmap", "world", metric=True)
    assert not ok and "metric scale" in why
    g.add_transform("depth_model", "world", TRUE, source="laser-measured baseline", scale_anchored=True)
    assert g.comparable("depth_model", "world", metric=True)[0]


def test_transform_chains_compose_and_invert():
    g = graph()
    g.add_transform("colmap", "world", TRUE, source="a")
    second = Sim3.from_arrays(1.0, rot_z(10), np.array([0.0, 1.0, 0.0]))
    g.add_transform("world", "lonely", second, source="b", scale_anchored=True)
    p = cloud(8)
    via = g.transform_points(p, "colmap", "lonely")
    assert np.allclose(via, second.apply(TRUE.apply(p)), atol=1e-9)
    assert np.allclose(g.transform_points(via, "lonely", "colmap"), p, atol=1e-8)
    with pytest.raises(AlignmentError):
        g.add_transform("colmap", "world", TRUE, source="")


# ------------------------------------------------------------------------------------------------ camera poses

def test_an_inconsistent_camera_pose_is_flagged_and_too_few_cameras_flag_nothing():
    rng = np.random.default_rng(5)
    poses_a = {f"c{i}": (rot_z(i * 7.0), rng.uniform(-2, 2, 3)) for i in range(10)}
    ident = Sim3.from_arrays(1.0, np.eye(3), np.zeros(3))
    poses_b = {k: (r.copy(), c + rng.normal(0, 0.002, 3)) for k, (r, c) in poses_a.items()}
    poses_b["c4"] = (poses_b["c4"][0], poses_b["c4"][1] + np.array([0.9, 0.0, 0.0]))      # one badly registered camera
    res = pose_residuals(poses_a, poses_b, ident)
    assert flag_inconsistent_poses(res) == ["c4"]
    assert flag_inconsistent_poses({k: res[k] for k in list(res)[:3]}) == []


# ------------------------------------------------------------------------------------------------ consensus: frames, independence

def est(engine, value, group=None, frame="world", cov=None, oid=None):
    return Estimate(engine, oid or f"obs_{engine}", group or engine, frame, tuple(value),
                    None if cov is None else tuple(map(tuple, cov)))


def test_estimates_in_different_frames_are_never_compared():
    with pytest.raises(FrameMismatch):
        resolve_estimates([est("a", [1, 2, 3], frame="colmap"), est("b", [1, 2, 3], frame="world")], tolerance=1.0)
    with pytest.raises(ConsensusError):
        resolve_estimates([est("a", [1, 2, 3]), est("b", [1, 2])], tolerance=1.0)


def test_three_correlated_engines_from_one_family_are_one_source_not_three_votes():
    """3 engines of one family agree on 5.0; one independent engine says 9.0. A majority vote would call this 3:1 for 5.0.
    There are only TWO independent sources and they disagree, so the honest answer is a conflict."""
    ests = [est("dino_s", [5.0], "dino"), est("dino_b", [5.01], "dino"), est("dino_l", [4.99], "dino"), est("other", [9.0])]
    res = resolve_estimates(ests, tolerance=0.5)
    assert res.status == "conflict" and res.value is None
    assert len(res.groups) == 2 and any("collapsed into 2 independent" in n for n in res.notes)
    assert res.disagreement["max_pairwise_distance"] == pytest.approx(4.0, abs=0.05)


def test_agreement_between_two_sources_resolves_without_inventing_a_tighter_uncertainty():
    res = resolve_estimates([est("a", [5.0, 1.0]), est("b", [5.2, 1.0])], tolerance=0.5)
    assert res.resolved and np.allclose(res.value, [5.1, 1.0])
    assert res.uncertainty["method"] == "observed_spread" and np.allclose(res.uncertainty["half_range"], [0.1, 0.0])
    assert res.verification == "unverified" and "NOT established" in res.independence


def test_without_covariance_or_tolerance_agreement_cannot_be_judged():
    res = resolve_estimates([est("a", [1.0]), est("b", [1.0])])
    assert res.status == "insufficient" and res.value is None and "cannot be judged" in res.notes[0]


def test_a_single_source_is_reported_as_uncorroborated_with_no_invented_confidence():
    res = resolve_estimates([est("a", [1.0, 2.0])])
    assert res.resolved and res.uncertainty is None and any("nothing was corroborated" in n for n in res.notes)


def test_three_independent_sources_reject_one_outlier_and_say_which():
    res = resolve_estimates([est("a", [1.00, 2.00]), est("b", [1.02, 1.98]), est("c", [0.99, 2.01]), est("d", [8.0, -4.0])],
                            tolerance=0.2)
    assert res.resolved and [o.group for o in res.outliers] == ["d"]
    assert np.allclose(res.value, [1.0, 2.0], atol=0.03) and "obs_d" not in res.supporting
    assert res.method.startswith("consensus_set_")


def test_three_sources_that_all_disagree_are_a_conflict_not_an_average():
    res = resolve_estimates([est("a", [0.0]), est("b", [5.0]), est("c", [10.0])], tolerance=0.5)
    assert res.status == "conflict" and res.value is None


def test_two_equally_large_camps_are_a_conflict_the_tie_is_not_broken_arbitrarily():
    res = resolve_estimates([est("a", [0.0]), est("b", [0.1]), est("c", [5.0]), est("d", [5.1])], tolerance=0.5)
    assert res.status == "conflict" and res.value is None and "equally large" in res.notes[0]


# ------------------------------------------------------------------------------------------------ consensus: covariances

def test_covariances_decide_agreement_by_chi_square_not_a_fixed_distance():
    tight = [[0.01, 0], [0, 0.01]]
    loose = [[4.0, 0], [0, 4.0]]
    far_but_uncertain = resolve_estimates([est("a", [0, 0], cov=loose), est("b", [2.0, 0], cov=loose)])
    far_and_certain = resolve_estimates([est("a", [0, 0], cov=tight), est("b", [2.0, 0], cov=tight)])
    assert far_but_uncertain.resolved and far_and_certain.status == "conflict"


def test_inverse_variance_fusion_shrinks_uncertainty_only_when_independence_is_attested():
    c = [[1.0, 0], [0, 1.0]]
    pair = [est("a", [1.0, 1.0], cov=c), est("b", [1.2, 0.8], cov=c)]
    not_attested = resolve_estimates(pair)
    attested = resolve_estimates(pair, assume_independent_groups=True)
    assert np.allclose(not_attested.uncertainty["covariance"], c)                     # no shrinkage
    assert np.allclose(attested.uncertainty["covariance"], 0.5 * np.eye(2))           # halved: attested independent
    assert np.allclose(attested.value, [1.1, 0.9])


# ------------------------------------------------------------------------------------------------ consensus: original image evidence

K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
CAMS = [(K, np.eye(3), np.zeros(3)), (K, np.eye(3), np.array([-0.5, 0.0, 0.0])), (K, rot_z(4.0), np.array([-1.0, 0.1, 0.0]))]
X_TRUE = np.array([0.3, 0.2, 4.0])


def observed(X):
    out = []
    for Kc, R, t in CAMS:
        p = Kc @ (R @ X + t)
        out.append(p[:2] / p[2])
    return out


CHECK = make_reprojection_check(CAMS, observed(X_TRUE))


def test_reprojection_error_is_zero_for_the_true_point_and_infinite_behind_the_camera():
    assert reprojection_error(K, np.eye(3), np.zeros(3), X_TRUE, observed(X_TRUE)[0]) == pytest.approx(0.0, abs=1e-9)
    assert CHECK(X_TRUE) == pytest.approx(0.0, abs=1e-9) and CHECK(X_TRUE + [0.4, 0, 0.9]) > 10
    assert reprojection_error(K, np.eye(3), np.zeros(3), np.array([0, 0, -1.0]), np.zeros(2)) == float("inf")
    assert make_reprojection_check(CAMS[:1], observed(X_TRUE)[:1])(X_TRUE) is None       # one view cannot judge a 3D point
    with pytest.raises(ConsensusError):
        make_reprojection_check(CAMS, observed(X_TRUE)[:2])


def test_two_independent_but_biased_engines_agreeing_do_not_outvote_the_image_evidence():
    """A and B are different model families that share the same systematic bias (e.g. both over-estimate depth) and agree with each
    other. C is correct. Agreement is not proof: the reprojection check from the original images must overrule the majority."""
    biased = X_TRUE + np.array([0.45, 0.0, 0.9])
    ests = [est("A", biased + 0.01), est("B", biased - 0.01), est("C", X_TRUE)]
    statistical = resolve_estimates(ests, tolerance=0.3)
    assert np.allclose(statistical.value, biased, atol=0.05) and statistical.verification == "unverified"   # no check: majority wins
    res = resolve_estimates(ests, tolerance=0.3, geometric_residual=CHECK, geometric_margin=1.0)
    assert res.resolved and res.verification == "geometry_preferred_minority" and res.method == "geometric_residual"
    assert np.allclose(res.value, X_TRUE, atol=1e-9)
    assert any("overruled" in n for n in res.notes)


def test_when_the_image_evidence_supports_the_consensus_the_result_says_so():
    ests = [est("A", X_TRUE + 0.002), est("B", X_TRUE - 0.002), est("C", X_TRUE + 0.001)]
    res = resolve_estimates(ests, tolerance=0.3, geometric_residual=CHECK, geometric_margin=1.0)
    assert res.resolved and res.verification == "geometry_supported"


def test_two_disagreeing_sources_are_resolved_by_the_image_evidence_only_when_it_clearly_separates_them():
    ests = [est("A", X_TRUE), est("B", X_TRUE + [0.0, 0.0, 1.0])]
    res = resolve_estimates(ests, tolerance=0.3, geometric_residual=CHECK, geometric_margin=1.0)
    assert res.resolved and np.allclose(res.value, X_TRUE) and res.verification == "geometry_preferred_minority"
    ambiguous = resolve_estimates(ests, tolerance=0.3, geometric_residual=lambda v: 1.0, geometric_margin=1.0)
    assert ambiguous.status == "conflict"
    cannot_judge = resolve_estimates(ests, tolerance=0.3, geometric_residual=lambda v: None)
    assert cannot_judge.status == "conflict"


# ------------------------------------------------------------------------------------------------ categorical reconciliation

def claim(engine, label, group=None):
    return LabelClaim(engine, f"obs_{engine}", group or engine, label)


def test_unanimous_labels_are_agreed_but_explicitly_unverified():
    r = reconcile_labels([claim("a", "sofa"), claim("b", "sofa")])
    assert r.status == "agreed_unverified" and r.label == "sofa" and "not proof" in r.notes[0]


def test_a_two_to_one_label_split_is_a_conflict_not_a_majority_win_and_correlated_engines_count_once():
    split = reconcile_labels([claim("a", "sofa", "g1"), claim("b", "sofa", "g2"), claim("c", "bed", "g3")])
    assert split.status == "conflict" and split.label is None and split.candidates == ("bed", "sofa")
    same_family = reconcile_labels([claim("a", "sofa", "g1"), claim("b", "sofa", "g1"), claim("c", "bed", "g3")])
    assert same_family.sources["sofa"] == ("g1",)             # two engines of one family are one independent source


def test_image_evidence_can_separate_labels_but_only_by_a_margin_and_only_if_every_label_is_scored():
    claims = [claim("a", "sofa"), claim("b", "bed")]
    scores = {"sofa": 0.2, "bed": 0.9}
    assert reconcile_labels(claims, evidence_score=scores.get, margin=0.3).label == "bed"
    assert reconcile_labels(claims, evidence_score=scores.get, margin=0.8).status == "conflict"
    assert reconcile_labels(claims, evidence_score={"sofa": 0.2}.get, margin=0.0).status == "conflict"
