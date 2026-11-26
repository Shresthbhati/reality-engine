"""Tests for pairwise-rotation comparison and pose-graph optimisation.

The Day-4 report claimed a 21 deg median cycle disagreement and explained it as
planar degeneracy. Two defects produced that number: the cycle residual
multiplied its three rotations in an order that is not the closure (so it
reported tens of degrees on perfectly consistent poses), and the pair
intrinsics came from the reference model's original 3072x2304 frames while the
images were the 1024x768 derivatives (principal point outside the image), which
made every essential matrix -- and every recovered rotation -- wrong. These
tests pin the corrected closure and the pose-graph/measurement machinery.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from perception.camera_pose_graph import (
    ReferencePose,
    compare_pair_rotations,
    cycle_residuals,
    read_colmap_text_model,
    relative_rotation,
    rotation_angle_deg,
    rotation_exp,
    rotation_log,
    solve_pose_graph,
)


def _rand_R(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _scene(n: int, seed: int):
    """n cameras with random world->camera rotations and centres spread in a unit box."""
    rng = np.random.default_rng(seed)
    R = {f"cam{i}": _rand_R(seed + i) for i in range(n)}
    c = {f"cam{i}": rng.uniform(-1, 1, 3) for i in range(n)}
    return R, c


def _edges(R, c, **extra):
    out = []
    names = list(R)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            Rab = relative_rotation(R[a], R[b])
            # x_b ~ Rab x_a + t  <=>  c_b - c_a = -R_b^T t
            t = -R[b] @ (c[b] - c[a])
            n = float(np.linalg.norm(t))
            if n < 1e-9:
                continue
            out.append({"a": a, "b": b, "R": Rab, "t_unit": t / n, "inliers": 100, **extra})
    return out


def _reference(R, c, prefix=""):
    return {f"{prefix}{k}": ReferencePose(image=f"{prefix}{k}", R=R[k] @ np.eye(3), t=-R[k] @ c[k], center=c[k])
            for k in R}


# --------------------------------------------------------------------------- closure metric


def test_cycle_residual_is_zero_for_consistent_poses():
    """The regression for the product-order defect that produced the 21 deg claim."""
    R1, R2, R3 = _rand_R(1), _rand_R(2), _rand_R(3)
    edges = [("a", "b", relative_rotation(R1, R2)),
             ("b", "c", relative_rotation(R2, R3)),
             ("a", "c", relative_rotation(R1, R3))]
    rows = cycle_residuals(["a", "b", "c"], edges)
    assert len(rows) == 1
    assert rows[0]["residual_deg"] < 1e-6


def test_cycle_residual_flags_an_inconsistent_triangle():
    R1, R2, R3 = _rand_R(4), _rand_R(5), _rand_R(6)
    bad = relative_rotation(R1, R3) @ rotation_exp(np.array([0.0, 0.0, math.radians(12.0)]))
    edges = [("a", "b", relative_rotation(R1, R2)),
             ("b", "c", relative_rotation(R2, R3)),
             ("a", "c", bad)]
    row = cycle_residuals(["a", "b", "c"], edges)[0]
    assert 11.0 < row["residual_deg"] < 13.0


def test_cycle_residual_reads_edges_in_either_direction():
    R1, R2, R3 = _rand_R(7), _rand_R(8), _rand_R(9)
    forward = [("a", "b", relative_rotation(R1, R2)),
               ("b", "c", relative_rotation(R2, R3)),
               ("a", "c", relative_rotation(R1, R3))]
    backward = [("b", "a", relative_rotation(R1, R2).T),
                ("c", "b", relative_rotation(R2, R3).T),
                ("c", "a", relative_rotation(R1, R3).T)]
    assert cycle_residuals(["a", "b", "c"], forward)[0]["residual_deg"] < 1e-6
    assert cycle_residuals(["a", "b", "c"], backward)[0]["residual_deg"] < 1e-6


# --------------------------------------------------------------------------- reference model


def test_reference_model_poses_are_parsed_with_colmab_conventions(tmp_path):
    angle = math.radians(20.0)
    qw, qz = math.cos(angle / 2.0), math.sin(angle / 2.0)
    # world->camera: a 20 deg rotation about +z (quaternion w,x,y,z = cos,0,0,sin)
    (tmp_path / "images.txt").write_text(
        f"1 {qw:.12f} 0 0 {qz:.12f} 1 2 3 1 frame_a.JPG\n"
        "10.5 20.5 -1\n"
        f"2 {qw:.12f} 0 0 {qz:.12f} -4 0.5 2 1 frame_b.JPG\n"
        "11.5 21.5 55\n",
        encoding="utf-8")
    poses = read_colmap_text_model(tmp_path)
    assert sorted(poses) == ["frame_a.JPG", "frame_b.JPG"]
    expected = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    assert np.allclose(poses["frame_a.JPG"].R, expected, atol=1e-9)
    assert np.allclose(poses["frame_a.JPG"].t, [1.0, 2.0, 3.0])
    # camera centre = -R^T t, the convention the pair comparison relies on
    assert np.allclose(poses["frame_a.JPG"].center, -expected.T @ np.array([1.0, 2.0, 3.0]))
    assert np.allclose(relative_rotation(poses["frame_a.JPG"].R, poses["frame_b.JPG"].R),
                       poses["frame_b.JPG"].R @ poses["frame_a.JPG"].R.T)


def test_a_binary_reference_model_is_reported_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_colmap_text_model(tmp_path)


def test_pair_rotations_are_compared_in_a_gauge_free_way():
    R, c = _scene(4, seed=11)
    reference = _reference(R, c)
    pairs = [{"engine": "synthetic", "a": e["a"], "b": e["b"], "inliers": e["inliers"],
              "relative_pose": {"R": e["R"], "t_unit": e["t_unit"]}} for e in _edges(R, c)]
    result = compare_pair_rotations(pairs, reference)
    assert result["status"] == "measured"
    assert result["pairs_compared"] == len(pairs)
    assert result["median_rotation_error_deg"] < 1e-6
    assert result["max_rotation_error_deg"] < 1e-6
    assert result["median_translation_direction_error_deg"] < 1e-6
    assert result["pairs_beyond_far_deg"] == 0


def test_pair_comparison_flags_a_wrong_rotation_and_reports_skips():
    R, c = _scene(3, seed=21)
    reference = _reference(R, c)
    names = sorted(R)
    def _row(a, b, R_rel):
        return {"a": a, "b": b, "engine": "synthetic",
                "relative_pose": {"R": R_rel, "t_unit": np.array([1.0, 0.0, 0.0])}}

    good = _row(names[0], names[1], relative_rotation(R[names[0]], R[names[1]]))
    bad = _row(names[0], names[2],
               relative_rotation(R[names[0]], R[names[2]]) @ rotation_exp(np.array([0.0, math.radians(25.0), 0.0])))
    unknown = _row(names[0], "not_in_model.JPG", np.eye(3))
    result = compare_pair_rotations([good, bad, unknown], reference)
    assert result["pairs_compared"] == 2
    assert result["pairs_skipped"] == 1
    assert "not_in_model.JPG" in result["skipped"][0]["reason"]
    errs = {f"{r['a']}|{r['b']}": r["rotation_error_deg"] for r in result["pairs"]}
    assert errs[f"{names[0]}|{names[2]}"] == pytest.approx(25.0, abs=1e-6)
    assert errs[f"{names[0]}|{names[1]}"] < 1e-6


# --------------------------------------------------------------------------- pose graph


def test_pose_graph_recovers_known_cameras_up_to_the_gauge():
    R, c = _scene(8, seed=31)
    result = solve_pose_graph(sorted(R), _edges(R, c), reference=_reference(R, c))
    assert result["status"] == "optimised"
    assert result["edges_accepted"] == result["edges_given"]
    assert result["n_components"] == 1
    cmp_ = result["reference_comparison"]
    assert cmp_["status"] == "measured"
    assert cmp_["cameras_compared"] == 8
    assert cmp_["rotation_error_deg"]["max"] < 1e-4
    assert cmp_["centre_error_model_units"]["max"] < 1e-4
    assert cmp_["rotation_error_deg"]["n_within_2_deg"] == 8


def test_an_outlier_edge_is_rejected_and_kept_with_its_reason():
    R, c = _scene(8, seed=41)
    edges = _edges(R, c)
    names = sorted(R)
    outlier = {"a": edges[0]["a"], "b": edges[0]["b"],
               "R": np.asarray(edges[0]["R"]) @ rotation_exp(np.array([0.0, math.radians(40.0), 0.0])),
               "t_unit": edges[0]["t_unit"], "inliers": 3}
    result = solve_pose_graph(names, edges + [outlier], reference=_reference(R, c))
    assert result["edges_rejected"] >= 1
    rejected = result["rejected_edges"][0]
    assert rejected["residual_deg"] > 30.0
    assert "disagrees with the averaged rotations" in rejected["reason"]
    # the accepted solution is still the correct one
    assert result["reference_comparison"]["rotation_error_deg"]["median"] < 0.5


def test_disconnected_groups_keep_their_own_gauge():
    R1, c1 = _scene(4, seed=51)
    R2, c2 = _scene(4, seed=61)
    R = {f"g1_{k}": R1[k] for k in R1} | {f"g2_{k}": R2[k] for k in R2}
    c = {f"g1_{k}": c1[k] for k in c1} | {f"g2_{k}": c2[k] for k in c2}
    edges = [e for e in _edges(R, c) if e["a"].split("_")[0] == e["b"].split("_")[0]]
    result = solve_pose_graph(sorted(R), edges, reference=_reference(R, c))
    assert result["n_components"] == 2
    assert result["main_component_cameras"] == 4
    cmp_ = result["reference_comparison"]
    # each group is scored in its own gauge, so both groups must come out accurate
    assert cmp_["cameras_compared"] == 8
    assert len(cmp_["components"]) == 2
    assert all(g["status"] == "measured" for g in cmp_["components"])
    assert "cameras_outside_main_component" in cmp_
    assert cmp_["rotation_error_deg"]["max"] < 1e-4


def test_the_spanning_tree_takes_the_strongest_edges_not_the_first_ones_found():
    """Kruskal by inliers, not breadth-first order (the 0308 mis-orientation).

    On a 0-1-2-3 chain with a weak wrong 0->2 edge, a breadth-first walk leaves 0,
    discovers 2 through the wrong edge, and orients everything downstream from it;
    Kruskal reaches 2 through the strong 1->2 edge before the weak edge is considered.
    """
    R, c = _scene(4, seed=81)
    names = sorted(R)
    chain = []
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        t = -R[b] @ (c[b] - c[a])
        chain.append({"a": a, "b": b, "R": relative_rotation(R[a], R[b]),
                      "t_unit": t / np.linalg.norm(t), "inliers": 50})
    wrong = {"a": names[0], "b": names[2],
             "R": relative_rotation(R[names[0]], R[names[2]]) @ rotation_exp(
                 np.array([0.0, 0.0, math.radians(35.0)])),
             "t_unit": chain[0]["t_unit"], "inliers": 5}
    result = solve_pose_graph(names, chain + [wrong], reference=_reference(R, c))
    assert [(Path(r["a"]).name, Path(r["b"]).name) for r in result["rejected_edges"]] == [
        (names[0], names[2])]
    # rounding perturbs each edge by ~1e-6, which the averaging carries as a small bias
    assert result["reference_comparison"]["rotation_error_deg"]["max"] < 0.1


def test_no_edges_is_reported_as_unavailable_not_as_success():
    result = solve_pose_graph(["a", "b"], [])
    assert result["status"] == "unavailable"
    assert "no edges" in result["reason"]


def test_a_single_link_between_two_groups_is_reported_as_a_bridge():
    """No cycle means no verification: the hinge must be named, not trusted."""
    R, c = _scene(6, seed=91)
    names = sorted(R)
    inside = [e for e in _edges(R, c)
              if (e["a"] in names[:3] and e["b"] in names[:3])
              or (e["a"] in names[3:] and e["b"] in names[3:])]
    await_bridge = {"a": names[0], "b": names[3],
                    "R": relative_rotation(R[names[0]], R[names[3]]),
                    "t_unit": np.array([1.0, 0.0, 0.0]), "inliers": 100}
    result = solve_pose_graph(names, inside + [await_bridge])
    assert result["n_components"] == 1
    assert result["n_bridges"] == 1
    assert result["bridges"][0]["a"] == names[0]
    assert result["bridges"][0]["b"] == names[3]
    # a fully redundant graph has no bridge to distrust
    assert solve_pose_graph(names, _edges(R, c))["n_bridges"] == 0


def test_rotation_log_and_exp_round_trip():
    for seed in range(5):
        R = _rand_R(seed + 70)
        assert np.allclose(rotation_exp(rotation_log(R)), R, atol=1e-9)
        assert np.allclose(rotation_exp(rotation_log(R.T)), R.T, atol=1e-9)
    # near pi, where the axis must be recovered from R + I
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]):
        R = rotation_exp(np.array(axis) / np.linalg.norm(axis) * math.pi)
        assert np.allclose(rotation_exp(rotation_log(R)), R, atol=1e-6)


# --------------------------------------------------------------------------- conventions
# Day 5 Phase A.1: the coordinate conventions the whole pipeline depends on, asserted
# directly instead of only through the pose-graph result.


def test_relative_rotation_composes_world_to_camera_poses():
    """x_b ~ R_ab x_a with R_ab = R_b R_a^T, i.e. R_b == R_ab @ R_a."""
    Ra, Rb = _rand_R(101), _rand_R(102)
    Rab = relative_rotation(Ra, Rb)
    assert np.allclose(Rab @ Ra, Rb)
    assert np.allclose(np.linalg.det(Rab), 1.0)
    assert np.allclose(Rab @ Rab.T, np.eye(3))


def test_camera_centre_and_translation_follow_the_world_to_camera_convention():
    """A camera at centre c has t = -R c, so R c + t == 0 and the pair translation is
    -R_b (c_b - c_a) in camera a's frame."""
    Ra, Rb = _rand_R(103), _rand_R(104)
    ca, cb = np.array([0.3, -1.2, 2.0]), np.array([-0.7, 0.4, 1.1])
    ta, tb = -Ra @ ca, -Rb @ cb
    assert np.allclose(Ra @ ca + ta, 0.0)
    d_in_a = Ra @ (cb - ca)                     # direction of travel expressed in camera a
    # the pair translation is t = t_b - R_b R_a^T t_a (NOT t_b - t_a: two cameras'
    # translations live in different frames, which is exactly the mistake to pin here)
    t_pair = tb - Rb @ Ra.T @ ta
    assert np.allclose(t_pair, -Rb @ (cb - ca))
    # expressed in camera a it is the bearing of b: R_a (c_b - c_a), i.e. -R_b^T t in a
    assert np.allclose(-Rb.T @ t_pair, cb - ca)
    assert np.allclose(Ra @ (-Rb.T @ t_pair), d_in_a)
    assert np.allclose(-t_pair, Rb @ (cb - ca))   # and in b: the bearing of a is negated


def test_relative_pose_inversion_transposes_the_rotation_and_flips_the_baseline():
    Ra, Rb = _rand_R(105), _rand_R(106)
    ca, cb = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.5])
    Rab = relative_rotation(Ra, Rb)
    t_ab = -Rb @ (cb - ca)
    # x_a = Rab^T x_b - Rab^T t_ab, so the inverse edge (b -> a) is (Rab^T, -Rab^T t_ab)
    assert np.allclose(Rab.T, relative_rotation(Rb, Ra))
    assert np.allclose(Rab @ Rab.T, np.eye(3))
    # -Rab^T t_ab is the direction of travel in camera a, i.e. the inverse edge's bearing
    assert np.allclose(-Rab.T @ t_ab, Ra @ (cb - ca))


def test_longer_loops_close_only_with_the_closure_product_order():
    """A 5-cycle: R_ab R_bc R_cd R_de R_ea == I ... in the correct order."""
    Rs = [_rand_R(110 + i) for i in range(5)]
    names = [f"c{i}" for i in range(5)]
    rel = {(names[i], names[(i + 1) % 5]): relative_rotation(Rs[i], Rs[(i + 1) % 5])
           for i in range(5)}
    closure = np.eye(3)
    for i in range(5):
        closure = rel[(names[i], names[(i + 1) % 5])] @ closure
    assert rotation_angle_deg(closure) < 1e-6
    # the wrong order (the defect that produced the 21 deg claim) is far from identity
    wrong = rel[(names[0], names[1])]
    for i in range(1, 5):
        wrong = wrong @ rel[(names[i], names[(i + 1) % 5])]
    assert rotation_angle_deg(wrong) > 20.0


@pytest.mark.parametrize("seed", [201, 202, 203, 204, 205])
def test_randomised_graphs_with_known_ground_truth(seed):
    """12 cameras, a random spanning tree plus redundant edges, noise-free: the graph must
    come back with the exact rotations and centres in its own gauge."""
    rng = np.random.default_rng(seed)
    R, c = _scene(12, seed=seed)
    names = sorted(R)
    every = _edges(R, c)
    # keep a random spanning tree, then add ~half of the remaining edges
    tree, seen = [], {names[0]}
    for e in rng.permutation(len(every)):
        e = every[e]
        if (e["a"] in seen) ^ (e["b"] in seen):
            tree.append(e)
            seen |= {e["a"], e["b"]}
    rest = [e for e in every if e not in tree]
    keep = [rest[i] for i in rng.permutation(len(rest))[: len(rest) // 2]]
    result = solve_pose_graph(names, tree + keep, reference=_reference(R, c))
    assert result["status"] == "optimised"
    assert result["n_components"] == 1
    assert result["n_bridges"] == 0
    cmp_ = result["reference_comparison"]
    assert cmp_["rotation_error_deg"]["max"] < 1e-4
    assert cmp_["rotation_error_deg"]["mean"] < 1e-4
    assert cmp_["rotation_error_deg"]["p95"] < 1e-4
    assert cmp_["centre_error_model_units"]["max"] < 1e-3


# --------------------------------------------------------------------------- degeneracies


def test_a_reflection_edge_is_rejected_not_averaged():
    """det = -1 is not a camera rotation; orthonormalising it silently would invent one."""
    R, c = _scene(6, seed=211)
    edges = _edges(R, c)
    reflected = dict(edges[0])
    reflected["R"] = np.diag([1.0, 1.0, -1.0]) @ np.asarray(edges[0]["R"])
    assert abs(float(np.linalg.det(reflected["R"])) + 1.0) < 1e-9
    result = solve_pose_graph(sorted(R), edges + [reflected], reference=_reference(R, c))
    assert result["n_invalid_edges"] == 1
    assert "determinant" in result["invalid_edges"][0]["reason"]
    assert result["edges_accepted"] == len(edges)
    assert result["reference_comparison"]["rotation_error_deg"]["max"] < 1e-4


def test_a_rotation_rounded_to_six_decimals_is_still_a_rotation():
    """Measured rotations are serialised rounded; det then reads 0.999999. A 1e-6 test
    wrongly discarded five real edges on the south_building capture."""
    R, c = _scene(6, seed=215)
    rounded = [{**e, "R": np.round(np.asarray(e["R"]), 6)} for e in _edges(R, c)]
    result = solve_pose_graph(sorted(R), rounded, reference=_reference(R, c))
    assert result["n_invalid_edges"] == 0
    assert result["edges_rejected"] == 0
    # rounding each edge by 1e-6 leaves ~0.04 deg of bias in the fitted solution:
    # negligible against the 0.8 deg pair error measured on real data, but not zero
    assert result["reference_comparison"]["rotation_error_deg"]["max"] < 0.1


def test_a_consistent_frame_mistake_is_invisible_to_cycles_and_caught_by_the_reference():
    """Every edge given in the inverse convention is self-consistent, so no cycle and no
    residual flags it; only the comparison with the reference model does. This is why the
    two checks are not interchangeable."""
    R, c = _scene(6, seed=221)
    edges = _edges(R, c)
    inverted = [{**e, "R": np.asarray(e["R"]).T} for e in edges]
    result = solve_pose_graph(sorted(R), inverted, reference=_reference(R, c))
    # each edge is a valid rotation, so nothing is 'invalid'; the graph is simply wrong,
    # and only the reference model sees it
    assert result["n_invalid_edges"] == 0
    assert result["reference_comparison"]["rotation_error_deg"]["median"] > 20.0
    assert result["reference_comparison"]["rotation_error_deg"]["max"] > 60.0


def test_a_planar_camera_layout_withholds_the_centre_accuracy():
    """Centres on one plane leave the missing direction unconstrained: the translation
    solution is ambiguous, so no centre error may be reported (rotations still are)."""
    R, c = _scene(8, seed=231)
    flat = {k: np.array([v[0], v[1], 0.0]) for k, v in c.items()}
    result = solve_pose_graph(sorted(R), _edges(R, flat), reference=_reference(R, flat))
    assert result["translation"]["layout_rank"] < 3
    assert result["translation"]["status"] == "ambiguous"
    assert result["reference_comparison"]["centre_error_model_units"]["status"] == "unavailable"
    assert result["reference_comparison"]["rotation_error_deg"]["max"] < 1e-4


def test_a_flipped_bearing_convention_produces_a_mirrored_layout():
    """Negating every t_unit turns (c_b - c_a) = s d into the mirrored layout, which is a
    perfectly consistent solution of the direction constraints: the solver cannot reject
    it, and the centre error against the reference exposes it. This is the documented
    limit of a directions-only translation solve -- cycles cannot see handedness."""
    R, c = _scene(6, seed=241)
    edges = [{**e, "t_unit": -np.asarray(e["t_unit"])} for e in _edges(R, c)]
    result = solve_pose_graph(sorted(R), edges, reference=_reference(R, c))
    # the rotations are unaffected: the bearing sign says nothing about rotation
    assert result["reference_comparison"]["rotation_error_deg"]["max"] < 1e-4
    # the layout is mirrored, so it is a valid solve but a wrong room
    assert result["translation"]["direction_residual"]["max"] < 1e-6
    assert result["reference_comparison"]["centre_error_model_units"]["median"] > 0.2


def test_a_camera_hinged_on_one_bridge_is_named_unobservable():
    R, c = _scene(6, seed=251)
    names = sorted(R)
    # group 1 is fully connected (cycles); group 2 is a chain hanging off one bridge, so
    # none of its cameras has a cycle that could verify its placement
    def edge(a, b):
        t = -R[b] @ (c[b] - c[a])
        return {"a": a, "b": b, "R": relative_rotation(R[a], R[b]),
                "t_unit": t / np.linalg.norm(t), "inliers": 100}
    edges = [edge(names[0], names[1]), edge(names[1], names[2]), edge(names[0], names[2]),
             edge(names[0], names[3]), edge(names[3], names[4]), edge(names[4], names[5])]
    result = solve_pose_graph(names, edges)
    named = {u["camera"]: u["reason"] for u in result["observability"]["unobservable"]}
    assert set(named) == {names[3], names[4], names[5]}
    assert "bridge" in named[names[3]]
    assert result["observability"]["observable_cameras"] == 3
    assert result["rotations"]  # the fit still runs; the limitation is reported, not hidden


def test_translation_baselines_are_positive_on_a_noise_free_scene():
    R, c = _scene(6, seed=261)
    result = solve_pose_graph(sorted(R), _edges(R, c), reference=_reference(R, c))
    tr = result["translation"]
    assert tr["status"] == "estimated"
    assert tr["baselines_non_negative"] is True
    assert tr["baselines_at_zero"] == 0
    assert all(v > 0.0 for v in tr["per_edge_scale"].values())
    assert tr["direction_residual"]["max"] < 1e-6
    assert tr["scale"].startswith("gauge:")
    assert result["reference_comparison"]["centre_error_model_units"]["max"] < 1e-3
