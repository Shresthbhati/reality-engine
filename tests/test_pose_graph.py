"""Pose-graph optimisation on synthetic trajectories with KNOWN ground truth (deterministic, seeded)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from registration.pose_graph import Edge, GnssPrior, Pose, PoseGraphError, optimize_pose_graph

N = 24


def ground_truth():
    """Closed circular path, radius 10 m, heading tangent to the circle."""
    gt = {}
    for k in range(N):
        a = 2 * np.pi * k / N
        gt[f"p{k:02d}"] = Pose((10 * np.cos(a), 10 * np.sin(a), 0.1 * k),
                               tuple(Rotation.from_euler("z", a + np.pi / 2).as_quat()))
    return gt


def rel(a: Pose, b: Pose, rng=None, st=0.0, sr=0.0):
    ra, rb = a.rot(), b.rot()
    t = ra.inv().apply(np.array(b.t) - np.array(a.t))
    r = ra.inv() * rb
    if rng is not None:
        t = t + rng.normal(0, st, 3)
        r = r * Rotation.from_rotvec(rng.normal(0, sr, 3))
    return tuple(t), tuple(r.as_quat())


def drifted_chain(gt, rng, st=0.03, sr=0.01):
    """Odometry edges + the dead-reckoned initial guess they imply (accumulates drift)."""
    names = sorted(gt)
    edges, est = [], {names[0]: gt[names[0]]}
    for a, b in zip(names, names[1:]):
        t, q = rel(gt[a], gt[b], rng, st, sr)
        edges.append(Edge(f"o_{a}_{b}", a, b, t, q, sigma_t=st, sigma_r=sr))
        pa = est[a]
        ra = pa.rot()
        est[b] = Pose(tuple(np.array(pa.t) + ra.apply(t)), tuple((ra * Rotation.from_quat(q)).as_quat()))
    return names, edges, est


def ate(est, gt):
    """RMS translation error after anchoring on the first pose (graph gauge) -- metres."""
    return float(np.sqrt(np.mean([np.sum((np.array(est[k].t) - np.array(gt[k].t)) ** 2) for k in gt])))


def test_loop_closure_removes_drift():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(0))
    t, q = rel(gt[names[-1]], gt[names[0]])
    edges.append(Edge("loop", names[-1], names[0], t, q, sigma_t=0.03, sigma_r=0.01))
    res = optimize_pose_graph(est, edges)
    assert res.converged and not res.rejected_edges
    assert ate(res.poses, gt) < 0.5 * ate(est, gt)           # drift measurably reduced
    assert res.final_cost < res.initial_cost
    assert res.initial == est and res.initial is not res.poses       # raw input preserved


def test_gross_outlier_loop_is_rejected_and_reported():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(1))
    t, q = rel(gt[names[-1]], gt[names[0]])
    edges.append(Edge("loop_good", names[-1], names[0], t, q, sigma_t=0.03, sigma_r=0.01))
    mid = names[N // 2]
    wt, wq = rel(gt[names[3]], gt[mid])
    edges.append(Edge("loop_bad", names[3], mid, (wt[0] + 4.0, wt[1] - 3.0, wt[2]), wq, sigma_t=0.03, sigma_r=0.01))
    res = optimize_pose_graph(est, edges)
    assert [r["id"] for r in res.rejected_edges] == ["loop_bad"]
    assert ate(res.poses, gt) < 0.5 * ate(est, gt)


def test_gnss_priors_pin_world_frame():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(2))
    # initial guess is expressed in a rotated/shifted frame; GNSS anchors define the true world frame
    off = Rotation.from_euler("z", 0.7)
    est = {k: Pose(tuple(off.apply(v.t) + [5, -2, 1]), tuple((off * v.rot()).as_quat())) for k, v in est.items()}
    rng = np.random.default_rng(3)
    priors = [GnssPrior(n, tuple(np.array(gt[n].t) + rng.normal(0, 0.2, 3)), sigma=0.2) for n in names[::4]]
    res = optimize_pose_graph(est, edges, priors)
    assert ate(res.poses, gt) < 0.5            # lands in the GNSS frame, not the rotated initial frame
    assert ate(est, gt) > 5.0


def test_disconnected_graph_raises_with_components():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(4))
    edges = [e for e in edges if e.id != f"o_{names[5]}_{names[6]}"]
    with pytest.raises(PoseGraphError, match="disconnected: 2 components"):
        optimize_pose_graph(est, edges)


def test_underconstrained_gnss_and_bad_input_raise():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(5))
    two = [GnssPrior(names[0], gt[names[0]].t), GnssPrior(names[9], gt[names[9]].t)]
    with pytest.raises(PoseGraphError, match="orientation"):
        optimize_pose_graph(est, edges, two)
    with pytest.raises(PoseGraphError, match="sigma"):
        optimize_pose_graph(est, [Edge("x", names[0], names[1], (0, 0, 0), (0, 0, 0, 1), sigma_t=0.0)])
    with pytest.raises(PoseGraphError, match="unknown or identical"):
        optimize_pose_graph(est, [Edge("x", names[0], "nope", (0, 0, 0), (0, 0, 0, 1))])


def test_deterministic():
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(6))
    a = optimize_pose_graph(est, edges).to_dict()
    b = optimize_pose_graph(est, edges).to_dict()
    assert a == b


def test_cli_pose_graph_roundtrip_and_errors(tmp_path):
    import json
    from apps.cli.main import main
    gt = ground_truth()
    names, edges, est = drifted_chain(gt, np.random.default_rng(0))
    t, q = rel(gt[names[-1]], gt[names[0]])
    doc = {"poses": {k: {"t": list(v.t), "q": list(v.q)} for k, v in est.items()},
           "edges": [{"id": e.id, "i": e.i, "j": e.j, "t": list(e.t), "q": list(e.q), "sigma_t": 0.03, "sigma_r": 0.01}
                     for e in edges] + [{"id": "loop", "i": names[-1], "j": names[0], "t": list(t), "q": list(q),
                                         "sigma_t": 0.03, "sigma_r": 0.01}]}
    src, out = tmp_path / "g.json", tmp_path / "o.json"
    src.write_text(json.dumps(doc))
    assert main(["pose-graph", str(src), "-o", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["converged"] and set(res["optimized"]) == set(est) and res["initial"]["p05"]["t"] == list(est["p05"].t)
    doc["edges"] = doc["edges"][:5]                                          # disconnects the graph
    src.write_text(json.dumps(doc))
    assert main(["pose-graph", str(src), "-o", str(out)]) == 1


def test_single_loop_with_one_bad_edge_is_flagged_ambiguous_not_pruned():
    ident = (0.0, 0.0, 0.0, 1.0)
    poses = {k: Pose((float(i), 0.0, 0.0), ident) for i, k in enumerate("abc")}
    edges = [Edge("ab", "a", "b", (1, 0, 0), ident, 0.01, 0.01), Edge("bc", "b", "c", (1, 0, 0), ident, 0.01, 0.01),
             Edge("ac", "a", "c", (5, 0, 0), ident, 0.01, 0.01)]          # should be 2
    res = optimize_pose_graph(poses, edges)
    assert res.rejected_edges == [] and sorted(res.ambiguous) == ["ab", "ac", "bc"]
    assert any("not identifiable" in w for w in res.warnings)
