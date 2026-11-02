"""Cross-session registration refined by the pose-graph optimiser, against KNOWN ground-truth session frames."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from engine.math import Quat, Vec3
from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from reconstruction.calibration.transforms import RigidTransform
from registration.cross_session import (
    SessionAlignment,
    align_reconstructed_sessions,
    align_session_chain,
    record_cross_session_registration,
    refine_session_chain,
)
from world_ir import WorldIR

# ground truth: session frame -> reference frame "A" (p_A = R p_s + t)
GT = {"B": (Rotation.from_euler("z", 0.4), np.array([3.0, -1.0, 0.5])),
      "C": (Rotation.from_euler("xyz", [0.1, -0.2, 0.9]), np.array([-2.0, 4.0, 1.0]))}
GT["A"] = (Rotation.identity(), np.zeros(3))


def rt(frm, to, R, t):
    x, y, z, w = R.as_quat()
    return RigidTransform(from_frame=frm, to_frame=to, rotation=Quat(w=w, x=x, y=y, z=z).normalized(),
                          translation=Vec3(*map(float, t)))


def true_pair(frm, to):
    """Transform mapping `frm` coordinates into `to` coordinates."""
    Rf, tf = GT[frm]
    Rt, tt = GT[to]
    return Rt.inv() * Rf, Rt.inv().apply(tf - tt)


def anchor(frm, to, rmse, err_t=(0.0, 0.0, 0.0), err_r=0.0):
    R, t = true_pair(frm, to)
    R = R * Rotation.from_rotvec([0, 0, err_r])
    return SessionAlignment(frm, to, "accepted", "icp", rt(frm, to, R, np.array(t) + err_t), rmse, 0.9, "ok", 100, 100)


def err(report, s):
    """Translation error (m) of session s's transform into A against ground truth."""
    t = report.transforms[s].translation
    return float(np.linalg.norm(np.array([t.x, t.y, t.z]) - GT[s][1]))


def anchors(*items):
    return {(a.from_session, a.to_session): a for a in items}


def test_loop_refinement_beats_bfs_when_the_direct_alignment_is_poor():
    # C->A direct is poor (0.4 m off, large rmse); B->A and C->B are accurate and jointly imply a better C.
    a = anchors(anchor("B", "A", 0.01, (0.01, 0, 0)), anchor("C", "B", 0.01, (0, 0.01, 0)),
                anchor("C", "A", 0.4, (0.4, 0, 0)))
    bfs = align_session_chain(a, ["A", "B", "C"])
    ref = refine_session_chain(a, ["A", "B", "C"])
    assert ref.refinement["status"] == "refined" and ref.refinement["loop_closures"] == 1
    assert err(bfs, "C") > 0.3
    assert err(ref, "C") < 0.5 * err(bfs, "C")
    assert "refined by pose graph" in ref.reasons["C"]
    assert ref.refinement["initial_transforms"]["C"] == bfs.transforms["C"].to_dict()      # evidence preserved
    assert ref.transforms["A"] is None and ref.status_by_session["A"] == "reference"


def test_single_loop_inconsistency_is_reported_ambiguous_not_arbitrarily_pruned():
    # one wrong edge in a 3-session loop: any of the three edges could be the culprit
    a = anchors(anchor("B", "A", 0.01), anchor("C", "B", 0.01), anchor("C", "A", 0.01, (3.0, -2.0, 0.0)))
    ref = refine_session_chain(a, ["A", "B", "C"])
    assert ref.refinement["status"] == "ambiguous"
    assert set(ref.refinement["ambiguous_edges"]) >= {"C->A"} and len(ref.refinement["ambiguous_edges"]) >= 2
    bfs = align_session_chain(a, ["A", "B", "C"])
    assert ref.transforms["C"].to_dict() == bfs.transforms["C"].to_dict()      # untouched BFS result, not a guess


def test_wrong_alignment_is_identified_when_other_loops_corroborate_the_rest():
    GT["D"] = (Rotation.from_euler("y", -0.5), np.array([1.0, 2.0, -3.0]))
    try:
        good = [anchor("B", "A", 0.01), anchor("C", "B", 0.01), anchor("D", "C", 0.01), anchor("D", "A", 0.01),
                anchor("D", "B", 0.01)]
        a = anchors(*good, anchor("C", "A", 0.01, (3.0, -2.0, 0.0)))
        ref = refine_session_chain(a, ["A", "B", "C", "D"])
        assert ref.refinement["status"] == "refined"
        assert [r["id"] for r in ref.refinement["rejected_edges"]] == ["C->A"]
        assert max(err(ref, s) for s in ("B", "C", "D")) < 0.05
    finally:
        GT.pop("D")


def test_no_loop_is_skipped_with_reason_and_equals_bfs():
    a = anchors(anchor("B", "A", 0.01), anchor("C", "B", 0.01))
    ref = refine_session_chain(a, ["A", "B", "C"])
    bfs = align_session_chain(a, ["A", "B", "C"])
    assert ref.refinement["status"] == "skipped" and "no loop" in ref.refinement["reason"]
    assert {k: v.to_dict() for k, v in ref.transforms.items() if v} == {k: v.to_dict() for k, v in bfs.transforms.items() if v}


def test_unresolved_session_stays_unresolved_and_serialises():
    a = anchors(anchor("B", "A", 0.01))
    ref = refine_session_chain(a, ["A", "B", "C"])
    assert ref.status_by_session["C"] == "unresolved" and ref.transforms["C"] is None
    assert "refinement" in ref.to_dict()
    assert "refinement" not in align_session_chain(a, ["A", "B", "C"]).to_dict()      # old shape unchanged


def test_record_in_worldir_has_provenance_and_survives_roundtrip():
    a = anchors(anchor("B", "A", 0.01), anchor("C", "B", 0.01), anchor("C", "A", 0.4, (0.4, 0, 0)))
    world = WorldIR(id="w1")
    rec = record_cross_session_registration(world, refine_session_chain(a, ["A", "B", "C"]))
    assert rec["provenance"] == "registration:pose_graph"
    back = WorldIR.from_dict(world.to_dict())
    assert back.metadata["registration"]["cross_session"]["report"]["refinement"]["status"] == "refined"
    plain = WorldIR(id="w2")
    assert record_cross_session_registration(plain, align_session_chain(a, ["A", "B", "C"]))["provenance"] == \
        "registration:chain"


def _structure():
    rng = np.random.default_rng(7)
    xy = rng.uniform(-5, 5, (500, 2))
    z = 0.6 * np.sin(xy[:, 0]) + 0.4 * np.cos(1.3 * xy[:, 1]) + 0.15 * xy[:, 0] * xy[:, 1] / 5
    return np.column_stack([xy, z])


def _result(points_in_session_frame):
    pts = [ReconstructedPoint(position=tuple(map(float, p)), track_id=f"t{i}", source_evidence_ids=[],
                              uncertainty=Uncertainty(confidence=0.9), color=None)
           for i, p in enumerate(points_in_session_frame)]
    return ReconstructionResult(points=pts, camera_poses=[], registration_status="success")


def test_all_pairs_refine_runs_real_icp_end_to_end():
    world_pts = _structure()            # shared structure in the reference frame A
    results = {}
    for s in ("A", "B", "C"):
        R, t = GT[s]
        results[s] = _result(R.inv().apply(world_pts - t))      # the same points as seen in session s's frame
    rep = align_reconstructed_sessions(results, "A", all_pairs=True, refine=True)
    for s in ("B", "C"):
        assert rep.status_by_session[s] == "aligned", rep.reasons[s]
    # Measured (not designed): the existing ICP acceptance gates ACCEPT a 4.3 m-wrong C->A alignment (rmse 0.55 m,
    # inlier fraction 1.0) while B->A and C->B are exact. In a bare 3-session loop every edge explains that conflict
    # equally well, so the optimiser must report the loop ambiguous and leave the BFS result alone -- not guess.
    assert rep.refinement["status"] == "ambiguous", rep.refinement
    assert "C->A" in rep.refinement["ambiguous_edges"]
    assert err(rep, "B") < 1e-6                       # the exact pair is untouched
