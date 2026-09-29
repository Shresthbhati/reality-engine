"""World-level acceptance: a valid candidate is not automatically a better one.

Model-free: snapshots are plain dicts, so every verdict is provable in milliseconds.
"""

from __future__ import annotations

import math

import numpy as np

from engine.pipeline import world_delta as wd


def snap(registered, inputs=None, level=2, cams=None, types=None, points=1000):
    return {
        "registered_ids": list(registered), "input_ids": list(inputs or registered), "level": level,
        "model_state": "PARTIAL", "points": points, "cameras": cams or {},
        "entity_types": types if types is not None else {"wall": 4, "floor": 1}, "provenance": {},
    }


def ring(n=6, radius=3.0):
    return {f"e{i}": [radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), 0.1 * i]
            for i in range(n)}


def test_camera_check_is_frame_independent_rotated_scaled_shifted_copy_did_not_move():
    prev = ring()
    th = 0.7
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    cand = {k: (2.5 * rot @ np.array(v) + np.array([10, -4, 3])).tolist() for k, v in prev.items()}
    cc = wd.camera_consistency(prev, cand)
    assert cc["common"] == 6 and cc["relative"] < 1e-6


def test_camera_check_detects_real_movement_and_refuses_when_not_measurable():
    prev = ring()
    cand = {k: [v[0] + (1.5 if i % 2 else -1.5), v[1], v[2]] for i, (k, v) in enumerate(prev.items())}
    assert wd.camera_consistency(prev, cand)["relative"] > wd.CAMERA_MOVED_REL
    assert wd.camera_consistency({"a": [0, 0, 0]}, {"a": [1, 1, 1]}) is None      # < 4 shared: not measurable


def test_first_version_is_accepted_and_described():
    d = wd.compute_delta(None, snap(["a", "b", "c"], inputs=["a", "b", "c", "d"]))
    assert wd.decide(d)["verdict"] == wd.ACCEPT
    assert d["evidence"]["still_waiting"] == ["d"]
    assert "still waiting" in " ".join(wd.describe_changes(d))


def test_more_registered_evidence_is_accepted_and_reported():
    prev = snap(["a", "b", "c"])
    cand = snap(["a", "b", "c", "d", "e"], types={"wall": 6, "floor": 1})
    d = wd.compute_delta(prev, cand)
    dec = wd.decide(d)
    assert dec["verdict"] == wd.ACCEPT and d["evidence"]["gained"] == ["d", "e"]
    text = " ".join(wd.describe_changes(d))
    assert "2 more photos placed" in text and "+2 walls (4 -> 6)" in text


def test_candidate_that_loses_registered_evidence_without_gain_is_rejected():
    d = wd.compute_delta(snap(["a", "b", "c"]), snap(["a"], inputs=["a", "b", "c", "d"], level=2))
    dec = wd.decide(d)
    assert dec["verdict"] == wd.REJECT and "no net gain" in dec["reasons"][0]


def test_fall_back_to_single_views_is_rejected_when_head_had_a_real_reconstruction():
    d = wd.compute_delta(snap(["a", "b", "c"], level=2), snap([], inputs=["a", "b", "c", "d"], level=0))
    assert wd.decide(d)["verdict"] == wd.REJECT


def test_loss_offset_by_bigger_gain_is_adopted_but_flagged_uncertain():
    d = wd.compute_delta(snap(["a", "b", "c"]), snap(["a", "b", "d", "e", "f"], inputs=list("abcdef")))
    dec = wd.decide(d)
    assert dec["verdict"] == wd.ACCEPT_WITH_UNCERTAINTY and "offset by 3" in dec["uncertainties"][0]


def test_shrinking_structure_and_moved_cameras_are_uncertainty_not_rejection():
    prev = snap(list("abcdef"), cams=ring(), types={"wall": 6})
    cand = snap(list("abcdef") + ["g"], cams={k: [v[0] + (2 if i % 2 else -2), v[1], v[2]]
                                             for i, (k, v) in enumerate(ring().items())}, types={"wall": 2})
    dec = wd.decide(wd.compute_delta(prev, cand))
    assert dec["verdict"] == wd.ACCEPT_WITH_UNCERTAINTY
    assert any("wall count fell from 6 to 2" in u for u in dec["uncertainties"])
    assert any("moved by" in u for u in dec["uncertainties"])


def test_missing_history_is_reported_not_measurable_never_silently_passed():
    prev = snap(["a", "b", "c"], cams={})
    prev["entity_types"] = None                     # an older version that never recorded structure
    d = wd.compute_delta(prev, snap(["a", "b", "c", "d"]))
    assert len(d["not_measurable"]) == 2
    assert wd.decide(d)["verdict"] == wd.ACCEPT


def test_snapshot_from_report_round_trip_and_none_for_empty():
    assert wd.snapshot_from_report({}) is None
    rep = {"level": 2, "model_state": "PARTIAL", "evidence": {"registered_ids": ["a"], "input_ids": ["a", "b"]},
           "structure": {"entity_types": {"wall": 1}, "provenance": {}}, "cameras": {"a": [0, 0, 0]},
           "stages": {"reconstruction": {"points": 5}}}
    s = wd.snapshot_from_report(rep)
    assert s["registered_ids"] == ["a"] and s["points"] == 5 and s["entity_types"] == {"wall": 1}


def test_rejected_candidate_reports_only_evidence_facts_never_its_own_structure():
    d = wd.compute_delta(snap(["a", "b", "c"], types={"wall": 2}), snap([], inputs=list("abcd"), level=0, types={"wall": 18}))
    text = " ".join(wd.describe_changes(d, structure=False))
    assert "wall" not in text and "could no longer be placed" in text
    assert "+16 walls" in " ".join(wd.describe_changes(d))            # adopted runs do report structure


def test_single_view_anchor_is_not_a_registration_so_a_fused_candidate_is_not_a_regression():
    """V1 = one photo (level 0, its own anchor listed as 'registered'). A candidate covering more
    photos as independent views registers nothing -- but nothing real was registered before."""
    d = wd.compute_delta(snap(["a"], level=0), snap([], inputs=["a", "b", "c"], level=0))
    assert d["evidence"]["lost"] == [] and wd.decide(d)["verdict"] == wd.ACCEPT
    # and level-1+ registration still counts
    d2 = wd.compute_delta(snap(["a", "b"], level=1), snap([], inputs=["a", "b", "c"], level=0))
    assert wd.decide(d2)["verdict"] == wd.REJECT
