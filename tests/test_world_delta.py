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
    assert len(d["not_measurable"]) == 3
    assert any("entity continuity" in m for m in d["not_measurable"])
    assert any("structure counts" in m for m in d["not_measurable"])
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


# ------------------------------------------------------------------ entity continuity + conflicts


def _place(cams_map, shift=(0.0, 0.0, 0.0), scale=1.0):
    return {k: [scale * v[0] + shift[0], scale * v[1] + shift[1], scale * v[2] + shift[2]] for k, v in cams_map.items()}


def _ent(t, c, prov="RECONSTRUCTED"):
    return {"id": f"{t}-{c[0]}", "type": t, "center": list(c), "provenance": prov, "confidence": 0.8}


def test_entity_continuity_is_measured_in_a_common_frame_not_by_id():
    cams = ring()
    walls = [_ent("wall", (3, 0, 0)), _ent("wall", (0, 3, 0)), _ent("wall", (-3, 0, 0)), _ent("floor", (0, 0, -1))]
    # candidate lives in a different frame (scaled x2.5 and shifted) and re-derives every id
    def mv(c):
        return [2.5 * c[0] + 10, 2.5 * c[1] - 4, 2.5 * c[2] + 3]
    cand_ents = [dict(e, id="new-" + e["id"], center=mv(e["center"])) for e in walls]
    cand_ents[2] = dict(cand_ents[2], center=mv((-3, 2.0, 0)))                 # this wall genuinely moved
    cand_ents.append(_ent("wall", tuple(mv((0, -3, 0)))))                     # and one wall is new
    res = wd.entity_continuity(walls, cand_ents, cams, _place(cams, (10, -4, 3), 2.5))
    # two walls + the floor unchanged; the third wall moved 2 units (> 20% of extent) so it is NOT
    # matched: it is reported removed, and its new position plus the extra wall count as new
    assert (res["preserved"], res["refined"], len(res["removed"]), res["new"]) == (3, 0, 1, 2)
    assert res["removed"][0]["type"] == "wall" and "centre distance" in res["matched_by"] and res["signals"] == {"center_only": 3}


def test_entity_continuity_not_measurable_without_a_camera_alignment():
    assert wd.entity_continuity([_ent("wall", (0, 0, 0))], [_ent("wall", (0, 0, 0))], {}, {}) is None


def _lost_walls_delta(extra_registered=()):
    cams = ring()
    prev = snap(list(cams), cams=cams, types={"wall": 4})
    prev["entities"] = [_ent("wall", (3, 0, 0)), _ent("wall", (0, 3, 0)), _ent("wall", (-3, 0, 0)), _ent("wall", (0, -3, 0))]
    cand = snap(list(cams) + list(extra_registered), cams=cams, types={"wall": 4})
    cand["entities"] = [_ent("wall", (3, 0, 0))]                               # 3 of 4 walls not reproduced
    return wd.compute_delta(prev, cand)


def test_established_structure_lost_with_nothing_to_justify_it_is_a_regression():
    d = _lost_walls_delta()
    dec = wd.decide(d)
    assert d["entities"]["preserved"] == 1 and len(d["entities"]["removed"]) == 3
    assert dec["verdict"] == wd.REJECT and "no additional photograph" in dec["reasons"][0]


def test_the_same_loss_alongside_newly_placed_photos_is_uncertainty_not_rejection():
    dec = wd.decide(_lost_walls_delta(extra_registered=["g"]))
    assert dec["verdict"] == wd.ACCEPT_WITH_UNCERTAINTY and any("not reproduced" in u for u in dec["uncertainties"])


def _moved_delta(displaced=("e0",), amount=3.0):
    cams = ring()
    cand = {k: list(v) for k, v in cams.items()}
    for k in displaced:
        cand[k][0] += amount
    return wd.compute_delta(snap(list(cams), cams=cams), snap(list(cams), cams=cand))


def test_conflict_is_opened_with_both_hypotheses_and_provenance_then_preserved():
    d = _moved_delta()
    conflicts = wd.reconcile_conflicts(None, d, "v2")
    assert len(conflicts) == 1 and conflicts[0]["status"] == "unresolved" and conflicts[0]["subject"] == "e0"
    hyp = conflicts[0]["hypotheses"]
    assert [h["source"] for h in hyp] == ["previous_version", "v2"] and hyp[0]["position"] != hyp[1]["position"]
    assert all(h["provenance"] == ["e0"] and h["confidence"] is None for h in hyp)   # confidence NOT invented
    assert wd.decide(d, conflicts)["verdict"] == wd.ACCEPT_WITH_UNCERTAINTY
    # a later version that cannot see the camera at all must NOT drop the conflict
    quiet = wd.compute_delta(snap(["a", "b", "c"]), snap(["a", "b", "c"]))
    kept = wd.reconcile_conflicts(conflicts, quiet, "v3")
    assert len(kept) == 1 and kept[0]["status"] == "unresolved"


def test_conflict_is_resolved_only_when_a_later_version_measurably_agrees_and_history_is_kept():
    opened = wd.reconcile_conflicts(None, _moved_delta(), "v2")
    agree = _moved_delta(displaced=(), amount=0.0)                              # every camera consistent
    resolved = wd.reconcile_conflicts(opened, agree, "v3")
    assert resolved[0]["status"] == "resolved"
    assert [h["event"] for h in resolved[0]["history"]] == ["opened", "resolved"]
    assert len(resolved[0]["hypotheses"]) == 2                                  # provenance retained after resolution
    text = " ".join(wd.describe_changes(agree, conflicts=resolved))
    assert "earlier conflict resolved" in text
    # moved AGAIN keeps the conflict open and records the new hypothesis
    again = wd.reconcile_conflicts(opened, _moved_delta(), "v3")
    assert again[0]["status"] == "unresolved" and again[0]["hypotheses"][-1]["source"] == "v3"
    assert again[0]["history"][-1]["event"] == "moved_again"


# ------------------------------------------------------ geometry-aware reconciliation, end to end
import numpy as np                                                     # noqa: E402

from tests.test_spatial_continuity import WALL, lattice, rec           # noqa: E402


def _world(entities, registered=("a", "b", "c", "d", "e", "f"), extra_input=()):
    cams = ring()
    s = snap(list(registered), inputs=list(registered) + list(extra_input), cams=cams)
    s["entities"] = entities
    return s


def test_split_is_not_a_loss_and_the_gate_accepts_it_as_a_representation_change():
    a1, a2 = lattice(0, 3, 0, 2), lattice(3.05, 6, 0, 2)
    o1, o2 = lattice(0, 3, 0, 2, origin=(0, 0, 8)), lattice(0, 3, 0, 2, origin=(0, 0, -8))   # two distinct walls
    prev = _world([rec("w", "wall", WALL), rec("o1", "wall", o1), rec("o2", "wall", o2)])
    cand = _world([rec("f1", "wall", a1), rec("f2", "wall", a2), rec("o1", "wall", o1), rec("o2", "wall", o2)])
    d = wd.compute_delta(prev, cand)
    assert d["entities"]["counts"]["split"] == 1 and d["entities"]["counts"]["removed"] == 0
    assert wd.decide(d)["verdict"] == wd.ACCEPT
    phys = wd.physical_changes(d)
    assert phys["represented_differently"] == ["one wall is now 2 fragments (same surface, different grouping)"]
    assert phys["added"] == [] and phys["not_reproduced"] == []
    # the status carries ALL ten relation categories, so the Studio can show each one (zero included)
    assert list(phys["counts"]) == ["preserved", "refined", "extended", "reduced", "split", "merge", "regrouped",
                                    "ambiguous", "removed", "new"]
    assert phys["counts"]["split"] == 1 and phys["counts"]["preserved"] == 2 and phys["counts"]["removed"] == 0


def test_unsupported_move_opens_a_geometry_conflict_keeping_both_positions_with_real_provenance():
    shifted = WALL + np.array([0, 0, 0.25])
    prev = _world([rec("w", "wall", WALL, evidence=("a", "b"))])
    cand = _world([rec("w2", "wall", shifted, evidence=("a", "b"))])          # same evidence, different place
    d = wd.compute_delta(prev, cand)
    conflicts = wd.reconcile_conflicts(None, d, "v2")
    (c,) = [c for c in conflicts if c["kind"] == "geometry"]
    assert c["status"] == "unresolved" and c["subject_type"] == "wall"
    assert [h["source"] for h in c["hypotheses"]] == ["previous_version", "v2"]
    assert c["hypotheses"][0]["provenance"] == ["a", "b"] and c["hypotheses"][0]["confidence"] == 0.8    # real values
    dec = wd.decide(d, conflicts)
    assert dec["verdict"] == wd.ACCEPT_WITH_UNCERTAINTY and any("conflict" in u for u in dec["uncertainties"])


def test_a_move_backed_by_new_photographs_is_a_refinement_not_a_conflict():
    shifted = WALL + np.array([0, 0, 0.25])
    prev = _world([rec("w", "wall", WALL, evidence=("a", "b"))])
    cand = _world([rec("w2", "wall", shifted, evidence=("a", "b", "g"))], registered=("a", "b", "c", "d", "e", "f", "g"))
    d = wd.compute_delta(prev, cand)
    assert not [c for c in wd.reconcile_conflicts(None, d, "v2") if c["kind"] == "geometry"]
    phys = wd.physical_changes(d)
    assert phys["unsupported_moves"] == 0 and any("backed by new photographs" in x for x in phys["refined"])


def test_geometry_conflict_lifecycle_open_persist_then_resolve_with_history_and_provenance_kept():
    shifted = WALL + np.array([0, 0, 0.25])
    v1 = _world([rec("w", "wall", WALL, evidence=("a", "b"))])
    v2 = _world([rec("w2", "wall", shifted, evidence=("a", "b"))])
    c2 = wd.reconcile_conflicts(None, wd.compute_delta(v1, v2), "v2")             # V2: conflict introduced
    assert [c["status"] for c in c2 if c["kind"] == "geometry"] == ["unresolved"]

    v3_same = _world([rec("w3", "wall", shifted, evidence=("a", "b"))])            # V3: nothing new supports it
    c3 = wd.reconcile_conflicts(c2, wd.compute_delta(v2, v3_same), "v3")
    (g3,) = [c for c in c3 if c["kind"] == "geometry"]
    assert g3["status"] == "unresolved" and g3["history"][-1]["event"] == "still_unresolved"

    v4 = _world([rec("w4", "wall", shifted, evidence=("a", "b", "g"))], registered=("a", "b", "c", "d", "e", "f", "g"))
    d4 = wd.compute_delta(v3_same, v4)                                             # V4: a new photograph sees it there
    c4 = wd.reconcile_conflicts(c3, d4, "v4")
    (g4,) = [c for c in c4 if c["kind"] == "geometry"]
    assert g4["status"] == "resolved" and g4["resolved_to"] == "v2"
    assert [h["event"] for h in g4["history"]] == ["opened", "still_unresolved", "resolved"]
    assert len(g4["hypotheses"]) == 2 and g4["hypotheses"][0]["provenance"] == ["a", "b"]     # nothing erased
    assert "resolved by this version" in " ".join(wd.describe_changes(d4, conflicts=c4))


def test_conflicts_are_carried_into_the_newest_frame_and_never_dropped_when_the_frame_is_lost():
    shifted = WALL + np.array([0, 0, 0.25])
    d = wd.compute_delta(_world([rec("w", "wall", WALL, evidence=("a",))]), _world([rec("w2", "wall", shifted, evidence=("a",))]))
    c = wd.reconcile_conflicts(None, d, "v2")
    quiet = wd.compute_delta(snap(["a", "b", "c"]), snap(["a", "b", "c"]))       # no shared cameras -> no alignment
    kept = wd.reconcile_conflicts(c, quiet, "v3")
    assert [k["status"] for k in kept if k["kind"] == "geometry"] == ["unresolved"]
    assert [k for k in kept if k["kind"] == "geometry"][0].get("frame_lost") is True


def test_physical_summary_names_new_and_extended_surfaces_and_the_regions_new_photos_touched():
    bigger = lattice(0, 12, 0, 2)
    far = lattice(0, 4, 0, 2, origin=(0, 0, 9))
    prev = _world([rec("w", "wall", WALL, evidence=("a",))], extra_input=())
    cand = _world([rec("w2", "wall", bigger, evidence=("a", "g")), rec("n", "wall", far, evidence=("g",))],
                  registered=("a", "b", "c", "d", "e", "f", "g"))
    phys = wd.physical_changes(wd.compute_delta(prev, cand))
    assert phys["extended"][0].startswith("1 wall")
    assert phys["added"] == ["1 wall"]
    touched = [r for r in phys["regions"] if r["affected_by_new_evidence"]]
    assert touched and all(r["affected_by_new_evidence"] == ["g"] for r in touched)


def test_physical_summary_reports_not_available_with_the_reason_instead_of_inventing_one():
    d = wd.compute_delta(snap(["a", "b", "c"]), snap(["a", "b", "c", "d"]))
    phys = wd.physical_changes(d)
    assert phys["available"] is False and "entity continuity" in phys["reason"]


def test_an_identical_rerun_is_unchanged_but_anything_new_is_not():
    prev = snap(["a", "b", "c", "d"], level=2)
    same = snap(["a", "b", "c", "d"], level=2)
    assert wd.is_unchanged(prev, same, wd.compute_delta(prev, same))
    for cand in (snap(["a", "b", "c", "d", "e"], level=2),                       # a newly placed photo
                 snap(["a", "b", "c"], inputs=["a", "b", "c", "d"], level=2),       # a photo lost
                 snap(["a", "b", "c", "d"], inputs=["a", "b", "c", "d", "x"], level=2),  # new waiting evidence
                 snap(["a", "b", "c", "d"], level=3)):                              # a higher level (e.g. dense)
        assert not wd.is_unchanged(prev, cand, wd.compute_delta(prev, cand))
    assert not wd.is_unchanged(None, same, wd.compute_delta(None, same))            # first version is never "unchanged"
