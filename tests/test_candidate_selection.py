"""Incremental-vs-full is decided by how each candidate evolves the WORLD, never by camera count alone.

Model-free: snapshots are plain dicts run through the same world_delta machinery that gates adoption.
"""

from __future__ import annotations

from engine.pipeline import candidate_selection as cs
from engine.pipeline import world_delta as wd
from tests.test_world_delta import _ent, ring, snap

CAMS = ring()
IDS = list(CAMS)
WALLS = [(3, 0, 0), (0, 3, 0), (-3, 0, 0), (0, -3, 0)]


def _prev():
    s = snap(IDS, cams=CAMS, types={"wall": 4})
    s["entities"] = [_ent("wall", c) for c in WALLS]
    return s


def _cand(extra=(), walls=WALLS, cams=None, extra_walls=()):
    s = snap(IDS + list(extra), inputs=IDS + ["g", "h", "i", "j"], cams=cams or CAMS,
             types={"wall": len(walls) + len(extra_walls)})
    s["entities"] = [_ent("wall", c) for c in list(walls) + list(extra_walls)]
    return s


def _pick(inc, full):
    prev = _prev()
    return cs.choose(cs.assess(prev, inc), cs.assess(prev, full))


def test_more_cameras_but_established_wall_lost_keeps_the_incremental_world():          # Case A
    inc = _cand(extra=["g"])                                     # 7 placed, every wall preserved
    full = _cand(extra=["g", "h"], walls=WALLS[:1])              # 8 placed, three walls not reproduced
    out = _pick(inc, full)
    assert out["choice"] == "incremental" and out["deciding"] == "established surfaces lost"


def test_more_cameras_and_new_surface_with_everything_preserved_lets_full_win():        # Case B
    inc = _cand(extra=["g"])
    full = _cand(extra=["g", "h"], extra_walls=[(0, 0, 6)])
    out = _pick(inc, full)
    assert out["choice"] == "full" and out["deciding"] == "useful evidence placed"


def test_same_cameras_better_geometry_lets_full_win_registration_count_not_consulted(): # Case C
    inc = _cand(extra=["g"], walls=WALLS[:3])                    # one established wall not reproduced
    full = _cand(extra=["g"])                                    # identical camera count, wall kept
    out = _pick(inc, full)
    assert out["choice"] == "full" and out["deciding"] == "established surfaces lost"


def test_indistinguishable_candidates_prefer_incremental_to_keep_the_frame():           # Case D
    out = _pick(_cand(extra=["g"]), _cand(extra=["g"]))
    assert out["choice"] == "incremental" and out["deciding"] == "tie"


def test_more_cameras_but_drifted_established_cameras_lets_the_stable_full_win():       # Case E
    drifted = {k: list(v) for k, v in CAMS.items()}
    drifted["e0"][0] += 3.0
    inc = _cand(extra=["g", "h", "i", "j"], cams=drifted)        # 10 placed, an established camera jumped
    full = _cand(extra=["g", "h", "i"])                          # 9 placed, stable
    out = _pick(inc, full)
    m = cs.assess(_prev(), inc)["measures"]
    assert out["choice"] == "full"
    assert m["open_conflicts"] >= 1 and m["camera_drift"] > 0.05      # the incremental model regressed the world


def test_both_questionable_forces_no_winner_and_is_flagged_for_the_adoption_gate():     # Case F
    inc = snap(["e0"], inputs=IDS, cams={"e0": CAMS["e0"]})
    full = snap(["e0", "e1"], inputs=IDS, cams={k: CAMS[k] for k in ("e0", "e1")})
    a, b = cs.assess(_prev(), inc), cs.assess(_prev(), full)
    assert a["verdict"] == b["verdict"] == wd.REJECT
    out = cs.choose(a, b)
    assert out["both_questionable"] and out["deciding"] == "acceptance"


def test_a_rejected_candidate_never_beats_an_acceptable_one_whatever_it_registered():
    bad = snap(["e0", "e1"], inputs=IDS, cams={k: CAMS[k] for k in ("e0", "e1")})
    good = _cand(extra=["g"])
    assert cs.choose(cs.assess(_prev(), bad), cs.assess(_prev(), good))["choice"] == "full"
    assert cs.choose(cs.assess(_prev(), good), cs.assess(_prev(), bad))["choice"] == "incremental"


def test_missing_candidate_is_decided_by_validity():
    a = cs.assess(_prev(), _cand())
    assert cs.choose(None, a)["choice"] == "full" and cs.choose(a, None)["choice"] == "incremental"


def test_camera_drift_below_the_margin_is_noise_not_a_reason_to_replace():
    a = {"verdict": wd.ACCEPT, "reasons": [], "uncertainties": [],
         "measures": {"cameras_lost": 0, "structural_lost": 0, "open_conflicts": 0, "camera_drift": 0.0,
                      "uncertainties": 0, "registered": 8, "new_understanding": 0}}
    b = {**a, "measures": {**a["measures"], "camera_drift": cs.CAMERA_DRIFT_MARGIN / 2, "registered": 8}}
    assert cs.choose(a, b)["deciding"] == "tie"


def test_first_version_has_nothing_to_regress_against_and_falls_to_evidence_then_tie():
    a = cs.assess(None, _cand(extra=["g"]))
    b = cs.assess(None, _cand(extra=["g", "h"]))
    assert cs.choose(a, b)["choice"] == "full"


def test_with_an_established_frame_one_more_new_entity_does_not_justify_a_full_resolve():
    """Regression from the real rich-world journey: equal registration, nothing established lost, the full rebuild
    has one more refined/extended/new entity. Frame preservation outranks that tie-break."""
    base = {"verdict": wd.ACCEPT, "reasons": [], "uncertainties": [],
            "measures": {"cameras_lost": 0, "structural_lost": 0, "open_conflicts": 0, "camera_drift": 0.0,
                         "uncertainties": 0, "registered": 7, "new_understanding": 1, "first_version": False}}
    full = {**base, "measures": {**base["measures"], "new_understanding": 2}}
    out = cs.choose(base, full)
    assert out["choice"] == "incremental" and out["deciding"] == "tie"


def test_without_an_established_frame_new_understanding_still_decides():
    base = {"verdict": wd.ACCEPT, "reasons": [], "uncertainties": [],
            "measures": {"cameras_lost": 0, "structural_lost": 0, "open_conflicts": 0, "camera_drift": 0.0,
                         "uncertainties": 0, "registered": 7, "new_understanding": 1, "first_version": True}}
    full = {**base, "measures": {**base["measures"], "new_understanding": 2}}
    assert cs.choose(base, full)["choice"] == "full"
