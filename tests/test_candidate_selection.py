"""Incremental-vs-full is decided by how each candidate evolves the WORLD, never by camera count alone.

Model-free: snapshots are plain dicts run through the same world_delta machinery that gates adoption.
"""

from __future__ import annotations

import pytest

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


# ----------------------------------------------------------------------------------------------------------------
# The ARBITRATION BOUNDARY: the object the COLMAP backend actually consults (WorldArbiter.assess/choose on whatever
# its to_snapshot returns for a reconstruction result), not the helpers it is built from.
# ----------------------------------------------------------------------------------------------------------------

import math  # noqa: E402

import numpy as np  # noqa: E402


class _Result:
    """Stands in for a ReconstructionResult: the arbiter only ever hands it to ``to_snapshot``."""

    def __init__(self, snapshot):
        self.snapshot = snapshot


def _arbiter(prev=None):
    return cs.WorldArbiter(prev if prev is not None else _prev(), None, lambda res: res.snapshot)


def _arbitrate(inc_snapshot, full_snapshot, prev=None):
    arb = _arbiter(prev)
    return arb.choose(arb.assess(_Result(inc_snapshot), "incremental"), arb.assess(_Result(full_snapshot), "full"))


def _in_new_frame(snapshot, *, deg=90.0, scale=2.5, shift=(10.0, -4.0, 3.0)):
    """The SAME world expressed in another coordinate frame (what a COLMAP full re-solve produces)."""
    th = math.radians(deg)
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])

    def mv(p):
        return (scale * rot @ np.asarray(p, float) + np.asarray(shift)).tolist()

    out = dict(snapshot)
    out["cameras"] = {k: mv(v) for k, v in snapshot["cameras"].items()}
    out["entities"] = [dict(e, center=mv(e["center"])) for e in snapshot["entities"]]
    return out


def test_the_arbiter_incremental_candidate_that_registers_more_cameras_but_loses_the_world_is_rejected():  # B
    inc = _cand(extra=["g", "h"], walls=WALLS[:1])               # 8 placed, three established walls not reproduced
    full = _cand(extra=["g"])                                    # 7 placed, every wall preserved
    out = _arbitrate(inc, full)
    assert out["choice"] == "full" and out["deciding"] == "established surfaces lost"
    arb = _arbiter()
    assert arb.assess(_Result(inc), "i")["measures"]["registered"] > arb.assess(_Result(full), "f")["measures"]["registered"]


def test_the_arbiter_full_candidate_that_registers_more_cameras_but_loses_continuity_is_rejected():  # C
    drifted = {k: list(v) for k, v in CAMS.items()}
    drifted["e0"][0] += 3.0                                      # an established camera jumped in the full re-solve
    inc = _cand(extra=["g"])                                     # stable, 7 placed
    full = _cand(extra=["g", "h", "i"], cams=drifted)            # 9 placed but the established world moved
    out = _arbitrate(inc, full)
    assert out["choice"] == "incremental" and out["deciding"] in ("established surfaces lost", "unsupported moves / conflicts", "camera stability")
    # and the other side of the same coin: it is continuity, not "incremental is always preferred"
    clean_full = _cand(extra=["g", "h", "i"])
    assert _arbitrate(inc, clean_full)["choice"] == "full"


def test_the_frame_shift_of_an_identical_frame_is_zero_and_of_a_moved_frame_is_measured():
    arb = _arbiter()
    same = arb.assess(_Result(_cand(extra=["g"])), "x")["measures"]
    assert same["frame_changed"] is False and same["frame_shift"]["rotation_deg"] < 1e-3
    moved = arb.assess(_Result(_in_new_frame(_cand(extra=["g"]))), "x")["measures"]
    fs = moved["frame_shift"]
    assert moved["frame_changed"] is True
    assert fs["rotation_deg"] == pytest.approx(90.0, abs=0.1) and fs["scale_change"] == pytest.approx(math.log(2.5), abs=1e-3)
    # the geometry check itself is blind to this by design: nothing was lost, nothing drifted
    assert moved["cameras_lost"] == 0 and moved["structural_lost"] == 0 and (moved["camera_drift"] or 0) < 1e-3


def test_comparable_geometry_in_a_needlessly_new_frame_keeps_the_established_frame():  # D
    """HEAD exists. The full re-solve is the same world (nothing lost or drifted) with ONE more photo placed, but it
    re-expresses everything in a new coordinate frame. Before the frame rule 'useful evidence placed' made it win."""
    inc = _cand(extra=["g"])                                     # 7 placed, HEAD's frame
    full = _in_new_frame(_cand(extra=["g", "h"]))                # 8 placed, new frame
    out = _arbitrate(inc, full)
    assert out["choice"] == "incremental" and out["deciding"] == "coordinate frame preserved", out
    # symmetric: the roles do not matter, the frame-preserving candidate wins either way round
    arb = _arbiter()
    swapped = arb.choose(arb.assess(_Result(full), "incremental"), arb.assess(_Result(inc), "full"))
    assert swapped["choice"] == "full" and swapped["deciding"] == "coordinate frame preserved"


def test_a_material_gain_in_evidence_justifies_replacing_the_frame_and_says_so():
    inc = _cand(extra=["g"])                                     # 7 placed
    full = _in_new_frame(_cand(extra=["g", "h", "i"]))           # 9 placed (+2 == FRAME_CHANGE_MIN_EXTRA_PHOTOS)
    out = _arbitrate(inc, full)
    assert out["choice"] == "full" and out["deciding"] == "useful evidence placed"
    assert "coordinate frame is replaced" in out["why"]


def test_a_new_frame_never_outranks_established_geometry_in_either_direction():
    lossy_new_frame = _in_new_frame(_cand(extra=["g", "h"], walls=WALLS[:1]))
    assert _arbitrate(_cand(extra=["g"]), lossy_new_frame)["deciding"] == "established surfaces lost"
    good_new_frame = _in_new_frame(_cand(extra=["g"]))
    lossy_kept_frame = _cand(extra=["g", "h"], walls=WALLS[:1])
    out = _arbitrate(lossy_kept_frame, good_new_frame)               # the frame-preserving one lost walls
    assert out["choice"] == "full" and out["deciding"] == "established surfaces lost"


def test_frame_preservation_does_not_apply_without_an_established_frame_to_lose():
    arb = cs.WorldArbiter(None, None, lambda res: res.snapshot)
    a = arb.assess(_Result(_cand(extra=["g"])), "i")
    b = arb.assess(_Result(_in_new_frame(_cand(extra=["g", "h"]))), "f")
    assert a["measures"]["frame_changed"] is None and b["measures"]["frame_changed"] is None
    assert cs.choose(a, b)["choice"] == "full"                       # first version: evidence placed decides
