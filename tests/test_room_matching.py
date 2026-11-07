"""M4 (decision: propose, then the user confirms): ranked match proposals with evidence; nothing is ever applied by proposing.

Rooms come from the synthetic box room (tests/layout_synth.py), each in its own id prefix like separate sessions.
"""
import copy

import pytest

from engine.pipeline.room_matching import propose_matches, room_signature, similarity
from engine.pipeline.single_image import fuse_single_views
from tests.test_room_connect import _door, _room


@pytest.fixture()
def world(tmp_path, monkeypatch):
    return fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2")]).world


def test_a_room_signature_carries_its_measured_dimensions_and_openings(world):
    sig = room_signature(world, "boot1")
    assert sig["height_m"] == pytest.approx(2.6, abs=0.4) and sig["long_m"] >= sig["short_m"] > 0
    assert [o["kind"] for o in sig["openings"]] == ["door"]


def test_the_same_room_seen_twice_is_a_high_confidence_revisit_proposal(world):
    props = propose_matches(world)["revisits"]
    assert len(props) == 1 and {props[0]["room_a"], props[0]["room_b"]} == {"boot1", "boot2"}
    assert props[0]["confidence"] >= 0.8 and props[0]["state"] == "proposed"
    assert {"dimensions", "ceiling_height", "openings"} <= set(props[0]["evidence"])


def test_door_pairs_between_rooms_are_proposed_with_their_size_difference(world):
    joins = propose_matches(world)["door_joins"]
    assert len(joins) == 1
    j = joins[0]
    assert {j["door_a"], j["door_b"]} == {_door(world, "boot1").id, _door(world, "boot2").id}
    assert j["width_diff_m"] == pytest.approx(0.0, abs=0.1) and j["confidence"] >= 0.8 and j["state"] == "proposed"


def test_proposing_changes_nothing(world):
    before = copy.deepcopy(world.to_dict())
    propose_matches(world)
    assert world.to_dict() == before


def test_a_much_wider_door_is_not_proposed_and_dissimilar_rooms_score_low(tmp_path, monkeypatch):
    w = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2", door=(-1.5, 1.5, 0.0, 2.1))]).world
    assert propose_matches(w)["door_joins"] == []
    assert similarity(room_signature(w, "boot1"), room_signature(w, "boot2"))["score"] < 0.95


def test_a_connected_door_is_not_proposed_again(world):
    from engine.pipeline.rooms import connect_rooms

    connect_rooms(world, _door(world, "boot1").id, _door(world, "boot2").id)
    assert propose_matches(world)["door_joins"] == []


def _states(cs, kind):
    return sorted(c["state"] for c in cs["changes"] if c["kind"] == kind)


def test_the_same_room_twice_is_all_matched(world):
    from engine.pipeline.room_matching import change_set

    cs = change_set(world, "boot1", "boot2")
    assert set(_states(cs, "wall")) == {"matched"} and _states(cs, "opening") == ["matched"]
    assert set(cs["summary"]) == {"matched"} and cs["summary"]["matched"] >= 2 and cs["state"] == "proposed"


def test_a_changed_opening_is_conflicting_and_one_that_vanished_is_uncertain(tmp_path, monkeypatch):
    from engine.pipeline.room_matching import change_set

    w = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2", door=(-1.5, 1.5, 0.0, 2.1))]).world
    cs = change_set(w, "boot1", "boot2")
    assert _states(cs, "opening") == ["conflicting"]       # a 1 m door and a 3 m opening on the same wall: it cannot just be "updated"
    gone = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2")]).world
    for e in list(gone.entities.values()):
        if e.id.startswith("boot2") and e.type.value == "door":
            gone.entities[e.id].custom_properties["host_wall_id"] = ""
    for e in gone.entities.values():
        slab = (e.custom_properties or {}).get("wall_slab")
        if slab and e.id.startswith("boot2"):
            slab["openings"] = []
    assert _states(change_set(gone, "boot1", "boot2"), "opening") == ["uncertain"]   # seen before, not seen now: not assumed gone


def test_change_set_is_read_only_and_says_positions_are_not_compared(world):
    import copy

    from engine.pipeline.room_matching import change_set

    before = copy.deepcopy(world.to_dict())
    cs = change_set(world, "boot1", "boot2")
    assert world.to_dict() == before and "shared frame" in cs["note"]


def test_visual_evidence_joins_the_confidence_and_is_shown_as_evidence(world):
    high = propose_matches(world, visual=lambda a, b: {"score": 1.0, "best_inliers": 40, "pairs_tested": 2})["revisits"][0]
    low = propose_matches(world, visual=lambda a, b: {"score": 0.0, "best_inliers": 2, "pairs_tested": 2})["revisits"][0]
    none = propose_matches(world, visual=lambda a, b: None)["revisits"][0]
    assert high["evidence"]["visual"] == 1.0 and low["evidence"]["visual"] == 0.0 and none["evidence"]["visual"] is None
    assert high["confidence"] >= none["confidence"] > low["confidence"]
    assert high["visual_detail"]["best_inliers"] == 40
