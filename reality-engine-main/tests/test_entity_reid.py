"""Tests for cross-session entity re-identification (world_ir/entity_reid.py)."""

from __future__ import annotations

import pytest

from world_ir.entity_reid import MatchKind, reidentify_entities
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def _positioned(entity_id, x, y, z, entity_type=EntityType.WALL, **kwargs):
    return Entity(id=entity_id, type=entity_type, transform={"position": {"x": x, "y": y, "z": z}}, **kwargs)


def _world_with(*entities) -> WorldIR:
    world = WorldIR()
    for e in entities:
        world.entities[e.id] = e
    return world


def test_same_position_same_type_is_a_match():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 0.0, 0.0, 0.0))

    result = reidentify_entities(before, after)
    assert len(result.matched) == 1
    match = result.matched[0]
    assert match.before_entity_id == "wall-old"
    assert match.after_entity_id == "wall-new"
    assert match.distance_m == 0.0


def test_moderate_move_is_possible_match_not_match():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 1.0, 0.0, 0.0))  # 1.0m: beyond 0.5m match, within 2.0m possible

    result = reidentify_entities(before, after)
    assert len(result.possible_matches) == 1
    assert result.possible_matches[0].kind is MatchKind.POSSIBLE_MATCH


def test_large_move_is_no_match():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 10.0, 0.0, 0.0))

    result = reidentify_entities(before, after)
    assert len(result.unmatched) == 1
    assert result.unmatched[0].after_entity_id is None
    assert result.unmatched[0].distance_m == 10.0


def test_different_type_never_matches_even_at_same_position():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0, entity_type=EntityType.WALL))
    after = _world_with(_positioned("floor-new", 0.0, 0.0, 0.0, entity_type=EntityType.FLOOR))

    result = reidentify_entities(before, after)
    assert len(result.unmatched) == 1
    assert "no entity of type wall exists" in result.unmatched[0].reason


def test_unresolved_when_before_entity_has_no_position():
    before = _world_with(Entity(id="ent-bare", type=EntityType.WALL))
    after = _world_with(_positioned("wall-new", 0.0, 0.0, 0.0))

    result = reidentify_entities(before, after)
    assert len(result.unresolved) == 1
    assert result.unresolved[0].after_entity_id is None
    assert result.unresolved[0].distance_m is None


def test_new_entity_ids_excludes_matched_and_possibly_matched():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(
        _positioned("wall-new", 0.0, 0.0, 0.0),        # matched to wall-old
        _positioned("wall-extra", 50.0, 0.0, 0.0),      # genuinely new
    )

    result = reidentify_entities(before, after)
    assert result.new_entity_ids(after) == ("wall-extra",)


def test_ties_broken_deterministically_by_candidate_id():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(
        _positioned("wall-b", 0.5, 0.0, 0.0),
        _positioned("wall-a", -0.5, 0.0, 0.0),  # same distance, but "wall-a" < "wall-b"
    )

    result = reidentify_entities(before, after)
    assert result.matched[0].after_entity_id == "wall-a"


def test_deterministic_repeated_runs_produce_identical_result():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 0.1, 0.0, 0.0))

    r1 = reidentify_entities(before, after).to_dict()
    r2 = reidentify_entities(before, after).to_dict()
    assert r1 == r2


def test_rejects_invalid_thresholds():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 0.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        reidentify_entities(before, after, match_distance_m=-1.0)
    with pytest.raises(ValueError):
        reidentify_entities(before, after, match_distance_m=5.0, possible_match_distance_m=1.0)


def test_empty_after_world_yields_no_match_for_every_before_entity():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = WorldIR()

    result = reidentify_entities(before, after)
    assert len(result.unmatched) == 1
    assert result.unmatched[0].distance_m is None


def test_never_mutates_either_world():
    before = _world_with(_positioned("wall-old", 0.0, 0.0, 0.0))
    after = _world_with(_positioned("wall-new", 0.0, 0.0, 0.0))
    before_snapshot = before.entities["wall-old"].to_dict()
    after_snapshot = after.entities["wall-new"].to_dict()

    reidentify_entities(before, after)

    assert before.entities["wall-old"].to_dict() == before_snapshot
    assert after.entities["wall-new"].to_dict() == after_snapshot
