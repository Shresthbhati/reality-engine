"""T-6.7: undo/redo that survives a reload and never crosses worlds, on a really reconstructed world.

Each edit is a committed WorldStore version; undo/redo move HEAD along the version chain (the redo list is in the database).
"Reload" is a fresh read of /history and /worldir: nothing lives in memory between calls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_commit_validation_contract import _commit, world  # noqa: E402,F401  (fixture reuse)


def _name(c, wid, eid):
    return c.get(f"/api/worlds/{wid}/worldir").json()["entities"][eid]["name"]


def _rename(c, wid, eid, name):
    r = _commit(c, wid, {"entity_id": eid, "changes": {"name": name}})
    assert r.status_code == 200, r.text


def test_a_reconstruction_is_the_floor_undo_never_crosses(world):
    c, wid, _ = world
    h = c.get(f"/api/worlds/{wid}/history").json()
    assert not h["can_undo"] and not h["can_redo"]
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 409


def test_undo_and_redo_walk_the_edit_chain_and_survive_a_reload(world):
    c, wid, eid = world
    original = _name(c, wid, eid)
    _rename(c, wid, eid, "First name")
    _rename(c, wid, eid, "Second name")
    h = c.get(f"/api/worlds/{wid}/history").json()
    assert h["can_undo"] and h["undo_depth"] == 2 and not h["can_redo"]

    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert _name(c, wid, eid) == "First name"
    # a "reload": nothing but the database and the store hold the state
    h = c.get(f"/api/worlds/{wid}/history").json()
    assert h["can_undo"] and h["can_redo"] and h["undo_depth"] == 1 and h["redo_depth"] == 1

    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert _name(c, wid, eid) == original
    assert not c.get(f"/api/worlds/{wid}/history").json()["can_undo"]
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 409            # the reconstruction itself stays

    assert c.post(f"/api/worlds/{wid}/redo").status_code == 200
    assert c.post(f"/api/worlds/{wid}/redo").status_code == 200
    assert _name(c, wid, eid) == "Second name"
    assert c.post(f"/api/worlds/{wid}/redo").status_code == 409


def test_a_new_edit_after_an_undo_drops_the_stale_redo(world):
    c, wid, eid = world
    _rename(c, wid, eid, "Kept then undone")
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    _rename(c, wid, eid, "Branch")
    h = c.get(f"/api/worlds/{wid}/history").json()
    assert not h["can_redo"], "redoing the old edit would silently discard 'Branch'"
    assert c.post(f"/api/worlds/{wid}/redo").status_code == 409
    assert _name(c, wid, eid) == "Branch"


def test_undoing_one_world_never_touches_another(world):
    c, wid, eid = world
    dup = c.post(f"/api/worlds/{wid}/duplicate", json={})
    assert dup.status_code in (200, 201), dup.text
    other_id = dup.json().get("id") or dup.json().get("world_id")
    assert other_id and other_id != wid
    _rename(c, wid, eid, "Only in the original")
    other_before = c.get(f"/api/worlds/{other_id}/history").json()
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert c.get(f"/api/worlds/{other_id}/history").json() == other_before
    assert c.post(f"/api/worlds/{other_id}/undo").status_code == 409        # B has no edits of its own to undo
