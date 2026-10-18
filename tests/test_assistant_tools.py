"""Assistant tool layer (Phase F first slice): deterministic read tools, propose/confirm edits, HTTP surface."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.schema_v1 import Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _shell() -> WorldIR:
    wir = WorldIR(id="x")
    for name, hi in {"floor": (6, 0.1, 5), "wall-n": (6, 2.6, 0.2)}.items():
        gid = f"g-{name}"
        wir.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0),
                                       bounds_max=Vector3(*hi), provenance=Provenance.RECONSTRUCTED)
        wir.entities[name] = Entity(id=name, type=EntityType.FLOOR if name == "floor" else EntityType.WALL,
                                    name=name, geometry_ids=[gid], provenance=Provenance.RECONSTRUCTED,
                                    confidence=0.8)
    # nothing recorded about this one: default confidence, no evidence, UNKNOWN provenance
    wir.entities["mystery"] = Entity(id="mystery", type=EntityType.UNKNOWN, name="mystery thing")
    return wir


def _shell_world(client) -> str:
    wid = client.post("/api/worlds", json={"name": "Shell"}).json()["id"]
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    async def go():
        async with db_mod.get_sessionmaker()() as db:
            await svc.commit_version(db, world_id=wid, world=_shell(), parent=None)

    asyncio.run(go())
    return wid


def _proc_world(client) -> str:
    wid = client.post("/api/worlds", json={"name": "Proc"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/rooms", json={"name": "A", "width": 4, "depth": 4, "height": 2.6}).status_code == 201
    r = client.post(f"/api/worlds/{wid}/corridors", json={"name": "c", "length": 5, "attach": {"to": "A-room", "side": "east"}})
    assert r.status_code == 201, r.text
    return wid


def _head(client, wid) -> str:
    return client.get(f"/api/worlds/{wid}").json()["current_version_id"]


def _call(client, wid, tool, **args):
    return client.post(f"/api/worlds/{wid}/assistant/call", json={"tool": tool, "args": args})


def _ok(client, wid, tool, **args):
    r = _call(client, wid, tool, **args)
    assert r.status_code == 200, r.text
    return r.json()["result"]


def _op(**kw):
    return {"operation": kw}


# ---------------------------------------------------------------------------- registry


def test_assistant_tools_listing_is_json_schema_style(client):
    tools = client.get("/api/assistant/tools").json()["tools"]
    names = {t["name"] for t in tools}
    assert {"list_entities", "get_entity", "find_by_name", "measure", "explain_provenance", "search", "propose_edit"} <= names
    for t in tools:
        assert t["parameters"]["type"] == "object" and "properties" in t["parameters"]
    assert [t["name"] for t in tools if t["mutating"]] == ["propose_edit"]


# ------------------------------------------------------------------- read tools: procedural


def test_assistant_read_tools_on_procedural_world(client):
    wid = _proc_world(client)
    proc = _ok(client, wid, "list_entities", provenance="PROCEDURAL")
    assert proc["total"] > 0 and all(e["provenance"] == "PROCEDURAL" for e in proc["entities"])
    walls = _ok(client, wid, "list_entities", kind="wall")
    assert walls["total"] > 0 and all(e["type"] == "wall" for e in walls["entities"])
    assert _ok(client, wid, "list_entities", provenance="OBSERVED")["total"] == 0

    ent = _ok(client, wid, "get_entity", entity_id="A-room")
    assert ent["spec"]["kind"] == "room" and ent["spec"]["params"]["width"] == 4
    assert ent["provenance"] == "PROCEDURAL"

    name = ent["name"]
    found = _ok(client, wid, "find_by_name", name=name.upper())
    assert any(m["id"] == "A-room" for m in found["matches"]), (name, found)
    assert found["exact"] >= 1 and found["matches"][0]["name"] == name

    s1, s2 = _ok(client, wid, "search", query="corridor c"), _ok(client, wid, "search", query="corridor c")
    assert s1 == s2 and s1["results"][0]["id"] == "c-corridor"  # name+id hit outranks the rest, stable order

    expl = _ok(client, wid, "explain_provenance", entity_id="A-room")
    assert "PROCEDURAL" in expl["text"] and "room_grammar" in expl["text"] and "seed 0" in expl["text"]


def test_assistant_measure_unit_honesty(client):
    wid = _proc_world(client)
    m = _ok(client, wid, "measure", kind="distance", points=[[0, 0, 0], [3, 4, 0]])
    assert m["value"] == 5.0 and m["unit"] == "model_units" and m["scale_state"] == "unknown"
    m = _ok(client, wid, "measure", kind="distance", points=[[0, 0, 0], [3, 4, 0]], scale_state="metric")
    assert m["unit"] == "m"
    assert _call(client, wid, "measure", kind="distance", points=[[0, 0, 0]]).status_code == 422


# ------------------------------------------------------------ read tools: reconstructed shell


def test_assistant_read_tools_on_reconstructed_shell(client):
    wid = _shell_world(client)
    rec = _ok(client, wid, "list_entities", provenance="reconstructed")
    assert {e["id"] for e in rec["entities"]} == {"floor", "wall-n"}
    wall = _ok(client, wid, "get_entity", entity_id="wall-n")
    assert wall["confidence"] == 0.8 and wall["bounds"][0]["max"] == [6, 2.6, 0.2] and wall["spec"] is None
    assert _ok(client, wid, "search", query="wall")["results"][0]["id"] == "wall-n"
    assert _call(client, wid, "get_entity", entity_id="nope").status_code == 422


def test_assistant_explain_never_states_unrecorded_values(client):
    wid = _shell_world(client)
    text = _ok(client, wid, "explain_provenance", entity_id="mystery")["text"]
    assert "Provenance: Unknown" in text
    assert "Evidence: Not recorded" in text and "Confidence: Not recorded" in text
    assert "Uncertainty note: Not recorded" in text
    assert "0.5" not in text  # the 0.5 placeholder default must not be presented as a confidence
    wall = _ok(client, wid, "explain_provenance", entity_id="wall-n")
    assert "RECONSTRUCTED" in wall["text"] and "Confidence: 0.8" in wall["text"]
    assert "Evidence: Not recorded" in wall["text"]  # no evidence ids on this fixture: say so, do not invent
    assert wall["evidence"] == {"observations": [], "external_references": [], "evidence_ids": []}


# ------------------------------------------------------------------------ propose / confirm


def test_assistant_propose_does_not_change_head_and_confirm_applies(client):
    wid = _proc_world(client)
    head = _head(client, wid)
    r = _call(client, wid, "propose_edit", **_op(op="update_generated", kind="room", name="A", patch={"width": 7}))
    assert r.status_code == 200, r.text
    plan = r.json()["result"]
    assert plan["requires_confirmation"] is True and plan["preview"]["entities_changed"] > 0
    assert plan["preview"]["entities_added"] == 0 and plan["preview"]["entities_removed"] == 0
    assert _head(client, wid) == head
    assert _ok(client, wid, "get_entity", entity_id="A-room")["spec"]["params"]["width"] == 4

    c = client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]})
    assert c.status_code == 200, c.text
    assert c.json()["version_id"] == _head(client, wid) != head
    assert _ok(client, wid, "get_entity", entity_id="A-room")["spec"]["params"]["width"] == 7
    # a confirmed plan cannot be replayed
    assert client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]}).status_code == 409


def test_assistant_stale_plan_is_409_and_stays_stale(client):
    wid = _proc_world(client)
    plan = _call(client, wid, "propose_edit", **_op(op="update_generated", kind="room", name="A", patch={"width": 7})).json()["result"]
    other = client.put(f"/api/worlds/{wid}/rooms/A", json={"depth": 6})  # HEAD moves under the plan
    assert other.status_code == 200, other.text
    moved = _head(client, wid)
    c = client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]})
    assert c.status_code == 409 and "changed" in c.text
    assert _head(client, wid) == moved
    again = client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]})
    assert again.status_code == 409 and "stale" in again.text


def test_assistant_delete_with_dependents_and_cascade(client):
    wid = _proc_world(client)
    head = _head(client, wid)
    r = _call(client, wid, "propose_edit", **_op(op="delete_generated", kind="room", name="A"))
    assert r.status_code == 422 and "dependents" in r.text and _head(client, wid) == head
    plan = _call(client, wid, "propose_edit", **_op(op="delete_generated", kind="room", name="A", cascade=True)).json()["result"]
    assert plan["preview"]["entities_removed"] > 0 and any("c" in w for w in plan["warnings"])
    assert _head(client, wid) == head
    assert client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]}).status_code == 200
    assert _ok(client, wid, "list_entities")["total"] == 0


@pytest.mark.parametrize("args", [
    _op(op="rename", kind="room", name="A"),
    _op(op="explode", kind="room", name="A"),
    _op(op="update_generated", kind="room", name="A", patch={"name": "B"}),      # rename via patch
    _op(op="update_generated", kind="room", name="A", patch={"width": -1}),     # grammar rejects
    _op(op="update_generated", kind="room", name="A", patch={}),                # empty
    _op(op="update_generated", kind="room", name="missing", patch={"width": 5}),
    _op(op="update_generated", kind="spaceship", name="A", patch={"width": 5}),
    {"operation": "delete everything"},
])
def test_assistant_invalid_operations_are_422_and_leave_head(client, args):
    wid = _proc_world(client)
    head = _head(client, wid)
    r = _call(client, wid, "propose_edit", **args)
    assert r.status_code == 422, r.text
    assert _head(client, wid) == head


def test_assistant_unknown_tool_and_bad_args_are_422(client):
    wid = _proc_world(client)
    assert _call(client, wid, "delete_world").status_code == 422
    assert _call(client, wid, "list_entities", limit=0).status_code == 422
    assert _call(client, wid, "list_entities", bogus=1).status_code == 422
    assert client.post("/api/worlds/nope/assistant/call", json={"tool": "search", "args": {"query": "x"}}).status_code == 404
    assert client.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": "plan_missing"}).status_code == 404
