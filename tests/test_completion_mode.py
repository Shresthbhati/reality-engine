"""Completion mode (Phase D): GENERATED interior proposals over a reconstructed shell."""

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


def _shell(width=6.0, depth=5.0, prov=Provenance.RECONSTRUCTED) -> WorldIR:
    wir = WorldIR(id="x")
    box = {"floor": ((0, 0, 0), (width, 0.1, depth)), "wall-n": ((0, 0, 0), (width, 2.6, 0.2))}
    for name, (lo, hi) in box.items():
        gid = f"g-{name}"
        wir.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo),
                                       bounds_max=Vector3(*hi), provenance=prov)
        wir.entities[name] = Entity(id=name, type=EntityType.FLOOR if name == "floor" else EntityType.WALL,
                                    name=name, geometry_ids=[gid], provenance=prov, confidence=0.8)
    return wir


def _seed(client, wir: WorldIR | None) -> str:
    wid = client.post("/api/worlds", json={"name": "Shell"}).json()["id"]
    if wir is not None:
        from apps.api import worldstore_service as svc
        import apps.api.db as db_mod

        async def go():
            async with db_mod.get_sessionmaker()() as db:
                await svc.commit_version(db, world_id=wid, world=wir, parent=None)

        asyncio.run(go())
    return wid


def _head(client, wid) -> str:
    return client.get(f"/api/worlds/{wid}").json()["current_version_id"]


def _propose(client, wid, **kw):
    return client.post(f"/api/worlds/{wid}/completion/proposals", json={"template": "studio", **kw})


def test_proposal_does_not_change_head(client):
    wid = _seed(client, _shell())
    before = _head(client, wid)
    r = _propose(client, wid, alternatives=2)
    assert r.status_code == 201, r.text
    assert _head(client, wid) == before
    props = r.json()["proposals"]
    assert 1 <= len(props) <= 2
    assert props[0]["summary"]["unit"] == "model_units"
    assert props[0]["summary"]["footprint"]["method"] == "axis_aligned_bbox"
    assert len(client.get(f"/api/worlds/{wid}/completion/proposals").json()) == len(props)


def test_diff_counts_and_never_observed(client):
    wid = _seed(client, _shell())
    pid = _propose(client, wid).json()["proposals"][0]["id"]
    d = client.get(f"/api/worlds/{wid}/completion/proposals/{pid}/diff").json()
    assert d["counts"]["added"] == len(d["added"]) > 0
    assert d["counts"]["changed"] == 0 and d["counts"]["removed"] == 0
    assert d["provenance"]["existing"] == {"RECONSTRUCTED": 2}
    assert set(d["provenance"]["added"]) == {"GENERATED"}
    assert "floor" not in d["added"]


def test_accept_adds_generated_entities_and_stores_spec(client):
    wid = _seed(client, _shell())
    before = _head(client, wid)
    pid = _propose(client, wid).json()["proposals"][0]["id"]
    added = client.get(f"/api/worlds/{wid}/completion/proposals/{pid}/diff").json()["added"]
    r = client.post(f"/api/worlds/{wid}/completion/proposals/{pid}/accept")
    assert r.status_code == 200, r.text
    assert _head(client, wid) != before
    assert sorted(r.json()["entity_ids_added"]) == added
    gen = client.get(f"/api/worlds/{wid}/generated").json()
    assert [g["kind"] for g in gen] == ["apartment"]
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    async def load():
        async with db_mod.get_sessionmaker()() as db:
            return await svc.get_current_worldir(db, wid)

    wir = asyncio.run(load())
    new = [wir.entities[i] for i in added]
    assert all(e.provenance is Provenance.GENERATED for e in new)
    assert all(e.custom_properties.get("completion_proposal") == pid for e in new)
    assert wir.entities["floor"].provenance is Provenance.RECONSTRUCTED
    assert client.get(f"/api/worlds/{wid}/completion/proposals").json()[0]["status"] == "accepted"
    assert client.post(f"/api/worlds/{wid}/completion/proposals/{pid}/accept").status_code == 409


def test_reject_leaves_world_identical(client):
    wid = _seed(client, _shell())
    before = _head(client, wid)
    pid = _propose(client, wid).json()["proposals"][0]["id"]
    r = client.post(f"/api/worlds/{wid}/completion/proposals/{pid}/reject")
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert _head(client, wid) == before
    assert client.get(f"/api/worlds/{wid}/generated").json() == []
    assert client.post(f"/api/worlds/{wid}/completion/proposals/{pid}/accept").status_code == 409


def test_deterministic_by_seed(client):
    a = _seed(client, _shell())
    b = _seed(client, _shell())
    pa = _propose(client, a, seed=7).json()["proposals"]
    pb = _propose(client, b, seed=7).json()["proposals"]
    from apps.api.models import CompletionProposal  # noqa: F401  (table is part of the metadata)

    def specs(wid):
        return client.get(f"/api/worlds/{wid}/completion/proposals").json()

    assert [p["summary"] for p in specs(a)] == [p["summary"] for p in specs(b)]
    rects = lambda w, p: client.get(f"/api/worlds/{w}/completion/proposals/{p}/diff").json()["added"]  # noqa: E731
    assert [rects(a, p["id"]) for p in pa] == [rects(b, p["id"]) for p in pb]


@pytest.mark.parametrize("wir", [None, WorldIR(id="x"), _shell(prov=Provenance.GENERATED), _shell(1.0, 1.0)])
def test_422_without_usable_footprint(client, wir):
    wid = _seed(client, wir)
    r = _propose(client, wid)
    assert r.status_code == 422
    assert "shell" in r.text or "footprint" in r.text or "committed" in r.text


def test_404_and_stale_shell_409(client):
    assert _propose(client, "wld_nope").status_code == 404
    wid = _seed(client, _shell())
    pid = _propose(client, wid).json()["proposals"][0]["id"]
    # shell grows after the proposal was made -> proposal is stale
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    async def grow():
        async with db_mod.get_sessionmaker()() as db:
            cur = await svc.get_current_worldir(db, wid)
            cur.geometries["g-floor"].bounds_max = Vector3(9.0, 0.1, 5.0)
            from apps.api.models import World
            w = await db.get(World, wid)
            await svc.commit_version(db, world_id=wid, world=cur, parent=w.current_version_id,
                                     expect_parent=w.current_version_id)

    asyncio.run(grow())
    assert client.post(f"/api/worlds/{wid}/completion/proposals/{pid}/accept").status_code == 409


def test_proposals_carry_non_overlapping_rects_within_footprint(client):
    wid = _seed(client, _shell())
    props = _propose(client, wid, alternatives=3).json()["proposals"]
    assert props
    for p in props:
        plan = p["plan"]
        assert plan["provenance"] == "GENERATED"
        fw, fd = plan["footprint"]
        assert fw <= 6.0 and fd <= 5.0
        rects = plan["rects"]
        assert set(rects) == set(plan["kinds"]) and len(rects) >= 3
        eps = 1e-6
        for x, y, w, d in rects.values():
            assert w > 0 and d > 0 and x >= -eps and y >= -eps and x + w <= fw + eps and y + d <= fd + eps
        items = list(rects.values())
        for i, (ax, ay, aw, ad) in enumerate(items):
            for bx, by, bw, bd in items[i + 1:]:
                assert ax + aw <= bx + eps or bx + bw <= ax + eps or ay + ad <= by + eps or by + bd <= ay + eps
    listed = client.get(f"/api/worlds/{wid}/completion/proposals").json()
    assert [x["plan"] for x in listed] == [p["plan"] for p in props]


def test_plan_rects_deterministic_by_seed(client):
    a, b = _seed(client, _shell()), _seed(client, _shell())
    pa = _propose(client, a, seed=11, alternatives=2).json()["proposals"]
    pb = _propose(client, b, seed=11, alternatives=2).json()["proposals"]
    assert [p["plan"] for p in pa] == [p["plan"] for p in pb]
