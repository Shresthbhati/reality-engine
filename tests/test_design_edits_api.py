"""T-4b.1 through the real API: each design edit is one committed version, the survey is untouched, Undo (T-6.7) reverses it."""
from __future__ import annotations

import asyncio
import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient  # noqa: E402

from provenance import Provenance  # noqa: E402
from world_ir import Entity, EntityType  # noqa: E402
from world_ir.schema_v1 import Geometry, GeometryType, Vector3  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402


def _dim(v, scale: bool = True):
    return {"value": v, "unit": "m" if scale else "model_units", "state": "ESTIMATED" if scale else "RECONSTRUCTED", "basis": "test"}


def _room(scale: bool = True) -> WorldIR:
    w = WorldIR(id="x")
    if scale:
        w.metadata["scale"] = {"state": "metric"}
    w.geometries["g-wall"] = Geometry(id="g-wall", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(5, 2.6, 0.2),
                                      provenance=Provenance.RECONSTRUCTED)
    w.entities["wall-n"] = Entity(id="wall-n", type=EntityType.WALL, name="wall", geometry_ids=["g-wall"],
                                  provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    w.geometries["g-sofa"] = Geometry(id="g-sofa", type=GeometryType.BOX, bounds_min=Vector3(1, 0, 1), bounds_max=Vector3(3, 0.85, 2),
                                      provenance=Provenance.ESTIMATED)
    w.entities["sofa"] = Entity(
        id="sofa", type=EntityType.FURNITURE, name="Sofa", transform={"position": {"x": 2.0, "y": 0.0, "z": 1.5}},
        geometry_ids=["g-sofa"], semantic_labels=["furniture_candidate"], provenance=Provenance.INFERRED, confidence=0.6,
        custom_properties={"category": "sofa", "entity_class": "furniture.sofa", "yaw_rad": 0.0,
                           "dimensions": {"length_m": _dim(2.0, scale), "width_m": _dim(1.0, scale), "height_m": _dim(0.85, scale)}})
    return w


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_EMBEDDED_WORKER", "0")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None

    def make(scale: bool = True) -> str:
        wid = c.post("/api/worlds", json={"name": "Flat"}).json()["id"]

        async def seed():
            async with db_mod.get_sessionmaker()() as db:
                wir = _room(scale)
                wir.id = wid
                await ws.commit_version(db, world_id=wid, world=wir, parent=None)

        asyncio.run(seed())
        return wid

    with TestClient(main_mod.app) as c:
        yield c, make


def _entities(c, wid):
    return c.get(f"/api/worlds/{wid}/worldir").json()["entities"]


def test_move_rotate_replace_delete_each_make_one_version_and_leave_the_survey_alone(studio):
    c, make = studio
    wid = make()
    wall_before = _entities(c, wid)["wall-n"]
    base = f"/api/worlds/{wid}/furniture/sofa"

    r = c.post(f"{base}/move", json={"x": 3.0, "z": 2.0})
    assert r.status_code == 200, r.text
    assert r.json()["operation"] == "design_move"
    assert _entities(c, wid)["sofa"]["transform"]["position"] == {"x": 3.0, "y": 0.0, "z": 2.0}

    assert c.post(f"{base}/rotate", json={"yaw_rad": 1.0}).status_code == 200
    assert _entities(c, wid)["sofa"]["custom_properties"]["yaw_rad"] == pytest.approx(1.0)

    r = c.post(f"{base}/replace", json={"catalog_id": "armchair"})
    assert r.status_code == 200, r.text
    sofa = _entities(c, wid)["sofa"]
    assert sofa["name"] == "Armchair" and sofa["custom_properties"]["category"] == "armchair"

    assert c.post(f"{base}/delete", json={}).status_code == 200
    ents = _entities(c, wid)
    assert "sofa" not in ents
    assert ents["wall-n"] == wall_before, "the as-built survey is never touched by a design edit"
    assert len(c.get(f"/api/worlds/{wid}/versions").json()["items"]) == 5          # the survey + 4 edits


def test_undo_reverses_a_design_edit_and_redo_restores_it(studio):
    c, make = studio
    wid = make()
    assert c.post(f"/api/worlds/{wid}/furniture/sofa/move", json={"x": 4.0, "z": 3.0}).status_code == 200
    assert c.post(f"/api/worlds/{wid}/furniture/sofa/delete", json={}).status_code == 200
    assert "sofa" not in _entities(c, wid)
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert _entities(c, wid)["sofa"]["transform"]["position"]["x"] == 4.0               # the deleted sofa is back, where it was moved to
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert _entities(c, wid)["sofa"]["transform"]["position"]["x"] == 2.0
    assert c.post(f"/api/worlds/{wid}/redo").status_code == 200
    assert _entities(c, wid)["sofa"]["transform"]["position"]["x"] == 4.0


def test_refusals_are_422_with_a_reason_and_create_no_version(studio):
    c, make = studio
    wid = make()
    n = len(c.get(f"/api/worlds/{wid}/versions").json()["items"])
    for path, body, code in (("furniture/wall-n/move", {"x": 1, "z": 1}, 422), ("furniture/sofa/replace", {"catalog_id": "nope"}, 422),
                             ("furniture/ghost/delete", {}, 404)):
        r = c.post(f"/api/worlds/{wid}/{path}", json=body)
        assert r.status_code == code, (path, r.text)
    assert len(c.get(f"/api/worlds/{wid}/versions").json()["items"]) == n


def test_replace_needs_a_real_scale_but_move_does_not(studio):
    c, make = studio
    wid = make(scale=False)
    r = c.post(f"/api/worlds/{wid}/furniture/sofa/replace", json={"catalog_id": "armchair"})
    assert r.status_code == 422 and "Set scale" in r.text
    assert c.post(f"/api/worlds/{wid}/furniture/sofa/move", json={"x": 1.0, "z": 1.0}).status_code == 200


def test_the_catalog_endpoint_lists_nominal_items(studio):
    c, _ = studio
    body = c.get("/api/furniture-catalog").json()
    assert {i["id"] for i in body["items"]} >= {"sofa-3-seat", "armchair", "bed-double"} and "not measurements" in body["note"]
