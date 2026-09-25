"""Tests for Space Graph API endpoints: /api/worlds/{id}/space-graph and /query/spaces."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    import apps.api.db as db_mod
    importlib.reload(db_mod)

    import apps.api.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def test_space_graph_api_endpoints(client, tmp_path):
    ws_root = tmp_path / "worldstore"
    ws_root.mkdir(parents=True, exist_ok=True)
    store = WorldStore(str(ws_root))

    # 1. Create a world
    res = client.post("/api/worlds", json={"name": "Space Graph World"})
    assert res.status_code == 201
    wid = res.json()["id"]

    # 2. Build WorldIR with an InteriorSpaceGraph
    space_graph_payload = {
        "building_id": wid,
        "levels": [
            {
                "level_id": "lvl-0",
                "elevation_m": 0.0,
                "room_ids": ["room-101", "room-102"],
                "corridor_ids": ["corr-01"],
                "stair_ids": ["stair-01"],
            },
            {
                "level_id": "lvl-1",
                "elevation_m": 3.0,
                "room_ids": ["room-201"],
                "corridor_ids": [],
                "stair_ids": ["stair-01"],
            },
        ],
        "rooms": [
            {
                "room_id": "room-101",
                "floor_area_m2": 15.0,
                "adjacent_room_ids": ["room-102", "corr-01"],
                "corridor_ids": ["corr-01"],
                "boundary_completeness": 1.0,
                "status": "detected",
            },
            {
                "room_id": "room-102",
                "floor_area_m2": 18.0,
                "adjacent_room_ids": ["room-101", "corr-01"],
                "corridor_ids": ["corr-01"],
                "boundary_completeness": 1.0,
                "status": "detected",
            },
            {
                "room_id": "room-201",
                "floor_area_m2": 24.0,
                "adjacent_room_ids": [],
                "corridor_ids": [],
                "boundary_completeness": 1.0,
                "status": "detected",
            },
        ],
        "corridors": [
            {
                "corridor_id": "corr-01",
                "aspect_ratio": 3.0,
                "width_m": 1.2,
                "length_m": 3.6,
                "height_m": 2.4,
                "floor_area_m2": 4.32,
                "connected_room_ids": ["room-101", "room-102"],
                "longitudinal_axis": [1.0, 0.0, 0.0],
                "bounds_min": [0.0, 0.0, 0.0],
                "bounds_max": [3.6, 1.2, 2.4],
            }
        ],
        "openings": [
            {
                "opening_id": "op-01",
                "kind": "doorway",
                "width_m": 0.9,
                "height_m": 2.1,
                "sill_height_m": 0.0,
                "source_space_id": "room-101",
                "connected_space_ids": ["room-102"],
            }
        ],
        "stairs": [
            {
                "stair_id": "stair-01",
                "step_count": 15,
                "total_rise_m": 3.0,
                "total_run_m": 3.75,
                "connected_level_ids": ["lvl-0", "lvl-1"],
            }
        ],
        "summary": {
            "level_count": 2,
            "room_count": 3,
            "corridor_count": 1,
            "opening_count": 1,
            "stair_count": 1,
        },
    }

    worldir = WorldIR(id=wid, name="Space Graph World")
    worldir.metadata["interior_space_graph"] = space_graph_payload
    v = store.save_version(worldir, parent=None)

    # Adopt version in DB
    import asyncio
    import apps.api.db as db_mod
    from apps.api.models import World, WorldVersion

    async def _adopt():
        maker = db_mod.get_sessionmaker()
        async with maker() as session:
            w = await session.get(World, wid)
            w.current_version_id = v.version_id
            session.add(
                WorldVersion(
                    id=v.version_id,
                    world_id=wid,
                    artifact_uri=v.artifact_uri,
                    artifact_hash=v.artifact_hash,
                )
            )
            await session.commit()

    asyncio.run(_adopt())

    # 3. Test GET /api/worlds/{wid}/space-graph
    sg_res = client.get(f"/api/worlds/{wid}/space-graph")
    assert sg_res.status_code == 200
    data = sg_res.json()
    assert data["building_id"] == wid
    assert len(data["levels"]) == 2
    assert len(data["corridors"]) == 1
    assert data["corridors"][0]["corridor_id"] == "corr-01"

    # 4. Test corridor reachability query
    q_corr = client.get(f"/api/worlds/{wid}/query/spaces?corridor_id=corr-01")
    assert q_corr.status_code == 200
    q_corr_data = q_corr.json()
    assert q_corr_data["query"] == "corridor_reachability"
    assert q_corr_data["reachable_rooms"] == ["room-101", "room-102"]

    # 5. Test room adjacency query
    q_room = client.get(f"/api/worlds/{wid}/query/spaces?room_id=room-101")
    assert q_room.status_code == 200
    q_room_data = q_room.json()
    assert q_room_data["query"] == "connected_rooms"
    assert "room-102" in q_room_data["connected_spaces"]
    assert "corr-01" in q_room_data["connected_spaces"]

    # 6. Test opening connectivity query
    q_op = client.get(f"/api/worlds/{wid}/query/spaces?space_a=room-101&space_b=room-102")
    assert q_op.status_code == 200
    q_op_data = q_op.json()
    assert len(q_op_data["openings"]) == 1
    assert q_op_data["openings"][0]["opening_id"] == "op-01"

    # 7. Test stair connectivity query
    q_st = client.get(f"/api/worlds/{wid}/query/spaces?level_a=lvl-0&level_b=lvl-1")
    assert q_st.status_code == 200
    q_st_data = q_st.json()
    assert len(q_st_data["stairs"]) == 1
    assert q_st_data["stairs"][0]["stair_id"] == "stair-01"
