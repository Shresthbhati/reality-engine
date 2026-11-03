"""Tests for Spatial Intelligence API endpoint: /api/worlds/{id}/spatial-intelligence."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient
from world_ir.coordinates import Frame
from world_ir.schema_v1 import Entity, EntityType, Provenance
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


def test_spatial_intelligence_endpoint(client, tmp_path):
    ws_root = tmp_path / "worldstore"
    ws_root.mkdir(parents=True, exist_ok=True)
    store = WorldStore(str(ws_root))

    # 1. Create a world
    res = client.post("/api/worlds", json={"name": "Spatial Intelligence Test World"})
    assert res.status_code == 201
    wid = res.json()["id"]

    # 2. Build WorldIR with spatial intelligence metadata
    world = WorldIR(id=wid, name="Spatial Intelligence Test World")
    world.coordinate_frame = Frame.ENU
    world.metadata["georeference"] = {
        "status": "georeferenced",
        "frame": "ENU",
        "origin": {"latitude_deg": 37.7749, "longitude_deg": -122.4194, "altitude_m": 15.0},
    }
    world.metadata["design_style"] = {
        "primary_style": "MODERN_MINIMALIST",
        "confidence": 0.92,
        "ornamentation": "NONE_MINIMAL",
    }
    world.metadata["building_envelope"] = {
        "total_storeys": 2,
        "gross_floor_area_m2": 240.5,
        "building_height_m": 6.2,
    }
    world.environment["lighting"] = {
        "dominant_light_direction": [0.0, 0.0, 1.0],
        "ambient_irradiance_rgb": [0.4, 0.4, 0.4],
    }

    # Add entities
    room = Entity(
        id="room-101",
        name="Living Room",
        type=EntityType.ROOM,
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.98,
        custom_properties={"elevation_m": 0.0},
    )
    wall = Entity(
        id="wall-01",
        name="Exterior Wall",
        type=EntityType.WALL,
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
        custom_properties={"is_exterior": True, "thickness_m": 0.25},
    )
    world.entities[room.id] = room
    world.entities[wall.id] = wall

    # Save to store
    v = store.save_version(world, parent=None)

    # Adopt version in DB
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
                    parent_version_id=None,
                )
            )
            await session.commit()

    asyncio.run(_adopt())

    # Call endpoint without version query param (should query current_version_id)
    res = client.get(f"/api/worlds/{wid}/spatial-intelligence")
    assert res.status_code == 200
    data = res.json()

    assert data["world_id"] == wid
    assert data["version_id"] == v.version_id
    assert data["coordinate_frame"] == "ENU"
    assert data["georeference"]["status"] == "georeferenced"
    assert data["design_style"]["primary_style"] == "MODERN_MINIMALIST"
    assert data["building_envelope"]["total_storeys"] == 2
    assert "scene_completeness" in data
    assert "physical_plausibility" in data
    assert "lighting" in data
    assert data["lighting"]["dominant_light_direction"] == [0.0, 0.0, 1.0]
