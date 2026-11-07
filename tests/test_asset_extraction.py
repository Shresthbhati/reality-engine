"""Acceptance tests for Furniture & Asset Extraction slice (T-2.6).

Turnaround plan T-2.6:
Select furniture -> 'Extract as asset' -> independent copy (own ID + URL), downloadable.
Slice: data model -> API -> UI.
Done when: extracted asset keeps geometry, materials, dimensions, provenance; original untouched (test).
"""

from __future__ import annotations

import copy
import pytest
from apps.api.models import utcnow
from engine.assets.extraction import extract_entity_asset
from world_ir.schema_v1 import (
    ConfidenceVector,
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Provenance,
    Vector3,
)
from world_ir.world_v1 import WorldIR


def _sample_world_with_furniture() -> WorldIR:
    world = WorldIR(
        id="world-penthouse-01",
        name="Penthouse Suite",
        version=1,
        created_at=100.0,
        modified_at=100.0,
        metadata={"scale": {"state": "metric", "meters_per_unit": 1.0}},
    )

    # 1. Room
    world.entities["room-lounge"] = Entity(
        id="room-lounge",
        name="Living Lounge",
        type=EntityType.ROOM,
        custom_properties={"footprint_polygon_m": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]},
    )

    # 2. Material
    world.materials["mat-leather"] = Material(
        id="mat-leather",
        name="Cognac Leather",
        class_name="leather",
        color_rgb=(0.55, 0.27, 0.07),
    )

    # 3. Geometry (Box)
    world.geometries["geom-sofa-box"] = Geometry(
        id="geom-sofa-box",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(2.2, 0.95, 0.82),
    )

    # 4. Furniture Entity
    world.entities["furn-sofa-01"] = Entity(
        id="furn-sofa-01",
        name="Mid-Century Leather Sofa",
        type=EntityType.FURNITURE,
        transform={
            "position": [1.5, 2.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        geometry_ids=["geom-sofa-box"],
        material_ids=["mat-leather"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.94,
        confidence_vector=ConfidenceVector(geometry=0.92, material=0.95, classification=0.96),
        custom_properties={
            "dimensions": {
                "width_m": {"value": 2.2, "unit": "m", "state": "MEASURED"},
                "depth_m": {"value": 0.95, "unit": "m", "state": "MEASURED"},
                "height_m": {"value": 0.82, "unit": "m", "state": "MEASURED"},
            },
            "category": "seating",
        },
    )
    return world


def test_extract_furniture_preserves_geometry_materials_dimensions_provenance():
    world = _sample_world_with_furniture()
    original_world_dict = copy.deepcopy(world.to_dict())

    extracted, asset_world, glb_bytes = extract_entity_asset(
        world,
        "furn-sofa-01",
        name="Designer Leather Sofa",
        category="furniture",
    )

    # 1. Extracted asset record identity & fields
    assert extracted.id.startswith("ast-")
    assert extracted.name == "Designer Leather Sofa"
    assert extracted.category == "furniture"
    assert extracted.source_world_id == "world-penthouse-01"
    assert extracted.source_entity_id == "furn-sofa-01"

    # 2. Dimensions preserved accurately
    assert extracted.dimensions["width_m"] == pytest.approx(2.2, abs=1e-3)
    assert extracted.dimensions["depth_m"] == pytest.approx(0.95, abs=1e-3)
    assert extracted.dimensions["height_m"] == pytest.approx(0.82, abs=1e-3)

    # 3. Provenance preserved
    assert extracted.provenance["source_world_id"] == "world-penthouse-01"
    assert extracted.provenance["source_entity_id"] == "furn-sofa-01"
    assert extracted.provenance["confidence"] == pytest.approx(0.94, abs=1e-3)
    assert extracted.provenance["entity_provenance"] == Provenance.RECONSTRUCTED.value

    # 4. Geometry and materials preserved in extracted world
    assert "geom-sofa-box" in asset_world.geometries
    assert "mat-leather" in asset_world.materials
    assert len(asset_world.entities) == 1
    asset_ent = list(asset_world.entities.values())[0]
    assert asset_ent.geometry_ids == ["geom-sofa-box"]
    assert asset_ent.material_ids == ["mat-leather"]

    # 5. Asset transform centered at local origin
    assert asset_ent.transform["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert asset_ent.custom_properties["original_world_transform"]["position"] == [1.5, 2.0, 0.0]

    # 6. Valid downloadable GLB
    assert glb_bytes.startswith(b"glTF")
    assert len(glb_bytes) > 100

    # 7. CRITICAL: Original world was completely untouched!
    assert world.to_dict() == original_world_dict
    assert len(world.entities) == 2
    assert "furn-sofa-01" in world.entities
    assert world.entities["furn-sofa-01"].transform["position"] == [1.5, 2.0, 0.0]


def test_extract_nonexistent_entity_raises_keyerror():
    world = _sample_world_with_furniture()
    with pytest.raises(KeyError, match="not found"):
        extract_entity_asset(world, "furn-does-not-exist")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from pathlib import Path
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    db_mod._engine = None
    db_mod._sessionmaker = None
    import apps.api.main as main_mod

    from starlette.testclient import TestClient

    with TestClient(main_mod.app) as c:
        yield c



def test_api_extract_asset_and_download_roundtrip(client, tmp_path, monkeypatch):
    """Test full API slice: POST extract-asset -> GET asset -> GET download -> GET list."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))

    import apps.api.worldstore_service as ws_svc
    ws_svc._store = None

    # 1. Create a world via API
    res = client.post("/api/worlds", json={"name": "Loft Space"})
    assert res.status_code == 201
    wid = res.json()["id"]

    # Commit a world version with our sample furniture into WorldStore
    from apps.api.db import get_sessionmaker
    import asyncio
    from apps.api.models import World, WorldVersion

    async def _seed_world_version():
        w = _sample_world_with_furniture()
        w.id = wid
        sm = get_sessionmaker()
        async with sm() as db:
            await ws_svc.commit_version(db, world_id=wid, world=w, parent=None)

    asyncio.run(_seed_world_version())

    # 2. Extract asset via API endpoint
    r = client.post(
        f"/api/worlds/{wid}/extract-asset",
        json={"entity_id": "furn-sofa-01", "name": "Studio Leather Sofa"},
    )
    assert r.status_code == 201, r.text
    data = r.json()

    assert data["id"].startswith("ast-")
    assert data["name"] == "Studio Leather Sofa"
    assert data["category"] == "furniture"
    assert data["source_world_id"] == wid
    assert data["source_entity_id"] == "furn-sofa-01"
    assert data["dimensions"]["width_m"] == pytest.approx(2.2, abs=1e-3)
    assert "download_url" in data
    asset_id = data["id"]

    # 3. GET /api/assets/{id}
    r_get = client.get(f"/api/assets/{asset_id}")
    assert r_get.status_code == 200
    asset_data = r_get.json()
    assert asset_data["id"] == asset_id
    assert asset_data["name"] == "Studio Leather Sofa"

    # 4. GET /api/assets/{id}/download
    r_dl = client.get(f"/api/assets/{asset_id}/download")
    assert r_dl.status_code == 200
    assert r_dl.headers["content-type"] in ("model/gltf-binary", "application/octet-stream")
    assert r_dl.content.startswith(b"glTF")
    assert len(r_dl.content) > 100

    # 5. GET /api/assets
    r_list = client.get(f"/api/assets?world_id={wid}")
    assert r_list.status_code == 200
    items = r_list.json()["items"]
    assert any(item["id"] == asset_id for item in items)

    # 6. Verify original world is untouched
    r_world = client.get(f"/api/worlds/{wid}")
    assert r_world.status_code == 200
    assert r_world.json()["name"] == "Loft Space"


def test_api_extract_asset_404_handling(client):
    # Non-existent world
    r = client.post("/api/worlds/wld-nonexistent/extract-asset", json={"entity_id": "any"})
    assert r.status_code == 404

    # Non-existent asset
    r = client.get("/api/assets/ast-nonexistent")
    assert r.status_code == 404

    # Non-existent asset download
    r = client.get("/api/assets/ast-nonexistent/download")
    assert r.status_code == 404
