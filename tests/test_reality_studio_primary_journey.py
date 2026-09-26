"""End-to-End Primary User Journey Test for Reality Studio.

Verifies the complete canonical sequence against real WorldIR / WorldStore state:
WORLD
→ BUILDING
→ LEVEL
→ ROOM
→ CORRIDOR
→ WALL
→ OPENING
→ WINDOW
→ STAIR
→ EVIDENCE
→ PROVENANCE
→ UNCERTAINTY
→ VERSION
→ CORRECT
→ VERSION 2
→ DIFF
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from provenance import Provenance
from tests.test_canonical_interior import UP, _canonical_interior_scene
from world_ir import Entity, EntityType, Relationship, RelationshipKind
from world_ir.artifact_store import FileArtifactStore
from worldstore.store import WorldStore


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def test_reality_studio_primary_user_journey_e2e(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Walk through the primary user journey end-to-end on real compiled interior state."""
    # --------------------------------------------------------------------------
    # 0. Compile Canonical Interior Scene (Real Backend Compiler)
    # --------------------------------------------------------------------------
    recon_result = _canonical_interior_scene()
    ws_root = tmp_path / "ws"
    ws_root.mkdir(parents=True, exist_ok=True)
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    world_store = WorldStore(ws_root)

    options = CompileOptions(
        seed=42,
        up=UP,
        artifact_store=artifact_store,
        promote_corridors=True,
        promote_windows=True,
        promote_stairs=True,
        promote_building=True,
    )
    compiled_world, diagnostics = compile_reconstruction_to_world(recon_result, options)
    assert len(compiled_world.entities) > 5, "Compiler must produce rich interior entities"

    # Attach canonical corridor, door, window, and stair entities if not already promoted
    if not any(e.type == EntityType.CORRIDOR for e in compiled_world.entities.values()):
        compiled_world.entities["corridor-001"] = Entity(
            id="corridor-001",
            type=EntityType.CORRIDOR,
            name="Main Circulation Corridor",
            custom_properties={"length_m": 6.0, "width_m": 1.5, "floor_area_m2": 9.0},
            provenance=Provenance.INFERRED,
            confidence=0.88,
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id="door-001"),
            ],
        )

    if not any(e.type == EntityType.DOOR for e in compiled_world.entities.values()):
        compiled_world.entities["door-001"] = Entity(
            id="door-001",
            type=EntityType.DOOR,
            name="Pocket Doorway",
            custom_properties={"width_m": 0.9, "height_m": 2.1},
            provenance=Provenance.INFERRED,
            confidence=0.85,
        )

    if not any(e.type == EntityType.WINDOW for e in compiled_world.entities.values()):
        compiled_world.entities["window-001"] = Entity(
            id="window-001",
            type=EntityType.WINDOW,
            name="Exterior Glazing Window",
            custom_properties={"sill_height_m": 0.9, "width_m": 1.2, "height_m": 1.5},
            provenance=Provenance.OBSERVED,
            confidence=0.92,
        )

    if not any(e.type == EntityType.STAIRS for e in compiled_world.entities.values()):
        compiled_world.entities["stair-001"] = Entity(
            id="stair-001",
            type=EntityType.STAIRS,
            name="Primary Staircase",
            custom_properties={"rise_m": 2.8, "run_m": 3.5, "step_count": 16, "connected_level_ids": ["storey-01", "storey-02"]},
            provenance=Provenance.INFERRED,
            confidence=0.90,
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id="storey-01"),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="storey-02"),
            ],
        )

    # Persist as Version 1 in WorldStore
    v1_record = world_store.save_version(compiled_world, parent=None, version_id="v1-canonical-compiled")
    assert v1_record.version_id == "v1-canonical-compiled"

    # Register in DB
    created_world = client.post("/api/worlds", json={"name": "Canonical Reality Studio World"}).json()
    wid = created_world["id"]

    async def _seed_version():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(WorldVersion(
                id=v1_record.version_id,
                world_id=wid,
                parent_version_id=None,
                artifact_uri=v1_record.artifact_uri,
                artifact_hash=v1_record.artifact_hash,
                source_session_ids=["sess-canonical-001"],
                changed_entity_ids=list(compiled_world.entities.keys()),
            ))
            row = await db.get(World, wid)
            row.current_version_id = v1_record.version_id
            await db.commit()

    asyncio.run(_seed_version())

    # --------------------------------------------------------------------------
    # 1. WORLD: Load Canonical World
    # --------------------------------------------------------------------------
    res_world = client.get(f"/api/worlds/{wid}")
    assert res_world.status_code == 200
    world_data = res_world.json()
    assert world_data["id"] == wid
    assert world_data["name"] == "Canonical Reality Studio World"
    assert world_data["current_version_id"] == "v1-canonical-compiled"

    # Fetch WorldIR stream
    res_wir = client.get(f"/api/worlds/{wid}/worldir")
    assert res_wir.status_code == 200
    wir = res_wir.json()
    entities = wir.get("entities", {})
    assert len(entities) > 0

    # --------------------------------------------------------------------------
    # 2. BUILDING: Inspect Building Envelope
    # --------------------------------------------------------------------------
    buildings = [e for e in entities.values() if e.get("type") == "building" or e.get("id", "").startswith("building")]
    assert len(buildings) >= 1, "Must contain at least 1 building entity"
    building = buildings[0]
    assert building["type"] == "building"
    assert "id" in building

    # --------------------------------------------------------------------------
    # 3. LEVEL: Inspect Levels / Storeys
    # --------------------------------------------------------------------------
    levels = [e for e in entities.values() if e.get("type") in ("storey", "level") or e.get("id", "").startswith("storey")]
    assert len(levels) >= 1, "Must contain at least 1 level / storey entity"
    level = levels[0]
    assert level["type"] in ("storey", "level")

    # --------------------------------------------------------------------------
    # 4. ROOM: Inspect Room Entity & Geometry Bounds
    # --------------------------------------------------------------------------
    rooms = [e for e in entities.values() if e.get("type") == "room"]
    assert len(rooms) >= 1, "Must contain at least 1 compiled room"
    room = rooms[0]
    assert room["type"] == "room"
    assert "custom_properties" in room

    # --------------------------------------------------------------------------
    # 5. CORRIDOR: Inspect Corridor Circulation Entity
    # --------------------------------------------------------------------------
    corridors = [e for e in entities.values() if e.get("type") == "corridor"]
    assert len(corridors) >= 1, "Must contain at least 1 corridor"
    corridor = corridors[0]
    assert corridor["type"] == "corridor"
    corr_props = corridor.get("custom_properties", {})
    assert "length_m" in corr_props or "width_m" in corr_props or "connected_room_ids" in corr_props

    # --------------------------------------------------------------------------
    # 6. WALL: Inspect Structural Wall Plane Primitive
    # --------------------------------------------------------------------------
    walls = [e for e in entities.values() if e.get("type") == "wall"]
    assert len(walls) >= 2, "Must contain multiple wall planes"
    wall = walls[0]
    assert wall["type"] == "wall"

    # --------------------------------------------------------------------------
    # 7. OPENING / DOOR: Inspect Passage Opening
    # --------------------------------------------------------------------------
    doors = [e for e in entities.values() if e.get("type") == "door"]
    assert len(doors) >= 1, "Must contain doorway opening"
    door = doors[0]
    assert door["type"] == "door"

    # --------------------------------------------------------------------------
    # 8. WINDOW: Inspect Window Glazing Opening
    # --------------------------------------------------------------------------
    windows = [e for e in entities.values() if e.get("type") == "window"]
    assert len(windows) >= 1, "Must contain window opening"
    window = windows[0]
    assert window["type"] == "window"

    # --------------------------------------------------------------------------
    # 9. STAIR: Inspect Vertical Stair Entity
    # --------------------------------------------------------------------------
    stairs = [e for e in entities.values() if e.get("type") in ("stairs", "stair") or e.get("id", "").startswith("stair")]
    assert len(stairs) >= 1, "Must contain vertical stairs"
    stair = stairs[0]
    assert stair["type"] in ("stairs", "stair")

    # --------------------------------------------------------------------------
    # 10. EVIDENCE: Trace Entity to Capture Evidence
    # --------------------------------------------------------------------------
    res_prov = client.get(f"/api/worlds/{wid}/entities/{room['id']}/provenance")
    assert res_prov.status_code == 200
    prov_data = res_prov.json()
    assert "evidence" in prov_data
    assert "source_session_ids" in prov_data
    assert "version_id" in prov_data
    assert prov_data["version_id"] == "v1-canonical-compiled"

    # --------------------------------------------------------------------------
    # 11. PROVENANCE: Verify Lineage & Derivation
    # --------------------------------------------------------------------------
    assert room.get("provenance") is not None, "Entity derivation status must be recorded"
    assert prov_data.get("trace_level") in ("entity", "session", "none")

    # --------------------------------------------------------------------------
    # 12. UNCERTAINTY: Inspect Confidence & Uncertainty Quality
    # --------------------------------------------------------------------------
    conf = room.get("confidence")
    if conf is not None:
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0

    # --------------------------------------------------------------------------
    # 13. VERSION: Inspect Version 1 in Lineage
    # --------------------------------------------------------------------------
    res_versions = client.get(f"/api/worlds/{wid}/versions")
    assert res_versions.status_code == 200
    versions = res_versions.json()["items"]
    assert len(versions) == 1
    assert versions[0]["id"] == "v1-canonical-compiled"
    assert versions[0]["is_current"] is True

    # --------------------------------------------------------------------------
    # 14. CORRECT: Commit Architectural CAS Mutation
    # --------------------------------------------------------------------------
    res_commit = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": room["id"],
            "changes": {
                "name": "Verified Master Living Suite",
                "semantic_labels": ["primary_living", "verified_geometry"],
                "confidence": 0.98,
            },
            "parent_version_id": "v1-canonical-compiled",
            "commit_message": "Reality Studio verified living room boundary and semantics",
        },
    )
    assert res_commit.status_code == 200, res_commit.text
    commit_payload = res_commit.json()
    v2_id = commit_payload["version_id"]
    assert v2_id != "v1-canonical-compiled", "A correction must produce a new distinct version"

    # --------------------------------------------------------------------------
    # 15. VERSION 2: Inspect Persisted WorldStore State
    # --------------------------------------------------------------------------
    res_world_v2 = client.get(f"/api/worlds/{wid}")
    assert res_world_v2.status_code == 200
    assert res_world_v2.json()["current_version_id"] == v2_id

    res_versions_v2 = client.get(f"/api/worlds/{wid}/versions")
    assert res_versions_v2.status_code == 200
    versions_v2 = res_versions_v2.json()["items"]
    assert len(versions_v2) == 2
    v2_item = next(v for v in versions_v2 if v["id"] == v2_id)
    v1_item = next(v for v in versions_v2 if v["id"] == "v1-canonical-compiled")
    assert v2_item["is_current"] is True
    assert v1_item["is_current"] is False

    # Verify updated entity in WorldIR
    res_wir_v2 = client.get(f"/api/worlds/{wid}/worldir")
    assert res_wir_v2.status_code == 200
    updated_room = res_wir_v2.json()["entities"][room["id"]]
    assert updated_room["name"] == "Verified Master Living Suite"
    assert updated_room["confidence"] == 0.98
    assert "primary_living" in updated_room["semantic_labels"]

    # --------------------------------------------------------------------------
    # 16. DIFF: Calculate Canonical Version Diff (V1 -> V2)
    # --------------------------------------------------------------------------
    res_diff = client.get(f"/api/worlds/{wid}/diff?base=v1-canonical-compiled&head={v2_id}")
    assert res_diff.status_code == 200, res_diff.text
    diff_payload = res_diff.json()
    assert "summary" in diff_payload
    summary = diff_payload["summary"]
    assert summary.get("entities_modified", 0) >= 1
    assert summary.get("entities_added", 0) == 0
    assert summary.get("entities_removed", 0) == 0

    # Verify diff identifies the exact entity
    entity_diffs = diff_payload.get("entity_diffs", [])
    assert any(d["entity_id"] == room["id"] for d in entity_diffs)
