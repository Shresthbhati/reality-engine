"""T-4b.1 & T-6.7: Furniture Catalog, Move/Rotate/Delete/Replace Commands, and Persistent Undo/Redo.

Verification contracts:
1. T-4b.1: Small built-in catalog of CC0 3D furniture models:
   - Query catalog (optionally by category), get item details, download valid binary glTF (GLB).
   - Move, rotate, delete, or replace furniture with an item from the CC0 catalog.
   - All mutations advance HEAD and are stored as commands with provenance "generated".
2. T-6.7: Persistent per-world undo/redo:
   - Reload test: commands persist across sessions/reloads; undo works after reload.
   - Cross-world isolation: undo in World A never affects World B.
   - Redo re-applies reverted commands accurately.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401


def test_cc0_furniture_catalog_endpoints(client: TestClient):  # noqa: F811
    """T-4b.1: CC0 furniture catalog inspection and GLB export."""
    # 1. List all catalog items
    res = client.get("/api/furniture/catalog")
    assert res.status_code == 200
    catalog = res.json()
    assert len(catalog) >= 5, "Catalog should have at least 5 built-in items"

    # Verify every item carries honest CC0 licensing and dimensions
    for item in catalog:
        assert item["id"].startswith("cc0-")
        assert "CC0" in item["license"]
        assert "width" in item["dimensions"]
        assert "depth" in item["dimensions"]
        assert "height" in item["dimensions"]
        assert item["dimensions"]["unit"] == "m"

    # 2. Filter by category
    chairs_res = client.get("/api/furniture/catalog?category=chair")
    assert chairs_res.status_code == 200
    chairs = chairs_res.json()
    assert len(chairs) >= 2
    for c in chairs:
        assert c["category"] == "chair"

    # 3. Get single item details
    first_id = catalog[0]["id"]
    item_res = client.get(f"/api/furniture/catalog/{first_id}")
    assert item_res.status_code == 200
    assert item_res.json()["id"] == first_id

    # 4. Download binary glTF (GLB)
    glb_res = client.get(f"/api/furniture/catalog/{first_id}/glb")
    assert glb_res.status_code == 200
    assert glb_res.headers["content-type"] == "model/gltf-binary"
    assert glb_res.content[:4] == b"glTF", "Must start with glTF magic header bytes"
    assert "CC0" in glb_res.headers.get("x-license", "")


def test_furniture_commands_and_provenance(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """T-4b.1: Move, rotate, delete, replace furniture; verify edits stored with provenance 'generated'."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Reconstruct world to obtain initial entities
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    wir_res = client.get(f"/api/worlds/{wid}/worldir")
    assert wir_res.status_code == 200
    entities = wir_res.json()["entities"]
    assert entities, "World must contain entities"
    target_id = list(entities.keys())[0]

    # 2. Command: Move
    move_res = client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "move",
            "entity_id": target_id,
            "params": {"position": [2.5, 0.0, -1.2]},
            "description": "Move sofa toward center window",
        },
    )
    assert move_res.status_code == 201
    assert move_res.json()["provenance"] == "generated"

    # Check updated position and provenance in WorldIR
    wir_after_move = client.get(f"/api/worlds/{wid}/worldir").json()
    moved_ent = wir_after_move["entities"][target_id]
    assert moved_ent["transform"]["position"]["x"] == 2.5
    assert moved_ent["transform"]["position"]["z"] == -1.2
    assert moved_ent["provenance"].lower() == "generated"

    # 3. Command: Rotate
    rot_res = client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "rotate",
            "entity_id": target_id,
            "params": {"yaw_deg": 90.0},
            "description": "Rotate 90 degrees",
        },
    )
    assert rot_res.status_code == 201
    assert rot_res.json()["provenance"] == "generated"

    wir_after_rot = client.get(f"/api/worlds/{wid}/worldir").json()
    rot_ent = wir_after_rot["entities"][target_id]
    assert rot_ent["provenance"].lower() == "generated"
    assert abs(rot_ent["custom_properties"]["yaw_rad"] - 1.57079) < 0.01

    # 4. Command: Replace with CC0 catalog item
    replace_res = client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "replace",
            "entity_id": target_id,
            "params": {"catalog_id": "cc0-chair-dining"},
            "description": "Replace with CC0 dining chair",
        },
    )
    assert replace_res.status_code == 201
    assert replace_res.json()["provenance"] == "generated"

    wir_after_rep = client.get(f"/api/worlds/{wid}/worldir").json()
    rep_ent = wir_after_rep["entities"][target_id]
    assert rep_ent["name"] == "Modern Dining Chair"
    assert rep_ent["provenance"].lower() == "generated"
    assert rep_ent["custom_properties"]["catalog_id"] == "cc0-chair-dining"
    assert "CC0" in rep_ent["custom_properties"]["license"]


def test_persistent_undo_redo_and_reload(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """T-6.7: Persistent undo/redo and reload resilience."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    wir_init = client.get(f"/api/worlds/{wid}/worldir").json()
    target_id = list(wir_init["entities"].keys())[0]
    orig_pos = wir_init["entities"][target_id]["transform"]["position"]

    # 1. Execute Move command
    client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "move",
            "entity_id": target_id,
            "params": {"position": [4.0, 0.0, 1.0]},
            "description": "Move entity to (4, 0, 1)",
        },
    )

    # 2. Execute Replace command
    client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "replace",
            "entity_id": target_id,
            "params": {"catalog_id": "cc0-table-coffee"},
            "description": "Replace with CC0 coffee table",
        },
    )

    # Check command log
    cmds_res = client.get(f"/api/worlds/{wid}/commands")
    assert cmds_res.status_code == 200
    cmds_data = cmds_res.json()
    assert cmds_data["can_undo"] is True
    assert cmds_data["can_redo"] is False
    assert len(cmds_data["commands"]) == 2

    # 3. Undo replace
    undo_1 = client.post(f"/api/worlds/{wid}/undo")
    assert undo_1.status_code == 200
    assert undo_1.json()["undone"] is True
    assert undo_1.json()["command_type"] == "replace"

    # WorldIR entity is restored to pre-replace state (name & properties restored, position remains [4, 0, 1])
    wir_u1 = client.get(f"/api/worlds/{wid}/worldir").json()
    assert wir_u1["entities"][target_id]["name"] != "Round Coffee Table"
    assert wir_u1["entities"][target_id]["transform"]["position"]["x"] == 4.0

    # 4. Reload test: Query commands again (as if after browser refresh)
    cmds_reloaded = client.get(f"/api/worlds/{wid}/commands").json()
    assert cmds_reloaded["can_undo"] is True
    assert cmds_reloaded["can_redo"] is True

    # 5. Undo move after reload
    undo_2 = client.post(f"/api/worlds/{wid}/undo")
    assert undo_2.status_code == 200
    assert undo_2.json()["undone"] is True
    assert undo_2.json()["command_type"] == "move"

    # Position is restored to original
    wir_u2 = client.get(f"/api/worlds/{wid}/worldir").json()
    assert wir_u2["entities"][target_id]["transform"]["position"]["x"] == orig_pos["x"]

    # 6. Redo move
    redo_1 = client.post(f"/api/worlds/{wid}/redo")
    assert redo_1.status_code == 200
    assert redo_1.json()["redone"] is True
    assert redo_1.json()["command_type"] == "move"

    wir_r1 = client.get(f"/api/worlds/{wid}/worldir").json()
    assert wir_r1["entities"][target_id]["transform"]["position"]["x"] == 4.0


def test_cross_world_undo_isolation(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """T-6.7: Undo in World A never touches World B."""
    sid_a, wid_a = _setup(client, tmp_path, monkeypatch)
    sid_b, wid_b = _setup(client, tmp_path, monkeypatch)

    r_a = client.post(f"/api/sessions/{sid_a}/reconstruct")
    _wait(client, r_a.json()["job_id"])
    r_b = client.post(f"/api/sessions/{sid_b}/reconstruct")
    _wait(client, r_b.json()["job_id"])

    wir_a = client.get(f"/api/worlds/{wid_a}/worldir").json()
    wir_b = client.get(f"/api/worlds/{wid_b}/worldir").json()
    ent_a = list(wir_a["entities"].keys())[0]
    ent_b = list(wir_b["entities"].keys())[0]

    # Command on World A
    client.post(
        f"/api/worlds/{wid_a}/commands",
        json={
            "command_type": "move",
            "entity_id": ent_a,
            "params": {"position": [10.0, 0.0, 0.0]},
            "description": "Move in World A",
        },
    )

    # Command on World B
    client.post(
        f"/api/worlds/{wid_b}/commands",
        json={
            "command_type": "move",
            "entity_id": ent_b,
            "params": {"position": [20.0, 0.0, 0.0]},
            "description": "Move in World B",
        },
    )

    # Check commands for World A contains only A's commands
    cmds_a = client.get(f"/api/worlds/{wid_a}/commands").json()
    assert len(cmds_a["commands"]) == 1
    assert cmds_a["commands"][0]["entity_id"] == ent_a

    # Check commands for World B contains only B's commands
    cmds_b = client.get(f"/api/worlds/{wid_b}/commands").json()
    assert len(cmds_b["commands"]) == 1
    assert cmds_b["commands"][0]["entity_id"] == ent_b

    # Undo in World A
    undo_res = client.post(f"/api/worlds/{wid_a}/undo")
    assert undo_res.status_code == 200

    # World A is undone (position reverts to original)
    wir_a_after = client.get(f"/api/worlds/{wid_a}/worldir").json()
    assert wir_a_after["entities"][ent_a]["transform"]["position"]["x"] != 10.0

    # World B is completely untouched!
    wir_b_after = client.get(f"/api/worlds/{wid_b}/worldir").json()
    assert wir_b_after["entities"][ent_b]["transform"]["position"]["x"] == 20.0
    cmds_b_after = client.get(f"/api/worlds/{wid_b}/commands").json()
    assert cmds_b_after["can_undo"] is True
    assert cmds_b_after["commands"][0]["undone"] is False
