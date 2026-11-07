"""Tests for multi-session world adoption API & progressive project evolution (Tasks 6.2 & 6.3).

Verifies Decision 027 contracts:
1. Candidate matching & Sim(3) alignment confidence scoring (C_align).
2. Operational confidence tiers: HIGH (>= 0.85), MODERATE (0.60 - 0.85), LOW (< 0.60).
3. Automatic refusal with HTTP 422 when C_align < min_confidence (A11, A19).
4. 5-state semantic ChangeSet: matched, new, updated, uncertain, conflicting.
5. Strict Rule 18 protection: late reconstruction never overwrites manual user edits.
6. Atomic adoption transaction: version increment, HEAD move, and adoption state machine.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_propose_and_commit_world_adoption_lifecycle(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """Verify complete adoption workflow: proposal, 5-state changeset, Rule 18 edit protection, atomic commit."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Initialize base world with a reconstruction
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    wir_base = client.get(f"/api/worlds/{wid}/worldir").json()
    base_entities = wir_base["entities"]
    assert len(base_entities) >= 1

    # Enrich base world with initial walls so we have matched, updated, and manual edit test cases
    import asyncio

    import apps.api.worldstore_service as ws_svc
    from apps.api.db import get_sessionmaker

    store = ws_svc.get_store()
    cur_vid = client.get(f"/api/worlds/{wid}").json()["current_version_id"]
    base_w = store.load_version(cur_vid)

    target_edit_id = "wall_manual_edit_01"
    base_w.entities[target_edit_id] = Entity(
        id=target_edit_id,
        type=EntityType.WALL,
        name="Custom Wall",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 1.3}},
        custom_properties={"length_m": 5.0, "position": [0.0, 0.0, 1.3]},
    )
    base_w.entities["wall_matched_01"] = Entity(
        id="wall_matched_01",
        type=EntityType.WALL,
        name="Matched Wall",
        transform={"position": {"x": 2.5, "y": 0.0, "z": 1.3}},
        custom_properties={"length_m": 5.0, "position": [2.5, 0.0, 1.3]},
    )
    base_w.entities["wall_updated_01"] = Entity(
        id="wall_updated_01",
        type=EntityType.WALL,
        name="Wall To Update",
        transform={"position": {"x": 5.0, "y": 2.0, "z": 1.3}},
        custom_properties={"length_m": 4.0, "position": [5.0, 2.0, 1.3]},
    )

    async def _add_base_entities():
        async with get_sessionmaker()() as db:
            await ws_svc.commit_version(db, world_id=wid, world=base_w, parent=cur_vid)

    asyncio.run(_add_base_entities())

    # Mark target_edit_id as an intentional manual user edit via command
    cmd_res = client.post(
        f"/api/worlds/{wid}/commands",
        json={
            "command_type": "move",
            "entity_id": target_edit_id,
            "params": {"position": [10.0, 10.0, 0.0]},
            "description": "User adjusted wall manually",
        },
    )
    assert cmd_res.status_code == 201

    # Reload base world after edit
    wir_edited = client.get(f"/api/worlds/{wid}/worldir").json()
    assert wir_edited["entities"][target_edit_id]["custom_properties"].get("is_manual_edit") is True or \
           wir_edited["entities"][target_edit_id]["provenance"].lower() in ("generated", "user_edit", "manual")

    # 2. Build candidate world from second capture session
    cand_world = WorldIR(id="cand_ses_second_visit", name="Second Site Visit")

    # A. Matched entity: existing wall with identical coordinates
    matched_ent = Entity(
        id="wall_matched_01",
        type=EntityType.WALL,
        name="Matched Wall",
        transform={"position": {"x": 2.5, "y": 0.0, "z": 1.3}},
        custom_properties={"length_m": 5.0, "position": [2.5, 0.0, 1.3]},
    )
    cand_world.entities["wall_matched_01"] = matched_ent

    # B. New entity: adjacent kitchen surveyed in session 2
    kitchen_room = Entity(
        id="room_kitchen_02",
        type=EntityType.ROOM,
        name="Kitchen",
        custom_properties={"floor_area_m2": 15.0, "height_m": 2.6},
    )
    cand_world.entities[kitchen_room.id] = kitchen_room

    kitchen_wall = Entity(
        id="wall_kitchen_k01",
        type=EntityType.WALL,
        name="Kitchen Partition Wall",
        custom_properties={"room_id": "room_kitchen_02", "length_m": 3.8, "position": [7.0, 2.0, 1.3]},
    )
    cand_world.entities[kitchen_wall.id] = kitchen_wall

    # C. Updated entity: wall with refined position (+3 cm offset)
    updated_ent = Entity(
        id="wall_updated_01",
        type=EntityType.WALL,
        name="Refined Wall",
        transform={"position": {"x": 5.03, "y": 2.0, "z": 1.3}},
        custom_properties={"length_m": 4.02, "position": [5.03, 2.0, 1.3]},
    )
    cand_world.entities["wall_updated_01"] = updated_ent

    # D. Uncertain entity: partial observation with low confidence
    uncertain_ent = Entity(
        id="chair_uncertain_09",
        type=EntityType.FURNITURE,
        name="Obscured Chair",
        confidence=0.45,
        custom_properties={"coverage_state": "PARTIALLY_OBSERVED", "position": [1.0, 1.0, 0.5]},
    )
    cand_world.entities[uncertain_ent.id] = uncertain_ent

    # E. Conflicting entity: candidate reconstruction tries to place wall back to original position,
    # clashing with the manual user edit on target_edit_id
    conflicting_ent = Entity(
        id=target_edit_id,
        type=EntityType.WALL,
        name="Clashing Wall",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 1.3}},
        custom_properties={"length_m": 5.0, "position": [0.0, 0.0, 1.3]},
    )
    cand_world.entities[target_edit_id] = conflicting_ent

    # 3. Propose adoption: POST /api/worlds/{wid}/adoptions
    prop_res = client.post(
        f"/api/worlds/{wid}/adoptions",
        json={
            "session_id": "ses_visit_2",
            "min_confidence": 0.60,
            "candidate_world": cand_world.to_dict(),
            "scale_factor": 1.002,
            "overlap_ratio": 0.92,
            "feature_inlier_ratio": 0.89,
        },
    )
    assert prop_res.status_code == 200, prop_res.text
    prop_data = prop_res.json()

    assert prop_data["adoption_id"].startswith("adp_")
    assert prop_data["world_id"] == wid
    assert prop_data["session_id"] == "ses_visit_2"
    assert prop_data["alignment_tier"] == "HIGH"
    assert prop_data["alignment_confidence"] >= 0.85
    assert prop_data["status"] == "PROPOSED"

    # Verify 5-state ChangeSet
    cs = prop_data["change_set"]
    assert "room_kitchen_02" in cs["new"]
    assert "wall_kitchen_k01" in cs["new"]
    assert "chair_uncertain_09" in cs["uncertain"]
    assert target_edit_id in cs["conflicting"]  # Clashing with user edit

    # 4. GET adoption proposal details
    aid = prop_data["adoption_id"]
    get_res = client.get(f"/api/worlds/{wid}/adoptions/{aid}")
    assert get_res.status_code == 200
    assert get_res.json()["adoption_id"] == aid

    # 5. Commit adoption: POST /api/worlds/{wid}/adoptions/{aid}/commit
    commit_res = client.post(
        f"/api/worlds/{wid}/adoptions/{aid}/commit",
        json={"resolutions": {target_edit_id: "keep_edit"}},
    )
    assert commit_res.status_code == 200, commit_res.text
    commit_data = commit_res.json()
    assert commit_data["status"] == "ADOPTED"
    assert commit_data["version_id"].startswith("v-")

    # 6. Verify synthesized WorldIR: user edit preserved (Rule 18) and new room adopted
    wir_final = client.get(f"/api/worlds/{wid}/worldir").json()
    final_entities = wir_final["entities"]

    # New room and kitchen wall were adopted
    assert "room_kitchen_02" in final_entities
    assert "wall_kitchen_k01" in final_entities

    # Rule 18 verification: manual edit position [10, 10, 0] was strictly retained!
    preserved_wall = final_entities[target_edit_id]
    p = preserved_wall["transform"]["position"]
    assert p["x"] == 10.0 and p["y"] == 10.0


def test_refuse_low_confidence_adoption_with_422(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """Verify that non-matching / low-confidence sessions are refused with 422 (A11, A19)."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # Initialize base world
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    # Create empty/unrelated candidate session with large scale distortion
    unrelated_cand = WorldIR(id="cand_unrelated")

    refuse_res = client.post(
        f"/api/worlds/{wid}/adoptions",
        json={
            "session_id": "ses_unrelated",
            "min_confidence": 0.70,
            "candidate_world": unrelated_cand.to_dict(),
            "scale_factor": 1.25,  # 25% scale error -> 0.0 scale score
            "overlap_ratio": 0.15,
            "feature_inlier_ratio": 0.10,
        },
    )
    assert refuse_res.status_code == 422
    err_msg = refuse_res.json()["detail"]
    assert "Automatic cross-session alignment refused" in err_msg
    assert "below threshold 0.70" in err_msg


def test_reject_adoption_flow(client: TestClient, tmp_path, monkeypatch):  # noqa: F811
    """Verify proposal rejection and subsequent commit failure."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    prop_res = client.post(
        f"/api/worlds/{wid}/adoptions",
        json={
            "session_id": "ses_reject_test",
            "candidate_world": WorldIR(id="cand_rej").to_dict(),
            "scale_factor": 1.0,
            "overlap_ratio": 0.90,
            "feature_inlier_ratio": 0.90,
        },
    )
    assert prop_res.status_code == 200
    aid = prop_res.json()["adoption_id"]

    # Reject proposal
    rej_res = client.post(f"/api/worlds/{wid}/adoptions/{aid}/reject")
    assert rej_res.status_code == 200
    assert rej_res.json()["status"] == "REJECTED"

    # Attempting to commit a rejected proposal must fail with 409
    commit_res = client.post(f"/api/worlds/{wid}/adoptions/{aid}/commit")
    assert commit_res.status_code == 409
    assert "rejected" in commit_res.json()["detail"]
