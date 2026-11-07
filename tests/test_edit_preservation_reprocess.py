"""T-4.2: Late reconstruction results never overwrite accepted manual edits (rule 18).

Contract:
- User-accepted manual edits (dimensions, reclassifications, splits, merges) must survive
  when a session or world is reprocessed or late reconstruction runs.
- Tested both at the WorldIR level (carry_corrections, carry_decisions by exact ID and spatial anchor)
  and through the full HTTP API job execution pipeline.
"""

from __future__ import annotations

import copy
import pytest

from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401
from tests.test_room_fidelity_fixture import _compiled
from world_ir import EntityType
from world_ir.corrections import (
    carry_corrections,
    carry_decisions,
    make_correction,
    reclassify_entity,
    record_correction,
    split_entity,
)
from world_ir.measure import stamp_world_dimensions


def _metric_world():
    _obs, world, _diag = copy.deepcopy(_compiled("albedo", False))
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0, "dimension_state": "MEASURED"}
    stamp_world_dimensions(world)
    return world


def _slab(world):
    return next(
        e for e in world.entities.values()
        if e.type == EntityType.WALL and (e.custom_properties or {}).get("wall_slab")
    )


def test_carry_decisions_reclassify_by_exact_id():
    """A user reclassification of an entity survives a rebuild with identical IDs."""
    world = _metric_world()
    wall = _slab(world)
    old_class = (wall.custom_properties or {}).get("entity_class")

    dec = reclassify_entity(world, wall.id, "soundproof_partition", note="studio test note", user_id="tester")
    assert wall.custom_properties["entity_class"] == "soundproof_partition"
    assert wall.custom_properties["classification_basis"] == "user"
    assert "soundproof_partition" in wall.semantic_labels
    assert world.metadata["user_decisions"] == [dec]

    # Rebuild world (fresh compile where entity starts back with its automatic class)
    rebuilt = _metric_world()
    rebuilt_wall = rebuilt.entities[wall.id]
    assert rebuilt_wall.custom_properties.get("entity_class") == old_class

    # Carry user decisions onto rebuilt world
    res = carry_decisions(rebuilt, world.metadata["user_decisions"])
    assert len(res["applied"]) == 1
    assert res["applied"][0]["entity_id"] == wall.id
    assert res["applied"][0]["matched_by"] == "id"

    # Assert edit survived
    assert rebuilt_wall.custom_properties["entity_class"] == "soundproof_partition"
    assert rebuilt_wall.custom_properties["classification_basis"] == "user"
    assert "soundproof_partition" in rebuilt_wall.semantic_labels
    assert rebuilt.metadata["user_decisions"] == [dec]


def test_carry_decisions_reclassify_by_spatial_anchor():
    """A user reclassification survives even if compiler regenerated different entity IDs."""
    world = _metric_world()
    wall = _slab(world)

    dec = reclassify_entity(world, wall.id, "glass_curtain_wall", note="laser verified", user_id="surveyor")

    # Freshly compiled world where entity ID has changed
    rebuilt = _metric_world()
    target = rebuilt.entities.pop(wall.id)
    target.id = "regenerated_wall_999"
    target.custom_properties["entity_class"] = "generic_wall"
    target.custom_properties.pop("classification_basis", None)
    rebuilt.entities["regenerated_wall_999"] = target

    res = carry_decisions(rebuilt, [dec])
    assert len(res["applied"]) == 1
    assert res["applied"][0]["entity_id"] == "regenerated_wall_999"
    assert res["applied"][0]["matched_by"].startswith("nearest")

    # Assert edit survived on the spatially matched entity
    assert target.custom_properties["entity_class"] == "glass_curtain_wall"
    assert target.custom_properties["classification_basis"] == "user"
    assert rebuilt.metadata["user_decisions"] == [dec]


def test_carry_decisions_and_corrections_combined():
    """Both dimension corrections and classification decisions survive together."""
    world = _metric_world()
    wall = _slab(world)

    # 1. User corrects dimension
    dim_rec = record_correction(world, make_correction(world, wall.id, "length_m", 6.45, "tape"))
    # 2. User reclassifies
    class_dec = reclassify_entity(world, wall.id, "load_bearing_wall", note="engineer report")

    rebuilt = _metric_world()
    rebuilt_wall = rebuilt.entities[wall.id]

    carry_corrections(rebuilt, world.metadata["user_corrections"])
    carry_decisions(rebuilt, world.metadata["user_decisions"])

    # Both manual edits are intact
    assert rebuilt_wall.custom_properties["length_m"] == pytest.approx(6.45)
    assert rebuilt_wall.custom_properties["dimensions"]["length_m"]["state"] == "VERIFIED"
    assert rebuilt_wall.custom_properties["entity_class"] == "load_bearing_wall"
    assert rebuilt_wall.custom_properties["classification_basis"] == "user"
    assert rebuilt.metadata["user_corrections"] == [dim_rec]
    assert rebuilt.metadata["user_decisions"] == [class_dec]


def test_api_reprocess_preserves_accepted_manual_edits(client, tmp_path, monkeypatch):
    """End-to-end API pipeline test: edit committed to world survives subsequent reconstruction job."""
    from tests.test_reconstruction_lifecycle_semantics import _setup, _wait

    sid, wid = _setup(client, tmp_path, monkeypatch)

    # Initial reconstruction
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200, r.text
    job1 = _wait(client, r.json()["job_id"])
    assert job1["status"] in ("succeeded", "partial"), job1

    v1_world = client.get(f"/api/worlds/{wid}/worldir").json()
    assert v1_world["entities"], "initial world must have entities"
    eid = next(iter(v1_world["entities"].keys()))

    # User accepts a manual edit: reclassifying the entity
    commit_res = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": eid,
            "changes": {
                "reclassify": {
                    "new_class": "feature_column",
                    "note": "architect on-site confirmation",
                }
            },
            "commit_message": "Accept manual reclassification of column",
        },
    )
    assert commit_res.status_code == 200, commit_res.text

    # Verify committed state
    v2_world = client.get(f"/api/worlds/{wid}/worldir").json()
    assert v2_world["entities"][eid]["custom_properties"]["entity_class"] == "feature_column"
    assert v2_world["entities"][eid]["custom_properties"]["classification_basis"] == "user"

    # Now simulate a late reprocess / second reconstruction job
    r2 = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r2.status_code == 200, r2.text
    job2 = _wait(client, r2.json()["job_id"])
    assert job2["status"] in ("succeeded", "partial"), job2

    # Verify that the reconstructed world preserved the manual edit (Rule 18 / T-4.2)
    v3_world = client.get(f"/api/worlds/{wid}/worldir").json()
    assert eid in v3_world["entities"]
    survived_ent = v3_world["entities"][eid]
    assert survived_ent["custom_properties"].get("entity_class") == "feature_column"
    assert survived_ent["custom_properties"].get("classification_basis") == "user"
    assert any(
        d.get("new_class") == "feature_column"
        for d in (v3_world["metadata"].get("user_decisions") or [])
    )
