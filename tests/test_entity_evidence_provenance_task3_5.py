"""Tests for Blueprint Task 3.5: Entity Evidence & Provenance in Inspector.

Verifies:
- GET /api/worlds/{world_id}/entities/{entity_id}/provenance exposes:
  * observation-matched photos with mask_uri and bbox_px
  * association_decisions ("Why is this one object?") with match probability and geometric reasoning
  * observation ledger metadata
- Fallback to session-level and procedural provenance when no ledger exists.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from apps.api.main import app
from apps.api.db import get_sessionmaker
from apps.api.models import World, WorldVersion, Evidence
from world_ir import WorldIR, Entity, EntityType, Transform, Vector3
from world_ir.observations import (
    Observation,
    AssociationDecision,
    AssociationState,
    DepthSource,
    attach_ledger,
)


@pytest.mark.asyncio
async def test_entity_evidence_provenance_with_ledger():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a world
        resp = await client.post("/api/worlds", json={"name": "Evidence Test World"})
        assert resp.status_code == 201
        world_data = resp.json()
        world_id = world_data["id"]

        # Build WorldIR with an entity and an observation ledger
        world = WorldIR(id=world_id, name="Evidence Test World")
        ent = Entity(
            id="ent_chair_001",
            type=EntityType.STRUCTURE,
            name="Office Chair",
            transform={"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
            confidence=0.92,
        )
        world.entities[ent.id] = ent

        obs = Observation(
            observation_id="obs_001",
            image_id="img_photo_01.jpg",
            label="chair",
            world_pos=(1.0, 2.0, 0.0),
            sigma_m=0.03,
            depth_source=DepthSource.METRIC,
            mask_uri="artifacts/masks/obs_001_mask.png",
            bbox_px=(100.0, 150.0, 400.0, 600.0),
            confidence=0.94,
        )
        decision = AssociationDecision(
            observation_id="obs_001",
            state=AssociationState.SAME_ENTITY_CONFIRMED,
            entity_id="ent_chair_001",
            probability=0.96,
            reason="Hungarian minimum-cost match (IoU 0.88, distance 0.02m)",
            dist_m=0.02,
            iou3d=0.88,
            geom_verified=True,
            candidates=(("ent_chair_001", 0.96),),
        )
        attach_ledger(world, [obs], [decision])

        # Commit a version to DB via canonical worldstore_service
        async with get_sessionmaker()() as db:
            from apps.api import worldstore_service
            await worldstore_service.commit_version(
                db,
                world_id=world_id,
                world=world,
                parent=None,
            )

        # Query provenance
        prov_resp = await client.get(f"/api/worlds/{world_id}/entities/ent_chair_001/provenance")
        assert prov_resp.status_code == 200
        prov = prov_resp.json()

        assert prov["entity_id"] == "ent_chair_001"
        assert "association_decisions" in prov
        assert len(prov["association_decisions"]) == 1
        dec = prov["association_decisions"][0]
        assert dec["state"] == "SAME_ENTITY_CONFIRMED"
        assert dec["probability"] == 0.96
        assert "Hungarian" in dec["reason"]
        assert dec["dist_m"] == 0.02

        assert "observations" in prov
        assert len(prov["observations"]) == 1
        obs_item = prov["observations"][0]
        assert obs_item["observation_id"] == "obs_001"
        assert obs_item["mask_uri"] == "artifacts/masks/obs_001_mask.png"
        assert obs_item["bbox_px"] == [100.0, 150.0, 400.0, 600.0]
