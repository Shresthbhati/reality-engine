"""Unit and API integration tests for World-space CaptureRequests (Blueprint Task 6.1).

Verifies:
1. World-space CaptureRequests generated from unobserved/occluded surfaces carry target_region, suggested_poses, instruction, and expected_gain.
2. Fully observed surfaces (coverage >= 80%) do not trigger unnecessary capture requests.
3. API endpoint GET /api/worlds/{id}/capture-requests returns structured capture requests.
4. API endpoint PATCH /api/worlds/{id}/capture-requests/{id} updates status (open -> satisfied / dismissed).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from apps.api.main import app
from apps.api.db import get_db
from apps.api.models import CaptureRequest, World, WorldVersion, utcnow
from engine.pipeline.guidance import generate_world_capture_requests
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType


def _build_test_world_with_gaps() -> WorldIR:
    """Build a WorldIR with one fully observed wall and one unobserved wall."""
    world = WorldIR(id="wld-guidance-01")

    # Fully observed North wall (coverage 0.95)
    world.entities["wall-north"] = Entity(
        id="wall-north",
        name="North Wall",
        type=EntityType.WALL,
        confidence=0.95,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "bounds_min": [-2.0, 3.0, 0.0],
            "bounds_max": [2.0, 3.0, 2.5],
            "normal": [0.0, -1.0, 0.0],
            "coverage": 0.95,
            "evidence_state": "OBSERVED",
        },
    )

    # Unobserved South wall (coverage 0.10)
    world.entities["wall-south"] = Entity(
        id="wall-south",
        name="South Wall",
        type=EntityType.WALL,
        confidence=0.70,
        provenance=Provenance.INFERRED,
        custom_properties={
            "bounds_min": [-2.0, -3.0, 0.0],
            "bounds_max": [2.0, -3.0, 2.5],
            "normal": [0.0, 1.0, 0.0],
            "coverage": 0.10,
            "evidence_state": "UNOBSERVED",
        },
    )

    # Partially occluded East wall (coverage 0.45)
    world.entities["wall-east"] = Entity(
        id="wall-east",
        name="East Wall",
        type=EntityType.WALL,
        confidence=0.80,
        provenance=Provenance.INFERRED,
        custom_properties={
            "bounds_min": [3.0, -2.0, 0.0],
            "bounds_max": [3.0, 2.0, 2.5],
            "normal": [-1.0, 0.0, 0.0],
            "coverage": 0.45,
            "evidence_state": "OCCLUDED",
        },
    )

    return world


def test_generate_world_capture_requests():
    """Verify CaptureRequest generation logic in engine/pipeline/guidance.py."""
    world = _build_test_world_with_gaps()
    reqs = generate_world_capture_requests(world, world_id="wld-01")

    # North wall is 95% covered, so only South and East wall should generate requests
    assert len(reqs) == 2
    target_ids = {r["target_ids"][0] for r in reqs}
    assert target_ids == {"wall-south", "wall-east"}

    south_req = next(r for r in reqs if r["target_ids"][0] == "wall-south")
    assert south_req["status"] == "open"
    assert south_req["expected_gain"] == 0.90
    assert "target_region" in south_req
    assert "bounds_min" in south_req["target_region"]
    assert "center" in south_req["target_region"]
    assert south_req["suggested_poses"]
    assert "position" in south_req["suggested_poses"][0]
    assert "look_at" in south_req["suggested_poses"][0]
    assert "instruction" in south_req
    assert "South Wall" in south_req["instruction"]

    east_req = next(r for r in reqs if r["target_ids"][0] == "wall-east")
    assert east_req["expected_gain"] == 0.55
    assert "occlusion" in east_req["instruction"].lower() or "bypass" in east_req["instruction"].lower()


@pytest.mark.asyncio
async def test_capture_requests_api_flow(tmp_path, monkeypatch):
    """Verify GET and PATCH /api/worlds/{id}/capture-requests."""
    from uuid import uuid4
    from apps.api.db import get_sessionmaker, init_db

    await init_db()
    session_maker = get_sessionmaker()

    world_id = f"wld-cr-{uuid4().hex[:8]}"
    req_id = f"cr-{uuid4().hex[:8]}"

    # Insert test world and capture request in DB
    async with session_maker() as db:
        w = World(id=world_id, name="Living Room")
        db.add(w)

        cr = CaptureRequest(
            id=req_id,
            world_id=world_id,
            reason="Surface South Wall is unobserved",
            target_region={"bounds_min": [-2, -3, 0], "bounds_max": [2, -3, 2.5], "center": [0, -3, 1.25]},
            target_ids=["wall-south"],
            suggested_poses=[{"position": [0, -1, 1.25], "look_at": [0, -3, 1.25]}],
            expected_gain=0.85,
            instruction="Stand 2m back and photograph South Wall directly.",
            status="open",
            created_version="v1",
        )
        db.add(cr)
        await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET /api/worlds/{id}/capture-requests
        res = await client.get(f"/api/worlds/{world_id}/capture-requests")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 1
        assert data["items"][0]["id"] == req_id
        assert data["items"][0]["status"] == "open"
        assert data["items"][0]["expected_gain"] == 0.85
        assert data["items"][0]["instruction"] == "Stand 2m back and photograph South Wall directly."

        # 2. PATCH status to satisfied
        patch_res = await client.patch(
            f"/api/worlds/{world_id}/capture-requests/{req_id}",
            json={"status": "satisfied"},
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["status"] == "satisfied"

        # 3. GET with status filter
        filter_res = await client.get(f"/api/worlds/{world_id}/capture-requests?status=satisfied")
        assert filter_res.status_code == 200
        assert filter_res.json()["count"] == 1

        filter_open = await client.get(f"/api/worlds/{world_id}/capture-requests?status=open")
        assert filter_open.status_code == 200
        assert filter_open.json()["count"] == 0

        # 4. Invalid status rejection
        bad_res = await client.patch(
            f"/api/worlds/{world_id}/capture-requests/{req_id}",
            json={"status": "invalid_state"},
        )
        assert bad_res.status_code == 422
