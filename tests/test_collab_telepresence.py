"""Tests for Multi-User Real-Time Collaborative Telepresence & Live Spatial Markup.

Verifies:
1. Peer presence tracking, heartbeat updates, pose sync, and session expiry.
2. 3D spatial annotations, pins, and stroke markups.
3. WorldIR custom_properties persistence round-trip.
4. buildingSMART BCF 2.1 zip archive export with XML validation.
5. FastAPI REST API endpoints for presence, heartbeats, markups, and BCF export.
"""

from __future__ import annotations

import io
import time
from typing import Any
import uuid
import xml.etree.ElementTree as ET
import zipfile
import pytest

from world_ir.world_v1 import WorldIR
from engine.pipeline.collab_presence import (
    CollabPresenceRegistry,
    MarkupStatus,
    MarkupTrade,
    MarkupType,
    PeerPresence,
    SpatialMarkup,
    delete_world_markup,
    export_markups_to_bcf,
    get_presence_registry,
    load_world_markups,
    save_world_markup,
)


def test_peer_presence_registry():
    """Verify registry peer addition, heartbeat updates, active listing, and timeout cleanup."""
    registry = CollabPresenceRegistry()
    world_id = "world-collab-test"

    # Add peer 1
    p1 = registry.update_peer(
        world_id=world_id,
        client_id="client-alice",
        user_name="Alice (Lead Architect)",
        role="architect",
        color="#3b82f6",
        camera_position=(1.0, 2.0, 1.8),
        look_at=(3.0, 2.0, 1.5),
        field_of_view=65.0,
    )
    assert p1.client_id == "client-alice"
    assert p1.user_name == "Alice (Lead Architect)"

    # Add peer 2
    p2 = registry.update_peer(
        world_id=world_id,
        client_id="client-bob",
        user_name="Bob (Structural Engineer)",
        role="structural_engineer",
        color="#10b981",
        camera_position=(4.0, 5.0, 2.0),
        look_at=(4.0, 1.0, 0.5),
        field_of_view=60.0,
    )
    assert p2.client_id == "client-bob"

    # List active peers
    active = registry.get_active_peers(world_id, timeout_sec=30.0)
    assert len(active) == 2
    names = [p.user_name for p in active]
    assert "Alice (Lead Architect)" in names
    assert "Bob (Structural Engineer)" in names

    # Simulate heartbeat expiry on peer 1
    p1.last_heartbeat = time.time() - 40.0
    active_after_expire = registry.get_active_peers(world_id, timeout_sec=30.0)
    assert len(active_after_expire) == 1
    assert active_after_expire[0].client_id == "client-bob"

    # Remove peer 2
    removed = registry.remove_peer(world_id, "client-bob")
    assert removed is True
    assert len(registry.get_active_peers(world_id)) == 0


def test_spatial_markup_worldir_persistence():
    """Verify spatial markup creation, serialization, and WorldIR persistence."""
    world = WorldIR(id="world-markup-test")

    # Initially empty
    assert len(load_world_markups(world)) == 0

    # Add pin markup
    markup1 = SpatialMarkup(
        markup_id="mk-pin-1",
        world_id="world-markup-test",
        markup_type=MarkupType.PIN,
        trade=MarkupTrade.ARCHITECTURE,
        title="Check Column Alignment",
        description="Verify architectural finish vs grid line 4",
        author_name="Alice Architect",
        position=(2.5, 3.0, 1.2),
        normal=(0.0, 1.0, 0.0),
        camera_viewpoint={"position": [2.5, 1.0, 1.8], "direction": [0.0, 1.0, -0.3], "fov": 60.0},
        status=MarkupStatus.OPEN,
    )

    save_world_markup(world, markup1)
    loaded = load_world_markups(world)
    assert len(loaded) == 1
    assert loaded[0].markup_id == "mk-pin-1"
    assert loaded[0].title == "Check Column Alignment"
    assert loaded[0].position == (2.5, 3.0, 1.2)
    assert loaded[0].trade == MarkupTrade.ARCHITECTURE

    # Add 3D stroke markup
    markup2 = SpatialMarkup(
        markup_id="mk-stroke-2",
        world_id="world-markup-test",
        markup_type=MarkupType.STROKE,
        trade=MarkupTrade.MEP,
        title="Conduit Penetration Path",
        stroke_points=[(1.0, 1.0, 2.7), (2.0, 1.0, 2.7), (2.0, 3.0, 2.7)],
        position=(1.0, 1.0, 2.7),
        status=MarkupStatus.IN_REVIEW,
    )

    save_world_markup(world, markup2)
    loaded2 = load_world_markups(world)
    assert len(loaded2) == 2

    # Delete first markup
    deleted = delete_world_markup(world, "mk-pin-1")
    assert deleted is True
    loaded3 = load_world_markups(world)
    assert len(loaded3) == 1
    assert loaded3[0].markup_id == "mk-stroke-2"

    # Delete non-existent
    assert delete_world_markup(world, "non-existent") is False


def test_bcf_export_validation():
    """Verify buildingSMART BCF 2.1 zip generation and XML schema compliance."""
    world_id = "world-bcf-test"
    markups = [
        SpatialMarkup(
            markup_id="mk-bcf-1",
            world_id=world_id,
            markup_type=MarkupType.PIN,
            trade=MarkupTrade.STRUCTURE,
            title="Slab Deflection Warning",
            description="Exceeds L/360 limit at center span",
            author_name="Structural Reviewer",
            position=(4.0, 6.0, 0.0),
            camera_viewpoint={"position": [3.0, 5.0, 1.8], "direction": [0.7, 0.7, -0.3], "fov": 55.0},
            status=MarkupStatus.OPEN,
        )
    ]

    zip_bytes = export_markups_to_bcf(markups, world_id=world_id)
    assert len(zip_bytes) > 200

    # Unpack zip and validate XML
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        file_list = zf.namelist()
        assert "bcf.version" in file_list

        ver_xml = zf.read("bcf.version").decode("utf-8")
        assert 'VersionId="2.1"' in ver_xml

        # Check topic folder
        topic_guid = str(uuid.uuid5(uuid.NAMESPACE_URL, "reality-engine://markup/mk-bcf-1"))
        markup_file = f"{topic_guid}/markup.bcf"
        viewpoint_file = f"{topic_guid}/viewpoint.bcfv"
        assert markup_file in file_list
        assert viewpoint_file in file_list

        # Validate markup.bcf XML structure
        markup_xml = zf.read(markup_file)
        root = ET.fromstring(markup_xml)
        assert "Markup" in root.tag
        topic = root.find("{http://www.buildingsmart-tech.org/specifications/bcf-releases}Topic")
        assert topic is not None
        assert topic.attrib["Guid"] == topic_guid
        title_el = topic.find("{http://www.buildingsmart-tech.org/specifications/bcf-releases}Title")
        assert title_el is not None and title_el.text == "Slab Deflection Warning"

        # Validate viewpoint.bcfv XML structure
        vp_xml = zf.read(viewpoint_file)
        vp_root = ET.fromstring(vp_xml)
        assert "VisualizationInfo" in vp_root.tag
        persp = vp_root.find("{http://www.buildingsmart-tech.org/specifications/bcf-releases}PerspectiveCamera")
        assert persp is not None
        cam_pt = persp.find("{http://www.buildingsmart-tech.org/specifications/bcf-releases}CameraViewPoint")
        assert cam_pt is not None
        assert cam_pt.attrib["X"] == "3.0"


@pytest.mark.asyncio
async def test_collab_api_endpoints():
    """Verify collaborative presence and spatial markup HTTP endpoints."""
    from starlette.testclient import TestClient
    from apps.api.main import app
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    client = TestClient(app)

    world_id = f"wld_collab_{uuid.uuid4().hex[:8]}"
    w = WorldIR(id=world_id)

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Collaborative Model Studio",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Base Version for Collab",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. POST /api/worlds/{id}/collab/heartbeat
    res_hb = client.post(
        f"/api/worlds/{w.id}/collab/heartbeat",
        json={
            "client_id": "test-client-1",
            "user_name": "Carol (MEP Coordinator)",
            "role": "mep_engineer",
            "color": "#f59e0b",
            "camera_position": [2.0, 3.0, 1.8],
            "look_at": [2.0, 5.0, 1.8],
            "field_of_view": 60.0,
        },
    )
    assert res_hb.status_code == 200, res_hb.text
    hb_data = res_hb.json()
    assert hb_data["status"] == "ok"
    assert hb_data["peer"]["client_id"] == "test-client-1"

    # 2. GET /api/worlds/{id}/collab/presence
    res_pres = client.get(f"/api/worlds/{w.id}/collab/presence")
    assert res_pres.status_code == 200, res_pres.text
    pres_data = res_pres.json()
    assert pres_data["peer_count"] >= 1
    assert pres_data["peers"][0]["user_name"] == "Carol (MEP Coordinator)"

    # 3. POST /api/worlds/{id}/collab/markups (Create pin)
    res_mk1 = client.post(
        f"/api/worlds/{w.id}/collab/markups",
        json={
            "markup_type": "pin",
            "trade": "mep",
            "title": "HVAC Duct Clash",
            "description": "Supply duct intersects structural beam W12x26",
            "author_name": "Carol",
            "position": [2.0, 4.0, 2.6],
            "camera_viewpoint": {"position": [2.0, 2.5, 1.8], "direction": [0.0, 1.0, 0.4]},
            "color": "#ef4444",
            "status": "open",
        },
    )
    assert res_mk1.status_code == 200, res_mk1.text
    mk1_data = res_mk1.json()
    assert mk1_data["status"] == "created"
    markup_id1 = mk1_data["markup"]["markup_id"]

    # 4. POST /api/worlds/{id}/collab/markups (Create stroke)
    res_mk2 = client.post(
        f"/api/worlds/{w.id}/collab/markups",
        json={
            "markup_type": "stroke",
            "trade": "architecture",
            "title": "Partition Wall Relocation",
            "description": "Shift drywall by 200mm to clear doorway",
            "author_name": "Alice",
            "position": [5.0, 2.0, 0.0],
            "stroke_points": [[5.0, 2.0, 0.0], [5.0, 4.0, 0.0]],
            "color": "#3b82f6",
            "status": "in_review",
        },
    )
    assert res_mk2.status_code == 200, res_mk2.text
    mk2_data = res_mk2.json()
    markup_id2 = mk2_data["markup"]["markup_id"]

    # 5. GET /api/worlds/{id}/collab/markups
    res_list = client.get(f"/api/worlds/{w.id}/collab/markups")
    assert res_list.status_code == 200, res_list.text
    list_data = res_list.json()
    assert list_data["markup_count"] == 2
    titles = [m["title"] for m in list_data["markups"]]
    assert "HVAC Duct Clash" in titles
    assert "Partition Wall Relocation" in titles

    # 6. GET /api/worlds/{id}/collab/export/bcf
    res_bcf = client.get(f"/api/worlds/{w.id}/collab/export/bcf")
    assert res_bcf.status_code == 200, res_bcf.text
    assert len(res_bcf.content) > 300
    with zipfile.ZipFile(io.BytesIO(res_bcf.content), "r") as zf:
        assert "bcf.version" in zf.namelist()

    # 7. DELETE /api/worlds/{id}/collab/markups/{markup_id}
    res_del = client.delete(f"/api/worlds/{w.id}/collab/markups/{markup_id1}")
    assert res_del.status_code == 200, res_del.text
    del_data = res_del.json()
    assert del_data["status"] == "deleted"

    # Verify count is now 1
    res_list_after = client.get(f"/api/worlds/{w.id}/collab/markups")
    assert res_list_after.status_code == 200
    assert res_list_after.json()["markup_count"] == 1
    assert res_list_after.json()["markups"][0]["markup_id"] == markup_id2
