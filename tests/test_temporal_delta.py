"""Tests for Multi-Epoch 4D As-Built Temporal Delta & Automated Site Change Detection (Track D)."""

from __future__ import annotations

import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World, WorldVersion
from apps.api.worldstore_service import get_store
from engine.pipeline.temporal_delta import (
    SpatialChangeCluster,
    TemporalChangeKind,
    TemporalEntityDelta,
    TemporalSiteDeltaSummary,
    TradeCategory,
    classify_trade,
    compute_temporal_delta,
    detect_spatial_change_clusters,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_classify_trade():
    """Verify trade categorization of entities."""
    assert classify_trade("WALL") == TradeCategory.STRUCTURE
    assert classify_trade("COLUMN") == TradeCategory.STRUCTURE
    assert classify_trade("DOOR") == TradeCategory.ARCHITECTURE
    assert classify_trade("WINDOW") == TradeCategory.ARCHITECTURE
    assert classify_trade("ELECTRICAL") == TradeCategory.MEP
    assert classify_trade("FIXTURE", {"trade": "mep"}) == TradeCategory.MEP
    assert classify_trade("FURNITURE") == TradeCategory.EQUIPMENT


def test_temporal_delta_entity_classification():
    """Verify comprehensive entity delta classification: added, removed, moved, renovated, unchanged."""
    w1 = WorldIR(id=f"wld_ep1_{uuid.uuid4().hex[:8]}")
    w2 = WorldIR(id=f"wld_ep2_{uuid.uuid4().hex[:8]}")

    # 1. Static Anchor (Exterior Wall): stays in exact same place
    wall_static_1 = Entity(
        id="wall_ext_north",
        name="North Exterior Shear Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 8.0, "depth_m": 0.3, "height_m": 3.0, "trade": "structure"},
    )
    wall_static_2 = Entity(
        id="wall_ext_north",
        name="North Exterior Shear Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.005, "y": 0.0, "z": 0.005}},  # 7mm drift (below 3cm tolerance)
        custom_properties={"width_m": 8.0, "depth_m": 0.3, "height_m": 3.0, "trade": "structure"},
    )
    w1.entities[wall_static_1.id] = wall_static_1
    w2.entities[wall_static_2.id] = wall_static_2

    # 2. Moved Equipment (Generator / HVAC Unit): displaced by 1.2 meters
    hvac_1 = Entity(
        id="hvac_unit_1",
        name="Roof HVAC Chiller Unit",
        type="fixture",
        transform={"position": {"x": 10.0, "y": 4.0, "z": 2.0}},
        custom_properties={"width_m": 1.5, "depth_m": 1.2, "height_m": 1.0, "trade": "mep"},
    )
    hvac_2 = Entity(
        id="hvac_unit_1",
        name="Roof HVAC Chiller Unit",
        type="fixture",
        transform={"position": {"x": 11.0, "y": 4.0, "z": 2.5}},  # Displaced by sqrt(1^2 + 0.5^2) = 1.118m
        custom_properties={"width_m": 1.5, "depth_m": 1.2, "height_m": 1.0, "trade": "mep"},
    )
    w1.entities[hvac_1.id] = hvac_1
    w2.entities[hvac_2.id] = hvac_2

    # 3. Renovated Element (Partition Wall): expanded/carved
    part_1 = Entity(
        id="wall_int_office",
        name="Office Partition Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 4.0, "y": 0.0, "z": 5.0}},
        custom_properties={"width_m": 4.0, "depth_m": 0.15, "height_m": 2.8},
    )
    part_2 = Entity(
        id="wall_int_office",
        name="Office Partition Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 4.0, "y": 0.0, "z": 5.0}},
        custom_properties={"width_m": 6.0, "depth_m": 0.15, "height_m": 2.8},  # 50% width expansion
    )
    w1.entities[part_1.id] = part_1
    w2.entities[part_2.id] = part_2

    # 4. Removed Element (Temporary Scaffolding): exists in w1 only
    scaffold = Entity(
        id="scaffold_temp_a",
        name="Temporary Shoring Scaffolding",
        type="equipment",
        transform={"position": {"x": 2.0, "y": 0.0, "z": 3.0}},
        custom_properties={"width_m": 2.0, "depth_m": 2.0, "height_m": 4.0, "trade": "equipment"},
    )
    w1.entities[scaffold.id] = scaffold

    # 5. Added Element (Distribution Panel): newly installed in w2
    panel = Entity(
        id="elec_panel_b1",
        name="Main Electrical Distribution Panel",
        type="fixture",
        transform={"position": {"x": 1.5, "y": 1.2, "z": 0.1}},
        custom_properties={"width_m": 0.8, "depth_m": 0.3, "height_m": 1.8, "trade": "mep"},
    )
    w2.entities[panel.id] = panel

    # Compute delta
    delta = compute_temporal_delta(
        world_from=w1,
        world_to=w2,
        from_epoch_id="epoch_2026_09",
        to_epoch_id="epoch_2026_11",
        movement_tolerance_m=0.03,
    )

    assert delta.from_epoch_id == "epoch_2026_09"
    assert delta.to_epoch_id == "epoch_2026_11"
    assert delta.total_unchanged == 1
    assert delta.total_moved == 1
    assert delta.total_renovated == 1
    assert delta.total_removed == 1
    assert delta.total_added == 1
    assert delta.static_anchor_count == 1
    assert delta.residual_alignment_error_m < 0.01

    # Check moved entity details
    moved_item = next(c for c in delta.changes if c.entity_id == "hvac_unit_1")
    assert moved_item.change_kind == TemporalChangeKind.MOVED
    assert pytest.approx(moved_item.displacement_m, rel=0.01) == 1.118
    assert pytest.approx(moved_item.displacement_vector[0], abs=1e-3) == 1.0
    assert pytest.approx(moved_item.displacement_vector[2], abs=1e-3) == 0.5

    # Check trade counts
    assert delta.trade_counts["mep"]["added"] == 1
    assert delta.trade_counts["mep"]["moved"] == 1
    assert delta.trade_counts["equipment"]["removed"] == 1
    assert delta.trade_counts["structure"]["unchanged"] == 1

    d = delta.to_dict()
    assert d["counts"]["total_analyzed"] == 5


def test_spatial_change_clusters():
    """Verify 3D spatial voxel grid differencing for volumetric material changes."""
    # Group of points for Epoch 1 (e.g. demolished partition at X=2)
    pts1 = np.array([[2.0 + i * 0.1, j * 0.1, 0.0] for i in range(4) for j in range(4)], dtype=np.float32)
    # Group of points for Epoch 2 (e.g. newly built wall at X=8)
    pts2 = np.array([[8.0 + i * 0.1, j * 0.1, 0.0] for i in range(4) for j in range(4)], dtype=np.float32)

    clusters = detect_spatial_change_clusters(pts1, pts2, voxel_size=0.15, min_cluster_voxels=5)
    assert len(clusters) == 2

    added_cl = next(c for c in clusters if c.change_kind == TemporalChangeKind.ADDED)
    removed_cl = next(c for c in clusters if c.change_kind == TemporalChangeKind.REMOVED)

    assert added_cl.centroid[0] > 7.0
    assert removed_cl.centroid[0] < 3.0
    assert added_cl.volume_m3 > 0.0


@pytest.mark.asyncio
async def test_temporal_api_endpoints():
    """Verify REST API endpoints: /api/worlds/{id}/epochs and /temporal/compare."""
    client = TestClient(app)

    # 1. Setup World with 2 Epochs in DB and WorldStore
    store = get_store()

    # Epoch 1: Framing
    w1 = WorldIR(id=f"wld_temp_{uuid.uuid4().hex[:8]}")
    wall1 = Entity(
        id="wall_1",
        name="Foundation Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 5.0, "depth_m": 0.3, "height_m": 2.5},
    )
    w1.entities[wall1.id] = wall1
    sv1 = store.save_version(w1, parent=None)

    # Epoch 2: MEP & Finishes
    w2 = WorldIR(id=w1.id)
    wall1_copy = Entity(
        id="wall_1",
        name="Foundation Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width_m": 5.0, "depth_m": 0.3, "height_m": 2.5},
    )
    panel = Entity(
        id="panel_1",
        name="Breaker Box",
        type=EntityType.INFRASTRUCTURE,
        transform={"position": {"x": 1.0, "y": 1.5, "z": 0.0}},
        custom_properties={"width_m": 0.6, "depth_m": 0.2, "height_m": 1.0, "trade": "mep"},
    )
    w2.entities[wall1_copy.id] = wall1_copy
    w2.entities[panel.id] = panel
    sv2 = store.save_version(w2, parent=sv1.version_id)

    # Database records
    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w1.id,
            name="Construction Site 4D Test",
            current_version_id=sv2.version_id,
            status="active",
        )
        session.add(world_row)

        ver1_row = WorldVersion(
            id=sv1.version_id,
            world_id=w1.id,
            label="Epoch 1 - Framing",
        )
        ver2_row = WorldVersion(
            id=sv2.version_id,
            world_id=w1.id,
            parent_version_id=sv1.version_id,
            label="Epoch 2 - MEP Rough-In",
        )
        session.add(ver1_row)
        session.add(ver2_row)
        await session.commit()

    # 2. Test GET /api/worlds/{id}/epochs
    res_ep = client.get(f"/api/worlds/{w1.id}/epochs")
    assert res_ep.status_code == 200
    ep_data = res_ep.json()
    assert ep_data["world_id"] == w1.id
    assert len(ep_data["epochs"]) >= 2
    assert any(e["epoch_id"] == sv1.version_id for e in ep_data["epochs"])
    assert any(e["epoch_id"] == sv2.version_id for e in ep_data["epochs"])

    # 3. Test GET /api/worlds/{id}/temporal/compare with explicit epochs
    res_comp = client.get(
        f"/api/worlds/{w1.id}/temporal/compare?from_epoch={sv1.version_id}&to_epoch={sv2.version_id}"
    )
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert comp_data["from_epoch_id"] == sv1.version_id
    assert comp_data["to_epoch_id"] == sv2.version_id
    assert comp_data["counts"]["added"] == 1
    assert comp_data["counts"]["unchanged"] == 1
    assert comp_data["counts"]["removed"] == 0
    assert comp_data["counts"]["moved"] == 0

    # 4. Test GET /api/worlds/{id}/temporal/compare with default epochs (parent -> current)
    res_def = client.get(f"/api/worlds/{w1.id}/temporal/compare")
    assert res_def.status_code == 200
    assert res_def.json()["to_epoch_id"] == sv2.version_id
