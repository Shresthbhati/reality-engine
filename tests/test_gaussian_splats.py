"""Tests for 3D Gaussian Splatting & Hybrid Viewport (Track C)."""

from __future__ import annotations

import io
import math
import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World, WorldVersion
from apps.api.storage import store_bytes
from apps.api.worldstore_service import get_store
from engine.pipeline.gaussian_splats import (
    GaussianPrimitive,
    GaussianSplatCloud,
    decimate_splats,
    export_ply_splats,
    export_splat_binary,
    generate_splats_from_points,
    parse_ply_splats,
    parse_splat_binary,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR


def test_splat_primitive_and_cloud():
    """Verify GaussianPrimitive and GaussianSplatCloud basic behaviors."""
    p1 = GaussianPrimitive(
        position=(1.0, 2.0, 3.0),
        scale=(0.05, 0.05, 0.05),
        rotation=(0.0, 0.0, 0.0, 1.0),
        color=(255, 128, 64),
        opacity=0.9,
    )
    p2 = GaussianPrimitive(
        position=(4.0, 5.0, 6.0),
        scale=(0.1, 0.1, 0.02),
        rotation=(0.0, 0.7071, 0.0, 0.7071),
        color=(10, 20, 30),
        opacity=0.5,
    )
    cloud = GaussianSplatCloud.from_primitives([p1, p2])
    assert len(cloud) == 2
    assert cloud.count == 2
    assert cloud.byte_size == 64

    b = cloud.bounds
    assert b["min"] == [1.0, 2.0, 3.0]
    assert b["max"] == [4.0, 5.0, 6.0]

    prims = cloud.to_primitives()
    assert len(prims) == 2
    assert prims[0].position == (1.0, 2.0, 3.0)
    assert prims[0].color == (255, 128, 64)
    assert pytest.approx(prims[0].opacity, abs=1e-3) == 0.9


def test_splat_binary_roundtrip():
    """Verify standard 32-byte .splat chunk packing and unpacking."""
    positions = np.array([[1.5, -2.5, 3.25], [-0.5, 4.0, 1.0]], dtype=np.float32)
    scales = np.array([[0.04, 0.05, 0.06], [0.1, 0.1, 0.02]], dtype=np.float32)
    rotations = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    colors = np.array([[240, 120, 60], [30, 90, 180]], dtype=np.uint8)
    opacities = np.array([0.95, 0.45], dtype=np.float32)

    cloud = GaussianSplatCloud(positions, scales, rotations, colors, opacities)
    data = export_splat_binary(cloud)
    assert len(data) == 2 * 32

    cloud_dec = parse_splat_binary(data)
    assert len(cloud_dec) == 2
    np.testing.assert_allclose(cloud_dec.positions, positions, atol=1e-5)
    np.testing.assert_allclose(cloud_dec.scales, scales, atol=1e-5)
    np.testing.assert_allclose(cloud_dec.colors, colors, atol=2)
    np.testing.assert_allclose(cloud_dec.opacities, opacities, atol=1.0 / 255.0)
    np.testing.assert_allclose(cloud_dec.rotations, rotations, atol=0.02)


def test_ply_splats_binary_roundtrip():
    """Verify 3DGS binary PLY format export and parsing."""
    positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    scales = np.array([[0.03, 0.03, 0.01], [0.05, 0.05, 0.05]], dtype=np.float32)
    rotations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.7071, 0.0, 0.7071]], dtype=np.float32)
    colors = np.array([[200, 150, 100], [50, 100, 200]], dtype=np.uint8)
    opacities = np.array([0.8, 0.95], dtype=np.float32)

    cloud = GaussianSplatCloud(positions, scales, rotations, colors, opacities)
    ply_bytes = export_ply_splats(cloud, binary=True)
    assert ply_bytes.startswith(b"ply\nformat binary_little_endian 1.0\n")

    cloud_parsed = parse_ply_splats(ply_bytes)
    assert len(cloud_parsed) == 2
    np.testing.assert_allclose(cloud_parsed.positions, positions, atol=1e-4)
    np.testing.assert_allclose(cloud_parsed.scales, scales, rtol=0.05)
    np.testing.assert_allclose(cloud_parsed.colors, colors, atol=3)
    np.testing.assert_allclose(cloud_parsed.opacities, opacities, atol=0.03)


def test_ply_splats_ascii_roundtrip():
    """Verify 3DGS ASCII PLY format export and parsing."""
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    scales = np.array([[0.04, 0.04, 0.04]], dtype=np.float32)
    rotations = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[180, 210, 240]], dtype=np.uint8)
    opacities = np.array([0.9], dtype=np.float32)

    cloud = GaussianSplatCloud(positions, scales, rotations, colors, opacities)
    ply_ascii = export_ply_splats(cloud, binary=False)
    assert b"format ascii 1.0" in ply_ascii

    cloud_parsed = parse_ply_splats(ply_ascii)
    assert len(cloud_parsed) == 1
    np.testing.assert_allclose(cloud_parsed.positions, positions, atol=1e-4)
    np.testing.assert_allclose(cloud_parsed.colors, colors, atol=3)


def test_generate_splats_from_points():
    """Verify generating splat primitives from 3D points and surface normals."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    cloud = generate_splats_from_points(pts, colors=colors, normals=normals, default_radius=0.05)
    assert len(cloud) == 3
    # Check anisotropic tangent disk scaling for normals
    assert cloud.scales[0, 2] < cloud.scales[0, 0]
    # Check normalized quaternions
    norms = np.linalg.norm(cloud.rotations, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_decimate_splats():
    """Verify spatial decimation of splats for LOD streaming."""
    pts = np.random.uniform(-5.0, 5.0, size=(100, 3)).astype(np.float32)
    cloud = generate_splats_from_points(pts, default_radius=0.03)
    assert len(cloud) == 100

    dec_50 = decimate_splats(cloud, target_count=50)
    assert len(dec_50) == 50
    # Decimation should adjust scale to maintain volumetric coverage
    assert dec_50.scales[0, 0] > cloud.scales[0, 0]

    dec_10 = decimate_splats(cloud, target_count=10)
    assert len(dec_10) == 10

    # Decimating to higher than count returns original
    dec_200 = decimate_splats(cloud, target_count=200)
    assert len(dec_200) == 100


@pytest.mark.asyncio
async def test_splats_api_endpoints():
    """Verify REST API endpoints: metadata, upload, stream, and LOD decimation."""
    client = TestClient(app)

    # 1. Setup a World in DB and WorldStore
    store = get_store()
    world_ir = WorldIR(id=f"wld_splat_{uuid.uuid4().hex[:8]}")
    sv = store.save_version(world_ir, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        w = World(
            id=world_ir.id,
            name="Splat Testing Space",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(w)
        # Add version mirror row
        v_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            report={},
        )
        session.add(v_row)
        await session.commit()

    # 2. GET /api/worlds/{id}/splats before splats are uploaded
    res_meta0 = client.get(f"/api/worlds/{world_ir.id}/splats")
    assert res_meta0.status_code == 200
    m0 = res_meta0.json()
    assert m0["available"] is False
    assert m0["splat_count"] == 0

    # 3. Create synthetic splat binary and upload via POST /api/worlds/{id}/splats/upload
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[220, 50, 50], [50, 220, 50], [50, 50, 220]], dtype=np.uint8)
    sample_cloud = generate_splats_from_points(pts, colors=colors)
    splat_bytes = export_splat_binary(sample_cloud)

    upload_file = io.BytesIO(splat_bytes)
    res_up = client.post(
        f"/api/worlds/{world_ir.id}/splats/upload",
        files={"file": ("model.splat", upload_file, "application/octet-stream")},
    )
    assert res_up.status_code == 200
    up_data = res_up.json()
    assert up_data["success"] is True
    assert up_data["splat_count"] == 3
    assert up_data["byte_size"] == 3 * 32
    assert "artifact_uri" in up_data

    # 4. GET /api/worlds/{id}/splats after upload
    res_meta1 = client.get(f"/api/worlds/{world_ir.id}/splats")
    assert res_meta1.status_code == 200
    m1 = res_meta1.json()
    assert m1["available"] is True
    assert m1["splat_count"] == 3
    assert m1["byte_size"] == 96
    assert m1["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert m1["bounds"]["max"] == [2.0, 1.0, 1.0]

    # 5. GET /api/worlds/{id}/splats/stream?format=splat
    res_stream_splat = client.get(f"/api/worlds/{world_ir.id}/splats/stream?format=splat")
    assert res_stream_splat.status_code == 200
    assert len(res_stream_splat.content) == 96
    assert res_stream_splat.headers["x-splat-count"] == "3"
    assert res_stream_splat.headers["x-splat-format"] == "splat"

    # 6. GET /api/worlds/{id}/splats/stream?format=ply
    res_stream_ply = client.get(f"/api/worlds/{world_ir.id}/splats/stream?format=ply")
    assert res_stream_ply.status_code == 200
    assert res_stream_ply.content.startswith(b"ply\n")
    assert res_stream_ply.headers["x-splat-format"] == "ply"

    # 7. GET /api/worlds/{id}/splats/stream?max_points=2
    res_stream_dec = client.get(f"/api/worlds/{world_ir.id}/splats/stream?format=splat&max_points=2")
    assert res_stream_dec.status_code == 200
    assert len(res_stream_dec.content) == 2 * 32
    assert res_stream_dec.headers["x-splat-count"] == "2"

    # 8. Upload 3DGS PLY format
    ply_bytes = export_ply_splats(sample_cloud, binary=True)
    res_up_ply = client.post(
        f"/api/worlds/{world_ir.id}/splats/upload",
        files={"file": ("model.ply", io.BytesIO(ply_bytes), "application/octet-stream")},
    )
    assert res_up_ply.status_code == 200
    assert res_up_ply.json()["splat_count"] == 3
