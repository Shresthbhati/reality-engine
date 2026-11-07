"""Tests for Task 13.3: Extended Capture Ingestion.

Covers:
- HEIC/HEIF mobile capture parsing (ISOBMFF ispe spatial extents without external C extensions)
- 360° / Equirectangular panorama detection (2:1 aspect ratio, projection metadata)
- Point cloud (PLY, PCD, LAS) ingestion and header parsing
- Mesh (OBJ, glTF, GLB) ingestion and header parsing
- Pre-flight upload checking via POST /api/uploads/check
- 3D model ingestion via POST /api/reconstructions
"""

from __future__ import annotations

import asyncio
import importlib
import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from apps.api.models import Evidence, Session, World
from evidence.image_check import _parse_heic_container, inspect_image


def _build_synthetic_heic(width: int = 1920, height: int = 1080) -> bytes:
    """Build a minimal valid ISOBMFF HEIC container with an 'ispe' box."""
    # ftyp box
    ftyp = struct.pack(">I4s4sI4s4s", 28, b"ftyp", b"heic", 0, b"mif1", b"heic")
    # ispe box: length 20 (4 len, 4 type, 4 flags/ver, 4 w, 4 h)
    ispe = struct.pack(">I4sIIII", 20, b"ispe", 0, width, height, 0)
    return ftyp + ispe


def _build_synthetic_360_image(width: int = 1000, height: int = 500, seed: int = 42) -> bytes:
    """Create a 2:1 aspect ratio JPEG image (equirectangular panorama) with natural noise."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_standard_image(width: int = 800, height: int = 600, seed: int = 99) -> bytes:
    """Create a standard aspect ratio (4:3) JPEG image with natural noise."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_synthetic_ply(vertex_count: int = 3) -> bytes:
    """Build a minimal ASCII PLY point cloud."""
    return (
        f"ply\n"
        f"format ascii 1.0\n"
        f"element vertex {vertex_count}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"end_header\n"
        f"0.0 0.0 0.0\n"
        f"1.0 0.0 0.0\n"
        f"0.0 1.0 0.0\n"
    ).encode("ascii")


def _build_synthetic_pcd(point_count: int = 4) -> bytes:
    """Build a minimal ASCII PCD point cloud."""
    return (
        f"# .PCD v.7 - Point Cloud Data file format\n"
        f"VERSION .7\n"
        f"FIELDS x y z\n"
        f"SIZE 4 4 4\n"
        f"TYPE F F F\n"
        f"COUNT 1 1 1\n"
        f"WIDTH {point_count}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        f"DATA ascii\n"
        f"0.0 0.0 0.0\n"
        f"1.0 0.0 0.0\n"
        f"0.0 1.0 0.0\n"
        f"1.0 1.0 1.0\n"
    ).encode("ascii")


def _build_synthetic_obj() -> bytes:
    """Build a minimal Wavefront OBJ mesh with 3 vertices and 1 face."""
    return b"# minimal OBJ\nv 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n"


def _build_synthetic_glb() -> bytes:
    """Build a minimal binary glTF (GLB) header."""
    magic = b"glTF"
    version = struct.pack("<I", 2)
    length = struct.pack("<I", 20)
    chunk_len = struct.pack("<I", 8)
    chunk_type = b"JSON"
    chunk_data = b"{}" + b" " * 6
    return magic + version + length + chunk_len + chunk_type + chunk_data


@pytest.fixture
def client_env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))
    monkeypatch.setenv("REALITY_API_TOKEN", "test-token")

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    asyncio.run(db_mod.init_db())

    with TestClient(main_mod.app, headers={"Authorization": "Bearer test-token"}) as client:
        yield client, db_mod


def test_heic_container_parser(tmp_path):
    """Test HEIC container ispe spatial extents extraction without external dependencies."""
    data = _build_synthetic_heic(2560, 1440)
    w, h, exif = _parse_heic_container(data)
    assert w == 2560
    assert h == 1440
    assert isinstance(exif, dict)

    heic_file = tmp_path / "capture.heic"
    heic_file.write_bytes(data)

    facts = inspect_image(heic_file)
    assert facts.ok is True
    assert facts.width == 2560
    assert facts.height == 1440
    assert facts.evidence_class == "photograph"
    assert "ISOBMFF ispe" in facts.class_basis
    assert facts.geometry_eligible is True


def test_heic_container_guards(tmp_path):
    """Test HEIC container validation respects pixel cap and minimum dimensions."""
    # 1. Too small (below MIN_SIDE_PX: 64px)
    small_data = _build_synthetic_heic(50, 50)
    small_file = tmp_path / "small.heic"
    small_file.write_bytes(small_data)
    small_facts = inspect_image(small_file)
    assert small_facts.ok is False
    assert "too small" in (small_facts.reason or "")

    # 2. Huge (exceeds cap)
    huge_data = _build_synthetic_heic(50000, 50000)
    huge_file = tmp_path / "huge.heic"
    huge_file.write_bytes(huge_data)
    huge_facts = inspect_image(huge_file)
    assert huge_facts.ok is False
    assert "decoded-pixel cap" in (huge_facts.reason or "")


def test_360_equirectangular_detection(tmp_path):
    """Test that 2:1 aspect ratio images are recognized as 360 equirectangular panoramas."""
    pano_data = _build_synthetic_360_image(1200, 600)
    pano_file = tmp_path / "pano_360.jpg"
    pano_file.write_bytes(pano_data)

    pano_facts = inspect_image(pano_file)
    assert pano_facts.ok is True
    assert pano_facts.is_360 is True
    assert pano_facts.projection == "equirectangular"

    std_data = _build_standard_image(800, 600)
    std_file = tmp_path / "standard.jpg"
    std_file.write_bytes(std_data)

    std_facts = inspect_image(std_file)
    assert std_facts.ok is True
    assert std_facts.is_360 is False
    assert std_facts.projection is None


def test_uploads_check_with_point_clouds_meshes_and_360(client_env):
    """POST /api/uploads/check surfaces point cloud, mesh and 360 metadata."""
    client, _ = client_env

    pano_data = _build_synthetic_360_image(1000, 500)
    ply_data = _build_synthetic_ply(vertex_count=42)
    pcd_data = _build_synthetic_pcd(point_count=50)
    obj_data = _build_synthetic_obj()
    glb_data = _build_synthetic_glb()

    files = [
        ("files", ("room_pano.jpg", pano_data, "image/jpeg")),
        ("files", ("survey.ply", ply_data, "application/octet-stream")),
        ("files", ("lidar.pcd", pcd_data, "application/octet-stream")),
        ("files", ("furniture.obj", obj_data, "text/plain")),
        ("files", ("fixture.glb", glb_data, "model/gltf-binary")),
    ]

    r = client.post("/api/uploads/check", files=files)
    assert r.status_code == 200, r.text
    data = r.json()

    files_by_name = {f["name"]: f for f in data["files"]}

    # Check 360 panorama detection
    pano = files_by_name["room_pano.jpg"]
    assert pano["kind"] == "photo"
    assert pano["is_360"] is True
    assert pano["projection"] == "equirectangular"

    # Check point clouds
    ply = files_by_name["survey.ply"]
    assert ply["kind"] == "point_cloud"
    assert ply["point_cloud_format"] == "ply"
    assert ply["vertex_count"] == 42
    assert ply["decodable"] is True

    pcd = files_by_name["lidar.pcd"]
    assert pcd["kind"] == "point_cloud"
    assert pcd["point_cloud_format"] == "pcd"
    assert pcd["vertex_count"] == 50
    assert pcd["decodable"] is True

    # Check meshes
    obj = files_by_name["furniture.obj"]
    assert obj["kind"] == "mesh"
    assert obj["mesh_format"] == "obj"
    assert obj["vertex_count"] == 3
    assert obj["decodable"] is True

    glb = files_by_name["fixture.glb"]
    assert glb["kind"] == "mesh"
    assert glb["mesh_format"] == "glb"
    assert glb["decodable"] is True

    # Summary: all 5 are usable, none flagged as broken
    assert data["summary"]["usable"] == 5
    assert data["summary"]["flagged"] == 0


def test_reconstructions_point_cloud_and_mesh_direct_ingestion(client_env):
    """POST /api/reconstructions supports direct ingestion of point clouds and meshes without requiring photos."""
    client, db_mod = client_env

    ply_data = _build_synthetic_ply(vertex_count=10)
    obj_data = _build_synthetic_obj()

    files = [
        ("files", ("scan.ply", ply_data, "application/octet-stream")),
        ("files", ("cad.obj", obj_data, "text/plain")),
    ]

    r = client.post("/api/reconstructions", files=files, data={"name": "3D Survey Site"})
    assert r.status_code == 202, r.text
    res = r.json()

    assert res["world_id"] is not None
    assert res["session_id"] is not None
    # No photos were uploaded, so SfM reconstruct job is not required
    assert res["job_id"] is None
    assert len(res["evidence"]) == 2

    ev_types = {e["name"]: e["type"] for e in res["evidence"]}
    assert ev_types["scan.ply"] == "point_cloud"
    assert ev_types["cad.obj"] == "mesh"

    # Verify database state
    async def _verify():
        session_factory = db_mod.get_sessionmaker()
        async with session_factory() as db:
            session = await db.get(Session, res["session_id"])
            assert session is not None
            assert session.status == "completed"

            world = await db.get(World, res["world_id"])
            assert world is not None
            assert world.name == "3D Survey Site"

    asyncio.run(_verify())


def test_reconstructions_mixed_photos_and_3d_assets(client_env, monkeypatch):
    """POST /api/reconstructions handles combined photo + 3D capture, keeping all evidence and triggering reconstruction."""
    client, db_mod = client_env

    monkeypatch.setattr(
        "apps.api.jobs.enqueue_job",
        lambda *args, **kwargs: type("J", (), {"id": "job_reconstruct_123", "type": "reconstruct", "status": "queued"})()
    )

    pano_data = _build_synthetic_360_image(1000, 500)
    std_data = _build_standard_image(800, 600)
    glb_data = _build_synthetic_glb()

    files = [
        ("files", ("living_room_360.jpg", pano_data, "image/jpeg")),
        ("files", ("detail_shot.jpg", std_data, "image/jpeg")),
        ("files", ("existing_mesh.glb", glb_data, "model/gltf-binary")),
    ]

    r = client.post("/api/reconstructions", files=files, data={"name": "Mixed Capture"})
    assert r.status_code == 202, r.text
    res = r.json()

    assert res["job_id"] == "job_reconstruct_123"
    assert len(res["evidence"]) == 3

    # Check evidence in db
    async def _verify():
        session_factory = db_mod.get_sessionmaker()
        async with session_factory() as db:
            session = await db.get(Session, res["session_id"])
            assert session is not None
            assert session.status == "processing"

    asyncio.run(_verify())
