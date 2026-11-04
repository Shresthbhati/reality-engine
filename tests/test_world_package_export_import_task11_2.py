"""Unit and API tests for World Package Export/Import (Blueprint Task 11.2).

Done when: export -> import yields an identical world.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from provenance import Provenance
from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geo import GeoAnchor
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Observation,
    PhysicalProperties,
    SurfaceFinish,
    Vector3,
)
from world_ir.serialization import (
    IR_FILENAME,
    MANIFEST_FILENAME,
    WORLD_SAVE_FORMAT_VERSION,
    WorldFormatVersionError,
    WorldPackageError,
    export_world_package_zip,
    import_world_package_zip,
)
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR


def _create_rich_test_world() -> WorldIR:
    world = WorldIR(id="world-package-rich", name="Architectural Suite")
    world.version = 3
    world.created_at = 1700000000.0

    # Add geo anchor
    world.geo_anchor = GeoAnchor(
        latitude=51.5074,
        longitude=-0.1278,
        altitude_m=35.0,
        heading_deg=45.0,
        crs="EPSG:4326",
        accuracy_m=0.5,
        source="rtk_gnss",
    )

    # Add materials
    mat = Material(
        id="mat-oak",
        name="Oak Wood",
        class_name="wood",
        surface_finish=SurfaceFinish.SMOOTH,
        color_rgb=(0.6, 0.4, 0.2),
        properties=PhysicalProperties(friction_coefficient=0.45, restitution=0.2),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.90,
    )
    world.materials["mat-oak"] = mat

    # Add geometries
    floor_geom = Geometry(
        id="geom-floor",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(8.0, 0.2, 10.0),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    world.geometries["geom-floor"] = floor_geom

    wall_geom = Geometry(
        id="geom-wall",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(8.0, 2.8, 0.2),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.92,
    )
    world.geometries["geom-wall"] = wall_geom

    # Add entities
    floor_ent = Entity(
        id="ent-floor",
        type=EntityType.FLOOR,
        name="Hardwood Floor",
        geometry_ids=["geom-floor"],
        material_ids=["mat-oak"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
        statement_state=StatementState.DERIVED,
        observations=[Observation(id="obs-floor-1", sensor_type="camera")],
        custom_properties={"evidence_state": "OBSERVED", "area_m2": 80.0},
    )
    world.entities["ent-floor"] = floor_ent

    wall_ent = Entity(
        id="ent-wall",
        type=EntityType.WALL,
        name="North Wall",
        geometry_ids=["geom-wall"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.92,
        statement_state=StatementState.DERIVED,
        observations=[Observation(id="obs-wall-1", sensor_type="camera")],
        custom_properties={"evidence_state": "OBSERVED", "thickness_m": 0.2},
    )
    world.entities["ent-wall"] = wall_ent

    return world


def test_export_import_world_package_zip_yields_identical_world(tmp_path: Path):
    orig = _create_rich_test_world()

    # 1. Export to in-memory bytes
    zip_bytes = export_world_package_zip(orig)
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0

    # 2. Inspect ZIP archive structure
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        names = zf.namelist()
        assert MANIFEST_FILENAME in names
        assert IR_FILENAME in names

        manifest = json.loads(zf.read(MANIFEST_FILENAME).decode("utf-8"))
        assert manifest["format_version"] == WORLD_SAVE_FORMAT_VERSION
        assert manifest["world_id"] == orig.id
        assert manifest["world_version"] == orig.version
        assert manifest["entity_count"] == 2
        assert manifest["geometry_count"] == 2
        assert manifest["material_count"] == 1
        assert manifest["geo_anchor"]["latitude"] == pytest.approx(51.5074)

        ir_bytes = zf.read(IR_FILENAME)
        actual_hash = hashlib.sha256(ir_bytes).hexdigest()
        assert manifest["ir_sha256"] == actual_hash

    # 3. Export to file path
    file_path = tmp_path / "world_package.zip"
    out_path = export_world_package_zip(orig, output=file_path)
    assert out_path == file_path
    assert file_path.exists()

    # 4. Import back from bytes
    imported_from_bytes = import_world_package_zip(zip_bytes)
    assert imported_from_bytes.to_dict() == orig.to_dict()

    # 5. Import back from file path
    imported_from_file = import_world_package_zip(file_path)
    assert imported_from_file.to_dict() == orig.to_dict()


def test_export_import_world_package_zip_with_artifacts():
    orig = _create_rich_test_world()
    art_store_src = MemoryArtifactStore()

    # Create dummy mesh payload
    mesh_payload = b"FAKE_BINARY_MESH_DATA_REALITY_ENGINE_12345"
    digest = hashlib.sha256(mesh_payload).hexdigest()
    uri, _ = art_store_src.put(mesh_payload)

    # Attach artifact uri to geometry
    orig.geometries["geom-floor"].data_uri = uri
    orig.geometries["geom-floor"].data_hash = digest

    # Export with artifact store
    zip_bytes = export_world_package_zip(orig, artifact_store=art_store_src)

    # Verify artifact is enclosed in zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        assert f"artifacts/{digest}" in zf.namelist()
        assert zf.read(f"artifacts/{digest}") == mesh_payload

    # Import into a fresh artifact store
    art_store_dst = MemoryArtifactStore()
    imported = import_world_package_zip(zip_bytes, artifact_store=art_store_dst)

    # Assert world identity and artifact restoration
    assert imported.to_dict() == orig.to_dict()
    assert art_store_dst.get(uri) == mesh_payload


def test_package_corruption_and_checksum_verification():
    orig = _create_rich_test_world()
    zip_bytes = export_world_package_zip(orig)

    # 1. Invalid zip bytes
    with pytest.raises(WorldPackageError, match="Invalid or corrupted zip package"):
        import_world_package_zip(b"NOT_A_ZIP_FILE")

    # 2. Missing manifest
    buf_no_manifest = io.BytesIO()
    with zipfile.ZipFile(buf_no_manifest, mode="w") as zf:
        zf.writestr(IR_FILENAME, b'{"id": "test"}')
    with pytest.raises(WorldPackageError, match="missing required world.manifest"):
        import_world_package_zip(buf_no_manifest.getvalue())

    # 3. Missing world.ir
    buf_no_ir = io.BytesIO()
    with zipfile.ZipFile(buf_no_ir, mode="w") as zf:
        zf.writestr(MANIFEST_FILENAME, json.dumps({"format_version": 1, "world_id": "test"}))
    with pytest.raises(WorldPackageError, match="missing required world.ir"):
        import_world_package_zip(buf_no_ir.getvalue())

    # 4. Tampered world.ir with checksum mismatch
    buf_tampered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as src_zf:
        manifest_data = src_zf.read(MANIFEST_FILENAME)
    with zipfile.ZipFile(buf_tampered, mode="w") as dst_zf:
        dst_zf.writestr(MANIFEST_FILENAME, manifest_data)
        # Write altered IR bytes
        dst_zf.writestr(IR_FILENAME, b'{"id": "world-package-rich", "tampered": true}')

    with pytest.raises(WorldPackageError, match="Checksum mismatch"):
        import_world_package_zip(buf_tampered.getvalue())

    # 5. Unsupported format version
    buf_bad_ver = io.BytesIO()
    with zipfile.ZipFile(buf_bad_ver, mode="w") as zf:
        zf.writestr(MANIFEST_FILENAME, json.dumps({"format_version": 999, "world_id": "test"}))
        zf.writestr(IR_FILENAME, b'{"id": "test"}')
    with pytest.raises(WorldFormatVersionError, match="unsupported world.manifest format_version 999"):
        import_world_package_zip(buf_bad_ver.getvalue())


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as c:
        yield c


def test_api_world_package_export_and_import(client):
    orig = _create_rich_test_world()

    # Create world and commit initial version via API/db
    wid = client.post("/api/worlds", json={"name": orig.name}).json()["id"]

    from apps.api import worldstore_service
    import apps.api.db as db_mod
    import asyncio

    async def seed():
        async with db_mod.get_sessionmaker()() as db:
            await worldstore_service.commit_version(db, world_id=wid, world=orig, parent=None)

    asyncio.run(seed())

    # 1. Test GET /api/worlds/{id}/export/package
    res_export = client.get(f"/api/worlds/{wid}/export/package")
    assert res_export.status_code == 200
    assert res_export.headers["content-type"] == "application/zip"
    assert "attachment" in res_export.headers.get("content-disposition", "")
    zip_bytes = res_export.content

    # Validate exported zip structure
    imported_world = import_world_package_zip(zip_bytes)
    assert imported_world.name == orig.name
    assert len(imported_world.entities) == len(orig.entities)

    # 2. Test POST /api/worlds/import/package
    files = {"file": ("my_world_package.zip", zip_bytes, "application/zip")}
    res_import = client.post("/api/worlds/import/package", files=files)
    assert res_import.status_code == 201
    import_data = res_import.json()

    assert "id" in import_data
    assert import_data["name"] == orig.name
    assert import_data["entity_count"] == 2
    assert import_data["status"] == "imported"

    # Verify imported world is retrievable via API
    get_res = client.get(f"/api/worlds/{import_data['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == orig.name
