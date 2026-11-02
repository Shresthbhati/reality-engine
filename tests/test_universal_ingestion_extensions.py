"""Universal Ingestion Extensions tests (Phase 2, Tasks 2.4, 2.7, 2.8, 2.9).

Covers:
- Point cloud ingestion (.ply, .pcd, .e57) with header inspection
- Mesh ingestion (.obj, .gltf, .glb)
- 360-degree / equirectangular image detection
- Capture-session grouping (temporal & spatial clustering)
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image

from evidence.packages import DeterministicPackageBuilder, CorruptEvidenceError
from evidence.importers import import_file, import_folder
from evidence.session import EvidenceKind, EvidenceItem
from evidence.session_grouping import (
    haversine_distance_m,
    group_evidence_by_session,
    SessionGroup,
)


def test_haversine_distance():
    # San Francisco to San Jose (~65-70 km)
    sf_lat, sf_lon = 37.7749, -122.4194
    sj_lat, sj_lon = 37.3382, -121.8863
    dist = haversine_distance_m(sf_lat, sf_lon, sj_lat, sj_lon)
    assert 60000.0 < dist < 80000.0


def test_session_grouping_by_time_and_gps():
    items = [
        # Session 1: Site A, morning
        EvidenceItem(id="ev-1", kind=EvidenceKind.PHOTO, source_uri="file:///a1.jpg", captured_at=1000.0,
                     metadata={"gps": (37.7749, -122.4194)}),
        EvidenceItem(id="ev-2", kind=EvidenceKind.PHOTO, source_uri="file:///a2.jpg", captured_at=1060.0,
                     metadata={"gps": (37.7750, -122.4195)}),
        # Session 2: Site A, afternoon (gap > 1800s)
        EvidenceItem(id="ev-3", kind=EvidenceKind.PHOTO, source_uri="file:///a3.jpg", captured_at=5000.0,
                     metadata={"gps": (37.7751, -122.4194)}),
        # Session 3: Site B, afternoon (distance > 500m)
        EvidenceItem(id="ev-4", kind=EvidenceKind.PHOTO, source_uri="file:///b1.jpg", captured_at=5100.0,
                     metadata={"gps": (37.7850, -122.4000)}),
    ]

    groups = group_evidence_by_session(items, max_time_gap_s=1800.0, max_distance_m=500.0)
    assert len(groups) == 3
    assert groups[0].item_count == 2
    assert groups[0].items[0].id == "ev-1"
    assert groups[0].items[1].id == "ev-2"
    assert groups[1].item_count == 1
    assert groups[1].items[0].id == "ev-3"
    assert groups[2].item_count == 1
    assert groups[2].items[0].id == "ev-4"

    d = groups[0].to_dict()
    assert d["group_id"] == "session-001"
    assert d["item_count"] == 2
    assert d["start_time"] == 1000.0
    assert d["end_time"] == 1060.0
    assert d["center_gps"] is not None


def test_equirectangular_360_image_detection(tmp_path: Path):
    # 2:1 ratio image = 360 equirectangular panorama
    img = Image.new("RGB", (200, 100), color=(100, 150, 200))
    path = tmp_path / "pano_360.jpg"
    img.save(path)

    builder = DeterministicPackageBuilder(seed="test")
    imported = import_file(builder, str(path))
    package = builder.build()
    asset = package.assets[imported.asset_id]

    assert asset.sensor_metadata.get("is_360") is True
    assert asset.sensor_metadata.get("projection") == "equirectangular"


def test_point_cloud_ply_ingestion(tmp_path: Path):
    ply_content = b"""ply
format ascii 1.0
element vertex 42
property float x
property float y
property float z
end_header
0 0 0
1 1 1
"""
    path = tmp_path / "cloud.ply"
    path.write_bytes(ply_content)

    builder = DeterministicPackageBuilder(seed="test")
    imported = import_file(builder, str(path))
    assert imported.kind == EvidenceKind.POINT_CLOUD

    package = builder.build()
    asset = package.assets[imported.asset_id]
    assert asset.kind == EvidenceKind.POINT_CLOUD
    assert asset.sensor_metadata.get("format") == "ply"
    assert asset.sensor_metadata.get("vertex_count") == 42


def test_point_cloud_pcd_ingestion(tmp_path: Path):
    pcd_content = b"""# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 15
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 15
DATA ascii
0 0 0
"""
    path = tmp_path / "cloud.pcd"
    path.write_bytes(pcd_content)

    builder = DeterministicPackageBuilder(seed="test")
    imported = import_file(builder, str(path))
    assert imported.kind == EvidenceKind.POINT_CLOUD

    package = builder.build()
    asset = package.assets[imported.asset_id]
    assert asset.sensor_metadata.get("point_count") == 15


def test_mesh_obj_ingestion(tmp_path: Path):
    obj_content = b"""# Test OBJ file
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""
    path = tmp_path / "model.obj"
    path.write_bytes(obj_content)

    builder = DeterministicPackageBuilder(seed="test")
    imported = import_file(builder, str(path))
    assert imported.kind == EvidenceKind.OTHER

    package = builder.build()
    asset = package.assets[imported.asset_id]
    assert asset.sensor_metadata.get("role") == "mesh"
    assert asset.sensor_metadata.get("mesh_format") == "obj"


def test_mesh_gltf_and_glb_ingestion(tmp_path: Path):
    gltf_content = b'{\n  "asset": {"version": "2.0"},\n  "scene": 0\n}'
    gltf_path = tmp_path / "scene.gltf"
    gltf_path.write_bytes(gltf_content)

    glb_content = b"glTF\x02\x00\x00\x00\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(glb_content)

    builder = DeterministicPackageBuilder(seed="test")
    report = import_folder(builder, str(tmp_path))
    assert len(report.imported) == 2
    assert all(a.kind == EvidenceKind.OTHER for a in report.imported)

    package = builder.build()
    formats = {a.sensor_metadata.get("mesh_format") for a in package.all_assets()}
    assert formats == {"gltf", "glb"}
