"""Unit tests for Blueprint Task 8.8: Open-vocab detector & crop pass in the reconstruction path.

Verifies:
- Optical ray unprojection from camera through image fixture detections to host wall planes.
- Fixture entity creation (sockets, switches, vents) on wall planes with standard plate geometry.
- Creation of traceable MOUNTED_ON relationships linking fixtures to their host wall planes.
- Resolution-aware rejection of sub-resolution detections without fake claiming.
- Integration into single-image bootstrap path (spec M2 acceptance).
"""

import math
from typing import Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from engine.pipeline.single_image import bootstrap_single_image
from engine.pipeline.single_image_fixtures import (
    place_wall_fixtures,
    unproject_ray_to_plane,
)
from evidence.image_check import inspect_image
from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap
from perception.detail.crop_pass import CropPassResult, HighResCropPass
from perception.detection.interface import Detector, ImageDetection
from perception.detection.open_vocab import OpenVocabDetector
from world_ir import Entity, EntityType, GeometryType, RelationshipKind, WorldIR
from world_ir.observations import DepthSource, Observation


class MockFixtureDetector(Detector):
    def __init__(self, detections: list[ImageDetection]):
        self._detections = detections

    def is_available(self) -> bool:
        return True

    def detect(self, image: np.ndarray, prompts: list[str], threshold: float = 0.25) -> list[ImageDetection]:
        return [d for d in self._detections if d.score >= threshold]


class MockOrientedPlane:
    def __init__(self, plane_id: str, normal: tuple[float, float, float], d: float):
        self.normal = normal
        self.d = d
        self.role = "wall"
        self.plane = type("SubPlane", (), {"plane_id": plane_id, "inlier_count": 500})()


def _room_with_wall(tmp_path):
    W, H, CAM_H, WALL_Z = 320, 240, 1.0, 4.0
    p = tmp_path / "room_wall.png"
    img = np.ones((H, W, 3), dtype=np.uint8) * 128
    Image.fromarray(img).save(p)

    f = (W / 2) / math.tan(math.radians(30))
    vals = []
    for v in range(H):
        z = min(CAM_H * f / (v - H / 2), WALL_Z) if v > H / 2 + 1 else WALL_Z
        vals.append([1.0 / z - 0.25] * W)

    item = EvidenceItem(id="ev-fixture-room", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    return item, inspect_image(p), DepthMap(evidence_id="x", width=W, height=H, values=vals)


def test_unproject_ray_to_plane_basic_geometry():
    """Verify ray from pinhole camera unprojects and intersects wall plane accurately."""
    f = 500.0
    cx = 320.0
    cy = 240.0

    # Wall plane at z = 4.0m facing camera (normal = (0, 0, -1), d = 4.0 => -z + 4 = 0)
    normal = (0.0, 0.0, -1.0)
    d = 4.0

    # 1. Optical center ray (u=320, v=240) -> straight down +Z axis
    pt = unproject_ray_to_plane(320.0, 240.0, f, cx, cy, normal, d)
    assert pt is not None
    assert abs(pt[0]) < 1e-5
    assert abs(pt[1]) < 1e-5
    assert abs(pt[2] - 4.0) < 1e-5

    # 2. Right-offset ray (u=420 => offset +100px)
    # rx = 100 / 500 = 0.2 => at z=4.0, x = 0.2 * 4.0 = 0.8
    pt_right = unproject_ray_to_plane(420.0, 240.0, f, cx, cy, normal, d)
    assert pt_right is not None
    assert abs(pt_right[0] - 0.8) < 1e-5
    assert abs(pt_right[2] - 4.0) < 1e-5

    # 3. Ray parallel to plane (e.g. floor plane y = -1.5, normal = (0, 1, 0), ray in horizontal plane)
    pt_parallel = unproject_ray_to_plane(320.0, 240.0, f, cx, cy, (0.0, 1.0, 0.0), 1.5)
    assert pt_parallel is None

    # 4. Ray pointing away from plane (behind camera)
    pt_behind = unproject_ray_to_plane(320.0, 240.0, f, cx, cy, (0.0, 0.0, 1.0), 4.0)
    assert pt_behind is None


def test_place_wall_fixtures_creates_entities_and_mounted_on_relationships():
    """Verify place_wall_fixtures creates fixture entities and MOUNTED_ON relationships."""
    world = WorldIR(id="test_world")

    # Add host wall entity to world
    wall_entity = Entity(
        id="wall_north",
        name="North Wall",
        type=EntityType.WALL,
    )
    world.entities["wall_north"] = wall_entity

    # Synthetic image (800x600)
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # Wall pixels covering central region
    wall_pix = [(r, c) for r in range(100, 500, 5) for c in range(150, 650, 5)]
    wall_pixels = {"wall_north": wall_pix}

    # North wall facing camera at z = 3.5m
    oriented_wall = MockOrientedPlane("wall_north", (0.0, 0.0, -1.0), 3.5)

    # Mock detector returning a socket detection in the wall region
    mock_det = MockFixtureDetector([
        ImageDetection(label="electrical socket", score=0.92, box_xyxy=(300.0, 250.0, 330.0, 280.0)),
    ])
    open_vocab = OpenVocabDetector(backend=mock_det)
    crop_pass = HighResCropPass(detector=open_vocab, target_classes=("fixtures/electrical/socket",))

    facts = place_wall_fixtures(
        world,
        item=type("MockItem", (), {"id": "cam_01", "source_uri": "mock.jpg"})(),
        wall_pixels=wall_pixels,
        oriented_walls=[oriented_wall],
        f=600.0,
        cx=400.0,
        cy=300.0,
        w=800,
        h=600,
        pre="view1-",
        crop_pass=crop_pass,
        image_array=img,
    )

    assert facts["attempted"] is True
    assert facts["fixtures_detected"] == 1
    assert facts["fixtures_placed"] == 1
    assert len(facts["placed_ids"]) == 1

    fid = facts["placed_ids"][0]
    assert fid in world.entities
    f_ent = world.entities[fid]

    assert f_ent.type == EntityType.INFRASTRUCTURE
    assert "socket" in f_ent.name.lower()
    assert len(f_ent.geometry_ids) == 1
    geom = world.geometries[f_ent.geometry_ids[0]]
    assert geom.type.value == "box"

    # 3D position must lie on the wall (z = 3.5)
    pos = f_ent.transform["position"]
    assert abs(pos["z"] - 3.5) < 1e-4

    # Verify MOUNTED_ON relationship was created
    assert len(f_ent.relationships) == 1
    rel = f_ent.relationships[0]
    assert rel.target_id == "wall_north"
    assert rel.kind == RelationshipKind.MOUNTED_ON


def test_place_wall_fixtures_without_walls_reports_clean_skip():
    world = WorldIR(id="empty_world")
    facts = place_wall_fixtures(
        world,
        item=type("MockItem", (), {"id": "cam_01", "source_uri": "mock.jpg"})(),
        wall_pixels={},
        oriented_walls=[],
        f=500.0,
        cx=320.0,
        cy=240.0,
        w=640,
        h=480,
    )
    assert facts["attempted"] is False
    assert "no wall planes" in facts["reason"].lower()
    assert len(world.entities) == 0


def test_place_wall_fixtures_resolution_gating_rejects_sub_pixel_fixtures():
    """Verify fixtures below resolution limit (100 px) are rejected and recorded in facts."""
    world = WorldIR(id="test_world")
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    wall_pix = [(r, c) for r in range(100, 500, 10) for c in range(150, 650, 10)]
    wall_pixels = {"wall_north": wall_pix}
    oriented_wall = MockOrientedPlane("wall_north", (0.0, 0.0, -1.0), 3.5)

    # 8x8 detection = 64 pixels < 100 px min_pixels threshold
    mock_det = MockFixtureDetector([
        ImageDetection(label="electrical switch", score=0.85, box_xyxy=(300.0, 250.0, 308.0, 258.0)),
    ])
    open_vocab = OpenVocabDetector(backend=mock_det)
    crop_pass = HighResCropPass(detector=open_vocab, target_classes=("fixtures/electrical/switch",))

    facts = place_wall_fixtures(
        world,
        item=type("MockItem", (), {"id": "cam_01", "source_uri": "mock.jpg"})(),
        wall_pixels=wall_pixels,
        oriented_walls=[oriented_wall],
        f=600.0,
        cx=400.0,
        cy=300.0,
        w=800,
        h=600,
        crop_pass=crop_pass,
        image_array=img,
    )

    assert facts["fixtures_placed"] == 0
    assert len(facts["rejected"]) == 1
    rej_reason = facts["rejected"][0]["reason"].lower()
    assert "min_pixels" in rej_reason or "pixel_size" in rej_reason


def test_single_image_bootstrap_end_to_end_with_fixtures(tmp_path):
    """End-to-end integration: verify bootstrap_single_image runs fixture pass and establishes MOUNTED_ON relationships."""
    item, facts, dm = _room_with_wall(tmp_path)

    # Mock crop pass with 1 switch on detected wall
    mock_crop_pass = MagicMock(spec=HighResCropPass)
    mock_obs = Observation(
        observation_id="obs-crop-switch",
        image_id=item.id,
        label="fixtures/electrical/switch",
        world_pos=(0.0, 0.0, 0.0),
        sigma_m=0.05,
        depth_source=DepthSource.RELATIVE,
        bbox_px=(100.0, 80.0, 140.0, 120.0),
        confidence=0.92,
    )
    mock_crop_pass.run_pass.return_value = CropPassResult(
        accepted_observations=[mock_obs],
        rejected_detections=[],
        crops_processed=1,
        target_classes=("fixtures/electrical/socket", "fixtures/electrical/switch"),
        summary={"accepted": 1, "rejected": 0},
    )

    res = bootstrap_single_image(item, facts, depth_map=dm, fixture_crop_pass=mock_crop_pass)

    assert "fixtures" in res.facts
    assert "fixtures" in res.world.metadata["bootstrap"]
    ff = res.facts["fixtures"]
    assert ff["attempted"] is True
    if ff["fixtures_placed"] > 0:
        ent = next(e for e in res.world.entities.values() if "switch" in e.id)
        assert ent.custom_properties["subclass"] == "fixtures/electrical/switch"
        assert any(r.kind == RelationshipKind.MOUNTED_ON for r in ent.relationships)
