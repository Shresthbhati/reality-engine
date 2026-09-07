import json

import pytest

from provenance import Provenance
from world_ir.coordinates import Frame, Transform
from world_ir.entity import Entity, Relationship
from world_ir.serialization import (
    IR_FILENAME,
    MANIFEST_FILENAME,
    WORLD_SAVE_FORMAT_VERSION,
    WorldFormatVersionError,
    WorldPackageError,
    load_world,
    save_world,
)
from world_ir.world import WorldIR


def _sample_world() -> WorldIR:
    world = WorldIR(id="world_demo_building", version=3, coordinate_system=Frame.WORLD)
    world.entities.add(Entity(
        id="building_01",
        type="building",
        transform=Transform.identity(Frame.WORLD, timestamp=0.0),
        provenance=Provenance.RECONSTRUCTED,
    ))
    world.entities.add(Entity(
        id="window_01",
        type="window",
        provenance=Provenance.OBSERVED,
        relationships=[Relationship(kind="attached_to", target_id="building_01")],
        semantic_labels=["glass"],
    ))
    return world


def test_save_creates_manifest_and_ir(tmp_path):
    world = _sample_world()
    package_dir = save_world(world, tmp_path / "world_demo_building")
    assert (package_dir / MANIFEST_FILENAME).exists()
    assert (package_dir / IR_FILENAME).exists()

    manifest = json.loads((package_dir / MANIFEST_FILENAME).read_text())
    assert manifest["format_version"] == WORLD_SAVE_FORMAT_VERSION
    assert manifest["world_id"] == "world_demo_building"
    assert manifest["entity_count"] == 2


def test_roundtrip_preserves_world(tmp_path):
    world = _sample_world()
    package_dir = save_world(world, tmp_path / "world_demo_building")
    restored = load_world(package_dir)

    assert restored.id == world.id
    assert restored.version == world.version
    assert restored.coordinate_system == world.coordinate_system
    assert len(restored.entities) == len(world.entities)
    assert restored.entities.get("window_01").relationships[0].target_id == "building_01"
    assert restored.to_dict() == world.to_dict()


def test_load_missing_package_raises(tmp_path):
    with pytest.raises(WorldPackageError):
        load_world(tmp_path / "does_not_exist")


def test_load_rejects_future_format_version(tmp_path):
    world = _sample_world()
    package_dir = save_world(world, tmp_path / "world_demo_building")
    manifest_path = package_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = WORLD_SAVE_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(WorldFormatVersionError):
        load_world(package_dir)


def test_load_rejects_id_mismatch_between_manifest_and_ir(tmp_path):
    world = _sample_world()
    package_dir = save_world(world, tmp_path / "world_demo_building")
    manifest_path = package_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["world_id"] = "some_other_world"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(WorldPackageError):
        load_world(package_dir)
