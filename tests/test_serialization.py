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


# ===== Canonical V1 WorldIR round-trip (regression for the save/load P0) =====
#
# `save_world`/`load_world` used to be implemented only against the legacy
# `world_ir.world.WorldIR` class above. A canonical `world_ir.world_v1.WorldIR`
# (what WorldRuntime, the compiler, WorldStore, exporters and the SDK all
# actually hold) would *save* successfully -- both classes expose to_dict()
# -- but *loading* it back crashed: V1 serializes `entities` as an id-keyed
# dict (legacy's loader iterates it as a list of entity dicts and blows up
# on the first key) and writes the frame under `coordinate_frame` (legacy
# reads `coordinate_system`, so it would have silently defaulted to
# Frame.WORLD even if the crash hadn't happened first).
#
# These tests fail against the pre-fix implementation (TypeError on load)
# and pass against the corrected one.

from provenance import Provenance as ProvenanceV1, Uncertainty
from world_ir.schema_v1 import (
    CausalRelation,
    Entity as EntityV1,
    EntityType,
    Geometry,
    GeometryType,
    Relationship as RelationshipV1,
    RelationshipKind,
    TemporalEvent,
)
from world_ir.world_v1 import Branch, WorldIR as WorldIRV1


def _sample_world_v1() -> WorldIRV1:
    world = WorldIRV1(
        id="world_v1_demo",
        name="v1 demo",
        version=2,
        coordinate_frame=Frame.ENU,  # non-default frame -- exercises the
        # coordinate_system/coordinate_frame key mismatch directly
        global_provenance=ProvenanceV1.RECONSTRUCTED,
        global_confidence=0.77,
        global_uncertainty=Uncertainty(confidence=0.6, note="multi-view triangulation"),
    )

    world.geometries["geom_wall"] = Geometry(
        id="geom_wall",
        type=GeometryType.MESH,
        vertex_count=48,
        triangle_count=80,
        data_uri="artifact://geom_wall.glb",
        data_hash="deadbeef",
        provenance=ProvenanceV1.OBSERVED,
        confidence=0.95,
    )

    world.entities["building_01"] = EntityV1(
        id="building_01",
        type=EntityType.BUILDING,
        transform=Transform.identity(Frame.ENU, timestamp=1.5).to_dict(),
        geometry_ids=["geom_wall"],
        semantic_labels=["reconstructed-primary"],
        provenance=ProvenanceV1.RECONSTRUCTED,
        confidence=0.82,
        uncertainty=Uncertainty(confidence=0.7, note="sparse cloud"),
        custom_properties={"floor_count": 3},
    )
    world.entities["window_01"] = EntityV1(
        id="window_01",
        type=EntityType.WINDOW,
        relationships=[
            RelationshipV1(kind=RelationshipKind.PART_OF, target_id="building_01", confidence=1.0)
        ],
        semantic_labels=["glass"],
        provenance=ProvenanceV1.OBSERVED,
        confidence=0.9,
    )

    world.temporal_events["evt_1"] = TemporalEvent(
        id="evt_1", entity_id="building_01", timestamp=3.0,
        provenance=ProvenanceV1.OBSERVED,
    )
    world.causal_relations.append(
        CausalRelation(cause_id="evt_1", effect_id="evt_1", relationship_type="self-test")
    )
    world.branches[world.main_branch_id] = Branch(id=world.main_branch_id, name="main")
    world.metadata["source"] = "test_serialization"

    return world


def test_v1_roundtrip_preserves_semantic_content(tmp_path):
    world = _sample_world_v1()
    package_dir = save_world(world, tmp_path / "world_v1_demo")
    restored = load_world(package_dir)

    assert isinstance(restored, WorldIRV1)

    # Entity identity
    assert set(restored.entities.keys()) == {"building_01", "window_01"}

    # Geometry identity and data
    assert set(restored.geometries.keys()) == {"geom_wall"}
    restored_geom = restored.geometries["geom_wall"]
    assert restored_geom.vertex_count == 48
    assert restored_geom.triangle_count == 80
    assert restored_geom.data_hash == "deadbeef"
    assert restored_geom.provenance == ProvenanceV1.OBSERVED
    assert restored_geom.confidence == 0.95

    # Coordinate frame -- the field legacy's loader would have silently
    # defaulted to Frame.WORLD
    assert restored.coordinate_frame == Frame.ENU

    # Transform (externalized dict on the entity)
    restored_building = restored.entities["building_01"]
    assert restored_building.transform["source_frame"] == "ENU"
    assert restored_building.transform["target_frame"] == "ENU"
    assert restored_building.transform["timestamp"] == 1.5

    # Semantics
    assert restored_building.semantic_labels == ["reconstructed-primary"]
    assert restored_building.custom_properties == {"floor_count": 3}
    assert restored.entities["window_01"].relationships[0].target_id == "building_01"
    assert restored.entities["window_01"].relationships[0].kind == RelationshipKind.PART_OF

    # Provenance and uncertainty -- both entity-level and world-level,
    # must not collapse to a fabricated default
    assert restored_building.provenance == ProvenanceV1.RECONSTRUCTED
    assert restored_building.confidence == 0.82
    assert restored_building.uncertainty.confidence == 0.7
    assert restored_building.uncertainty.note == "sparse cloud"
    assert restored.global_provenance == ProvenanceV1.RECONSTRUCTED
    assert restored.global_confidence == 0.77
    assert restored.global_uncertainty.confidence == 0.6

    # Metadata and everything else, by full semantic equality
    assert restored.metadata == {"source": "test_serialization"}
    assert restored.to_dict() == world.to_dict()


def test_v1_manifest_records_ir_kind(tmp_path):
    world = _sample_world_v1()
    package_dir = save_world(world, tmp_path / "world_v1_demo")
    manifest = json.loads((package_dir / MANIFEST_FILENAME).read_text())
    assert manifest["ir_kind"] == "v1"


def test_legacy_package_without_ir_kind_still_loads_as_legacy(tmp_path):
    """Packages written before this fix have no `ir_kind` manifest field.
    They were legacy-only in practice (a V1 save+load never survived), so
    the loader must default a missing `ir_kind` to legacy, not guess V1."""
    world = _sample_world()
    package_dir = save_world(world, tmp_path / "world_demo_building")
    manifest_path = package_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["ir_kind"]
    manifest_path.write_text(json.dumps(manifest))

    restored = load_world(package_dir)
    assert type(restored) is type(world)
    assert restored.to_dict() == world.to_dict()


def test_load_rejects_manifest_ir_kind_mismatch(tmp_path):
    """A manifest claiming 'legacy' while world.ir is actually V1 content
    (or vice versa) is corruption, not something to silently resolve."""
    world = _sample_world_v1()
    package_dir = save_world(world, tmp_path / "world_v1_demo")
    manifest_path = package_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["ir_kind"] = "legacy"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(WorldPackageError):
        load_world(package_dir)
