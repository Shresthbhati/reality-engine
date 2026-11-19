"""Tests for Universal Object, Appliance, Fixture, and Architectural Ontology."""


from world.ontology.universal_object_ontology import (
    ObjectCategory,
    SpatialRelationType,
    get_universal_ontology,
)


def test_ontology_category_and_class_population() -> None:
    ont = get_universal_ontology()

    chair = ont.get_class("chair")
    assert chair is not None
    assert chair.category == ObjectCategory.FURNITURE
    assert "seat" in chair.allowed_parts
    assert "legs" in chair.allowed_parts

    refrigerator = ont.get_class("refrigerator")
    assert refrigerator is not None
    assert refrigerator.category == ObjectCategory.APPLIANCE
    assert refrigerator.is_container is True
    assert "door" in refrigerator.allowed_parts

    socket = ont.get_class("electrical_socket")
    assert socket is not None
    assert socket.category == ObjectCategory.FIXTURE
    assert "faceplate" in socket.allowed_parts

    wall = ont.get_class("wall")
    assert wall is not None
    assert wall.category == ObjectCategory.BUILDING_ELEMENT
    assert wall.is_structural is True

    # Priority 4 fine-grained classes
    radiator = ont.get_class("radiator")
    assert radiator is not None
    assert radiator.category == ObjectCategory.FIXTURE
    assert "fins" in radiator.allowed_parts

    tree = ont.get_class("tree")
    assert tree is not None
    assert tree.category == ObjectCategory.URBAN_INFRASTRUCTURE
    assert "trunk" in tree.allowed_parts

    mug = ont.get_class("mug")
    assert mug is not None
    assert mug.category == ObjectCategory.EVERYDAY_OBJECT
    assert mug.is_container is True


def test_ontology_synonym_resolution() -> None:
    ont = get_universal_ontology()

    assert ont.resolve_class_name("desk") == "table"
    assert ont.resolve_class_name("couch") == "sofa"
    assert ont.resolve_class_name("fridge") == "refrigerator"
    assert ont.resolve_class_name("wall outlet") == "electrical_socket"
    assert ont.resolve_class_name("pavement") == "footpath"
    assert ont.resolve_class_name("unknown_xyz") is None


def test_ontology_part_validation() -> None:
    ont = get_universal_ontology()

    assert ont.is_valid_part("chair", "seat") is True
    assert ont.is_valid_part("chair", "legs") is True
    assert ont.is_valid_part("chair", "steering_wheel") is False

    assert ont.is_valid_part("refrigerator", "door") is True
    assert ont.is_valid_part("refrigerator", "exhaust_grille") is False


def test_ontology_spatial_relation_validation() -> None:
    ont = get_universal_ontology()

    # Valid: leg is part of chair
    ok, _ = ont.validate_spatial_relation("legs", "chair", SpatialRelationType.PART_OF)
    assert ok is True

    # Invalid: steering wheel part of chair
    ok, msg = ont.validate_spatial_relation("steering_wheel", "chair", SpatialRelationType.PART_OF)
    assert ok is False
    assert "not an established part" in msg

    # Valid: floor supports chair
    ok, _ = ont.validate_spatial_relation("floor", "chair", SpatialRelationType.SUPPORTS)
    assert ok is True

    # Invalid: socket contained in non-container chair
    ok, msg = ont.validate_spatial_relation("socket", "chair", SpatialRelationType.CONTAINED_IN)
    assert ok is False
    assert "not a container" in msg
