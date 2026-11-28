"""Phase 1 completeness and validation tests for the canonical entity taxonomy."""
from __future__ import annotations

from perception.ontology import VALID_CAPABILITY_STATUSES, get_default_ontology


def test_canonical_ontology_validates_cleanly() -> None:
    ont = get_default_ontology()
    issues = ont.validate()
    assert issues == [], f"Ontology validation issues: {issues}"


def test_all_phase1_domains_and_classes_resolve() -> None:
    ont = get_default_ontology()
    required_classes = [
        # A. Architecture
        "building", "floor", "room", "corridor", "wall", "floor surface", "ceiling",
        "column", "beam", "structural opening", "door", "door frame", "window",
        "window frame", "staircase", "landing", "railing", "roof", "facade",
        "balcony", "entrance", "architectural recess",
        # B. Furniture & interior
        "table", "chair", "sofa", "bed", "cabinet", "wardrobe", "shelf",
        "desk", "counter", "appliance", "storage unit",
        # C. Electrical & mechanical
        "electrical socket", "plug point", "switch", "switchboard",
        "distribution panel", "light fixture", "ceiling fan", "vent",
        "air-conditioning unit", "duct", "pipe", "radiator", "plumbing fixture",
        "smoke detector", "fire alarm", "cable tray",
        # D. Fine architectural
        "door handle", "door hinge", "lock", "window latch", "window sill",
        "skirting", "molding", "trim", "grille", "handrail", "fastener",
        # E. Outdoor & landscape
        "tree", "trunk", "branch", "canopy", "shrub", "plant", "grass",
        "hedge", "flower bed", "soil", "footpath", "road", "curb",
        "streetlight", "lamppost", "signpost", "bollard", "bench", "drain",
        "manhole", "fence", "gate", "boundary wall",
        # F. Building & infrastructure systems
        "exterior wall", "facade opening", "window assembly", "door assembly",
        "roof assembly", "stair assembly", "drainage component", "lighting infrastructure",
    ]
    missing = [c for c in required_classes if ont.resolve_or_unknown(c).path.startswith("fallback")]
    assert missing == [], f"Missing Phase 1 classes from ontology: {missing}"


def test_every_node_has_valid_capability_status() -> None:
    ont = get_default_ontology()
    for path, node in ont.nodes.items():
        assert node.capability_status in VALID_CAPABILITY_STATUSES, (
            f"{path} has invalid capability_status {node.capability_status!r}"
        )
