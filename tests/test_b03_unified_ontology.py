"""BUILD_SPEC B03: Tests for unified ontology, parts, and attributes."""

from __future__ import annotations
import pytest
from perception.ontology import get_default_ontology, load_ontology, Ontology, OntologyNode
import perception.entities.taxonomy as tax

def test_b03_ontology_validate_returns_empty():
    onto = get_default_ontology()
    issues = onto.validate()
    assert issues == [], f"Ontology validation failed: {issues}"

def test_b03_alias_lookup_wardrobe_and_almirah():
    onto = get_default_ontology()
    w = onto.resolve_or_unknown("wardrobe")
    a = onto.resolve_or_unknown("almirah")
    assert w.path == a.path
    assert w.path in ("furniture/storage/almirah", "furniture/storage/wardrobe")

def test_b03_ontology_node_attributes_and_aliases():
    onto = get_default_ontology()
    almirah = onto.resolve_or_unknown("almirah")
    assert isinstance(almirah.attributes, dict)
    assert "handle_type" in almirah.attributes
    assert "unknown" in almirah.attributes["handle_type"]
    assert "wardrobe" in almirah.aliases or "almirah" in almirah.aliases

def test_b03_ontology_children_and_ancestors():
    onto = get_default_ontology()
    assert isinstance(onto.children, dict)
    # Check ancestors
    anc = onto.ancestors("electrical.switchboard.module.socket")
    assert "electrical.switchboard.module" in anc or "electrical.switchboard" in anc

def test_b03_all_old_taxonomy_classes_resolve():
    classes = tax.all_classes()
    assert len(classes) >= 117
    for c in classes:
        ec = tax.get(c.id)
        assert ec.id == c.id

def test_b03_section_4_families_represented():
    onto = get_default_ontology()
    # Storage, Electrical, Doors, Windows, Seating/tables, Fixtures, Structure
    # 1. Electrical
    socket = onto.resolve_or_unknown("fixtures/electrical/socket")
    assert "socket_standard" in socket.attributes
    assert "unknown" in socket.attributes["socket_standard"]
    # 2. Doors
    door = onto.resolve_or_unknown("architecture/door")
    assert "handle_type" in door.attributes
    assert "unknown" in door.attributes["handle_type"]
    # 3. Windows
    win = onto.resolve_or_unknown("architecture/window")
    assert "pane_count" in win.attributes
    assert "unknown" in win.attributes["pane_count"]
    # 4. Structure
    wall = onto.resolve_or_unknown("architecture/wall")
    assert "finish" in wall.attributes
    assert "unknown" in wall.attributes["finish"]
