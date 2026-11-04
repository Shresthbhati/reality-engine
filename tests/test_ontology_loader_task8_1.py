"""Tests for Blueprint Task 8.1: Ontology file + loader (Blueprint 17)."""

import pytest
from perception.ontology import (
    FALLBACK_PATH,
    Ontology,
    OntologyNode,
    get_default_ontology,
    load_ontology,
    resolve_entity_class,
)


def test_load_ontology_from_default_config():
    onto = get_default_ontology()
    assert isinstance(onto, Ontology)
    assert onto.version == "1.0.0"
    assert len(onto.nodes) >= 30, f"Expected >= 30 v1 classes, got {len(onto.nodes)}"

    # Zero validation issues
    issues = onto.validate()
    assert issues == [], f"Ontology validation failed: {issues}"


def test_v1_class_groups_represented():
    onto = get_default_ontology()
    expected_groups = {
        "architecture",
        "furniture",
        "decor",
        "electronics",
        "fixtures",
        "plants",
        "fallback",
    }
    present_groups = {node.group for node in onto.nodes.values()}
    for g in expected_groups:
        assert g in present_groups, f"Missing expected ontology group: {g}"


def test_sectional_sofa_hierarchy_and_components():
    onto = get_default_ontology()
    sectional = onto.get("furniture/seating/sofa/sectional")
    assert sectional is not None
    assert sectional.parent == "furniture/seating/sofa"
    assert sectional.is_child_of("furniture/seating/sofa")
    assert sectional.is_child_of("furniture")
    assert "seat_cushion" in sectional.components
    assert "chaise" in sectional.components
    assert sectional.typical_dims_m is not None
    assert "w" in sectional.typical_dims_m
    assert sectional.min_pixels >= 1500


def test_resolve_or_unknown_known_classes():
    onto = get_default_ontology()

    # Exact path
    n1 = onto.resolve_or_unknown("architecture/door")
    assert n1.path == "architecture/door"

    # Case-insensitive name
    n2 = onto.resolve_or_unknown("Dining table")
    assert n2.path == "furniture/tables/dining_table"

    # Detector prompt
    n3 = onto.resolve_or_unknown("coffee table")
    assert n3.path == "furniture/tables/coffee_table"

    n4 = onto.resolve_or_unknown("power outlet")
    assert n4.path == "fixtures/electrical/socket"

    n5 = onto.resolve_or_unknown("television")
    assert n5.path == "electronics/display/tv"


def test_resolve_or_unknown_fallback_on_unrecognized():
    onto = get_default_ontology()

    # Completely unrecognized queries must map to unknown_object (P9)
    queries = [
        "alien spaceship",
        "random unclassified debris",
        "unknown widget 999",
        "",
        "asdkjh2389y",
    ]
    for q in queries:
        resolved = onto.resolve_or_unknown(q)
        assert resolved.path == FALLBACK_PATH
        assert resolved.name == "unknown_object"
        assert resolved.geometry == "proxy_box"


def test_detector_prompt_mapping():
    onto = get_default_ontology()
    all_prompts = onto.all_prompts()
    assert len(all_prompts) >= 30
    flat_list = onto.flat_prompt_list()
    assert len(flat_list) >= len(all_prompts)
    assert "sectional sofa" in flat_list
    assert "sofa" in flat_list
    assert "window" in flat_list


def test_convenience_function():
    resolved = resolve_entity_class("armchair")
    assert resolved.path == "furniture/seating/armchair"
    assert resolved.mobility == "movable"
