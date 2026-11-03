"""Tests for Infrastructure and Utility Network Intelligence (Wave 2, Block 15)."""

from __future__ import annotations

import pytest

from perception.infrastructure.utility_network import (
    FlowRole,
    UtilityDomain,
    UtilityNetworkReport,
    build_utility_networks,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


def test_electrical_circuit_topology():
    world = WorldIR()

    # Main Distribution Board at (0, 1.5, 0)
    g_panel = Geometry(id="g-p", type=GeometryType.BOX, bounds_min=Vector3(x=-0.2, y=1.2, z=-0.05), bounds_max=Vector3(x=0.2, y=1.8, z=0.05))
    e_panel = Entity(id="ent-panel", name="Main Breaker Panel", type=EntityType.STRUCTURE, geometry_ids=["g-p"],
                     custom_properties={"entity_class": "electrical.distribution_board"})
    world.geometries["g-p"] = g_panel
    world.entities["ent-panel"] = e_panel

    # Wall Conduit run at (1.0, 1.5, 0)
    g_conduit = Geometry(id="g-c", type=GeometryType.BOX, bounds_min=Vector3(x=0.2, y=1.45, z=-0.02), bounds_max=Vector3(x=2.0, y=1.55, z=0.02))
    e_conduit = Entity(id="ent-conduit", name="Wall Conduit", type=EntityType.STRUCTURE, geometry_ids=["g-c"],
                       custom_properties={"entity_class": "electrical.conduit"})
    world.geometries["g-c"] = g_conduit
    world.entities["ent-conduit"] = e_conduit

    # Switchboard at (2.2, 1.2, 0)
    g_sw = Geometry(id="g-sw", type=GeometryType.BOX, bounds_min=Vector3(x=2.1, y=1.1, z=-0.02), bounds_max=Vector3(x=2.3, y=1.3, z=0.02))
    e_sw = Entity(id="ent-sw", name="Living Room Switch", type=EntityType.STRUCTURE, geometry_ids=["g-sw"],
                  custom_properties={"entity_class": "electrical.switchboard.module.switch"})
    world.geometries["g-sw"] = g_sw
    world.entities["ent-sw"] = e_sw

    # Ceiling light at (2.2, 2.8, 0)
    g_light = Geometry(id="g-lt", type=GeometryType.BOX, bounds_min=Vector3(x=2.0, y=2.7, z=-0.2), bounds_max=Vector3(x=2.4, y=2.9, z=0.2))
    e_light = Entity(id="ent-light", name="Ceiling Downlight", type=EntityType.STRUCTURE, geometry_ids=["g-lt"],
                     custom_properties={"entity_class": "electrical.light_fitting"})
    world.geometries["g-lt"] = g_light
    world.entities["ent-light"] = e_light

    report = build_utility_networks(world, max_connection_distance_m=4.0)

    assert isinstance(report, UtilityNetworkReport)
    assert report.total_utility_entities == 4
    assert report.domain_counts["electrical"] == 4
    assert len(report.circuits) >= 1

    # Check circuit properties
    ckt = report.circuits[0]
    assert ckt.domain == UtilityDomain.ELECTRICAL
    assert ckt.source_entity_id == e_panel.id
    assert len(ckt.member_entity_ids) == 4

    # Check CONNECTS relationships in WorldIR
    panel_conn = [r for r in e_panel.relationships if r.kind == RelationshipKind.CONNECTS and r.target_id == e_conduit.id]
    assert len(panel_conn) == 1
    assert panel_conn[0].metadata["utility_domain"] == "electrical"


def test_plumbing_and_isolated_fixture_detection():
    world = WorldIR()

    # Sink with tap at (0, 0.8, 0)
    g_sink = Geometry(id="g-sk", type=GeometryType.BOX, bounds_min=Vector3(x=-0.3, y=0.5, z=-0.2), bounds_max=Vector3(x=0.3, y=0.9, z=0.2))
    e_sink = Entity(id="ent-sink", name="Kitchen Sink", type=EntityType.STRUCTURE, geometry_ids=["g-sk"],
                    custom_properties={"entity_class": "plumbing.sink"})
    world.geometries["g-sk"] = g_sink
    world.entities["ent-sink"] = e_sink

    # Pipe directly adjacent to sink at (0, 0.4, 0)
    g_pipe = Geometry(id="g-pi", type=GeometryType.BOX, bounds_min=Vector3(x=-0.05, y=0.0, z=-0.05), bounds_max=Vector3(x=0.05, y=0.5, z=0.05))
    e_pipe = Entity(id="ent-pipe", name="Supply Pipe", type=EntityType.STRUCTURE, geometry_ids=["g-pi"],
                    custom_properties={"entity_class": "plumbing.pipe"})
    world.geometries["g-pi"] = g_pipe
    world.entities["ent-pipe"] = e_pipe

    # Distant isolated drain at (50.0, 0, 50.0)
    g_drain = Geometry(id="g-dr", type=GeometryType.BOX, bounds_min=Vector3(x=49.9, y=0, z=49.9), bounds_max=Vector3(x=50.1, y=0.1, z=50.1))
    e_drain = Entity(id="ent-drain-far", name="Distant Yard Drain", type=EntityType.STRUCTURE, geometry_ids=["g-dr"],
                     custom_properties={"entity_class": "plumbing.drain"})
    world.geometries["g-dr"] = g_drain
    world.entities["ent-drain-far"] = e_drain

    report = build_utility_networks(world, max_connection_distance_m=3.0)

    # Sink and pipe should be connected, distant drain must be isolated
    assert "ent-drain-far" in report.isolated_entities
    assert "ent-sink" not in report.isolated_entities
    assert "ent-pipe" not in report.isolated_entities
