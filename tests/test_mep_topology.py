"""Tests for Automated MEP & Conduit Routing Perception with IFC4x3 Facility Topology.

Verifies:
1. Perception of electrical, HVAC, and plumbing fixtures from WorldIR.
2. Orthogonal 3D Manhattan conduit/pipe routing with plenum heights and 90° elbows.
3. Distribution graph topology, ports, and DAG connectivity.
4. NEC / IPC spatial clearance verification between electrical and plumbing systems.
5. IFC4X3 Building Services STEP physical file export and IfcOpenShell roundtrip.
6. 3D LineString GeoJSON network export.
7. FastAPI REST API endpoints.
"""

from __future__ import annotations

import json
import math
from typing import Any
import pytest

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from engine.pipeline.mep_topology import (
    CrossSectionKind,
    FlowDirection,
    MepDistributionNetwork,
    MepElementType,
    MepFacilityTopology,
    MepFitting,
    MepPort,
    MepSegment,
    MepSystemType,
    MepTerminal,
    detect_mep_elements_from_world,
    export_mep_to_geojson,
    export_mep_to_ifc4x3,
    infer_orthogonal_routing,
    verify_code_clearances,
)


def _build_test_world() -> WorldIR:
    """Create a sample world with electrical panel, outlets, lights, diffusers, and plumbing."""
    world = WorldIR(id="world-facility-test")

    # Ceiling entity for plenum height
    world.entities["ceiling-1"] = Entity(
        id="ceiling-1",
        name="Main Ceiling Slab",
        type=EntityType.CEILING,
        transform={"position": [2.5, 2.5, 2.7]},
    )

    # 1. Main Electrical Distribution Board
    world.entities["panel-main"] = Entity(
        id="panel-main",
        name="Main Electrical Distribution Panel (MDP)",
        type=EntityType.STRUCTURE,
        transform={"position": [0.2, 0.5, 1.4]},
    )

    # 2. Electrical Outlets
    world.entities["outlet-1"] = Entity(
        id="outlet-1",
        name="Duplex Wall Outlet #1",
        type=EntityType.WALL,
        transform={"position": [3.5, 0.1, 0.45]},
    )
    world.entities["outlet-2"] = Entity(
        id="outlet-2",
        name="Duplex Wall Outlet #2",
        type=EntityType.WALL,
        transform={"position": [4.8, 3.2, 0.45]},
    )

    # 3. Light Switch
    world.entities["switch-1"] = Entity(
        id="switch-1",
        name="3-Gang Light Switch",
        type=EntityType.WALL,
        transform={"position": [0.3, 1.2, 1.1]},
    )

    # 4. Ceiling Light Fixture
    world.entities["light-1"] = Entity(
        id="light-1",
        name="Recessed LED Troffer Luminaire",
        type=EntityType.CEILING,
        transform={"position": [2.5, 2.5, 2.65]},
    )

    # 5. HVAC Supply Air Diffuser
    world.entities["diffuser-1"] = Entity(
        id="diffuser-1",
        name="HVAC Supply Air Diffuser 4-Way",
        type=EntityType.CEILING,
        transform={"position": [1.5, 3.5, 2.65]},
    )

    # 6. Plumbing Fixture
    world.entities["sink-1"] = Entity(
        id="sink-1",
        name="Commercial Stainless Kitchen Sink",
        type=EntityType.FURNITURE,
        transform={"position": [3.6, 0.15, 0.85]},
    )

    return world


def test_infer_orthogonal_routing():
    """Verify 3D Manhattan orthogonal routing with ceiling plenum rise and vertical drops."""
    start = (0.2, 0.5, 1.4)
    end = (3.5, 0.1, 0.45)
    ceiling_z = 2.7

    route = infer_orthogonal_routing(start, end, ceiling_z=ceiling_z)
    assert len(route) >= 4
    assert route[0] == start
    assert route[-1] == end

    # Check that highest waypoint reaches the ceiling plenum
    max_z = max(p[2] for p in route)
    assert max_z >= 2.7

    # Check total length is strictly positive and >= direct Euclidean distance
    direct_d = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, end)))
    route_len = sum(
        math.sqrt(sum((b - a) ** 2 for a, b in zip(route[i], route[i + 1])))
        for i in range(len(route) - 1)
    )
    assert route_len >= direct_d


def test_detect_mep_elements_and_routing():
    """Verify automated perception, system grouping, and conduit/pipe routing."""
    world = _build_test_world()
    topology = detect_mep_elements_from_world(world)

    assert topology.world_id == "world-facility-test"
    assert len(topology.systems) >= 2  # Electrical, Plumbing, HVAC

    elec_sys = next((s for s in topology.systems if s.system_type == MepSystemType.ELECTRICAL), None)
    assert elec_sys is not None
    assert elec_sys.root_panel_id == "panel-main"
    assert len(elec_sys.terminals) >= 4
    assert len(elec_sys.segments) >= 3  # Conduits to outlets, switch, light
    assert elec_sys.total_length_m > 5.0
    assert elec_sys.is_connected_dag is True

    # Check segments have ports
    for seg in elec_sys.segments:
        assert len(seg.ports) >= 2
        assert seg.diameter_mm == 25.0
        assert seg.cross_section == CrossSectionKind.CIRCULAR
        assert len(seg.waypoints) >= 2

    # Check plumbing system
    plumb_sys = next((s for s in topology.systems if s.system_type == MepSystemType.PLUMBING), None)
    assert plumb_sys is not None
    assert len(plumb_sys.terminals) >= 1
    assert len(plumb_sys.segments) >= 1


def test_code_clearance_verification():
    """Verify NEC/IPC code clearance check between electrical and water runs."""
    # Create electrical segment
    eseg = MepSegment(
        segment_id="conduit-test",
        name="Branch Circuit Conduit",
        system_type=MepSystemType.ELECTRICAL,
        element_type=MepElementType.CONDUIT,
        start_point=(1.0, 0.0, 0.5),
        end_point=(3.0, 0.0, 0.5),
    )
    esys = MepDistributionNetwork(
        system_id="sys-elec",
        name="Electrical System",
        system_type=MepSystemType.ELECTRICAL,
        segments=[eseg],
    )

    # 1. Conflicting water pipe 0.05m away (< 0.15m required threshold)
    pseg_conflict = MepSegment(
        segment_id="pipe-water-close",
        name="Domestic Water Pipe",
        system_type=MepSystemType.PLUMBING,
        element_type=MepElementType.PIPE,
        start_point=(1.0, 0.05, 0.5),
        end_point=(3.0, 0.05, 0.5),
    )
    psys_conflict = MepDistributionNetwork(
        system_id="sys-plumb",
        name="Plumbing System",
        system_type=MepSystemType.PLUMBING,
        segments=[pseg_conflict],
    )

    violations = verify_code_clearances([esys, psys_conflict], min_clearance_m=0.15)
    assert len(violations) == 1
    v = violations[0]
    assert v.actual_clearance_m < 0.15
    assert v.element_a_id == "conduit-test"
    assert v.element_b_id == "pipe-water-close"
    assert "Clearance conflict" in v.description

    # 2. Compliant water pipe 0.30m away (>= 0.15m)
    pseg_compliant = MepSegment(
        segment_id="pipe-water-far",
        name="Domestic Water Pipe Compliant",
        system_type=MepSystemType.PLUMBING,
        element_type=MepElementType.PIPE,
        start_point=(1.0, 0.35, 0.5),
        end_point=(3.0, 0.35, 0.5),
    )
    psys_compliant = MepDistributionNetwork(
        system_id="sys-plumb-ok",
        name="Plumbing System OK",
        system_type=MepSystemType.PLUMBING,
        segments=[pseg_compliant],
    )
    violations_ok = verify_code_clearances([esys, psys_compliant], min_clearance_m=0.15)
    assert len(violations_ok) == 0


def test_export_mep_to_ifc4x3_and_ifcopenshell_roundtrip():
    """Verify IFC4X3 Building Services export and ifcopenshell validation."""
    ifcopenshell = pytest.importorskip("ifcopenshell")

    world = _build_test_world()
    topology = detect_mep_elements_from_world(world)

    ifc_text = export_mep_to_ifc4x3(topology, world=world)

    # Basic STEP physical file checks
    assert "ISO-10303-21;" in ifc_text
    assert "IFC4X3" in ifc_text
    assert "END-ISO-10303-21;" in ifc_text
    assert "IFCDISTRIBUTIONSYSTEM" in ifc_text
    assert "IFCCABLECARRIERSEGMENT" in ifc_text
    assert "IFCDISTRIBUTIONPORT" in ifc_text
    assert "IFCRELCONNECTSPORTS" in ifc_text

    # Parse with IfcOpenShell to prove standard conformity
    model = ifcopenshell.file.from_string(ifc_text)
    assert model.schema == "IFC4X3"

    systems = model.by_type("IfcDistributionSystem")
    assert len(systems) >= 2

    segments = model.by_type("IfcCableCarrierSegment")
    assert len(segments) >= 3

    ports = model.by_type("IfcDistributionPort")
    assert len(ports) >= 6

    port_connections = model.by_type("IfcRelConnectsPorts")
    assert len(port_connections) >= 2


def test_export_mep_to_geojson():
    """Verify 3D LineString GeoJSON network export."""
    world = _build_test_world()
    topology = detect_mep_elements_from_world(world)

    gj = export_mep_to_geojson(topology)
    assert gj["type"] == "FeatureCollection"
    assert "features" in gj
    assert len(gj["features"]) >= 5

    line_features = [f for f in gj["features"] if f["geometry"]["type"] == "LineString"]
    assert len(line_features) >= 3
    for lf in line_features:
        coords = lf["geometry"]["coordinates"]
        assert len(coords) >= 2
        assert len(coords[0]) == 3  # [x, y, z]
        assert "system_type" in lf["properties"]
        assert "diameter_mm" in lf["properties"]

    point_features = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    assert len(point_features) >= 1
    for pf in point_features:
        coords = pf["geometry"]["coordinates"]
        assert len(coords) == 3


def test_mep_facility_topology_serialization():
    """Verify dictionary serialization contract."""
    world = _build_test_world()
    topology = detect_mep_elements_from_world(world)

    d = topology.to_dict()
    assert d["world_id"] == "world-facility-test"
    assert d["total_conduit_length_m"] > 0
    assert "summary" in d
    assert "systems" in d
    assert isinstance(d["systems"], list)

    # JSON roundtrip proof
    json_str = json.dumps(d)
    assert len(json_str) > 100
    loaded = json.loads(json_str)
    assert loaded["world_id"] == "world-facility-test"


@pytest.mark.asyncio
async def test_mep_api_endpoints():
    """Verify HTTP endpoints for topology retrieval, detection, routing, IFC export, and GeoJSON."""
    from starlette.testclient import TestClient
    import uuid
    from apps.api.main import app
    from apps.api.db import get_sessionmaker, init_db
    from apps.api.models import World, WorldVersion
    from apps.api.worldstore_service import get_store

    client = TestClient(app)

    world_id = f"wld_mep_{uuid.uuid4().hex[:8]}"
    w = _build_test_world()
    w.id = world_id

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Facility Operations Center",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Version for MEP Topology",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET /api/worlds/{id}/mep/topology
    res_topo = client.get(f"/api/worlds/{w.id}/mep/topology")
    assert res_topo.status_code == 200, res_topo.text
    topo_data = res_topo.json()
    assert topo_data["world_id"] == w.id
    assert len(topo_data["systems"]) >= 2
    assert "summary" in topo_data

    # 2. POST /api/worlds/{id}/mep/detect with system_filter
    res_detect = client.post(
        f"/api/worlds/{w.id}/mep/detect",
        json={"system_filter": "electrical"},
    )
    assert res_detect.status_code == 200, res_detect.text
    detect_data = res_detect.json()
    assert len(detect_data["systems"]) == 1
    assert detect_data["systems"][0]["system_type"] == "electrical"

    # 3. POST /api/worlds/{id}/mep/route
    res_route = client.post(
        f"/api/worlds/{w.id}/mep/route",
        json={
            "start_point": [0.2, 0.5, 1.4],
            "end_point": [3.5, 0.1, 0.45],
            "ceiling_z": 2.7,
            "system_type": "electrical",
            "diameter_mm": 25.0,
        },
    )
    assert res_route.status_code == 200, res_route.text
    route_data = res_route.json()
    assert route_data["total_length_m"] > 3.0
    assert len(route_data["waypoints"]) >= 4

    # 4. GET /api/worlds/{id}/mep/export/ifc
    res_ifc = client.get(f"/api/worlds/{w.id}/mep/export/ifc")
    assert res_ifc.status_code == 200, res_ifc.text
    assert "ISO-10303-21;" in res_ifc.text
    assert "IFC4X3" in res_ifc.text
    assert "IFCDISTRIBUTIONSYSTEM" in res_ifc.text

    # 5. GET /api/worlds/{id}/mep/export/geojson
    res_gj = client.get(f"/api/worlds/{w.id}/mep/export/geojson")
    assert res_gj.status_code == 200, res_gj.text
    gj_data = res_gj.json()
    assert gj_data["type"] == "FeatureCollection"
    assert len(gj_data["features"]) >= 5

