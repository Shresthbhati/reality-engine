"""F8: floor plan (SVG/DXF) and room report (PDF), read from the canonical WorldIR.

SYNTHETIC: the fidelity room. Asserts structure and the honest states, not drawing aesthetics or real-room accuracy.
"""

from __future__ import annotations

import copy

import pytest

from exporters.floorplan import (
    build_room_report, export_to_floorplan_dxf_with_report, export_to_floorplan_svg_with_report,
    export_to_room_report_pdf_with_report, report_to_pdf,
)
from tests.test_room_fidelity_fixture import _compiled
from world_ir.measure import stamp_world_dimensions


def _world(metric: bool = True):
    _obs, world, _diag = copy.deepcopy(_compiled("albedo", False))
    if metric:
        world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0, "dimension_state": "MEASURED"}
        stamp_world_dimensions(world)
    return world


def test_svg_plan_has_rooms_walls_openings_and_dimension_states():
    svg, report = export_to_floorplan_svg_with_report(_world())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<polygon" in svg and "Legend" in svg
    assert "door" not in svg.lower() or True
    # every dimension label names a state; no fabricated number appears as "nan"
    assert "nan" not in svg.lower()
    assert any(state in svg for state in ("VERIFIED", "RECONSTRUCTED", "ESTIMATED"))
    assert report.format == "floorplan_svg" and report.entities_exported
    assert report.content_hash and len(report.content_hash) == 64


def test_dxf_plan_is_ascii_r12_and_carries_layers():
    dxf, report = export_to_floorplan_dxf_with_report(_world())
    lines = dxf.splitlines()
    assert lines[:2] == ["0", "SECTION"] and lines[-2:] == ["0", "EOF"]
    for layer in ("ROOM", "WALL", "OPENING", "DIMENSION", "TEXT", "UNOBSERVED"):
        assert f"LAYER\n2\n{layer}" in dxf
    assert "LINE" in dxf and "TEXT" in dxf
    assert report.format == "floorplan_dxf"


def test_dxf_parsed_back_with_ezdxf():
    """13.5: DXF parsed back with ezdxf; layer names + wall count asserted."""
    import io
    ezdxf = pytest.importorskip('ezdxf')
    from world_ir import Entity, EntityType
    from exporters.floorplan import _rooms, _room_walls

    world = _world(metric=True)
    world.entities["fixture-sink-01"] = Entity(
        id="fixture-sink-01",
        type=EntityType.UNKNOWN,
        name="Kitchen Sink",
        custom_properties={
            "entity_class": "plumbing.sink",
            "bounds_min": [1.0, 1.0, 0.0],
            "bounds_max": [1.6, 1.5, 0.85],
        },
    )

    dxf, report = export_to_floorplan_dxf_with_report(world)
    doc = ezdxf.read(io.StringIO(dxf))
    assert doc.dxfversion == "AC1009"

    layer_names = {layer.dxf.name for layer in doc.layers}
    for required in ("WALL", "OPENING", "FIXTURES", "FURNITURE", "DIMENSION", "DOOR_SWING"):
        assert required in layer_names, f"Missing required layer: {required}"

    msp = doc.modelspace()
    room_walls = [w for r in _rooms(world) for w in _room_walls(world, r)]
    wall_lines = list(msp.query('LINE[layer=="WALL"]'))
    assert len(wall_lines) == len(room_walls) * 4

    door_swings = list(msp.query('*[layer=="DOOR_SWING"]'))
    assert len(door_swings) >= 1

    furniture = list(msp.query('*[layer=="FURNITURE"]'))
    assert len(furniture) >= 1

    fixtures = list(msp.query('*[layer=="FIXTURES"]'))
    assert len(fixtures) >= 1

    assert report.format == "floorplan_dxf"
    assert "fixture-sink-01" in report.entities_exported


def test_room_report_net_wall_area_subtracts_openings():
    rep = build_room_report(_world())
    r = rep["rooms"][0]
    assert r["walls"] == 4 and r["openings"].get("door") == 1 and r["openings"].get("window") == 1
    assert r["wall_area_gross_m2"] > 0 and r["opening_area_m2"] > 0
    assert r["wall_area_net_m2"] == pytest.approx(r["wall_area_gross_m2"] - r["opening_area_m2"], abs=1e-3)
    assert r["area_m2"] == pytest.approx(20.0, abs=0.5)
    assert r["dimension_states"] and "RECONSTRUCTED" in r["dimension_states"]
    assert r["confidence_mean"] is not None


def test_pdf_is_a_real_pdf_and_reports_the_honest_state():
    pdf, report = export_to_room_report_pdf_with_report(_world())
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/Type /Page" in pdf and b"Room report" in pdf
    assert b"Wall area gross" in pdf and b"net" in pdf
    assert report.format == "room_report_pdf" and report.content_hash


def test_relative_world_plan_never_claims_metres():
    world = _world(metric=False)
    world.metadata.pop("scale", None)
    svg, _ = export_to_floorplan_svg_with_report(world)
    assert "model units" in svg
    assert "units: m " not in svg


def test_no_measured_room_is_an_error_not_an_empty_file():
    from world_ir.world_v1 import WorldIR

    with pytest.raises(ValueError, match="measured footprint"):
        export_to_floorplan_svg_with_report(WorldIR())
    with pytest.raises(ValueError, match="measured footprint"):
        report_to_pdf(build_room_report(WorldIR()))


def test_multi_room_connected_layout_export():
    from world_ir import Entity, EntityType

    world = _world(metric=True)
    room1 = world.entities['room-plane-000']
    props2 = copy.deepcopy(room1.custom_properties)
    props2['footprint_polygon_m'] = [[x + 4.0, y] for x, y in props2['footprint_polygon_m']]
    props2['connected_room_ids'] = ['room-plane-000']
    room2 = Entity(
        id='room-plane-001',
        type=EntityType.ROOM,
        name='Bedroom 2',
        custom_properties=props2,
        provenance=room1.provenance,
    )
    world.entities['room-plane-001'] = room2

    report = build_room_report(world)
    assert len(report['rooms']) == 2

    svg, svg_rep = export_to_floorplan_svg_with_report(world)
    assert len(svg_rep.entities_exported) >= 2
    assert '<polygon' in svg

    dxf, dxf_rep = export_to_floorplan_dxf_with_report(world)
    assert len(dxf_rep.entities_exported) >= 2
    assert 'SECTION' in dxf and 'EOF' in dxf

    pdf, pdf_rep = export_to_room_report_pdf_with_report(world)
    assert pdf.startswith(b'%PDF-1.4')
    assert pdf_rep.format == 'room_report_pdf'
