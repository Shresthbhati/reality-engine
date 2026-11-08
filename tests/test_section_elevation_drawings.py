"""Tests for Architectural Section & Elevation Drawing Generator (DXF & Vector SVG)."""

from __future__ import annotations

import uuid
import pytest
from starlette.testclient import TestClient

from apps.api.db import get_sessionmaker, init_db
from apps.api.main import app
from apps.api.models import World, WorldVersion
from apps.api.worldstore_service import get_store
from engine.pipeline.section_elevation_drawings import (
    ArchitecturalDrawing,
    DrawingType,
    SheetSize,
    export_drawing_to_dxf,
    export_drawing_to_svg,
    generate_architectural_drawing,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_generate_longitudinal_section_drawing():
    """Verify architectural section slicing through structural walls, floors, and openings."""
    w = WorldIR(id="wld_dwg_test")

    # Floor slab: center [0, 0, 0], 10m wide, 10m deep, 0.3m thick
    w.entities["slab-1"] = Entity(
        id="slab-1",
        name="Ground Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width": 10.0, "depth": 10.0, "height": 0.3},
    )

    # South Wall: center [0, 1.4, -4.9], 10m long, 2.8m high, 0.25m thick
    w.entities["wall-south"] = Entity(
        id="wall-south",
        name="South Exterior Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.4, "z": -4.9}},
        custom_properties={"width": 10.0, "depth": 0.25, "height": 2.8},
    )

    # North Wall: center [0, 1.4, 4.9]
    w.entities["wall-north"] = Entity(
        id="wall-north",
        name="North Exterior Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.4, "z": 4.9}},
        custom_properties={"width": 10.0, "depth": 0.25, "height": 2.8},
    )

    # Interior Cross-Partition Wall: cuts at center Z=0.0
    w.entities["wall-cut"] = Entity(
        id="wall-cut",
        name="Central Structural Spine Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.4, "z": 0.0}},
        custom_properties={"width": 8.0, "depth": 0.30, "height": 2.8},
    )

    # Door opening in projection: center [2.0, 1.0, 2.0]
    w.entities["door-1"] = Entity(
        id="door-1",
        name="Interior Timber Door",
        type=EntityType.DOOR,
        transform={"position": {"x": 2.0, "y": 1.0, "z": 2.0}},
        custom_properties={"width": 0.9, "depth": 0.1, "height": 2.1},
    )

    # Window in projection: center [-2.0, 1.5, 3.0]
    w.entities["win-1"] = Entity(
        id="win-1",
        name="Double Glazed Window",
        type=EntityType.WINDOW,
        transform={"position": {"x": -2.0, "y": 1.5, "z": 3.0}},
        custom_properties={"width": 1.4, "depth": 0.15, "height": 1.2},
    )

    drawing = generate_architectural_drawing(
        world=w,
        drawing_type=DrawingType.SECTION_LONGITUDINAL,
        cut_offset_m=0.0,
        scale_label="1:50",
        sheet_size=SheetSize.A2,
    )

    assert drawing.drawing_type == DrawingType.SECTION_LONGITUDINAL
    assert drawing.scale_label == "1:50"
    assert len(drawing.lines) > 0
    assert len(drawing.cut_polygons) >= 1
    assert any("A-WALL-CUT" in p.layer for p in drawing.cut_polygons)
    assert any("A-FLOR-CUT" in l.layer for l in drawing.lines)
    assert any("A-DOOR" in l.layer for l in drawing.lines)
    assert any("A-GLAZ" in l.layer for l in drawing.lines)
    assert len(drawing.levels) >= 2
    assert any("F.F.L." in lvl.label for lvl in drawing.levels)
    assert len(drawing.dimensions) >= 1


def test_generate_elevation_drawing():
    """Verify exterior facade elevation drawing generation with ground datum."""
    w = WorldIR(id="wld_elev_test")
    w.entities["wall-front"] = Entity(
        id="wall-front",
        name="Facade Curtain Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.5, "z": -3.0}},
        custom_properties={"width": 12.0, "depth": 0.3, "height": 3.0},
    )

    drawing = generate_architectural_drawing(
        world=w,
        drawing_type=DrawingType.ELEVATION_NORTH,
        scale_label="1:100",
        sheet_size=SheetSize.A1,
    )

    assert drawing.drawing_type == DrawingType.ELEVATION_NORTH
    assert "NORTH EXTERIOR ELEVATION" in drawing.title
    assert any("A-SITE-GRND" in l.layer for l in drawing.lines)
    assert len(drawing.levels) >= 1


def test_export_drawing_to_svg():
    """Verify vector SVG generation in dark and light themes with title block."""
    w = WorldIR(id="wld_svg_test")
    w.entities["wall-1"] = Entity(
        id="wall-1",
        name="Main Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.4, "z": 0.0}},
        custom_properties={"width": 6.0, "depth": 0.25, "height": 2.8},
    )

    drawing = generate_architectural_drawing(w, DrawingType.SECTION_LONGITUDINAL)

    # Dark studio canvas SVG
    svg_dark = export_drawing_to_svg(drawing, project_name="Studio Pavilion", theme="dark")
    assert svg_dark.startswith("<svg")
    assert svg_dark.endswith("</svg>")
    assert "viewBox" in svg_dark
    assert "title-block" in svg_dark
    assert "Studio Pavilion" in svg_dark
    assert "arch_hatch" in svg_dark

    # Light print SVG
    svg_light = export_drawing_to_svg(drawing, project_name="Studio Pavilion", theme="light")
    assert svg_light.startswith("<svg")
    assert "#ffffff" in svg_light


def test_export_drawing_to_dxf():
    """Verify AutoCAD R12/R2000 compliant ASCII DXF generation."""
    w = WorldIR(id="wld_dxf_test")
    w.entities["wall-1"] = Entity(
        id="wall-1",
        name="Loadbearing Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.5, "z": 0.0}},
        custom_properties={"width": 6.0, "depth": 0.3, "height": 3.0},
    )

    drawing = generate_architectural_drawing(w, DrawingType.SECTION_TRANSVERSE)
    dxf_str = export_drawing_to_dxf(drawing, project_name="Civic Center")

    assert "SECTION" in dxf_str
    assert "HEADER" in dxf_str
    assert "$ACADVER" in dxf_str
    assert "AC1009" in dxf_str
    assert "TABLES" in dxf_str
    assert "A-WALL-CUT" in dxf_str
    assert "ENTITIES" in dxf_str
    assert "LINE" in dxf_str
    assert "POLYLINE" in dxf_str
    assert "EOF" in dxf_str


@pytest.mark.asyncio
async def test_drawings_api_endpoints():
    """Verify HTTP endpoints for preset listings, generation, SVG rendering, and DXF download."""
    client = TestClient(app)

    world_id = f"wld_dwg_{uuid.uuid4().hex[:8]}"
    w = WorldIR(id=world_id)
    w.entities["wall-1"] = Entity(
        id="wall-1",
        name="Core Wall",
        type=EntityType.WALL,
        transform={"position": {"x": 0.0, "y": 1.5, "z": 0.0}},
        custom_properties={"width": 8.0, "depth": 0.3, "height": 3.0},
    )
    w.entities["slab-1"] = Entity(
        id="slab-1",
        name="First Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"width": 10.0, "depth": 10.0, "height": 0.3},
    )

    store = get_store()
    sv = store.save_version(w, parent=None)

    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        world_row = World(
            id=w.id,
            name="Architectural Center",
            current_version_id=sv.version_id,
            status="active",
        )
        session.add(world_row)

        ver_row = WorldVersion(
            id=sv.version_id,
            world_id=w.id,
            label="Version for Drawings",
            report={},
        )
        session.add(ver_row)
        await session.commit()

    # 1. GET presets
    res_presets = client.get(f"/api/worlds/{w.id}/drawings/presets")
    assert res_presets.status_code == 200, res_presets.text
    presets_data = res_presets.json()
    assert len(presets_data["presets"]) >= 4
    assert any(p["id"] == "section_longitudinal" for p in presets_data["presets"])

    # 2. POST generate
    res_gen = client.post(
        f"/api/worlds/{w.id}/drawings/generate",
        json={"drawing_type": "section_longitudinal", "scale_label": "1:50", "sheet_size": "A2"},
    )
    assert res_gen.status_code == 200, res_gen.text
    dwg_data = res_gen.json()
    assert dwg_data["drawing_type"] == "section_longitudinal"
    assert dwg_data["scale_label"] == "1:50"
    assert dwg_data["line_count"] > 0
    assert dwg_data["polygon_count"] > 0

    # 3. GET SVG rendering
    res_svg = client.get(
        f"/api/worlds/{w.id}/drawings/render/svg?drawing_type=section_longitudinal&theme=dark"
    )
    assert res_svg.status_code == 200, res_svg.text
    assert res_svg.headers["content-type"] == "image/svg+xml"
    assert res_svg.text.startswith("<svg")

    # 4. GET DXF export
    res_dxf = client.get(
        f"/api/worlds/{w.id}/drawings/export/dxf?drawing_type=section_longitudinal"
    )
    assert res_dxf.status_code == 200, res_dxf.text
    assert res_dxf.headers["content-type"] == "application/dxf"
    assert "AC1009" in res_dxf.text
    assert "EOF" in res_dxf.text
