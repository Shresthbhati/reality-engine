"""Automated Architectural Section & Elevation Drawing Generator.

Generates professional 2D architectural building sections, cross-sections, and exterior elevations
from 3D WorldIR models, with cut plane geometry carving, background projection lines,
datum level markers (F.F.L., C.L., T.O.P.), dimension chains, and drawing sheet layouts.

Exports to:
- Standard AutoCAD-compatible DXF (R12 / R2000 format with architectural layers)
- High-DPI Vector SVG with Dark Studio preview and print-ready architectural title blocks
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

from world_ir.schema_v1 import EntityType
from world_ir.world_v1 import WorldIR


class DrawingType(str, Enum):
    SECTION_LONGITUDINAL = "section_longitudinal"  # Section A-A (cuts east-west along X)
    SECTION_TRANSVERSE = "section_transverse"      # Section B-B (cuts north-south along Z)
    ELEVATION_NORTH = "elevation_north"            # North exterior elevation (view looking South -Z)
    ELEVATION_SOUTH = "elevation_south"            # South exterior elevation (view looking North +Z)
    ELEVATION_EAST = "elevation_east"              # East exterior elevation (view looking West -X)
    ELEVATION_WEST = "elevation_west"              # West exterior elevation (view looking East +X)
    CUSTOM_SECTION = "custom_section"              # User-defined cutting plane


class SheetSize(str, Enum):
    A1 = "A1"          # 841 x 594 mm
    A2 = "A2"          # 594 x 420 mm
    A3 = "A3"          # 420 x 297 mm
    ARCH_D = "ARCH_D"  # 914 x 610 mm (36 x 24 in)


@dataclass
class DrawingLine:
    """A vector line segment in drawing space (units: meters)."""
    layer: str  # A-WALL-CUT, A-WALL-PROJ, A-FLOR-CUT, A-DOOR, A-GLAZ, A-DIMS, A-ANNO
    start: Tuple[float, float]  # (u, v) where u is horizontal axis, v is vertical elevation Y
    end: Tuple[float, float]
    stroke_width_mm: float = 0.35
    is_dashed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "start": [round(self.start[0], 4), round(self.start[1], 4)],
            "end": [round(self.end[0], 4), round(self.end[1], 4)],
            "stroke_width_mm": self.stroke_width_mm,
            "is_dashed": self.is_dashed,
        }


@dataclass
class DrawingPolygon:
    """A solid or hatched cut polygon representing sliced structural mass."""
    layer: str
    vertices: List[Tuple[float, float]]
    fill_hatch: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "vertices": [[round(v[0], 4), round(v[1], 4)] for v in self.vertices],
            "fill_hatch": self.fill_hatch,
        }


@dataclass
class LevelMarker:
    """Architectural datum level callout (e.g., F.F.L. +0.000)."""
    label: str  # "GROUND FLOOR F.F.L.", "FIRST FLOOR F.F.L.", "CEILING LEVEL C.L."
    elevation_m: float
    datum_u: float  # Horizontal position of marker on sheet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "elevation_m": round(self.elevation_m, 3),
            "datum_u": round(self.datum_u, 3),
            "formatted_elevation": f"{'+' if self.elevation_m > 0 else ''}{self.elevation_m:.3f} m",
        }


@dataclass
class DimensionString:
    """Architectural dimension witness line and measurement text."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    offset_u: float
    offset_v: float
    measured_length_m: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": [round(self.start[0], 3), round(self.start[1], 3)],
            "end": [round(self.end[0], 3), round(self.end[1], 3)],
            "measured_length_m": round(self.measured_length_m, 3),
            "label": self.label,
        }


@dataclass
class ArchitecturalDrawing:
    """Complete 2D architectural drawing with cut geometry, projections, dimensions, and sheet metadata."""
    drawing_id: str
    world_id: str
    title: str
    drawing_type: DrawingType
    scale_label: str  # "1:50", "1:100"
    scale_factor: float  # e.g., 0.01 for 1:100 (1m = 10mm)
    sheet_size: SheetSize
    cut_plane_offset_m: float
    extent_u: Tuple[float, float]  # (min_u, max_u)
    extent_v: Tuple[float, float]  # (min_v, max_v)
    lines: List[DrawingLine] = field(default_factory=list)
    cut_polygons: List[DrawingPolygon] = field(default_factory=list)
    levels: List[LevelMarker] = field(default_factory=list)
    dimensions: List[DimensionString] = field(default_factory=list)
    created_at_iso: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drawing_id": self.drawing_id,
            "world_id": self.world_id,
            "title": self.title,
            "drawing_type": self.drawing_type.value,
            "scale_label": self.scale_label,
            "scale_factor": self.scale_factor,
            "sheet_size": self.sheet_size.value,
            "cut_plane_offset_m": round(self.cut_plane_offset_m, 3),
            "extent_u": [round(self.extent_u[0], 3), round(self.extent_u[1], 3)],
            "extent_v": [round(self.extent_v[0], 3), round(self.extent_v[1], 3)],
            "line_count": len(self.lines),
            "polygon_count": len(self.cut_polygons),
            "level_count": len(self.levels),
            "dimension_count": len(self.dimensions),
            "lines": [l.to_dict() for l in self.lines],
            "cut_polygons": [p.to_dict() for p in self.cut_polygons],
            "levels": [lvl.to_dict() for lvl in self.levels],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "created_at_iso": self.created_at_iso,
        }


def _extract_entity_geometry(entity: Any) -> Tuple[float, float, float, float, float, float]:
    """Extract (cx, cy, cz, width, height, depth) for an entity."""
    transform = getattr(entity, "transform", None)
    cx, cy, cz = 0.0, 0.0, 0.0
    if isinstance(transform, dict):
        p = transform.get("position", transform.get("pos", {}))
        if isinstance(p, dict):
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            cz = float(p.get("z", 0.0))
    elif transform is not None:
        p = getattr(transform, "position", None)
        if isinstance(p, dict):
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            cz = float(p.get("z", 0.0))
        elif p is not None:
            cx = float(getattr(p, "x", 0.0))
            cy = float(getattr(p, "y", 0.0))
            cz = float(getattr(p, "z", 0.0))

    props = getattr(entity, "custom_properties", {}) or {}
    w = float(props.get("width_m", props.get("width", 1.0)))
    h = float(props.get("height_m", props.get("height", 2.8)))
    d = float(props.get("depth_m", props.get("depth", props.get("thickness_m", 0.25))))

    return cx, cy, cz, w, h, d


def generate_architectural_drawing(
    world: WorldIR,
    drawing_type: DrawingType = DrawingType.SECTION_LONGITUDINAL,
    cut_offset_m: Optional[float] = None,
    projection_depth_m: float = 12.0,
    scale_label: str = "1:100",
    sheet_size: SheetSize = SheetSize.A2,
    drawing_title: Optional[str] = None,
) -> ArchitecturalDrawing:
    """Generate a high-precision 2D architectural section or elevation drawing from WorldIR."""
    import datetime
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    world_id = getattr(world, "id", "world_default")

    # Determine projection mapping:
    # In section/elevation, vertical axis v is always elevation Y.
    # Horizontal axis u is:
    # - X for SECTION_LONGITUDINAL, ELEVATION_NORTH, ELEVATION_SOUTH (looking along Z)
    # - Z for SECTION_TRANSVERSE, ELEVATION_EAST, ELEVATION_WEST (looking along X)
    is_along_x = drawing_type in (
        DrawingType.SECTION_LONGITUDINAL,
        DrawingType.ELEVATION_NORTH,
        DrawingType.ELEVATION_SOUTH,
    )

    entities = getattr(world, "entities", {})

    # Calculate model bounding box to establish automatic cut plane and projection extent
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    for ent in entities.values():
        cx, cy, cz, w, h, d = _extract_entity_geometry(ent)
        min_x = min(min_x, cx - w / 2.0)
        max_x = max(max_x, cx + w / 2.0)
        min_y = min(min_y, cy - h / 2.0)
        max_y = max(max_y, cy + h / 2.0)
        min_z = min(min_z, cz - d / 2.0)
        max_z = max(max_z, cz + d / 2.0)

    if min_x == float("inf"):
        min_x, max_x = -5.0, 5.0
        min_y, max_y = 0.0, 3.2
        min_z, max_z = -5.0, 5.0

    # Nominal cut plane position
    if is_along_x:
        nominal_cut = (min_z + max_z) / 2.0 if cut_offset_m is None else cut_offset_m
    else:
        nominal_cut = (min_x + max_x) / 2.0 if cut_offset_m is None else cut_offset_m

    lines: List[DrawingLine] = []
    cut_polys: List[DrawingPolygon] = []
    level_heights: set[float] = {0.0}

    # Derive drawing title
    if not drawing_title:
        titles = {
            DrawingType.SECTION_LONGITUDINAL: "BUILDING SECTION A-A (LONGITUDINAL)",
            DrawingType.SECTION_TRANSVERSE: "BUILDING SECTION B-B (TRANSVERSE)",
            DrawingType.ELEVATION_NORTH: "NORTH EXTERIOR ELEVATION",
            DrawingType.ELEVATION_SOUTH: "SOUTH EXTERIOR ELEVATION",
            DrawingType.ELEVATION_EAST: "EAST EXTERIOR ELEVATION",
            DrawingType.ELEVATION_WEST: "WEST EXTERIOR ELEVATION",
            DrawingType.CUSTOM_SECTION: "ARCHITECTURAL SECTION CUT",
        }
        drawing_title = titles.get(drawing_type, "ARCHITECTURAL SECTION")

    # Scale factor
    scale_factors = {"1:20": 0.05, "1:50": 0.02, "1:100": 0.01, "1:200": 0.005}
    s_factor = scale_factors.get(scale_label, 0.01)

    # 1. Process World Elements
    for eid, ent in entities.items():
        etype = getattr(ent, "type", None)
        cx, cy, cz, w, h, d = _extract_entity_geometry(ent)

        y_bot = cy - h / 2.0
        y_top = cy + h / 2.0
        level_heights.add(round(y_bot, 2))
        level_heights.add(round(y_top, 2))

        if is_along_x:
            # Looking along Z. Horizontal coordinate is X.
            u_min = cx - w / 2.0
            u_max = cx + w / 2.0
            norm_pos = cz
            thick = d
        else:
            # Looking along X. Horizontal coordinate is Z.
            u_min = cz - d / 2.0
            u_max = cz + d / 2.0
            norm_pos = cx
            thick = w

        is_cut = False
        # Check if plane cuts through this element
        if abs(norm_pos - nominal_cut) <= (thick / 2.0 + 0.05):
            is_cut = True

        # Classify and draw
        if etype in (EntityType.WALL, "wall", EntityType.STRUCTURE, "structure"):
            if is_cut:
                # Heavy Cut Rectangle
                cut_poly = DrawingPolygon(
                    layer="A-WALL-CUT",
                    vertices=[
                        (u_min, y_bot),
                        (u_max, y_bot),
                        (u_max, y_top),
                        (u_min, y_top),
                    ],
                    fill_hatch=True,
                )
                cut_polys.append(cut_poly)
                # Outer perimeter cut lines (heavy 0.50mm)
                lines.append(DrawingLine("A-WALL-CUT", (u_min, y_bot), (u_max, y_bot), 0.50))
                lines.append(DrawingLine("A-WALL-CUT", (u_max, y_bot), (u_max, y_top), 0.50))
                lines.append(DrawingLine("A-WALL-CUT", (u_max, y_top), (u_min, y_top), 0.50))
                lines.append(DrawingLine("A-WALL-CUT", (u_min, y_top), (u_min, y_bot), 0.50))
            else:
                # Background projection (light 0.18mm)
                if abs(norm_pos - nominal_cut) <= projection_depth_m:
                    lines.append(DrawingLine("A-WALL-PROJ", (u_min, y_bot), (u_max, y_bot), 0.18))
                    lines.append(DrawingLine("A-WALL-PROJ", (u_max, y_bot), (u_max, y_top), 0.18))
                    lines.append(DrawingLine("A-WALL-PROJ", (u_max, y_top), (u_min, y_top), 0.18))
                    lines.append(DrawingLine("A-WALL-PROJ", (u_min, y_top), (u_min, y_bot), 0.18))

        elif etype in (EntityType.FLOOR, "floor", EntityType.ROOF, "roof", EntityType.CEILING, "ceiling"):
            # Horizontal slab cut
            slab_thick = min(0.30, h)
            lines.append(DrawingLine("A-FLOR-CUT", (u_min, y_top), (u_max, y_top), 0.50))
            lines.append(DrawingLine("A-FLOR-CUT", (u_max, y_top), (u_max, y_top - slab_thick), 0.50))
            lines.append(DrawingLine("A-FLOR-CUT", (u_max, y_top - slab_thick), (u_min, y_top - slab_thick), 0.50))
            lines.append(DrawingLine("A-FLOR-CUT", (u_min, y_top - slab_thick), (u_min, y_top), 0.50))
            cut_polys.append(
                DrawingPolygon(
                    layer="A-FLOR-CUT",
                    vertices=[
                        (u_min, y_top),
                        (u_max, y_top),
                        (u_max, y_top - slab_thick),
                        (u_min, y_top - slab_thick),
                    ],
                    fill_hatch=True,
                )
            )

        elif etype in (EntityType.BEAM, "beam"):
            # Structural beam section cut
            beam_depth = min(0.60, h)
            lines.append(DrawingLine("A-BEAM-CUT", (u_min, y_top), (u_max, y_top), 0.40))
            lines.append(DrawingLine("A-BEAM-CUT", (u_max, y_top), (u_max, y_top - beam_depth), 0.40))
            lines.append(DrawingLine("A-BEAM-CUT", (u_max, y_top - beam_depth), (u_min, y_top - beam_depth), 0.40))
            lines.append(DrawingLine("A-BEAM-CUT", (u_min, y_top - beam_depth), (u_min, y_top), 0.40))

        elif etype in (EntityType.DOOR, "door"):
            # Door opening in projection
            lines.append(DrawingLine("A-DOOR", (u_min, y_bot), (u_max, y_bot), 0.25))
            lines.append(DrawingLine("A-DOOR", (u_max, y_bot), (u_max, y_top), 0.25))
            lines.append(DrawingLine("A-DOOR", (u_max, y_top), (u_min, y_top), 0.25))
            lines.append(DrawingLine("A-DOOR", (u_min, y_top), (u_min, y_bot), 0.25))
            # Door swing projection line (dashed)
            lines.append(DrawingLine("A-DOOR", (u_min, y_bot), (u_max, y_top / 2.0), 0.15, is_dashed=True))
            lines.append(DrawingLine("A-DOOR", (u_max, y_top / 2.0), (u_min, y_top), 0.15, is_dashed=True))

        elif etype in (EntityType.WINDOW, "window"):
            # Window opening in projection with mullions
            lines.append(DrawingLine("A-GLAZ", (u_min, y_bot), (u_max, y_bot), 0.25))
            lines.append(DrawingLine("A-GLAZ", (u_max, y_bot), (u_max, y_top), 0.25))
            lines.append(DrawingLine("A-GLAZ", (u_max, y_top), (u_min, y_top), 0.25))
            lines.append(DrawingLine("A-GLAZ", (u_min, y_top), (u_min, y_bot), 0.25))
            # Center transom
            mid_u = (u_min + u_max) / 2.0
            lines.append(DrawingLine("A-GLAZ", (mid_u, y_bot), (mid_u, y_top), 0.18))

    # 2. Ground Terrain Foundation Line
    ground_y = 0.0
    u_left = min_x - 1.5 if is_along_x else min_z - 1.5
    u_right = max_x + 1.5 if is_along_x else max_z + 1.5
    lines.append(DrawingLine("A-SITE-GRND", (u_left, ground_y), (u_right, ground_y), 0.70))

    # 3. Datum Level Markers (Finished Floor Level, Ceiling, Parapet)
    levels: List[LevelMarker] = []
    marker_u = u_left - 0.8
    sorted_levels = sorted(list(level_heights))

    # Keep key architectural datums
    for y_lvl in sorted_levels:
        if abs(y_lvl - 0.0) < 0.05:
            label = "GROUND FLOOR F.F.L."
        elif y_lvl > 2.4 and y_lvl < 3.2:
            label = "CEILING LEVEL C.L."
        elif y_lvl >= 3.2 and y_lvl < 4.0:
            label = "PARAPET / ROOF LEVEL"
        elif y_lvl > 0.0 and abs(y_lvl - round(y_lvl)) < 0.1:
            label = f"LEVEL {int(round(y_lvl))} F.F.L."
        else:
            continue

        levels.append(LevelMarker(label=label, elevation_m=y_lvl, datum_u=marker_u))
        # Thin dashed projection datum guide across the drawing
        lines.append(DrawingLine("A-ANNO-DATM", (marker_u, y_lvl), (u_right + 0.5, y_lvl), 0.15, is_dashed=True))

    # 4. Vertical and Horizontal Dimension Chains
    dimensions: List[DimensionString] = []
    # Vertical storey dimension
    top_y = max(max_y, 2.8)
    dimensions.append(
        DimensionString(
            start=(u_left - 0.4, ground_y),
            end=(u_left - 0.4, top_y),
            offset_u=-0.4,
            offset_v=0.0,
            measured_length_m=top_y - ground_y,
            label=f"{(top_y - ground_y):.2f} m",
        )
    )
    # Horizontal building span dimension
    dimensions.append(
        DimensionString(
            start=(u_left + 1.5, ground_y - 0.6),
            end=(u_right - 1.5, ground_y - 0.6),
            offset_u=0.0,
            offset_v=-0.6,
            measured_length_m=(u_right - u_left - 3.0),
            label=f"{(u_right - u_left - 3.0):.2f} m",
        )
    )

    extent_u = (u_left - 2.0, u_right + 2.0)
    extent_v = (ground_y - 1.5, top_y + 1.5)

    drawing_id = f"dwg_{drawing_type.value[:4]}_{datetime.datetime.now().strftime('%H%M%S')}"

    return ArchitecturalDrawing(
        drawing_id=drawing_id,
        world_id=world_id,
        title=drawing_title,
        drawing_type=drawing_type,
        scale_label=scale_label,
        scale_factor=s_factor,
        sheet_size=sheet_size,
        cut_plane_offset_m=nominal_cut,
        extent_u=extent_u,
        extent_v=extent_v,
        lines=lines,
        cut_polygons=cut_polys,
        levels=levels,
        dimensions=dimensions,
        created_at_iso=created_at,
    )


def export_drawing_to_svg(
    drawing: ArchitecturalDrawing,
    project_name: str = "Architectural Reality Model",
    theme: str = "dark",  # "dark" (studio canvas) or "light" (print sheet)
) -> str:
    """Render drawing to standard vector SVG with title block, scales, and layers."""
    min_u, max_u = drawing.extent_u
    min_v, max_v = drawing.extent_v
    u_span = max(1.0, max_u - min_u)
    v_span = max(1.0, max_v - min_v)

    # SVG sheet dimensions in points/pixels (A2 default: 1190 x 842 pt)
    SHEET_W = 1190.0
    SHEET_H = 842.0
    MARGIN = 50.0

    # Usable drawing viewport area
    viewport_w = SHEET_W - 2 * MARGIN
    viewport_h = SHEET_H - 2 * MARGIN - 70.0  # Reserve 70pt for title block at bottom

    scale_u = viewport_w / u_span
    scale_v = viewport_h / v_span
    view_scale = min(scale_u, scale_v) * 0.90

    # Center drawing in viewport
    center_u = (min_u + max_u) / 2.0
    center_v = (min_v + max_v) / 2.0
    origin_x = SHEET_W / 2.0 - center_u * view_scale
    origin_y = (MARGIN + viewport_h / 2.0) + center_v * view_scale

    def to_svg(u: float, v: float) -> Tuple[float, float]:
        sx = origin_x + u * view_scale
        sy = origin_y - v * view_scale  # Flip Y for SVG downwards axis
        return sx, sy

    # Theme colors
    if theme == "dark":
        bg_sheet = "#0d1117"
        bg_border = "#30363d"
        text_primary = "#f0f6fc"
        text_muted = "#8b949e"
        stroke_cut = "#ffffff"
        stroke_proj = "#7ee787"
        stroke_glaz = "#79c0ff"
        stroke_dims = "#d2a8ff"
        stroke_ground = "#ffa657"
        hatch_fill = "rgba(255, 255, 255, 0.08)"
    else:
        bg_sheet = "#ffffff"
        bg_border = "#222222"
        text_primary = "#111111"
        text_muted = "#666666"
        stroke_cut = "#000000"
        stroke_proj = "#444444"
        stroke_glaz = "#0066cc"
        stroke_dims = "#660099"
        stroke_ground = "#8b4513"
        hatch_fill = "rgba(0, 0, 0, 0.06)"

    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SHEET_W:.1f} {SHEET_H:.1f}" '
        f'width="{SHEET_W:.1f}" height="{SHEET_H:.1f}" style="background-color: {bg_sheet}; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;">'
    )

    # Defs: Hatch pattern
    svg_parts.append(
        f'<defs>'
        f'<pattern id="arch_hatch" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">'
        f'<line x1="0" y1="0" x2="0" y2="8" stroke="{stroke_cut}" stroke-width="0.8" opacity="0.3"/>'
        f'</pattern>'
        f'</defs>'
    )

    # Sheet Border & Margins
    svg_parts.append(
        f'<rect x="{MARGIN:.1f}" y="{MARGIN:.1f}" width="{SHEET_W - 2 * MARGIN:.1f}" height="{SHEET_H - 2 * MARGIN:.1f}" '
        f'fill="none" stroke="{bg_border}" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<rect x="{MARGIN + 6:.1f}" y="{MARGIN + 6:.1f}" width="{SHEET_W - 2 * MARGIN - 12:.1f}" height="{SHEET_H - 2 * MARGIN - 12:.1f}" '
        f'fill="none" stroke="{bg_border}" stroke-width="0.5"/>'
    )

    # Cut Polygons (with architectural diagonal hatching)
    svg_parts.append('<g id="layer-cuts-hatch">')
    for poly in drawing.cut_polygons:
        pts_str = " ".join(f"{to_svg(vx, vy)[0]:.2f},{to_svg(vx, vy)[1]:.2f}" for vx, vy in poly.vertices)
        svg_parts.append(f'<polygon points="{pts_str}" fill="url(#arch_hatch)" stroke="none"/>')
    svg_parts.append('</g>')

    # Vector Lines
    svg_parts.append('<g id="layer-vector-lines">')
    for line in drawing.lines:
        x1, y1 = to_svg(line.start[0], line.start[1])
        x2, y2 = to_svg(line.end[0], line.end[1])

        # Layer color
        if "CUT" in line.layer:
            col = stroke_cut
            w_px = max(1.2, line.stroke_width_mm * 2.8)
        elif "PROJ" in line.layer:
            col = stroke_proj
            w_px = 0.8
        elif "GLAZ" in line.layer:
            col = stroke_glaz
            w_px = 1.0
        elif "GRND" in line.layer:
            col = stroke_ground
            w_px = 2.4
        elif "DATM" in line.layer:
            col = text_muted
            w_px = 0.6
        else:
            col = stroke_dims
            w_px = 0.8

        dash_attr = 'stroke-dasharray="4,3"' if line.is_dashed else ""
        svg_parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{col}" stroke-width="{w_px:.2f}" {dash_attr}/>'
        )
    svg_parts.append('</g>')

    # Level Markers
    svg_parts.append('<g id="layer-level-markers">')
    for lvl in drawing.levels:
        lx, ly = to_svg(lvl.datum_u, lvl.elevation_m)
        # Architectural datum target symbol: inverted triangle + circle
        svg_parts.append(
            f'<polygon points="{lx:.2f},{ly:.2f} {lx - 10:.2f},{ly - 8:.2f} {lx - 10:.2f},{ly + 8:.2f}" '
            f'fill="{stroke_cut}" stroke="{stroke_cut}" stroke-width="0.5"/>'
        )
        elev_str = f"{'+' if lvl.elevation_m > 0 else ''}{lvl.elevation_m:.3f}"
        svg_parts.append(
            f'<text x="{lx - 16:.2f}" y="{ly - 3:.2f}" fill="{text_primary}" font-size="10" font-weight="bold" text-anchor="end">{lvl.label}</text>'
        )
        svg_parts.append(
            f'<text x="{lx - 16:.2f}" y="{ly + 9:.2f}" fill="{text_muted}" font-size="9" text-anchor="end">{elev_str} m</text>'
        )
    svg_parts.append('</g>')

    # Dimension Strings
    svg_parts.append('<g id="layer-dimension-strings">')
    for dim in drawing.dimensions:
        x1, y1 = to_svg(dim.start[0], dim.start[1])
        x2, y2 = to_svg(dim.end[0], dim.end[1])
        # Dimension witness line
        svg_parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke_dims}" stroke-width="0.8"/>')
        # Ticks
        svg_parts.append(f'<line x1="{x1 - 4:.2f}" y1="{y1 - 4:.2f}" x2="{x1 + 4:.2f}" y2="{y1 + 4:.2f}" stroke="{stroke_dims}" stroke-width="1.2"/>')
        svg_parts.append(f'<line x1="{x2 - 4:.2f}" y1="{y2 - 4:.2f}" x2="{x2 + 4:.2f}" y2="{y2 + 4:.2f}" stroke="{stroke_dims}" stroke-width="1.2"/>')
        # Text at center
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0 - 5.0
        svg_parts.append(f'<text x="{mid_x:.2f}" y="{mid_y:.2f}" fill="{stroke_dims}" font-size="10" text-anchor="middle" font-weight="bold">{dim.label}</text>')
    svg_parts.append('</g>')

    # Standard Title Block at Bottom Right (300 x 65 pt)
    tb_x = SHEET_W - MARGIN - 300.0
    tb_y = SHEET_H - MARGIN - 65.0
    svg_parts.append('<g id="title-block">')
    svg_parts.append(f'<rect x="{tb_x:.1f}" y="{tb_y:.1f}" width="294" height="59" fill="{bg_sheet}" stroke="{bg_border}" stroke-width="1.2"/>')
    svg_parts.append(f'<line x1="{tb_x:.1f}" y1="{tb_y + 28:.1f}" x2="{tb_x + 294:.1f}" y2="{tb_y + 28:.1f}" stroke="{bg_border}" stroke-width="0.8"/>')
    svg_parts.append(f'<line x1="{tb_x + 190:.1f}" y1="{tb_y:.1f}" x2="{tb_x + 190:.1f}" y2="{tb_y + 59:.1f}" stroke="{bg_border}" stroke-width="0.8"/>')

    # Project & Sheet Text
    svg_parts.append(f'<text x="{tb_x + 8:.1f}" y="{tb_y + 14:.1f}" fill="{text_muted}" font-size="8" font-weight="bold">PROJECT</text>')
    svg_parts.append(f'<text x="{tb_x + 8:.1f}" y="{tb_y + 24:.1f}" fill="{text_primary}" font-size="11" font-weight="bold">{project_name[:28]}</text>')
    svg_parts.append(f'<text x="{tb_x + 8:.1f}" y="{tb_y + 42:.1f}" fill="{text_muted}" font-size="8" font-weight="bold">DRAWING TITLE</text>')
    svg_parts.append(f'<text x="{tb_x + 8:.1f}" y="{tb_y + 53:.1f}" fill="{text_primary}" font-size="10" font-weight="bold">{drawing.title[:30]}</text>')

    # Right block: Scale & Sheet No
    svg_parts.append(f'<text x="{tb_x + 198:.1f}" y="{tb_y + 14:.1f}" fill="{text_muted}" font-size="8" font-weight="bold">SCALE: {drawing.scale_label}</text>')
    svg_parts.append(f'<text x="{tb_x + 198:.1f}" y="{tb_y + 24:.1f}" fill="{text_muted}" font-size="8">{drawing.sheet_size.value} SHEET</text>')
    svg_parts.append(f'<text x="{tb_x + 198:.1f}" y="{tb_y + 44:.1f}" fill="{text_muted}" font-size="8" font-weight="bold">DRAWING NO.</text>')
    svg_parts.append(f'<text x="{tb_x + 198:.1f}" y="{tb_y + 55:.1f}" fill="{stroke_proj}" font-size="12" font-weight="bold">{drawing.drawing_id[:10].upper()}</text>')
    svg_parts.append('</g>')

    # Drawing Title Bar under section
    svg_parts.append(
        f'<text x="{MARGIN + 12:.1f}" y="{SHEET_H - MARGIN - 20:.1f}" fill="{text_primary}" font-size="14" font-weight="bold">{drawing.title}</text>'
    )
    svg_parts.append(
        f'<text x="{MARGIN + 12:.1f}" y="{SHEET_H - MARGIN - 8:.1f}" fill="{text_muted}" font-size="10">SCALE {drawing.scale_label}  |  REALITY ENGINE BIM PERCEPTION</text>'
    )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def export_drawing_to_dxf(
    drawing: ArchitecturalDrawing,
    project_name: str = "Architectural Section",
) -> str:
    """Export drawing to standard AutoCAD DXF (R12 / R2000 compatible ASCII DXF format)."""
    dxf_lines: List[str] = []

    # 1. HEADER SECTION
    dxf_lines.extend(["0", "SECTION", "2", "HEADER"])
    dxf_lines.extend(["9", "$ACADVER", "1", "AC1009"])  # AutoCAD R11/R12 standard
    dxf_lines.extend(["9", "$INSUNITS", "70", "6"])      # 6 = Meters
    dxf_lines.extend(["0", "ENDSEC"])

    # 2. TABLES SECTION (Architectural Layers with AIA Cad standards)
    dxf_lines.extend(["0", "SECTION", "2", "TABLES"])
    dxf_lines.extend(["0", "TABLE", "2", "LAYER", "70", "8"])

    layers = [
        ("0", 7),
        ("A-WALL-CUT", 7),     # White / Heavy
        ("A-WALL-PROJ", 8),    # Gray / Light
        ("A-FLOR-CUT", 7),     # White
        ("A-BEAM-CUT", 3),     # Green
        ("A-DOOR", 4),         # Cyan
        ("A-GLAZ", 5),         # Blue
        ("A-SITE-GRND", 1),    # Red
        ("A-DIMS", 6),         # Magenta
        ("A-ANNO-DATM", 2),    # Yellow
    ]

    for lay_name, color in layers:
        dxf_lines.extend([
            "0", "LAYER",
            "2", lay_name,
            "70", "0",
            "62", str(color),
            "6", "CONTINUOUS",
        ])
    dxf_lines.extend(["0", "ENDTAB"])
    dxf_lines.extend(["0", "ENDSEC"])

    # 3. ENTITIES SECTION
    dxf_lines.extend(["0", "SECTION", "2", "ENTITIES"])

    # Lines
    for line in drawing.lines:
        u1, v1 = line.start
        u2, v2 = line.end
        dxf_lines.extend([
            "0", "LINE",
            "8", line.layer,
            "10", f"{u1:.4f}",
            "20", f"{v1:.4f}",
            "30", "0.0",
            "11", f"{u2:.4f}",
            "21", f"{v2:.4f}",
            "31", "0.0",
        ])

    # Cut Polygons (as closed LWPOLYLINE)
    for poly in drawing.cut_polygons:
        v_count = len(poly.vertices)
        dxf_lines.extend([
            "0", "POLYLINE",
            "8", poly.layer,
            "66", "1",
            "70", "1",  # 1 = Closed
            "10", "0.0", "20", "0.0", "30", "0.0",
        ])
        for vx, vy in poly.vertices:
            dxf_lines.extend([
                "0", "VERTEX",
                "8", poly.layer,
                "10", f"{vx:.4f}",
                "20", f"{vy:.4f}",
                "30", "0.0",
            ])
        dxf_lines.extend(["0", "SEQEND"])

    # Level Markers Text
    for lvl in drawing.levels:
        elev_str = f"{lvl.label} ({'+' if lvl.elevation_m > 0 else ''}{lvl.elevation_m:.3f}m)"
        dxf_lines.extend([
            "0", "TEXT",
            "8", "A-ANNO-DATM",
            "10", f"{lvl.datum_u:.4f}",
            "20", f"{lvl.elevation_m:.4f}",
            "30", "0.0",
            "40", "0.20",  # Text height
            "1", elev_str,
        ])

    # Dimensions
    for dim in drawing.dimensions:
        mid_u = (dim.start[0] + dim.end[0]) / 2.0
        mid_v = (dim.start[1] + dim.end[1]) / 2.0 + 0.1
        dxf_lines.extend([
            "0", "TEXT",
            "8", "A-DIMS",
            "10", f"{mid_u:.4f}",
            "20", f"{mid_v:.4f}",
            "30", "0.0",
            "40", "0.18",
            "1", dim.label,
        ])

    # Title Block Text
    min_u, _ = drawing.extent_u
    min_v, _ = drawing.extent_v
    dxf_lines.extend([
        "0", "TEXT",
        "8", "A-ANNO-DATM",
        "10", f"{min_u:.4f}",
        "20", f"{min_v - 0.5:.4f}",
        "30", "0.0",
        "40", "0.35",
        "1", f"{project_name} - {drawing.title} (SCALE {drawing.scale_label})",
    ])

    dxf_lines.extend(["0", "ENDSEC"])
    dxf_lines.extend(["0", "EOF\n"])

    return "\n".join(dxf_lines)
