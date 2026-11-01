"""2D floor plan (SVG + DXF) and a room report (PDF) with wall areas net of openings (plan F8).

Everything is read from the canonical WorldIR: room footprint polygons, solid wall slabs, the openings carved into
them, and the D1 dimension objects. No geometry is regenerated here and no measurement is invented -- a dimension
without a measured value is printed as "not measured", never as a plausible number.

* ``export_to_floorplan_svg_with_report`` -- a paper-style plan: walls as solid bands, door/window gaps and symbols,
  dimension lines carrying the dimension STATE, unobserved/inferred walls dashed, and a legend.
* ``export_to_floorplan_dxf_with_report`` -- the same plan as an ASCII DXF (R12): layers ROOM/WALL/OPENING/DIMENSION/
  TEXT/UNOBSERVED. Loadable by CAD without a Python dependency.
* ``export_to_room_report_pdf_with_report`` -- a compact multi-page PDF (base-14 Helvetica, hand-built; no dependency):
  per room area, height, gross and net wall area after subtracting openings, opening counts, colour/material basis and
  a confidence summary.

Units: the plan is drawn in the room's plan metres when the world is metric; a RELATIVE world is drawn in model units
and the legend says so (it never claims metres).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from exporters.report import ExportReport, content_hash
from world_ir import EntityType

SVG_MM_PER_M = 20.0          # 1:50 on paper
SVG_MARGIN_M = 0.8
DIM_OFFSET_M = 0.45
DOOR_ARC_DEG = 90.0


def _scale_units(world) -> Tuple[str, float]:
    scale = (getattr(world, "metadata", None) or {}).get("scale") or {}
    if scale.get("state") == "metric":
        return "m", 1.0
    return "model units", 1.0


def _basis(room) -> Tuple[List[float], List[float]]:
    b = (room.custom_properties or {}).get("footprint_basis") or {}
    e1 = b.get("e1") or [1.0, 0.0, 0.0]
    e2 = b.get("e2") or [0.0, 1.0, 0.0]
    return [float(v) for v in e1], [float(v) for v in e2]


def _project(point_xyz: Sequence[float], e1: Sequence[float], e2: Sequence[float]) -> Tuple[float, float]:
    p = [float(v) for v in point_xyz]
    return (sum(a * b for a, b in zip(p, e1)), sum(a * b for a, b in zip(p, e2)))


def _rooms(world) -> List[object]:
    return [e for e in world.entities.values() if e.type == EntityType.ROOM
            and (e.custom_properties or {}).get("footprint_polygon_m")
            and not (e.custom_properties or {}).get("superseded_by")]


def _poly_area(poly: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for i, (x, y) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        s += float(x) * float(y2) - float(x2) * float(y)
    return abs(s) / 2.0


def _room_walls(world, room) -> List[dict]:
    e1, e2 = _basis(room)
    out: List[dict] = []
    for ent in world.entities.values():
        cp = ent.custom_properties or {}
        if ent.type != EntityType.WALL or cp.get("room_id") != room.id or not cp.get("wall_slab"):
            continue
        s = cp["wall_slab"]
        a = _project(s.get("start_xyz_m"), e1, e2)
        b = _project(s.get("end_xyz_m"), e1, e2)
        out.append({
            "id": ent.id, "a": a, "b": b, "length_m": float(cp.get("length_m") or math.dist(a, b)),
            "height_m": float(cp.get("height_m") or 0.0), "thickness_m": float(cp.get("thickness_m") or 0.1),
            "confidence": ent.confidence, "observed_fraction": cp.get("observed_fraction"),
            "inferred": "inferred_wall_segment" in (ent.semantic_labels or []) or str(ent.provenance.value) == "INFERRED",
            "openings": list(s.get("openings") or []),
            "dimension": (cp.get("dimensions") or {}).get("length_m"),
        })
    return sorted(out, key=lambda w: math.atan2(w["b"][1] - w["a"][1], w["b"][0] - w["a"][0]))


def _world_extent(rooms, walls) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for room in rooms:
        for p in room.custom_properties["footprint_polygon_m"]:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    for w in walls:
        xs += [w["a"][0], w["b"][0]]
        ys += [w["a"][1], w["b"][1]]
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def _svg_num(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def _svg_text(x: float, y: float, text: str, size: float, *, anchor: str = "middle", fill: str = "#111",
              weight: str = "normal") -> str:
    return (f'<text x="{_svg_num(x)}" y="{_svg_num(y)}" font-size="{_svg_num(size)}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="Helvetica, Arial, sans-serif">{text}</text>')


def _dim_label(dim: Optional[dict], unit: str) -> str:
    if not isinstance(dim, dict) or dim.get("value") is None:
        return "not measured"
    return f"{float(dim['value']):.2f} {unit} · {dim.get('state', 'UNKNOWN')}"


def _wall_line_with_gaps(wall: dict, unit_scale: float) -> str:
    """Wall band + door gaps/window panes. Offsets in metres along the wall."""
    a, b = wall["a"], wall["b"]
    length = max(1e-9, wall["length_m"])
    d = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    thick = max(0.06, wall["thickness_m"])
    parts: List[str] = []
    # wall body split around openings so a door reads as a real gap and a window as a glazed band
    edges = sorted((max(0.0, float(o.get("offset_m", 0.0))),
                    min(length, float(o.get("offset_m", 0.0)) + float(o.get("width_m", 0.0))))
                   for o in wall["openings"])
    cursor = 0.0
    for start, end in edges:
        if start > cursor:
            p0 = (a[0] + d[0] * cursor, a[1] + d[1] * cursor)
            p1 = (a[0] + d[0] * start, a[1] + d[1] * start)
            parts.append(f'<line x1="{_svg_num(p0[0])}" y1="{_svg_num(p0[1])}" x2="{_svg_num(p1[0])}" '
                         f'y2="{_svg_num(p1[1])}" stroke="#2b2b2b" stroke-width="{_svg_num(thick)}" '
                         f'stroke-linecap="butt"/>')
        cursor = max(cursor, end)
    if cursor < length:
        p0 = (a[0] + d[0] * cursor, a[1] + d[1] * cursor)
        parts.append(f'<line x1="{_svg_num(p0[0])}" y1="{_svg_num(p0[1])}" x2="{_svg_num(b[0])}" '
                     f'y2="{_svg_num(b[1])}" stroke="#2b2b2b" stroke-width="{_svg_num(thick)}" '
                     f'stroke-linecap="butt"/>')
    for o in wall["openings"]:
        off, width = float(o.get("offset_m", 0.0)), float(o.get("width_m", 0.0))
        kind = str(o.get("kind", "door")).lower()
        p0 = (a[0] + d[0] * off, a[1] + d[1] * off)
        p1 = (a[0] + d[0] * (off + width), a[1] + d[1] * (off + width))
        if "window" in kind:
            parts.append(f'<line x1="{_svg_num(p0[0])}" y1="{_svg_num(p0[1])}" x2="{_svg_num(p1[0])}" '
                         f'y2="{_svg_num(p1[1])}" stroke="#2f6fb0" stroke-width="{_svg_num(thick * 0.35)}"/>')
        else:
            # a door swing arc, drawn from the opening's start
            r = width
            sweep = 1 if (d[0] * 0 + d[1]) >= 0 else 0
            end = (p0[0] - d[1] * r * (-1 if sweep else 1), p0[1] + d[0] * r * (-1 if sweep else 1))
            parts.append(f'<path d="M {_svg_num(p0[0])} {_svg_num(p0[1])} L {_svg_num(end[0])} {_svg_num(end[1])}" '
                         f'fill="none" stroke="#8a6d3b" stroke-width="0.03"/>')
            parts.append(f'<path d="M {_svg_num(p0[0] + d[0] * r)} {_svg_num(p0[1] + d[1] * r)} '
                         f'A {_svg_num(r)} {_svg_num(r)} 0 0 {sweep} {_svg_num(end[0])} {_svg_num(end[1])}" '
                         f'fill="none" stroke="#8a6d3b" stroke-width="0.015"/>')
    if wall.get("inferred"):
        parts.append(f'<line x1="{_svg_num(a[0])}" y1="{_svg_num(a[1])}" x2="{_svg_num(b[0])}" '
                     f'y2="{_svg_num(b[1])}" stroke="#d97706" stroke-width="{_svg_num(thick)}" '
                     f'stroke-dasharray="0.12 0.08" opacity="0.75"/>')
    return "".join(parts)


def export_to_floorplan_svg_with_report(world, artifact_store=None) -> Tuple[str, ExportReport]:
    unit, _ = _scale_units(world)
    rooms = _rooms(world)
    walls = [w for r in rooms for w in _room_walls(world, r)]
    if not rooms:
        raise ValueError("floor plan export needs at least one room with a measured footprint")
    minx, miny, maxx, maxy = _world_extent(rooms, walls)
    pad = SVG_MARGIN_M
    vb = (minx - pad, miny - pad, (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_svg_num(vb[0])} {_svg_num(vb[1])} '
        f'{_svg_num(vb[2])} {_svg_num(vb[3])}" width="{_svg_num(vb[2] * SVG_MM_PER_M)}mm" '
        f'height="{_svg_num(vb[3] * SVG_MM_PER_M)}mm" role="img" aria-label="Measured floor plan">',
        f'<rect x="{_svg_num(vb[0])}" y="{_svg_num(vb[1])}" width="{_svg_num(vb[2])}" height="{_svg_num(vb[3])}" '
        f'fill="#ffffff"/>',
    ]
    for room in rooms:
        cp = room.custom_properties
        poly = " ".join(f"{_svg_num(float(p[0]))},{_svg_num(float(p[1]))}" for p in cp["footprint_polygon_m"])
        parts.append(f'<polygon points="{poly}" fill="#f4f1ea" stroke="#111111" stroke-width="0.03"/>')
        cx = sum(float(p[0]) for p in cp["footprint_polygon_m"]) / len(cp["footprint_polygon_m"])
        cy = sum(float(p[1]) for p in cp["footprint_polygon_m"]) / len(cp["footprint_polygon_m"])
        area = float(cp.get("floor_area_m2") or _poly_area(cp["footprint_polygon_m"]))
        parts.append(_svg_text(cx, cy, f"{room.name} · {area:.2f} {unit}²", 0.16, weight="bold"))
    for wall in walls:
        parts.append(_wall_line_with_gaps(wall, 1.0))
        if wall["dimension"]:
            mid = ((wall["a"][0] + wall["b"][0]) / 2, (wall["a"][1] + wall["b"][1]) / 2)
            parts.append(_svg_text(mid[0], mid[1] - 0.10, _dim_label(wall["dimension"], unit), 0.10,
                                   fill="#334155"))
    # overall dimension lines per room
    for room in rooms:
        cp = room.custom_properties
        xs = [float(p[0]) for p in cp["footprint_polygon_m"]]
        ys = [float(p[1]) for p in cp["footprint_polygon_m"]]
        dims = cp.get("dimensions") or {}
        uw = dims.get("layout_dimension_u_m") or dims.get("floor_dimension_u_m")
        vw = dims.get("layout_dimension_v_m") or dims.get("floor_dimension_v_m")
        y = min(ys) - DIM_OFFSET_M
        parts.append(f'<path d="M {_svg_num(min(xs))} {_svg_num(min(ys) - 0.12)} L {_svg_num(min(xs))} {_svg_num(y)} '
                     f'M {_svg_num(max(xs))} {_svg_num(min(ys) - 0.12)} L {_svg_num(max(xs))} {_svg_num(y)} '
                     f'M {_svg_num(min(xs))} {_svg_num(y)} L {_svg_num(max(xs))} {_svg_num(y)}" stroke="#2563eb" '
                     f'stroke-width="0.015" fill="none"/>')
        parts.append(_svg_text((min(xs) + max(xs)) / 2, y - 0.08, _dim_label(uw, unit), 0.11, fill="#1d4ed8"))
        x = min(xs) - DIM_OFFSET_M
        parts.append(f'<path d="M {_svg_num(min(xs) - 0.12)} {_svg_num(min(ys))} L {_svg_num(x)} {_svg_num(min(ys))} '
                     f'M {_svg_num(min(xs) - 0.12)} {_svg_num(max(ys))} L {_svg_num(x)} {_svg_num(max(ys))} '
                     f'M {_svg_num(x)} {_svg_num(min(ys))} L {_svg_num(x)} {_svg_num(max(ys))}" stroke="#2563eb" '
                     f'stroke-width="0.015" fill="none"/>')
        parts.append(f'<text x="{_svg_num(x - 0.07)}" y="{_svg_num((min(ys) + max(ys)) / 2)}" font-size="0.11" '
                     f'fill="#1d4ed8" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
                     f'transform="rotate(-90 {_svg_num(x - 0.07)} {_svg_num((min(ys) + max(ys)) / 2)})">'
                     f'{_dim_label(vw, unit)}</text>')
    # legend -- states, not decoration
    lx, ly = maxx + 0.25, maxy
    legend = [
        ("Solid wall", "#2b2b2b"), ("Inferred / unobserved", "#d97706"), ("Window", "#2f6fb0"),
        ("Door swing", "#8a6d3b"), ("Dimension (state shown)", "#2563eb"),
    ]
    parts.append(_svg_text(lx, ly + 0.05, "Legend", 0.13, anchor="start", weight="bold"))
    for i, (label, colour) in enumerate(legend):
        yy = ly - 0.12 * (i + 1)
        parts.append(f'<line x1="{_svg_num(lx)}" y1="{_svg_num(yy)}" x2="{_svg_num(lx + 0.35)}" '
                     f'y2="{_svg_num(yy)}" stroke="{colour}" stroke-width="0.04"/>')
        parts.append(_svg_text(lx + 0.42, yy + 0.04, label, 0.10, anchor="start"))
    parts.append(_svg_text(vb[0] + 0.1, vb[1] + vb[3] - 0.12,
                           f"Measured floor plan · units: {unit} · 1:50 · every dimension states its evidence",
                           0.11, anchor="start", fill="#475569"))
    parts.append("</svg>")
    svg = "\n".join(parts)
    report = ExportReport(
        format="floorplan_svg", world_id=world.id, world_version=world.version,
        entities_exported=tuple([r.id for r in rooms] + [w["id"] for w in walls]),
        entities_skipped=tuple(e.id for e in world.entities.values() if e.type not in
                               (EntityType.ROOM, EntityType.WALL, EntityType.DOOR, EntityType.WINDOW)),
        skip_reasons=("not a room/wall/opening: the plan exports structure only",) * len(
            [e for e in world.entities.values() if e.type not in (EntityType.ROOM, EntityType.WALL,
                                                                  EntityType.DOOR, EntityType.WINDOW)]),
        content_hash=content_hash(svg),
    )
    return svg, report


def _dxf_header() -> List[str]:
    return [
        "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", "7",
        *sum((["0", "LAYER", "2", name, "70", "0", "62", str(colour), "6", "CONTINUOUS"]
              for name, colour in (("ROOM", 7), ("WALL", 7), ("OPENING", 5), ("DIMENSION", 5),
                                   ("TEXT", 7), ("UNOBSERVED", 30))), []),
        "0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES",
    ]


def _dxf_line(layer: str, a: Sequence[float], b: Sequence[float]) -> List[str]:
    return ["0", "LINE", "8", layer, "10", f"{a[0]:.4f}", "20", f"{a[1]:.4f}", "30", "0",
            "11", f"{b[0]:.4f}", "21", f"{b[1]:.4f}", "31", "0"]


def _dxf_text(x: float, y: float, text: str, height: float = 0.12) -> List[str]:
    safe = str(text).replace("\n", " ")
    return ["0", "TEXT", "8", "TEXT", "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0", "40", f"{height:.3f}",
            "1", safe]


def export_to_floorplan_dxf_with_report(world, artifact_store=None) -> Tuple[str, ExportReport]:
    unit, _ = _scale_units(world)
    rooms = _rooms(world)
    walls = [w for r in rooms for w in _room_walls(world, r)]
    if not rooms:
        raise ValueError("floor plan export needs at least one room with a measured footprint")
    lines: List[str] = _dxf_header()
    for room in rooms:
        poly = room.custom_properties["footprint_polygon_m"]
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            lines += _dxf_line("ROOM", p, q)
        cx = sum(float(p[0]) for p in poly) / len(poly)
        cy = sum(float(p[1]) for p in poly) / len(poly)
        area = float(room.custom_properties.get("floor_area_m2") or _poly_area(poly))
        lines += _dxf_text(cx, cy, f"{room.name} {area:.2f} {unit}2", 0.14)
    for wall in walls:
        a, b = wall["a"], wall["b"]
        length = max(1e-9, wall["length_m"])
        d = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        n = (-d[1], d[0])
        half = max(0.03, wall["thickness_m"] / 2)
        corners = [(a[0] + n[0] * half, a[1] + n[1] * half), (b[0] + n[0] * half, b[1] + n[1] * half),
                   (b[0] - n[0] * half, b[1] - n[1] * half), (a[0] - n[0] * half, a[1] - n[1] * half)]
        layer = "UNOBSERVED" if wall.get("inferred") or (wall.get("observed_fraction") or 1.0) < 0.5 else "WALL"
        for i, p in enumerate(corners):
            lines += _dxf_line(layer, p, corners[(i + 1) % len(corners)])
        for o in wall["openings"]:
            off, width = float(o.get("offset_m", 0.0)), float(o.get("width_m", 0.0))
            p0 = (a[0] + d[0] * off, a[1] + d[1] * off)
            p1 = (a[0] + d[0] * (off + width), a[1] + d[1] * (off + width))
            lines += _dxf_line("OPENING", p0, p1)
        if wall["dimension"]:
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            lines += _dxf_text(mid[0], mid[1], _dim_label(wall["dimension"], unit), 0.10)
    # room dimensions
    for room in rooms:
        cp = room.custom_properties
        xs = [float(p[0]) for p in cp["footprint_polygon_m"]]
        ys = [float(p[1]) for p in cp["footprint_polygon_m"]]
        dims = cp.get("dimensions") or {}
        for key, position, label in (
            ("layout_dimension_u_m", ((min(xs) + max(xs)) / 2, min(ys) - DIM_OFFSET_M), "L"),
            ("layout_dimension_v_m", (min(xs) - DIM_OFFSET_M, (min(ys) + max(ys)) / 2), "W"),
        ):
            dim = dims.get(key) or dims.get(key.replace("layout_", "floor_"))
            if dim:
                lines += _dxf_text(position[0], position[1], f"{label} {_dim_label(dim, unit)}", 0.11)
    lines += ["0", "ENDSEC", "0", "EOF"]
    dxf = "\n".join(lines) + "\n"
    report = ExportReport(
        format="floorplan_dxf", world_id=world.id, world_version=world.version,
        entities_exported=tuple([r.id for r in rooms] + [w["id"] for w in walls]),
        entities_skipped=(), skip_reasons=(), content_hash=content_hash(dxf),
    )
    return dxf, report


# --------------------------------------------------------------------------------------------------- room report

def build_room_report(world) -> dict:
    """Machine-readable room report: per room area, height, gross/net wall area, openings, colours and confidence."""
    unit, _ = _scale_units(world)
    rooms = _rooms(world)
    if not rooms:
        raise ValueError("room report needs at least one room with a measured footprint")
    out = {"schema": "room_report/1", "world_id": world.id, "world_version": world.version, "units": unit,
           "scale_state": (((world.metadata or {}).get("scale") or {}).get("state") or "relative"),
           "rooms": []}
    for room in rooms:
        cp = room.custom_properties
        walls = _room_walls(world, room)
        gross = sum(w["length_m"] * w["height_m"] for w in walls)
        opening_area = 0.0
        counts: Dict[str, int] = {}
        for w in walls:
            for o in w["openings"]:
                opening_area += float(o.get("width_m", 0.0)) * float(o.get("height_m", 0.0))
                kind = str(o.get("kind", "opening")).lower()
                counts[kind] = counts.get(kind, 0) + 1
        dims = cp.get("dimensions") or {}
        states = sorted({d.get("state") for d in dims.values() if isinstance(d, dict) and d.get("state")})
        confs = [float(w["confidence"]) for w in walls if w.get("confidence") is not None]
        out["rooms"].append({
            "id": room.id, "name": room.name,
            "area_m2": round(float(cp.get("floor_area_m2") or _poly_area(cp["footprint_polygon_m"])), 3),
            "perimeter_m": round(sum(w["length_m"] for w in walls), 3),
            "height_m": cp.get("height_m"),
            "height_layers_m": (cp.get("dimensional_consistency") or {}).get("height_layers_m") or cp.get("layout_height_layers_m"),
            "wall_area_gross_m2": round(gross, 3),
            "opening_area_m2": round(opening_area, 3),
            "wall_area_net_m2": round(max(0.0, gross - opening_area), 3),
            "walls": len(walls), "openings": counts,
            "floor_color_rgb": cp.get("floor_color_rgb"), "wall_color_rgb": cp.get("wall_color_rgb"),
            "color_basis": cp.get("color_source") or cp.get("color_state") or "measured from points" if (
                cp.get("mean_color_rgb") or cp.get("wall_color_rgb")) else "not measured",
            "dimension_states": states,
            "confidence_mean": round(sum(confs) / len(confs), 3) if confs else None,
            "inferred_walls": sum(1 for w in walls if w.get("inferred")),
            "unobserved_walls": [w["id"] for w in walls if (w.get("observed_fraction") or 1.0) < 0.5],
        })
    return out


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def report_to_pdf(report: dict) -> bytes:
    """A compact PDF (hand-built, base-14 Helvetica) so the report needs no runtime dependency."""
    unit = report.get("units", "m")
    lines: List[Tuple[str, int, bool]] = []        # (text, font size, bold-ish via Helvetica-Bold)
    lines.append((f"Room report — {report.get('world_id', 'world')}", 16, True))
    lines.append((f"Scale: {report.get('scale_state', 'relative')} · units: {unit}", 9, False))
    lines.append(("", 6, False))
    for r in report.get("rooms", []):
        lines.append((f"Room {r.get('name', r.get('id'))}", 13, True))
        lines.append((f"Floor area {r['area_m2']:.2f} {unit}²   Perimeter {r['perimeter_m']:.2f} {unit}", 10, False))
        layers = r.get("height_layers_m") or []
        lines.append((f"Height {r.get('height_m')} {unit}" + (f" (layers {layers})" if layers else ""), 10, False))
        lines.append((f"Wall area gross {r['wall_area_gross_m2']:.2f} {unit}²  − openings "
                      f"{r['opening_area_m2']:.2f} {unit}²  = net {r['wall_area_net_m2']:.2f} {unit}²", 10, False))
        lines.append((f"Walls {r['walls']} ({r['inferred_walls']} inferred) · openings "
                      f"{r['openings'] or 'none'}", 10, False))
        lines.append((f"Dimension states: {', '.join(r.get('dimension_states') or []) or 'none recorded'}", 10, False))
        lines.append((f"Colour: {r.get('color_basis')} · wall {r.get('wall_color_rgb')} · floor "
                      f"{r.get('floor_color_rgb')}", 9, False))
        conf = r.get("confidence_mean")
        lines.append((f"Mean wall confidence: {'not scored' if conf is None else f'{conf:.2f}'}", 9, False))
        if r.get("unobserved_walls"):
            lines.append((f"Walls with weak coverage: {', '.join(r['unobserved_walls'])}", 9, False))
        lines.append(("", 8, False))
    lines.append(("Every number above is read from the canonical WorldIR; a dimension without a measured value is "
                  "reported as not measured.", 8, False))

    page_h, top, bottom = 842.0, 800.0, 50.0
    pages: List[List[Tuple[str, int, bool]]] = []
    current: List[Tuple[str, int, bool]] = []
    y = top
    for text, size, bold in lines:
        advance = size * 1.7
        if y - advance < bottom:
            pages.append(current)
            current, y = [], top
        current.append((text, size, bold))
        y -= advance
    if current:
        pages.append(current)

    def wrap(text: str, size: int) -> List[str]:
        max_chars = max(20, int((515.0) / (size * 0.52)))
        words, lines_out, cur = text.split(), [], ""
        for w in words:
            candidate = (cur + " " + w).strip()
            if len(candidate) > max_chars and cur:
                lines_out.append(cur)
                cur = w
            else:
                cur = candidate
        if cur or not lines_out:
            lines_out.append(cur)
        return lines_out

    objects: List[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    page_obj_numbers: List[int] = []
    content_numbers: List[int] = []
    for page in pages:
        ops: List[str] = []
        yy = top
        for text, size, bold in page:
            for chunk in wrap(text, size):
                if yy < bottom:
                    break
                ops.append(f"BT /{'F2' if bold else 'F1'} {size} Tf 40 {yy:.1f} Td ({_pdf_escape(chunk)}) Tj ET")
                yy -= size * 1.55
            yy -= size * 0.25
        stream = "\n".join(ops).encode("latin-1", "replace")
        content_numbers.append(add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream +
                                   b"\nendstream"))
    # pages object needs the page object numbers; reserve it after pages are known
    pages_obj = len(objects) + 1
    for content in content_numbers:
        page_obj_numbers.append(add(
            f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 595 842] /Contents {content} 0 R "
            f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> >>".encode()))
    objects.append(f"<< /Type /Pages /Kids [{' '.join(f'{n} 0 R' for n in page_obj_numbers)}] "
                   f"/Count {len(page_obj_numbers)} >>".encode())
    catalog = add(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def export_to_room_report_pdf_with_report(world, artifact_store=None) -> Tuple[bytes, ExportReport]:
    report = build_room_report(world)
    pdf = report_to_pdf(report)
    rooms = report.get("rooms", [])
    rep = ExportReport(
        format="room_report_pdf", world_id=world.id, world_version=world.version,
        entities_exported=tuple(r["id"] for r in rooms),
        entities_skipped=(), skip_reasons=(), content_hash=content_hash(pdf),
    )
    return pdf, rep
