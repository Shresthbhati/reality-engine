"""Still render service for saved camera views and client decks (T-4b.6).

Produces crisp, high-resolution perspective still renders from saved camera poses,
incorporating derived surface finish layers, material colors, and element geometry.
"""

from __future__ import annotations

import io
import math
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_ir.world_v1 import WorldIR


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / norm


def _coord(pt: Any, axis: str, default: float = 0.0) -> float:
    """Safely get numeric coordinate from Vector3, dict, or object."""
    if pt is None:
        return default
    if hasattr(pt, axis):
        val = getattr(pt, axis)
        if val is not None:
            return float(val)
    if isinstance(pt, dict) and axis in pt:
        val = pt[axis]
        if val is not None:
            return float(val)
    return default


def _get_entity_color(ent: Any, world: WorldIR) -> tuple[int, int, int]:
    """Resolve face color from entity surface finishes, materials, or architectural type."""
    # 1. Check material assignment
    if ent.material_ids:
        for mid in ent.material_ids:
            mat = world.materials.get(mid)
            if mat and mat.color_rgb:
                r, g, b = mat.color_rgb
                return int(r * 255), int(g * 255), int(b * 255)

    # 2. Check custom properties
    props = ent.custom_properties or {}
    if "color_rgb" in props:
        c = props["color_rgb"]
        if isinstance(c, (list, tuple)) and len(c) == 3:
            return int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)

    mat_label = str(props.get("material") or props.get("material_label") or "").lower()
    if "wood" in mat_label or "oak" in mat_label:
        return 190, 145, 105
    if "paint" in mat_label or "white" in mat_label or "linen" in mat_label:
        return 235, 235, 230
    if "concrete" in mat_label:
        return 140, 142, 145
    if "tile" in mat_label:
        return 210, 215, 220

    # 3. Default architectural colors by entity type
    etype = getattr(ent.type, "value", str(ent.type)).lower()
    if etype == "wall":
        return 225, 225, 220
    if etype in ("floor", "slab"):
        return 180, 155, 125
    if etype == "ceiling":
        return 245, 245, 248
    if etype in ("door", "window"):
        return 160, 200, 220
    if etype in ("furniture", "structure"):
        return 120, 140, 160
    return 180, 180, 185


def _get_entity_boxes(world: WorldIR) -> list[tuple[np.ndarray, tuple[int, int, int], str]]:
    """Extract bounding boxes and approximate volumes for entities."""
    boxes = []
    for eid, ent in world.entities.items():
        color = _get_entity_color(ent, world)
        name = ent.name or eid
        t = ent.transform or {}
        pos = t.get("position", {}) if isinstance(t, dict) else (getattr(t, "position", None) or {})
        cx = _coord(pos, "x", 0.0)
        cy = _coord(pos, "y", 0.0)
        cz = _coord(pos, "z", 0.0)

        # Check geometry bounds or custom properties
        sx, sy, sz = 1.0, 1.0, 1.0
        geom_found = False
        for gid in ent.geometry_ids:
            geom = world.geometries.get(gid)
            if geom and hasattr(geom, "bounds_min") and hasattr(geom, "bounds_max"):
                bmin = geom.bounds_min
                bmax = geom.bounds_max
                if bmin is not None and bmax is not None:
                    min_x = _coord(bmin, "x", cx)
                    max_x = _coord(bmax, "x", cx)
                    min_y = _coord(bmin, "y", cy)
                    max_y = _coord(bmax, "y", cy)
                    min_z = _coord(bmin, "z", cz)
                    max_z = _coord(bmax, "z", cz)

                    cx = (min_x + max_x) / 2.0
                    cy = (min_y + max_y) / 2.0
                    cz = (min_z + max_z) / 2.0
                    sx = max(0.1, abs(max_x - min_x))
                    sy = max(0.1, abs(max_y - min_y))
                    sz = max(0.1, abs(max_z - min_z))
                    geom_found = True
                    break

        if not geom_found:
            etype = getattr(ent.type, "value", str(ent.type)).lower()
            if etype == "wall":
                sx, sy, sz = 3.0, 2.5, 0.2
            elif etype in ("floor", "slab"):
                sx, sy, sz = 5.0, 0.2, 5.0
            elif etype == "door":
                sx, sy, sz = 0.9, 2.1, 0.1
            elif etype == "window":
                sx, sy, sz = 1.2, 1.5, 0.1
            else:
                sx, sy, sz = 1.5, 1.5, 1.5

        # 8 box corners
        dx, dy, dz = sx / 2.0, sy / 2.0, sz / 2.0
        corners = np.array([
            [cx - dx, cy - dy, cz - dz],
            [cx + dx, cy - dy, cz - dz],
            [cx + dx, cy + dy, cz - dz],
            [cx - dx, cy + dy, cz - dz],
            [cx - dx, cy - dy, cz + dz],
            [cx + dx, cy - dy, cz + dz],
            [cx + dx, cy + dy, cz + dz],
            [cx - dx, cy + dy, cz + dz],
        ])
        boxes.append((corners, color, name))
    return boxes


def render_still_png(
    world: WorldIR,
    view_name: str,
    position: dict[str, float],
    target: dict[str, float],
    fov_deg: float = 60.0,
    width: int = 1280,
    height: int = 720,
    world_name: str = "Architectural Model",
    version_label: str = "as-built",
) -> bytes:
    """Render a perspective still image from the given camera pose."""
    # Setup camera coordinates
    eye = np.array([float(position.get("x", 0.0)), float(position.get("y", 1.6)), float(position.get("z", -4.0))])
    look = np.array([float(target.get("x", 0.0)), float(target.get("y", 1.2)), float(target.get("z", 0.0))])

    fwd = _normalize(look - eye)
    up_world = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))

    fov_rad = math.radians(max(15.0, min(140.0, fov_deg)))
    focal = (height / 2.0) / math.tan(fov_rad / 2.0)

    # Initialize Image with gradient backdrop
    img = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(img)

    # Gradient background: gentle sky-to-ground transition
    for y in range(height):
        ratio = y / float(height)
        r = int(230 + ratio * 15)
        g = int(235 + ratio * 12)
        b = int(242 + ratio * 8)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Horizon line
    cam_height = eye[1]
    horizon_y = int(height / 2.0 + (cam_height * focal / 50.0))
    if 0 <= horizon_y < height:
        draw.line([(0, horizon_y), (width, horizon_y)], fill=(215, 220, 228), width=1)

    # Project and sort faces
    face_indices = [
        ([0, 1, 2, 3], np.array([0, 0, -1])),  # Back
        ([4, 5, 6, 7], np.array([0, 0, 1])),   # Front
        ([0, 1, 5, 4], np.array([0, -1, 0])),  # Bottom
        ([2, 3, 7, 6], np.array([0, 1, 0])),   # Top
        ([0, 3, 7, 4], np.array([-1, 0, 0])),  # Left
        ([1, 2, 6, 5], np.array([1, 0, 0])),   # Right
    ]

    light_dir = _normalize(np.array([0.6, 1.2, -0.4]))

    boxes = _get_entity_boxes(world)
    projected_faces = []

    for corners, base_color, ent_name in boxes:
        for idxs, normal in face_indices:
            pts_3d = corners[idxs]
            center_3d = np.mean(pts_3d, axis=0)
            dist_to_cam = np.linalg.norm(center_3d - eye)

            # Project each vertex to 2D
            pts_2d = []
            valid = True
            for pt in pts_3d:
                v = pt - eye
                z_cam = np.dot(v, fwd)
                if z_cam <= 0.05:
                    valid = False
                    break
                x_cam = np.dot(v, right)
                y_cam = np.dot(v, up)
                u = width / 2.0 + (x_cam / z_cam) * focal
                v_coord = height / 2.0 - (y_cam / z_cam) * focal
                pts_2d.append((u, v_coord))

            if valid and len(pts_2d) >= 3:
                # Shading based on diffuse lighting
                cos_theta = max(0.25, float(np.dot(normal, light_dir)))
                shaded = (
                    int(min(255, base_color[0] * cos_theta)),
                    int(min(255, base_color[1] * cos_theta)),
                    int(min(255, base_color[2] * cos_theta)),
                )
                projected_faces.append((dist_to_cam, pts_2d, shaded))

    # Painter's algorithm: sort faces from back to front
    projected_faces.sort(key=lambda f: f[0], reverse=True)

    for _, pts_2d, color in projected_faces:
        draw.polygon(pts_2d, fill=color, outline=(70, 75, 85))

    # ---------------------------------------------------------------------------------------------
    # Client Deck Presentation Overlays & Annotations
    # ---------------------------------------------------------------------------------------------
    # Header card
    draw.rectangle([(24, 20), (520, 95)], fill=(255, 255, 255), outline=(210, 215, 222), width=1)
    draw.text((36, 28), view_name.upper(), fill=(18, 22, 28))
    deck_sub = f"{world_name}  •  Version: {version_label}"
    draw.text((36, 50), deck_sub, fill=(90, 98, 110))
    elem_info = f"Entities: {len(world.entities)}  |  Materials: {len(world.materials)}"
    draw.text((36, 70), elem_info, fill=(120, 130, 145))

    # Camera telemetry badge (bottom-left)
    draw.rectangle([(24, height - 60), (380, height - 20)], fill=(255, 255, 255), outline=(210, 215, 222), width=1)
    cam_str = f"Eye: ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})m  |  FOV: {fov_deg:.0f}°"
    draw.text((36, height - 48), cam_str, fill=(45, 52, 64))

    # Watermark / stamp (bottom-right)
    draw.text((width - 240, height - 36), "Reality Engine Client Deck", fill=(140, 148, 160))

    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()
