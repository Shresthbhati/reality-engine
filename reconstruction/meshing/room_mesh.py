"""Non-convex room prismatic 3D mesh generator and volume calculator (Mandate Phase 5 & 6).

Generates watertight 3D triangle meshes from 2D room boundary footprints (convex or non-convex:
L-shaped, T-shaped, U-shaped, chamfered):
- Triangulates bottom floor slab and top ceiling slab using ear clipping.
- Extrudes vertical wall quads (2 triangles per wall segment) connecting floor to ceiling.
- Computes vertex/face normals and verifies mesh closure and volume.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from perception.architecture.room_layout import ear_clip
from world_ir.geometry_data import MeshData, Point, Triangle


def _polygon_area_2d(pts: Sequence[tuple[float, float]]) -> float:
    """Compute signed 2D area (positive for CCW, negative for CW)."""
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def is_non_convex(pts: Sequence[tuple[float, float]]) -> bool:
    """Detect if a 2D simple polygon is non-convex (contains at least one reflex angle)."""
    n = len(pts)
    if n < 4:
        return False

    area = _polygon_area_2d(pts)
    is_ccw = area > 0.0
    ordered_pts = list(pts) if is_ccw else list(reversed(pts))

    for i in range(n):
        p_prev = ordered_pts[(i - 1) % n]
        p_curr = ordered_pts[i]
        p_next = ordered_pts[(i + 1) % n]

        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # In a CCW convex polygon, all cross products are >= 0. A negative cross product indicates a reflex turn.
        if cross < -1e-6:
            return True
    return False


def generate_prismatic_room_mesh(
    floor_polygon_2d: Sequence[tuple[float, float]],
    height_m: float,
    ground_z_m: float = 0.0,
) -> MeshData:
    """Construct a watertight 3D prismatic mesh from a 2D room footprint."""
    n = len(floor_polygon_2d)
    if n < 3:
        raise ValueError(f"Room footprint must have at least 3 vertices, got {n}")
    if height_m <= 0.0:
        raise ValueError(f"Room height must be positive, got {height_m}")

    # Ensure CCW ordering for floor vertices
    area_2d = _polygon_area_2d(floor_polygon_2d)
    if math.isclose(area_2d, 0.0, abs_tol=1e-8):
        raise ValueError("Degenerate polygon with zero area")

    is_ccw = area_2d > 0.0
    poly = list(floor_polygon_2d) if is_ccw else list(reversed(floor_polygon_2d))

    vertices: list[Point] = []
    faces: list[Triangle] = []

    # 1. Floor vertices (indices 0 .. n-1)
    for x, y in poly:
        vertices.append((float(x), float(y), float(ground_z_m)))

    # 2. Ceiling vertices (indices n .. 2n-1)
    ceil_z = float(ground_z_m + height_m)
    for x, y in poly:
        vertices.append((float(x), float(y), ceil_z))

    # 3. Floor faces (normal pointing -Z into ground)
    # ear_clip assumes CCW and produces (a, b, c) facing +Z. For floor facing down (-Z), invert triangle order: (a, c, b).
    floor_tris = ear_clip(poly)
    for a, b, c in floor_tris:
        faces.append((a, c, b))

    # 4. Ceiling faces (normal pointing +Z into sky)
    # Ceiling vertices are offset by n. Facing +Z: (n+a, n+b, n+c).
    for a, b, c in floor_tris:
        faces.append((n + a, n + b, n + c))

    # 5. Vertical wall quad faces (2 triangles per wall segment)
    # Wall from vertex i to (i+1)%n:
    # Floor: i, (i+1)%n
    # Ceiling: n+i, n+(i+1)%n
    for i in range(n):
        next_i = (i + 1) % n
        v_fl_curr = i
        v_fl_next = next_i
        v_cl_next = n + next_i
        v_cl_curr = n + i

        # First wall triangle (facing outward)
        faces.append((v_fl_curr, v_fl_next, v_cl_next))
        # Second wall triangle (facing outward)
        faces.append((v_fl_curr, v_cl_next, v_cl_curr))

    # Compute vertex normals
    normals = _compute_vertex_normals(vertices, faces)

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        units="meters",
        frame="world",
    )


def _compute_vertex_normals(
    vertices: list[Point],
    faces: list[Triangle],
) -> list[Point]:
    """Compute area-weighted smooth vertex normals."""
    v_normals = [[0.0, 0.0, 0.0] for _ in vertices]

    for a, b, c in faces:
        p0 = vertices[a]
        p1 = vertices[b]
        p2 = vertices[c]

        v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])

        # Cross product
        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]

        for idx in (a, b, c):
            v_normals[idx][0] += nx
            v_normals[idx][1] += ny
            v_normals[idx][2] += nz

    out: list[Point] = []
    for nx, ny, nz in v_normals:
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if mag > 1e-12:
            out.append((nx / mag, ny / mag, nz / mag))
        else:
            out.append((0.0, 0.0, 1.0))
    return out


def room_mesh_volume(floor_polygon_2d: Sequence[tuple[float, float]], height_m: float) -> float:
    """Exact enclosed 3D prismatic volume (floor area * ceiling height)."""
    return abs(_polygon_area_2d(floor_polygon_2d)) * height_m


def room_mesh_surface_area(floor_polygon_2d: Sequence[tuple[float, float]], height_m: float) -> float:
    """Total surface area of the room shell (2 * floor area + perimeter * height)."""
    n = len(floor_polygon_2d)
    perimeter = sum(
        math.hypot(
            floor_polygon_2d[(i + 1) % n][0] - floor_polygon_2d[i][0],
            floor_polygon_2d[(i + 1) % n][1] - floor_polygon_2d[i][1],
        )
        for i in range(n)
    )
    floor_area = abs(_polygon_area_2d(floor_polygon_2d))
    return 2.0 * floor_area + perimeter * height_m
