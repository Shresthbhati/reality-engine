"""Outdoor Large-Scale Geospatial Terrain Elevation & Surface Mesh Fusion.

Provides production-grade terrain surface generation, bare-earth digital terrain
modeling (DTM/DEM), slope and aspect estimation, watertight manifold skirt
extrusion, and seamless integration into WorldIR geographic spaces.

Coordinates:
  - Canonical ENU (East-North-Up): x = East (+m), y = Elevation (+m), z = South (-m North).
  - Aligned with OpenStreetMap context in `geo.osm_context`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from provenance import Provenance, Uncertainty
from reconstruction.meshing.mesh import MeshData
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR


@dataclass
class TerrainElevationGrid:
    """A 2D regular elevation grid (DEM/DTM) in metric or geographic coordinates."""

    elevation: np.ndarray  # 2D array [rows, cols] (meters)
    bounds: Tuple[float, float, float, float]  # (min_x, min_z, max_x, max_z) in local metric ENU
    resolution_m: float = 1.0  # Cell size in meters
    crs: str = "EPSG:4326"  # Coordinate Reference System
    nodata: float = -9999.0
    geo_anchor: Optional[Tuple[float, float]] = None  # (lat, lon) reference anchor

    def __post_init__(self) -> None:
        if self.elevation.ndim != 2:
            raise ValueError(f"Elevation grid must be 2D, got shape {self.elevation.shape}")
        if self.resolution_m <= 0:
            raise ValueError(f"Resolution must be positive, got {self.resolution_m}")


@dataclass
class TerrainMesh:
    """A triangulated, analysis-ready 3D terrain surface mesh."""

    vertices: np.ndarray  # (N, 3) [x, y, z] float64
    faces: np.ndarray  # (M, 3) uint32
    normals: np.ndarray  # (N, 3) float64 per-vertex normals
    slope_degrees: np.ndarray  # (N,) float64 slope angle from horizontal
    aspect_degrees: np.ndarray  # (N,) float64 compass direction of slope (0=North, 90=East, 180=South, 270=West)
    is_watertight: bool = False
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        default_factory=lambda: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    metadata: Dict[str, Union[float, int, str, bool]] = field(default_factory=dict)

    def to_mesh_data(self) -> MeshData:
        """Convert to canonical MeshData representation."""
        return MeshData(
            vertices=self.vertices.astype(np.float32),
            faces=self.faces.astype(np.uint32),
            normals=self.normals.astype(np.float32),
            units="meter",
            frame="world",
        )


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smoothed per-vertex normal vectors by accumulating face normals."""
    num_verts = len(vertices)
    normals = np.zeros((num_verts, 3), dtype=np.float64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    lens[lens < 1e-12] = 1.0
    face_normals /= lens

    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    vert_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    vert_lens[vert_lens < 1e-12] = 1.0
    normals /= vert_lens
    return normals


def compute_slope_and_aspect(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate slope (degrees from horizontal) and aspect (degrees from North) from Y-up normal vectors."""
    # Canonical Y-up: vertical UP vector is [0, 1, 0].
    # North is [0, 0, -1], East is [1, 0, 0], South is [0, 0, 1].
    ny = np.clip(normals[:, 1], -1.0, 1.0)
    # Slope is angle between surface normal and vertical UP
    slope_rad = np.arccos(ny)
    slope_deg = np.degrees(slope_rad)

    # In-plane horizontal projection: x = East, z = South (-z = North)
    # Azimuth: North (0 deg) = -z, East (90 deg) = +x, South (180 deg) = +z, West (270 deg) = -x
    nx = normals[:, 0]
    nz = normals[:, 2]
    # compass angle = arctan2(east, north) = arctan2(nx, -nz)
    aspect_rad = np.arctan2(nx, -nz)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0

    return slope_deg, aspect_deg


def generate_terrain_mesh_from_grid(
    grid: TerrainElevationGrid,
    add_skirt: bool = True,
    skirt_depth_m: float = 10.0,
) -> TerrainMesh:
    """Convert a regular 2D elevation grid into a continuous 3D terrain surface mesh.

    If add_skirt is True, boundary skirt walls and a bottom enclosure plate are added,
    making the volume 100% watertight and manifold for physical collision and volumetric queries.
    """
    rows, cols = grid.elevation.shape
    if rows < 2 or cols < 2:
        raise ValueError("Elevation grid must be at least 2x2 cells")

    min_x, min_z, max_x, max_z = grid.bounds
    x_coords = np.linspace(min_x, max_x, cols)
    z_coords = np.linspace(min_z, max_z, rows)

    # Replace nodata values with minimum valid elevation
    valid_mask = grid.elevation != grid.nodata
    if not np.any(valid_mask):
        raise ValueError("Elevation grid contains only nodata values")

    min_valid_elev = float(np.min(grid.elevation[valid_mask]))
    clean_elevation = np.where(valid_mask, grid.elevation, min_valid_elev)

    # Create top surface vertices (rows * cols, 3)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)
    top_vertices = np.stack([grid_x, clean_elevation, grid_z], axis=-1).reshape(-1, 3)

    # Create triangulation faces for regular quad grid
    faces_list = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = i0 + 1
            i2 = (r + 1) * cols + c
            i3 = i2 + 1

            # Two triangles per cell: (i0, i2, i1) and (i1, i2, i3) (CCW when viewed from above)
            faces_list.append([i0, i2, i1])
            faces_list.append([i1, i2, i3])

    top_faces = np.array(faces_list, dtype=np.uint32)

    if not add_skirt:
        normals = compute_vertex_normals(top_vertices, top_faces)
        slope, aspect = compute_slope_and_aspect(normals)
        bbox = (
            (float(np.min(top_vertices[:, 0])), float(np.min(top_vertices[:, 1])), float(np.min(top_vertices[:, 2]))),
            (float(np.max(top_vertices[:, 0])), float(np.max(top_vertices[:, 1])), float(np.max(top_vertices[:, 2]))),
        )
        return TerrainMesh(
            vertices=top_vertices,
            faces=top_faces,
            normals=normals,
            slope_degrees=slope,
            aspect_degrees=aspect,
            is_watertight=False,
            bounding_box=bbox,
            metadata={
                "rows": rows,
                "cols": cols,
                "min_elevation_m": float(np.min(clean_elevation)),
                "max_elevation_m": float(np.max(clean_elevation)),
                "mean_elevation_m": float(np.mean(clean_elevation)),
                "area_m2": float((max_x - min_x) * (max_z - min_z)),
            },
        )

    # Add watertight skirt and base
    base_elevation = float(np.min(clean_elevation)) - max(1.0, float(skirt_depth_m))
    perimeter_top_indices: List[int] = []

    # Top edge (r=0, c=0..cols-1)
    for c in range(cols):
        perimeter_top_indices.append(c)
    # Right edge (c=cols-1, r=1..rows-1)
    for r in range(1, rows):
        perimeter_top_indices.append(r * cols + (cols - 1))
    # Bottom edge (r=rows-1, c=cols-2..0)
    for c in range(cols - 2, -1, -1):
        perimeter_top_indices.append((rows - 1) * cols + c)
    # Left edge (c=0, r=rows-2..1)
    for r in range(rows - 2, 0, -1):
        perimeter_top_indices.append(r * cols)

    num_top_verts = len(top_vertices)
    num_perimeter = len(perimeter_top_indices)

    # Bottom skirt vertices: matching x, z with y = base_elevation
    skirt_bottom_vertices = np.zeros((num_perimeter, 3), dtype=np.float64)
    for idx, top_idx in enumerate(perimeter_top_indices):
        skirt_bottom_vertices[idx] = [
            top_vertices[top_idx, 0],
            base_elevation,
            top_vertices[top_idx, 2],
        ]

    # Center base vertex for fan triangulation
    center_base_vertex = np.array(
        [[float(np.mean(top_vertices[:, 0])), base_elevation, float(np.mean(top_vertices[:, 2]))]],
        dtype=np.float64,
    )

    all_vertices = np.vstack([top_vertices, skirt_bottom_vertices, center_base_vertex])
    center_base_idx = len(all_vertices) - 1

    skirt_faces = []
    base_faces = []

    for k in range(num_perimeter):
        k_next = (k + 1) % num_perimeter

        t0 = perimeter_top_indices[k]
        t1 = perimeter_top_indices[k_next]
        b0 = num_top_verts + k
        b1 = num_top_verts + k_next

        # Skirt quad side: (t0, b0, t1) and (t1, b0, b1)
        skirt_faces.append([t0, b0, t1])
        skirt_faces.append([t1, b0, b1])

        # Bottom base fan face (facing downwards, CCW viewed from below)
        base_faces.append([center_base_idx, b0, b1])

    all_faces = np.vstack(
        [
            top_faces,
            np.array(skirt_faces, dtype=np.uint32),
            np.array(base_faces, dtype=np.uint32),
        ]
    )

    normals = compute_vertex_normals(all_vertices, all_faces)
    slope, aspect = compute_slope_and_aspect(normals)
    bbox = (
        (float(np.min(all_vertices[:, 0])), float(np.min(all_vertices[:, 1])), float(np.min(all_vertices[:, 2]))),
        (float(np.max(all_vertices[:, 0])), float(np.max(all_vertices[:, 1])), float(np.max(all_vertices[:, 2]))),
    )

    return TerrainMesh(
        vertices=all_vertices,
        faces=all_faces,
        normals=normals,
        slope_degrees=slope,
        aspect_degrees=aspect,
        is_watertight=True,
        bounding_box=bbox,
        metadata={
            "rows": rows,
            "cols": cols,
            "min_elevation_m": float(np.min(clean_elevation)),
            "max_elevation_m": float(np.max(clean_elevation)),
            "mean_elevation_m": float(np.mean(clean_elevation)),
            "base_elevation_m": base_elevation,
            "area_m2": float((max_x - min_x) * (max_z - min_z)),
            "watertight": True,
        },
    )


def generate_terrain_mesh_from_point_cloud(
    points: np.ndarray,
    cell_size_m: float = 2.0,
    add_skirt: bool = True,
    skirt_depth_m: float = 5.0,
) -> TerrainMesh:
    """Create a digital terrain model (DTM) from an unorganized 3D point cloud (e.g. LiDAR or photogrammetry).

    Bins the points into a regular 2D metric grid, extracts minimum elevation per cell
    (bare-earth filtering), interpolates missing cells, and generates a manifold mesh.
    """
    if len(points) < 4:
        raise ValueError("At least 4 points required to reconstruct terrain surface")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_z, max_z = float(np.min(z)), float(np.max(z))

    cols = max(2, int(math.ceil((max_x - min_x) / cell_size_m)))
    rows = max(2, int(math.ceil((max_z - min_z) / cell_size_m)))

    grid_elev = np.full((rows, cols), -9999.0, dtype=np.float64)

    # Bin points into cells
    col_indices = np.clip(((x - min_x) / (max_x - min_x + 1e-9) * cols).astype(int), 0, cols - 1)
    row_indices = np.clip(((z - min_z) / (max_z - min_z + 1e-9) * rows).astype(int), 0, rows - 1)

    for i in range(len(points)):
        r = row_indices[i]
        c = col_indices[i]
        pt_y = y[i]
        if grid_elev[r, c] == -9999.0 or pt_y < grid_elev[r, c]:
            grid_elev[r, c] = pt_y

    # Interpolate unfilled cells using nearest valid neighbor
    unfilled = grid_elev == -9999.0
    if np.all(unfilled):
        raise ValueError("Could not populate any terrain cells from point cloud")

    if np.any(unfilled):
        valid_r, valid_c = np.nonzero(~unfilled)
        valid_y = grid_elev[valid_r, valid_c]

        from scipy.spatial import cKDTree

        tree = cKDTree(np.stack([valid_r, valid_c], axis=-1))
        empty_r, empty_c = np.nonzero(unfilled)
        _, nearest_idx = tree.query(np.stack([empty_r, empty_c], axis=-1))
        grid_elev[empty_r, empty_c] = valid_y[nearest_idx]

    grid = TerrainElevationGrid(
        elevation=grid_elev,
        bounds=(min_x, min_z, max_x, max_z),
        resolution_m=cell_size_m,
    )

    return generate_terrain_mesh_from_grid(grid, add_skirt=add_skirt, skirt_depth_m=skirt_depth_m)


def fuse_terrain_into_world(
    world: WorldIR,
    terrain_mesh: TerrainMesh,
    entity_id: str = "terrain-ground",
    name: str = "Site Ground Terrain",
    geo_anchor: Optional[Tuple[float, float]] = None,
) -> Entity:
    """Register a reconstructed terrain mesh surface as a first-class WorldIR entity.

    Links spatial geometry, slope/aspect properties, bounding volumes, and geo-anchor coordinates.
    """
    min_pt, max_pt = terrain_mesh.bounding_box
    center_y = (min_pt[1] + max_pt[1]) / 2.0

    geom_id = f"geom-{entity_id}"
    geom = Geometry(
        id=geom_id,
        type=GeometryType.MESH,
        bounds_min=Vector3(min_pt[0], min_pt[1], min_pt[2]),
        bounds_max=Vector3(max_pt[0], max_pt[1], max_pt[2]),
        vertex_count=len(terrain_mesh.vertices),
        triangle_count=len(terrain_mesh.faces),
        provenance=Provenance.OBSERVED,
        confidence=0.98,
    )
    world.geometries[geom_id] = geom

    entity = Entity(
        id=entity_id,
        name=name,
        type=EntityType.TERRAIN,
        geometry_ids=[geom_id],
        statement_state=StatementState.OBSERVED,
        confidence=0.98,
        uncertainty=Uncertainty(
            confidence=0.98,
            note="geospatial terrain elevation reconstruction",
        ),
        provenance=Provenance.OBSERVED,
        custom_properties={
            "is_terrain": True,
            "is_watertight": terrain_mesh.is_watertight,
            "min_elevation_m": terrain_mesh.metadata.get("min_elevation_m", min_pt[1]),
            "max_elevation_m": terrain_mesh.metadata.get("max_elevation_m", max_pt[1]),
            "mean_elevation_m": terrain_mesh.metadata.get("mean_elevation_m", center_y),
            "area_m2": terrain_mesh.metadata.get("area_m2", 0.0),
            "mean_slope_degrees": float(np.mean(terrain_mesh.slope_degrees)),
            "max_slope_degrees": float(np.max(terrain_mesh.slope_degrees)),
            "bounds_min": list(min_pt),
            "bounds_max": list(max_pt),
        },
    )

    world.entities[entity_id] = entity

    # Enrich world metadata
    if world.metadata is None:
        world.metadata = {}

    world.metadata["terrain"] = {
        "entity_id": entity_id,
        "is_watertight": terrain_mesh.is_watertight,
        "vertex_count": len(terrain_mesh.vertices),
        "face_count": len(terrain_mesh.faces),
        "bounding_box": [list(min_pt), list(max_pt)],
        **terrain_mesh.metadata,
    }

    if geo_anchor is not None:
        if "geo" not in world.metadata:
            world.metadata["geo"] = {}
        world.metadata["geo"]["anchor"] = {"lat": geo_anchor[0], "lon": geo_anchor[1]}

    return entity
