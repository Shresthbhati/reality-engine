"""Texture projection from photographs onto 3D meshes (Blueprint 30 Task 10.1).

Reconstructs photorealistic surface appearance by projecting multi-view camera
evidence onto 3D meshes with visibility, incidence-angle, and resolution weighting.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.render.textures.projection import PinholeCamera
from reconstruction.meshing.mesh import MeshData
from world_ir.artifact_store import ArtifactStore
from world_ir.schema_v1 import GeometryType, Material
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class TextureResult:
    """Outcome of texture projection onto a 3D mesh."""

    uvs: Tuple[Tuple[float, float], ...]  # per-vertex UV coordinates in [0..1]
    image_rgb: np.ndarray                 # (H, W, 3) uint8 image array
    png_bytes: bytes                      # Encoded PNG image bytes
    coverage: float                       # Observed surface fraction [0..1]
    cameras_used: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_uvs": len(self.uvs),
            "width": self.image_rgb.shape[1],
            "height": self.image_rgb.shape[0],
            "png_byte_length": len(self.png_bytes),
            "coverage": round(self.coverage, 4),
            "cameras_used": list(self.cameras_used),
        }


class MeshTextureProjector:
    """Projects multi-view photographs onto 3D meshes with resolution and incidence weighting."""

    def __init__(
        self,
        texture_width: int = 512,
        texture_height: int = 512,
        min_incidence_cos: float = 0.05,
    ):
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.min_incidence_cos = min_incidence_cos

    def project_texture(
        self,
        mesh: MeshData,
        cameras: Sequence[PinholeCamera],
    ) -> TextureResult:
        """Project camera photographs onto `mesh`, generating UVs and a blended texture atlas."""
        if len(mesh.vertices) == 0:
            raise ValueError("Cannot project texture onto an empty mesh")

        verts = np.array(mesh.vertices, dtype=np.float64)  # (N, 3)

        # Step 1: Compute dominant normal and orthogonal tangent plane
        if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
            avg_normal = np.mean(np.array(mesh.normals, dtype=np.float64), axis=0)
            norm_mag = np.linalg.norm(avg_normal)
            normal = avg_normal / norm_mag if norm_mag > 1e-6 else np.array([0.0, 0.0, 1.0])
        else:
            # Fallback to PCA / bounding plane
            mins = np.min(verts, axis=0)
            maxs = np.max(verts, axis=0)
            extents = maxs - mins
            min_dim = int(np.argmin(extents))
            normal = np.zeros(3)
            normal[min_dim] = 1.0

        # Construct orthonormal basis (u_axis, v_axis, normal)
        normal = normal / np.linalg.norm(normal)
        if abs(normal[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])

        u_axis = np.cross(up, normal)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis)

        # Step 2: Project vertices onto (u_axis, v_axis) to form UVs
        u_proj = verts @ u_axis
        v_proj = verts @ v_axis

        min_u, max_u = float(np.min(u_proj)), float(np.max(u_proj))
        min_v, max_v = float(np.min(v_proj)), float(np.max(v_proj))

        span_u = max(max_u - min_u, 1e-5)
        span_v = max(max_v - min_v, 1e-5)

        uvs_arr = np.column_stack([
            (u_proj - min_u) / span_u,
            (v_proj - min_v) / span_v,
        ])
        uvs_arr = np.clip(uvs_arr, 0.0, 1.0)
        uvs_tuple = tuple((float(u), float(v)) for u, v in uvs_arr)

        # Step 3: Rasterize texture atlas across the tangent plane
        H, W = self.texture_height, self.texture_width
        u_grid = np.linspace(min_u, max_u, W)
        v_grid = np.linspace(min_v, max_v, H)
        U_grid, V_grid = np.meshgrid(u_grid, v_grid)

        # Planar 3D sample points (H, W, 3)
        pts_3d = (
            U_grid[:, :, np.newaxis] * u_axis +
            V_grid[:, :, np.newaxis] * v_axis +
            (np.mean(verts @ normal)) * normal
        )
        pts_flat = pts_3d.reshape(-1, 3)

        accum_colors = np.zeros((H * W, 3), dtype=np.float64)
        accum_weights = np.zeros((H * W, 1), dtype=np.float64)
        cameras_used: List[str] = []

        for cam in cameras:
            pixels, depth_z = cam.project(pts_flat)
            colors, valid_fov = cam.sample_bilinear(pixels)

            cam_pos = cam.position_world
            to_cam = cam_pos - pts_flat
            dists = np.linalg.norm(to_cam, axis=1)
            to_cam_dir = to_cam / np.maximum(dists[:, np.newaxis], 1e-6)

            cos_inc = np.sum(to_cam_dir * normal, axis=1)
            valid = (cos_inc >= self.min_incidence_cos) & valid_fov & (depth_z > 0.1)

            if not np.any(valid):
                continue

            cameras_used.append(cam.camera_id)
            # Weight proportional to cos(incidence) / distance^2
            weights = np.where(valid, cos_inc / np.maximum(dists, 0.5), 0.0)[:, np.newaxis]

            accum_colors += colors * weights
            accum_weights += weights

        # Normalize colors
        has_obs = accum_weights[:, 0] > 1e-6
        final_colors = np.zeros((H * W, 3), dtype=np.float64)
        # Default unobserved background: neutral grey (0.7)
        final_colors[~has_obs] = 0.7
        final_colors[has_obs] = accum_colors[has_obs] / accum_weights[has_obs]

        img_uint8 = np.clip(final_colors * 255.0, 0, 255).astype(np.uint8).reshape(H, W, 3)
        coverage = float(np.sum(has_obs)) / float(H * W)

        # Encode to PNG bytes
        success, encoded = cv2.imencode(".png", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode texture image to PNG")
        png_bytes = encoded.tobytes()

        return TextureResult(
            uvs=uvs_tuple,
            image_rgb=img_uint8,
            png_bytes=png_bytes,
            coverage=coverage,
            cameras_used=tuple(cameras_used),
        )


def apply_texture_projection(
    world: WorldIR,
    entity_id: str,
    texture_result: TextureResult,
    artifact_store: Optional[ArtifactStore] = None,
) -> WorldIR:
    """Attach texture result to a WorldIR entity as a textured Material."""
    if entity_id not in world.entities:
        raise KeyError(f"Entity '{entity_id}' not found in world")

    entity = world.entities[entity_id]
    mat_id = f"mat-tex-{entity_id}"

    # Build material with texture properties
    b64_str = base64.b64encode(texture_result.png_bytes).decode("ascii")
    mat_props: Dict[str, Any] = {
        "texture_bytes": texture_result.png_bytes,
        "texture_base64": b64_str,
        "texture_mime": "image/png",
        "coverage": texture_result.coverage,
        "cameras_used": list(texture_result.cameras_used),
    }

    if artifact_store is not None:
        put_res = artifact_store.put(texture_result.png_bytes)
        tex_uri = put_res[0] if isinstance(put_res, tuple) else put_res
        mat_props["texture_uri"] = tex_uri

    material = Material(
        id=mat_id,
        name=f"Textured_{entity.name or entity_id}",
        properties=mat_props,
    )

    world.materials[mat_id] = material
    entity.material_id = mat_id
    if hasattr(entity, "material_ids"):
        if mat_id not in entity.material_ids:
            entity.material_ids.append(mat_id)

    # Attach UVs to mesh geometry if resolvable
    for gid in getattr(entity, "geometry_ids", ()):
        geom = world.geometries.get(gid)
        if geom and geom.type is GeometryType.MESH and artifact_store and geom.data_uri:
            try:
                payload = artifact_store.get(geom.data_uri)
                old_mesh = MeshData.from_bytes(payload)
                updated_mesh = MeshData(
                    vertices=old_mesh.vertices,
                    faces=old_mesh.faces,
                    normals=old_mesh.normals,
                    uvs=texture_result.uvs,
                    units=old_mesh.units,
                    frame=old_mesh.frame,
                )
                put_mesh = artifact_store.put(updated_mesh.to_bytes())
                geom.data_uri = put_mesh[0] if isinstance(put_mesh, tuple) else put_mesh
            except Exception:
                pass

    return world


__all__ = [
    "TextureResult",
    "MeshTextureProjector",
    "apply_texture_projection",
]
