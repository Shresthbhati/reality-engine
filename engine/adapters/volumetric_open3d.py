"""Open3D ScalableTSDFVolume and Marching Cubes Meshing Adapter.

Uses Open3D's native ScalableTSDFVolume C++ engine to integrate multi-view
depth maps and camera poses into a truncated signed distance field, extracting
watertight isosurface triangle meshes via Marching Cubes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.adapters.base import (
    BaseEngineAdapter,
    InputValidationError,
)


class Open3DTSDFAdapter(BaseEngineAdapter):
    """Adapter for Open3D ScalableTSDFVolume integration and meshing."""

    def __init__(
        self,
        voxel_length_m: float = 0.03,
        sdf_trunc_m: float = 0.09,
        device: str | None = None,
    ) -> None:
        super().__init__(
            engine_name="open3d_tsdf_mesher",
            task_category="volumetric_surface_reconstruction",
            licence="MIT",
            device=device,
        )
        self.voxel_length_m = voxel_length_m
        self.sdf_trunc_m = sdf_trunc_m
        self._o3d = None

    def verify_dependencies(self) -> tuple[bool, str]:
        try:
            import open3d as o3d
            self._o3d = o3d
            return True, f"open3d {o3d.__version__} available"
        except ImportError as e:
            return False, f"open3d is not installed: {e}"

    def load_model(self) -> None:
        if self._o3d is None:
            import open3d as o3d
            self._o3d = o3d

    def validate_inputs(self, inputs: Any) -> None:
        if not isinstance(inputs, dict):
            raise InputValidationError("Inputs must be a dict containing 'frames' and 'intrinsics'.")
        if "frames" not in inputs or len(inputs["frames"]) == 0:
            raise InputValidationError("Inputs must contain a non-empty 'frames' list.")
        if "intrinsics" not in inputs:
            raise InputValidationError("Inputs must specify camera 'intrinsics'.")

    def _execute_inference(
        self,
        inputs: dict[str, Any],
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        import open3d as o3d

        voxel_len = float(options.get("voxel_length_m", self.voxel_length_m))
        sdf_trunc = float(options.get("sdf_trunc_m", self.sdf_trunc_m))

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_len,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        intr_dict = inputs["intrinsics"]
        width = int(intr_dict["width"])
        height = int(intr_dict["height"])
        fx = float(intr_dict["fx"])
        fy = float(intr_dict.get("fy", fx))
        cx = float(intr_dict["cx"])
        cy = float(intr_dict["cy"])

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        frames = inputs["frames"]
        integrated_count = 0

        for frame in frames:
            depth_arr = np.asarray(frame["depth"], dtype=np.float32)
            color_arr = frame.get("color")
            if color_arr is not None:
                color_arr = np.asarray(color_arr, dtype=np.uint8)
            else:
                color_arr = np.ones((height, width, 3), dtype=np.uint8) * 200

            depth_o3d = o3d.geometry.Image(depth_arr)
            color_o3d = o3d.geometry.Image(color_arr)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=float(frame.get("depth_scale", 1000.0)),
                depth_trunc=float(frame.get("max_depth", 5.0)),
                convert_rgb_to_intensity=False,
            )

            extrinsic = np.asarray(frame["extrinsic_4x4"], dtype=np.float64)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))
            integrated_count += 1

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        is_watertight = mesh.is_watertight() if len(vertices) > 0 else False
        has_vertices = len(vertices) > 0

        extent = [0.0, 0.0, 0.0]
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
        if has_vertices:
            bounds_min = np.min(vertices, axis=0).tolist()
            bounds_max = np.max(vertices, axis=0).tolist()
            extent = (np.max(vertices, axis=0) - np.min(vertices, axis=0)).tolist()

        structured_results = {
            "num_vertices": len(vertices),
            "num_faces": len(triangles),
            "is_watertight": is_watertight,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "extent_m": extent,
            "integrated_frames": integrated_count,
            "voxel_length_m": voxel_len,
            "sdf_trunc_m": sdf_trunc,
        }

        diagnostics = {
            "has_normals": mesh.has_vertex_normals(),
            "has_vertex_colors": mesh.has_vertex_colors(),
            "surface_area": float(mesh.get_surface_area()) if has_vertices else 0.0,
        }

        return structured_results, mesh, diagnostics
