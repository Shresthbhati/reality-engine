"""In-Process PyColmap Structure-from-Motion Adapter.

Uses the C++ bindings of pycolmap (v4.2.1) directly in-process. Supports both
Global SfM (`pycolmap.global_mapping`, GLOMAP-derived) and Incremental SfM
(`pycolmap.incremental_mapping`), returning typed camera pose records and
sparse 3D point clouds with reprojection metrics.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from engine.adapters.base import (
    BaseEngineAdapter,
    InputValidationError,
)


class PycolmapSfMAdapter(BaseEngineAdapter):
    """Adapter for in-process pycolmap Global and Incremental SfM."""

    def __init__(self, device: str | None = None) -> None:
        super().__init__(
            engine_name="pycolmap_sfm",
            task_category="structure_from_motion",
            licence="BSD-3-Clause",
            device=device,
        )
        self._pycolmap = None

    def verify_dependencies(self) -> tuple[bool, str]:
        try:
            import pycolmap
            self._pycolmap = pycolmap
            return True, f"pycolmap {pycolmap.__version__} available"
        except ImportError as e:
            return False, f"pycolmap is not installed: {e}"

    def load_model(self) -> None:
        if self._pycolmap is None:
            import pycolmap
            self._pycolmap = pycolmap

    def validate_inputs(self, inputs: Any) -> None:
        if not isinstance(inputs, (list, tuple)):
            raise InputValidationError("Inputs must be a list of image paths or image records.")
        if len(inputs) < 2:
            raise InputValidationError(f"SfM requires at least 2 images, got {len(inputs)}.")
        for item in inputs:
            p = Path(item) if isinstance(item, (str, Path)) else getattr(item, "path", None)
            if p is not None and not p.exists():
                raise InputValidationError(f"Image path does not exist: {p}")

    def _execute_inference(
        self,
        inputs: Sequence[Union[str, Path]],
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        import pycolmap

        mapping_mode = options.get("mapping_mode", "global")
        image_paths = [Path(p) for p in inputs]

        # Use temporary directory for workspace database if not provided
        work_dir = Path(options.get("work_dir", tempfile.mkdtemp(prefix="pycolmap_sfm_")))
        database_path = work_dir / "database.db"
        image_dir = image_paths[0].parent

        # 1. Feature Extraction
        extract_options = pycolmap.SiftExtractionOptions()
        extract_options.use_gpu = options.get("use_gpu", True)
        extract_options.max_image_size = options.get("max_image_size", 2000)

        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            sift_options=extract_options,
        )

        # 2. Exhaustive or Sequential Feature Matching
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.use_gpu = extract_options.use_gpu
        pycolmap.match_exhaustive(database_path=database_path, sift_options=matching_options)

        # 3. Mapping: Attempt Global Mapping first, fall back to Incremental if requested or needed
        reconstructions = {}
        used_mode = mapping_mode
        if mapping_mode == "global" and hasattr(pycolmap, "global_mapping"):
            try:
                reconstructions = pycolmap.global_mapping(
                    database_path=database_path,
                    image_path=image_dir,
                )
            except Exception:
                used_mode = "incremental_fallback"

        if not reconstructions:
            inc_options = pycolmap.IncrementalMapperOptions()
            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=image_dir,
                options=inc_options,
            )

        if not reconstructions:
            return {
                "status": "failed",
                "registered_cameras": 0,
                "points_3d_count": 0,
                "camera_poses": [],
                "points_3d": [],
            }, None, {"mapping_mode": used_mode, "error": "Zero sub-models reconstructed"}

        # Select the largest reconstructed sub-model
        model_idx = max(reconstructions.keys(), key=lambda k: len(reconstructions[k].images))
        recon = reconstructions[model_idx]

        # Extract structured camera poses
        camera_poses = []
        for img_id, img in recon.images.items():
            R = img.rotmat()
            t = img.translation
            pos = -R.T @ t
            qvec = img.qvec  # (w, x, y, z)
            camera_poses.append({
                "image_id": img.name,
                "camera_id": str(img.camera_id),
                "position_m": pos.tolist(),
                "rotation_quaternion": qvec.tolist(),
                "num_points_2d": len(img.points2D),
            })

        # Extract sparse 3D points
        points_3d = []
        errors = []
        for p3d_id, pt in recon.points3D.items():
            points_3d.append({
                "point_id": p3d_id,
                "xyz": pt.xyz.tolist(),
                "rgb": pt.color.tolist(),
                "error": float(pt.error),
                "track_length": len(pt.track.elements),
            })
            errors.append(float(pt.error))

        mean_error = float(np.mean(errors)) if errors else 0.0

        structured_results = {
            "status": "success" if len(camera_poses) >= 2 else "partial",
            "mapping_mode": used_mode,
            "registered_cameras": len(camera_poses),
            "points_3d_count": len(points_3d),
            "mean_reprojection_error_px": mean_error,
            "camera_poses": camera_poses,
            "points_3d": points_3d,
        }

        diagnostics = {
            "num_submodels": len(reconstructions),
            "selected_submodel": model_idx,
            "total_images": len(image_paths),
            "registration_ratio": len(camera_poses) / max(1, len(image_paths)),
            "database_path": str(database_path),
        }

        return structured_results, recon, diagnostics
