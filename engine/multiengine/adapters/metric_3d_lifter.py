"""Metric 3D Lifter Engine Adapter.

Implements the BaseModelEngineAdapter contract to lift 2D detections and segmentation
masks into metric 3D entities using calibrated camera poses and metric depth maps.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from engine.multiengine.adapters.base import BaseModelEngineAdapter
from engine.multiengine.contract import (
    Device,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
)
from engine.multiengine.metric_lifting import Metric3DLifter
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.visibility_reprojection import VisibilityReprojectionValidator


class Metric3DLifterAdapter(BaseModelEngineAdapter):
    """Lifts 2D bounding boxes and masks into metric 3D entities."""

    def __init__(self, spec: EngineSpec) -> None:
        super().__init__(spec)
        self._lifter = Metric3DLifter()

    def _load_model_weights(self, device: Device) -> None:
        # Algorithmic/geometric adapter: no heavy neural weights to download
        self._model = self._lifter

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        if self._model is None:
            raise EngineError(
                EngineErrorKind.UNAVAILABLE,
                f"Adapter {self.spec.engine_id} not loaded. Call load() before infer().",
            )

        drafts: list[ObservationDraft] = []
        options = request.payload or {}  # EngineRequest carries ``payload``; it has never had an ``options`` attribute

        # 1. Camera parameters
        camera_meta = options.get("camera")
        if not camera_meta:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                "Metric3DLifter requires 'camera' metadata with intrinsics and extrinsics.",
            )

        camera = VisibilityReprojectionValidator.camera_from_metadata(camera_meta)
        if camera is None:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                "Failed to construct valid PinholeCamera from provided 'camera' metadata.",
            )

        # 2. Scale metadata validation (must fail closed)
        if "is_metric_depth" not in options and "scale_state" not in options:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                "Missing scale metadata: 'is_metric_depth' or 'scale_state' must be explicitly specified (fail closed).",
            )

        is_metric = bool(options.get("is_metric_depth", False))
        scale_state = options.get("scale_state")
        scale_units = options.get("scale_units")

        if is_metric and scale_state not in ("METRIC_SCALE_ALIGNED", "metric_anchored", None):
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                f"Contradictory scale metadata: is_metric_depth=True cannot be used with scale_state={scale_state!r}.",
            )

        if scale_state in ("RELATIVE_UNSCALED", "relative_depth", "UNKNOWN", "unknown") or (not is_metric and scale_state is None):
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                "Metric3DLifter requires verified metric or scale-aligned depth; relative monocular depth without scale alignment cannot be unprojected.",
            )

        depth_map = options.get("depth_map")
        if depth_map is None:
            # Check if file path is provided in sources or options
            depth_path = options.get("depth_path")
            if depth_path and Path(depth_path).exists():
                try:
                    from PIL import Image
                    d_img = Image.open(depth_path)
                    depth_map = np.array(d_img, dtype=np.float32)
                    scale = float(options.get("depth_scale", 1.0))
                    depth_map = depth_map * scale
                except Exception as exc:
                    raise EngineError(
                        EngineErrorKind.INVALID_INPUT,
                        f"Failed to load depth map from {depth_path}: {exc}",
                    ) from exc
            else:
                raise EngineError(
                    EngineErrorKind.INVALID_INPUT,
                    "Metric3DLifter requires 'depth_map' array or valid 'depth_path'.",
                )

        from engine.multiengine.depth_scale_alignment import validate_depth_map
        ok, err = validate_depth_map(depth_map)
        if not ok:
            raise EngineError(EngineErrorKind.INVALID_INPUT, f"Invalid depth map: {err}")

        # 3. 2D Detections / Observations to lift
        detections = options.get("detections", [])
        if not detections:
            return drafts

        eff_scale_state = scale_state or ("METRIC_SCALE_ALIGNED" if is_metric else "RELATIVE_UNSCALED")
        eff_scale_units = scale_units or ("METRES" if is_metric else "RELATIVE_DISPARITY")

        lifted_entities = []
        for idx, det in enumerate(detections):
            ent_id = det.get("entity_id", f"entity_{idx:03d}")
            cls_name = det.get("label") or det.get("class_name", "object")
            box_2d = det.get("box_2d") or det.get("image_region")
            if not box_2d or len(box_2d) != 4:
                continue

            mask = det.get("mask_2d")
            conf = det.get("confidence") or det.get("score")

            lifted = self._lifter.lift_detection(
                entity_id=ent_id,
                class_name=cls_name,
                box_2d=box_2d,
                depth_map=depth_map,
                camera=camera,
                target_frame=options.get("target_frame", "world:canonical"),
                mask_2d=mask,
                is_metric_depth=is_metric,
                scale_state=eff_scale_state,
                scale_units=eff_scale_units,
                depth_source=options.get("depth_source", "calibrated_metric_depth"),
                calibration_source=options.get("calibration_source", "camera_metadata"),
                confidence=float(conf) if conf is not None else None,
                source_id=str(request.sources[0]) if request.sources else "camera0",
            )
            if lifted is not None:
                lifted_entities.append(lifted)

        # 4. Associate subparts to parents in 3D
        associated = self._lifter.associate_parts_and_parents(lifted_entities)

        # 5. Output observation drafts
        for ent in associated:
            drafts.extend(self._lifter.to_observation_drafts(ent))

        return drafts
