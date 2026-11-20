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
        self.rejections: list[dict] = []  # per-detection rejections of the LAST infer call (entity_id, class_name, reason)

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
        self.rejections = []
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

        # 2. Depth map + an EXPLICIT scale declaration. Nothing defaults to metres: a missing declaration is refused, and the
        #    scale class (relative / model-metric / gauge / physically anchored) decides what the lift may claim.
        scale_state = options.get("scale_state")
        if not scale_state:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                "Metric3DLifter requires an explicit 'scale_state' for the depth map (e.g. PHYSICALLY_ANCHORED with 'scale_source', "
                "METRIC_TRAINED_MODEL, SFM_GAUGE); an undeclared scale is never assumed to be metric.",
            )
        is_metric_depth = bool(options.get("is_metric_depth", False))

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

        # 3. 2D Detections / Observations to lift
        detections = options.get("detections", [])
        if not detections:
            return drafts

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
                is_metric_depth=is_metric_depth,
                scale_state=str(scale_state),
                scale_units=str(options.get("scale_units", "UNSPECIFIED")),
                scale_source=str(options.get("scale_source", "")),
                depth_source=str(options.get("depth_source", "unspecified")),
                calibration_source=str(options.get("calibration_source", "unspecified")),
                confidence=float(conf) if conf is not None else None,
                source_id=str(request.sources[0]) if request.sources else "camera0",
            )
            if lifted is not None:
                lifted_entities.append(lifted)
            else:  # never discard silently: the reason stays on the adapter and, if nothing lifted, in the raised error  # noqa: E501
                self.rejections.append({"entity_id": ent_id, "class_name": cls_name, "reason": self._lifter.last_rejection_reason})

        if not lifted_entities and self.rejections:
            reasons = sorted({str(r["reason"]) for r in self.rejections})
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                f"all {len(self.rejections)} detection(s) were rejected by the lifter: {reasons}; the 2D evidence is unchanged",
            )

        # 4. Associate subparts to parents in 3D
        associated = self._lifter.associate_parts_and_parents(lifted_entities)

        # 5. Output observation drafts
        for ent in associated:
            drafts.extend(self._lifter.to_observation_drafts(ent))

        return drafts
