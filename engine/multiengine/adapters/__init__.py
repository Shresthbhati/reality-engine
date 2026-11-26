"""Concrete Multi-Engine Model Adapters.

Exports all model adapters across reconstruction, geometry, depth, perception,
segmentation, and tracking disciplines.
"""

from __future__ import annotations

from engine.multiengine.adapters.appearance_tracking import (
    Dinov2EngineAdapter,
    TrackingEngineAdapter,
    probe_dinov2,
    probe_tracking,
)
from engine.multiengine.adapters.base import BaseModelEngineAdapter
from engine.multiengine.adapters.depth_adapters import (
    DepthAnythingV2EngineAdapter,
    DepthAnythingV2MetricEngineAdapter,
    DepthAnythingV3EngineAdapter,
    MidasEngineAdapter,
    probe_depth_anything_v2,
    probe_depth_anything_v2_metric,
    probe_depth_anything_v3,
    probe_midas,
)
from engine.multiengine.adapters.geometry_dust3r import (
    Dust3rEngineAdapter,
    Mast3rEngineAdapter,
    probe_dust3r,
    probe_mast3r,
)
from engine.multiengine.adapters.geometry_vggt import (
    VggtEngineAdapter,
    VggtOmegaEngineAdapter,
    probe_vggt,
    probe_vggt_omega,
)
from engine.multiengine.adapters.metric_3d_lifter import Metric3DLifterAdapter
from engine.multiengine.adapters.perception_detectors import (
    Florence2EngineAdapter,
    GroundingDinoEngineAdapter,
    MaskRCNNEngineAdapter,
    OwlVitEngineAdapter,
    probe_florence2,
    probe_grounding_dino,
    probe_owlvit,
)
from engine.multiengine.adapters.perception_segmenters import (
    Sam2EngineAdapter,
    Sam3EngineAdapter,
    SamEngineAdapter,
    probe_sam,
    probe_sam2,
    probe_sam3,
)
from engine.multiengine.adapters.sfm_colmap import ColmapEngineAdapter
from engine.multiengine.adapters.sfm_mapanything import MapAnythingEngineAdapter, probe_mapanything
from engine.multiengine.adapters.volumetric_surface import Open3DTSDFEngineAdapter, probe_open3d
from engine.multiengine.contract import EngineAdapter
from engine.multiengine.registry import EngineRegistry


def create_adapters_for_registry(reg: EngineRegistry) -> dict[str, EngineAdapter]:
    """Instantiate concrete EngineAdapter objects for every registered engine specification."""
    adapters: dict[str, EngineAdapter] = {}

    mapping = {
        "colmap": ColmapEngineAdapter,
        "mapanything": MapAnythingEngineAdapter,
        "vggt": VggtEngineAdapter,
        "vggt_omega": VggtOmegaEngineAdapter,
        "dust3r": Dust3rEngineAdapter,
        "mast3r": Mast3rEngineAdapter,
        "depth_anything_v3": DepthAnythingV3EngineAdapter,
        "depth_anything": DepthAnythingV2EngineAdapter,
        "depth_anything_v2": DepthAnythingV2EngineAdapter,
        "midas": MidasEngineAdapter,
        "grounding_dino": GroundingDinoEngineAdapter,
        "florence2": Florence2EngineAdapter,
        "owlvit": OwlVitEngineAdapter,
        "mask_rcnn": MaskRCNNEngineAdapter,
        "sam": SamEngineAdapter,
        "sam2": Sam2EngineAdapter,
        "sam3": Sam3EngineAdapter,
        "dinov2": Dinov2EngineAdapter,
        "tracking": TrackingEngineAdapter,
        "open3d_tsdf": Open3DTSDFEngineAdapter,
        "metric_3d_lifter": Metric3DLifterAdapter,
    }

    for spec in reg.all():
        cls = mapping.get(spec.engine_id)
        if cls is not None:
            adapters[spec.engine_id] = cls(spec)

    return adapters


__all__ = [
    "BaseModelEngineAdapter",
    "ColmapEngineAdapter",
    "MapAnythingEngineAdapter",
    "VggtEngineAdapter",
    "VggtOmegaEngineAdapter",
    "Dust3rEngineAdapter",
    "Mast3rEngineAdapter",
    "DepthAnythingV3EngineAdapter",
    "DepthAnythingV2EngineAdapter",
    "DepthAnythingV2MetricEngineAdapter",
    "MidasEngineAdapter",
    "GroundingDinoEngineAdapter",
    "Florence2EngineAdapter",
    "MaskRCNNEngineAdapter",
    "OwlVitEngineAdapter",
    "SamEngineAdapter",
    "Sam2EngineAdapter",
    "Sam3EngineAdapter",
    "Dinov2EngineAdapter",
    "TrackingEngineAdapter",
    "Open3DTSDFEngineAdapter",
    "Metric3DLifterAdapter",
    "create_adapters_for_registry",
    "probe_mapanything",
    "probe_vggt",
    "probe_vggt_omega",
    "probe_dust3r",
    "probe_mast3r",
    "probe_depth_anything_v3",
    "probe_depth_anything_v2",
    "probe_depth_anything_v2_metric",
    "probe_midas",
    "probe_grounding_dino",
    "probe_florence2",
    "probe_owlvit",
    "probe_sam",
    "probe_sam2",
    "probe_sam3",
    "probe_dinov2",
    "probe_tracking",
    "probe_open3d",
]
