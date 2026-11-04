"""High-Resolution Crop Pass for Fine Detail Detection (Task 9.1).

Multi-scale processing (Blueprint 15 P11):
1. Whole image at reduced size: architecture & large furniture.
2. High-res crops only where needed: small objects (sockets, switches, fixtures)
   inside wall / furniture regions of interest.
3. Resolution-aware min_pixels gating (Blueprint 15 P10/P11 & Blueprint 17):
   Detector output carries pixel_size; detections strictly below the class's min_pixels
   are rejected and NEVER claimed as real observations.
Rule Zero: Outputs Observations only; never creates entities or surfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np

from perception.detection.open_vocab import OpenVocabDetector
from perception.ontology import Ontology, get_default_ontology
from world_ir.observations import DepthSource, Observation


@dataclass(frozen=True)
class ImageCrop:
    """A high-resolution crop extracted from a full-resolution image."""
    crop_id: str
    crop_image: np.ndarray
    origin_x: int  # X offset in full image pixels
    origin_y: int  # Y offset in full image pixels
    width: int
    height: int
    source_roi_id: Optional[str] = None


@dataclass(frozen=True)
class RejectedDetailDetection:
    """A fine detail detection rejected by resolution-aware min_pixels gating."""
    class_path: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in full image
    pixel_size: float
    min_pixels: int
    reason: str


@dataclass(frozen=True)
class CropPassResult:
    """Output of the high-res crop pass."""
    accepted_observations: List[Observation]
    rejected_detections: List[RejectedDetailDetection]
    crops_processed: int
    target_classes: Tuple[str, ...]
    summary: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "accepted_count": len(self.accepted_observations),
            "rejected_count": len(self.rejected_detections),
            "crops_processed": self.crops_processed,
            "target_classes": list(self.target_classes),
            "summary": self.summary,
        }


class HighResCropPass:
    """Executes high-resolution crop inspection for small architectural fixtures."""

    DEFAULT_TARGET_CLASSES = ("fixtures/electrical/socket", "fixtures/electrical/switch")

    def __init__(
        self,
        detector: Optional[OpenVocabDetector] = None,
        ontology: Optional[Ontology] = None,
        target_classes: Optional[Sequence[str]] = None,
        crop_padding_ratio: float = 0.15,
        min_confidence: float = 0.25,
    ):
        self.ontology = ontology or get_default_ontology()
        self.detector = detector or OpenVocabDetector(ontology=self.ontology, confidence_threshold=min_confidence)
        self.target_classes = tuple(target_classes) if target_classes is not None else self.DEFAULT_TARGET_CLASSES
        self.crop_padding_ratio = crop_padding_ratio
        self.min_confidence = min_confidence

    def extract_crops(
        self,
        image: np.ndarray,
        regions: Sequence[Tuple[float, float, float, float]],
        region_ids: Optional[Sequence[str]] = None,
    ) -> List[ImageCrop]:
        """Extract high-resolution sub-image crops from pixel bounding boxes [x_min, y_min, x_max, y_max]."""
        H, W = image.shape[:2]
        crops: List[ImageCrop] = []

        for i, box in enumerate(regions):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            if w <= 0 or h <= 0:
                continue

            # Add context margin
            pad_x = w * self.crop_padding_ratio
            pad_y = h * self.crop_padding_ratio
            cx0 = max(0, int(math.floor(x_min - pad_x)))
            cy0 = max(0, int(math.floor(y_min - pad_y)))
            cx1 = min(W, int(math.ceil(x_max + pad_x)))
            cy1 = min(H, int(math.ceil(y_max + pad_y)))

            if cx1 <= cx0 or cy1 <= cy0:
                continue

            crop_sub = image[cy0:cy1, cx0:cx1]
            rid = region_ids[i] if region_ids and i < len(region_ids) else f"roi_{i}"
            crops.append(ImageCrop(
                crop_id=f"crop_{uuid4().hex[:8]}",
                crop_image=crop_sub,
                origin_x=cx0,
                origin_y=cy0,
                width=cx1 - cx0,
                height=cy1 - cy0,
                source_roi_id=rid,
            ))

        return crops

    def run_pass(
        self,
        full_image: np.ndarray,
        candidate_regions: Sequence[Tuple[float, float, float, float]],
        frame_id: str = "",
        depth_map: Optional[np.ndarray] = None,
        is_metric_depth: bool = False,
        camera_intrinsics: Optional[dict] = None,
    ) -> CropPassResult:
        """Run the high-res crop pass over candidate regions in an image.

        1. Extracts sub-image crops for each candidate ROI.
        2. Detects target fine-detail classes.
        3. Maps bounding boxes back to full-resolution image space.
        4. Calculates pixel_size in full-resolution pixels.
        5. Enforces min_pixels threshold from the ontology:
           - pixel_size < min_pixels -> REJECTED (never claimed)
           - pixel_size >= min_pixels -> ACCEPTED observation
        """
        crops = self.extract_crops(full_image, candidate_regions)
        accepted: List[Observation] = []
        rejected: List[RejectedDetailDetection] = []
        per_class_accepted: Dict[str, int] = {cls: 0 for cls in self.target_classes}

        for crop in crops:
            # Run open-vocabulary detector on the crop
            detections = self.detector.detect_in_crop(
                crop_image=crop.crop_image,
                target_classes=self.target_classes,
                confidence_threshold=self.min_confidence,
            )

            for det in detections:
                # Map box from crop coordinates to full image coordinates
                local_box = det["box"]  # (x0, y0, x1, y1) in crop
                global_box = (
                    crop.origin_x + local_box[0],
                    crop.origin_y + local_box[1],
                    crop.origin_x + local_box[2],
                    crop.origin_y + local_box[3],
                )

                box_w = max(0.0, global_box[2] - global_box[0])
                box_h = max(0.0, global_box[3] - global_box[1])
                pixel_size = box_w * box_h

                canonical_class = det["class_path"]
                node = self.ontology.get(canonical_class)
                min_pixels = node.min_pixels if node is not None and node.min_pixels else 100

                # Resolution-aware min_pixels gating (Blueprint 15 P10/P11)
                if pixel_size < min_pixels:
                    rejected.append(RejectedDetailDetection(
                        class_path=canonical_class,
                        confidence=float(det["confidence"]),
                        box=global_box,
                        pixel_size=float(pixel_size),
                        min_pixels=min_pixels,
                        reason=f"pixel_size {pixel_size:.1f} < min_pixels {min_pixels}",
                    ))
                    continue

                # Accept observation
                per_class_accepted[canonical_class] = per_class_accepted.get(canonical_class, 0) + 1

                obs_meta = {
                    "detected_class": canonical_class,
                    "target_classes": list(self.target_classes),
                    "confidence": float(det["confidence"]),
                    "pixel_size": float(pixel_size),
                    "min_pixels": min_pixels,
                    "box_full": list(global_box),
                    "crop_origin": [crop.origin_x, crop.origin_y],
                    "crop_dims": [crop.width, crop.height],
                    "crop_id": crop.crop_id,
                    "source_roi_id": crop.source_roi_id,
                    "detection_stage": "high_res_crop_pass",
                }
                if camera_intrinsics:
                    obs_meta["camera_intrinsics"] = camera_intrinsics

                obs = Observation(
                    observation_id=f"obs-crop-{uuid4().hex[:10]}",
                    image_id=frame_id or "camera_main",
                    label=canonical_class,
                    world_pos=(0.0, 0.0, 0.0),
                    sigma_m=0.1 if is_metric_depth else 1.0,
                    depth_source=DepthSource.METRIC if is_metric_depth else DepthSource.RELATIVE,
                    model=getattr(self.detector, "model_label", "high_res_crop@1.0"),
                    confidence=float(det["confidence"]),
                    bbox_px=(float(global_box[0]), float(global_box[1]), float(global_box[2]), float(global_box[3])),
                )
                accepted.append(obs)

        return CropPassResult(
            accepted_observations=accepted,
            rejected_detections=rejected,
            crops_processed=len(crops),
            target_classes=self.target_classes,
            summary={
                "accepted_total": len(accepted),
                "rejected_total": len(rejected),
                **per_class_accepted,
            },
        )
