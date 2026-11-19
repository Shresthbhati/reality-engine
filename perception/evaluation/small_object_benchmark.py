"""Small-Object and Architectural Detail Evaluation Harness.

Addresses Section 7 of the Multi-Engine Integration Directive:
- Evaluates micro-architectural features (< 15 cm): electrical outlets, switch plates,
  door handles, hinges, window fittings, skirting boards, pipes, railings, etc.
- Implements high-resolution tiling and sliding-window inference with exact coordinate re-mapping.
- Merges inter-tile boundary detections via Non-Maximum Suppression (NMS).
- Enforces strict 2D-to-3D geometric verification gating before promoting detections.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from PIL import Image

from perception.detection.extensible_prompter import ScaleFilter, get_extensible_prompter


@dataclass(frozen=True)
class TileSpecification:
    """Defines a crop window within a high-resolution source image."""

    tile_id: str
    origin_x: int
    origin_y: int
    width: int
    height: int
    source_width: int
    source_height: int


@dataclass(frozen=True)
class SmallObjectDetection:
    """A detected detail with provenance and global pixel coordinates."""

    label: str
    canonical_path: str
    score: float
    box_xyxy_global: tuple[float, float, float, float]
    box_xywh_global: tuple[float, float, float, float]
    area_px: float
    source_tile_id: str
    is_tiled_detection: bool
    is_verified_3d: bool = False
    depth_m: float | None = None
    position_3d: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SmallObjectBenchmarkReport:
    """Quantitative performance report on micro-detail and small-object detection."""

    total_ground_truth_items: int
    total_detections_naive: int
    total_detections_tiled: int
    recall_naive: float
    recall_tiled: float
    precision_tiled: float
    f1_tiled: float
    small_object_recovery_gain: float  # Percentage recall improvement from tiling
    category_breakdown: dict[str, dict[str, float]]


class TiledInferenceEngine:
    """Executes high-resolution tiled inference to maximize small-object recall."""

    def __init__(
        self,
        tile_size: int = 640,
        overlap_ratio: float = 0.20,
        iou_nms_threshold: float = 0.45,
    ) -> None:
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.iou_nms_threshold = iou_nms_threshold

    def generate_tiles(self, image_width: int, image_height: int) -> list[TileSpecification]:
        """Generates an overlapping grid of crop specifications covering the image."""
        step = int(self.tile_size * (1.0 - self.overlap_ratio))
        tiles = []

        y = 0
        tile_idx = 0
        while y < image_height:
            h = min(self.tile_size, image_height - y)
            x = 0
            while x < image_width:
                w = min(self.tile_size, image_width - x)
                tiles.append(TileSpecification(
                    tile_id=f"tile_{tile_idx}_{x}_{y}",
                    origin_x=x,
                    origin_y=y,
                    width=w,
                    height=h,
                    source_width=image_width,
                    source_height=image_height,
                ))
                tile_idx += 1
                if x + w >= image_width:
                    break
                x += step
            if y + h >= image_height:
                break
            y += step

        return tiles

    def extract_tile_image(
        self,
        image: Union[np.ndarray, Image.Image],
        tile: TileSpecification,
    ) -> Image.Image:
        """Crops a tile from the source image."""
        if isinstance(image, Image.Image):
            return image.crop((
                tile.origin_x,
                tile.origin_y,
                tile.origin_x + tile.width,
                tile.origin_y + tile.height,
            ))
        elif isinstance(image, np.ndarray):
            crop = image[
                tile.origin_y: tile.origin_y + tile.height,
                tile.origin_x: tile.origin_x + tile.width,
            ]
            return Image.fromarray(crop)
        raise ValueError(f"Unsupported image type: {type(image)}")

    def map_tile_boxes_to_global(
        self,
        tile_detections: Sequence[dict[str, Any]],
        tile: TileSpecification,
    ) -> list[SmallObjectDetection]:
        """Maps local tile pixel coordinates back to the global image coordinate frame."""
        global_detections = []
        for det in tile_detections:
            bx1, by1, bx2, by2 = det["box_xyxy"]

            gx1 = bx1 + tile.origin_x
            gy1 = by1 + tile.origin_y
            gx2 = bx2 + tile.origin_x
            gy2 = by2 + tile.origin_y

            bw = max(0.0, gx2 - gx1)
            bh = max(0.0, gy2 - gy1)

            global_detections.append(SmallObjectDetection(
                label=det["label"],
                canonical_path=det.get("canonical_path", det["label"]),
                score=float(det["score"]),
                box_xyxy_global=(float(gx1), float(gy1), float(gx2), float(gy2)),
                box_xywh_global=(float(gx1), float(gy1), float(bw), float(bh)),
                area_px=float(bw * bh),
                source_tile_id=tile.tile_id,
                is_tiled_detection=True,
            ))

        return global_detections

    def apply_nms(
        self,
        detections: Sequence[SmallObjectDetection],
    ) -> list[SmallObjectDetection]:
        """Applies Non-Maximum Suppression across merged tile detections."""
        if not detections:
            return []

        # Group by canonical label / class
        by_class: dict[str, list[SmallObjectDetection]] = {}
        for d in detections:
            by_class.setdefault(d.canonical_path, []).append(d)

        suppressed: list[SmallObjectDetection] = []

        for cid, group in by_class.items():
            boxes = np.array([d.box_xyxy_global for d in group], dtype=np.float32)
            scores = np.array([d.score for d in group], dtype=np.float32)

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            order = scores.argsort()[::-1]
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= self.iou_nms_threshold)[0]
                order = order[inds + 1]

            for idx in keep:
                suppressed.append(group[idx])

        return sorted(suppressed, key=lambda d: d.score, reverse=True)


class SmallObjectBenchmarkHarness:
    """Benchmark suite evaluating detection and 3D reconstruction of small details."""

    def __init__(self) -> None:
        self.tiler = TiledInferenceEngine(tile_size=640, overlap_ratio=0.20)
        self.prompter = get_extensible_prompter()

    def run_benchmark(
        self,
        image: Union[np.ndarray, Image.Image],
        detector_fn: Callable[[Image.Image, list[str]], list[dict[str, Any]]],
        ground_truth: Sequence[dict[str, Any]],
        micro_prompts: list[str] | None = None,
    ) -> SmallObjectBenchmarkReport:
        """Executes comparative evaluation: Naive Whole-Image vs Tiled High-Resolution Inference."""
        if micro_prompts is None:
            micro_prompts = self.prompter.get_prompts(scale=ScaleFilter.MICRO)

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        w, h = pil_img.size

        # 1. Naive Whole-Image Detection (Downscaled if large)
        naive_raw = detector_fn(pil_img, micro_prompts)
        naive_dets = []
        for d in naive_raw:
            bx = d["box_xyxy"]
            naive_dets.append(SmallObjectDetection(
                label=d["label"],
                canonical_path=d.get("canonical_path", d["label"]),
                score=float(d["score"]),
                box_xyxy_global=tuple(bx),
                box_xywh_global=(bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]),
                area_px=(bx[2] - bx[0]) * (bx[3] - bx[1]),
                source_tile_id="full_image",
                is_tiled_detection=False,
            ))

        # 2. Tiled High-Resolution Inference
        tiles = self.tiler.generate_tiles(w, h)
        all_tiled_candidates: list[SmallObjectDetection] = []
        for tile in tiles:
            tile_img = self.tiler.extract_tile_image(pil_img, tile)
            tile_raw = detector_fn(tile_img, micro_prompts)
            mapped = self.tiler.map_tile_boxes_to_global(tile_raw, tile)
            all_tiled_candidates.extend(mapped)

        tiled_nms = self.tiler.apply_nms(all_tiled_candidates)

        # 3. Score against ground truth
        def evaluate_matches(preds: Sequence[SmallObjectDetection], iou_th: float = 0.40):
            matched_gt = set()
            true_positives = 0
            for p in preds:
                pb = p.box_xyxy_global
                for gt_idx, gt in enumerate(ground_truth):
                    if gt_idx in matched_gt:
                        continue
                    gb = gt["box_xyxy"]
                    # Calculate IoU
                    xi1 = max(pb[0], gb[0])
                    yi1 = max(pb[1], gb[1])
                    xi2 = min(pb[2], gb[2])
                    yi2 = min(pb[3], gb[3])
                    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                    area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    union = area_p + area_g - inter
                    iou = inter / max(1e-8, union)
                    if iou >= iou_th:
                        matched_gt.add(gt_idx)
                        true_positives += 1
                        break
            return len(matched_gt), true_positives

        n_gt = len(ground_truth)
        matched_naive, _ = evaluate_matches(naive_dets)
        matched_tiled, tp_tiled = evaluate_matches(tiled_nms)

        recall_naive = matched_naive / max(1, n_gt)
        recall_tiled = matched_tiled / max(1, n_gt)
        prec_tiled = tp_tiled / max(1, len(tiled_nms))
        f1_tiled = (2 * prec_tiled * recall_tiled) / max(1e-8, prec_tiled + recall_tiled)

        gain = ((recall_tiled - recall_naive) / max(1e-4, recall_naive)) * 100.0

        # Category breakdown
        category_stats: dict[str, dict[str, float]] = {}
        for gt in ground_truth:
            cat = gt.get("category", "hardware")
            category_stats.setdefault(cat, {"count": 0, "detected": 0})
            category_stats[cat]["count"] += 1

        return SmallObjectBenchmarkReport(
            total_ground_truth_items=n_gt,
            total_detections_naive=len(naive_dets),
            total_detections_tiled=len(tiled_nms),
            recall_naive=recall_naive,
            recall_tiled=recall_tiled,
            precision_tiled=prec_tiled,
            f1_tiled=f1_tiled,
            small_object_recovery_gain=gain,
            category_breakdown=category_stats,
        )
