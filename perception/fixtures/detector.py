"""Fine-grained architectural fixture detector and 3D surface localizer (Task 3.4, Epic E05-1).

Integrates:
1. Two-tier Patch Inspection Engine (orthorectified wall crops -> fine-grained detector).
2. Open-vocabulary taxonomy alignment (electrical, plumbing, mechanical, opening hardware).
3. Metric size-prior validation against taxonomy plausibility limits.
4. Exact 3D wall-surface back-projection and orientation.
5. Multi-view 3D association and non-maximum suppression (NMS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from perception.detection.interface import ImageDetection
from perception.entities import taxonomy
from perception.fixtures.patch_extractor import OrthorectifiedPatchExtractor, WallPatch
from provenance import Provenance, Uncertainty
from reconstruction.calibration.camera import PinholeCamera
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Relationship,
    RelationshipKind,
    Vector3,
)


class FixtureCategory(str, Enum):
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    MECHANICAL = "hvac"
    HARDWARE = "hardware"


# Mapping from common open-vocabulary detection queries to taxonomy class IDs
FIXTURE_PROMPT_TO_CLASS: Dict[str, str] = {
    # Electrical
    "power socket": "electrical.switchboard.module.socket",
    "wall socket": "electrical.switchboard.module.socket",
    "electrical socket": "electrical.switchboard.module.socket",
    "plug point": "electrical.switchboard.module.socket",
    "light switch": "electrical.switchboard.module.switch",
    "wall switch": "electrical.switchboard.module.switch",
    "rocker switch": "electrical.switchboard.module.switch",
    "switchboard": "electrical.switchboard",
    "switch plate": "electrical.switchboard",
    "wall socket plate": "electrical.switchboard",
    "electrical faceplate": "electrical.switchboard",
    "distribution board": "electrical.distribution_board",
    "fuse box": "electrical.distribution_board",
    "circuit breaker panel": "electrical.distribution_board",
    "breaker panel": "electrical.distribution_board",
    # Plumbing
    "water faucet": "plumbing.tap",
    "tap": "plumbing.tap",
    "faucet": "plumbing.tap",
    "shower head": "plumbing.tap",
    "sink drain": "plumbing.drain",
    "floor drain": "plumbing.drain",
    "drain": "plumbing.drain",
    # Mechanical / HVAC
    "hvac vent": "hvac.vent",
    "air vent": "hvac.vent",
    "ventilation grille": "hvac.vent",
    "vent": "hvac.vent",
    "thermostat": "hvac.thermostat",
    "radiator": "hvac.radiator",
    # Opening hardware
    "door handle": "opening.door.handle",
    "door knob": "opening.door.handle",
    "lever handle": "opening.door.handle",
}


@dataclass(frozen=True)
class DetectedFixture:
    """A fine-grained architectural fixture localized in 3D world space."""

    fixture_id: str
    class_id: str
    category: FixtureCategory
    label: str
    confidence: float
    center_world: Tuple[float, float, float]
    normal_world: Tuple[float, float, float]
    up_world: Tuple[float, float, float]
    dimensions_m: Tuple[float, float, float]  # (width, height, depth/protrusion)
    patch_bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in patch px
    source_evidence_ids: List[str]
    source_patch_id: str
    host_plane_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_world_entity(self, entity_id: Optional[str] = None) -> Entity:
        """Convert localized fixture into a canonical WorldIR Entity."""
        eid = entity_id or self.fixture_id
        cx, cy, cz = self.center_world
        w, h, d = self.dimensions_m

        # AABB bounds
        half_w, half_h, half_d = w * 0.5, h * 0.5, d * 0.5
        bmin = [cx - half_w, cy - half_h, cz - half_d]
        bmax = [cx + half_w, cy + half_h, cz + half_d]

        observations = [
            Observation(
                id=f"obs-fixture-{ev_id}",
                sensor_type="camera",
                frame_id=ev_id,
                confidence=self.confidence,
            )
            for ev_id in self.source_evidence_ids
        ]

        custom_props: Dict[str, Any] = {
            "entity_class": self.class_id,
            "category": self.category.value,
            "normal": list(self.normal_world),
            "up": list(self.up_world),
            "dimensions_m": [round(float(v), 4) for v in self.dimensions_m],
            "bounds_min": [round(float(v), 4) for v in bmin],
            "bounds_max": [round(float(v), 4) for v in bmax],
            "attributes": self.attributes,
        }
        if self.host_plane_id is not None:
            custom_props["host_plane_id"] = self.host_plane_id

        return Entity(
            id=eid,
            type=EntityType.STRUCTURE,
            name=self.label,
            relationships=[],
            provenance=Provenance.RECONSTRUCTED,
            confidence=self.confidence,
            uncertainty=Uncertainty(confidence=self.confidence),
            observations=observations,
            custom_properties=custom_props,
        )


class FixtureDetector:
    """Detects and 3D-localizes fine-grained architectural fixtures on walls and ceilings."""

    def __init__(
        self,
        detector_backend: Any = None,
        min_confidence: float = 0.35,
        iou_threshold: float = 0.45,
        merge_distance_3d_m: float = 0.08,
        patch_extractor: Optional[OrthorectifiedPatchExtractor] = None,
    ):
        self.detector_backend = detector_backend
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.merge_distance_3d_m = merge_distance_3d_m
        self.patch_extractor = patch_extractor or OrthorectifiedPatchExtractor()

    @staticmethod
    def _category_for_class(class_id: str) -> FixtureCategory:
        if class_id.startswith("electrical"):
            return FixtureCategory.ELECTRICAL
        elif class_id.startswith("plumbing"):
            return FixtureCategory.PLUMBING
        elif class_id.startswith("hvac"):
            return FixtureCategory.MECHANICAL
        else:
            return FixtureCategory.HARDWARE

    def _validate_size_prior(self, class_id: str, width_m: float, height_m: float) -> bool:
        """Check whether detected dimensions are physically plausible for the taxonomy class."""
        try:
            ent_cls = taxonomy.get(class_id)
        except KeyError:
            return True

        largest_dim = max(width_m, height_m)
        min_p, max_p = ent_cls.size_m

        # Allow 40% margin below and 60% margin above plausibility prior for small features
        if largest_dim < min_p * 0.4 or largest_dim > max_p * 1.6:
            return False
        return True

    def detect_in_patch(self, patch: WallPatch) -> List[DetectedFixture]:
        """Detect fixtures within a single orthorectified wall patch."""
        detections: List[DetectedFixture] = []
        raw_detections: List[Tuple[str, float, Tuple[float, float, float, float]]] = []

        queries = list(FIXTURE_PROMPT_TO_CLASS.keys())

        # If backend is provided, call backend
        if self.detector_backend is not None:
            if hasattr(self.detector_backend, "detect"):
                results = self.detector_backend.detect(patch.image, queries)
                for res in results:
                    lbl = res.label.lower().strip()
                    if res.score >= self.min_confidence and lbl in FIXTURE_PROMPT_TO_CLASS:
                        b = getattr(res, "box_xyxy", getattr(res, "box", None))
                        if b is not None:
                            raw_detections.append((FIXTURE_PROMPT_TO_CLASS[lbl], res.score, b))
        else:
            # Deterministic visual inspection heuristic for test/offline environments
            raw_detections = self._heuristic_patch_detector(patch)

        # 2D NMS on patch detections
        filtered_raw = self._apply_2d_nms(raw_detections, self.iou_threshold)

        for idx, (class_id, score, box) in enumerate(filtered_raw):
            center_3d, (w_m, h_m, d_m), norm_3d = patch.box_to_world_bounds(box)

            # Plausibility check against taxonomy size prior
            if not self._validate_size_prior(class_id, w_m, h_m):
                continue

            try:
                ent_cls = taxonomy.get(class_id)
                label = ent_cls.label
            except KeyError:
                label = class_id.split(".")[-1].replace("_", " ").title()

            cat = self._category_for_class(class_id)
            fix_id = f"fix-{patch.source_evidence_id}-{class_id.split('.')[-1]}-{idx}"

            detections.append(
                DetectedFixture(
                    fixture_id=fix_id,
                    class_id=class_id,
                    category=cat,
                    label=label,
                    confidence=float(score),
                    center_world=center_3d,
                    normal_world=norm_3d,
                    up_world=patch.v_axis_world,
                    dimensions_m=(w_m, h_m, d_m),
                    patch_bbox=box,
                    source_evidence_ids=[patch.source_evidence_id],
                    source_patch_id=patch.patch_id,
                    host_plane_id=patch.plane_id,
                    attributes={"detector": "patch_inspection", "stage": "fine"},
                )
            )

        return detections

    def _heuristic_patch_detector(
        self,
        patch: WallPatch,
    ) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """High-precision deterministic contour/gradient/color detector for rectangular wall fixtures."""
        results: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        gray = cv2.cvtColor(patch.image, cv2.COLOR_RGB2GRAY)

        # Find high-contrast edges and contour rectangles
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ppm = float(patch.pixels_per_meter)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (0.02 * ppm) ** 2:  # Smaller than 2cm x 2cm
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # Check for quadrilateral (rectangles/squares typical of sockets and switch plates)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                w_m, h_m = w / ppm, h / ppm
                aspect = max(w_m, h_m) / max(min(w_m, h_m), 1e-4)

                # 1. Square or near-square modular socket / switchboard
                if aspect <= 1.4:
                    if 0.04 <= max(w_m, h_m) <= 0.12:
                        results.append(("electrical.switchboard.module.socket", 0.92, (float(x), float(y), float(x + w), float(y + h))))
                    elif 0.12 < max(w_m, h_m) <= 0.35:
                        results.append(("electrical.switchboard", 0.88, (float(x), float(y), float(x + w), float(y + h))))
                # 2. Rectangular multi-gang switchboard or distribution board
                elif aspect <= 3.0:
                    if 0.08 <= max(w_m, h_m) <= 0.35:
                        results.append(("electrical.switchboard", 0.86, (float(x), float(y), float(x + w), float(y + h))))
                    elif 0.35 < max(w_m, h_m) <= 1.0:
                        results.append(("electrical.distribution_board", 0.84, (float(x), float(y), float(x + w), float(y + h))))

        return results

    @staticmethod
    def _apply_2d_nms(
        detections: List[Tuple[str, float, Tuple[float, float, float, float]]],
        iou_thresh: float,
    ) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d[1], reverse=True)
        keep: List[Tuple[str, float, Tuple[float, float, float, float]]] = []

        def iou(b1, b2):
            ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            inter = ix * iy
            union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
            return inter / union if union > 0 else 0.0

        for cand in sorted_dets:
            c_cls, c_score, c_box = cand
            if all(k[0] != c_cls or iou(k[2], c_box) < iou_thresh for k in keep):
                keep.append(cand)

        return keep

    def merge_multiview_fixtures(
        self,
        fixtures: List[DetectedFixture],
    ) -> List[DetectedFixture]:
        """Associate and merge duplicate 3D fixture observations across multiple views."""
        if not fixtures:
            return []

        merged: List[DetectedFixture] = []
        visited = set()

        for i in range(len(fixtures)):
            if i in visited:
                continue

            cluster = [fixtures[i]]
            visited.add(i)

            c_pos = np.array(fixtures[i].center_world, dtype=np.float64)
            c_cls = fixtures[i].class_id

            for j in range(i + 1, len(fixtures)):
                if j in visited:
                    continue

                if fixtures[j].class_id == c_cls:
                    o_pos = np.array(fixtures[j].center_world, dtype=np.float64)
                    dist = float(np.linalg.norm(c_pos - o_pos))

                    if dist <= self.merge_distance_3d_m:
                        cluster.append(fixtures[j])
                        visited.add(j)

            # Fuse cluster
            all_ev_ids = sorted(set(ev for fix in cluster for ev in fix.source_evidence_ids))
            weights = np.array([fix.confidence for fix in cluster], dtype=np.float64)
            total_weight = float(np.sum(weights))

            pos_arr = np.array([fix.center_world for fix in cluster], dtype=np.float64)
            avg_center = np.average(pos_arr, axis=0, weights=weights)

            # Boost confidence for multi-view verified fixtures
            fused_conf = min(0.99, max(f.confidence for f in cluster) + 0.05 * (len(cluster) - 1))

            primary = cluster[0]
            merged_fixture = DetectedFixture(
                fixture_id=f"fused-{primary.class_id.split('.')[-1]}-{len(merged)}",
                class_id=primary.class_id,
                category=primary.category,
                label=primary.label,
                confidence=float(fused_conf),
                center_world=(float(avg_center[0]), float(avg_center[1]), float(avg_center[2])),
                normal_world=primary.normal_world,
                up_world=primary.up_world,
                dimensions_m=primary.dimensions_m,
                patch_bbox=primary.patch_bbox,
                source_evidence_ids=all_ev_ids,
                source_patch_id=primary.source_patch_id,
                host_plane_id=primary.host_plane_id,
                attributes={
                    **primary.attributes,
                    "multi_view_observations": len(cluster),
                },
            )
            merged.append(merged_fixture)

        return merged

    def detect_in_scene(
        self,
        cameras: Dict[str, PinholeCamera],
        images_rgb: Dict[str, np.ndarray],
        planes: Sequence[Tuple[str, Sequence[float], float]],  # (plane_id, normal, d)
        plane_bounds: Optional[Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None,
    ) -> List[DetectedFixture]:
        """Run full-scene fine-grained fixture inspection across all planes and camera views."""
        all_raw_fixtures: List[DetectedFixture] = []

        for plane_id, normal, d in planes:
            bounds = plane_bounds.get(plane_id) if plane_bounds else None

            for ev_id, cam in cameras.items():
                if ev_id not in images_rgb:
                    continue

                img = images_rgb[ev_id]
                patches = self.patch_extractor.extract_tiled_patches(
                    camera=cam,
                    image_rgb=img,
                    plane_normal=normal,
                    plane_d=d,
                    evidence_id=ev_id,
                    plane_id=plane_id,
                    plane_bounds_world=bounds,
                )

                for patch in patches:
                    fixtures = self.detect_in_patch(patch)
                    all_raw_fixtures.extend(fixtures)

        # Merge co-visible observations across views
        fused = self.merge_multiview_fixtures(all_raw_fixtures)
        return fused
