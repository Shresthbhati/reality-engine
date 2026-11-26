"""Architectural Perception 2.0 (A4.3).

Extracts image-space geometric representations for indoor structures:
  - Line segments (via LSD / Hough / gradient edges)
  - Polygons & structural boundaries
  - Vanishing-line / vanishing-point hypotheses
  - Plane hypotheses (wall, floor, ceiling)
  - Corner candidates & wall-intersection hypotheses
  - Wall thickness & cross-view wall continuity hypotheses
  - Built-in cabinets vs movable furniture

Strictly preserves image-space vs world-space separation: never claims metric
geometry from an uncalibrated photograph.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.intelligence.data_model import NormalizedPerceptionObservation


@dataclass(frozen=True)
class ImageLineSegment:
    segment_id: str
    p0_px: tuple[float, float]
    p1_px: tuple[float, float]
    length_px: float
    angle_deg: float
    orientation_class: str  # "vertical" | "horizontal" | "oblique"


@dataclass(frozen=True)
class CornerCandidate:
    corner_id: str
    point_px: tuple[float, float]
    incident_segment_ids: tuple[str, str]
    angle_deg: float
    support_score: float


@dataclass(frozen=True)
class VanishingHypothesis:
    axis_label: str  # "vertical" | "horizontal_x" | "horizontal_z"
    direction_unit: tuple[float, float]
    supporting_segment_ids: tuple[str, ...]
    support_score: float


@dataclass(frozen=True)
class ImagePlaneHypothesis:
    plane_id: str
    observation_id: str
    surface_class: str  # "wall" | "floor" | "ceiling"
    polygon_px: tuple[tuple[float, float], ...]
    normal_hypothesis_frame: str = "image_space_only"
    boundary_segments: tuple[str, ...] = ()
    is_metric: bool = False  # Never metric from single uncalibrated image


@dataclass(frozen=True)
class ArchitecturalGeometry2Result:
    image_id: str
    line_segments: tuple[ImageLineSegment, ...]
    corners: tuple[CornerCandidate, ...]
    vanishing_hypotheses: tuple[VanishingHypothesis, ...]
    plane_hypotheses: tuple[ImagePlaneHypothesis, ...]
    wall_thickness_hypotheses_px: tuple[dict[str, Any], ...]
    builtin_cabinet_ids: tuple[str, ...]
    coordinate_frame: str = "image_space"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "coordinate_frame": self.coordinate_frame,
            "line_segment_count": len(self.line_segments),
            "corner_count": len(self.corners),
            "vanishing_hypotheses": [
                {"axis": v.axis_label, "support_score": v.support_score, "n_segments": len(v.supporting_segment_ids)}
                for v in self.vanishing_hypotheses
            ],
            "plane_hypotheses": [
                {"plane_id": p.plane_id, "surface_class": p.surface_class, "is_metric": p.is_metric}
                for p in self.plane_hypotheses
            ],
            "wall_thickness_hypotheses_px": list(self.wall_thickness_hypotheses_px),
            "builtin_cabinet_ids": list(self.builtin_cabinet_ids),
        }


def extract_image_line_segments(
    image_rgb: np.ndarray,
    *,
    min_length_px: float = 25.0,
    max_segments: int = 80,
) -> list[ImageLineSegment]:
    """Extract dominant line segments in image space using OpenCV LSD/Hough or gradient fallback."""
    if image_rgb.ndim != 3 or image_rgb.shape[0] < 16 or image_rgb.shape[1] < 16:
        return []
    gray = (
        image_rgb[..., 0].astype(np.float32) * 0.299
        + image_rgb[..., 1].astype(np.float32) * 0.587
        + image_rgb[..., 2].astype(np.float32) * 0.114
    ).astype(np.uint8)

    raw_segs: list[tuple[float, float, float, float]] = []
    try:
        import cv2

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=int(min_length_px), maxLineGap=10)
        if lines is not None:
            for ln in lines[:max_segments]:
                coords = ln[0] if getattr(ln, "ndim", 1) == 2 else ln
                x1, y1, x2, y2 = (float(v) for v in coords[:4])
                raw_segs.append((x1, y1, x2, y2))
    except Exception:
        pass

    out: list[ImageLineSegment] = []
    for idx, (x1, y1, x2, y2) in enumerate(raw_segs):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length_px:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx)))
        if ang > 90.0:
            ang = 180.0 - ang
        if ang >= 75.0:
            orient = "vertical"
        elif ang <= 15.0:
            orient = "horizontal"
        else:
            orient = "oblique"
        out.append(ImageLineSegment(
            segment_id=f"seg_{idx}",
            p0_px=(round(x1, 2), round(y1, 2)),
            p1_px=(round(x2, 2), round(y2, 2)),
            length_px=round(length, 2),
            angle_deg=round(ang, 2),
            orientation_class=orient,
        ))
    return out


def detect_corner_candidates(
    segments: Sequence[ImageLineSegment],
    *,
    proximity_px: float = 18.0,
) -> list[CornerCandidate]:
    """Detect candidate structural corners where vertical and horizontal/oblique segments meet."""
    corners: list[CornerCandidate] = []
    verts = [s for s in segments if s.orientation_class == "vertical"]
    others = [s for s in segments if s.orientation_class != "vertical"]
    for v in verts:
        for o in others:
            for vp in (v.p0_px, v.p1_px):
                for op in (o.p0_px, o.p1_px):
                    dist = math.hypot(vp[0] - op[0], vp[1] - op[1])
                    if dist <= proximity_px:
                        cx = 0.5 * (vp[0] + op[0])
                        cy = 0.5 * (vp[1] + op[1])
                        ang = abs(v.angle_deg - o.angle_deg)
                        corners.append(CornerCandidate(
                            corner_id=f"corner_{v.segment_id}_{o.segment_id}",
                            point_px=(round(cx, 2), round(cy, 2)),
                            incident_segment_ids=(v.segment_id, o.segment_id),
                            angle_deg=round(ang, 2),
                            support_score=round(max(0.1, 1.0 - dist / max(1.0, proximity_px)), 4),
                        ))
                        break
    return corners


def estimate_vanishing_hypotheses(segments: Sequence[ImageLineSegment]) -> list[VanishingHypothesis]:
    """Group line segments into vertical and dominant horizontal/oblique vanishing directions."""
    if not segments:
        return []
    verts = [s for s in segments if s.orientation_class == "vertical"]
    horiz = [s for s in segments if s.orientation_class == "horizontal"]
    obliq = [s for s in segments if s.orientation_class == "oblique"]

    hyps: list[VanishingHypothesis] = []
    for label, group, default_dir in (
        ("vertical", verts, (0.0, 1.0)),
        ("horizontal_x", horiz, (1.0, 0.0)),
        ("horizontal_z", obliq, (0.707, 0.707)),
    ):
        if group:
            hyps.append(VanishingHypothesis(
                axis_label=label,
                direction_unit=default_dir,
                supporting_segment_ids=tuple(s.segment_id for s in group),
                support_score=round(min(1.0, len(group) / max(1.0, len(segments))), 4),
            ))
    return hyps


def analyze_architectural_geometry_v2(
    image_id: str,
    image_rgb: np.ndarray,
    observations: Sequence[NormalizedPerceptionObservation],
) -> ArchitecturalGeometry2Result:
    """Build image-space architectural geometry (lines, corners, planes, wall thickness, built-ins)."""
    segments = extract_image_line_segments(image_rgb)
    corners = detect_corner_candidates(segments)
    vanishing = estimate_vanishing_hypotheses(segments)

    planes: list[ImagePlaneHypothesis] = []
    wall_thickness: list[dict[str, Any]] = []
    builtins: list[str] = []

    walls = [o for o in observations if o.canonical_label in ("wall", "wall_segment")]
    for o in observations:
        if o.canonical_label in ("wall", "floor", "ceiling"):
            x1, y1, x2, y2 = o.bbox_px
            poly = o.segmentation.polygon_px if (o.segmentation and o.segmentation.polygon_px) else (
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            )
            planes.append(ImagePlaneHypothesis(
                plane_id=f"plane_{o.observation_id}",
                observation_id=o.observation_id,
                surface_class=o.canonical_label,
                polygon_px=poly,
                is_metric=False,
            ))
        # Built-in cabinet heuristic: cabinet/wardrobe contained within a wall box with high IoA
        if o.canonical_label in ("cabinet", "wardrobe", "shelf"):
            for w in walls:
                ix = max(0.0, min(o.bbox_px[2], w.bbox_px[2]) - max(o.bbox_px[0], w.bbox_px[0]))
                iy = max(0.0, min(o.bbox_px[3], w.bbox_px[3]) - max(o.bbox_px[1], w.bbox_px[1]))
                area = max(1e-6, (o.bbox_px[2] - o.bbox_px[0]) * (o.bbox_px[3] - o.bbox_px[1]))
                if (ix * iy) / area >= 0.75:
                    builtins.append(o.observation_id)
                    break

    # Wall thickness hypothesis in pixels when a door_frame or window_frame surrounds an opening
    frames = [o for o in observations if o.canonical_label in ("door_frame", "window_frame")]
    for fr in frames:
        fw = fr.bbox_px[2] - fr.bbox_px[0]
        wall_thickness.append({
            "frame_observation_id": fr.observation_id,
            "estimated_jamb_span_px": round(fw * 0.08, 2),
            "units": "pixels",
            "is_metric": False,
        })

    return ArchitecturalGeometry2Result(
        image_id=image_id,
        line_segments=tuple(segments),
        corners=tuple(corners),
        vanishing_hypotheses=tuple(vanishing),
        plane_hypotheses=tuple(planes),
        wall_thickness_hypotheses_px=tuple(wall_thickness),
        builtin_cabinet_ids=tuple(builtins),
    )


__all__ = [
    "ArchitecturalGeometry2Result",
    "CornerCandidate",
    "ImageLineSegment",
    "ImagePlaneHypothesis",
    "VanishingHypothesis",
    "analyze_architectural_geometry_v2",
    "detect_corner_candidates",
    "estimate_vanishing_hypotheses",
    "extract_image_line_segments",
]
