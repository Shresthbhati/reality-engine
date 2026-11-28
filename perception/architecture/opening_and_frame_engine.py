"""Team C — Opening and Frame Engine.

Specialized analysis of doors, windows, openings, frames, columns and beams
in indoor RGB imagery.

Capabilities:
  - Door/window/opening/frame/column/beam classification
  - Cross-view structural continuity for openings
  - Contradiction detection (e.g., door wider than its frame)
  - Alternative hypothesis preservation
  - Geometric attribute extraction (aspect ratio, relative height)

Design principles:
  - Never asserts a door type without detection evidence.
  - Frame-without-opening and opening-without-frame are both valid hypotheses.
  - Cross-view contradictions are preserved, not silently resolved.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("reality.architecture.opening_and_frame_engine")


# ---------------------------------------------------------------------------
# Opening types
# ---------------------------------------------------------------------------
class OpeningType(str, Enum):
    DOOR      = "door"
    WINDOW    = "window"
    ARCHWAY   = "archway"
    PASS_THROUGH = "pass_through"
    SKYLIGHT  = "skylight"
    UNKNOWN   = "unknown"


class FrameType(str, Enum):
    DOOR_FRAME    = "door_frame"
    WINDOW_FRAME  = "window_frame"
    COLUMN        = "column"
    BEAM          = "beam"
    LINTEL        = "lintel"
    UNKNOWN       = "unknown"


# ---------------------------------------------------------------------------
# Opening observation
# ---------------------------------------------------------------------------
@dataclass
class OpeningObservation:
    """A detected architectural opening."""
    observation_id: str
    image_id: str
    opening_type: OpeningType
    bbox_px: tuple[float, float, float, float]
    image_width: int
    image_height: int
    confidence: float
    aspect_ratio: float         # width / height of bounding box
    relative_height: float      # bbox height / image height
    relative_bottom: float      # bbox bottom / image height (1.0 = touches bottom)
    is_full_height: bool        # occupies nearly full image height
    likely_door: bool           # geometric heuristic
    likely_window: bool         # geometric heuristic
    source_engine: str
    alternative_types: tuple[OpeningType, ...]
    notes: list[str]

    @staticmethod
    def from_bbox(
        observation_id: str,
        image_id: str,
        label: str,
        bbox_px: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
        confidence: float,
        source_engine: str = "opening_frame_engine",
    ) -> "OpeningObservation":
        x1, y1, x2, y2 = bbox_px
        w = x2 - x1
        h = y2 - y1
        aspect = w / max(h, 1.0)
        rel_h = h / max(image_height, 1.0)
        rel_bottom = y2 / max(image_height, 1.0)
        is_full = rel_h > 0.75

        # Geometric heuristics
        likely_door = aspect < 0.7 and rel_bottom > 0.85 and rel_h > 0.4
        likely_window = 0.5 < aspect < 3.0 and rel_bottom < 0.90 and rel_h < 0.6

        # Map label to type
        label_l = label.lower()
        if "door" in label_l:
            otype = OpeningType.DOOR
            alts = (OpeningType.ARCHWAY,) if likely_door else ()
        elif "window" in label_l or "skylight" in label_l:
            otype = OpeningType.WINDOW if "sky" not in label_l else OpeningType.SKYLIGHT
            alts = ()
        elif "arch" in label_l:
            otype = OpeningType.ARCHWAY
            alts = (OpeningType.DOOR,)
        else:
            otype = OpeningType.UNKNOWN
            alts = (OpeningType.DOOR, OpeningType.WINDOW)

        notes: list[str] = []
        if is_full:
            notes.append("Full-height opening — likely door or archway.")
        if likely_door and otype == OpeningType.WINDOW:
            notes.append("Geometric profile suggests door despite 'window' label.")
        if likely_window and otype == OpeningType.DOOR:
            notes.append("Geometric profile suggests window despite 'door' label.")

        return OpeningObservation(
            observation_id=observation_id,
            image_id=image_id,
            opening_type=otype,
            bbox_px=bbox_px,
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
            aspect_ratio=aspect,
            relative_height=rel_h,
            relative_bottom=rel_bottom,
            is_full_height=is_full,
            likely_door=likely_door,
            likely_window=likely_window,
            source_engine=source_engine,
            alternative_types=alts,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Frame observation
# ---------------------------------------------------------------------------
@dataclass
class FrameObservation:
    """A detected architectural frame, column, or beam."""
    observation_id: str
    image_id: str
    frame_type: FrameType
    bbox_px: tuple[float, float, float, float]
    image_width: int
    image_height: int
    confidence: float
    aspect_ratio: float
    is_vertical: bool    # column-like (height > width)
    is_horizontal: bool  # beam-like (width > height)
    source_engine: str
    associated_opening_id: str | None
    notes: list[str]


def classify_frame(
    observation_id: str,
    image_id: str,
    label: str,
    bbox_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    confidence: float,
    source_engine: str = "opening_frame_engine",
) -> FrameObservation:
    """Classify a detected frame, column, or beam observation.

    Args:
        observation_id: Observation identifier.
        image_id: Image identifier.
        label: Detector output label.
        bbox_px: (x1, y1, x2, y2) in image pixels.
        image_width: Image width for geometric analysis.
        image_height: Image height for geometric analysis.
        confidence: Detector confidence.
        source_engine: Source engine identifier.

    Returns:
        ``FrameObservation`` with type classification.
    """
    x1, y1, x2, y2 = bbox_px
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    aspect = w / h
    is_vert = h > w * 1.5
    is_horiz = w > h * 1.5
    label_l = label.lower()
    notes: list[str] = []

    if "column" in label_l or "pillar" in label_l or "post" in label_l:
        ftype = FrameType.COLUMN
        if not is_vert:
            notes.append(f"Label 'column' but aspect ratio {aspect:.2f} is not vertical.")
    elif "beam" in label_l or "joist" in label_l or "lintel" in label_l:
        ftype = FrameType.BEAM if "lintel" not in label_l else FrameType.LINTEL
        if not is_horiz:
            notes.append(f"Label 'beam' but aspect ratio {aspect:.2f} is not horizontal.")
    elif "door frame" in label_l or "door_frame" in label_l:
        ftype = FrameType.DOOR_FRAME
    elif "window frame" in label_l or "window_frame" in label_l:
        ftype = FrameType.WINDOW_FRAME
    else:
        # Geometric fallback
        if is_vert:
            ftype = FrameType.COLUMN
            notes.append("Classified as column from vertical aspect ratio.")
        elif is_horiz and y1 / image_height < 0.3:
            ftype = FrameType.BEAM
            notes.append("Classified as beam from horizontal aspect ratio in upper region.")
        else:
            ftype = FrameType.UNKNOWN

    return FrameObservation(
        observation_id=observation_id,
        image_id=image_id,
        frame_type=ftype,
        bbox_px=bbox_px,
        image_width=image_width,
        image_height=image_height,
        confidence=confidence,
        aspect_ratio=aspect,
        is_vertical=is_vert,
        is_horizontal=is_horiz,
        source_engine=source_engine,
        associated_opening_id=None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Opening-frame association
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpeningFrameAssociation:
    """Linkage between an opening and its surrounding frame."""
    association_id: str
    opening_observation_id: str
    frame_observation_id: str
    image_id: str
    spatial_consistency: float    # 0–1: how well frame surrounds opening
    contradiction: str | None     # human-readable if contradiction detected


def _box_iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-6)


def associate_openings_with_frames(
    openings: list[OpeningObservation],
    frames: list[FrameObservation],
    *,
    min_iou_for_association: float = 0.15,
    max_size_ratio: float = 1.5,
) -> list[OpeningFrameAssociation]:
    """Associate detected openings with their surrounding frames.

    Args:
        openings: Detected opening observations.
        frames: Detected frame observations.
        min_iou_for_association: Minimum bbox IoU to consider association.
        max_size_ratio: Maximum frame/opening size ratio before flagging contradiction.

    Returns:
        List of ``OpeningFrameAssociation`` with spatial consistency scores.
    """
    associations: list[OpeningFrameAssociation] = []
    for opening in openings:
        for frame in frames:
            if opening.image_id != frame.image_id:
                continue
            iou = _box_iou(opening.bbox_px, frame.bbox_px)
            if iou < min_iou_for_association:
                continue

            ox1, oy1, ox2, oy2 = opening.bbox_px
            fx1, fy1, fx2, fy2 = frame.bbox_px
            opening_area = max(1.0, (ox2-ox1)*(oy2-oy1))
            frame_area = max(1.0, (fx2-fx1)*(fy2-fy1))
            size_ratio = frame_area / opening_area

            contradiction: str | None = None
            if size_ratio < 0.5:
                contradiction = (
                    f"Frame ({frame.frame_type}) is smaller than its opening "
                    f"({opening.opening_type}) — geometrically implausible."
                )
            elif size_ratio > max_size_ratio * 3:
                contradiction = (
                    f"Frame is {size_ratio:.1f}x larger than opening — "
                    f"likely misassociation."
                )

            spatial_consistency = min(1.0, iou * 2.0)
            assoc_id = hashlib.sha1(
                f"{opening.observation_id}:{frame.observation_id}".encode()
            ).hexdigest()[:8]

            associations.append(OpeningFrameAssociation(
                association_id=f"assoc_{assoc_id}",
                opening_observation_id=opening.observation_id,
                frame_observation_id=frame.observation_id,
                image_id=opening.image_id,
                spatial_consistency=spatial_consistency,
                contradiction=contradiction,
            ))

    return associations


# ---------------------------------------------------------------------------
# Contradiction detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructuralContradiction:
    """A detected logical contradiction between structural observations."""
    contradiction_id: str
    description: str
    involved_observation_ids: tuple[str, ...]
    severity: str   # "minor" | "moderate" | "critical"


def detect_structural_contradictions(
    openings: list[OpeningObservation],
    frames: list[FrameObservation],
    associations: list[OpeningFrameAssociation],
) -> list[StructuralContradiction]:
    """Find logical contradictions in structural observations.

    Args:
        openings: Opening observations.
        frames: Frame observations.
        associations: Opening-frame associations.

    Returns:
        List of detected contradictions.
    """
    contradictions: list[StructuralContradiction] = []

    # From association contradictions
    for assoc in associations:
        if assoc.contradiction:
            cid = hashlib.sha1(assoc.association_id.encode()).hexdigest()[:6]
            contradictions.append(StructuralContradiction(
                contradiction_id=f"cont_{cid}",
                description=assoc.contradiction,
                involved_observation_ids=(
                    assoc.opening_observation_id,
                    assoc.frame_observation_id,
                ),
                severity="moderate",
            ))

    # Duplicate openings in same location
    for i in range(len(openings)):
        for j in range(i + 1, len(openings)):
            o1, o2 = openings[i], openings[j]
            if o1.image_id != o2.image_id:
                continue
            iou = _box_iou(o1.bbox_px, o2.bbox_px)
            if iou > 0.80:
                cid = hashlib.sha1(f"{o1.observation_id}:{o2.observation_id}".encode()).hexdigest()[:6]
                contradictions.append(StructuralContradiction(
                    contradiction_id=f"cont_{cid}",
                    description=(
                        f"Opening {o1.observation_id} ({o1.opening_type}) and "
                        f"{o2.observation_id} ({o2.opening_type}) highly overlap "
                        f"(IoU={iou:.2f}) — likely duplicate detections."
                    ),
                    involved_observation_ids=(o1.observation_id, o2.observation_id),
                    severity="minor",
                ))

    return contradictions
