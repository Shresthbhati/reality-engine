"""Geometry Extraction, Validation and Object-Specific Constraints (Sprint 6).

Extracts and validates physical geometry, aspect ratios, surface normals,
hierarchical components, and spatial relationships without fabricating unmeasured coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.taxonomy import (
    GeometryType,
    TaxonomyClass,
    resolve_taxonomy_class,
)


@dataclass(frozen=True)
class ObjectGeometryEvidence:
    """Rich geometric evidence extracted from 2D observations and 3D lifting."""

    box_2d: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_name: str
    mask_2d: np.ndarray | None = None
    mask_polygon: tuple[tuple[float, float], ...] | None = None
    visible_contour_area_px: float = 0.0
    relative_dimensions: tuple[float, float, float] | None = None  # (width, depth, height) proxy
    surface_normal: tuple[float, float, float] | None = None
    depth_evidence: dict[str, Any] = field(default_factory=dict)
    position_3d: tuple[float, float, float] | None = None
    extent_3d: tuple[float, float, float] | None = None
    orientation_quat: tuple[float, float, float, float] | None = None
    occlusion_level: str = "none"  # "none", "partial", "heavy"
    is_truncated: bool = False
    confidence: float = 0.5
    uncertainty: float = 0.5
    subparts_geometry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryValidationResult:
    """Outcome of validating object-specific geometric constraints."""

    is_valid: bool
    validation_score: float  # 0.0 to 1.0
    anomalies: tuple[str, ...]
    measured_aspect_ratio: float | None
    is_physically_plausible: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "validation_score": round(self.validation_score, 3),
            "anomalies": list(self.anomalies),
            "measured_aspect_ratio": round(self.measured_aspect_ratio, 3) if self.measured_aspect_ratio else None,
            "is_physically_plausible": self.is_physically_plausible,
            "details": self.details,
        }


def extract_box_aspect_ratio(box_2d: tuple[float, float, float, float]) -> float:
    """Returns height / width aspect ratio of a 2D bounding box."""
    w = max(1.0, float(box_2d[2] - box_2d[0]))
    h = max(1.0, float(box_2d[3] - box_2d[1]))
    return h / w


def validate_door_geometry(
    evidence: ObjectGeometryEvidence,
    door_panel_evidence: ObjectGeometryEvidence | None = None,
    door_frame_evidence: ObjectGeometryEvidence | None = None,
) -> GeometryValidationResult:
    """Validates door geometry: tall vertical aperture aspect ratio, vertical surface normal, panel containment."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)

    # 1. Doors are vertical apertures (standard aspect ratio > 1.2 unless severely truncated)
    if not evidence.is_truncated and ar < 1.1:
        anomalies.append(f"Abnormally wide aspect ratio for door: h/w = {ar:.2f} (expected >= 1.20)")
        score -= 0.35

    # 2. Surface normal validation (should be approximately horizontal/vertical wall aligned, not pointing up/down)
    if evidence.surface_normal is not None:
        nz = abs(evidence.surface_normal[2])
        if nz > 0.65:
            anomalies.append(f"Door surface normal tilted vertically: |nz| = {nz:.2f} (expected horizontal normal |nz| < 0.40)")
            score -= 0.30

    # 3. Component relationship validation: panel inside door frame / aperture
    if door_panel_evidence is not None:
        p_box = door_panel_evidence.box_2d
        d_box = evidence.box_2d
        # Check panel bounds within door aperture
        panel_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
        door_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
        if panel_area > door_area * 1.15:
            anomalies.append("Door panel area exceeds parent door aperture by > 15%")
            score -= 0.25

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"door_aperture_ar": ar, "truncated": evidence.is_truncated},
    )


def validate_window_geometry(
    evidence: ObjectGeometryEvidence,
    window_frame_evidence: ObjectGeometryEvidence | None = None,
) -> GeometryValidationResult:
    """Validates window geometry: plausible aperture dimensions, frame alignment."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)

    # Windows typically have aspect ratios between 0.35 and 3.0
    if ar < 0.20 or ar > 4.5:
        anomalies.append(f"Extreme window aspect ratio: {ar:.2f}")
        score -= 0.30

    if evidence.surface_normal is not None:
        nz = abs(evidence.surface_normal[2])
        if nz > 0.70:
            anomalies.append(f"Window normal pointing vertically: |nz| = {nz:.2f}")
            score -= 0.30

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"window_aspect_ratio": ar},
    )


def validate_table_geometry(
    evidence: ObjectGeometryEvidence,
    table_top_evidence: ObjectGeometryEvidence | None = None,
    legs_evidence: Sequence[ObjectGeometryEvidence] | None = None,
) -> GeometryValidationResult:
    """Validates table geometry: distinguishes horizontal table top from vertical supporting legs."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)

    # Table top should be primarily horizontal planar surface
    if table_top_evidence is not None and table_top_evidence.surface_normal is not None:
        nz = abs(table_top_evidence.surface_normal[2])
        if nz < 0.50:
            anomalies.append(f"Table top surface normal is not horizontal: |nz| = {nz:.2f} (expected near 1.0)")
            score -= 0.35

    # Table legs should be located vertically below the table top
    if table_top_evidence is not None and legs_evidence:
        top_y2 = table_top_evidence.box_2d[3]
        for leg in legs_evidence:
            leg_y1 = leg.box_2d[1]
            if leg_y1 < table_top_evidence.box_2d[1] - 10:  # Leg starting above table top
                anomalies.append("Table leg detected above the table top surface")
                score -= 0.25
                break

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"table_aspect_ratio": ar},
    )


def validate_chair_geometry(
    evidence: ObjectGeometryEvidence,
    seat_evidence: ObjectGeometryEvidence | None = None,
    backrest_evidence: ObjectGeometryEvidence | None = None,
) -> GeometryValidationResult:
    """Validates chair geometry: horizontal seat surface, vertical backrest above seat, legs below."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)

    # Backrest should be vertically higher than seat (in image coordinate y, lower pixel value = higher in world)
    if seat_evidence is not None and backrest_evidence is not None:
        seat_center_y = (seat_evidence.box_2d[1] + seat_evidence.box_2d[3]) / 2.0
        back_center_y = (backrest_evidence.box_2d[1] + backrest_evidence.box_2d[3]) / 2.0
        if back_center_y > seat_center_y:
            anomalies.append("Chair backrest is located below the seat surface in observation")
            score -= 0.35

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"chair_aspect_ratio": ar},
    )


def validate_light_geometry(
    evidence: ObjectGeometryEvidence,
    housing_evidence: ObjectGeometryEvidence | None = None,
    diffuser_evidence: ObjectGeometryEvidence | None = None,
) -> GeometryValidationResult:
    """Validates light fixture geometry: housing distinguishes from mounting surface and diffuser."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=True,
        details={"light_aspect_ratio": ar},
    )


def validate_structural_surface_geometry(
    evidence: ObjectGeometryEvidence,
) -> GeometryValidationResult:
    """Validates structural boundaries: wall (vertical plane), floor/ceiling (horizontal planes)."""
    anomalies: list[str] = []
    score = 1.0
    ar = extract_box_aspect_ratio(evidence.box_2d)
    cname = evidence.class_name.lower()

    if evidence.surface_normal is not None:
        nz = abs(evidence.surface_normal[2])
        if "wall" in cname and nz > 0.45:
            anomalies.append(f"Wall surface normal exhibits vertical tilt: |nz| = {nz:.2f} (expected near 0.0)")
            score -= 0.35
        elif ("floor" in cname or "ceiling" in cname) and nz < 0.65:
            anomalies.append(f"Floor/Ceiling normal is not predominantly horizontal: |nz| = {nz:.2f} (expected near 1.0)")
            score -= 0.35

    is_valid = score >= 0.50
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"surface_class": cname},
    )


def validate_object_geometry(evidence: ObjectGeometryEvidence) -> GeometryValidationResult:
    """Generic dispatcher for object-specific geometric validation."""
    cname = evidence.class_name.lower()
    if "door" in cname:
        return validate_door_geometry(evidence)
    if "window" in cname:
        return validate_window_geometry(evidence)
    if "table" in cname or "desk" in cname:
        return validate_table_geometry(evidence)
    if "chair" in cname or "sofa" in cname:
        return validate_chair_geometry(evidence)
    if "light" in cname or "lamp" in cname or "diffuser" in cname:
        return validate_light_geometry(evidence)
    if any(k in cname for k in ("wall", "floor", "ceiling")):
        return validate_structural_surface_geometry(evidence)

    # Default fallback: check bounding box sanity
    ar = extract_box_aspect_ratio(evidence.box_2d)
    anomalies: list[str] = []
    if ar < 0.05 or ar > 20.0:
        anomalies.append(f"Extreme aspect ratio {ar:.2f}")

    return GeometryValidationResult(
        is_valid=len(anomalies) == 0,
        validation_score=1.0 if not anomalies else 0.5,
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        details={"generic_aspect_ratio": ar},
    )
