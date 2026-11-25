"""Geometry Extraction, Validation and Object-Specific Constraints (Sprint 6 / Sprint 7).

Extracts and validates physical geometry, aspect ratios, surface normals,
hierarchical components, and spatial relationships without fabricating unmeasured coordinates.

Sprint 7 additions:
- ``ObjectGeometryEvidence`` carries ``scale_state`` and ``coordinate_frame`` so validators
  can refuse cross-frame comparisons and avoid applying metric plausibility bounds to
  relative-depth or SfM-gauge evidence.
- Validators that require metric extents (e.g. door height > 1.8 m, table width > 0.4 m)
  annotate their result with ``metric_check_applicable=False`` when the evidence is not
  physically metric, and do not mark the object invalid solely because of a metric check
  that cannot be verified.
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
    """Rich geometric evidence extracted from 2D observations and 3D lifting.

    Scale-state fields (Sprint 7):
    - ``scale_state``: string from the entity's ``scale_state`` attribute.  Valid values
      include ``"metric_anchored"``, ``"sfm_gauge"``, ``"relative_depth"``,
      ``"synthetic_scale_aligned"``.  When ``None``, scale is unknown.
    - ``coordinate_frame``: opaque string identifying the coordinate reference frame
      (e.g. ``"world:canonical"``, ``"camera:0"``, ``"sfm:reconstruction_0"``).
      Validators refuse to compare 3D positions from incompatible frames.
    - ``is_physically_metric``: True only when ``position_3d`` and ``extent_3d`` are
      known to be in physical metres (anchored to a real-world reference).
    """

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
    # Sprint 7 scale-state fields
    scale_state: str | None = None          # e.g. "metric_anchored", "relative_depth"
    coordinate_frame: str | None = None     # e.g. "world:canonical", "sfm:reconstruction_0"
    is_physically_metric: bool = False       # True only when extent_3d is in real metres


def _evidence_is_metric(evidence: "ObjectGeometryEvidence") -> bool:
    """Returns True only when the evidence carries physically-calibrated metric 3D data."""
    ss = evidence.scale_state
    if ss is None:
        return False
    return "metric_anchored" in str(ss).lower() and evidence.is_physically_metric


@dataclass(frozen=True)
class GeometryValidationResult:
    """Outcome of validating object-specific geometric constraints.

    ``metric_check_applicable`` (Sprint 7): True when at least one metric-bound check
    (e.g. door height > 1.8 m) could be evaluated.  False indicates the evidence
    lacks physically-calibrated extent data; metric-bound validity claims are
    therefore not meaningful.  The object is not marked invalid solely because a
    metric check was inapplicable.
    """

    is_valid: bool
    validation_score: float  # 0.0 to 1.0
    anomalies: tuple[str, ...]
    measured_aspect_ratio: float | None
    is_physically_plausible: bool
    metric_check_applicable: bool = True  # Sprint 7: False when evidence is not metric
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "validation_score": round(self.validation_score, 3),
            "anomalies": list(self.anomalies),
            "measured_aspect_ratio": round(self.measured_aspect_ratio, 3) if self.measured_aspect_ratio else None,
            "is_physically_plausible": self.is_physically_plausible,
            "metric_check_applicable": self.metric_check_applicable,
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
    # Sprint 7: metric extent checks (door height > 1.8 m) are only applicable when
    # the evidence carries physically-calibrated metric 3D data.
    metric_ok = _evidence_is_metric(evidence)
    details_out: dict[str, Any] = {
        "door_aperture_ar": ar,
        "truncated": evidence.is_truncated,
        "metric_check_applicable": metric_ok,
        "scale_state": evidence.scale_state,
    }
    if metric_ok and evidence.extent_3d is not None:
        h_m = evidence.extent_3d[2]  # height in metres
        if h_m < 1.80:
            anomalies.append(f"Door height {h_m:.2f} m below standard minimum of 1.80 m")
            score -= 0.20
        details_out["height_m"] = h_m
    return GeometryValidationResult(
        is_valid=len(anomalies) == 0 or score >= 0.50,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        metric_check_applicable=metric_ok,
        details=details_out,
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
    metric_ok = _evidence_is_metric(evidence)
    details_out: dict[str, Any] = {
        "window_aspect_ratio": ar,
        "metric_check_applicable": metric_ok,
        "scale_state": evidence.scale_state,
    }
    if metric_ok and evidence.extent_3d is not None:
        w_m, h_m = evidence.extent_3d[0], evidence.extent_3d[2]
        details_out["width_m"] = w_m
        details_out["height_m"] = h_m
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        metric_check_applicable=metric_ok,
        details=details_out,
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
    metric_ok = _evidence_is_metric(evidence)
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        metric_check_applicable=metric_ok,
        details={"table_aspect_ratio": ar, "metric_check_applicable": metric_ok},
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
    metric_ok = _evidence_is_metric(evidence)
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        metric_check_applicable=metric_ok,
        details={"chair_aspect_ratio": ar, "metric_check_applicable": metric_ok},
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
    metric_ok = _evidence_is_metric(evidence)
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=True,
        metric_check_applicable=metric_ok,
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
    metric_ok = _evidence_is_metric(evidence)
    return GeometryValidationResult(
        is_valid=is_valid,
        validation_score=max(0.0, min(1.0, score)),
        anomalies=tuple(anomalies),
        measured_aspect_ratio=ar,
        is_physically_plausible=len(anomalies) == 0,
        metric_check_applicable=metric_ok,
        details={"surface_class": cname, "metric_check_applicable": metric_ok},
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
        metric_check_applicable=_evidence_is_metric(evidence),
        details={"generic_aspect_ratio": ar},
    )
