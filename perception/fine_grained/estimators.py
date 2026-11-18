"""Fine-grained fixture and component parameter estimators (Mandate Phase 3 & 6).

Provides deterministic physical parameter estimation for:
1. Electrical switchboards: gang count, aspect ratio, socket grounding, and orientation.
2. Plumbing pipe runs: nominal standard diameters (15mm to 110mm), outer diameters, and roles.
3. Plumbing ball valves: handle alignment angle, open/closed/partial state, and confidence.
4. Support surfaces: horizontal plane classification, usable support area, elevation, and tilt.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SwitchboardAttributes:
    """Estimated parameters for an electrical switchboard or outlet faceplate."""

    gang_count: int
    orientation_deg: float
    is_grounded: bool
    socket_standard: str
    aspect_ratio: float
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SwitchboardEstimator:
    """Estimates electrical faceplate gang count and orientation from physical geometry."""

    # Standard gang dimensions (metres): 1-gang: 86mm, 2-gang: 146mm, 3-gang: 206mm, 4-gang: 266mm
    GANG_ASPECT_THRESHOLDS = (
        (1, 1.35),  # aspect < 1.35 -> 1-gang
        (2, 2.05),  # 1.35 <= aspect < 2.05 -> 2-gang
        (3, 2.75),  # 2.05 <= aspect < 2.75 -> 3-gang
    )

    @classmethod
    def estimate(
        cls,
        width_m: float,
        height_m: float,
        depth_m: float = 0.02,
        module_detections: Sequence[str] | None = None,
        is_vertical: bool | None = None,
    ) -> SwitchboardAttributes:
        """Estimate gang count, orientation, and grounding from faceplate dimensions."""
        w = max(0.01, float(width_m))
        h = max(0.01, float(height_m))
        major_dim = max(w, h)
        minor_dim = min(w, h)
        aspect = major_dim / minor_dim

        # Determine gang count
        if module_detections and len(module_detections) > 0:
            gang_count = len(module_detections)
            conf = 0.95
        else:
            if aspect < 1.35:
                gang_count = 1
            elif aspect < 2.05:
                gang_count = 2
            elif aspect < 2.75:
                gang_count = 3
            else:
                gang_count = min(6, int(round(aspect)))
            conf = 0.88 if gang_count in (1, 2, 3) else 0.75

        # Orientation: 0 deg (horizontal) vs 90 deg (vertical)
        if is_vertical is not None:
            orientation = 90.0 if is_vertical else 0.0
        else:
            orientation = 90.0 if h > w * 1.15 else 0.0

        # Grounding detection: 3-pin standards or grounded modules
        grounded = True
        std = "universal_grounded"
        if module_detections:
            grounded = any("ground" in m.lower() or "3pin" in m.lower() or "socket" in m.lower() for m in module_detections)

        return SwitchboardAttributes(
            gang_count=gang_count,
            orientation_deg=orientation,
            is_grounded=grounded,
            socket_standard=std,
            aspect_ratio=round(aspect, 3),
            confidence=round(conf, 3),
            notes=f"Estimated {gang_count}-gang faceplate (aspect {aspect:.2f})",
        )


@dataclass(frozen=True)
class PipeAttributes:
    """Estimated parameters for a plumbing pipe run."""

    nominal_diameter_mm: float
    outer_diameter_mm: float
    nominal_size_inch: str
    standard_role: str
    length_m: float
    error_mm: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipeRunEstimator:
    """Estimates standard nominal plumbing diameters from observed pipe thickness."""

    # Standard nominal sizes: (nominal_mm, outer_mm, inch_str, typical_role)
    NOMINAL_STANDARDS = (
        (15.0, 15.0, '1/2"', "potable_supply"),
        (22.0, 22.0, '3/4"', "supply_branch"),
        (28.0, 28.0, '1"', "main_supply"),
        (32.0, 36.0, '1-1/4"', "basin_waste"),
        (40.0, 43.0, '1-1/2"', "sink_waste"),
        (50.0, 56.0, '2"', "branch_drain"),
        (75.0, 82.0, '3"', "vent_drain"),
        (110.0, 110.0, '4"', "soil_stack"),
    )

    @classmethod
    def estimate(
        cls,
        measured_thickness_m: float,
        length_m: float = 1.0,
    ) -> PipeAttributes:
        """Snap measured pipe outer thickness to closest nominal standard schedule."""
        measured_mm = max(1.0, float(measured_thickness_m) * 1000.0)

        best_diff = float("inf")
        best_spec = cls.NOMINAL_STANDARDS[0]

        for spec in cls.NOMINAL_STANDARDS:
            outer_mm = spec[1]
            diff = abs(measured_mm - outer_mm)
            if diff < best_diff:
                best_diff = diff
                best_spec = spec

        # Confidence decays as deviation from nominal grows
        # A 5mm deviation on a 22mm pipe reduces confidence
        rel_error = best_diff / best_spec[1]
        conf = max(0.40, min(0.98, 1.0 - rel_error * 1.5))

        return PipeAttributes(
            nominal_diameter_mm=best_spec[0],
            outer_diameter_mm=best_spec[1],
            nominal_size_inch=best_spec[2],
            standard_role=best_spec[3],
            length_m=round(max(0.05, float(length_m)), 4),
            error_mm=round(best_diff, 2),
            confidence=round(conf, 3),
        )


@dataclass(frozen=True)
class ValveAttributes:
    """Estimated state and angle for a plumbing shutoff ball valve."""

    state: str  # "OPEN", "CLOSED", "PARTIAL"
    handle_angle_deg: float
    valve_type: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BallValveEstimator:
    """Estimates shutoff valve open/closed state from handle orientation relative to pipe axis."""

    @classmethod
    def estimate(
        cls,
        handle_vector: Sequence[float],
        pipe_axis: Sequence[float],
        valve_type: str = "ball_valve",
    ) -> ValveAttributes:
        """Compute alignment angle between valve handle and pipe flow axis."""
        h = [float(v) for v in handle_vector[:3]]
        p = [float(v) for v in pipe_axis[:3]]

        norm_h = math.sqrt(sum(x * x for x in h))
        norm_p = math.sqrt(sum(x * x for x in p))

        if norm_h < 1e-6 or norm_p < 1e-6:
            return ValveAttributes(
                state="UNKNOWN",
                handle_angle_deg=0.0,
                valve_type=valve_type,
                confidence=0.0,
            )

        dot = abs(sum(h[i] * p[i] for i in range(3))) / (norm_h * norm_p)
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        # In a ball valve:
        # Handle parallel to pipe (angle near 0 deg) -> OPEN
        # Handle perpendicular to pipe (angle near 90 deg) -> CLOSED
        if angle_deg <= 20.0:
            state = "OPEN"
            conf = 0.95 - (angle_deg / 20.0) * 0.15
        elif angle_deg >= 70.0:
            state = "CLOSED"
            conf = 0.95 - ((90.0 - angle_deg) / 20.0) * 0.15
        else:
            state = "PARTIAL"
            # Middle of range has higher confidence in being partially throttled
            conf = 0.85

        return ValveAttributes(
            state=state,
            handle_angle_deg=round(angle_deg, 2),
            valve_type=valve_type,
            confidence=round(conf, 3),
        )


@dataclass(frozen=True)
class SupportSurfaceInfo:
    """Classification parameters for horizontal support planes (tables, counters, shelves)."""

    is_support_surface: bool
    role: str  # "support_surface", "floor", "ceiling", "other"
    usable_area_m2: float
    elevation_above_floor_m: float
    tilt_from_horizontal_deg: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupportSurfaceClassifier:
    """Classifies horizontal entity surfaces into support surfaces vs floors/ceilings."""

    # Upward normal vector threshold (cos(15 deg) ~ 0.966)
    MIN_VERTICAL_COS = 0.96

    @classmethod
    def classify(
        cls,
        normal: Sequence[float],
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        floor_elevation_m: float = 0.0,
        up_axis: int = 2,  # 2 for Z-up, 1 for Y-up
    ) -> SupportSurfaceInfo:
        """Classify plane into support surface vs floor/ceiling."""
        norm = [float(v) for v in normal[:3]]
        mag = math.sqrt(sum(x * x for x in norm))
        if mag > 1e-6:
            norm = [x / mag for x in norm]
        else:
            norm = [0.0, 0.0, 1.0]

        up_component = norm[up_axis]
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, abs(up_component)))))

        # Compute bounding dimensions
        bmin = [float(v) for v in bounds_min[:3]]
        bmax = [float(v) for v in bounds_max[:3]]

        # Elevation of top surface
        top_elevation = bmax[up_axis]
        elevation_above_floor = top_elevation - float(floor_elevation_m)

        # Planar area perpendicular to up_axis
        horiz_axes = [i for i in (0, 1, 2) if i != up_axis]
        span_a = max(0.01, bmax[horiz_axes[0]] - bmin[horiz_axes[0]])
        span_b = max(0.01, bmax[horiz_axes[1]] - bmin[horiz_axes[1]])
        usable_area = span_a * span_b

        is_near_horizontal = abs(up_component) >= cls.MIN_VERTICAL_COS

        if not is_near_horizontal:
            return SupportSurfaceInfo(
                is_support_surface=False,
                role="wall_or_slanted",
                usable_area_m2=round(usable_area, 4),
                elevation_above_floor_m=round(elevation_above_floor, 4),
                tilt_from_horizontal_deg=round(tilt_deg, 2),
                confidence=0.90,
            )

        # Check elevation ranges:
        # Floor: elevation <= 0.15m
        # Support surface: 0.20m to 2.20m (tables ~0.75m, counters ~0.90m, shelves 0.3m-2.0m)
        # Ceiling: > 2.20m
        if elevation_above_floor <= 0.15:
            role = "floor"
            is_support = False
            conf = 0.95
        elif 0.20 <= elevation_above_floor <= 2.20:
            role = "support_surface"
            is_support = True
            conf = 0.92
        else:
            role = "ceiling"
            is_support = False
            conf = 0.90

        return SupportSurfaceInfo(
            is_support_surface=is_support,
            role=role,
            usable_area_m2=round(usable_area, 4),
            elevation_above_floor_m=round(elevation_above_floor, 4),
            tilt_from_horizontal_deg=round(tilt_deg, 2),
            confidence=round(conf, 3),
        )
