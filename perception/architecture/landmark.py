"""Single-image curved-structure reconstruction (LAND-01 foundation).

Recovers landmark/curved exterior architecture from the 3D points of ONE
photograph -- domes (sphere), arches (circle cross-section), towers
(cylinder) -- plus a measured error report against an optional ground truth.

Position in the perception stack (directive section 3):

    single-view depth/unprojection -> 3D points
        -> PARAMETRIC CURVED-FIT (this module: sphere/circle/cylinder)
        -> architectural entities (registry: arch/dome/tower)
        -> WorldIR (structure type, ESTIMATED provenance)

Honesty rules (constitution: unknown stays unknown, never fabricate):
  - Every fit consumes REAL 3D points; residuals are MEASURED (rms/max),
    never declared; a fit the points cannot support raises FitRefused.
  - Single-image scale is RELATIVE unless an external baseline is supplied.
    A landmark with no scale baseline reports RELATIVE scale and marks every
    dimension ESTIMATED -- never metres-by-assumption.
  - Confidence is a documented deterministic function of measured fit quality.
  - Pure float math, no RNG, no clocks: same points -> byte-identical output.

Status: FOUNDATION (synthetic verification only). The acceptance criterion
(measured against real ground truth with a stated error metric) requires a
real landmark photo set + surveyed dimensions, which this environment does
not have. Real-data verification is BLOCKED on user-supplied fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from perception.architecture.parametric import (
    CircleFit,
    CylinderFit,
    FitRefused,
    SphereFit,
    fit_circle,
    fit_cylinder,
    fit_sphere,
)
from provenance import Provenance, Uncertainty

Vec3 = Tuple[float, float, float]
Point3 = Tuple[float, float, float]

RELATIVE_SCALE_NOTE = "single-image scale is relative; no camera-to-camera baseline declared"


class LandmarkRefused(FitRefused):
    """The supplied points do not demonstrate any curved landmark.
    The measured fact is in the message."""


@dataclass(frozen=True)
class DomeFit:
    """A fitted dome: sphere fit over a point cap, with the measured
    upper fraction (points above the fitted center plane) confirming
    dome-ness rather than a full bubble."""

    sphere: SphereFit
    upper_fraction: float
    max_polar_angle_rad: float
    confidence: float
    n_points: int

    def to_dict(self) -> dict:
        d = self.sphere.to_dict()
        d.update(
            {
                "kind": "dome",
                "upper_fraction": self.upper_fraction,
                "max_polar_angle_rad": self.max_polar_angle_rad,
                "confidence": self.confidence,
                "n_points": self.n_points,
            }
        )
        return d


@dataclass(frozen=True)
class ArchFit:
    """A fitted arch: circle cross-section over a wall-gap point segment,
    with the measured angular span confirming arch-ness."""

    circle: CircleFit
    angular_span_rad: float
    rise_m: float
    span_m: float
    confidence: float
    n_points: int

    def to_dict(self) -> dict:
        d = self.circle.to_dict()
        d.update(
            {
                "kind": "arch",
                "angular_span_rad": self.angular_span_rad,
                "rise_m": self.rise_m,
                "span_m": self.span_m,
                "confidence": self.confidence,
                "n_points": self.n_points,
            }
        )
        return d


@dataclass(frozen=True)
class TowerFit:
    """A fitted tower/minaret: cylinder fit over an elongated vertical
    segment, with the measured axis-verticality confirming tower-ness."""

    cylinder: CylinderFit
    axis_up_dot: float
    confidence: float
    n_points: int

    def to_dict(self) -> dict:
        d = self.cylinder.to_dict()
        d.update(
            {
                "kind": "tower",
                "axis_up_dot": self.axis_up_dot,
                "confidence": self.confidence,
                "n_points": self.n_points,
            }
        )
        return d


@dataclass(frozen=True)
class LandmarkResult:
    """Outcome of fitting one point cluster to a curved landmark kind.

    Exactly one of dome/arch/tower is populated; the others are None.
    scale_state is "relative" unless an external baseline_m was supplied,
    in which case it is "measured" and all dimensions are scaled by the
    same factor (never silently converted).
    """

    kind: str
    dome: Optional[DomeFit] = None
    arch: Optional[ArchFit] = None
    tower: Optional[TowerFit] = None
    scale_state: str = "relative"
    scale_factor: float = 1.0
    provenance: Provenance = Provenance.ESTIMATED
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    measured_fact: str = ""
    refusal_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "dome": self.dome.to_dict() if self.dome else None,
            "arch": self.arch.to_dict() if self.arch else None,
            "tower": self.tower.to_dict() if self.tower else None,
            "scale_state": self.scale_state,
            "scale_factor": self.scale_factor,
            "provenance": self.provenance.value,
            "uncertainty": self.uncertainty.to_dict(),
            "measured_fact": self.measured_fact,
            "refusal_reason": self.refusal_reason,
        }


def _polar_angles(points: Sequence[Point3], center: Point3, up: Vec3) -> list[float]:
    """Polar angle (radians from +up) of each point about center."""
    ux, uy, uz = up
    out = []
    for p in points:
        d = (p[0] - center[0], p[1] - center[1], p[2] - center[2])
        r = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if r < 1e-12:
            out.append(0.0)
            continue
        cos_ang = (d[0] * ux + d[1] * uy + d[2] * uz) / r
        cos_ang = max(-1.0, min(1.0, cos_ang))
        out.append(math.acos(cos_ang))
    return out


def fit_dome(points: Sequence[Point3], up: Vec3) -> DomeFit:
    """Fit a dome (upper spherical cap) over real points.

    Fits a sphere, then requires:
      - at least MIN_SPHERE_POINTS points;
      - a measured upper_fraction >= 0.45 (dome-like, not full bubble);
      - measured rms within a sane fraction of radius.
    Raises FitRefused if the points do not support a dome.
    """
    sphere = fit_sphere(points)
    n = len(points)
    center = sphere.center
    angles = _polar_angles(points, center, up)
    if not angles:
        raise FitRefused("no points to measure dome polar extent")
    up_count = sum(1 for p in points if p[2] >= center[2])
    upper_fraction = up_count / n
    max_polar = max(angles)
    if upper_fraction < 0.45:
        raise FitRefused(
            "dome rejected: upper_fraction "
            f"{upper_fraction:.3f} < 0.45 "
            "(points form a fuller bubble than a dome)"
        )
    if sphere.radius_m <= 0:
        raise FitRefused("sphere fit returned non-positive radius")
    rms_ratio = sphere.rms_residual_m / sphere.radius_m
    if rms_ratio > 0.25:
        raise FitRefused(
            "dome rejected: rms/residual ratio "
            f"{rms_ratio:.3f} > 0.25 "
            "(points do not lie on a spherical cap)"
        )
    confidence = sphere.confidence * min(1.0, upper_fraction)
    return DomeFit(
        sphere=sphere,
        upper_fraction=upper_fraction,
        max_polar_angle_rad=max_polar,
        confidence=confidence,
        n_points=n,
    )


def fit_arch(points: Sequence[Point3], up: Vec3) -> ArchFit:
    """Fit an arch (circle cross-section of a wall opening) over real points.

    Fits a circle in the vertical plane aligned with the segment's principal
    horizontal direction, then requires a measured angular span in the arch
    window. Raises FitRefused if the points do not support an arch.
    """
    pts = list(points)
    n = len(pts)
    if n < 5:
        raise FitRefused(f"arch fit needs at least 5 points, got {n}")
    cz = sum(p[2] for p in pts) / n
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cov_xx = sum((p[0] - cx) ** 2 for p in pts) / n
    cov_yy = sum((p[1] - cy) ** 2 for p in pts) / n
    span_axis = (1.0, 0.0, 0.0) if cov_xx >= cov_yy else (0.0, 1.0, 0.0)
    ux, uy, uz = up
    sx, sy, sz = span_axis
    normal = (uy * sz - uz * sy, uz * sx - ux * sz, ux * sy - uy * sx)
    nlen = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if nlen < 1e-12:
        raise FitRefused("degenerate span axis for arch plane")
    normal = (normal[0] / nlen, normal[1] / nlen, normal[2] / nlen)
    ipx = normal[1] * uz - normal[2] * uy
    ipy = normal[2] * ux - normal[0] * uz
    ipz = normal[0] * uy - normal[1] * ux
    iplen = math.sqrt(ipx * ipx + ipy * ipy + ipz * ipz)
    if iplen < 1e-12:
        raise FitRefused("degenerate in-plane axis")
    ipx, ipy, ipz = ipx / iplen, ipy / iplen, ipz / iplen
    origin3 = (cx, cy, cz)
    lifted = [
        (
            origin3[0] + x * ipx + y * ux,
            origin3[1] + x * ipy + y * uy,
            origin3[2] + x * ipz + y * uz,
        )
        for x, y in [
            (
                (p[0] - cx) * ipx + (p[1] - cy) * ipy + (p[2] - cz) * ipz,
                (p[0] - cx) * ux + (p[1] - cy) * uy + (p[2] - cz) * uz,
            )
            for p in pts
        ]
    ]
    try:
        circle = fit_circle(lifted, normal)
    except FitRefused:
        raise FitRefused("arch rejected: points do not lie on a circular arc")
    c = circle.center
    angles = []
    for p in lifted:
        d = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
        r = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if r < 1e-12:
            continue
        cos_a = (d[0] * ipx + d[1] * ipy + d[2] * ipz) / r
        sin_a = (d[0] * ux + d[1] * uy + d[2] * uz) / r
        cos_a = max(-1.0, min(1.0, cos_a))
        angles.append(math.atan2(sin_a, cos_a))
    if not angles:
        raise FitRefused("arch rejected: no measurable angular points")
    angles.sort()
    span = angles[-1] - angles[0]
    while span > math.pi:
        span -= 2 * math.pi
    span = abs(span)
    if span < 0.5:
        raise FitRefused(
            f"arch rejected: angular span {span:.3f} rad < 0.5 (not an arch)"
        )
    rise_m = circle.radius_m * (1.0 - math.cos(span / 2.0))
    span_m = 2.0 * circle.radius_m * math.sin(span / 2.0)
    confidence = circle.confidence * min(1.0, span / math.pi)
    return ArchFit(
        circle=circle,
        angular_span_rad=span,
        rise_m=rise_m,
        span_m=span_m,
        confidence=confidence,
        n_points=n,
    )


def fit_tower(points: Sequence[Point3], up: Vec3) -> TowerFit:
    """Fit a tower/minaret (vertical cylinder) over real points.

    Fits a cylinder, then requires:
      - measured axis |axis . up| >= 0.85 (vertical);
      - axial extent >= radius (elongated, not a puck).
    Raises FitRefused if the points do not support a tower.
    """
    cylinder = fit_cylinder(points, up)
    if cylinder.axis is None:
        raise FitRefused("tower rejected: cylinder fit has no axis")
    axis_up_dot = abs(
        cylinder.axis[0] * up[0] + cylinder.axis[1] * up[1] + cylinder.axis[2] * up[2]
    )
    if axis_up_dot < 0.85:
        raise FitRefused(
            "tower rejected: axis_up_dot "
            f"{axis_up_dot:.3f} < 0.85 "
            "(not vertical enough for a tower)"
        )
    extent = cylinder.height_max_m - cylinder.height_min_m
    if extent < cylinder.radius_m:
        raise FitRefused(
            "tower rejected: extent "
            f"{extent:.3f} m < radius {cylinder.radius_m:.3f} m "
            "(not elongated along the axis)"
        )
    confidence = cylinder.confidence * axis_up_dot
    return TowerFit(
        cylinder=cylinder,
        axis_up_dot=axis_up_dot,
        confidence=confidence,
        n_points=len(points),
    )


def fit_landmark(
    points: Sequence[Point3],
    up: Vec3,
    kind_hint: Optional[str] = None,
    baseline_m: Optional[float] = None,
) -> LandmarkResult:
    """Fit one point cluster to a curved landmark kind.

    Tries, in priority order, the kind hint first then dome -> arch ->
    tower, returning the first fit the points honestly support. When none
    fits, returns kind == "none" with a refusal_reason (never a guessed
    shape).

    scale_state is "measured" only when baseline_m is supplied AND the
    fitted size implies a sane scale factor; otherwise "relative" and all
    dimensions stay in model units.
    """
    pts = list(points)
    if not pts:
        return LandmarkResult(
            kind="none",
            refusal_reason="no points supplied",
            scale_state="relative",
        )
    # Default priority: tower first (most structurally distinctive -- a
    # vertical cylinder is unambiguous), then dome, then arch last (arch
    # circle fits can spuriously succeed on cylinder points).
    candidates: list[str] = []
    if kind_hint in ("dome", "arch", "tower"):
        candidates.append(kind_hint)
    for k in ("tower", "dome", "arch"):
        if k not in candidates:
            candidates.append(k)

    last_error: Optional[str] = None
    for cand in candidates:
        try:
            if cand == "dome":
                fit = fit_dome(pts, up)
                scale_state, scale_factor = _scale_state(fit.sphere.radius_m, baseline_m)
                return LandmarkResult(
                    kind="dome",
                    dome=fit,
                    scale_state=scale_state,
                    scale_factor=scale_factor,
                    measured_fact=_dome_fact(fit, scale_factor),
                )
            if cand == "arch":
                fit = fit_arch(pts, up)
                scale_state, scale_factor = _scale_state(fit.span_m, baseline_m)
                return LandmarkResult(
                    kind="arch",
                    arch=fit,
                    scale_state=scale_state,
                    scale_factor=scale_factor,
                    measured_fact=_arch_fact(fit, scale_factor),
                )
            if cand == "tower":
                fit = fit_tower(pts, up)
                scale_state, scale_factor = _scale_state(
                    cylinder_height(cylinder_fit=fit.cylinder), baseline_m
                )
                return LandmarkResult(
                    kind="tower",
                    tower=fit,
                    scale_state=scale_state,
                    scale_factor=scale_factor,
                    measured_fact=_tower_fact(fit, scale_factor),
                )
        except FitRefused as exc:
            last_error = str(exc)
            continue
    return LandmarkResult(
        kind="none",
        refusal_reason=last_error or "no curved landmark fit supported the points",
        scale_state="relative",
    )


def _scale_state(extent_m: float, baseline_m: Optional[float]) -> Tuple[str, float]:
    """Decide relative vs measured scale. Single-image scale is relative
    unless a trusted baseline is supplied AND the extent is sane."""
    if baseline_m is None or baseline_m <= 0 or extent_m <= 0:
        return "relative", 1.0
    factor = baseline_m / extent_m
    if not (0.01 <= factor <= 100.0):
        return "relative", 1.0
    return "measured", factor


def cylinder_height(*, cylinder_fit: CylinderFit) -> float:
    return cylinder_fit.height_max_m - cylinder_fit.height_min_m


def _dome_fact(fit: DomeFit, scale: float) -> str:
    r = fit.sphere.radius_m * scale
    return (
        f"dome: radius {r:.3f} m, rms {fit.sphere.rms_residual_m * scale:.3f} m, "
        f"upper_fraction {fit.upper_fraction:.3f}, confidence {fit.confidence:.3f}"
    )


def _arch_fact(fit: ArchFit, scale: float) -> str:
    return (
        f"arch: span {fit.span_m * scale:.3f} m, rise {fit.rise_m * scale:.3f} m, "
        f"angular_span {fit.angular_span_rad:.3f} rad, confidence {fit.confidence:.3f}"
    )


def _tower_fact(fit: TowerFit, scale: float) -> str:
    h = cylinder_height(cylinder_fit=fit.cylinder) * scale
    r = fit.cylinder.radius_m * scale
    return (
        f"tower: height {h:.3f} m, radius {r:.3f} m, "
        f"axis_up_dot {fit.axis_up_dot:.3f}, confidence {fit.confidence:.3f}"
    )


# ==================================================================
# Ground-truth error report (LAND-01 acceptance criterion support)
# ==================================================================


@dataclass(frozen=True)
class LandmarkErrorReport:
    """Measured error of a reconstructed landmark against ground truth.

    The LAND-01 acceptance criterion: "the reconstructed model is measured
    against a ground-truth model ... with a stated error metric and
    threshold; looking plausible by eye does not count." This is the
    evidence structure for that measurement. With no ground truth supplied
    all fields report "not measured" honestly.
    """

    kind: str
    radius_error_m: Optional[float] = None
    height_error_m: Optional[float] = None
    span_error_m: Optional[float] = None
    position_rms_m: Optional[float] = None
    threshold_m: Optional[float] = None
    passes_threshold: Optional[bool] = None
    measured_fact: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "radius_error_m": self.radius_error_m,
            "height_error_m": self.height_error_m,
            "span_error_m": self.span_error_m,
            "position_rms_m": self.position_rms_m,
            "threshold_m": self.threshold_m,
            "passes_threshold": self.passes_threshold,
            "measured_fact": self.measured_fact,
        }


def report_error(
    kind: str,
    *,
    radius_error_m: Optional[float] = None,
    height_error_m: Optional[float] = None,
    span_error_m: Optional[float] = None,
    position_rms_m: Optional[float] = None,
    threshold_m: Optional[float] = None,
) -> LandmarkErrorReport:
    """Build an error report. Any None dimension is honestly "not measured"."""
    passes = None
    if threshold_m is not None and position_rms_m is not None:
        passes = position_rms_m <= threshold_m
    parts = [f"{kind} landmark"]
    if position_rms_m is not None:
        parts.append(f"position RMS {position_rms_m:.4g} m")
    if threshold_m is not None:
        parts.append(f"threshold {threshold_m:.4g} m")
    if passes is not None:
        parts.append(f"passes={passes}")
    return LandmarkErrorReport(
        kind=kind,
        radius_error_m=radius_error_m,
        height_error_m=height_error_m,
        span_error_m=span_error_m,
        position_rms_m=position_rms_m,
        threshold_m=threshold_m,
        passes_threshold=passes,
        measured_fact="; ".join(parts),
    )