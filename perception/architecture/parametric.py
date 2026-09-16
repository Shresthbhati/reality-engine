"""Parametric architectural fitting (P7-03 expansion; directive
sections 12-14): cylinder/sphere/circle least-squares fits over REAL
segment points, with MEASURED residuals and HONEST refusals.

Position in the perception stack (directive section 3):

    low-level perception  ->  geometric perception
        (planes: perception/geometry/planes.py)
        (segments: perception/architecture/segments.py)
    ->  PARAMETRIC FITTING  (this module)
    ->  semantic perception (classify.py's plane roles)
    ->  architectural perception (components.py: hypothesis + entity
        resolution + pattern priors)

Rules (constitution: unknown stays unknown, never fabricate):
  - Fits consume real 3D points only. Residuals are measured
    (rms/max), never declared.
  - Insufficient or degenerate support raises FitRefused -- a fit that
    cannot be trusted must not exist downstream. A garbage giant-
    radius cylinder would silently "classify" as a column forever.
  - Axis/normal sign is canonicalized (positive dot with the supplied
    world up) so two fits of one physical column can never disagree
    by sign.
  - Confidence is a documented, deterministic function of the fit's
    measured quality (geometric deviation vs. a tolerance scale), not
    a made-up score.
  - Pure float math, no RNG, no clocks: the same points always give
    byte-identical fit values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]

#: A fit needs enough points to over-determine its parameters; below
#: this, refusal is honest ("we cannot fit"), not pessimism.
MIN_CYLINDER_POINTS = 8
MIN_SPHERE_POINTS = 6
MIN_CIRCLE_POINTS = 5

#: Radius sanity bounds (meters). A fit outside these is treated as
#: unconstrained/degenerate, not clamped into plausibility.
MIN_SANE_RADIUS_M = 0.01
MAX_SANE_RADIUS_M = 50.0

#: Below this normalized eigenvalue the perpendicular plane is
#: degenerate (points nearly collinear/coplanar) -- direction is not
#: measurable.
_EIGENVALUE_FLOOR = 1e-10

#: Confidence mapping: rms residual / tolerance scale. 0.1 m residual
#: on an architectural fit (columns ~0.3 m radius) is half-trusted;
#: millimeter fits are near-certain. Documented, deterministic.
_CONFIDENCE_TOLERANCE_M = 0.1

#: A real column segment is ELONGATED along its axis: the measured
#: axial extent must be at least this multiple of the fitted radius.
#: Without this, a flat patch of wall can "fit" a tiny cylinder
#: (small rms, small radius, zero extent) -- a geometrically valid
#: nonsense that must be refused, not classified.
MIN_EXTENT_RADIUS_RATIO = 1.0

#: The winning projected circle's rms must stay below this fraction of
#: the fitted radius -- the points must lie NEAR the cylinder surface,
#: not merely be splittable by some cylinder. Flat patches and solid
#: blobs fail this; real column shells pass.
SHELL_RMS_RADIUS_FRACTION = 0.25


class FitRefused(Exception):
    """The points cannot support this fit. Raised, never best-effort."""


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> Vec3:
    n = math.sqrt(_dot(v, v))
    if n < 1e-12:
        raise FitRefused("cannot normalize a near-zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _centroid(points: Sequence[Point3]) -> Point3:
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _covariance(points: Sequence[Point3]) -> Tuple[List[List[float]], Point3]:
    """3x3 covariance about the centroid (pure Python, deterministic)."""
    c = _centroid(points)
    cov = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
    n = len(points)
    for i in range(3):
        for j in range(3):
            cov[i][j] /= n
    return cov, c


def _eigen_symmetric_3x3(m: List[List[float]]) -> List[Tuple[float, Vec3]]:
    """Eigen-decomposition of a symmetric 3x3 matrix via cyclic Jacobi
    rotations -- robust for ALL symmetric inputs including repeated
    eigenvalues (the analytic cubic it replaces returned negative or
    zero eigenvalues for near-degenerate shells, e.g. a 360-degree
    column where Var(x) == Var(y), which silently zeroed downstream
    plane-flatness checks).

    Returns [(eigenvalue, eigenvector), ...] sorted ascending.
    Eigenvectors are orthogonal by construction (rotation products).
    """
    a = [row[:] for row in m]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _sweep in range(24):
        # Off-diagonal magnitude; converged when negligible vs diagonal.
        off = math.sqrt(a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2)
        diag = abs(a[0][0]) + abs(a[1][1]) + abs(a[2][2])
        if off <= 1e-14 * max(diag, 1e-30):
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-18:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = (
                math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                if abs(theta) < 1e18
                else 1.0 / (2.0 * theta)
            )
            c = 1.0 / math.sqrt(t * t + 1.0)
            s_ = t * c
            # Rotate rows/cols p,q of A, and columns of V.
            for k in range(3):
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s_ * akq
                a[k][q] = s_ * akp + c * akq
            for k in range(3):
                apk = a[p][k]
                aqk = a[q][k]
                a[p][k] = c * apk - s_ * aqk
                a[q][k] = s_ * apk + c * aqk
            for k in range(3):
                vkp = v[k][p]
                vkq = v[k][q]
                v[k][p] = c * vkp - s_ * vkq
                v[k][q] = s_ * vkp + c * vkq

    out = []
    for i in range(3):
        evec = (v[0][i], v[1][i], v[2][i])
        n = math.sqrt(evec[0] ** 2 + evec[1] ** 2 + evec[2] ** 2)
        if n < 1e-300:
            evec = (1.0, 0.0, 0.0)
            n = 1.0
        out.append((a[i][i], (evec[0] / n, evec[1] / n, evec[2] / n)))
    out.sort(key=lambda ev: ev[0])
    return out


def _dist2_point_line(p: Point3, a: Point3, u: Vec3) -> float:
    """Squared distance from point p to the line through a with unit
    direction u."""
    d = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    proj = _dot(d, u)
    perp = (d[0] - proj * u[0], d[1] - proj * u[1], d[2] - proj * u[2])
    return _dot(perp, perp)


def _confidence_from_residual(rms_m: float, scale_m: float = _CONFIDENCE_TOLERANCE_M) -> float:
    """Deterministic monotone map: perfect fit -> 1.0, residual at the
    tolerance scale -> 0.5, large residual -> ~0. Documented shape:
    1 / (1 + (rms/scale)^2)."""
    x = rms_m / scale_m
    return 1.0 / (1.0 + x * x)


def _measure_residuals(points: Sequence[Point3], signed: Sequence[float]) -> Tuple[float, float]:
    n = len(points)
    rms = math.sqrt(sum(s * s for s in signed) / n)
    mx = max(abs(s) for s in signed)
    return rms, mx


# ==================================================================
# Cylinder (column shaft)
# ==================================================================


@dataclass(frozen=True)
class CylinderFit:
    """A fitted infinite cylinder. `axis` is canonicalized to point in
    +up hemisphere; `height_min/max_m` are measured axial extents of
    the points (projections), not extrapolations."""

    axis: Vec3
    axis_point: Point3
    radius_m: float
    height_min_m: float
    height_max_m: float
    rms_residual_m: float
    max_residual_m: float
    n_points: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "axis": list(self.axis),
            "axis_point": list(self.axis_point),
            "radius_m": self.radius_m,
            "height_min_m": self.height_min_m,
            "height_max_m": self.height_max_m,
            "rms_residual_m": self.rms_residual_m,
            "max_residual_m": self.max_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def fit_cylinder(points: Sequence[Point3], up: Vec3) -> CylinderFit:
    """Least-squares cylinder over real points.

    Method (deterministic, no RNG):
      For a FIXED axis direction, the best-fit cylinder is exactly the
      best-fit CIRCLE of the points projected onto the plane
      perpendicular to that axis (closed-form Kasa solve). The axis
      direction itself is optimized by deterministic coordinate
      descent (tilt about two orthonormal directions, golden-section
      line searches on the exact inner objective). Seeds: covariance
      eigenvectors + a handful of centroid->point directions; the
      best final RMS wins.

      The final reported rms/max residuals are the true 3D
      perpendicular-distance residuals of the winning cylinder.
    """
    pts = list(points)
    n = len(pts)
    if n < MIN_CYLINDER_POINTS:
        raise FitRefused(
            f"cylinder fit needs at least {MIN_CYLINDER_POINTS} points, got {n}"
        )

    up = _norm(up)
    cov, centroid = _covariance(pts)
    eigs = _eigen_symmetric_3x3(cov)

    seeds: List[Vec3] = [eigs[i][1] for i in range(3)]
    for p in pts[:8]:  # deterministic: first 8 in input order
        d = (p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2])
        if _dot(d, d) > 1e-18:
            seeds.append(_norm(d))
    uniq: List[Vec3] = []
    for s in seeds:
        if all(abs(_dot(s, u)) < 0.9999 for u in uniq):
            uniq.append(s)

    best: Optional[Tuple[float, Vec3, Point3, float]] = None
    for seed in uniq:
        try:
            axis, apoint, radius, rms = _optimize_axis(pts, seed)
        except FitRefused:
            continue
        if best is None or rms < best[0]:
            best = (rms, axis, apoint, radius)

    if best is None:
        raise FitRefused("no cylinder seed produced a convergent fit")

    rms, axis, apoint, radius = best
    if not (MIN_SANE_RADIUS_M <= radius <= MAX_SANE_RADIUS_M):
        raise FitRefused(
            f"fitted radius {radius:.4g} m is outside plausible bounds "
            f"[{MIN_SANE_RADIUS_M}, {MAX_SANE_RADIUS_M}] -- the points do "
            f"not constrain a real cylinder"
        )

    # A cylinder is a SURFACE: points must lie near its shell. The
    # winning circle's rms is the in-plane shell deviation; requiring
    # it to be a small fraction of the radius is the geometric test a
    # flat patch (or any non-shell blob) fails while a real column
    # passes. A solid block fitting "through itself" is not a column.
    if rms > SHELL_RMS_RADIUS_FRACTION * radius:
        raise FitRefused(
            f"points do not lie near a cylindrical shell (in-plane rms "
            f"{rms:.4g} m > {SHELL_RMS_RADIUS_FRACTION}x radius "
            f"{radius:.4g} m) -- no cylinder is constrained by this set"
        )

    # Canonical sign: axis points into the +up hemisphere.
    if _dot(axis, up) < 0:
        axis = (-axis[0], -axis[1], -axis[2])

    ts = [
        (p[0] - apoint[0]) * axis[0]
        + (p[1] - apoint[1]) * axis[1]
        + (p[2] - apoint[2]) * axis[2]
        for p in pts
    ]
    if (max(ts) - min(ts)) < MIN_EXTENT_RADIUS_RATIO * radius:
        raise FitRefused(
            f"points are not elongated along the fitted axis (extent "
            f"{max(ts) - min(ts):.4g} m < {MIN_EXTENT_RADIUS_RATIO}x radius "
            f"{radius:.4g} m) -- a flat patch does not constrain a column; "
            f"refusing rather than classifying geometric nonsense"
        )

    signed = [
        math.sqrt(_dist2_point_line(p, apoint, axis)) - radius for p in pts
    ]
    rms_final, max_final = _measure_residuals(pts, signed)
    return CylinderFit(
        axis=axis,
        axis_point=apoint,
        radius_m=radius,
        height_min_m=min(ts),
        height_max_m=max(ts),
        rms_residual_m=rms_final,
        max_residual_m=max_final,
        n_points=n,
        confidence=_confidence_from_residual(rms_final),
    )


def _projected_circle_rms(
    pts: List[Point3], axis: Vec3
) -> Tuple[float, Point3, float]:
    """For a fixed axis: project points onto the perpendicular plane
    through the centroid, fit a circle there (closed form), and return
    (rms of in-plane radial residuals, circle center in 3D, radius).
    The projected circle IS the optimal fixed-axis cylinder (its
    radius/center minimize exactly the radial residuals)."""
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    u1 = _norm(_cross(axis, helper))
    u2 = _cross(axis, u1)
    centroid = _centroid(pts)
    rel = [(p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]) for p in pts]
    flat = [(_dot(d, u1), _dot(d, u2)) for d in rel]

    # Kasa 2D circle fit.
    rows = [[2.0 * a, 2.0 * b, 1.0] for a, b in flat]
    rhs = [a * a + b * b for a, b in flat]
    A = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    bv = [sum(r[i] * v for r, v in zip(rows, rhs)) for i in range(3)]
    D, E, F = _solve_linear(A, bv)
    r2 = F + D * D + E * E
    if r2 <= 0:
        raise FitRefused("projected points are collinear (no circle)")
    radius = math.sqrt(r2)
    c2d = (D, E)
    center3 = (
        centroid[0] + c2d[0] * u1[0] + c2d[1] * u2[0],
        centroid[1] + c2d[0] * u1[1] + c2d[1] * u2[1],
        centroid[2] + c2d[0] * u1[2] + c2d[1] * u2[2],
    )
    rad = [math.sqrt((a - c2d[0]) ** 2 + (b - c2d[1]) ** 2) - radius for a, b in flat]
    rms = math.sqrt(sum(r * r for r in rad) / len(rad))
    return rms, center3, radius


def _optimize_axis(
    pts: List[Point3], seed: Vec3
) -> Tuple[Vec3, Point3, float, float]:
    """Coordinate descent on the axis direction: alternate tilts about
    two orthonormal directions with golden-section line searches on the
    exact projected-circle objective. Deterministic; a local method by
    design (seeds provide global coverage)."""
    axis = _norm(seed)
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    u1 = _norm(_cross(axis, helper))
    u2 = _cross(axis, u1)

    def _tilt(base: Vec3, dir1: Vec3, dir2: Vec3, t1: float, t2: float) -> Vec3:
        cand = (
            base[0] + t1 * dir1[0] + t2 * dir2[0],
            base[1] + t1 * dir1[1] + t2 * dir2[1],
            base[2] + t1 * dir1[2] + t2 * dir2[2],
        )
        return _norm(cand)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0

    def _golden(f, lo, hi, iters=40):
        # Maximize -f over [lo, hi] (i.e. minimize f).
        a, b = lo, hi
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = f(d)
        return (a + b) / 2.0

    cur_rms, cur_center, cur_radius = _projected_circle_rms(pts, axis)

    # Escalating search spans: pi/8, pi/32, pi/128 radians.
    for span in (math.pi / 8, math.pi / 32, math.pi / 128):
        changed = False
        for _round in range(3):
            for direction in (u1, u2):
                def f(t, d=direction):
                    tilted = _tilt(axis, d, u2 if d is u1 else u1, t, 0.0)
                    try:
                        r, _, _ = _projected_circle_rms(pts, tilted)
                    except FitRefused:
                        return float("inf")
                    return r
                t_opt = _golden(f, -span, span)
                tilted = _tilt(axis, direction, u2 if d is u1 else u1, t_opt, 0.0) if False else None
                # (kept simple: evaluate candidate directly)
                cand = _tilt(axis, direction, u2 if direction is u1 else u1, t_opt, 0.0)
                try:
                    r, c3, rad = _projected_circle_rms(pts, cand)
                except FitRefused:
                    continue
                if r < cur_rms - 1e-15:
                    axis, cur_rms, cur_center, cur_radius = cand, r, c3, rad
                    changed = True
        if not changed:
            break

    return axis, cur_center, cur_radius, cur_rms


def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting; raises FitRefused on
    singular systems (an under-constrained fit must refuse, not guess)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            raise FitRefused("singular normal equations -- fit under-constrained")
        M[col], M[piv] = M[piv], M[col]
        inv = 1.0 / M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] * inv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


# ==================================================================
# Sphere (dome)
# ==================================================================


@dataclass(frozen=True)
class SphereFit:
    center: Point3
    radius_m: float
    rms_residual_m: float
    max_residual_m: float
    n_points: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius_m": self.radius_m,
            "rms_residual_m": self.rms_residual_m,
            "max_residual_m": self.max_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def fit_sphere(points: Sequence[Point3]) -> SphereFit:
    """Least-squares sphere (algebraic Kasa fit + one Newton polish on
    the geometric residual). Refuses coplanar/degenerate support."""
    pts = list(points)
    n = len(pts)
    if n < MIN_SPHERE_POINTS:
        raise FitRefused(f"sphere fit needs at least {MIN_SPHERE_POINTS} points, got {n}")

    # Kasa: minimize sum (x^2+y^2+z^2 + D x + E y + F z + G)^2
    # Linear least squares in (D, E, F, G).
    rows = []
    rhs = []
    for x, y, z in pts:
        rows.append([x, y, z, 1.0])
        rhs.append(-(x * x + y * y + z * z))
    A = [[sum(r[i] * r[j] for r in rows) for j in range(4)] for i in range(4)]
    bv = [sum(r[i] * v for r, v in zip(rows, rhs)) for i in range(4)]
    try:
        D, E, F, G = _solve_linear(A, bv)
    except FitRefused as exc:
        raise FitRefused(f"sphere fit under-constrained (coplanar points?): {exc}") from exc

    center = (-D / 2.0, -E / 2.0, -F / 2.0)
    r2 = _dot(center, center) - G
    if r2 <= 0:
        raise FitRefused("algebraic fit produced non-positive radius squared")
    radius = math.sqrt(r2)

    # Geometric polish: 2 Newton steps on the true residual.
    for _ in range(2):
        num = [0.0, 0.0, 0.0]
        den = 0.0
        for x, y, z in pts:
            d = (x - center[0], y - center[1], z - center[2])
            dist = math.sqrt(_dot(d, d))
            if dist < 1e-12:
                raise FitRefused("point at sphere center -- fit degenerate")
            w = 1.0 / max(dist, 1e-9)
            for i in range(3):
                num[i] += w * (dist - radius) * d[i] / dist
            den += w
        if den > 0:
            center = (center[0] + num[0] / den, center[1] + num[1] / den, center[2] + num[2] / den)

    radius = sum(math.sqrt(_dot((p[0] - center[0], p[1] - center[1], p[2] - center[2]),
                                (p[0] - center[0], p[1] - center[1], p[2] - center[2])))
                 for p in pts) / n
    signed = [
        math.sqrt(_dot((p[0] - center[0], p[1] - center[1], p[2] - center[2]),
                       (p[0] - center[0], p[1] - center[1], p[2] - center[2])))
        - radius
        for p in pts
    ]
    rms, mx = _measure_residuals(pts, signed)
    if not (MIN_SANE_RADIUS_M <= radius <= MAX_SANE_RADIUS_M):
        raise FitRefused(
            f"fitted radius {radius:.4g} m outside plausible bounds"
        )
    return SphereFit(
        center=center,
        radius_m=radius,
        rms_residual_m=rms,
        max_residual_m=mx,
        n_points=n,
        confidence=_confidence_from_residual(rms),
    )


# ==================================================================
# Circle (arch cross-section)
# ==================================================================


@dataclass(frozen=True)
class CircleFit:
    center: Point3
    radius_m: float
    plane_normal: Vec3
    #: Measured angular coverage of the observed arc (0..2pi]. An arch
    #: (~pi) vs a ring (2pi) is a measured fact, not assumed.
    angular_span_rad: float
    #: Unit direction (in the circle's plane) pointing at the middle of
    #: the largest angular gap -- the opening for an arch.
    opening_direction: Optional[Vec3]
    extrusion_depth_m: float
    rms_residual_m: float
    max_residual_m: float
    n_points: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius_m": self.radius_m,
            "plane_normal": list(self.plane_normal),
            "angular_span_rad": self.angular_span_rad,
            "opening_direction": list(self.opening_direction) if self.opening_direction else None,
            "extrusion_depth_m": self.extrusion_depth_m,
            "rms_residual_m": self.rms_residual_m,
            "max_residual_m": self.max_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def fit_circle(
    points: Sequence[Point3], normal: Optional[Vec3]
) -> CircleFit:
    """Fit a circle in 3D to real points. `normal`, when supplied,
    defines the circle plane; when None it is derived (smallest
    covariance eigenvector). Refuses collinear/near-degenerate sets.
    """
    pts = list(points)
    n = len(pts)
    if n < MIN_CIRCLE_POINTS:
        raise FitRefused(f"circle fit needs at least {MIN_CIRCLE_POINTS} points, got {n}")

    cov, centroid = _covariance(pts)
    eigs = _eigen_symmetric_3x3(cov)
    if normal is None:
        nvec = eigs[0][1]
    else:
        nvec = _norm(normal)
    # Plane spanned by u1, u2; project points.
    helper = (1.0, 0.0, 0.0) if abs(nvec[0]) < 0.9 else (0.0, 1.0, 0.0)
    u1 = _norm(_cross(nvec, helper))
    u2 = _cross(nvec, u1)

    flat = [(_dot(p, u1), _dot(p, u2)) for p in pts]

    # Kasa circle fit in 2D.
    rows = [[2.0 * a, 2.0 * b, 1.0] for a, b in flat]
    rhs = [a * a + b * b for a, b in flat]
    A = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    bv = [sum(r[i] * v for r, v in zip(rows, rhs)) for i in range(3)]
    try:
        D, E, F = _solve_linear(A, bv)
    except FitRefused as exc:
        raise FitRefused(f"circle fit under-constrained (collinear points?): {exc}") from exc

    c2d = (D, E)
    r2 = F + D * D + E * E
    if r2 <= 0:
        raise FitRefused("algebraic circle fit produced non-positive radius squared")
    radius = math.sqrt(r2)

    center3 = (
        c2d[0] * u1[0] + c2d[1] * u2[0],
        c2d[0] * u1[1] + c2d[1] * u2[1],
        c2d[0] * u1[2] + c2d[1] * u2[2],
    )

    # Measured angular span + largest-gap opening direction.
    angs = sorted(math.atan2(b - c2d[1], a - c2d[0]) for a, b in flat)
    gaps = []
    for i in range(len(angs)):
        nxt = angs[(i + 1) % len(angs)]
        gap = (nxt - angs[i]) % (2.0 * math.pi)
        gaps.append((gap, angs[i]))
    gaps.sort(reverse=True)
    largest_gap, gap_start = gaps[0]
    span = 2.0 * math.pi - largest_gap
    if len(angs) == 1:
        span = 0.0
        opening = None
    else:
        mid = gap_start + largest_gap / 2.0
        opening = (
            math.cos(mid) * u1[0] + math.sin(mid) * u2[0],
            math.cos(mid) * u1[1] + math.sin(mid) * u2[1],
            math.cos(mid) * u1[2] + math.sin(mid) * u2[2],
        )

    # Extrusion: axial extent of points about the fitted center.
    axial = [_dot((p[0] - center3[0], p[1] - center3[1], p[2] - center3[2]), nvec) for p in pts]
    extrusion = max(axial) - min(axial)

    # Residuals: 3D distance to the circle (radial + axial components).
    signed = []
    for p, (a, b) in zip(pts, flat):
        rad = math.sqrt((a - c2d[0]) ** 2 + (b - c2d[1]) ** 2) - radius
        signed.append(rad)
    # Include the axial (off-plane) component honestly: the 3D
    # residual to the curve is sqrt(radial^2 + axial^2).
    signed3d = [
        math.sqrt(rad * rad + ax * ax) for rad, ax in zip(signed, axial)
    ]
    rms, mx = _measure_residuals(pts, signed3d)
    if not (MIN_SANE_RADIUS_M <= radius <= MAX_SANE_RADIUS_M):
        raise FitRefused(f"fitted radius {radius:.4g} m outside plausible bounds")

    return CircleFit(
        center=center3,
        radius_m=radius,
        plane_normal=nvec,
        angular_span_rad=span,
        opening_direction=opening,
        extrusion_depth_m=extrusion,
        rms_residual_m=rms,
        max_residual_m=mx,
        n_points=n,
        confidence=_confidence_from_residual(rms),
    )
