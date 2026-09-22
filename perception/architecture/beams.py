"""Measured beam detection (P1 perception breadth; PR #86
stair-detector pattern).

The registry had no beam-producing fit at all (`cornice` maps to the
WorldIR beam TYPE via a plane fit -- a cornice is a wall-projection,
not a measured linear member). A beam is a LINEAR PRISMATIC member:
elongated along ONE horizontal axis, compact in the other two, with
a shell-like (face-sampled) cross-section. The detector below
measures exactly that structure and refuses (BeamRefused) when the
points do not demonstrate it -- a refusal is a fact, never
best-effort geometry.

Method (deterministic, no RNG):
  1. PCA of the supporting points (parametric's closed-form Jacobi
     solver -- reused, no duplicated eigensolver): e1 = long axis,
     e2/e3 = cross-section axes.
  2. Measured gates, each a refusal naming the fact:
     - single-axis elongation: sqrt(l1/l2) >= MIN_ELONGATION_RATIO
       (a blob is elongated in no direction; a wall panel in two)
     - horizontality: |e1 . up| <= MAX_AXIS_UP_DOT (a vertical post
       may be many things; a beam is not one; a 45-degree ramp is
       linear but not a horizontal structural member)
     - length: extent along e1 >= MIN_LENGTH_M
     - cross-section: BOTH lateral extents >= MIN_CROSS_SECTION_M
       (a planar patch has a degenerate lateral direction)
     - shell-like cross-section: on BOTH lateral axes the mean
       absolute projection must exceed FACE_SHELL_MIN_RATIO x half
       extent -- points hug the faces of a prism; face samples of a
       panel fill its extent (mean |proj| ~ half/2) and fail
  3. Residual: rms distance of points to the measured box shell
     (lateral only; the long ends are segment boundaries, not faces).
  4. Confidence: documented monotone function of the measured
     residual relative to the smaller cross-section half-extent,
     clamped to [0.05, 1.0] -- a fit is never more certain than its
     own consistency.

All extents are min/max over the member points (the same convention
as the stair fit's span; outlier robustness is segmentation's job,
upstream). Reproduction (deterministic):
tests/test_column_beam_detectors.py -- box fixtures give known
answers; post/blob/wall/ramp fixtures refuse with the measured fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from perception.architecture.parametric import (
    FitRefused,
    _covariance,
    _eigen_symmetric_3x3,
)

Point3 = Tuple[float, float, float]

#: Minimum point support to measure a prism at all.
MIN_POINTS = 16

#: The long axis must dominate the first cross-section axis by this
#: ratio of standard deviations (sqrt of the eigenvalue ratio).
MIN_ELONGATION_RATIO = 4.0

#: A beam is horizontal: the long axis's |up-dot| must not exceed
#: this (0.35 admits drainage slopes; a 45-degree ramp at 0.707
#: refuses).
MAX_AXIS_UP_DOT = 0.35

#: Minimum measured length (m).
MIN_LENGTH_M = 1.0

#: Minimum measured extent on EACH cross-section axis (m): below this
#: the cross-section is degenerate (a planar patch, not a prism).
MIN_CROSS_SECTION_M = 0.05

#: Shell test: mean |lateral projection| >= this fraction of the
#: half-extent on BOTH lateral axes (prism faces vs filled panel).
FACE_SHELL_MIN_RATIO = 0.35

#: Minimum length-to-cross-section aspect ratio for a "beam".
MIN_ASPECT_RATIO = 3.0


class BeamRefused(FitRefused):
    """The points do not demonstrate a beam. Carries the measured
    fact in the message; catching FitRefused still works for callers
    that treat all parametric refusals alike."""


@dataclass(frozen=True)
class BeamFit:
    """A measured beam: axis, cross-section, and residual -- every
    quantity measured from the supporting points, nothing guessed."""

    axis: Tuple[float, float, float]
    length_m: float
    height_m: float
    width_m: float
    aspect_ratio: float
    rms_residual_m: float
    n_points: int
    confidence: float
    #: Measured support centroid (the component's position downstream).
    position: Point3 = (0.0, 0.0, 0.0)
    kind: str = "beam"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "axis": list(self.axis),
            "position": list(self.position),
            "length_m": self.length_m,
            "height_m": self.height_m,
            "width_m": self.width_m,
            "aspect_ratio": self.aspect_ratio,
            "rms_residual_m": self.rms_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def _extent(values) -> float:
    return max(values) - min(values)


def detect_beam(
    points: Sequence[Point3], up: Point3 = (0.0, 0.0, 1.0)
) -> BeamFit:
    """Detect a beam in real points or refuse.

    Raises BeamRefused (a FitRefused subclass) when the measured
    structure does not demonstrate a horizontal linear prismatic
    member. Deterministic.
    """
    pts = list(points)
    n = len(pts)
    if n < MIN_POINTS:
        raise BeamRefused(
            f"beam detection needs >= {MIN_POINTS} points, got {n}"
        )

    cov, centroid = _covariance(pts)
    # The shared solver returns ascending eigenvalues; the long axis
    # is the LARGEST, so sort descending explicitly (documented
    # convention here, not an ordering assumption).
    eigs = sorted(_eigen_symmetric_3x3(cov), key=lambda t: t[0], reverse=True)
    (l1, e1), (l2, e2), (l3, e3) = eigs[0], eigs[1], eigs[2]

    # ---- 1. Single-axis elongation ----
    if l2 <= 0.0:
        raise BeamRefused(
            "points are collinear -- no cross-section is measurable; "
            "a line of points does not constrain a beam"
        )
    elongation = math.sqrt(l1 / l2)
    if elongation < MIN_ELONGATION_RATIO:
        raise BeamRefused(
            f"points are not elongated along one axis: measured "
            f"elongation {elongation:.2f} < {MIN_ELONGATION_RATIO} "
            f"(a blob is elongated in no direction, a panel in two)"
        )

    # Canonical sign: long axis into the +up hemisphere-agnostic form
    # (beams are horizontal; sign carries no meaning, fix it anyway
    # for determinism of the reported axis).
    if e1[2] < 0:
        e1 = (-e1[0], -e1[1], -e1[2])

    # ---- 2. Horizontality ----
    axis_up_dot = abs(e1[2])
    if axis_up_dot > MAX_AXIS_UP_DOT:
        raise BeamRefused(
            f"long axis is not horizontal: measured up-dot "
            f"{axis_up_dot:.3f} > {MAX_AXIS_UP_DOT} -- a vertical "
            f"member may be many things; a beam is not one"
        )

    ts = [sum((p[i] - centroid[i]) * e1[i] for i in range(3)) for p in pts]
    length = _extent(ts)
    if length < MIN_LENGTH_M:
        raise BeamRefused(
            f"measured length {length:.3f} m < {MIN_LENGTH_M} m"
        )

    # ---- 3. Cross-section extents (both lateral axes) ----
    us = [sum((p[i] - centroid[i]) * e2[i] for i in range(3)) for p in pts]
    ws = [sum((p[i] - centroid[i]) * e3[i] for i in range(3)) for p in pts]
    height = _extent(us)
    width = _extent(ws)
    if height < MIN_CROSS_SECTION_M or width < MIN_CROSS_SECTION_M:
        raise BeamRefused(
            f"cross-section is degenerate: measured lateral extents "
            f"{height:.3f} m x {width:.3f} m -- a planar patch does "
            f"not constrain a prismatic member"
        )

    # ---- 4. Shell-like cross-section (faces, not a filled panel) ----
    for name, projs, ext in (("first", us, height), ("second", ws, width)):
        half = ext / 2.0
        mean_abs = sum(abs(v) for v in projs) / n
        if mean_abs < FACE_SHELL_MIN_RATIO * half:
            raise BeamRefused(
                f"cross-section is not shell-like on the {name} lateral "
                f"axis: mean |projection| {mean_abs:.4f} m < "
                f"{FACE_SHELL_MIN_RATIO} x half-extent {half:.4f} m -- "
                f"face samples of a panel fill its extent; prism faces "
                f"hug it"
            )

    aspect = length / max(height, width)
    if aspect < MIN_ASPECT_RATIO:
        raise BeamRefused(
            f"measured aspect ratio {aspect:.2f} < "
            f"{MIN_ASPECT_RATIO} -- this is not an elongated member"
        )

    # ---- 5. Residual: rms distance to the measured box shell ----
    h_half, w_half = height / 2.0, width / 2.0
    sq = 0.0
    for u, w in zip(us, ws):
        du = abs(abs(u) - h_half)
        dw = abs(abs(w) - w_half)
        sq += min(du, dw) ** 2
    rms = math.sqrt(sq / n)

    # ---- 6. Confidence: documented function of measured residual ----
    scale = max(min(h_half, w_half), 1e-6)
    confidence = max(0.05, min(1.0, 1.0 - rms / scale))

    return BeamFit(
        axis=e1,
        length_m=length,
        height_m=height,
        width_m=width,
        aspect_ratio=aspect,
        rms_residual_m=rms,
        n_points=n,
        confidence=confidence,
        position=centroid,
    )
