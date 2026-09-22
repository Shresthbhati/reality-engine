"""Measured column detection (P1 perception breadth; PR #86
stair-detector pattern).

The registry declared `column` (registry.py Phase 1, gates on
min_axis_up_dot=0.9 at observation time) but nothing produced
class-level column EVIDENCE: `parametric.fit_cylinder` is a generic
cylinder fit that accepts any cylindrical shell -- a tilted pipe, a
horizontal silo, a flagpole on the ground. Column-ness is a measured
fact (a VERTICAL structural member of plausible proportions), so the
detector below measures it and REFUSES (ColumnRefused) when the
points do not demonstrate it -- a refusal is a fact, never
best-effort geometry.

Method (deterministic, no RNG):
  1. Geometry: parametric.fit_cylinder (least-squares cylinder with
     shell/elongation/sanity refusals) -- reused unchanged, no
     duplicated solver.
  2. Class gates, each a MEASURED refusal naming the fact:
     - verticality: |axis . up| >= MIN_AXIS_UP_DOT (the same threshold
       the registry declares; a 45-degree pipe is a real cylinder and
       NOT a column)
     - axial extent: height >= MIN_HEIGHT_M (a pipe cross-section
       ring or a manhole lid fits a cylinder but is not a column)
  3. Confidence: the fit's own residual-derived confidence, scaled by
     the measured verticality -- a barely-vertical member is a less
     certain column. Documented shape: cylinder.confidence *
     up_dot clamped to [0.05, 1.0].

Reproduction (deterministic): see tests/test_column_beam_detectors.py
-- vertical-shell fixtures give known answers; tilted/silo/flat/blob
fixtures refuse with the measured fact in the message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from perception.architecture.parametric import (
    CylinderFit,
    FitRefused,
    fit_cylinder,
)

Point3 = Tuple[float, float, float]

class ColumnRefused(FitRefused):
    """The points do not demonstrate a column. Carries the measured
    fact in the message; catching FitRefused still works for callers
    that treat all parametric refusals alike."""


#: The measured verticality a structural column must demonstrate
#: (mirrors registry.ArchClass("column").min_axis_up_dot).
MIN_AXIS_UP_DOT = 0.9

#: Minimum measured axial extent (m). A cylinder fit on a ring or a
#: lid has sub-meter extent; a real column (even a broken shaft) is
#: taller. Deliberately wide, not scene-tuned.
MIN_HEIGHT_M = 0.5


@dataclass(frozen=True)
class ColumnFit:
    """A measured column: the generic cylinder fit plus the class's
    own measured facts. `cylinder` carries the full geometry (and its
    measured residuals); nothing here re-guesses what the fit already
    measured."""

    cylinder: CylinderFit
    height_m: float
    n_points: int
    confidence: float
    #: The measured verticality this classification rests on.
    axis_up_dot: float
    min_axis_up_dot: float = MIN_AXIS_UP_DOT
    kind: str = "column"

    @property
    def radius_m(self) -> float:
        return self.cylinder.radius_m

    @property
    def axis(self) -> Tuple[float, float, float]:
        return self.cylinder.axis

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "radius_m": self.radius_m,
            "axis": list(self.axis),
            "axis_point": list(self.cylinder.axis_point),
            "height_m": self.height_m,
            "axis_up_dot": self.axis_up_dot,
            "rms_residual_m": self.cylinder.rms_residual_m,
            "max_residual_m": self.cylinder.max_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def detect_column(
    points: Sequence[Point3], up: Point3 = (0.0, 0.0, 1.0)
) -> ColumnFit:
    """Detect a column in real points or refuse.

    Raises ColumnRefused (a FitRefused subclass) when the measured
    structure does not demonstrate a vertical structural member.
    Deterministic.
    """
    try:
        cyl = fit_cylinder(list(points), up=up)
    except FitRefused as exc:
        raise ColumnRefused(
            f"points do not constrain a column: {exc}"
        ) from exc

    axis_up_dot = abs(cyl.axis[2])  # fit_cylinder canonicalizes to +up
    if axis_up_dot < MIN_AXIS_UP_DOT:
        raise ColumnRefused(
            f"fitted axis is not vertical enough for a column: measured "
            f"up-dot {axis_up_dot:.3f} < {MIN_AXIS_UP_DOT} -- a tilted "
            f"cylinder is a real cylinder but not a column"
        )

    height = cyl.height_max_m - cyl.height_min_m
    if height < MIN_HEIGHT_M:
        raise ColumnRefused(
            f"measured axial extent {height:.3f} m < {MIN_HEIGHT_M} m -- "
            f"a ring or lid fits a cylinder but is not a column"
        )

    # Documented confidence: the geometry fit's quality, discounted by
    # the measured verticality (a barely-vertical member is a less
    # certain column). Floored, never zero, never fabricated above 1.
    confidence = max(0.05, min(1.0, cyl.confidence * axis_up_dot))

    return ColumnFit(
        cylinder=cyl,
        height_m=height,
        n_points=cyl.n_points,
        confidence=confidence,
        axis_up_dot=axis_up_dot,
    )
