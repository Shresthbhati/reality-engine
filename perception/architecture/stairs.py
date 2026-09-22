"""Measured stair detection (P7-03 architectural-perception expansion).

A staircase's evidence is a RHYTHM of flat, walkable level bands
climbing monotonically: distinct z plateaus, each reached by a
consistent riser height, spaced by a consistent going. The detector
below measures exactly that structure from real points and refuses
when the points do not demonstrate it -- mirroring parametric.py's
FitRefused discipline (a refusal is a fact, never best-effort
geometry).

Method (deterministic, no RNG):
  1. Bin point z-values into a histogram at RISE_BIN_M resolution.
  2. Merge adjacent occupied bins into LEVEL BANDS (plateaus); a band
     must hold MIN_BAND_FRACTION of the max band population to count
     (sparse stray bins -- outliers, railing points -- do not become
     "steps").
  3. Measure the rhythm: rise = consecutive band-height gaps, going =
     consecutive band-centroid planimetric gaps along the principal
     horizontal direction (PCA of xy, deterministic closed form).
  4. Gates: >= MIN_STEPS distinct bands; rise/going consistency
     (relative deviation <= MAX_RHYTHM_DEVIATION); walkable riser
     window; enough points. Any failure raises StairRefused.

The reported fit carries only measured quantities. Confidence is a
documented function of the measured rhythm consistency -- a fit's
quality, not a made-up certainty about the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from perception.architecture.parametric import FitRefused

Point3 = Tuple[float, float, float]

# ---- Measured-structure gates (documented constants, not per-callsite
# magic numbers; residential/urban stair ranges, deliberately wide) ----

#: Histogram bin size for the z histogram (m).
RISE_BIN_M = 0.01

#: A band must hold at least this fraction of the LARGEST band's
#: population to count as a step (sparse bins are outliers, not treads).
MIN_BAND_FRACTION = 0.25

#: Fewer level bands than this cannot demonstrate a rhythm.
MIN_STEPS = 2

#: Consecutive rises/goings must agree within this relative deviation.
MAX_RHYTHM_DEVIATION = 0.35

#: Walkable riser window (m): below ~0.03 is measurement noise between
#: merged bins; above ~0.5 is not a stair a person climbs.
MIN_RISE_M = 0.03
MAX_RISE_M = 0.5

#: A step must advance (going) by at least this (m) -- vertical ladders
#: are not staircases.
MIN_GOING_M = 0.05

#: Absolute minimum point support.
MIN_POINTS = 16


@dataclass(frozen=True)
class StaircaseFit:
    """A measured staircase: the level-band rhythm of real points.

    All quantities are measured from the supporting points, never
    extrapolated: rise_m/going_m are the MEAN consecutive gaps,
    span_m the planimetric extent, n_steps the measured band count
    minus one (transitions).
    """

    n_steps: int
    rise_m: float
    going_m: float
    span_m: float
    rms_residual_m: float
    n_points: int
    confidence: float
    #: Measured support centroid (the component's position downstream).
    position: Point3 = (0.0, 0.0, 0.0)
    kind: str = "stairs"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "position": list(self.position),
            "n_steps": self.n_steps,
            "rise_m": self.rise_m,
            "going_m": self.going_m,
            "span_m": self.span_m,
            "rms_residual_m": self.rms_residual_m,
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def _level_bands(z_values: List[float]) -> List[Tuple[float, float, int]]:
    """Histogram z into merged level bands.

    Returns [(z_center_m, band_height_m, population), ...] ordered by
    z. Only bands meeting MIN_BAND_FRACTION of the largest band's
    population survive.
    """
    z_min = min(z_values)
    bins: dict = {}
    for z in z_values:
        key = math.floor((z - z_min) / RISE_BIN_M)
        bins[key] = bins.get(key, 0) + 1
    if not bins:
        return []

    # Merge adjacent occupied bins into candidate bands.
    occupied = sorted(bins.keys())
    bands: List[List[int]] = [[occupied[0]]]
    for key in occupied[1:]:
        if key == bands[-1][-1] + 1:
            bands[-1].append(key)
        else:
            bands.append([key])

    # Keep population-significant bands (relative to the largest).
    max_pop = max(sum(bins[k] for k in band) for band in bands)
    kept: List[Tuple[float, float, int]] = []
    for band in bands:
        pop = sum(bins[k] for k in band)
        if pop < MIN_BAND_FRACTION * max_pop:
            continue
        z_lo = z_min + band[0] * RISE_BIN_M
        z_hi = z_min + (band[-1] + 1) * RISE_BIN_M
        # Population-weighted z center of the band's own points.
        zc = sum(
            z for z in z_values
            if z_lo <= z < z_hi or (band[-1] == occupied[-1] and z == z_hi)
        ) / pop
        kept.append((zc, z_hi - z_lo, pop))
    return kept


def _ascent_direction(pts: Sequence[Point3]) -> Tuple[float, float]:
    """The horizontal direction along which height INCREASES: the
    normalized (cov(z,x), cov(z,y)) vector. NOT the dominant xy axis --
    a staircase is usually wider than its going, so a PCA direction
    would find the width. A vertically-stacked rhythm (ladder) yields
    a near-zero vector; the caller's going measurement then honestly
    measures ~0 and refuses."""
    n = len(pts)
    mz = sum(p[2] for p in pts) / n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    czx = sum((p[2] - mz) * (p[0] - mx) for p in pts) / n
    czy = sum((p[2] - mz) * (p[1] - my) for p in pts) / n
    norm = math.hypot(czx, czy)
    if norm < 1e-12:
        return 1.0, 0.0
    return czx / norm, czy / norm


def detect_stairs(points: Sequence[Point3]) -> StaircaseFit:
    """Detect a staircase in real points or refuse.

    Raises FitRefused when the measured structure does not
    demonstrate stairs. Deterministic.
    """
    pts = list(points)
    if len(pts) < MIN_POINTS:
        raise FitRefused(
            f"stair detection needs >= {MIN_POINTS} points, got {len(pts)}"
        )

    # ---- 1. Level bands from the z histogram ----
    bands = _level_bands([p[2] for p in pts])
    if len(bands) < MIN_STEPS + 1:
        raise FitRefused(
            f"expected >= {MIN_STEPS + 1} distinct level bands for a "
            f"{MIN_STEPS}-step staircase, measured {len(bands)}"
        )

    # ---- 2. Rhythm measurement ----
    centers = [b[0] for b in bands]
    rises = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

    # Planimetric going along the ASCENT direction (the horizontal
    # direction correlated with height -- see _ascent_direction).
    ux, uy = _ascent_direction(pts)
    proj = sorted(p[0] * ux + p[1] * uy for p in pts)

    # Band-support x-intervals: points near each band's z center.
    proj_centers: List[float] = []
    for zc, _h, _pop in bands:
        near = sorted(
            p[0] * ux + p[1] * uy for p in pts
            if abs(p[2] - zc) <= max(RISE_BIN_M, bands[0][1])
        )
        if near:
            proj_centers.append((near[0] + near[-1]) / 2.0)
    goings = [
        proj_centers[i + 1] - proj_centers[i]
        for i in range(len(proj_centers) - 1)
    ]

    # ---- 3. Gates: each a measured refusal, recorded in the message ----
    mean_rise = sum(rises) / len(rises)
    if mean_rise < MIN_RISE_M:
        raise FitRefused(
            f"mean level gap {mean_rise:.4f} m below the minimum walkable "
            f"riser {MIN_RISE_M} m (flat ground merges into one band)"
        )
    if mean_rise > MAX_RISE_M:
        raise FitRefused(
            f"mean level gap {mean_rise:.3f} m exceeds the maximum "
            f"walkable riser {MAX_RISE_M} m"
        )
    dev = max(
        abs(r - mean_rise) / max(mean_rise, 1e-9) for r in rises
    )
    if dev > MAX_RHYTHM_DEVIATION:
        raise FitRefused(
            f"rise rhythm irregular: max relative deviation {dev:.2f} > "
            f"{MAX_RHYTHM_DEVIATION} (gaps: "
            + ", ".join(f"{r:.3f}" for r in rises) + ")"
        )

    if goings:
        mean_going = sum(goings) / len(goings)
        if mean_going < MIN_GOING_M:
            raise FitRefused(
                f"mean going {mean_going:.4f} m below {MIN_GOING_M} m "
                f"(vertical rhythm without advance is not a staircase)"
            )
        gdev = max(
            abs(g - mean_going) / max(mean_going, 1e-9) for g in goings
        )
        if gdev > MAX_RHYTHM_DEVIATION:
            raise FitRefused(
                f"going rhythm irregular: max relative deviation "
                f"{gdev:.2f} > {MAX_RHYTHM_DEVIATION}"
            )
        going_m = mean_going
    else:
        going_m = 0.0

    # ---- 4. Piecewise-flat residual (measured fit quality) ----
    band_of = {}
    sq = 0.0
    for p in pts:
        bi = min(range(len(centers)), key=lambda i: abs(p[2] - centers[i]))
        sq += (p[2] - centers[bi]) ** 2
        band_of[bi] = band_of.get(bi, 0) + 1
    rms = math.sqrt(sq / len(pts))

    # ---- 5. Confidence: documented function of measured rhythm ----
    # Rhythm consistency (1 - normalized deviation), floored at 0.05:
    # a fit is never more certain than its own consistency.
    rise_dev = dev
    going_dev = (max(abs(g - sum(goings) / len(goings)) /
                     max(sum(goings) / len(goings), 1e-9)
                     for g in goings) if goings else 0.0)
    consistency = 1.0 - max(rise_dev, going_dev)
    confidence = max(0.05, min(1.0, consistency))

    span = proj[-1] - proj[0]
    centroid = (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )
    return StaircaseFit(
        n_steps=len(bands) - 1,
        rise_m=mean_rise,
        going_m=going_m,
        span_m=span,
        rms_residual_m=rms,
        n_points=len(pts),
        confidence=confidence,
        position=centroid,
    )
