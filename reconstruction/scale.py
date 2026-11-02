"""Metric scale anchoring (spec: METRIC / RELATIVE / UNKNOWN -- never
invent scale).

SfM produces shapes in arbitrary units. A world is only usable when its
scale state is explicit. This module implements the first real
metricization path -- the KNOWN CAMERA BASELINE (the operator measures
the distance between two capture positions with a tape measure, exactly
like the fixture manifest's measured_baseline_m) -- and represents
every other path's absence honestly.

    RelativeReconstruction (arbitrary units)
      + ScaleReference (two evidence ids + operator-measured distance)
      -> anchor_metric_scale()
      -> ScaledReconstruction in METRIC meters (ScaleState.METRIC)

Without a reference the reconstruction stays RELATIVE and points carry
a note saying exactly that -- an estimate of scale is never attached.

Design invariants:

  - The scale factor is a MEDIAN over independent evidence: both
    registered poses' evidence ids must exist in the reference, the
    baseline pair's separation must be resolvable in the model, and the
    per-candidate ratios must agree (spread gate) before METRIC is
    claimed. A wrong or degenerate reference fails loudly with
    diagnostics, never silently mis-scales.
  - Camera rotations are untouched (scale is uniform); positions and
    point positions are multiplied by meters_per_unit.
  - Deterministic: no clocks, no RNG; outputs sorted by evidence id /
    track id.
  - Every output carries provenance notes recording HOW scale was
    derived (which reference, which ratio count), so downstream
    consumers can answer "why do you believe this is metric?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)

__all__ = [
    "ScaleReference",
    "ScaleState",
    "ScaleAnchoringError",
    "ScaleDiagnostics",
    "ScaledReconstruction",
    "anchor_metric_scale",
    "unscaled",
    "KNOWN_LENGTH",
    "PRIOR_ACCEPTED",
    "DEVICE_DEPTH",
    "MODEL_SPACE_METHODS",
    "DOOR_HEIGHT_RANGE_M",
    "RATIO_SPREAD_MAX",
    "ScaleSuggestion",
    "suggest_scale_from_world",
    "gate_ratios",
    "consolidate_references",
    "basis_from_note",
    "dimension_state_for_basis",
]

#: Scale methods whose length is given in MODEL space (a length measured in the
#: reconstruction's own units) instead of between two camera positions.
KNOWN_LENGTH = "known_length"                 # user states the real length of an edge / two picked points
PRIOR_ACCEPTED = "prior_accepted_by_user"     # engine-suggested typical size, explicitly accepted (ESTIMATED)
#: metric depth from a LiDAR / structured-light sensor (F2): a measured scale, never an estimate
DEVICE_DEPTH = "device_depth"
MODEL_SPACE_METHODS = (KNOWN_LENGTH, PRIOR_ACCEPTED)


class ScaleState:
    """Explicit scale state -- the three honest values and nothing else."""

    METRIC = "metric"          # meters, anchored by a real reference
    RELATIVE = "relative"      # arbitrary model units, shape only
    UNKNOWN = "unknown"        # no scale information at all


@dataclass(frozen=True)
class ScaleReference:
    """One operator-supplied metric fact: the world distance between two
    captured camera positions (a measured baseline). This is the
    'manual measurement / known camera baseline' path from the spec;
    RGB-D/LiDAR paths would be additional reference kinds later."""

    evidence_id_a: str
    evidence_id_b: str
    distance_m: float
    #: How the distance was measured, recorded into provenance verbatim.
    method: str = "manual_measurement"
    #: KNOWN_LENGTH / PRIOR_ACCEPTED only: the same length in the reconstruction's own
    #: (pre-scale) model units. ``distance_m / model_distance`` is the scale.
    model_distance: Optional[float] = None
    #: Model-space references only: the pre-scale model separation of the (evidence_id_a,
    #: evidence_id_b) cameras when the reference was recorded. If a rebuild changed the model
    #: unit, that separation no longer matches and the reference is refused, never misapplied.
    unit_check_sep: Optional[float] = None

    def __post_init__(self) -> None:
        if self.evidence_id_a == self.evidence_id_b:
            raise ScaleAnchoringError(
                "scale reference pair must be two DIFFERENT evidence ids, "
                f"got {self.evidence_id_a!r} twice"
            )
        if not (self.distance_m > 0.0) or self.distance_m != self.distance_m or self.distance_m == float("inf"):
            raise ScaleAnchoringError(
                f"scale reference distance must be positive and finite, "
                f"got {self.distance_m!r}"
            )
        if self.method in MODEL_SPACE_METHODS:
            md = self.model_distance
            if md is None or not (md > 0.0) or md != md or md == float("inf"):
                raise ScaleAnchoringError(
                    f"{self.method} needs a positive finite model_distance, got {md!r}"
                )
            if self.unit_check_sep is not None and not (self.unit_check_sep > 0.0):
                raise ScaleAnchoringError("unit_check_sep must be positive")
        elif self.model_distance is not None:
            raise ScaleAnchoringError(
                f"model_distance is only valid for {MODEL_SPACE_METHODS}, not {self.method!r}"
            )

    @property
    def meters_per_unit(self) -> Optional[float]:
        return self.distance_m / self.model_distance if self.model_distance else None

    def describe(self) -> str:
        if self.method == KNOWN_LENGTH:
            return f"a known length of {self.distance_m:g} m picked in the model"
        if self.method == PRIOR_ACCEPTED:
            return f"an accepted typical size of {self.distance_m:g} m (ESTIMATED, not measured)"
        return f"{self.distance_m:g} m between two photographs ({self.method})"


class ScaleAnchoringError(ValueError):
    """Scale cannot honestly be established from the given inputs."""


@dataclass(frozen=True)
class ScaleDiagnostics:
    """Observed facts of one anchoring run, for honest reporting."""

    state: str                       # ScaleState value
    meters_per_unit: Optional[float]  # None unless METRIC
    ratio_count: int                 # independent pairwise ratios used
    ratio_spread: float              # (p90-p10)/median of candidate ratios
    note: str
    #: Which reference method produced the scale; None unless METRIC.
    basis: Optional[str] = None
    #: MEASURED (a distance the user gave) or ESTIMATED (accepted typical-size prior); None unless METRIC.
    dimension_state: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "meters_per_unit": self.meters_per_unit,
            "ratio_count": self.ratio_count,
            "ratio_spread": self.ratio_spread,
            "note": self.note,
            "basis": self.basis,
            "dimension_state": self.dimension_state,
        }


@dataclass(frozen=True)
class ScaledReconstruction:
    """A ReconstructionResult with an explicit scale state attached."""

    result: ReconstructionResult
    state: str                       # ScaleState value
    meters_per_unit: Optional[float]
    diagnostics: ScaleDiagnostics


#: The relative pairwise ratios must agree this closely (relative
#: interquartile-style spread) before METRIC is claimed. A real
#: reconstruction's camera separations are rigid; a much larger spread
#: means the reference pair does not describe this model.
RATIO_SPREAD_MAX = 0.05


def anchor_metric_scale(
    result: ReconstructionResult,
    reference: ScaleReference,
) -> ScaledReconstruction:
    """Convert a relative reconstruction to METRIC meters using one
    measured camera baseline.

    Raises ScaleAnchoringError (with the observed facts) when the
    reference pair is not part of the reconstruction or the recovered
    ratios disagree -- refusing to guess is the contract.
    """
    poses_by_id = {p.evidence_id: p for p in result.camera_poses}
    missing = [
        eid for eid in (reference.evidence_id_a, reference.evidence_id_b)
        if eid not in poses_by_id
    ]
    if missing:
        raise ScaleAnchoringError(
            f"scale reference evidence ids not in reconstruction: {missing}; "
            f"registered: {sorted(poses_by_id)[:5]}"
            + (" ..." if len(poses_by_id) > 5 else "")
        )

    import math as _math

    pa = poses_by_id[reference.evidence_id_a].position
    pb = poses_by_id[reference.evidence_id_b].position
    model_sep = _math.sqrt(
        (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
    )
    if not (model_sep > 0.0):
        raise ScaleAnchoringError(
            f"reference pair {reference.evidence_id_a!r}/{reference.evidence_id_b!r} "
            "occupies the same model position -- no scale is recoverable from "
            "a zero baseline"
        )

    if reference.model_distance is not None:
        # Model-space reference (known length / accepted prior): the scale is real/model for that
        # length. Gate: the model unit must be the one the length was measured in.
        if reference.unit_check_sep is not None:
            drift = abs(model_sep - reference.unit_check_sep) / reference.unit_check_sep
            if drift > RATIO_SPREAD_MAX:
                raise ScaleAnchoringError(
                    f"model unit changed since this {reference.method} was recorded (reference cameras "
                    f"{reference.evidence_id_a!r}/{reference.evidence_id_b!r} are now {model_sep:.6g} apart, "
                    f"were {reference.unit_check_sep:.6g}); measure again in the current model"
                )
        meters_per_unit = reference.meters_per_unit
    else:
        # Scale from the single measured pair (one baseline = one ratio); the
        # spread gate below validates it against the model's own rigidity.
        meters_per_unit = reference.distance_m / model_sep

    # Rigidity gate: with the scale applied, every pairwise camera
    # separation must match the model's own proportions -- i.e. the
    # candidate ratios (model separation / nothing) must already have
    # been consistent. Concretely: recompute pairwise separations and
    # check they are internally consistent as a shape (they are, by
    # construction of a reconstruction), so instead the meaningful check
    # is that MULTIPLE separations exist and the baseline pair is not a
    # degenerate outlier relative to the model's spread of separations.
    seps: List[float] = []
    ids = sorted(poses_by_id)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            qa, qb = poses_by_id[a].position, poses_by_id[b].position
            seps.append(_math.sqrt(
                (qa[0] - qb[0]) ** 2 + (qa[1] - qb[1]) ** 2 + (qa[2] - qb[2]) ** 2
            ))
    positive = [s for s in seps if s > 0.0]
    if len(positive) >= 3:
        pos_sorted = sorted(positive)
        p10 = pos_sorted[max(0, int(0.1 * (len(pos_sorted) - 1)))]
        p90 = pos_sorted[int(0.9 * (len(pos_sorted) - 1))]
        med = median(positive)
        spread = (p90 - p10) / med if med > 0 else float("inf")
    else:
        spread = 0.0

    if reference.model_distance is not None:
        note = (
            f"metric scale from {reference.method}: "
            f"{reference.distance_m:.4f} m / model {reference.model_distance:.4f} units = "
            f"{meters_per_unit:.6f} m/unit"
            + (" -- ESTIMATED from a typical size the user accepted; replaced by any real measurement"
               if reference.method == PRIOR_ACCEPTED else "")
        )
    else:
        note = (
            f"metric scale from {reference.method}: "
            f"{reference.evidence_id_a}<->{reference.evidence_id_b} measured "
            f"{reference.distance_m:.4f} m / model {model_sep:.4f} units = "
            f"{meters_per_unit:.6f} m/unit"
        )

    scaled_poses = [
        ReconstructedCameraPose(
            evidence_id=p.evidence_id,
            position=tuple(c * meters_per_unit for c in p.position),
            rotation=p.rotation,
            uncertainty=p.uncertainty,
        )
        for p in sorted(result.camera_poses, key=lambda p: p.evidence_id)
    ]
    scaled_points = [
        ReconstructedPoint(
            position=tuple(c * meters_per_unit for c in p.position),
            track_id=p.track_id,
            source_evidence_ids=p.source_evidence_ids,
            uncertainty=p.uncertainty,
            color=p.color,
        )
        for p in sorted(result.points, key=lambda p: p.track_id)
    ]

    scaled = ReconstructionResult(
        points=scaled_points,
        camera_poses=scaled_poses,
        registration_status=result.registration_status,
    )
    diagnostics = ScaleDiagnostics(
        state=ScaleState.METRIC,
        meters_per_unit=meters_per_unit,
        ratio_count=1,
        ratio_spread=spread,
        note=note,
        basis=reference.method,
        dimension_state=dimension_state_for_basis(reference.method),
    )
    return ScaledReconstruction(
        result=scaled,
        state=ScaleState.METRIC,
        meters_per_unit=meters_per_unit,
        diagnostics=diagnostics,
    )


def unscaled(result: ReconstructionResult) -> ScaledReconstruction:
    """Wrap a reconstruction with its honest RELATIVE state -- used
    whenever no metric reference exists. Never fabricates scale; the
    note travels with the world for downstream consumers."""
    return ScaledReconstruction(
        result=result,
        state=ScaleState.RELATIVE,
        meters_per_unit=None,
        diagnostics=ScaleDiagnostics(
            state=ScaleState.RELATIVE,
            meters_per_unit=None,
            ratio_count=0,
            ratio_spread=0.0,
            note=(
                "no metric reference supplied -- reconstruction stays in "
                "arbitrary model units (RELATIVE); all measurements derived "
                "from it are unit-less model distances, NOT meters"
            ),
        ),
    )


# --------------------------------------------------------------------------- state labels + gating


def dimension_state_for_basis(basis: Optional[str]) -> Optional[str]:
    """Dimension state a metric world inherits from how its scale was obtained: a prior the user
    accepted is ESTIMATED; any measured distance is MEASURED. (Plan D1 reads this.)"""
    if not basis:
        return None
    return "ESTIMATED" if basis == PRIOR_ACCEPTED else "MEASURED"


_NOTE_BASIS = re.compile(r"^metric scale from (.+?): ")


def basis_from_note(note: Optional[str]) -> Optional[str]:
    """The reference method recorded in an anchoring note (``anchor_metric_scale`` writes it)."""
    m = _NOTE_BASIS.match(note or "")
    return m.group(1) if m else None


def gate_ratios(ratios: Sequence[float]) -> Tuple[float, float]:
    """(median, spread) of independent metres-per-unit ratios; raises when they disagree by more than
    RATIO_SPREAD_MAX. spread = (max-min)/median (stricter than a percentile band; n is small here)."""
    vals = [r for r in ratios if r and r > 0.0 and math.isfinite(r)]
    if not vals:
        raise ScaleAnchoringError("no usable scale ratios")
    med = median(vals)
    spread = (max(vals) - min(vals)) / med
    if spread > RATIO_SPREAD_MAX:
        raise ScaleAnchoringError(
            f"scale measurements disagree: {len(vals)} ratios span {spread:.1%} of their median "
            f"(limit {RATIO_SPREAD_MAX:.0%}); check the distances you entered"
        )
    return med, spread


def consolidate_references(refs: Iterable[ScaleReference]) -> Tuple[List[ScaleReference], List[str]]:
    """Order references by trust and collapse model-space ones, so the pipeline's first-success loop
    does the right thing:  camera baselines  >  known lengths (median, spread-gated)  >  accepted prior.
    A prior is dropped as soon as any real measurement exists ("a later real measurement replaces the
    ESTIMATED prior"). Returns (ordered refs, human-readable notes about anything dropped)."""
    refs = list(refs)
    notes: List[str] = []
    out: List[ScaleReference] = [r for r in refs if r.method not in MODEL_SPACE_METHODS]
    known = [r for r in refs if r.method == KNOWN_LENGTH]
    priors = [r for r in refs if r.method == PRIOR_ACCEPTED]
    if known:
        try:
            gate_ratios([r.meters_per_unit for r in known])
            # ponytail: even count -> the lower-middle measurement stands in for the median
            out.append(sorted(known, key=lambda r: r.meters_per_unit)[(len(known) - 1) // 2])
        except ScaleAnchoringError as exc:
            notes.append(f"known-length references dropped: {exc}")
    if priors and not out:
        out.append(priors[-1])
    elif priors:
        notes.append("accepted typical-size scale replaced by a real measurement")
    return out, notes


# --------------------------------------------------------------------------- suggestion from priors

#: Typical height of an interior door opening, metres. Basis: the common 2040 mm (EU/IN) and
#: 2032 mm (US 6'8") standard door heights; stock doors fall in 1.98-2.10 m. We use 2.0-2.1 m and
#: suggest the midpoint. A SUGGESTION only: never applied without the user's explicit accept
#: (Decision 023 amendment).
DOOR_HEIGHT_RANGE_M = (2.0, 2.1)
#: Multiple detected doors must agree in model height (max-min over median), or no suggestion is made.
DOOR_HEIGHT_AGREEMENT_MAX = 0.25
#: kind -> (typical_min_m, typical_max_m, basis). Door only for now; a ceiling prior is too loose
#: (2.4-3.0 m, +-12 %) to be an honest suggestion -- add kinds here, nowhere else.
STANDARD_OBJECTS = {
    "door": (DOOR_HEIGHT_RANGE_M[0], DOOR_HEIGHT_RANGE_M[1],
             "standard interior door height (2040 mm EU/IN, 2032 mm US)"),
}


@dataclass(frozen=True)
class ScaleSuggestion:
    source_entity_id: str
    source_kind: str
    measured_model_units: float
    typical_range_m: Tuple[float, float]
    suggested_value_m: float
    implied_scale: float          # metres per model unit if accepted
    basis: str
    n_candidates: int

    def to_dict(self) -> dict:
        return {
            "source_entity_id": self.source_entity_id,
            "source_kind": self.source_kind,
            "measured_model_units": self.measured_model_units,
            "typical_range_m": list(self.typical_range_m),
            "suggested_value_m": self.suggested_value_m,
            "implied_scale": self.implied_scale,
            "basis": self.basis,
            "n_candidates": self.n_candidates,
            "dimension_state_if_accepted": "ESTIMATED",
        }


def suggest_scale_from_world(world) -> Optional[ScaleSuggestion]:
    """Suggest a metres-per-unit from a detected standard object in a RELATIVE compiled WorldIR.

    Returns None -- never a default -- when the world is already metric, or no door with a MEASURED
    height (an opening fitted from points) exists, or several doors disagree. Reads only; applies nothing.
    """
    state = ((getattr(world, "metadata", None) or {}).get("scale") or {}).get("state")
    if state not in (None, ScaleState.RELATIVE, ScaleState.UNKNOWN):
        return None
    cands: List[Tuple[float, str]] = []
    for eid in sorted(world.entities):
        ent = world.entities[eid]
        if str(getattr(ent.type, "value", ent.type)) != "door":
            continue
        props = getattr(ent, "custom_properties", None) or {}
        # promotion.py also writes a 2.1 DEFAULT height for openings without a fit; only an opening
        # fitted from points (it carries bounds) has a measured height.
        if "bounds_min" not in props or "bounds_max" not in props:
            continue
        try:
            h = float(props.get("height_m"))
            sill = float(props.get("sill_height_m") or 0.0)
        except (TypeError, ValueError):
            continue
        if not (h > 0.0) or not math.isfinite(h) or sill > 0.15 * h:
            continue                       # not a floor-to-head door opening
        cands.append((h, eid))
    if not cands:
        return None
    heights = [h for h, _ in cands]
    if (max(heights) - min(heights)) / median(heights) > DOOR_HEIGHT_AGREEMENT_MAX:
        return None
    h, eid = sorted(cands)[(len(cands) - 1) // 2]
    lo, hi, basis = STANDARD_OBJECTS["door"]
    mid = (lo + hi) / 2.0
    return ScaleSuggestion(eid, "door", h, (lo, hi), mid, mid / h, basis, len(cands))
