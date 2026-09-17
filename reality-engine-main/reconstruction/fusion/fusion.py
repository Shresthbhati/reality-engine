"""Evidence fusion (spec sec 14 EVIDENCE FUSION / sec 15 CONFLICT).

The subsystem that combines independent observations of the same
real-world quantity -- LiDAR distance, photogrammetry distance, two
competing reconstructions' camera estimates -- into ONE coherent
observation without destroying any source. Different backends disagree;
the disagreement itself is information and must survive into WorldIR as
Provenance.CONFLICT, not be averaged away or resolved by a winner.

What this module guarantees (and why it is custom technology):

  - Confidence weighting: sources are combined by inverse-variance
    weights -- a tighter measurement moves the fused value more, exactly
    as the uncertainty each source claims says it should.
  - Conflict detection: two observations conflict when their difference
    exceeds CONFLICT_SIGMA (5.0) combined standard deviations
    (sqrt(sigma_a^2 + sigma_b^2), independent errors). Below that they
    are statistically consistent; at 5 combined sigma the probability of
    agreement under honest uncertainties is < 3e-7, so disagreement is
    real signal, not noise.
  - Conflict resolution, honestly: the fused value is still the
    inverse-variance weighted mean of ALL contributing observations
    (never a winner-take-all pick, never a silent discard), its
    provenance becomes CONFLICT, its confidence drops to the weakest
    contributor's (a chain of evidence is only as strong as its least
    certain link), and every conflicting pair is preserved verbatim in
    the result. Raw evidence is never destroyed.
  - Provenance propagation: a single observation passes through with its
    own provenance; any multi-source fusion is a derived value, so it is
    ESTIMATED -- unless a conflict was detected, in which case CONFLICT
    (spec sec 15's semantic: "LiDAR = 3.17 / photogrammetry = 3.22 -->
    CONFLICT / fused estimate / uncertainty" -- all three survive).

Not yet implemented here, by design (labelled, not hidden): outlier
rejection / robust estimation, temporal consistency across epochs, and
geometric-consistency checks for poses. The fusion engine below is the
deterministic core those will extend.

Deterministic: canonical input ordering, pure float math, no clocks, no
RNG -- the same observations always produce the same fused result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from engine.core.units import Unit
from provenance import Provenance
from world_ir import Measurement

#: Two observations conflict when |a - b| exceeds this many combined
#: standard deviations. 5 sigma: under honest Gaussian uncertainties the
#: agreement probability is < 3e-7, so a larger difference is evidence
#: of a real problem (calibration, scale, correspondence), not noise.
CONFLICT_SIGMA = 5.0


class FusionError(ValueError):
    """Raised when observations cannot be fused honestly."""


@dataclass(frozen=True)
class FusableObservation:
    """One source's claim about one real-world quantity.

    `unit` is a canonical engine Unit (no subsystem may silently mix
    units -- engine/core/units.py). `precision` is the source's claimed
    standard deviation; zero would be a claim of perfect knowledge, which
    no real sensor or model has, so it is rejected at construction.
    `evidence_ids` tie the claim back to the evidence that produced it.
    """

    value: float
    unit: Unit
    source: str
    provenance: Provenance
    confidence: float
    precision: float
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.source:
            raise FusionError("observation must name its source")
        if not (0.0 <= self.confidence <= 1.0):
            raise FusionError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.precision <= 0.0:
            raise FusionError(
                f"precision must be > 0 (a source claiming zero uncertainty "
                f"cannot be fused honestly), got {self.precision}"
            )
        if not math.isfinite(self.value):
            raise FusionError(f"value must be finite, got {self.value}")


@dataclass(frozen=True)
class MeasurementConflict:
    """One preserved disagreement between two sources.

    `delta` and `combined_precision` are in the observations' shared
    unit; `sigma_multiple` = delta / combined_precision is how far beyond
    statistical agreement the pair is. Never discarded: the CONFLICT
    provenance on the fused result points here.
    """

    source_a: str
    source_b: str
    value_a: float
    value_b: float
    delta: float
    combined_precision: float
    sigma_multiple: float


@dataclass(frozen=True)
class FusedEstimate:
    """The fused result for one quantity, with its full audit trail.

    provenance: the single source's own provenance on passthrough,
    ESTIMATED for an uncontested multi-source fusion, CONFLICT when
    disagreement was detected (spec sec 15). `conflicts` is empty exactly
    when provenance is not CONFLICT. `contributing` preserves every input
    observation verbatim -- the fused mean never replaces the evidence.
    """

    quantity: str
    value: float
    unit: Unit
    precision: float
    confidence: float
    provenance: Provenance
    method: str
    contributing: Tuple[FusableObservation, ...]
    conflicts: Tuple[MeasurementConflict, ...]
    #: RMS of pairwise deltas among contributing observations -- the
    #: observed spread of the sources, independent of their claimed
    #: precisions. None on passthrough (no spread to measure).
    agreement_rms: Optional[float]

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


def _canonical_order(observations: Sequence[FusableObservation]) -> List[FusableObservation]:
    """Deterministic processing order regardless of caller input order."""
    return sorted(
        observations,
        key=lambda o: (o.value, o.source, o.provenance.value, o.confidence),
    )


def _pairwise_conflicts(
    ordered: List[FusableObservation],
) -> List[MeasurementConflict]:
    """All conflicting pairs (i < j in canonical order), deterministically."""
    conflicts: List[MeasurementConflict] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            combined = math.sqrt(a.precision ** 2 + b.precision ** 2)
            delta = abs(a.value - b.value)
            if delta > CONFLICT_SIGMA * combined:
                conflicts.append(MeasurementConflict(
                    source_a=a.source,
                    source_b=b.source,
                    value_a=a.value,
                    value_b=b.value,
                    delta=delta,
                    combined_precision=combined,
                    sigma_multiple=delta / combined,
                ))
    return conflicts


def _pairwise_spread(ordered: List[FusableObservation]) -> float:
    """RMS of pairwise deltas -- the sources' actual observed spread."""
    deltas: List[float] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            deltas.append(abs(ordered[i].value - ordered[j].value))
    if not deltas:
        return 0.0
    return math.sqrt(sum(d * d for d in deltas) / len(deltas))


def _inverse_variance_fusion(ordered: List[FusableObservation]) -> Tuple[float, float]:
    """(fused value, fused precision) by inverse-variance weights.

    All precisions are > 0 (enforced at construction), so the weights are
    always finite and positive.
    """
    weights = [1.0 / (o.precision ** 2) for o in ordered]
    total = sum(weights)
    value = sum(o.value * w for o, w in zip(ordered, weights)) / total
    precision = 1.0 / math.sqrt(total)
    return value, precision


def fuse_quantity(
    observations: Sequence[FusableObservation],
    quantity: str,
) -> FusedEstimate:
    """Fuse independent observations of ONE real-world quantity.

    Raises FusionError on an empty set or a unit mismatch -- mixing
    meters with kilograms would produce a silently wrong number, and a
    silently wrong world is the failure this engine exists to prevent.

    With one observation the result is an exact passthrough (same value,
    precision, confidence, provenance); with several it is the
    inverse-variance weighted mean, ESTIMATED, or CONFLICT when any pair
    disagrees beyond CONFLICT_SIGMA combined sigmas.
    """
    if not observations:
        raise FusionError(
            f"cannot fuse quantity {quantity!r}: no observations provided"
        )
    ordered = _canonical_order(observations)
    unit = ordered[0].unit
    mismatched = [o.source for o in ordered if o.unit != unit]
    if mismatched:
        wrong = next(o.unit.value for o in ordered if o.unit != unit)
        raise FusionError(
            f"unit mismatch while fusing {quantity!r}: "
            f"{unit.value} vs {wrong} from source {mismatched[0]!r} -- "
            "refusing to mix units"
        )

    if len(ordered) == 1:
        o = ordered[0]
        return FusedEstimate(
            quantity=quantity,
            value=o.value,
            unit=o.unit,
            precision=o.precision,
            confidence=o.confidence,
            provenance=o.provenance,
            method="passthrough",
            contributing=(o,),
            conflicts=(),
            agreement_rms=None,
        )

    conflicts = _pairwise_conflicts(ordered)
    value, precision = _inverse_variance_fusion(ordered)
    spread = _pairwise_spread(ordered)
    provenance = Provenance.CONFLICT if conflicts else Provenance.ESTIMATED
    # Confidence: the weakest contributor's. A fused value inherits every
    # input's uncertainty chain; the least certain link is the honest floor.
    confidence = min(o.confidence for o in ordered)
    return FusedEstimate(
        quantity=quantity,
        value=value,
        unit=unit,
        precision=precision,
        confidence=confidence,
        provenance=provenance,
        method="inverse_variance_weighted_mean",
        contributing=tuple(ordered),
        conflicts=tuple(conflicts),
        agreement_rms=spread,
    )


def fused_to_measurement(fused: FusedEstimate) -> Measurement:
    """Bridge a fused estimate into a WorldIR Measurement (same unit
    string, precision, provenance incl. CONFLICT, and confidence) so the
    fusion result can flow into WorldIR wherever units align."""
    return Measurement(
        value=fused.value,
        unit=fused.unit.value,
        precision=fused.precision,
        provenance=fused.provenance,
        confidence=fused.confidence,
    )
