"""Time synchronization: canonical clock model + alignment (P2.1,
spec sec 10; docs/future/synchronization/TIME_SYNCHRONIZATION.md).

The problem this solves: every SensorStream's timestamps are values on
that sensor's OWN clock (sensors.py documents this deliberately --
"not globally comparable"). Two devices' logs cannot be correlated
until each is mapped onto a common timeline. This module provides that
mapping WITHOUT ever overwriting the original sensor timestamps.

THE MODEL:

    t_global = a * t_sensor + b

`a` is clock scale/drift (a real device clock deviates from ideal
rate), `b` is the offset. Single-device streams typically need only
`b`; GNSS/PPS-anchored streams can estimate both.

Components (mirroring the spec):

- `Timestamp` -- an original sensor time + the clock it belongs to.
  The original value is retained everywhere; nothing in this module
  mutates a stream.
- `ClockModel(clock_id, a, b, method, uncertainty)` -- immutable; one
  per sensor clock.
- `TimeAlignment` -- the result of applying a model to a stream: one
  `SynchronizedSample` per input sample, each carrying original
  timestamp, global timestamp (or an explicit UNSYNCHRONIZED state),
  clock id, offset, drift, sync method, and per-sample state.
- `SynchronizationDiagnostics` -- method, counts, offset/drift
  estimates, rejected samples with reasons, notes.

METHODS (spec's preference order; the registry below implements the
metadata-alignment family today -- shared clock and known offset --
and the seam accepts further backends in preference order):

1. shared clock (`a=1`, `b=0`, uncertainty zero by construction)
2. known offset (manual `b` from capture metadata)
3. GNSS/PPS anchoring            -- not implemented yet (slot open)
4. trigger synchronization       -- not implemented yet (slot open)
5. signal correlation            -- not implemented yet (slot open)
6. optimization (bundle a,b)     -- not implemented yet (slot open)

RULES ENFORCED HERE (spec):

- Original sensor timestamps are never overwritten or discarded: the
  input stream is untouched; every aligned sample carries
  `original_t`.
- Never fabricate uncertainty: an offset declared without an
  uncertainty estimate carries `offset_s=None` with an explicit basis
  ("declared without uncertainty"), never a made-up 0. Shared-clock
  zero uncertainty IS honest (b=0 by construction of the hardware).
- Sync failure degrades honestly: a stream no backend can synchronize
  yields every sample `UNSYNCHRONIZED` with the reason recorded in
  diagnostics -- never a silent re-stamp, never a guessed offset.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from evidence.sensors import SensorStream


class SynchronizationError(ValueError):
    """Misuse error: invalid model parameters or contract violation."""


class SynchronizationUnavailable(Exception):
    """A backend cannot honestly synchronize the given stream/context.

    Distinct from SynchronizationError: this is the honest "not for
    me" signal a backend raises so the selection seam can try the next
    backend -- and record the reason if none succeeds.
    """


class SyncState(str, Enum):
    """Per-sample synchronization outcome. UNSYNCHRONIZED is an honest,
    visible state -- never silently re-stamped."""

    SYNCHRONIZED = "SYNCHRONIZED"
    UNSYNCHRONIZED = "UNSYNCHRONIZED"


class SyncMethod(str, Enum):
    """Synchronization method, in the spec's preference order. A
    ClockModel names the method that produced it; consumers can gate
    trust on it (hardware/shared-clock facts beat metadata guesses)."""

    SHARED_CLOCK = "shared_clock"
    KNOWN_OFFSET = "known_offset"
    GNSS_PPS = "gnss_pps"
    TRIGGER = "trigger"
    SIGNAL_CORRELATION = "signal_correlation"
    OPTIMIZATION = "optimization"


@dataclass(frozen=True)
class SyncUncertainty:
    """Estimate quality for a clock model. `None` means UNKNOWN --
    recorded as unknown, never coerced to zero (fabricating a tight
    uncertainty would be exactly the silent lie the spec forbids).
    `basis` says where each number came from."""

    offset_s: Optional[float] = None
    drift: Optional[float] = None
    basis: str = "unknown"

    def __post_init__(self):
        for name in ("offset_s", "drift"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise SynchronizationError(
                    f"uncertainty {name} must be None (unknown) or finite >= 0, got {v!r}"
                )

    def to_dict(self) -> dict:
        return {"offset_s": self.offset_s, "drift": self.drift, "basis": self.basis}

    @classmethod
    def from_dict(cls, data: dict) -> "SyncUncertainty":
        return cls(
            offset_s=data.get("offset_s"),
            drift=data.get("drift"),
            basis=data.get("basis", "unknown"),
        )


@dataclass(frozen=True)
class Timestamp:
    """An original sensor timestamp and the clock it was read on. The
    sensor's own value is authoritative and retained; synchronization
    adds a mapping, never a replacement."""

    sensor_t: float
    clock_id: str

    def __post_init__(self):
        if not math.isfinite(self.sensor_t):
            raise SynchronizationError(f"sensor_t must be finite, got {self.sensor_t!r}")
        if not self.clock_id:
            raise SynchronizationError("clock_id must be a non-empty string")

    def to_dict(self) -> dict:
        return {"sensor_t": self.sensor_t, "clock_id": self.clock_id}


@dataclass(frozen=True)
class ClockModel:
    """One sensor clock's mapping onto the global timeline:

        t_global = a * t_sensor + b

    Immutable. `a` must be finite and positive (a <= 0 would invert or
    collapse time -- a model that does that is a bug, not a clock).
    `b` and `a` may be exactly 1.0/0.0 for the shared-clock case."""

    clock_id: str
    a: float = 1.0
    b: float = 0.0
    method: SyncMethod = SyncMethod.KNOWN_OFFSET
    uncertainty: SyncUncertainty = field(default_factory=SyncUncertainty)

    def __post_init__(self):
        if not self.clock_id:
            raise SynchronizationError("clock_id must be a non-empty string")
        if not math.isfinite(self.a) or self.a <= 0.0:
            raise SynchronizationError(
                f"drift/scale a must be finite and > 0 (a<=0 inverts/collapses time), got {self.a!r}"
            )
        if not math.isfinite(self.b):
            raise SynchronizationError(f"offset b must be finite, got {self.b!r}")

    def apply(self, sensor_t: float) -> float:
        """Map one original sensor timestamp onto the global timeline.
        Raises on non-finite input; callers working with untrusted
        per-sample data should catch and reject that sample (see
        TimeAlignment construction)."""
        if not math.isfinite(sensor_t):
            raise SynchronizationError(f"sensor_t must be finite, got {sensor_t!r}")
        return self.a * sensor_t + self.b

    def to_dict(self) -> dict:
        return {
            "clock_id": self.clock_id,
            "a": self.a,
            "b": self.b,
            "method": self.method.value,
            "uncertainty": self.uncertainty.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClockModel":
        return cls(
            clock_id=data["clock_id"],
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            method=SyncMethod(data.get("method", "known_offset")),
            uncertainty=SyncUncertainty.from_dict(data.get("uncertainty", {})),
        )


@dataclass(frozen=True)
class SynchronizedSample:
    """One sample's view through a ClockModel. The original sensor
    timestamp is always present; `global_t` is None exactly when
    `sync_state` is UNSYNCHRONIZED (with the reason recorded)."""

    index: int
    original_t: float
    clock_id: str
    offset_s: float
    drift: float
    sync_method: str
    sync_state: SyncState
    global_t: Optional[float] = None
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        if self.sync_state is SyncState.SYNCHRONIZED and self.global_t is None:
            raise SynchronizationError(
                "a SYNCHRONIZED sample must carry global_t"
            )
        if self.sync_state is SyncState.UNSYNCHRONIZED and self.global_t is not None:
            raise SynchronizationError(
                "an UNSYNCHRONIZED sample must not carry a global_t -- that would "
                "be a fabricated re-stamp"
            )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "original_t": self.original_t,
            "clock_id": self.clock_id,
            "offset_s": self.offset_s,
            "drift": self.drift,
            "sync_method": self.sync_method,
            "sync_state": self.sync_state.value,
            "global_t": self.global_t,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class SynchronizationDiagnostics:
    """Observed facts of one synchronization run. Estimates are None
    when the method does not produce them; residuals only exist when a
    residual can actually be computed (spec rule: never fabricate)."""

    method: str
    n_samples: int
    n_synchronized: int
    n_rejected: int
    offset_estimate_s: Optional[float] = None
    drift_estimate: Optional[float] = None
    residual_max_s: Optional[float] = None
    stream_was_monotonic: Optional[bool] = None
    rejected: Tuple[dict, ...] = field(default_factory=tuple)  # {index, original_t, reason}
    backend_attempts: Tuple[dict, ...] = field(default_factory=tuple)  # {backend, ok, reason}
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "n_synchronized": self.n_synchronized,
            "n_rejected": self.n_rejected,
            "offset_estimate_s": self.offset_estimate_s,
            "drift_estimate": self.drift_estimate,
            "residual_max_s": self.residual_max_s,
            "stream_was_monotonic": self.stream_was_monotonic,
            "rejected": list(self.rejected),
            "backend_attempts": list(self.backend_attempts),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TimeAlignment:
    """The result of synchronizing one SensorStream with one
    ClockModel. Immutable; the source stream is untouched."""

    clock_id: str
    model: ClockModel
    samples: Tuple[SynchronizedSample, ...]
    diagnostics: SynchronizationDiagnostics

    @property
    def synchronized(self) -> Tuple[SynchronizedSample, ...]:
        """Only the samples that actually landed on the global
        timeline. Downstream cross-stream consumers MUST use this (or
        check sync_state themselves) -- iterating raw samples would
        silently include UNSYNCHRONIZED ones whose global_t is None."""
        return tuple(s for s in self.samples if s.sync_state is SyncState.SYNCHRONIZED)

    @property
    def rejected(self) -> Tuple[SynchronizedSample, ...]:
        return tuple(s for s in self.samples if s.sync_state is SyncState.UNSYNCHRONIZED)

    def global_times(self) -> Tuple[float, ...]:
        return tuple(s.global_t for s in self.synchronized)

    def to_dict(self) -> dict:
        return {
            "clock_id": self.clock_id,
            "model": self.model.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
            "diagnostics": self.diagnostics.to_dict(),
        }


class ISynchronizationBackend(ABC):
    """One strategy for producing a ClockModel for a stream.

    Contract: `build_model` returns a ClockModel or raises
    SynchronizationUnavailable with the reason it cannot honestly
    synchronize (the selection seam records attempts and, if every
    backend declines, degrades every sample to UNSYNCHRONIZED). It
    must never invent a plausible-looking model, and must never mutate
    the stream.
    """

    #: Registry name, also the default diagnostics label.
    name: str = "abstract"

    @abstractmethod
    def build_model(self, stream: "SensorStream", context: "SyncContext") -> ClockModel:
        ...


@dataclass(frozen=True)
class SyncContext:
    """What a backend may look at besides the stream itself: the
    clock's identity and any capture/source metadata where
    synchronization facts are recorded (offsets, shared-clock flags,
    drift). Backends that need more should extend this deliberately,
    not reach around it."""

    clock_id: str
    metadata: Dict[str, object] = field(default_factory=dict)


class MetadataAlignmentBackend(ISynchronizationBackend):
    """The first sync backend (spec preference 1-2): derive the clock
    model from explicit capture metadata.

    Accepted metadata keys (all optional, checked in this order):

    - ``shared_clock``: truthy -> SHARED_CLOCK model (a=1, b=0,
      uncertainty zero by construction of the hardware). This is the
      one case where zero uncertainty is honest: the spec names it
      ("uncertainty from spec"), not an estimate.
    - ``clock_offset_s``: number -> KNOWN_OFFSET model (b = value;
      a = ``clock_drift`` if also given, else 1.0). If
      ``clock_offset_uncertainty_s`` is present it is recorded as the
      offset uncertainty; otherwise the uncertainty is UNKNOWN with an
      explicit basis -- never fabricated as 0.
    - ``clock_drift`` alone (without an offset) -> KNOWN_OFFSET with
      b=0 and the given a.

    Raises SynchronizationUnavailable when none of these are present:
    "no synchronization metadata" is a fact about the capture, and the
    honest outcome is UNSYNCHRONIZED samples -- not a guessed offset.
    """

    name = "metadata_alignment"

    def build_model(self, stream: "SensorStream", context: SyncContext) -> ClockModel:
        meta = context.metadata or {}
        if meta.get("shared_clock"):
            return ClockModel(
                clock_id=context.clock_id,
                a=1.0,
                b=0.0,
                method=SyncMethod.SHARED_CLOCK,
                uncertainty=SyncUncertainty(
                    offset_s=0.0,
                    drift=0.0,
                    basis="shared clock by construction (spec preference 1)",
                ),
            )
        offset = meta.get("clock_offset_s")
        drift = meta.get("clock_drift")
        if offset is None and drift is None:
            raise SynchronizationUnavailable(
                f"no synchronization metadata for clock {context.clock_id!r} "
                "(checked shared_clock, clock_offset_s, clock_drift)"
            )
        if offset is not None:
            b = _require_number("clock_offset_s", offset, context.clock_id)
        else:
            b = 0.0
        a = 1.0
        if drift is not None:
            a = _require_number("clock_drift", drift, context.clock_id)
        unc_offset = meta.get("clock_offset_uncertainty_s")
        if unc_offset is not None:
            uncertainty = SyncUncertainty(
                offset_s=float(unc_offset),
                drift=None,
                basis="declared in capture metadata",
            )
        else:
            uncertainty = SyncUncertainty(
                offset_s=None,
                drift=None,
                basis="offset declared without an uncertainty estimate (unknown, not zero)",
            )
        return ClockModel(
            clock_id=context.clock_id,
            a=a,
            b=b,
            method=SyncMethod.KNOWN_OFFSET,
            uncertainty=uncertainty,
        )


def _require_number(key: str, value: object, clock_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynchronizationUnavailable(
            f"metadata {key!r} for clock {clock_id!r} must be a number, got {type(value).__name__}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise SynchronizationUnavailable(
            f"metadata {key!r} for clock {clock_id!r} must be finite, got {v!r}"
        )
    return v


class _EstimationBackend(ISynchronizationBackend):
    """Shared machinery for backends that ESTIMATE (a, b) from pairs of
    (sensor_t, global_t) correspondences instead of reading them from
    metadata. Estimates a as a slope and b as the intercept.

    Estimation quality is computed from residuals, never fabricated:
    - >= 3 distinct-time pairs -> least-squares slope/intercept with a
      real max residual (sync uncertainty derived from it);
    - exactly 2 distinct times -> exact 2-point solution with zero
      residual BY CONSTRUCTION (recorded as such in the basis);
    - 1 pair -> offset-only (a=1): estimating a slope from one point
      would be fabrication; uncertainty UNKNOWN unless declared.

    Outlier rejection: with >= 4 distinct times, pairs whose residual
    exceeds `outlier_sigma` estimated-scale-sigmas are dropped once and
    the fit is redone (rejected pairs ride in the context as
    ``rejected_pairs`` so callers/diagnostics can see them).
    """

    #: Registry name; also the SyncMethod the produced model carries.
    name = "abstract-estimator"
    method = SyncMethod.KNOWN_OFFSET

    #: Multiplicative band around the estimated residual scale within
    #: which a pair is considered an inlier (MAD-style, robust to the
    #: outliers this band is hunting).
    OUTLIER_SIGMA = 4.0

    def _fit(self, pairs, context):
        # pairs: sequence of (sensor_t, global_t) from trusted evidence.
        clean = []
        for st, gt in pairs:
            st_f = float(st)
            gt_f = float(gt)
            if not (math.isfinite(st_f) and math.isfinite(gt_f)):
                continue  # non-finite pair is corrupt evidence; not estimable
            clean.append((st_f, gt_f))
        distinct = sorted({st for st, _ in clean})
        if not clean:
            raise SynchronizationUnavailable(
                f"{self.name}: no finite (sensor_t, global_t) correspondences "
                "provided -- cannot estimate a clock model"
            )
        if len(distinct) < 2:
            # Offset-only: one pair pins b given a=1; a slope would be
            # fabricated from a single point.
            st, gt = clean[0]
            context.metadata.setdefault("_estimator_notes", []).append(
                "single correspondence: offset-only fit (a=1); drift not estimable"
           )
            return 1.0, gt - st, None, ()

        xs = [st for st, _ in clean]
        ys = [gt for _, gt in clean]
        if len(distinct) == 2:
            # Exact 2-point line through two distinct sensor times.
            (x1, y1), (x2, y2) = clean[0], clean[-1]
            if x2 == x1:  # cannot happen: distinct times guaranteed
                raise SynchronizationUnavailable("degenerate 2-point fit")
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            context.metadata.setdefault("_estimator_notes", []).append(
                "two correspondences: exact 2-point fit; residual zero by "
                "construction (no redundancy to estimate quality from)"
            )
            if a <= 0.0:
                raise SynchronizationUnavailable(
                    f"{self.name}: 2-point fit produced non-positive scale a={a!r} "
                    "-- the correspondences are inconsistent with a clock"
                )
            # No redundancy -> no measured residual. Reporting 0.0 here
            # would fabricate "perfect" estimate quality from a
            # construction that cannot fail; None is the honest value.
            return a, b, None, ()

        # >= 3 distinct times: least squares with one outlier pass.
        def _lstsq(indices):
            n = len(indices)
            sx = sum(xs[i] for i in indices)
            sy = sum(ys[i] for i in indices)
            sxx = sum(xs[i] * xs[i] for i in indices)
            sxy = sum(xs[i] * ys[i] for i in indices)
            denom = n * sxx - sx * sx
            if denom == 0.0:
                raise SynchronizationUnavailable(
                    f"{self.name}: correspondences collapse to a single distinct "
                    "sensor time -- slope not estimable"
               )
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
            return a, b

        indices = list(range(len(clean)))  # 0-based rows into xs/ys

        a, b = _lstsq(indices)
        rejected_pairs = ()
        inlier_idx = list(indices)
        # Gross-outlier rejection: deterministic RANSAC-style consensus.
        # A correspondence is an inlier only if it sits within a tight
        # relative band of a two-point seed line. Rejection fires only
        # when a strict-majority consensus subset fits near-exactly and
        # the remainder does not -- noisy-but-consistent data is never
        # silently trimmed (the band is far tighter than realistic noise,
        # so a noisy ensemble fails the majority-consensus condition and
        # is kept whole, outliers and all, visible in the residual).
        if len(clean) >= 4:
            y_span = max(ys) - min(ys)
            band = 1e-6 * max(1.0, y_span)
            n_total = len(clean)
            best = None  # (inlier_index_list, a_seed, b_seed); first best wins
            for si in range(n_total):
                for sj in range(si + 1, n_total):
                    if xs[sj] == xs[si]:
                        continue
                    a_seed = (ys[sj] - ys[si]) / (xs[sj] - xs[si])
                    if a_seed <= 0.0:
                        continue  # not a physical clock hypothesis
                    b_seed = ys[si] - a_seed * xs[si]
                    inliers = [
                        k for k in range(n_total)
                        if abs(ys[k] - (a_seed * xs[k] + b_seed)) <= band
                    ]
                    if best is None or len(inliers) > len(best[0]):
                        best = (inliers, a_seed, b_seed)
            if best is not None:
                inliers = best[0]
                if len(inliers) < n_total and len(inliers) >= max(3, (n_total + 1) // 2):
                    inlier_idx = inliers
                    rejected_pairs = tuple(
                        {"sensor_t": xs[k], "global_t": ys[k]}
                        for k in range(n_total) if k not in set(inliers)
                    )
                    context.metadata.setdefault("_estimator_notes", []).append(
                        f"outlier rejection: {len(rejected_pairs)} of {n_total} "
                        "correspondences outside the consensus band"
                    )
                    a, b = _lstsq(inlier_idx)
        if a <= 0.0:
            raise SynchronizationUnavailable(
                f"{self.name}: fit produced non-positive scale a={a!r} -- the "
                "correspondences are inconsistent with a clock"
            )
        # Residual is measured over the inlier set (the fit's own basis);
        # a rejected outlier's residual would overstate the error of the
        # model actually produced.
        residual_max = max(abs(ys[i] - (a * xs[i] + b)) for i in inlier_idx)
        context.metadata["_residual_max"] = residual_max
        return a, b, residual_max, rejected_pairs

    def _uncertainty_from(self, residual_max, notes, context):
        if residual_max is None:
            return SyncUncertainty(
                basis="; ".join(notes) or "estimate without residual basis",
            )
        return SyncUncertainty(
            offset_s=residual_max,
            drift=None,
            basis=(
                "; ".join(notes) + "; " if notes else ""
            ) + f"max correspondence residual {residual_max:.3e}s as estimate quality",
        )

    def _build(self, stream, context, pairs):
        a, b, residual_max, rejected_pairs = self._fit(pairs, context)
        if rejected_pairs:
            context.metadata["rejected_pairs"] = rejected_pairs
        notes = list(context.metadata.get("_estimator_notes", []))
        uncertainty = self._uncertainty_from(residual_max, notes, context)
        return ClockModel(
            clock_id=context.clock_id,
            a=a,
            b=b,
            method=self.method,
            uncertainty=uncertainty,
        )


class GnssPpsBackend(_EstimationBackend):
    """Spec preference 3: GNSS/PPS anchoring.

    Synchronizes against trusted GNSS time from the SAME capture: the
    stream's own GNSS samples (kind == "gnss") provide (sensor_t ->
    GNSS epoch time) correspondences, or declared anchor pairs arrive
    via metadata ``gnss_anchor_pairs`` as [(sensor_t, gnss_t), ...].

    A stream's own GNSS samples only anchor GNSS-kind streams -- an IMU
    stream has no GNSS receiver, so claiming its samples are GNSS time
    would be fabrication; such streams decline (and the caller should
    use declared anchors or another backend).
    """

    name = "gnss_pps"
    method = SyncMethod.GNSS_PPS

    def build_model(self, stream, context):
        pairs = []
        declared = (context.metadata or {}).get("gnss_anchor_pairs")
        if declared is not None:
            if not isinstance(declared, (list, tuple)):
                raise SynchronizationUnavailable(
                    "gnss_pps: metadata 'gnss_anchor_pairs' must be a sequence "
                    "of (sensor_t, gnss_t) pairs"
                )
            for pr in declared:
                if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                    raise SynchronizationUnavailable(
                        "gnss_pps: each anchor pair must be (sensor_t, gnss_t)"
                    )
                pairs.append((pr[0], pr[1]))
        elif stream.kind == "gnss":
            # The stream's own samples: sensor_t -> GNSS epoch time. A
            # PPS-anchored receiver timestamps samples in GNSS time, so
            # the identity model (a=1, b=0) holds BY CONSTRUCTION -- the
            # pairs are self-identical, so any "fit residual" measured on
            # them is trivially zero and carries no information. The
            # model is built directly with no measured residual rather
            # than routed through the estimator.
            if not stream.samples:
                raise SynchronizationUnavailable(
                    "gnss_pps: GNSS-kind stream carries no samples to anchor"
                )
            notes = context.metadata.setdefault("_estimator_notes", [])
            notes.append(
                "stream's own GNSS samples as anchors (PPS-disciplined receiver "
                "timestamps in GNSS time; identity model by construction -- "
                "no measured residual exists)"
            )
            return ClockModel(
                clock_id=context.clock_id,
                a=1.0,
                b=0.0,
                method=self.method,
                uncertainty=SyncUncertainty(basis="; ".join(notes)),
            )
        else:
            raise SynchronizationUnavailable(
                f"gnss_pps: stream kind {stream.kind!r} carries no GNSS time and "
                "no 'gnss_anchor_pairs' metadata was declared"
            )
        return self._build(stream, context, pairs)


class TriggerBackend(_EstimationBackend):
    """Spec preference 4: trigger synchronization.

    Correspondences come from hardware/software trigger EVENTS: pairs
    of (sensor_t, global_t) captured when a shared trigger fired, via
    metadata ``trigger_pairs``. Without declared trigger events there
    is nothing to synchronize on -- a trigger backend that invented
    correspondences would be guessing, so it declines.
    """

    name = "trigger"
    method = SyncMethod.TRIGGER

    def build_model(self, stream, context):
        pairs = (context.metadata or {}).get("trigger_pairs")
        if not pairs:
            raise SynchronizationUnavailable(
                "trigger: no 'trigger_pairs' metadata ([(sensor_t, global_t), ...]) "
                "-- no trigger events to synchronize on"
            )
        if not isinstance(pairs, (list, tuple)):
            raise SynchronizationUnavailable(
                "trigger: 'trigger_pairs' must be a sequence of (sensor_t, global_t) pairs"
            )
        for pr in pairs:
            if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                raise SynchronizationUnavailable(
                    "trigger: each trigger pair must be (sensor_t, global_t)"
                )
        return self._build(stream, context, list(pairs))


class SignalCorrelationBackend(_EstimationBackend):
    """Spec preference 5: signal correlation (audio/flash/event).

    Estimates the offset/drift by cross-correlating the stream against
    a reference EVENT sequence from the same physical process, via
    metadata ``reference_events``: [(global_t, value), ...] -- e.g. a
    flash at known global times, an audio chirp, a button press.

    The stream side is the sample times themselves: each sample is an
    event whose sensor_t is matched to the nearest reference event in
    the CORRELATED signal. This backend implements the event-correlation
    form: metadata ``observed_events`` = [(sensor_t, value), ...] from
    the stream's own sensor (e.g. detected flash times in camera
    frames). It cross-correlates the two event trains over a lag/drift
    search and returns the best (a, b) with the match residual as
    estimate quality. Without both trains it declines -- correlation
    against an imagined reference is fabrication.
    """

    name = "signal_correlation"
    method = SyncMethod.SIGNAL_CORRELATION

    def build_model(self, stream, context):
        meta = context.metadata or {}
        ref = meta.get("reference_events")
        obs = meta.get("observed_events")
        if ref is None or obs is None:
            raise SynchronizationUnavailable(
                "signal_correlation: needs metadata 'reference_events' "
                "([(global_t, value), ...]) AND 'observed_events' "
                "([(sensor_t, value), ...]) -- correlation against an "
                "imagined reference is fabrication"
        )
        if not isinstance(ref, (list, tuple)) or not isinstance(obs, (list, tuple)):
            raise SynchronizationUnavailable(
                "signal_correlation: 'reference_events' and 'observed_events' "
                "must be sequences"
            )
        for name_key, seq in (("reference_events", ref), ("observed_events", obs)):
            for ev in seq:
                if not isinstance(ev, (list, tuple)) or len(ev) != 2:
                    raise SynchronizationUnavailable(
                        f"signal_correlation: each {name_key} event must be (time, value)"
                    )
        ref_times = [float(ev[0]) for ev in ref]
        obs_times = [float(ev[0]) for ev in obs]
        if len(ref_times) < 2 or len(obs_times) < 2:
            raise SynchronizationUnavailable(
                "signal_correlation: each event train needs >= 2 events to "
                "correlate (one event cannot distinguish offset from drift)"
            )
        ref_vals = [float(ev[1]) for ev in ref]
        obs_vals = [float(ev[1]) for ev in obs]

        # Search (a, b) so that a*obs_t + b best aligns the observed
        # train onto the reference train. Search space: drift within
        # +-20% of 1.0, offset within the reference span. Score = mean
        # nearest-neighbor distance between mapped observed events and
        # reference events (values gate plausibility, not alignment).
        ref_span = max(ref_times) - min(ref_times)
        if ref_span <= 0.0:
            raise SynchronizationUnavailable(
                "signal_correlation: reference events share one timestamp"
            )
        # Decline gate: events are the signal, so a plausible alignment
        # puts each mapped observed event near SOME reference event
        # relative to how spread-out the reference events are. Gate on
        # 10% of the MEDIAN reference spacing -- a fixed fraction of the
        # span would pass trains that merely overlap part of the span
        # while aligning no actual events.
        sorted_ref = sorted(ref_times)
        spacings = [b - a for a, b in zip(sorted_ref, sorted_ref[1:])]
        median_spacing = sorted(spacings)[len(spacings) // 2]
        decline_gate = max(0.1 * median_spacing, 1e-9)
        best = None
        drifts = [1.0 + 0.02 * k for k in range(-10, 11)]  # 0.8..1.2
        n_offsets = 201
        for a_try in drifts:
            obs_min = min(a_try * t for t in obs_times)
            obs_max = max(a_try * t for t in obs_times)
            span = obs_max - obs_min
            for j in range(n_offsets):
                b_try = min(ref_times) - obs_min + ref_span * (j / (n_offsets - 1)) - span * (j / (n_offsets - 1))
                score = 0.0
                for t_obs in obs_times:
                    mapped = a_try * t_obs + b_try
                    score += min(abs(mapped - rt) for rt in ref_times)
                score /= len(obs_times)
                if best is None or score < best[0]:
                    best = (score, a_try, b_try)
        if best is None or best[0] > decline_gate:
            raise SynchronizationUnavailable(
                "signal_correlation: no drift/offset hypothesis aligned the "
                "event trains (best mean nearest-event distance "
                f"{best[0] if best else 'n/a'} exceeds the plausibility gate "
                f"{decline_gate:.3e}s = 10% of median reference event spacing)"
            )
        score, a, b = best
        context.metadata.setdefault("_estimator_notes", []).append(
            f"correlation match: mean nearest-event distance {score:.3e}s over "
            f"{len(obs_times)} observed vs {len(ref_times)} reference events"
        )
        context.metadata["_residual_max"] = score
        # Refine (a, b) by least squares on the greedy nearest-neighbor
        # correspondence to polish past the grid resolution.
        corr = []
        for t_obs in obs_times:
            mapped = a * t_obs + b
            rt = min(ref_times, key=lambda r: abs(mapped - r))
            corr.append((t_obs, rt))
        if len({st for st, _ in corr}) >= 2:
            a, b, residual_max, rejected_pairs = self._fit(corr, context)
            # The polish fits whatever correspondences it was given; a
            # fit that does not actually place mapped events near
            # reference events is not synchronization. Enforce the same
            # plausibility gate on the final measured residual.
            if residual_max is not None and residual_max > decline_gate:
                raise SynchronizationUnavailable(
                    "signal_correlation: refined fit does not align the event "
                    f"trains (max correspondence residual {residual_max:.3e}s "
                    f"exceeds the plausibility gate {decline_gate:.3e}s)"
                )
            if rejected_pairs:
                context.metadata["rejected_pairs"] = rejected_pairs
            notes = list(context.metadata.get("_estimator_notes", []))
            return ClockModel(
                clock_id=context.clock_id,
                a=a,
                b=b,
                method=self.method,
                uncertainty=self._uncertainty_from(residual_max, notes, context),
            )
        return ClockModel(
            clock_id=context.clock_id,
            a=a,
            b=b,
            method=self.method,
            uncertainty=self._uncertainty_from(score, ["correlation grid match"], context),
        )


class OptimizationBackend(_EstimationBackend):
    """Spec preference 6 (last): optimization-based alignment.

    Bundle-adjust (a, b) for MULTIPLE clocks jointly against cross-stream
    constraints: metadata ``cross_stream_constraints`` =
    [(clock_i, t_i, clock_j, t_j, dt_weight), ...] meaning "the event at
    clock i time t_i is the same physical event as clock j time t_j".

    This is deliberately the LAST backend: it will happily fit whatever
    it is given, so it must never win over a stronger method. Its
    advantage is that it works when no single stream has trusted time,
    by distributing one shared global timeline across clocks.
    """

    name = "optimization"
    method = SyncMethod.OPTIMIZATION

    def build_model(self, stream, context):
        constraints = (context.metadata or {}).get("cross_stream_constraints")
        if not constraints:
            raise SynchronizationUnavailable(
                "optimization: no 'cross_stream_constraints' metadata "
                "([(clock_i, t_i, clock_j, t_j), ...]) -- nothing to optimize"
            )
        for con in constraints:
            if not isinstance(con, (list, tuple)) or len(con) < 4:
                raise SynchronizationUnavailable(
                    "optimization: each constraint must be "
                    "(clock_i, t_i, clock_j, t_j)"
                )
        # The target clock must appear in the constraint graph, otherwise
        # the produced model would be unconstrained by any evidence.
        involved = {con[0] for con in constraints} | {con[2] for con in constraints}
        if context.clock_id not in involved:
            raise SynchronizationUnavailable(
                f"optimization: clock {context.clock_id!r} appears in no "
                "cross-stream constraint -- its model would be unconstrained"
            )
        # Solve: minimize sum over constraints of
        #   ((a_i*t_i + b_i) - (a_j*t_j + b_j))^2
        # with the gauge fixed by pinning the FIRST clock (deterministic
        # order) to a=1, b=0 -- without a gauge the system has a global
        # scale/offset null space and "the" answer would be arbitrary.
        import numpy as _np

        clocks = sorted(involved)
        gauge = clocks[0]
        free = [c for c in clocks if c != gauge]
        idx = {c: k for k, c in enumerate(free)}
        n_unknowns = 2 * len(free)
        rows = []
        rhs = []
        for con in constraints:
            ci, ti, cj, tj = con[0], float(con[1]), con[2], float(con[3])
            row = _np.zeros(n_unknowns)
            const = 0.0
            # residual r = (a_i t_i + b_i) - (a_j t_j + b_j); gauge a=1,b=0.
            for c, t, sign in ((ci, ti, 1.0), (cj, tj, -1.0)):
                if c == gauge:
                    const += sign * t
                else:
                    k = idx[c]
                    row[2 * k] += sign * t
                    row[2 * k + 1] += sign
            # r = row . x + const = 0  ->  row . x = -const
            rows.append(row)
            rhs.append(-const)
        M = _np.vstack(rows)
        v = _np.array(rhs)
        if M.shape[0] < n_unknowns:
            raise SynchronizationUnavailable(
                f"optimization: {M.shape[0]} constraints for {n_unknowns} unknowns "
                "-- underdetermined; add more cross-stream constraints"
            )
        x, *_ = _np.linalg.lstsq(M, v, rcond=None)
        params = {gauge: (1.0, 0.0)}
        for c in free:
            k = idx[c]
            params[c] = (float(x[2 * k]), float(x[2 * k + 1]))
        a, b = params[context.clock_id]
        if a <= 0.0:
            raise SynchronizationUnavailable(
                f"optimization: fit produced non-positive scale a={a!r} -- the "
                "constraint graph is inconsistent with a set of clocks"
            )
        residual_max = 0.0
        for con in constraints:
            ci, ti, cj, tj = con[0], float(con[1]), con[2], float(con[3])
            ai, bi = params[ci]
            aj, bj = params[cj]
            residual_max = max(residual_max, abs((ai * ti + bi) - (aj * tj + bj)))
        context.metadata.setdefault("_estimator_notes", []).append(
            f"joint fit over {len(clocks)} clocks, {len(constraints)} constraints; "
            f"gauge pinned to {gauge!r} (a=1, b=0)"
        )
        context.metadata["_residual_max"] = residual_max
        notes = list(context.metadata.get("_estimator_notes", []))
        return ClockModel(
            clock_id=context.clock_id,
            a=a,
            b=b,
            method=self.method,
            uncertainty=self._uncertainty_from(residual_max, notes, context),
        )


#: Backend registry in the spec's preference order. The estimation
#: backends run BEFORE metadata would let a weak declared offset beat a
#: measured fit? No -- the spec's order is authoritative: metadata
#: (preferences 1-2) first, then measurement backends in spec order.
#: A declared shared-clock fact outranks an estimated one.
DEFAULT_BACKENDS: Tuple[ISynchronizationBackend, ...] = (
    MetadataAlignmentBackend(),
    GnssPpsBackend(),
    TriggerBackend(),
    SignalCorrelationBackend(),
    OptimizationBackend(),
)


def synchronize_stream(
    stream: "SensorStream",
    clock_id: str,
    metadata: Optional[Dict[str, object]] = None,
    backends: Sequence[ISynchronizationBackend] = DEFAULT_BACKENDS,
) -> TimeAlignment:
    """Synchronize one SensorStream onto the global timeline.

    Tries `backends` in order; the first that produces a model wins.
    If every backend declines, EVERY sample degrades to
    UNSYNCHRONIZED (global_t=None) and the reasons are recorded in
    diagnostics -- the honest failure the spec requires. The input
    stream is never mutated; original timestamps ride along on every
    aligned sample.
    """
    if not backends:
        raise SynchronizationError("backends sequence must not be empty")
    if not clock_id:
        raise SynchronizationError("clock_id must be a non-empty string")

    context = SyncContext(clock_id=clock_id, metadata=dict(metadata or {}))
    attempts: List[dict] = []
    model: Optional[ClockModel] = None
    for backend in backends:
        try:
            candidate = backend.build_model(stream, context)
        except SynchronizationUnavailable as exc:
            attempts.append({"backend": backend.name, "ok": False, "reason": str(exc)})
            continue
        attempts.append({"backend": backend.name, "ok": True, "reason": None})
        model = candidate
        break

    notes: List[str] = []
    if model is None:
        notes.append(
            "no backend could synchronize this stream -- samples are "
            "UNSYNCHRONIZED (originals retained, nothing re-stamped)"
        )
        samples = tuple(
            SynchronizedSample(
                index=i,
                original_t=sample.t,
                clock_id=clock_id,
                offset_s=0.0,
                drift=1.0,
                sync_method="none",
                sync_state=SyncState.UNSYNCHRONIZED,
                global_t=None,
                rejection_reason="no synchronization backend produced a clock model",
            )
            for i, sample in enumerate(stream.samples)
        )
        diagnostics = SynchronizationDiagnostics(
            method="none",
            n_samples=len(samples),
            n_synchronized=0,
            n_rejected=len(samples),
            backend_attempts=tuple(attempts),
            notes=tuple(notes),
        )
        # A no-model alignment must not fabricate a ClockModel either;
        # the model field stays absent via a sentinel-free design:
        # TimeAlignment requires one, so construct the alignment with a
        # neutral identity model ONLY as a container label while every
        # sample remains UNSYNCHRONIZED and diagnostics.method="none".
        container_model = ClockModel(
            clock_id=clock_id,
            a=1.0,
            b=0.0,
            method=SyncMethod.KNOWN_OFFSET,
            uncertainty=SyncUncertainty(basis="no synchronization performed"),
        )
        return TimeAlignment(
            clock_id=clock_id,
            model=container_model,
            samples=samples,
            diagnostics=diagnostics,
        )

    samples: List[SynchronizedSample] = []
    rejected: List[dict] = []
    negative_time_notes: List[str] = []
    for i, sample in enumerate(stream.samples):
        original_t = sample.t
        try:
            global_t = model.apply(original_t)
        except SynchronizationError as exc:
            samples.append(SynchronizedSample(
                index=i, original_t=original_t, clock_id=clock_id,
                offset_s=model.b, drift=model.a, sync_method=model.method.value,
                sync_state=SyncState.UNSYNCHRONIZED, global_t=None,
                rejection_reason=str(exc),
            ))
            rejected.append({"index": i, "original_t": original_t, "reason": str(exc)})
            continue
        samples.append(SynchronizedSample(
            index=i, original_t=original_t, clock_id=clock_id,
            offset_s=model.b, drift=model.a, sync_method=model.method.value,
            sync_state=SyncState.SYNCHRONIZED, global_t=global_t,
        ))
        if global_t < 0.0:
            # Negative global time is suspicious for sensor logs
            # (epoch-relative clocks) but is NOT rejected: the global
            # timeline's zero is a convention, and a genuinely
            # pre-epoch capture is real data. Noted, not re-stamped.
            negative_time_notes.append(
                f"sample {i} maps to negative global time ({global_t})"
            )

    monotonic = stream.is_monotonic()
    if not monotonic:
        notes.append(
            "source stream was not monotonic in sensor time -- preserved as "
            "recorded; a real logging fault may exist (see source file)"
        )

    # Residual: with metadata-derived (a, b) there is no fit residual in
    # the estimation sense -- BUT estimation backends stash the measured
    # max correspondence residual in the context ("_residual_max"), and
    # that measured value belongs in the diagnostics. Metadata methods
    # leave it absent (None): they have no fit residual, and fabricating
    # one is forbidden.
    residual_max = context.metadata.get("_residual_max")
    diagnostics = SynchronizationDiagnostics(
        method=model.method.value,
        n_samples=len(samples),
        n_synchronized=sum(1 for s in samples if s.sync_state is SyncState.SYNCHRONIZED),
        n_rejected=len(rejected),
        offset_estimate_s=model.b,
        drift_estimate=model.a,
        residual_max_s=residual_max,  # measured by estimation backends; None for metadata methods
        stream_was_monotonic=monotonic,
        rejected=tuple(rejected),
        backend_attempts=tuple(attempts),
        notes=tuple(notes + negative_time_notes),
    )
    return TimeAlignment(
        clock_id=clock_id, model=model, samples=tuple(samples), diagnostics=diagnostics
    )
