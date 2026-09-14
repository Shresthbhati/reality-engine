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


#: Backend registry in the spec's preference order. Today the
#: metadata-alignment family is implemented; the open slots (GNSS/PPS,
#: trigger, signal correlation, optimization) are appended here as
#: they land, ahead of weaker methods.
DEFAULT_BACKENDS: Tuple[ISynchronizationBackend, ...] = (MetadataAlignmentBackend(),)


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

    # Residual: with metadata-derived (a, b) there is no fit residual
    # in the estimation sense; the honest observable is alignment
    # consistency of the applied mapping, which for a linear model is
    # exact by construction. Record that explicitly instead of a fake
    # residual number.
    residual_max = None
    diagnostics = SynchronizationDiagnostics(
        method=model.method.value,
        n_samples=len(samples),
        n_synchronized=sum(1 for s in samples if s.sync_state is SyncState.SYNCHRONIZED),
        n_rejected=len(rejected),
        offset_estimate_s=model.b,
        drift_estimate=model.a,
        residual_max_s=None,  # metadata methods have no fit residual; correlation backends will
        stream_was_monotonic=monotonic,
        rejected=tuple(rejected),
        backend_attempts=tuple(attempts),
        notes=tuple(notes + negative_time_notes),
    )
    return TimeAlignment(
        clock_id=clock_id, model=model, samples=tuple(samples), diagnostics=diagnostics
    )
