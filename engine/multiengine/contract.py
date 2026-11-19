"""The engine and adapter contract.

``EngineSpec`` is the declarative record of an engine (what it is, what it needs, what it is allowed to do, how to verify it
is really available). ``EngineAdapter`` is the executable side: a template ``run()`` that validates input, checks
availability, loads, infers, stamps observations with the facts the *runner* knows, records resources, and always unloads.

Model-specific behaviour lives in subclasses; nothing here assumes two models share inputs or output conventions beyond
"produces ``ObservationDraft``s in a stated coordinate frame".
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from engine.multiengine.observation import EngineObservation, ObservationDraft, make_observation


class Task(str, Enum):
    SFM = "sfm"
    MVS = "mvs"
    DEPTH = "depth"
    FEATURE_MATCHING = "feature_matching"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    SCENE_UNDERSTANDING = "scene_understanding"
    LAYOUT = "layout"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Availability:
    """The outcome of a REAL check. ``evidence`` lists what was verified (dependency importable, checkpoint file present and
    hash-checked, binary on PATH, ...); an AVAILABLE answer with no evidence is rejected."""

    status: AvailabilityStatus
    reason: str
    remediation: str = ""
    evidence: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.status is AvailabilityStatus.AVAILABLE and not self.evidence:
            raise ValueError("AVAILABLE requires evidence of what was verified (an import alone is not evidence of a usable model)")
        if self.status is AvailabilityStatus.BLOCKED and not self.reason:
            raise ValueError("a BLOCKED engine must say why")

    @property
    def ok(self) -> bool:
        return self.status is AvailabilityStatus.AVAILABLE


@dataclass(frozen=True)
class CheckpointRequirement:
    name: str
    path: Optional[str] = None            # where it should be; None = acquired by the framework's own cache
    sha256: Optional[str] = None          # None = no pinned hash (stated, not assumed)
    redistributable: bool = False


@dataclass(frozen=True)
class HardwareRequirement:
    gpu_required: bool = False
    cpu_supported: bool = True
    min_vram_mb: Optional[float] = None   # None = NOT MEASURED (never an estimate dressed up as a number)
    min_ram_mb: Optional[float] = None
    measured_source: str = ""             # where min_* came from (a benchmark record path); "" when None
    parallel_safe: bool = False           # may run concurrently with other engines on one device


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    name: str
    version: str
    tasks: FrozenSet[Task]
    input_schema: str
    output_schema: str
    probe: Callable[[], Availability]
    family: str = ""                       # shared base model / training-data lineage; engines in one family are NOT independent
    dependencies: Tuple[str, ...] = ()
    checkpoints: Tuple[CheckpointRequirement, ...] = ()
    hardware: HardwareRequirement = HardwareRequirement()
    licence: str = ""
    commercial_ok: Optional[bool] = None   # None = not established
    redistribution: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()
    tests: Tuple[str, ...] = ()
    benchmarks: Tuple[str, ...] = ()       # paths to benchmark records; empty = none exist
    priority: int = 100                    # lower runs first under ACCURATE (the single primary engine per task)
    integration: str = "registry-only"     # "registry-only" | "adapter" | "wired" -- how far the repo actually integrates it

    def __post_init__(self):
        if not self.engine_id or not self.version:
            raise ValueError("an engine needs an id and a model version")
        if not self.tasks:
            raise ValueError(f"engine {self.engine_id!r} declares no task")
        if self.integration not in ("registry-only", "adapter", "wired"):
            raise ValueError(f"unknown integration level {self.integration!r}")
        if self.hardware.min_vram_mb is not None and not self.hardware.measured_source:
            raise ValueError(f"engine {self.engine_id!r}: a VRAM requirement must cite the measurement it came from")

    def to_dict(self, availability: Optional[Availability] = None) -> dict:
        a = availability
        return {
            "engine_id": self.engine_id, "name": self.name, "version": self.version, "tasks": sorted(t.value for t in self.tasks),
            "input_schema": self.input_schema, "output_schema": self.output_schema, "family": self.family,
            "dependencies": list(self.dependencies),
            "checkpoints": [c.__dict__ for c in self.checkpoints],
            "hardware": self.hardware.__dict__, "licence": self.licence, "commercial_ok": self.commercial_ok,
            "redistribution": self.redistribution, "limitations": list(self.limitations), "tests": list(self.tests),
            "benchmarks": list(self.benchmarks), "integration": self.integration,
            "availability": None if a is None else {
                "status": a.status.value, "reason": a.reason, "remediation": a.remediation, "evidence": list(a.evidence)},
        }


# ------------------------------------------------------------------ execution-time types

class EngineErrorKind(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE = "unavailable"
    MISSING_CHECKPOINT = "missing_checkpoint"
    OUT_OF_MEMORY = "out_of_memory"
    TRANSIENT = "transient"
    MODEL_ERROR = "model_error"


class EngineError(Exception):
    def __init__(self, kind: EngineErrorKind, message: str, *, transient: bool = False):
        super().__init__(message)
        self.kind = kind
        self.transient = transient or kind is EngineErrorKind.TRANSIENT


class EngineRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Device:
    kind: str = "cpu"                      # "cpu" | "cuda"
    index: int = 0

    def __str__(self) -> str:
        return self.kind if self.kind == "cpu" else f"{self.kind}:{self.index}"


@dataclass(frozen=True)
class EngineRequest:
    sources: Tuple[str, ...]               # image / video ids
    payload: Mapping[str, Any] = field(default_factory=dict)
    max_resolution: Optional[int] = None   # long-side cap in px; None = native (the default under ACCURATE / MAXIMUM_FIDELITY)
    seed: Optional[int] = None


@dataclass(frozen=True)
class ResourceUsage:
    wall_s: float
    peak_vram_mb: Optional[float] = None   # None = not measured
    peak_ram_mb: Optional[float] = None


@dataclass(frozen=True)
class EngineRun:
    engine_id: str
    model_version: str
    status: EngineRunStatus
    observations: Tuple[EngineObservation, ...] = ()
    device: Optional[str] = None
    error_kind: Optional[str] = None
    error: str = ""
    usage: Optional[ResourceUsage] = None
    fallback: Optional[Mapping[str, Any]] = None
    reproducibility: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status is EngineRunStatus.SUCCEEDED


class EngineAdapter(ABC):
    """Executable side of an engine. Subclasses implement load/unload/infer/validate_input; ``run`` is the only entry point
    the execution manager uses, so the lifecycle guarantees (always unload, always stamp) hold for every engine."""

    spec: EngineSpec

    def check_available(self) -> Availability:
        return self.spec.probe()

    @abstractmethod
    def validate_input(self, request: EngineRequest) -> None:
        """Raise ``EngineError(INVALID_INPUT, ...)`` for a request this engine cannot honestly process."""

    @abstractmethod
    def load(self, device: Device) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]: ...

    def measured_usage(self) -> Tuple[Optional[float], Optional[float]]:
        """(peak_vram_mb, peak_ram_mb) as actually measured by the adapter, or None. Never an estimate."""
        return None, None

    def reproducibility(self, request: EngineRequest) -> Dict[str, Any]:
        return {"engine_id": self.spec.engine_id, "model_version": self.spec.version, "seed": request.seed,
                "config": dict(self.spec.config)}

    def run(self, request: EngineRequest, device: Device, *, fallback: Optional[Mapping[str, Any]] = None,
            attempts: int = 1) -> EngineRun:
        s = self.spec
        avail = self.check_available()
        if not avail.ok:
            kind = EngineErrorKind.MISSING_CHECKPOINT if "checkpoint" in avail.reason.lower() else EngineErrorKind.UNAVAILABLE
            return EngineRun(s.engine_id, s.version, EngineRunStatus.UNAVAILABLE, error_kind=kind.value, error=avail.reason,
                             attempts=attempts)
        try:
            self.validate_input(request)
        except EngineError as exc:
            return EngineRun(s.engine_id, s.version, EngineRunStatus.FAILED, error_kind=exc.kind.value, error=str(exc),
                             attempts=attempts)
        t0 = time.monotonic()
        loaded = False
        try:
            self.load(device)
            loaded = True
            drafts = list(self.infer(request))
            cfg = {**dict(s.config), "device": str(device), "max_resolution": request.max_resolution, "seed": request.seed}
            obs = tuple(make_observation(d, engine_id=s.engine_id, model_version=s.version, execution_config=cfg,
                                         fallback=fallback) for d in drafts)
            vram, ram = self.measured_usage()
            return EngineRun(s.engine_id, s.version, EngineRunStatus.SUCCEEDED, observations=obs, device=str(device),
                             usage=ResourceUsage(time.monotonic() - t0, vram, ram), fallback=fallback,
                             reproducibility=self.reproducibility(request), attempts=attempts)
        except EngineError as exc:
            return EngineRun(s.engine_id, s.version, EngineRunStatus.FAILED, device=str(device), error_kind=exc.kind.value,
                             error=str(exc), usage=ResourceUsage(time.monotonic() - t0), fallback=fallback, attempts=attempts)
        except MemoryError as exc:
            return EngineRun(s.engine_id, s.version, EngineRunStatus.FAILED, device=str(device),
                             error_kind=EngineErrorKind.OUT_OF_MEMORY.value, error=str(exc) or "out of memory",
                             usage=ResourceUsage(time.monotonic() - t0), fallback=fallback, attempts=attempts)
        except Exception as exc:  # noqa: BLE001 -- a model crash is a FAILED run with the real error, never a success
            return EngineRun(s.engine_id, s.version, EngineRunStatus.FAILED, device=str(device),
                             error_kind=EngineErrorKind.MODEL_ERROR.value, error=f"{type(exc).__name__}: {exc}",
                             usage=ResourceUsage(time.monotonic() - t0), fallback=fallback, attempts=attempts)
        finally:
            if loaded:
                try:
                    self.unload()
                except Exception:  # noqa: BLE001 -- an unload failure must not mask the run's own outcome
                    pass
