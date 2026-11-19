"""Execution manager: runs the engines a policy requires, sequentially unless memory is PROVEN to allow more, and reports
exactly what ran, what could not, and why.

Policies (``Policy``):
  MAXIMUM_FIDELITY  every approved engine that declares the task must run; any that cannot makes the report PARTIAL.
  ACCURATE          the single primary engine per task (lowest ``priority``); if it is unavailable the task FAILS -- a weaker
                    engine is never substituted silently.
  FAST              the cheapest available engine only; the report says in words that it is a reduced-fidelity result.

Rules that hold under every policy: an unavailable engine is reported with its reason (never skipped quietly); low memory
means waiting, then a *recorded* CPU fallback when the engine supports CPU, otherwise a FAILED run; the manager never lowers
resolution itself; transient errors are retried, other errors are not hidden.

Not implemented here (stated, not faked): remote GPU workers, a cross-run model cache, request batching.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Collection, List, Mapping, Optional, Protocol, Tuple

from engine.multiengine.contract import (
    Availability, Device, EngineAdapter, EngineErrorKind, EngineRequest, EngineRun, EngineRunStatus, EngineSpec, Task,
)
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.registry import EngineRegistry


class Policy(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"
    MAXIMUM_FIDELITY = "maximum_fidelity"


class ReportStatus(str, Enum):
    COMPLETE = "complete"      # every engine the policy required ran and succeeded
    PARTIAL = "partial"        # some required engine was unavailable or failed; the rest succeeded
    FAILED = "failed"          # nothing the policy required succeeded


class ResourceProbe(Protocol):
    """Honest resource facts. ``None`` means UNKNOWN -- the manager then runs sequentially and says it could not verify."""

    def gpu_available(self) -> Optional[bool]: ...
    def free_vram_mb(self) -> Optional[float]: ...


class UnknownResources:
    def gpu_available(self) -> Optional[bool]:
        return None

    def free_vram_mb(self) -> Optional[float]:
        return None


class TorchResourceProbe:
    """Reads ``torch.cuda`` when torch is installed; reports unknown (None) otherwise. Never guesses."""

    def _torch(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return None
        import torch  # imported only when the probe is asked

        return torch

    def gpu_available(self) -> Optional[bool]:
        t = self._torch()
        return None if t is None else bool(t.cuda.is_available())

    def free_vram_mb(self) -> Optional[float]:
        t = self._torch()
        if t is None or not t.cuda.is_available():
            return None
        free, _total = t.cuda.mem_get_info()
        return free / (1024 * 1024)


@dataclass(frozen=True)
class PlanEntry:
    engine_id: str
    state: str          # "planned" | "unavailable" | "not_approved" | "not_primary" | "not_selected"
    reason: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    task: Task
    policy: Policy
    entries: Tuple[PlanEntry, ...]
    required: Tuple[str, ...]       # engines the policy requires
    quality_claim: str

    @property
    def runnable(self) -> Tuple[str, ...]:
        return tuple(e.engine_id for e in self.entries if e.state == "planned")


@dataclass(frozen=True)
class ExecutionReport:
    task: Task
    policy: Policy
    plan: ExecutionPlan
    runs: Tuple[EngineRun, ...]
    status: ReportStatus
    quality_claim: str
    warnings: Tuple[str, ...] = ()
    memory_verification: str = ""    # what the manager could and could not verify about memory

    def observations(self):
        return tuple(o for r in self.runs for o in r.observations)

    def to_dict(self) -> dict:
        return {
            "task": self.task.value, "policy": self.policy.value, "status": self.status.value,
            "quality_claim": self.quality_claim, "warnings": list(self.warnings), "memory_verification": self.memory_verification,
            "plan": [e.__dict__ for e in self.plan.entries], "required": list(self.plan.required),
            "runs": [{"engine_id": r.engine_id, "model_version": r.model_version, "status": r.status.value, "device": r.device,
                      "error_kind": r.error_kind, "error": r.error, "attempts": r.attempts, "fallback": r.fallback,
                      "observations": len(r.observations)} for r in self.runs],
        }


class ExecutionManager:
    def __init__(self, registry: EngineRegistry, adapters: Mapping[str, EngineAdapter], *,
                 store: Optional[EvidenceStore] = None, probe: Optional[ResourceProbe] = None,
                 policy: Policy = Policy.MAXIMUM_FIDELITY, max_parallel: int = 1, max_retries: int = 2,
                 memory_wait_s: float = 0.0, sleep: Callable[[float], None] = time.sleep,
                 allow_unestablished_licence: bool = False) -> None:
        self.registry = registry
        self.adapters = dict(adapters)
        self.store = store
        self.probe = probe or UnknownResources()
        self.policy = policy
        self.max_parallel = max(1, int(max_parallel))
        self.max_retries = max(0, int(max_retries))
        self.memory_wait_s = float(memory_wait_s)
        self._sleep = sleep
        self.allow_unestablished_licence = allow_unestablished_licence

    # ------------------------------------------------------------------ planning
    def plan(self, task: Task, only: Optional[Collection[str]] = None) -> ExecutionPlan:
        """``only`` restricts the plan to the named engines (used by targeted refinement); an engine outside it is not reported."""
        entries: List[PlanEntry] = []
        approved: List[Tuple[EngineSpec, Availability]] = []
        for spec, avail in self.registry.candidates(task):
            if only is not None and spec.engine_id not in only:
                continue
            if spec.commercial_ok is False:
                entries.append(PlanEntry(spec.engine_id, "not_approved",
                                         f"licence {spec.licence or 'unknown'} is not approved for production use"))
            elif spec.commercial_ok is None and not self.allow_unestablished_licence:
                entries.append(PlanEntry(spec.engine_id, "not_approved", "licence not established in the registry"))
            else:
                approved.append((spec, avail))
        if self.policy is Policy.MAXIMUM_FIDELITY:
            chosen = approved
            required = tuple(s.engine_id for s, _ in chosen)
            claim = "every approved engine for this task was required" if required else "no approved engine exists for this task"
        elif self.policy is Policy.ACCURATE:
            chosen = approved[:1]   # the primary by priority, whether or not it is available: no silent substitution
            required = tuple(s.engine_id for s, _ in chosen)
            claim = "primary engine only (ACCURATE)"
            for s, _ in approved[1:]:
                entries.append(PlanEntry(s.engine_id, "not_primary", "ACCURATE runs the primary engine only"))
        else:
            chosen = [(s, a) for s, a in approved if a.ok][:1]
            required = tuple(s.engine_id for s, _ in chosen)
            claim = f"REDUCED FIDELITY: {len(chosen)} of {len(approved)} approved engines ran; this is not a full-fidelity result"
            chosen_ids = {s.engine_id for s, _ in chosen}
            for s, _ in approved:
                if s.engine_id not in chosen_ids:
                    entries.append(PlanEntry(s.engine_id, "not_selected", "FAST runs the cheapest available engine only"))
        for spec, avail in chosen:
            entries.append(PlanEntry(spec.engine_id, "planned") if avail.ok else PlanEntry(spec.engine_id, "unavailable", avail.reason))
        return ExecutionPlan(task, self.policy, tuple(sorted(entries, key=lambda e: e.engine_id)), required, claim)

    # ------------------------------------------------------------------ execution
    def run(self, task: Task, request: EngineRequest, *, only: Optional[Collection[str]] = None) -> ExecutionReport:
        plan = self.plan(task, only)
        warnings: List[str] = []
        if self.policy is Policy.FAST:
            warnings.append(plan.quality_claim)
        if request.max_resolution is not None:
            warnings.append(f"max_resolution={request.max_resolution} was requested: the cap is honoured and recorded in each "
                            "observation's execution_config, so this is not a native-resolution result")
        runs: List[EngineRun] = []
        for e in plan.entries:
            if e.state == "unavailable":
                spec = self.registry.get(e.engine_id)
                runs.append(EngineRun(e.engine_id, spec.version, EngineRunStatus.UNAVAILABLE,
                                      error_kind=EngineErrorKind.UNAVAILABLE.value, error=e.reason, attempts=0))
        runnable = [self.registry.get(eid) for eid in plan.runnable]
        free = self.probe.free_vram_mb()
        parallel_ok = (self.max_parallel > 1 and free is not None and len(runnable) > 1
                       and all(s.hardware.parallel_safe and s.hardware.min_vram_mb is not None for s in runnable)
                       and sum(s.hardware.min_vram_mb for s in runnable) <= free)
        if parallel_ok:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                runs.extend(pool.map(lambda s: self._run_one(s, request), runnable))
        else:
            if self.max_parallel > 1 and len(runnable) > 1:
                warnings.append("ran sequentially: parallel execution needs every engine to be parallel-safe with a measured "
                                "VRAM requirement and a probe proving they fit; at least one condition could not be verified")
            for s in runnable:
                runs.append(self._run_one(s, request))
        if self.store is not None:
            for r in runs:
                if r.ok:
                    self.store.add_many(r.observations)
        required_ok = [r for r in runs if r.engine_id in plan.required and r.ok]
        if not plan.required or not required_ok:
            status = ReportStatus.FAILED
        elif len(required_ok) == len(plan.required):
            status = ReportStatus.COMPLETE
        else:
            status = ReportStatus.PARTIAL
        gpu = self.probe.gpu_available()
        mem = ("GPU/VRAM state unknown to the manager (no probe): engines ran sequentially and memory was not verified"
               if gpu is None else f"gpu_available={gpu}, free_vram_mb={'unknown' if free is None else round(free)} at plan time")
        return ExecutionReport(task, self.policy, plan, tuple(sorted(runs, key=lambda r: r.engine_id)), status, plan.quality_claim,
                               tuple(warnings), mem)

    # ------------------------------------------------------------------ one engine
    def _choose_device(self, spec: EngineSpec) -> Tuple[Optional[Device], Optional[dict], str]:
        """(device, fallback_record, refusal_reason). A fallback is recorded only when a better device was planned and could
        not be used."""
        gpu = self.probe.gpu_available()
        if spec.hardware.gpu_required:
            if gpu is True:
                return self._wait_for_vram(spec)
            why = "none was detected" if gpu is False else "GPU availability could not be verified"
            return None, None, f"{spec.engine_id} requires a GPU and {why}"
        if gpu is True:
            return self._wait_for_vram(spec)
        return Device("cpu"), None, ""

    def _wait_for_vram(self, spec: EngineSpec) -> Tuple[Optional[Device], Optional[dict], str]:
        need = spec.hardware.min_vram_mb
        deadline = time.monotonic() + self.memory_wait_s
        while True:
            free = self.probe.free_vram_mb()
            if need is None or free is None or free >= need:
                return Device("cuda", 0), None, ""
            if time.monotonic() >= deadline:
                break
            self._sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        detail = f"needs {need:.0f} MB VRAM, {free:.0f} MB free after waiting {self.memory_wait_s:g}s"
        if spec.hardware.cpu_supported and not spec.hardware.gpu_required:
            return Device("cpu"), {"reason": detail, "from": "cuda:0", "to": "cpu"}, ""
        return None, None, f"{spec.engine_id} {detail}, and has no CPU path"

    def _run_one(self, spec: EngineSpec, request: EngineRequest) -> EngineRun:
        adapter = self.adapters.get(spec.engine_id)
        if adapter is None:
            return EngineRun(spec.engine_id, spec.version, EngineRunStatus.UNAVAILABLE, error_kind=EngineErrorKind.UNAVAILABLE.value,
                             error=f"no adapter is implemented for {spec.engine_id!r} (integration={spec.integration})", attempts=0)
        device, fallback, refusal = self._choose_device(spec)
        if device is None:
            kind = EngineErrorKind.OUT_OF_MEMORY if "VRAM" in refusal else EngineErrorKind.UNAVAILABLE
            return EngineRun(spec.engine_id, spec.version, EngineRunStatus.FAILED, error_kind=kind.value, error=refusal, attempts=0)
        attempts = 0
        oom_retried = False
        while True:
            attempts += 1
            run = adapter.run(request, device, fallback=fallback, attempts=attempts)
            if run.ok or run.status is EngineRunStatus.UNAVAILABLE:
                return run
            if run.error_kind == EngineErrorKind.TRANSIENT.value and attempts <= self.max_retries:
                self._sleep(min(2.0 ** (attempts - 1), 8.0))
                continue
            if run.error_kind == EngineErrorKind.OUT_OF_MEMORY.value:
                if not oom_retried:           # one sequential retry after the adapter has unloaded
                    oom_retried = True
                    continue
                if device.kind != "cpu" and spec.hardware.cpu_supported:
                    device = Device("cpu")
                    fallback = {"reason": "out of memory on the GPU after a retry", "from": "cuda:0", "to": "cpu"}
                    continue
            return run
