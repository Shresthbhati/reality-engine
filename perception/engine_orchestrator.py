"""Multi-Engine Orchestrator — runs complementary engines together with
GPU/CPU resource scheduling and cross-engine intelligence.

Architecture:
  1. ``EngineSpec`` — metadata for every registered engine (VRAM estimate,
     dependencies, licence, priority, availability probe, timeout)
  2. ``MultiEngineOrchestrator`` — dependency-ordered, priority-ordered,
     availability-checked, VRAM-aware execution with cancellation, per-engine
     timeouts and retry of transient failures
  3. ``EngineExecution`` — the execution contract: every engine reports its
     identity, model identity, status, runtime, memory, device, coordinate
     frames, units and provenance, whether it succeeded, failed, timed out,
     was cancelled, was skipped for an unmet dependency or was unavailable
  4. ``merge_engine_outputs`` — merges ``UnifiedEngineOutput`` from all engines
     into duplicate handling, contradiction reports and agreement counts

Invariants:
  * An engine whose ``run_condition`` reports a reason is NEVER executed; it is
    reported ``unavailable`` with that reason. Nothing is substituted for it.
  * VRAM is only gated when CUDA is actually present. On a CPU-only machine an
    engine is not deferred for "0 MB free VRAM" — it runs on CPU (or fails
    honestly if it cannot).
  * An engine whose dependency did not succeed is ``skipped`` with
    ``dependency_unmet``; it is never run against missing inputs.
  * A result is only ``succeeded``/``partial`` when it is a well-formed
    ``UnifiedEngineOutput``; malformed output is ``invalid_output``.

The orchestrator is designed to be called from the production pipeline:
``run_multi_engine(image_paths, ...) -> MultiEngineReport`` where
``MultiEngineReport`` is JSON-serialisable for the job report and the
UnifiedEvidenceStore (persistence layer).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
    bbox_iou,
    compare_observations,
    compute_observation_hash,
)

log = logging.getLogger("reality.engine_orchestrator")


@dataclass(frozen=True)
class EngineSpec:
    """Metadata for a registered engine."""
    engine_id: str
    architecture_id: str
    checkpoint_id: str = "none"
    version: str = "unknown"
    vram_mb: int = 1024           # estimated peak VRAM for this engine when run on GPU
    cpu_threads: int = 0          # 0 = uses default
    priority: int = 50            # higher runs first; default medium
    required_for_production: bool = False
    license: str = "unknown"
    non_commercial: bool = False
    dependencies: Tuple[str, ...] = ()  # engine_ids this depends on
    run_condition: Optional[Callable[[], str]] = None  # returns error or None
    #: False for engines that cannot run without CUDA (their adapter only has a GPU path).
    cpu_capable: bool = True
    #: 0 = no timeout. A timed-out engine is abandoned (see EngineExecution note).
    timeout_s: float = 0.0
    #: How long the orchestrator waits for a TIMED-OUT engine's call to actually return
    #: before it starts the next engine. Python cannot kill a thread, so an abandoned GPU
    #: engine would otherwise keep its allocations on the device while the next engine
    #: loads on top of it. The wait makes the device hand-over sequential; if the call is
    #: still running after this, further GPU engines are refused (``gpu_quarantined``)
    #: instead of silently sharing the device with it.
    drain_after_timeout_s: float = 60.0
    #: How many attempts a TRANSIENT failure (I/O, connection, timeout) gets.
    max_attempts: int = 2


@dataclass(frozen=True)
class EngineExecution:
    """The execution contract for one engine run.

    ``status`` is one of ``succeeded``, ``partial``, ``failed``, ``timed_out``,
    ``cancelled``, ``unavailable``, ``skipped``, ``invalid_output``.
    """
    engine_id: str
    architecture_id: str
    checkpoint_id: str = "none"
    version: str = "unknown"
    license: str = "unknown"
    non_commercial: bool = False
    status: str = "failed"
    error: str = ""
    device: str = "unknown"        # "cuda" | "cpu" | "unknown"
    attempts: int = 0
    wall_time_s: float = 0.0
    peak_vram_mb: Optional[float] = None
    n_observations: int = 0
    frame_ids: Tuple[str, ...] = ()
    scale_states: Tuple[str, ...] = ()
    units: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""               # why it did not run, when status is unavailable/skipped/cancelled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "architecture_id": self.architecture_id,
            "checkpoint_id": self.checkpoint_id,
            "version": self.version,
            "license": self.license,
            "non_commercial": self.non_commercial,
            "status": self.status,
            "error": self.error,
            "device": self.device,
            "attempts": self.attempts,
            "wall_time_s": round(self.wall_time_s, 3),
            "peak_vram_mb": self.peak_vram_mb,
            "n_observations": self.n_observations,
            "frame_ids": list(self.frame_ids),
            "scale_states": list(self.scale_states),
            "units": list(self.units),
            "provenance": dict(self.provenance),
            "reason": self.reason,
        }


@dataclass
class MultiEngineReport:
    """Top-level result of the multi-engine pipeline run."""
    run_id: str
    image_count: int
    total_wall_s: float
    engines_run: Dict[str, str]      # engine_id -> status
    observations: List[ObservationRecord]
    contradictions: List[Dict[str, Any]]
    unavailable: List[Dict[str, Any]]  # engine_id, reason
    vram_peak_mb: Optional[float]
    cross_engine_summary: Dict[str, Any]
    #: Full per-engine execution contract (every engine, including failures).
    engine_executions: List[EngineExecution] = field(default_factory=list)
    #: Observations dropped as content-identical repeats (not counted as extra evidence).
    duplicate_observations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "image_count": self.image_count,
            "total_wall_s": round(self.total_wall_s, 3),
            "engines_run": self.engines_run,
            "n_observations": len(self.observations),
            "n_contradictions": len(self.contradictions),
            "duplicate_observations": self.duplicate_observations,
            "unavailable": self.unavailable,
            "vram_peak_mb": self.vram_peak_mb,
            "cross_engine_summary": dict(self.cross_engine_summary),
            "contradictions": self.contradictions,
            "engine_executions": [e.to_dict() for e in self.engine_executions],
        }


# ---------------------------------------------------------------------------
# Engine Registry
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, EngineSpec] = {}
_ENGINE_FNS: Dict[str, Callable[..., UnifiedEngineOutput]] = {}
_REGISTRY_LOCK = threading.RLock()


def register_engine(spec: EngineSpec, fn: Callable[..., UnifiedEngineOutput]) -> None:
    """Register an engine with its spec and runner function (idempotent by engine_id)."""
    with _REGISTRY_LOCK:
        _ENGINES[spec.engine_id] = spec
        _ENGINE_FNS[spec.engine_id] = fn


def unregister_engine(engine_id: str) -> None:
    """Remove an engine from the registry (test isolation / plugin unload)."""
    with _REGISTRY_LOCK:
        _ENGINES.pop(engine_id, None)
        _ENGINE_FNS.pop(engine_id, None)


def list_engines() -> List[EngineSpec]:
    """All registered engines, sorted by priority (highest first)."""
    with _REGISTRY_LOCK:
        specs = list(_ENGINES.values())
    return sorted(specs, key=lambda s: -s.priority)


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _active_device() -> str:
    return "cuda" if _cuda_available() else "cpu"


def _availability_reason(spec: EngineSpec) -> Optional[str]:
    """Why this engine cannot run here, or None when it can. Never raises."""
    try:
        reason = spec.run_condition() if spec.run_condition else None
    except Exception as exc:  # a broken probe is an unavailable engine, not a crashed run
        return f"availability probe raised {type(exc).__name__}: {exc}"
    if reason:
        return str(reason)
    if not _cuda_available() and not spec.cpu_capable:
        return "needs CUDA: this engine has no CPU execution path"
    return None


def list_available_engines() -> List[EngineSpec]:
    """Engines that are runnable on the current system (probe passes, device available)."""
    return [spec for spec in list_engines() if _availability_reason(spec) is None]


def get_unavailable_engines() -> List[Dict[str, Any]]:
    """Engines that are registered but not runnable, with the reason from their own probe."""
    out = []
    for spec in list_engines():
        reason = _availability_reason(spec)
        if reason is not None:
            out.append({"engine_id": spec.engine_id, "reason": reason})
    return out


def _get_free_vram_mb() -> float:
    """Free VRAM in MB (0 if no CUDA)."""
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _get_vram_peak_mb() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _torch_empty_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Execution helpers — timeout, retry, validation
# ---------------------------------------------------------------------------

_TRANSIENT_ERRORS = (OSError, TimeoutError, ConnectionError)


def _is_transient(exc: BaseException) -> bool:
    """Only failures that may plausibly succeed on a retry.

    A missing model, a bad image or an invalid configuration is NOT transient:
    retrying it would double the runtime and still fail.
    """
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    text = str(exc).lower()
    return "timed out" in text or "temporarily unavailable" in text


def _call_with_timeout(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any],
                       timeout_s: float) -> Tuple[Any, Optional[BaseException], bool, Optional[threading.Thread]]:
    """Run ``fn`` in a daemon thread, waiting at most ``timeout_s``.

    Returns ``(result, error, timed_out, thread)``. ``thread`` is the live call's
    thread when the wait timed out (else ``None``), so the caller can wait for it
    before starting another engine: Python cannot interrupt a running call, so a
    timed-out engine is *abandoned*, not killed. The execution record says
    ``timed_out`` and its result is discarded; that limitation is recorded rather
    than hidden.
    """
    if timeout_s <= 0:
        try:
            return fn(*args, **kwargs), None, False, None
        except BaseException as exc:  # noqa: BLE001 -- reported, never swallowed
            return None, exc, False, None

    box: "queue.Queue[Tuple[Any, Optional[BaseException]]]" = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            box.put((fn(*args, **kwargs), None))
        except BaseException as exc:  # noqa: BLE001
            box.put((None, exc))

    thread = threading.Thread(target=_target, name="engine-call", daemon=True)
    thread.start()
    try:
        result, error = box.get(timeout=timeout_s)
    except queue.Empty:
        return None, None, True, thread
    return result, error, False, thread


def _drain_abandoned(thread: Optional[threading.Thread], timeout_s: float) -> Tuple[bool, float]:
    """Wait for an abandoned engine call to return. Returns ``(returned, waited_s)``.

    This is the device hand-over: the orchestrator will not start the next engine
    (and therefore cannot overlap its allocations) until the abandoned call has
    stopped touching the device, or until ``timeout_s`` expires and the call is
    declared still-running.
    """
    t0 = time.perf_counter()
    if thread is None or not thread.is_alive():
        return True, 0.0
    thread.join(timeout=max(0.0, timeout_s))
    return (not thread.is_alive()), time.perf_counter() - t0


def _validate_output(out: Any) -> Optional[str]:
    """Reason the engine output is not usable as evidence, or None when it is.

    "No engine is marked successful without a valid output": a success status
    with a malformed observation is downgraded, never trusted.
    """
    if not isinstance(out, UnifiedEngineOutput):
        return f"engine returned {type(out).__name__}, not UnifiedEngineOutput"
    if not out.engine_id:
        return "engine output has no engine_id"
    if out.status not in ("succeeded", "partial", "failed", "unavailable", "cancelled"):
        return f"unknown status {out.status!r}"
    for i, obs in enumerate(out.observations):
        if not isinstance(obs, ObservationRecord):
            return f"observation {i} is {type(obs).__name__}, not ObservationRecord"
        if not obs.evidence_id:
            return f"observation {i} has no evidence_id"
        if not obs.source_image_id:
            return f"observation {i} has no source_image_id"
        if not isinstance(obs.coordinate_frame, CoordinateFrame):
            return f"observation {i} has no coordinate frame"
    if out.status in ("succeeded", "partial") and not out.observations:
        return "engine reported success with no observations"
    return None


def _frames_of(observations: Sequence[ObservationRecord]) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    frame_ids = tuple(sorted({o.coordinate_frame.frame_id for o in observations}))
    scale_states = tuple(sorted({o.coordinate_frame.scale_state for o in observations}))
    units = tuple(sorted({o.coordinate_frame.units for o in observations}))
    return frame_ids, scale_states, units


# ---------------------------------------------------------------------------
# Intelligence Merger — cross-engine comparison and contradiction detection
# ---------------------------------------------------------------------------

#: Minimum bbox IoU for two observations from DIFFERENT engines on the SAME image
#: to be treated as the same target. Below this they are different objects and
#: comparing their labels would invent contradictions.
SAME_TARGET_MIN_IOU = 0.5


def _deduplicate(observations: Sequence[ObservationRecord]) -> Tuple[List[ObservationRecord], int]:
    """Drop content-identical repeats. The first-seen record is kept (stable for reproducibility)."""
    seen: Dict[str, int] = {}
    kept: List[ObservationRecord] = []
    for obs in observations:
        h = compute_observation_hash(obs)
        if h in seen:
            continue
        seen[h] = len(kept)
        kept.append(obs)
    return kept, len(observations) - len(kept)


def merge_engine_outputs(outputs: Sequence[UnifiedEngineOutput]) -> MultiEngineReport:
    """Merge multiple engine outputs into a single MultiEngineReport.

    Contradictions are only looked for between observations that can plausibly be
    the same target: the same source image AND overlapping bounding boxes. Two
    engines looking at different objects on one image are not in disagreement, and
    reading their labels as a conflict would invent evidence.
    """
    counted = [o for out in outputs for o in out.observations if out.status in ("succeeded", "partial")]
    all_obs, n_duplicates = _deduplicate(counted)

    all_contradictions: List[Dict[str, Any]] = []
    comparability_notes: List[Dict[str, Any]] = []
    agreements: List[Dict[str, Any]] = []

    by_image: Dict[str, List[ObservationRecord]] = {}
    for obs in all_obs:
        by_image.setdefault(obs.source_image_id, []).append(obs)

    for image_id, obs_list in by_image.items():
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                a, b = obs_list[i], obs_list[j]
                if a.model_id.architecture_id == b.model_id.architecture_id:
                    continue  # same engine twice is not cross-engine evidence
                iou = bbox_iou(a.bbox_px, b.bbox_px)
                if iou < SAME_TARGET_MIN_IOU:
                    continue  # different (or unplaced) targets: no claim either way
                issues = compare_observations(a, b)
                if not issues:
                    if a.label.lower() == b.label.lower():
                        agreements.append({
                            "image_id": image_id, "target_iou": round(iou, 3),
                            "label": a.label,
                            "architecture_a": a.model_id.architecture_id,
                            "architecture_b": b.model_id.architecture_id,
                        })
                    continue
                for issue in issues:
                    row = {
                        "image_id": image_id,
                        "target_iou": round(iou, 3),
                        "evidence_a": a.evidence_id,
                        "evidence_b": b.evidence_id,
                        "model_a": a.model_id.architecture_id,
                        "model_b": b.model_id.architecture_id,
                        "issue": issue,
                    }
                    (comparability_notes if issue.startswith("scale_state_mismatch") else all_contradictions).append(row)

    engine_stats: Dict[str, Dict[str, Any]] = {}
    for out in outputs:
        engine_stats[out.engine_id] = {
            "status": out.status,
            "n_observations": out.n_observations,
            "wall_time_s": round(out.wall_time_s, 3),
            "peak_vram_mb": out.peak_vram_mb,
            "error": out.error_message,
        }

    cross_summary = {
        "engines_run": sorted({o.engine_id for o in outputs}),
        "engines_by_status": dict(sorted(engine_stats.items())),
        "total_observations": len(all_obs),
        "duplicate_observations": n_duplicates,
        "n_contradictions": len(all_contradictions),
        "n_agreements": len(agreements),
        "n_comparability_notes": len(comparability_notes),
        "top_contradiction_types": _top_contradiction_types(all_contradictions),
        "agreements": agreements[:50],
        "comparability_notes": comparability_notes[:50],
    }

    total_wall = sum(o.wall_time_s for o in outputs)
    return MultiEngineReport(
        run_id=outputs[0].run_id if outputs else "",
        image_count=len({o.source_image_id for o in all_obs}),
        total_wall_s=total_wall,
        engines_run={o.engine_id: o.status for o in outputs},
        observations=all_obs,
        contradictions=all_contradictions,
        unavailable=[],
        vram_peak_mb=_get_vram_peak_mb(),
        cross_engine_summary=cross_summary,
        duplicate_observations=n_duplicates,
    )


def _top_contradiction_types(contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Top contradiction types by frequency."""
    from collections import Counter

    counts = Counter(str(c["issue"]).split(":")[0] for c in contradictions)
    return [{"type": k, "count": v} for k, v in counts.most_common(5)]


# ---------------------------------------------------------------------------
# MultiEngineOrchestrator — dependency- and VRAM-aware scheduling
# ---------------------------------------------------------------------------

class MultiEngineOrchestrator:
    """Runs all registered engines, respecting availability, dependencies,
    priority, VRAM and cancellation.

    Engines run in dependency order, priority order within the ready set. An
    engine whose dependency did not succeed is skipped. VRAM is checked only
    when CUDA is present; a deferred engine gets one retry after the GPU cache is
    released. An engine that cannot run here is recorded ``unavailable`` with its
    own probe's reason and never executed.
    """

    def __init__(self, vram_headroom_mb: int = 512):
        self._vram_headroom = vram_headroom_mb
        self._wave_results: List[UnifiedEngineOutput] = []
        self._deferred: List[EngineSpec] = []
        self._executions: List[EngineExecution] = []
        self._unavailable: List[Dict[str, Any]] = []
        #: engine_id -> the still-running thread of a timed-out call that has NOT
        #: returned. While this is non-empty, no further GPU engine is started.
        self._gpu_holders: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def _prune_released_devices(self) -> List[str]:
        """Drop holders whose abandoned call has since returned (the device is free again)."""
        released: List[str] = []
        with self._lock:
            for engine_id, thread in list(self._gpu_holders.items()):
                if not thread.is_alive():
                    self._gpu_holders.pop(engine_id, None)
                    released.append(engine_id)
        if released:
            log.info("abandoned engine(s) %s returned; the device is free again", ", ".join(released))
        return released

    def _gpu_blocked_reason(self, spec: EngineSpec) -> Optional[str]:
        """Why this engine may not start on the GPU now, or None.

        An engine deferred here is NOT reported as if it ran: it is ``unavailable``
        with the holder named, so the report says which engine kept the device.
        """
        self._prune_released_devices()
        if not _cuda_available() or spec.vram_mb <= 0:
            return None
        with self._lock:
            holders = sorted(self._gpu_holders)
        if not holders:
            return None
        return (f"gpu_quarantined: {', '.join(holders)} timed out and its call was still running after the "
                f"drain wait; starting this engine would share the device with it")

    # -- scheduling
    def _compute_vram_budget(self) -> Optional[float]:
        """VRAM available for the next engine, or None when there is no GPU to gate on."""
        if not _cuda_available():
            return None
        total = _get_free_vram_mb()
        return total - self._vram_headroom if total > self._vram_headroom else 0.0

    @staticmethod
    def _order(specs: Sequence[EngineSpec]) -> Tuple[List[EngineSpec], List[Tuple[str, str]]]:
        """Kahn topological order, priority tie-break. Returns (order, unmet_edges).

        An unmet edge means a dependency that is not in this run's set; the
        dependent engine is still ordered, and the executor decides whether to
        skip it based on the dependency's actual status.
        """
        by_id = {s.engine_id: s for s in specs}
        indegree = {s.engine_id: 0 for s in specs}
        unmet: List[Tuple[str, str]] = []
        for s in specs:
            for dep in s.dependencies:
                if dep in by_id:
                    indegree[s.engine_id] += 1
                else:
                    unmet.append((s.engine_id, dep))
        ready = [s for s in specs if indegree[s.engine_id] == 0]
        out: List[EngineSpec] = []
        while ready:
            ready.sort(key=lambda s: -s.priority)
            current = ready.pop(0)
            out.append(current)
            for s in specs:
                if current.engine_id in s.dependencies and s.engine_id in indegree:
                    indegree[s.engine_id] -= 1
                    if indegree[s.engine_id] == 0:
                        ready.append(s)
        if len(out) != len(specs):  # a cycle: report it, run nothing twice
            remaining = [s for s in specs if s not in out]
            log.warning("circular engine dependencies among %s; appending in priority order",
                        [s.engine_id for s in remaining])
            out.extend(sorted(remaining, key=lambda s: -s.priority))
        return out, unmet

    @staticmethod
    def _should_cancel(context: Dict[str, Any]) -> bool:
        fn = context.get("should_cancel")
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:  # a broken cancel probe must not crash the run
            return False

    def _execute_one(self, spec: EngineSpec, image_paths: Sequence[str],
                     context: Dict[str, Any]) -> Tuple[UnifiedEngineOutput, EngineExecution]:
        fn = _ENGINE_FNS[spec.engine_id]
        device = _active_device()
        attempts = 0
        last_error = ""
        while attempts < max(1, spec.max_attempts):
            attempts += 1
            t0 = time.perf_counter()
            result, error, timed_out, thread = _call_with_timeout(fn, (image_paths,), dict(context), spec.timeout_s)
            elapsed = time.perf_counter() - t0
            if timed_out:
                # The engine was abandoned at its timeout. The next engine must not start
                # while this call may still hold VRAM, so wait for it to return first.
                drained, drain_wait = _drain_abandoned(thread, spec.drain_after_timeout_s)
                _torch_empty_cache()
                if not drained and thread is not None:
                    with self._lock:
                        self._gpu_holders[spec.engine_id] = thread
                held = elapsed + drain_wait
                return (UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(context.get("run_id", "")),
                                            observations=(), status="timed_out",
                                            error_message=(f"engine exceeded its {spec.timeout_s:g}s timeout and was abandoned"
                                                           + ("" if drained else "; its call was still running after the "
                                                              f"{spec.drain_after_timeout_s:g}s drain wait")),
                                            wall_time_s=held),
                        EngineExecution(
                            engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                            checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                            non_commercial=spec.non_commercial, status="timed_out",
                            error=f"exceeded {spec.timeout_s:g}s timeout; the call cannot be interrupted in Python and was abandoned",
                            device=device, attempts=attempts, wall_time_s=held,
                            reason="timeout",
                            provenance={
                                "timeout_s": spec.timeout_s,
                                "abandoned_thread_drained": drained,
                                "drain_wait_s": round(drain_wait, 3),
                                "device_release": ("device released: the abandoned call returned before the next engine started"
                                                   if drained else
                                                   "device NOT released: the abandoned call was still running; further GPU engines "
                                                   "are refused until it returns (gpu_quarantined)"),
                            }))
            if error is None:
                invalid = _validate_output(result)
                if invalid is not None:
                    return (UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(context.get("run_id", "")),
                                                observations=(), status="invalid_output", error_message=invalid,
                                                wall_time_s=elapsed),
                            EngineExecution(
                                engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                                checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                                non_commercial=spec.non_commercial, status="invalid_output", error=invalid,
                                device=device, attempts=attempts, wall_time_s=elapsed,
                                reason="invalid output"))
                assert isinstance(result, UnifiedEngineOutput)
                frame_ids, scale_states, units = _frames_of(result.observations)
                stamped = UnifiedEngineOutput(
                    engine_id=result.engine_id or spec.engine_id, run_id=result.run_id,
                    observations=result.observations, status=result.status,
                    error_message=result.error_message, wall_time_s=elapsed,
                    peak_vram_mb=_get_vram_peak_mb(), summary=result.summary,
                )
                return stamped, EngineExecution(
                    engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                    checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                    non_commercial=spec.non_commercial, status=result.status,
                    error=result.error_message, device=device, attempts=attempts,
                    wall_time_s=elapsed, peak_vram_mb=_get_vram_peak_mb(),
                    n_observations=len(result.observations), frame_ids=frame_ids,
                    scale_states=scale_states, units=units,
                    provenance={"registered_engine_id": spec.engine_id,
                                "adapter_engine_id": result.engine_id,
                                "required_for_production": spec.required_for_production})
            last_error = f"{type(error).__name__}: {error}"
            if not _is_transient(error) or attempts >= max(1, spec.max_attempts):
                break
            log.info("engine %s transient failure, retrying (attempt %d): %s", spec.engine_id, attempts, last_error)

        elapsed = time.perf_counter() - t0
        return (UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(context.get("run_id", "")),
                                    observations=(), status="failed", error_message=last_error,
                                    wall_time_s=elapsed),
                EngineExecution(engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                                checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                                non_commercial=spec.non_commercial, status="failed", error=last_error,
                                device=device, attempts=attempts, wall_time_s=elapsed,
                                reason="exception"))

    def run_wave(self, image_paths: Sequence[str] = (), context: Optional[Dict[str, Any]] = None,
                 engine_ids: Optional[Sequence[str]] = None) -> List[UnifiedEngineOutput]:
        """Run the registered engines: dependency order, priority order, VRAM-aware, cancellable."""
        with self._lock:
            caller_context = dict(context or {})  # never mutate the caller's dict (per-job isolation)
            with _REGISTRY_LOCK:
                specs = [_ENGINES[e] for e in _ENGINES]
            if engine_ids is not None:
                wanted = set(engine_ids)
                specs = [s for s in specs if s.engine_id in wanted]

            results: List[UnifiedEngineOutput] = []
            executions: List[EngineExecution] = []
            unavailable: List[Dict[str, Any]] = []

            runnable: List[EngineSpec] = []
            for spec in specs:
                reason = _availability_reason(spec)
                if reason is None:
                    runnable.append(spec)
                else:
                    unavailable.append({"engine_id": spec.engine_id, "reason": reason})
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="unavailable", error_message=reason))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="unavailable", error=reason,
                        device=_active_device(), reason="unavailable"))

            ordered, unmet_edges = self._order(runnable)
            unmet_by_engine: Dict[str, List[str]] = {}
            for engine_id, dep in unmet_edges:
                unmet_by_engine.setdefault(engine_id, []).append(dep)

            statuses: Dict[str, str] = {s.engine_id: s.status for s in executions}
            deferred: List[EngineSpec] = []
            cancelled = False
            _torch_empty_cache()

            for spec in ordered:
                if cancelled or self._should_cancel(caller_context):
                    cancelled = True
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="cancelled",
                                                       error_message="run cancelled by the caller before this engine started"))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="cancelled", device=_active_device(),
                        reason="cancelled"))
                    statuses[spec.engine_id] = "cancelled"
                    continue
                unmet = list(unmet_by_engine.get(spec.engine_id, []))
                unmet += [d for d in spec.dependencies
                          if d in statuses and statuses[d] not in ("succeeded", "partial")]
                if unmet:
                    reason = f"dependency_unmet: {', '.join(sorted(set(unmet)))} did not succeed"
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="skipped", error_message=reason))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="skipped", error=reason,
                        device=_active_device(), reason="dependency_unmet"))
                    statuses[spec.engine_id] = "skipped"
                    continue
                blocked = self._gpu_blocked_reason(spec)
                if blocked:
                    log.warning("not starting engine %s: %s", spec.engine_id, blocked)
                    unavailable.append({"engine_id": spec.engine_id, "reason": blocked})
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="unavailable", error_message=blocked))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="unavailable", error=blocked,
                        device=_active_device(), reason="gpu_quarantined"))
                    statuses[spec.engine_id] = "unavailable"
                    continue
                budget = self._compute_vram_budget()
                if budget is not None and spec.vram_mb > 0 and spec.vram_mb > budget:
                    deferred.append(spec)
                    log.info("deferring engine %s: needs %d MB, available %.0f MB",
                             spec.engine_id, spec.vram_mb, budget)
                    continue
                ctx = dict(caller_context)
                out, execution = self._execute_one(spec, image_paths, ctx)
                results.append(out)
                executions.append(execution)
                statuses[spec.engine_id] = execution.status
                if execution.status in ("succeeded", "partial") and out.observations:
                    caller_context.update(_downstream_context(out, caller_context))
                _torch_empty_cache()

            for spec in deferred:  # one retry now that the cache has been released
                blocked = self._gpu_blocked_reason(spec)
                if blocked:
                    unavailable.append({"engine_id": spec.engine_id, "reason": blocked})
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="unavailable", error_message=blocked))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="unavailable", error=blocked,
                        device=_active_device(), reason="gpu_quarantined"))
                    continue
                budget = self._compute_vram_budget()
                if budget is not None and spec.vram_mb > budget:
                    reason = f"VRAM deferred: needs ~{spec.vram_mb} MB, free ~{budget:.0f} MB after the run"
                    unavailable.append({"engine_id": spec.engine_id, "reason": reason})
                    results.append(UnifiedEngineOutput(engine_id=spec.engine_id, run_id=str(caller_context.get("run_id", "")),
                                                       observations=(), status="unavailable", error_message=reason))
                    executions.append(EngineExecution(
                        engine_id=spec.engine_id, architecture_id=spec.architecture_id,
                        checkpoint_id=spec.checkpoint_id, version=spec.version, license=spec.license,
                        non_commercial=spec.non_commercial, status="unavailable", error=reason,
                        device=_active_device(), reason="vram"))
                    continue
                out, execution = self._execute_one(spec, image_paths, dict(caller_context))
                results.append(out)
                executions.append(execution)
                _torch_empty_cache()

            self._deferred = deferred
            self._wave_results.extend(results)
            self._executions.extend(executions)
            self._unavailable.extend(unavailable)
            return results

    def run_all(self, image_paths: Sequence[str], context: Optional[Dict[str, Any]] = None) -> MultiEngineReport:
        """Run ALL registered engines, merge results into a MultiEngineReport."""
        with self._lock:
            self._deferred = []
            self._unavailable = []
            self._executions = []
        outputs = self.run_wave(image_paths, context)
        report = merge_engine_outputs(outputs)
        with self._lock:
            report.engine_executions = list(self._executions)
            report.unavailable = list(self._unavailable)
        for out in outputs:
            log.info("engine %s: status=%s, n_obs=%d, wall=%.2fs",
                     out.engine_id, out.status, out.n_observations, out.wall_time_s)
        return report


def _downstream_context(out: UnifiedEngineOutput, context: Dict[str, Any]) -> Dict[str, Any]:
    """Facts a downstream engine legitimately needs from an upstream one.

    Keyed off the observation CATEGORY, not a hard-coded engine id, so a detector
    registered under any name still feeds the segmenter.
    """
    update: Dict[str, Any] = {}
    boxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for obs in out.observations:
        if obs.category in ("object", "architecture", "opening") and obs.bbox_px:
            boxes.setdefault(obs.source_image_id, []).append(tuple(obs.bbox_px))
    if boxes:
        merged = dict(context.get("bboxes") or {})
        for key, value in boxes.items():
            merged.setdefault(key, []).extend(value)
        update["bboxes"] = merged
    return update


__all__ = [
    "EngineSpec",
    "EngineExecution",
    "MultiEngineReport",
    "MultiEngineOrchestrator",
    "ModelIdentity",
    "ObservationRecord",
    "UnifiedEngineOutput",
    "register_engine",
    "unregister_engine",
    "list_engines",
    "list_available_engines",
    "get_unavailable_engines",
    "merge_engine_outputs",
    "SAME_TARGET_MIN_IOU",
]
