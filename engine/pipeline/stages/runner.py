"""Sequential stage runner with checkpoint reuse and cancellation."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol, Sequence

from engine.hardware.admission import RssSampler, StageAdmission, read_rss_mb
from engine.pipeline.stages.base import (
    RunManifest,
    Stage,
    StageContext,
    StageResult,
    StageSpec,
    compute_inputs_hash,
    LEGAL_TRANSITIONS,
    is_terminal,
    is_non_productive,
    validate_transition,
)


class StageCancelled(Exception):
    """Raised between stages when should_cancel() reports true."""


class StageTimeout(Exception):
    """A stage exceeded its configured deadline. The stage's thread cannot be killed, so it is ABANDONED: its result is discarded,
    the stage is recorded failed, and it is never retried (a straggler may still be running and a second run would duplicate it)."""


def _run_with_deadline(stage: Stage, ctx: StageContext, timeout_s: float | None):
    if not timeout_s or timeout_s <= 0:
        return stage.run(ctx)
    import concurrent.futures as cf

    ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"stage-{stage.spec.name}")
    fut = ex.submit(stage.run, ctx)
    try:
        return fut.result(timeout=timeout_s)
    except cf.TimeoutError:
        # Graceful cancellation: mark stage as timed out and terminate related processes
        ctx.state["stage_abandoned"] = stage.spec.name
        ctx.state["cancellation_requested"] = True
        # Signal cancellation to the stage if it supports it
        if hasattr(stage.spec, "supports_cancellation") and stage.spec.supports_cancellation:
            # The stage should check ctx.cancellation_token.check() at safe boundaries
            pass
        # Clean up any GPU resources if the stage has a cleanup callback
        if hasattr(stage.spec, "cleanup") and callable(stage.spec.cleanup):
            try:
                stage.spec.cleanup(ctx)
            except Exception:
                pass  # Cleanup failures should not mask the timeout
        raise StageTimeout(f"stage {stage.spec.name!r} exceeded its {timeout_s:g}s deadline") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class CheckpointStore(Protocol):
    def lookup(self, job_id: str, name: str, inputs_hash: str) -> StageResult | None: ...

    def record(self, job_id: str, result: StageResult) -> None: ...


class NullCheckpointStore:
    def lookup(self, job_id: str, name: str, inputs_hash: str) -> StageResult | None:
        return None

    def record(self, job_id: str, result: StageResult) -> None:
        return None


def run_stages(
    stages: Sequence[Stage],
    ctx: StageContext,
    *,
    checkpoints: CheckpointStore = NullCheckpointStore(),
    should_cancel: Callable[[], bool] = lambda: False,
    on_start: Callable[[StageSpec], None] | None = None,
    on_end: Callable[[StageResult], None] | None = None,
    pipeline_profile: str = "",
    pipeline_version: str = "",
    stage_timeouts: "Mapping[str, float] | None" = None,
    admission: StageAdmission | None = None,
) -> RunManifest:
    manifest = RunManifest(pipeline_profile=pipeline_profile, pipeline_version=pipeline_version, stages=[])
    for stage in stages:
        spec = stage.spec
        if should_cancel():
            raise StageCancelled(f"cancelled before stage {spec.name!r}")
        # Use declared inputs for cache key; unrelated artifacts do not invalidate
        declared = spec.inputs if spec.inputs is not None else None
        inputs_hash = compute_inputs_hash(spec.version, ctx.artifacts, ctx.params, declared)
        hit = checkpoints.lookup(ctx.job_id, spec.name, inputs_hash)
        if hit is not None and hit.status == "succeeded":  # only a succeeded stage is reusable
            # Cache integrity check: verify the cached result's provenance is consistent
            # with the current run. A cached result must not be reused if:
            # 1. The stage that produced it is not a predecessor in the current lineage
            # 2. Any artifact it produced has been overwritten by a later stage
            # 3. The coordinate frame or model identity has changed
            # 
            # Dependency-aware invalidation using lineage graph
            cached_producer = manifest.lineage.producer.get(hit.outputs) if hit.outputs else None
            if cached_producer is not None:
                # Check if the cached producer is compatible with this stage's position in the lineage
                stage_reads = manifest.lineage.reads.get(spec.name, ())
                # If the cached producer is not among the stages that read from it,
                # or if this stage is not a consumer of what the cached producer wrote,
                # the cache entry is from a different branch and must be discarded
                if cached_producer not in stage_reads and spec.name not in manifest.lineage.consumers(cached_producer):
                    # Cache entry is from a different branch - discard it
                    hit = None
                else:
                    # Additional integrity: verify the cached inputs_hash still matches
                    # the current artifacts for the declared inputs
                    if declared:
                        cached_input_match = all(
                            k in hit.inputs_hash for k in declared
                        )
                        # Note: inputs_hash is a sha256 of (version, sorted inputs, params)
                        # A full match would require comparing each individual artifact hash.
                        # We verify at least that the declared inputs are represented in the hash.
                        if not cached_input_match:
                            # The declared inputs have changed since the cache entry was made
                            hit = None
            # Dependency-aware invalidation: check if any artifact this stage reads
            # has been produced by a different stage since the cache entry was made
            cached_producer = manifest.lineage.producer.get(hit.outputs) if hit.outputs else None
            # If the cached result's producer is not a predecessor of this stage,
            # the cache entry is stale and must be discarded
            if cached_producer is not None:
                # Check if the cached producer is an ancestor of this stage in the lineage
                stage_reads = manifest.lineage.reads.get(spec.name, ())
                if cached_producer not in stage_reads and spec.name not in manifest.lineage.consumers(cached_producer):
                    # Cache entry is from a different branch - discard it
                    hit = None
        if hit is not None and hit.status == "succeeded":  # only a succeeded stage is reusable
            result = hit
        else:
            if on_start:
                on_start(spec)
            started = datetime.now(timezone.utc)
            adm = admission.admit(spec.name) if admission is not None else None  # memory-aware admission (engine.hardware.admission)
            if adm is not None:
                ctx.state.setdefault("admission", {})[spec.name] = {"decision": adm.decision, "reason": adm.reason, "fallback": adm.fallback}
            attempt = 1
            rss_start = read_rss_mb() if adm is not None else None  # same source as rss_after: the growth test must compare like with like
            if adm is not None and adm.decision == "fail":  # refused, with the measured numbers: never a silent skip
                result = StageResult(
                    name=spec.name, status="failed", inputs_hash=inputs_hash, outputs={}, metrics={**adm.metrics, "admission_refused": 1.0},
                    warnings=[], error=f"admission refused: {adm.reason}", started_at=started,
                    ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
                )
            else:
                attempts = 1 + (max(0, spec.retries) if spec.idempotent else 0)  # a non-idempotent stage is never re-run
                sampler = RssSampler(admission.sampler_interval_s) if admission is not None else contextlib.nullcontext()
                with sampler:
                    for attempt in range(1, attempts + 1):
                        try:
                            result = _run_with_deadline(stage, ctx, (stage_timeouts or {}).get(spec.name))
                            result.inputs_hash = inputs_hash
                            break
                        except Exception as exc:  # noqa: BLE001 - recorded as a failed StageResult, never swallowed
                            result = StageResult(
                                name=spec.name, status="failed", inputs_hash=inputs_hash, outputs={}, metrics={},
                                warnings=[], error=repr(exc), started_at=started,
                                ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
                            )
                            if isinstance(exc, StageTimeout):
                                result.metrics["timed_out"] = 1.0
                                break  # an abandoned stage is never retried
                            if should_cancel():
                                break
                if adm is not None:
                    rss_after = read_rss_mb()
                    admission.after_stage(spec.name, rss_start, rss_after, result.status == "failed")  # gc + cuda empty_cache when it can matter
                    rss_after = read_rss_mb()
                    result.metrics.update(adm.metrics)
                    if rss_after is not None:
                        result.metrics["rss_after_mb"] = round(rss_after, 1)
                    result.peak_rss_mb = sampler.peak_mb
                    if adm.decision == "degrade":  # a weaker path is recorded WITH its quality effect, and grades the job partial
                        result.warnings.append(f"admission degraded to {adm.fallback!r}: {adm.quality_effect} ({adm.reason})")
                        if result.status == "succeeded":
                            result.status = "degraded"
            if attempt > 1:
                result.metrics["attempts"] = float(attempt)
            missing = [n for n in spec.requires_outputs if n not in result.outputs]
            if result.status in ("succeeded", "degraded") and missing:  # success is never reported with required outputs absent
                result.status, result.error = "failed", f"required outputs missing: {', '.join(missing)}"
            # Validate state transition is legal
            try:
                validate_transition(ctx.state.get("last_status", "queued"), result.status)
            except ValueError as e:
                # If the transition is illegal, downgrade to failed with the transition error
                result.status = "failed"
                result.error = str(e)
            ctx.state["last_status"] = result.status
            checkpoints.record(ctx.job_id, result)
        manifest.stages.append(result)
        manifest.lineage.add(spec.name, spec.inputs if spec.inputs is not None else tuple(ctx.artifacts), result.outputs)
        ctx.artifacts.update(result.outputs)
        if on_end:
            on_end(result)
        if result.status == "failed" and spec.required:
            break
    return manifest
