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
        ctx.state["stage_abandoned"] = stage.spec.name
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
        declared_inputs = spec.inputs if spec.inputs is not None else tuple(ctx.artifacts)
        input_hashes = {n: ctx.artifacts[n] for n in declared_inputs if n in ctx.artifacts}
        inputs_hash = compute_inputs_hash(spec.version, input_hashes, ctx.params)
        hit = checkpoints.lookup(ctx.job_id, spec.name, inputs_hash)
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
            checkpoints.record(ctx.job_id, result)
        manifest.stages.append(result)
        manifest.lineage.add(spec.name, spec.inputs if spec.inputs is not None else tuple(ctx.artifacts), result.outputs)
        ctx.artifacts.update(result.outputs)
        if on_end:
            on_end(result)
        if result.status == "failed" and spec.required:
            break
    return manifest
