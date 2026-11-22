"""Sequential stage runner with checkpoint reuse and cancellation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

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
) -> RunManifest:
    manifest = RunManifest(pipeline_profile=pipeline_profile, pipeline_version=pipeline_version, stages=[])
    for stage in stages:
        spec = stage.spec
        if should_cancel():
            raise StageCancelled(f"cancelled before stage {spec.name!r}")
        inputs_hash = compute_inputs_hash(spec.version, ctx.artifacts, ctx.params)
        hit = checkpoints.lookup(ctx.job_id, spec.name, inputs_hash)
        if hit is not None and hit.status == "succeeded":  # only a succeeded stage is reusable
            result = hit
        else:
            if on_start:
                on_start(spec)
            started = datetime.now(timezone.utc)
            attempts = 1 + (max(0, spec.retries) if spec.idempotent else 0)  # a non-idempotent stage is never re-run
            for attempt in range(1, attempts + 1):
                try:
                    result = stage.run(ctx)
                    result.inputs_hash = inputs_hash
                    break
                except Exception as exc:  # noqa: BLE001 - recorded as a failed StageResult, never swallowed
                    result = StageResult(
                        name=spec.name, status="failed", inputs_hash=inputs_hash, outputs={}, metrics={},
                        warnings=[], error=repr(exc), started_at=started,
                        ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
                    )
                    if should_cancel():
                        break
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
