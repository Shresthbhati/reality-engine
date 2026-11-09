"""Stage-by-stage reconstruction observability (plan item H.38).

Records the input, output, runtime, memory consumption, warnings and failure
reason for each pipeline stage. An engineer can trace why a particular wall,
opening or object was missing without guessing which stage failed.

Honesty rules:
* Every stage record carries input_hash, output_hash, runtime, and either
  success or an explicit failure reason.
* A stage that was skipped records its skip reason, never "success".
* Memory is reported when measurable, otherwise None (not guessed).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class StageRecord:
    """One pipeline stage execution."""
    stage: str
    status: str                    # "success" | "skipped" | "failed"
    started_at: str
    finished_at: Optional[str]
    runtime_s: float
    input_hash: Optional[str]
    output_hash: Optional[str]
    memory_mb: Optional[float]
    warnings: List[str]
    failure_reason: Optional[str]
    artifact_uri: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineObservability:
    """Complete stage-by-stage observability for one reconstruction run."""
    run_id: str
    world_id: str
    stages: List[StageRecord]
    total_runtime_s: float
    failed_stages: List[str]
    skipped_stages: List[str]
    overall_status: str            # "success" | "partial" | "failed"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "world_id": self.world_id,
            "stages": [s.to_dict() for s in self.stages],
            "total_runtime_s": round(self.total_runtime_s, 2),
            "failed_stages": self.failed_stages,
            "skipped_stages": self.skipped_stages,
            "overall_status": self.overall_status,
        }


@dataclass
class _StageBuilder:
    """Mutable builder for a StageRecord during time_stage()."""
    stage: str
    started_at: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    memory_mb: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    artifact_uri: Optional[str] = None


class _TimedStage:
    """Context manager returned by PipelineObserver.time_stage()."""

    def __init__(self, observer: "PipelineObserver", stage: str):
        self._observer = observer
        self._stage = stage
        self._start: Optional[float] = None

    def __enter__(self) -> _StageBuilder:
        self._start = time.perf_counter()
        builder = _StageBuilder(
            stage=self._stage,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._observer._current_builder = builder
        return builder

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - (self._start or 0.0)
        if exc_type is not None:
            status = "failed"
            failure_reason = f"{exc_type.__name__}: {exc_val}"
        else:
            status = "success"
            failure_reason = None

        builder = self._observer._current_builder
        finished = StageRecord(
            stage=self._stage, status=status,
            started_at=builder.started_at if builder else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            runtime_s=round(elapsed, 3),
            input_hash=builder.input_hash if builder else None,
            output_hash=builder.output_hash if builder else None,
            memory_mb=builder.memory_mb if builder else None,
            warnings=list(builder.warnings) if builder else [],
            failure_reason=failure_reason,
            artifact_uri=builder.artifact_uri if builder else None,
        )
        self._observer._records.append(finished)
        self._observer._current_builder = None


class PipelineObserver:
    """Records and queries stage-by-stage pipeline observability."""

    def __init__(self):
        self._records: List[StageRecord] = []
        self._run_id: Optional[str] = None
        self._world_id: Optional[str] = None
        self._current_builder: Optional[_StageBuilder] = None

    def begin_run(self, run_id: str, world_id: str) -> None:
        self._run_id = run_id
        self._world_id = world_id
        self._records = []

    def record_stage(
        self,
        stage: str,
        *,
        status: str,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
        memory_mb: Optional[float] = None,
        warnings: Optional[Sequence[str]] = None,
        failure_reason: Optional[str] = None,
        artifact_uri: Optional[str] = None,
    ) -> StageRecord:
        """Record one stage's execution.

        status: "success" | "skipped" | "failed"
        """
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        runtime_s = 0.0

        record = StageRecord(
            stage=stage, status=status,
            started_at=started, finished_at=started,
            runtime_s=runtime_s,
            input_hash=input_hash, output_hash=output_hash,
            memory_mb=memory_mb,
            warnings=list(warnings or []),
            failure_reason=failure_reason,
            artifact_uri=artifact_uri,
        )
        self._records.append(record)
        return record

    def time_stage(self, stage: str):
        """Context manager that measures a stage's runtime and records it.

        Usage:
            with observer.time_stage("meshing") as rec:
                ... do meshing work ...
                rec.output_hash = "..."
                rec.warnings.append("...")
        """
        return _TimedStage(self, stage)

    def finalize(self) -> PipelineObservability:
        """Build the final observability report."""
        total_runtime = sum(r.runtime_s for r in self._records)
        failed = [r.stage for r in self._records if r.status == "failed"]
        skipped = [r.stage for r in self._records if r.status == "skipped"]

        if failed:
            overall = "failed"
        elif skipped and not any(r.status == "success" for r in self._records):
            overall = "partial"
        elif skipped:
            overall = "partial"
        else:
            overall = "success"

        return PipelineObservability(
            run_id=self._run_id or "unknown",
            world_id=self._world_id or "unknown",
            stages=list(self._records),
            total_runtime_s=round(total_runtime, 2),
            failed_stages=failed,
            skipped_stages=skipped,
            overall_status=overall,
        )

    def stage_for(self, stage: str) -> Optional[StageRecord]:
        for r in self._records:
            if r.stage == stage:
                return r
        return None


__all__ = ["PipelineObserver", "StageRecord", "PipelineObservability"]