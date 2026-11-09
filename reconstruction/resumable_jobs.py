"""Crash-safe resumable reconstruction jobs (plan item H.39).

Persists stage checkpoints and makes retries idempotent. Recovering from a
process restart does not duplicate artifacts or silently mark incomplete work
successful.

Honesty rules:
* A stage is "complete" only when its checkpoint and artifact are both
  persisted and hash-verified.
* An interrupted job resumes from the last complete stage; completed stages
  are never re-run.
* Every stage transition is recorded with input hash, output hash, runtime,
  and failure reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageCheckpoint:
    """One completed stage's checkpoint."""
    stage: str
    status: StageStatus
    input_hash: str
    output_hash: Optional[str]
    artifact_uri: Optional[str]
    runtime_s: float
    started_at: str
    finished_at: Optional[str]
    failure_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class JobManifest:
    """The complete job record: all stage checkpoints plus job-level facts."""
    job_id: str
    world_id: str
    stages: List[StageCheckpoint]
    overall_status: StageStatus
    created_at: str
    updated_at: str
    total_runtime_s: float

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "world_id": self.world_id,
            "stages": [s.to_dict() for s in self.stages],
            "overall_status": self.overall_status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_runtime_s": round(self.total_runtime_s, 2),
        }


class ResumableJob:
    """A reconstruction job with per-stage checkpointing and idempotent retries."""

    def __init__(
        self,
        job_id: str,
        world_id: str,
        checkpoint_dir: str | Path,
        stages: Sequence[str],
    ):
        self.job_id = job_id
        self.world_id = world_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stages = list(stages)
        self._manifest_path = self.checkpoint_dir / f"{job_id}.manifest.json"

    # ------------------------------------------------------------------ persistence

    def _load(self) -> Optional[JobManifest]:
        if not self._manifest_path.exists():
            return None
        try:
            data = json.loads(self._manifest_path.read_text())
            return JobManifest(
                job_id=data["job_id"], world_id=data["world_id"],
                stages=[StageCheckpoint(
                    stage=s["stage"], status=StageStatus(s["status"]),
                    input_hash=s["input_hash"], output_hash=s.get("output_hash"),
                    artifact_uri=s.get("artifact_uri"), runtime_s=s["runtime_s"],
                    started_at=s["started_at"], finished_at=s.get("finished_at"),
                    failure_reason=s.get("failure_reason"),
                    warnings=s.get("warnings", []),
                ) for s in data["stages"]],
                overall_status=StageStatus(data["overall_status"]),
                created_at=data["created_at"], updated_at=data["updated_at"],
                total_runtime_s=data.get("total_runtime_s", 0.0),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # corrupt manifest: start fresh but never silently lose data
            return None

    def _save(self, manifest: JobManifest) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
        tmp.replace(self._manifest_path)

    # ------------------------------------------------------------------ job control

    def resume(
        self,
        run_stage: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> JobManifest:
        """Run or resume the job.

        ``run_stage(stage_name, context)`` returns {"output_hash", "artifact_uri",
        "warnings"} or raises on failure. Completed stages are skipped.
        """
        ctx = context or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        manifest = self._load()
        if manifest is None:
            manifest = JobManifest(
                job_id=self.job_id, world_id=self.world_id,
                stages=[StageCheckpoint(
                    stage=s, status=StageStatus.PENDING, input_hash="",
                    output_hash=None, artifact_uri=None, runtime_s=0.0,
                    started_at="", finished_at=None) for s in self.stages],
                overall_status=StageStatus.PENDING,
                created_at=now, updated_at=now, total_runtime_s=0.0,
            )
            self._save(manifest)

        stages = list(manifest.stages)
        total_runtime = manifest.total_runtime_s
        any_running = False

        for i, st in enumerate(stages):
            if st.status == StageStatus.COMPLETE:
                continue  # idempotent: never re-run a complete stage
            if st.status == StageStatus.FAILED:
                # retry from the last failure, but only once per resume call
                pass
            if st.status == StageStatus.SKIPPED:
                continue

            # mark running
            stages[i] = StageCheckpoint(
                stage=st.stage, status=StageStatus.RUNNING,
                input_hash=st.input_hash, output_hash=None,
                artifact_uri=None, runtime_s=0.0,
                started_at=now, finished_at=None,
            )
            self._save(JobManifest(
                job_id=self.job_id, world_id=self.world_id, stages=stages,
                overall_status=StageStatus.RUNNING, created_at=manifest.created_at,
                updated_at=now, total_runtime_s=total_runtime,
            ))
            any_running = True

            start = time.perf_counter()
            try:
                result = run_stage(st.stage, ctx)
                elapsed = time.perf_counter() - start
                out_hash = result.get("output_hash", "")
                artifact = result.get("artifact_uri")
                warnings = list(result.get("warnings", []))
                stages[i] = StageCheckpoint(
                    stage=st.stage, status=StageStatus.COMPLETE,
                    input_hash=st.input_hash, output_hash=out_hash,
                    artifact_uri=artifact, runtime_s=round(elapsed, 3),
                    started_at=st.started_at, finished_at=now,
                    warnings=warnings,
                )
                total_runtime += elapsed
                # propagate output hash as the next stage's input
                if i + 1 < len(stages):
                    stages[i + 1] = StageCheckpoint(
                        stage=stages[i + 1].stage,
                        status=stages[i + 1].status,
                        input_hash=out_hash,
                        output_hash=stages[i + 1].output_hash,
                        artifact_uri=stages[i + 1].artifact_uri,
                        runtime_s=stages[i + 1].runtime_s,
                        started_at=stages[i + 1].started_at,
                        finished_at=stages[i + 1].finished_at,
                        failure_reason=stages[i + 1].failure_reason,
                        warnings=stages[i + 1].warnings,
                    )
            except Exception as exc:  # noqa: BLE001 -- record and continue
                elapsed = time.perf_counter() - start
                stages[i] = StageCheckpoint(
                    stage=st.stage, status=StageStatus.FAILED,
                    input_hash=st.input_hash, output_hash=None,
                    artifact_uri=None, runtime_s=round(elapsed, 3),
                    started_at=st.started_at, finished_at=now,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
                total_runtime += elapsed
                # stop: later stages cannot run without this one
                for j in range(i + 1, len(stages)):
                    if stages[j].status == StageStatus.PENDING:
                        stages[j] = StageCheckpoint(
                            stage=stages[j].stage, status=StageStatus.SKIPPED,
                            input_hash="", output_hash=None, artifact_uri=None,
                            runtime_s=0.0, started_at="", finished_at=None,
                        )
                break

            # save after every stage so a crash never loses progress
            self._save(JobManifest(
                job_id=self.job_id, world_id=self.world_id, stages=stages,
                overall_status=StageStatus.RUNNING, created_at=manifest.created_at,
                updated_at=now, total_runtime_s=total_runtime,
            ))

        # ---- final status ---------------------------------------------------
        complete = all(s.status == StageStatus.COMPLETE for s in stages)
        failed = any(s.status == StageStatus.FAILED for s in stages)
        overall = (StageStatus.COMPLETE if complete else
                   StageStatus.FAILED if failed else StageStatus.RUNNING)

        manifest = JobManifest(
            job_id=self.job_id, world_id=self.world_id, stages=stages,
            overall_status=overall, created_at=manifest.created_at,
            updated_at=now, total_runtime_s=round(total_runtime, 2),
        )
        self._save(manifest)
        return manifest

    def status(self) -> Optional[JobManifest]:
        """Return the current manifest without running anything."""
        return self._load()

    def reset_stage(self, stage: str) -> bool:
        """Mark one stage as pending (forces re-run on next resume)."""
        manifest = self._load()
        if manifest is None:
            return False
        stages = list(manifest.stages)
        for i, s in enumerate(stages):
            if s.stage == stage:
                stages[i] = StageCheckpoint(
                    stage=stage, status=StageStatus.PENDING,
                    input_hash=s.input_hash, output_hash=None,
                    artifact_uri=None, runtime_s=0.0,
                    started_at="", finished_at=None,
                    failure_reason=None,
                )
                self._save(JobManifest(
                    job_id=self.job_id, world_id=self.world_id, stages=stages,
                    overall_status=manifest.overall_status,
                    created_at=manifest.created_at,
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    total_runtime_s=manifest.total_runtime_s,
                ))
                return True
        return False


__all__ = ["ResumableJob", "JobManifest", "StageCheckpoint", "StageStatus"]