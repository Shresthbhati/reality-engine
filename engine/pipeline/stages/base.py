"""Stage contract (docs/blueprint/34_MASTER_PLAN.md section 3.3). Pure; no database."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Mapping, Protocol

ResourceClass = Literal["cpu", "gpu", "highmem", "export"]
StageStatus = Literal["succeeded", "degraded", "skipped", "failed"]


@dataclass(frozen=True)
class StageSpec:
    name: str
    version: str  # bump on behaviour change; part of the cache key
    resource_class: ResourceClass
    required: bool  # required + failed -> job FAILED; optional + failed -> PARTIAL


@dataclass
class StageResult:
    name: str
    status: StageStatus
    inputs_hash: str  # sha256 of (spec.version, sorted input artifact hashes, params json)
    outputs: dict[str, str]  # logical name -> artifact sha256
    metrics: dict[str, float]
    warnings: list[str]
    error: str | None
    started_at: datetime
    ended_at: datetime
    peak_rss_mb: float | None


@dataclass
class StageContext:
    job_id: str
    workdir: Path
    artifacts: dict[str, str]  # logical name -> sha256, grows as stages finish
    params: dict
    state: dict = field(default_factory=dict)  # in-memory objects shared within one process
    log: Callable[[str], None] = lambda msg: None


class Stage(Protocol):
    spec: StageSpec

    def run(self, ctx: StageContext) -> StageResult: ...


@dataclass
class RunManifest:
    pipeline_profile: str
    pipeline_version: str
    stages: list[StageResult]

    def grade(self, specs: list[StageSpec]) -> Literal["succeeded", "partial", "failed"]:
        """Any required stage failed -> failed; else any failed/degraded -> partial; else succeeded.

        Skipped never upgrades or downgrades.
        """
        required = {s.name: s.required for s in specs}
        if any(r.status == "failed" and required.get(r.name, True) for r in self.stages):
            return "failed"
        if any(r.status in ("failed", "degraded") for r in self.stages):
            return "partial"
        return "succeeded"


def compute_inputs_hash(version: str, input_hashes: Mapping[str, str], params: Mapping) -> str:
    payload = json.dumps([version, sorted(input_hashes.items()), params], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
