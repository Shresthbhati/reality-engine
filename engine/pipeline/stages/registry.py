"""Pipeline profiles and the segment adapter shared by the coarse stages (P106).

A "segment" is a callable the host (``apps/api/jobs.py``) supplies in ``ctx.state["segments"]`` keyed by stage
name. Each stage module is a thin wrapper that runs its segment and maps the outcome onto a ``StageResult``. The
runner and these modules stay free of ``apps.api`` imports; the host owns the DB/session objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from engine.pipeline.stages.base import Stage, StageContext, StageResult, StageSpec

PIPELINE_VERSION = "p106-coarse"
PROFILE_NAME = "interior_survey_standard"

#: P108 replaces ``reconstruct_legacy`` with the finer stages of plan section 3.4.
PIPELINE_PROFILES: dict[str, list[str]] = {
    PROFILE_NAME: [
        "evidence_resolve", "evidence_quality", "reconstruct_legacy", "compile_validate", "changeset_commit",
    ],
}


@dataclass
class SegmentReport:
    """What a segment reports back: any warnings make the stage ``degraded`` (graded PARTIAL)."""
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class SegmentStage:
    spec: StageSpec

    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(timezone.utc)
        try:
            report: SegmentReport = ctx.state["segments"][self.spec.name](ctx)
        except Exception as exc:
            # The runner records repr(exc) on the failed StageResult; the host re-raises the ORIGINAL exception
            # (type, message, failure_kind, retryable) so job error text stays byte-identical.
            ctx.state["stage_exc"] = exc
            raise
        return StageResult(
            name=self.spec.name, status="degraded" if report.warnings else "succeeded", inputs_hash="",
            outputs={}, metrics=dict(report.metrics), warnings=list(report.warnings), error=None,
            started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )


def build_stages(profile: str = PROFILE_NAME) -> list[Stage]:
    from engine.pipeline.stages.s01_evidence import EvidenceResolve
    from engine.pipeline.stages.s02_quality import EvidenceQuality
    from engine.pipeline.stages.s04_reconstruct_legacy import ReconstructLegacy
    from engine.pipeline.stages.s11_compile_validate import CompileValidate
    from engine.pipeline.stages.s13_changeset import ChangesetCommit

    by_name = {c.spec.name: c for c in (EvidenceResolve, EvidenceQuality, ReconstructLegacy, CompileValidate, ChangesetCommit)}
    return [by_name[n]() for n in PIPELINE_PROFILES[profile]]
