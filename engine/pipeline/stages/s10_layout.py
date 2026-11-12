"""P108 stage `layout`: single-image layout evidence + the structure compile (planes -> classification -> promotion ->
rooms -> validation gate), via ``vertical_slice.run_layout``.

Reads ``ctx.state`` ``pose`` / ``vertical_slice_options``; sets ``ctx.state["layout"]`` (LayoutOutcome: the compiled
WorldIR, its CompileDiagnostics, the CompileOptions and the layout-evidence facts). Required: a world that cannot be
compiled raises VerticalSliceError and fails the job. The layout-evidence sidecars are optional input; their absence
(no learned estimator is bundled) is the normal state and is reported in ``warnings``, not graded as damage -- the
stage's own job, the compile, ran.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import facts_status


class Layout:
    spec = StageSpec(name="layout", version="1", resource_class="cpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_layout

        started = datetime.now(timezone.utc)
        pose = ctx.state["pose"]
        outcome = run_layout(pose.result, pose.evidence_items, ctx.state["vertical_slice_options"], pose.backend)
        ctx.state["layout"] = outcome
        evidence_status, evidence_warnings = facts_status(outcome.layout_facts, "layout_evidence")
        warnings = evidence_warnings if evidence_status != "succeeded" else []
        diag = outcome.diagnostics
        metrics = {"entities": float(len(outcome.world.entities)),
                   "rooms_detected": float(diag.rooms_detected),
                   "layout_hypotheses": float((outcome.layout_facts or {}).get("hypotheses") or 0)}
        return StageResult(
            name=self.spec.name, status="succeeded", inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
