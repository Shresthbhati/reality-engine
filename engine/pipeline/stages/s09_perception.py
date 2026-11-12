"""P108 stage `perception`: Mask R-CNN instance masks lifted through metricized depth into world object entities, via
``vertical_slice.run_perception``.

Reads ``ctx.state`` ``pose`` / ``depth`` (the metricized depth maps) / ``layout`` / ``vertical_slice_options``;
promotes objects into the layout world in place and sets ``ctx.state["perception"]`` (PerceptionOutcome). ``skipped``
with the reason when perception is disabled (``perception_model=None``), there are no trusted intrinsics, no metric
depth survived, or the detector / its weights are unavailable. An invalid world after promotion raises
VerticalSliceError (the runner records ``failed``; the stage is optional, so the job grades PARTIAL).

Plan 3.4 also names the detail pass under perception. That pass runs AFTER dense fusion in ``vertical_slice()`` (it
reads the dense points and the fused world), so it cannot share a stage with perception without changing behaviour;
it stays in ``vertical_slice()`` as ``_detail_stage`` for now.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import facts_status


class Perception:
    spec = StageSpec(name="perception", version="1", resource_class="gpu", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_perception

        started = datetime.now(timezone.utc)
        pose = ctx.state["pose"]
        outcome = run_perception(pose.result, ctx.state["layout"].world, pose.evidence_items,
                                 ctx.state["depth"].metric_maps, ctx.state["vertical_slice_options"])
        ctx.state["perception"] = outcome
        status, warnings = facts_status(outcome.facts, "perception")
        metrics = {k: float(v) for k, v in outcome.facts.items()
                   if k in ("views_segmented", "masks_considered", "hypotheses_lifted", "candidates_merged",
                            "entities_promoted") and isinstance(v, (int, float))}
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
