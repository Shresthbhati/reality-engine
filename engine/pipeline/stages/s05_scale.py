"""P107 stage `scale`: metric scale anchoring, via ``vertical_slice.run_scale``.

Reads ``ctx.state["pose"]`` (set by the pose stage) and ``ctx.state["vertical_slice_options"]``; replaces
``ctx.state["pose"]`` with the scaled reconstruction and sets ``ctx.state["scale"]`` (ScaleOutcome). A world that
stays RELATIVE because measured baselines could not be registered is ``degraded`` (never silently metric).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


class Scale:
    spec = StageSpec(name="scale", version="1", resource_class="cpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_scale

        started = datetime.now(timezone.utc)
        pose = ctx.state["pose"]
        sc = run_scale(pose.result, pose.evidence_items, ctx.state["vertical_slice_options"])
        ctx.state["scale"] = sc
        ctx.state["pose"] = dataclasses.replace(pose, result=sc.result)
        warnings = [sc.scale_error] if sc.scale_error else []
        metrics = {"unit_k": float(sc.unit_k)}
        if sc.meters_per_unit is not None:
            metrics["meters_per_unit"] = float(sc.meters_per_unit)
        return StageResult(
            name=self.spec.name, status="degraded" if warnings else "succeeded", inputs_hash="", outputs={},
            metrics=metrics, warnings=warnings, error=None, started_at=started,
            ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
