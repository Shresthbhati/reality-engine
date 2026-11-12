"""P108 stage `depth`: optional monocular / sidecar depth -> dense metric points, via ``vertical_slice.run_depth``.

Reads ``ctx.state["pose"]`` (the scaled + frame-canonical reconstruction), ``["scale"]`` and
``["vertical_slice_options"]``. Appends depth points to the reconstruction in place and sets
``ctx.state["depth"]`` (DepthOutcome: facts + the metricized depth maps the perception stage lifts against).
``skipped`` with the reason when the stage is disabled (``depth_model=None``), has no trusted intrinsics, or its
model/dependencies are unavailable; ``degraded`` when the depth backend itself failed or sidecars were unparseable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import facts_status


class Depth:
    spec = StageSpec(name="depth", version="1", resource_class="gpu", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_depth

        started = datetime.now(timezone.utc)
        pose = ctx.state["pose"]
        outcome = run_depth(pose.result, pose.evidence_items, ctx.state["vertical_slice_options"],
                            ctx.state["scale"].scale_state)
        ctx.state["depth"] = outcome
        status, warnings = facts_status(outcome.facts, "depth")
        facts = outcome.facts or {}
        metrics = {k: float(facts[k]) for k in ("maps", "metricized", "dense_points") if isinstance(facts.get(k), (int, float))}
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
