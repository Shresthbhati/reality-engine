"""P108 stage `detail`: optional detail discovery + ROI refinement on the fused world, via ``vertical_slice.run_detail``.

Reads ``ctx.state`` ``pose`` / ``scale`` / ``layout`` (the world after the dense stage) / ``vertical_slice_options``;
sets ``ctx.state["detail"]`` (DetailOutcome). ``skipped`` with the reason when disabled (``detail_enabled=False``), when
there are no trusted intrinsics (GSD is unmeasurable), or when its optional dependencies are unavailable or refuse.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import facts_status


class Detail:
    spec = StageSpec(name="detail", version="1", resource_class="cpu", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_detail

        started = datetime.now(timezone.utc)
        outcome = run_detail(ctx.state["pose"].result, ctx.state["layout"].world, ctx.state["vertical_slice_options"],
                             ctx.state["scale"].scale_state)
        ctx.state["detail"] = outcome
        status, warnings = facts_status(outcome.facts, "detail")
        n = (outcome.facts or {}).get("n_candidates")
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={},
            metrics={"n_candidates": float(n)} if isinstance(n, (int, float)) else {}, warnings=warnings, error=None,
            started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
