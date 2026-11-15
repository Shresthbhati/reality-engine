"""P108 stage `finalize`: scale/frame/fact metadata on the world, reconstruction diagnostics + correction plan, and the
assembled ``VerticalSliceResult``, via ``vertical_slice.run_finalize``.

Reads every earlier stage's outcome from ``ctx.state`` (pose, scale, frame, depth, layout, perception, mesh, dense,
detail); sets ``ctx.state["slice_result"]`` (VerticalSliceResult -- what the ladder's multi-view rung returns).
Required. The diagnostics and correction plan never fail a build (their errors are recorded in the result's facts).
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


class Finalize:
    spec = StageSpec(name="finalize", version="1", resource_class="cpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_finalize

        started = datetime.now(timezone.utc)
        s = ctx.state
        pose = s["pose"]
        out = run_finalize(
            pose.evidence_items, pose.backend, pose.run, pose.result, s["scale"], s["frame"], s["layout"].layout_facts,
            s["depth"], s["perception"].facts, s["mesh"].facts, s["dense"], s["detail"].facts,
            clock=s.get("clock", {}), lap=s.get("lap", lambda name: None),
        )
        s["slice_result"] = out
        return StageResult(
            name=self.spec.name, status="succeeded", inputs_hash="", outputs={},
            metrics={"points": float(out.points_total), "entities": float(len(out.world.entities)),
                     "cameras_registered": float(out.cameras_registered)},
            warnings=[], error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
