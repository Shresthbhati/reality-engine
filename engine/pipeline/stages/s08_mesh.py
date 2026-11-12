"""P108 stage `mesh`: fused metric points -> Poisson surface mesh artifact -> WorldIR geometry, via
``vertical_slice.run_mesh``.

Reads ``ctx.state`` ``pose`` / ``scale`` / ``layout`` / ``vertical_slice_options``; mutates the layout world in place
and sets ``ctx.state["mesh"]`` (MeshOutcome). ``skipped`` with the reason when meshing is disabled, the world is not
metric, there is no artifact store, too few points, or the meshing dependencies / COLMAP poisson_mesher are
unavailable -- never ``succeeded`` for a mesh that was not built; ``degraded`` when meshing was attempted and failed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import facts_status


class Mesh:
    spec = StageSpec(name="mesh", version="1", resource_class="highmem", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_mesh

        started = datetime.now(timezone.utc)
        outcome = run_mesh(ctx.state["pose"].result, ctx.state["layout"].world, ctx.state["vertical_slice_options"],
                           ctx.state["scale"].scale_state)
        ctx.state["mesh"] = outcome
        status, warnings = facts_status(outcome.facts, "mesh")
        metrics = {k: float(v) for k, v in (outcome.facts or {}).items()
                   if k in ("vertices", "triangles", "input_points") and isinstance(v, (int, float))}
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
