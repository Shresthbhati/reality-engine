"""P108 stage `dense`: dense MVS + dense-structure pass + CPU plane sweep + plural-source fusion, via
``vertical_slice.run_dense``.

Reads ``ctx.state`` ``pose`` / ``scale`` / ``frame`` / ``layout`` / ``perception`` / ``mesh`` /
``vertical_slice_options``. May replace the world/diagnostics held in ``ctx.state["layout"]`` (an adopted dense
structure pass recompiles the world) and appends dense points to the reconstruction in place. Sets
``ctx.state["dense"]`` (DenseOutcome). Every sub-step that did not run keeps its reason in ``warnings``; the stage is
``skipped`` only when NOTHING ran (dense MVS disabled/unavailable, plane sweep off, fusion gated out) and
``degraded`` when a sub-step failed.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.facts import combine, facts_status


class Dense:
    spec = StageSpec(name="dense", version="1", resource_class="highmem", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_dense

        started = datetime.now(timezone.utc)
        pose, layout = ctx.state["pose"], ctx.state["layout"]
        outcome = run_dense(
            pose.result, layout.world, layout.diagnostics, layout.compile_options, pose.evidence_items,
            ctx.state["vertical_slice_options"], scale_state=ctx.state["scale"].scale_state,
            meters_per_unit=ctx.state["scale"].meters_per_unit, frame_rotation=ctx.state["frame"].frame_record.rotation,
            unit_k=ctx.state["frame"].unit_k, perception_facts=ctx.state["perception"].facts,
            mesh_facts=ctx.state["mesh"].facts,
        )
        ctx.state["dense"] = outcome
        ctx.state["layout"] = dataclasses.replace(layout, world=outcome.world, diagnostics=outcome.diagnostics)
        parts = [facts_status(outcome.dense_mvs_facts, "dense_mvs"),
                 facts_status(outcome.plane_sweep_facts, "plane_sweep"),
                 facts_status(outcome.fusion_facts, "fusion")]
        if outcome.structure_facts is not None:        # only attempted when dense ran and perception/mesh did not
            parts.append(facts_status(outcome.structure_facts, "dense_structure"))
        status, warnings = combine(parts)
        metrics = {}
        for key, facts, field in (("dense_points_added", outcome.dense_mvs_facts, "dense_points_added"),
                                  ("plane_sweep_points", outcome.plane_sweep_facts, "points_added"),
                                  ("fusion_associated", outcome.fusion_facts, "n_associated")):
            if isinstance((facts or {}).get(field), (int, float)):
                metrics[key] = float(facts[field])
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
