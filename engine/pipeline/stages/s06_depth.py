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


def refine_frame_from_dense(ctx: StageContext, result) -> dict | None:
    """The frame was fixed from the sparse cloud; dense monocular depth is systematically tilted against it, so its
    floors and ceilings can sit beyond the classifier's tolerance and be rejected. Re-fit "up" to the dense cloud's own
    floor/ceiling-like planes and, when the correction is real, rotate points and cameras in place and fold it into the
    recorded frame rotation (the dense MVS placement, world metadata and incremental updates all read that record).
    Returns the facts of the correction, or None when nothing was corrected."""
    import dataclasses

    import numpy as np

    from engine.compiler import CompileOptions
    from perception.geometry.planes import detect_planes
    from reconstruction.frame import refine_up_from_planes, rotate_in_place

    opts = CompileOptions()
    try:
        planes = detect_planes(result, seed=opts.seed, distance_tolerance_m=opts.distance_tolerance_m,
                               min_inliers=opts.min_inliers).planes
        found = refine_up_from_planes(planes)
    except Exception:  # noqa: BLE001 -- a refinement; never a reason to lose the depth stage
        return None
    if found is None:
        return None
    R_fix, angle = found
    rotate_in_place(result, R_fix)
    fr = ctx.state["frame"]
    rec = fr.frame_record
    composed = R_fix @ np.asarray(rec.rotation, dtype=float)
    ctx.state["frame"] = dataclasses.replace(fr, frame_record=dataclasses.replace(
        rec, rotation=tuple(tuple(float(v) for v in row) for row in composed), up_source=rec.up_source + "+dense_planes",
        note=rec.note + f" | up re-fitted {angle:.1f} deg to the dense cloud's floor/ceiling planes"))
    return {"correction_deg": round(angle, 2), "planes_used": len(planes), "up_source": rec.up_source + "+dense_planes"}


class Depth:
    spec = StageSpec(name="depth", version="1", resource_class="gpu", required=False)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_depth

        started = datetime.now(timezone.utc)
        pose = ctx.state["pose"]
        outcome = run_depth(pose.result, pose.evidence_items, ctx.state["vertical_slice_options"],
                            ctx.state["scale"].scale_state)
        if outcome.facts and outcome.facts.get("dense_points"):
            refined = refine_frame_from_dense(ctx, pose.result)
            if refined is not None:
                outcome.facts["frame_refined"] = refined
        ctx.state["depth"] = outcome
        status, warnings = facts_status(outcome.facts, "depth")
        facts = outcome.facts or {}
        metrics = {k: float(facts[k]) for k in ("maps", "metricized", "dense_points") if isinstance(facts.get(k), (int, float))}
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
