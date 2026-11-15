"""P108.6: the fine stages inside the progressive ladder.

``apps/api/jobs.py`` runs ``progressive.run_progressive`` -- a fallback ladder: multi-view reconstruction, falling back
to the single-image bootstrap. That behaviour is kept. The ladder takes a ``vertical_slice_fn``; ``run_fine_stages`` is
that function: it runs the real stage objects (pose ... finalize) in data-flow order in-process, collecting each
``StageResult`` in a sink, and returns the ``VerticalSliceResult`` exactly as ``vertical_slice()`` does. Any stage
exception propagates unchanged, so a multi-view failure still makes the ladder fall back.

After the ladder returns, ``LadderReport`` stages (one per fine stage, in the registry profile) put those results into
the run manifest -- or, when the multi-view rung was not the one used (failed, or rejected as not credible), a
``skipped`` result with the ladder's reason. ``skipped`` never upgrades or downgrades the grade, so a fallback run grades
``partial`` through the ``reconstruct`` stage's degradation warnings, never ``failed``.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


@dataclass
class Ladder:
    """What the ``reconstruct`` segment leaves in ``ctx.state["ladder"]``: the ladder's ``ProgressiveResult`` and the
    results the multi-view rung's fine stages produced (empty when that rung never ran or was abandoned part-way)."""

    prog: object
    fine: dict


#: timing keys written after each stage; the same names ``vertical_slice()`` records in ``stage_timings_s``
_LAPS_AFTER = {
    "scale": ("scale", "scale_and_frame", "frame"),
    "perception": ("perception",),
    "mesh": ("mesh", "perception_and_mesh"),
    "detail": ("detail", "fusion_and_detail"),
}


def fine_chain() -> list:
    from engine.pipeline.stages.s04_pose import Pose
    from engine.pipeline.stages.s05_scale import Scale
    from engine.pipeline.stages.s06_depth import Depth
    from engine.pipeline.stages.s07_dense import Dense
    from engine.pipeline.stages.s08_mesh import Mesh
    from engine.pipeline.stages.s09_perception import Perception
    from engine.pipeline.stages.s10_detail import Detail
    from engine.pipeline.stages.s10_finalize import Finalize
    from engine.pipeline.stages.s10_layout import Layout

    # layout/compile precedes perception, mesh and dense: they read the compiled world
    return [Pose, Scale, Depth, Layout, Perception, Mesh, Dense, Detail, Finalize]


def run_fine_stages(evidence_items, options, sink: dict | None = None):
    """The ladder's multi-view rung: ``vertical_slice(evidence_items, options)`` run as stages. Results land in ``sink``
    keyed by stage name as each stage finishes."""
    clock: dict[str, float] = {}
    mark = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        clock[name] = round(now - mark[0], 2)
        mark[0] = now

    ctx = StageContext(job_id="", workdir=Path("."), artifacts={}, params={},
                       state={"evidence_items": evidence_items, "vertical_slice_options": options,
                              "lap": lap, "clock": clock})
    for cls in fine_chain():
        stage = cls()
        result = stage.run(ctx)
        if sink is not None:
            sink[stage.spec.name] = result
        for name in _LAPS_AFTER.get(stage.spec.name, ()):
            lap(name)
    return ctx.state["slice_result"]


def _why_not_used(ladder: Ladder | None) -> str:
    if ladder is None:
        return "multi-view reconstruction did not run"
    level2 = [a for a in getattr(ladder.prog, "attempts", []) if a.get("level") == 2]
    if not level2:
        return "multi-view reconstruction not attempted (needs >= 2 usable photographs); single-image bootstrap used"
    last = level2[-1]
    return f"multi-view rung {last.get('outcome')}: {last.get('detail')}"[:400]


class LadderReport:
    """Reports one fine stage's outcome from the ladder (see module docstring). Its spec is the fine stage's own."""

    def __init__(self, spec: StageSpec) -> None:
        self.spec = spec

    def run(self, ctx: StageContext) -> StageResult:
        ladder: Ladder | None = ctx.state.get("ladder")
        if ladder is not None and getattr(ladder.prog, "vs_result", None) is not None and self.spec.name in ladder.fine:
            return dataclasses.replace(ladder.fine[self.spec.name])
        now = datetime.now(timezone.utc)
        return StageResult(
            name=self.spec.name, status="skipped", inputs_hash="", outputs={}, metrics={},
            warnings=[_why_not_used(ladder)], error=None, started_at=now, ended_at=now, peak_rss_mb=None,
        )
