"""P107 stage `pose`: backend run + WorldArbiter selection, via ``vertical_slice.run_pose``.

Inputs in ``ctx.state``: ``evidence_items`` (sequence of EvidenceItem) and ``vertical_slice_options``
(VerticalSliceOptions). Output: ``ctx.state["pose"]`` (PoseOutcome). A refused/failed reconstruction raises
VerticalSliceError, which the runner records as ``failed``; a partial registration is reported ``degraded``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


def _no_lap(name: str) -> None:
    return None


class Pose:
    spec = StageSpec(name="pose", version="1", resource_class="gpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_pose

        started = datetime.now(timezone.utc)
        lap = ctx.state.get("lap", _no_lap)   # the ladder driver supplies the per-stage timing hook (stage_timings_s)
        pose = run_pose(ctx.state["evidence_items"], ctx.state["vertical_slice_options"], lap=lap, lap_validation=lap)
        ctx.state["pose"] = pose
        result = pose.result
        warnings = []
        if result.registration_status != "success":
            warnings.append(f"registration_status={result.registration_status!r}: {len(result.camera_poses)}/"
                            f"{len(pose.evidence_items)} cameras registered")
        return StageResult(
            name=self.spec.name, status="degraded" if warnings else "succeeded", inputs_hash="", outputs={},
            metrics={"cameras_registered": float(len(result.camera_poses)),
                     "cameras_input": float(len(pose.evidence_items)), "points": float(len(result.points))},
            warnings=warnings, error=None, started_at=started, ended_at=datetime.now(timezone.utc),
            peak_rss_mb=None,
        )
