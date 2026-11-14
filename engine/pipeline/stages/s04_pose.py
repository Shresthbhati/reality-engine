"""P107 stage `pose`: backend run + WorldArbiter selection, via ``vertical_slice.run_pose``.

Inputs in ``ctx.state``: ``evidence_items`` (sequence of EvidenceItem) and ``vertical_slice_options``
(VerticalSliceOptions). Output: ``ctx.state["pose"]`` (PoseOutcome). A refused/failed reconstruction raises
VerticalSliceError, which the runner records as ``failed``; a partial registration is reported ``degraded``.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


def _no_lap(name: str) -> None:
    return None


def adopt_solved_intrinsics(ctx: StageContext, pose) -> dict | None:
    """No trusted or earlier intrinsics (EXIF stripped, first job of a world): use the focal length COLMAP just
    estimated for THIS run, so depth and perception stop skipping for want of a camera model.

    Only when every camera agrees, is pinhole-like and has the images' real size (reconstruction.colmap_intrinsics).
    Trust is ``estimated`` (COLMAP is not pinned to it, distortion is ignored). Returns the record, else None."""
    opts = ctx.state["vertical_slice_options"]
    session = getattr(pose.backend, "session", None)
    if opts.intrinsics is not None or session is None:
        return None
    from reconstruction.colmap_intrinsics import median_intrinsics, read_colmap_cameras

    found = median_intrinsics(read_colmap_cameras(session.staging / "sparse" / "0"), tuple(opts.image_size))
    if found is None:
        return None
    est, spread = found
    ctx.state["vertical_slice_options"] = dataclasses.replace(
        opts, intrinsics=tuple(float(v) for v in est), intrinsics_trust="estimated")
    return {"fx": est[0], "fy": est[1], "cx": est[2], "cy": est[3], "focal_spread": round(spread, 4),
            "source": "colmap_estimated_this_run", "trust": "estimated"}


class Pose:
    spec = StageSpec(name="pose", version="1", resource_class="gpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        from engine.pipeline.vertical_slice import run_pose

        started = datetime.now(timezone.utc)
        lap = ctx.state.get("lap", _no_lap)   # the ladder driver supplies the per-stage timing hook (stage_timings_s)
        pose = run_pose(ctx.state["evidence_items"], ctx.state["vertical_slice_options"], lap=lap, lap_validation=lap)
        ctx.state["pose"] = pose
        result = pose.result
        adopted = adopt_solved_intrinsics(ctx, pose)
        warnings = []
        if adopted is not None:
            warnings.append("no trusted intrinsics: using the focal length COLMAP estimated for this run "
                            f"(fx={adopted['fx']:.1f}px, trust=estimated)")
        if result.registration_status != "success":
            warnings.append(f"registration_status={result.registration_status!r}: {len(result.camera_poses)}/"
                            f"{len(pose.evidence_items)} cameras registered")
        return StageResult(
            name=self.spec.name, status="degraded" if warnings else "succeeded", inputs_hash="",
            outputs={"intrinsics_adopted": adopted} if adopted else {},
            metrics={"cameras_registered": float(len(result.camera_poses)),
                     "cameras_input": float(len(pose.evidence_items)), "points": float(len(result.points))},
            warnings=warnings, error=None, started_at=started, ended_at=datetime.now(timezone.utc),
            peak_rss_mb=None,
        )
