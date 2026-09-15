"""Canonical trajectory model (P3-01): a continuous, backend-neutral,
time-stamped pose sequence that VIO/SfM/EKF backends (P3-02) and
cross-source registration (P4-01) both consume and produce.

Reuses what exists rather than inventing a second representation:
poses are `RigidTransform` (`reconstruction.calibration.transforms`,
P2-02) so a trajectory composes with the same calibration chains as
everything else; timestamps are the global timeline produced by
`ClockModel.apply` (`evidence.clocks`, P2-01) -- a `Trajectory` never
carries a second, competing notion of time.

CONVENTIONS:

- Every frame's pose maps the SAME `from_frame` (the moving body/
  camera) into the SAME `to_frame` (the fixed world/map frame) --
  checked at construction. A trajectory that changes frame identity
  mid-sequence is a bug, not a valid trajectory.
- Timestamps are the global timeline (post `ClockModel.apply`),
  strictly increasing -- checked at construction (spec: "monotonic,
  strictly increasing timestamps (checked at construction)").
- `drift_estimate` is UNKNOWN unless a backend actually estimated it.
  UNKNOWN is a distinct, explicit state -- never silently treated as
  zero drift (the same uncertainty-honesty rule as P2-01/P2-02).
- `frame_source` records which backend produced the trajectory
  (vio/svo/sfm/ekf) plus free-form provenance text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from reconstruction.calibration.transforms import RigidTransform


class TrajectoryError(ValueError):
    """A trajectory/frame construction violated the documented contract
    (non-monotonic timestamps, inconsistent frame identity, invalid
    drift/covariance)."""


class FrameSource(str, Enum):
    """Which backend produced this trajectory. Matches the spec's
    ``vio | svo | sfm | ekf`` plus an explicit unknown for legacy/
    untagged data -- never guessed."""

    VIO = "vio"
    SVO = "svo"
    SFM = "sfm"
    EKF = "ekf"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DriftEstimate:
    """Trajectory drift: a scalar (meters, e.g. end-to-start error over
    a known loop) or a 6x6 pose covariance, or UNKNOWN when the
    producing backend did not estimate one. UNKNOWN is not zero -- a
    caller must not assume a trajectory without a drift estimate is
    drift-free."""

    known: bool
    scalar_m: Optional[float] = None
    covariance_6x6: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if not self.known:
            if self.scalar_m is not None or self.covariance_6x6 is not None:
                raise TrajectoryError("an UNKNOWN drift estimate must carry no values")
            return
        if self.scalar_m is None and self.covariance_6x6 is None:
            raise TrajectoryError("a known drift estimate needs scalar_m and/or covariance_6x6")
        if self.scalar_m is not None and self.scalar_m < 0.0:
            raise TrajectoryError(f"drift scalar_m must be >= 0, got {self.scalar_m!r}")
        if self.covariance_6x6 is not None and len(self.covariance_6x6) != 36:
            raise TrajectoryError(
                f"covariance_6x6 must have 36 entries (6x6 flattened row-major), "
                f"got {len(self.covariance_6x6)}"
            )

    @staticmethod
    def unknown() -> "DriftEstimate":
        return DriftEstimate(known=False)

    def to_dict(self) -> dict:
        return {
            "known": self.known,
            "scalar_m": self.scalar_m,
            "covariance_6x6": list(self.covariance_6x6) if self.covariance_6x6 else None,
        }

    @staticmethod
    def from_dict(data: dict) -> "DriftEstimate":
        cov = data.get("covariance_6x6")
        return DriftEstimate(
            known=bool(data.get("known", False)),
            scalar_m=data.get("scalar_m"),
            covariance_6x6=tuple(cov) if cov else None,
        )


@dataclass(frozen=True)
class TrajectoryFrame:
    """One pose sample: a global timestamp (nanoseconds, post
    `ClockModel.apply`) and the rigid pose at that instant. Pose
    covariance is optional -- absent means the backend did not report
    one, never zero uncertainty."""

    timestamp_ns: int
    pose: RigidTransform
    pose_covariance_6x6: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.pose_covariance_6x6 is not None and len(self.pose_covariance_6x6) != 36:
            raise TrajectoryError(
                f"pose_covariance_6x6 must have 36 entries, got {len(self.pose_covariance_6x6)}"
            )

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "pose": self.pose.to_dict(),
            "pose_covariance_6x6": (
                list(self.pose_covariance_6x6) if self.pose_covariance_6x6 else None
            ),
        }

    @staticmethod
    def from_dict(data: dict) -> "TrajectoryFrame":
        cov = data.get("pose_covariance_6x6")
        return TrajectoryFrame(
            timestamp_ns=int(data["timestamp_ns"]),
            pose=RigidTransform.from_dict(data["pose"]),
            pose_covariance_6x6=tuple(cov) if cov else None,
        )


@dataclass(frozen=True)
class Trajectory:
    """A continuous, ordered pose sequence in one consistent frame pair.

    Construction validates (spec acceptance criteria):
      - strictly increasing timestamps
      - frame identity: every frame's pose has the same `from_frame`
        (body) and `to_frame` (world) as the first frame

    `at` does no interpolation (a deliberate scope cut --
    resampling/interpolation is a P3-02/P3-03 consumer concern, not
    this representation's job).
    """

    frames: Tuple[TrajectoryFrame, ...]
    frame_source: FrameSource = FrameSource.UNKNOWN
    provenance: str = ""
    drift_estimate: DriftEstimate = field(default_factory=DriftEstimate.unknown)

    def __post_init__(self) -> None:
        if not self.frames:
            raise TrajectoryError("a trajectory must contain at least one frame")

        first_pose = self.frames[0].pose
        body_frame, world_frame = first_pose.from_frame, first_pose.to_frame

        prev_ts: Optional[int] = None
        for frame in self.frames:
            if frame.pose.from_frame != body_frame or frame.pose.to_frame != world_frame:
                raise TrajectoryError(
                    f"inconsistent frame identity: expected "
                    f"{body_frame!r}->{world_frame!r}, got "
                    f"{frame.pose.from_frame!r}->{frame.pose.to_frame!r} "
                    f"at timestamp_ns={frame.timestamp_ns}"
                )
            if prev_ts is not None and frame.timestamp_ns <= prev_ts:
                raise TrajectoryError(
                    f"timestamps must be strictly increasing: {frame.timestamp_ns} "
                    f"does not follow {prev_ts}"
                )
            prev_ts = frame.timestamp_ns

    @property
    def body_frame(self) -> str:
        return self.frames[0].pose.from_frame

    @property
    def world_frame(self) -> str:
        return self.frames[0].pose.to_frame

    @property
    def start_ns(self) -> int:
        return self.frames[0].timestamp_ns

    @property
    def end_ns(self) -> int:
        return self.frames[-1].timestamp_ns

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def __len__(self) -> int:
        return len(self.frames)

    def at(self, timestamp_ns: int) -> TrajectoryFrame:
        """The frame at an EXACT timestamp. No interpolation -- raises
        if no frame matches, rather than guessing a pose."""
        for frame in self.frames:
            if frame.timestamp_ns == timestamp_ns:
                return frame
        raise TrajectoryError(
            f"no frame at timestamp_ns={timestamp_ns} "
            f"(range [{self.start_ns}, {self.end_ns}]); this representation "
            f"does not interpolate"
        )

    def to_dict(self) -> dict:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "frame_source": self.frame_source.value,
            "provenance": self.provenance,
            "drift_estimate": self.drift_estimate.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "Trajectory":
        return Trajectory(
            frames=tuple(TrajectoryFrame.from_dict(f) for f in data["frames"]),
            frame_source=FrameSource(data.get("frame_source", "unknown")),
            provenance=data.get("provenance", ""),
            drift_estimate=DriftEstimate.from_dict(data.get("drift_estimate", {"known": False})),
        )
