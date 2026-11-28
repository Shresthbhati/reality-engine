"""Frame selection (spec sec 3 MEDIA PREPROCESSING: "Do not blindly
process every video frame. Create an extensible frame-selection
strategy").

Pure selection policy over ALREADY-ENUMERATED frame metadata: a video
importer enumerates FrameCandidates (index, timestamp, size -- cheap
grab-only probing, no full decode) and hands the list to a strategy,
which returns the subset worth reconstructing from. Because strategies
see only deterministic metadata and use no clocks or RNG, the same
video always selects the same frames -- replayable capture, the same
discipline as evidence/packages.py's content-derived ids.

Policies are deliberately separated from codecs (no cv2 import here):
strategies are unit-testable against synthetic candidate lists, and a
future real-time or GPU-side enumerator can reuse them unchanged.

The default policy is UniformTimeSamplingStrategy: evenly spaced
samples across the video's duration, always including first and last
frame. There is deliberately NO "process every frame" strategy in the
default set -- an unbounded all-frames policy is exactly the blind
processing the spec forbids; callers who truly want it can build a
trivial strategy themselves and own the consequences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


class FrameSelectionError(ValueError):
    """Raised when a strategy cannot make an honest selection."""


@dataclass(frozen=True)
class FrameCandidate:
    """Metadata for one frame of one video, as produced by enumeration.

    `timestamp_s` is the frame's position WITHIN the video (seconds
    from video start, from the container's own timing) -- it is not an
    acquisition timestamp and never claims to be one.
    """

    frame_index: int
    timestamp_s: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "width": self.width,
            "height": self.height,
        }


class IFrameSelectionStrategy(ABC):
    """Interface for frame-selection policies (spec sec 3: extensible)."""

    @abstractmethod
    def select(self, candidates: Sequence[FrameCandidate]) -> List[FrameCandidate]:
        """Return the subset of `candidates` worth reconstructing from.

        Implementations must be deterministic (same input -> same
        selection) and must preserve the input's order in their output.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable policy name, recorded in import metadata for provenance."""


class UniformTimeSamplingStrategy(IFrameSelectionStrategy):
    """Default policy: evenly spaced samples across the video.

    Selects `target_count` frames (or all frames when the video has
    fewer), spread by round(i * (n-1) / (target-1)) so the FIRST and
    LAST frames are always included and spacing is as uniform as an
    integer index grid allows. Pure integer/rational index math -- no
    RNG, no clocks, no floating accumulation error: the same candidate
    list selects byte-identically on every run and platform.

    Why uniform time sampling: SfM backends need overlapping views
    spread across the capture trajectory, not redundant adjacent
    frames; uniform spacing maximizes baseline per frame budget and is
    the honest default when nothing is known about scene motion.
    """

    def __init__(self, target_count: int = 24):
        if target_count < 2:
            raise FrameSelectionError(
                f"target_count must be >= 2 (a reconstruction needs "
                f"multiple views), got {target_count}"
            )
        self.target_count = target_count

    @property
    def name(self) -> str:
        return f"uniform_time_sampling(target={self.target_count})"

    def select(self, candidates: Sequence[FrameCandidate]) -> List[FrameCandidate]:
        n = len(candidates)
        if n == 0:
            return []
        if n <= self.target_count:
            return list(candidates)
        indices = []
        for i in range(self.target_count):
            indices.append(round(i * (n - 1) / (self.target_count - 1)))
        # Rounding can theoretically collide on tiny gaps; dedupe
        # preserving order (deterministically) -- first occurrence wins.
        seen: set = set()
        selected: List[FrameCandidate] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                selected.append(candidates[idx])
        return selected


__all__ = [
    "FrameCandidate",
    "FrameSelectionError",
    "IFrameSelectionStrategy",
    "UniformTimeSamplingStrategy",
]
