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
import math
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
    sharpness: float = 0.0
    motion_delta: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "width": self.width,
            "height": self.height,
        }
        if self.sharpness > 0.0:
            d["sharpness"] = round(self.sharpness, 2)
        if self.motion_delta > 0.0:
            d["motion_delta"] = round(self.motion_delta, 4)
        return d


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




class InformationGainSelectionStrategy(IFrameSelectionStrategy):
    """Information-gain based frame selection for walk-around videos (Blueprint 16 & 30 Task 9.2).

    Rather than blind uniform sampling, selects keyframes by balancing:
    1. Motion / parallax delta: Drops near-duplicate stationary frames while preserving small movements (useful parallax).
    2. Visual sharpness: Prefers sharp frames over motion-blurred frames.
    3. Coverage / temporal pacing: Enforces min_gap_s to prevent burst clusters, and max_gap_s to prevent excessive gaps.
    4. Deterministic bounding: Always includes first (index 0) and last (index n-1) frames.
    5. Adaptive budgeting: Targets between target_min and target_max (default 100-300 for walk-around, or proportional for short clips).
    """

    def __init__(
        self,
        target_min: int = 100,
        target_max: int = 300,
        min_gap_s: float = 0.25,
        max_gap_s: float = 3.0,
        min_motion_delta: float = 0.03,
    ):
        if target_min < 2:
            raise FrameSelectionError(f"target_min must be >= 2, got {target_min}")
        if target_max < target_min:
            raise FrameSelectionError(f"target_max ({target_max}) cannot be less than target_min ({target_min})")
        self.target_min = target_min
        self.target_max = target_max
        self.min_gap_s = min_gap_s
        self.max_gap_s = max_gap_s
        self.min_motion_delta = min_motion_delta

    @property
    def name(self) -> str:
        return f"information_gain_sampling(min={self.target_min}, max={self.target_max})"

    def select(self, candidates: Sequence[FrameCandidate]) -> List[FrameCandidate]:
        n = len(candidates)
        if n == 0:
            return []
        if n <= self.target_min:
            return list(candidates)

        # Check if candidates provide information gain metrics (sharpness/motion)
        has_metrics = any(c.motion_delta > 0.0 or c.sharpness > 0.0 for c in candidates)
        if not has_metrics:
            target = min(n, (self.target_min + self.target_max) // 2)
            return UniformTimeSamplingStrategy(target).select(candidates)

        # Windowed bucket partitioning by min_gap_s
        # Group candidates into buckets: [k * min_gap_s, (k + 1) * min_gap_s)
        buckets: List[List[FrameCandidate]] = []
        cur_bucket: List[FrameCandidate] = []
        cur_bucket_idx = 0

        for c in candidates:
            b_idx = int(c.timestamp_s / self.min_gap_s)
            if b_idx != cur_bucket_idx and cur_bucket:
                buckets.append(cur_bucket)
                cur_bucket = [c]
                cur_bucket_idx = b_idx
            else:
                cur_bucket.append(c)
                cur_bucket_idx = b_idx

        if cur_bucket:
            buckets.append(cur_bucket)

        selected: List[FrameCandidate] = [candidates[0]]
        last_kept = candidates[0]

        for b in buckets:
            # Score candidates in bucket by information gain
            best_cand = max(b, key=lambda cand: cand.sharpness * (1.0 + 2.0 * cand.motion_delta))
            dt = best_cand.timestamp_s - last_kept.timestamp_s

            is_max_gap = dt >= self.max_gap_s
            is_significant_motion = best_cand.motion_delta >= self.min_motion_delta

            if (is_significant_motion or is_max_gap) and best_cand.frame_index != last_kept.frame_index:
                selected.append(best_cand)
                last_kept = best_cand

        # Always include the last frame
        if selected[-1].frame_index != candidates[-1].frame_index:
            selected.append(candidates[-1])

        # Budget adjustment: prune or interpolate
        if len(selected) > self.target_max:
            while len(selected) > self.target_max:
                worst_idx = -1
                lowest_gain = float("inf")
                for k in range(1, len(selected) - 1):
                    prev_c = selected[k - 1]
                    cur_c = selected[k]
                    next_c = selected[k + 1]
                    temporal_spread = next_c.timestamp_s - prev_c.timestamp_s
                    gain = (cur_c.motion_delta + 0.05) * math.sqrt(max(0.1, cur_c.sharpness)) * temporal_spread
                    if gain < lowest_gain:
                        lowest_gain = gain
                        worst_idx = k
                if worst_idx > 0:
                    selected.pop(worst_idx)
                else:
                    break

        if len(selected) < self.target_min and n > len(selected):
            selected_indices = {c.frame_index for c in selected}
            while len(selected) < min(n, self.target_min):
                largest_gap = -1.0
                gap_pos = -1
                for k in range(len(selected) - 1):
                    gap = selected[k + 1].timestamp_s - selected[k].timestamp_s
                    if gap > largest_gap:
                        largest_gap = gap
                        gap_pos = k
                if gap_pos < 0 or largest_gap <= self.min_gap_s:
                    break

                left_idx = selected[gap_pos].frame_index
                right_idx = selected[gap_pos + 1].frame_index
                mid_candidates = [
                    c for c in candidates[left_idx + 1:right_idx]
                    if c.frame_index not in selected_indices
                ]
                if not mid_candidates:
                    break
                best_mid = max(mid_candidates, key=lambda c: c.sharpness * (1.0 + c.motion_delta))
                selected.insert(gap_pos + 1, best_mid)
                selected_indices.add(best_mid.frame_index)

        seen_idx = set()
        final_out: List[FrameCandidate] = []
        for c in selected:
            if c.frame_index not in seen_idx:
                seen_idx.add(c.frame_index)
                final_out.append(c)
        final_out.sort(key=lambda c: c.frame_index)
        return final_out


__all__ = [
    "FrameCandidate",
    "FrameSelectionError",
    "IFrameSelectionStrategy",
    "InformationGainSelectionStrategy",
    "UniformTimeSamplingStrategy",
]
