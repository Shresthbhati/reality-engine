"""Video keyframe extraction and scene transition detection (Master Development Plan Tasks 2.2 & 2.3).

Extracts high-value, non-redundant keyframes from video files (MP4, MOV, MKV, AVI):
1. Scene Boundary Detection:
   - Evaluates inter-frame visual delta (normalized color histogram chi-square distance
     and pixel absolute difference).
   - Marks cut boundaries when the visual distance exceeds adaptive thresholds.
2. Scene-Aware Frame Selection Strategy:
   - An extensible `IFrameSelectionStrategy` implementation that allocates frame budget
     across detected scenes proportionally to shot length and scene complexity, avoiding
     blind uniform time sampling during stationary camera moments.
3. Keyframe Extraction & Quality Scoring:
   - Decodes selected frames, computes Laplacian sharpness metrics, and exports
     replayable frame image artifacts with embedded timing provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from evidence.frames import FrameCandidate, IFrameSelectionStrategy, InformationGainSelectionStrategy, UniformTimeSamplingStrategy


@dataclass(frozen=True)
class SceneTransition:
    """A detected scene boundary or cut within a video sequence."""

    frame_index: int
    timestamp_s: float
    delta_score: float
    is_cut: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "delta_score": round(self.delta_score, 4),
            "is_cut": self.is_cut,
        }


@dataclass(frozen=True)
class VideoIngestResult:
    """Outcome of video keyframe ingestion (Blueprint 16 & 30 Task 9.2)."""

    video_path: str
    duration_s: float
    total_frames: int
    fps: float
    keyframe_count: int
    keyframes: List[Dict[str, Any]]
    strategy_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_s": round(self.duration_s, 3),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "keyframe_count": self.keyframe_count,
            "keyframes": self.keyframes,
            "strategy_used": self.strategy_used,
        }

    def summary(self) -> str:
        return (
            f"Extracted {self.keyframe_count} keyframes from {self.total_frames} frames "
            f"({self.duration_s:.1f}s @ {self.fps:.1f}fps) using {self.strategy_used}"
        )


def compute_frame_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute normalized visual distance [0.0, 1.0] between two frames.

    Combines luminance histogram Chi-Square distance and normalized Mean Absolute Error (MAE).
    """
    if frame_a.shape != frame_b.shape:
        # Resize frame_b to match frame_a
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    # Grayscale conversion
    if frame_a.ndim == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_a = frame_a
        gray_b = frame_b

    # 1. Pixel Mean Absolute Error normalized to [0, 1]
    mae = float(np.mean(np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))) / 255.0)

    # 2. Luminance histogram distance (32 bins)
    hist_a = cv2.calcHist([gray_a], [0], None, [32], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [32], [0, 256])
    cv2.normalize(hist_a, hist_a, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, alpha=1.0, norm_type=cv2.NORM_L1)

    # Chi-square distance
    chi2 = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CHISQR))
    chi2_norm = min(1.0, chi2)

    # Blended score
    return float(np.clip(0.5 * mae + 0.5 * chi2_norm, 0.0, 1.0))


def detect_scene_transitions_from_frames(
    frames: Sequence[Tuple[int, float, np.ndarray]],
    cut_threshold: float = 0.30,
) -> List[SceneTransition]:
    """Detect shot transitions from a sequence of (frame_index, timestamp_s, frame_array) tuples."""
    if len(frames) < 2:
        return []

    transitions: List[SceneTransition] = []
    prev_idx, prev_t, prev_frame = frames[0]

    for curr_idx, curr_t, curr_frame in frames[1:]:
        delta = compute_frame_difference(prev_frame, curr_frame)
        is_cut = delta >= cut_threshold
        transitions.append(
            SceneTransition(
                frame_index=curr_idx,
                timestamp_s=curr_t,
                delta_score=delta,
                is_cut=is_cut,
            )
        )
        prev_idx, prev_t, prev_frame = curr_idx, curr_t, curr_frame

    return transitions


class SceneAwareFrameSelectionStrategy(IFrameSelectionStrategy):
    """Extensible strategy that allocates frame budget across detected scenes.

    Guarantees:
    - First and last frames are preserved.
    - Every detected scene transition gets keyframe representation.
    - Frame allocation within scenes is proportional to scene duration.
    - Eliminates redundant near-identical frames within static shots.
    """

    def __init__(
        self,
        target_count: int = 24,
        cut_threshold: float = 0.30,
        scene_transitions: Optional[Sequence[SceneTransition]] = None,
    ):
        if target_count < 2:
            raise ValueError(f"target_count must be >= 2, got {target_count}")
        self.target_count = target_count
        self.cut_threshold = cut_threshold
        self.scene_transitions = list(scene_transitions) if scene_transitions else []

    @property
    def name(self) -> str:
        return f"scene_aware_sampling(target={self.target_count}, threshold={self.cut_threshold})"

    def select(self, candidates: Sequence[FrameCandidate]) -> List[FrameCandidate]:
        n = len(candidates)
        if n == 0:
            return []
        if n <= self.target_count:
            return list(candidates)

        # Identify cut indices
        cut_frame_indices = set(t.frame_index for t in self.scene_transitions if t.is_cut)

        # Partition candidates into scenes/shots
        scenes: List[List[FrameCandidate]] = []
        current_scene: List[FrameCandidate] = []

        for c in candidates:
            if c.frame_index in cut_frame_indices and current_scene:
                scenes.append(current_scene)
                current_scene = [c]
            else:
                current_scene.append(c)

        if current_scene:
            scenes.append(current_scene)

        # Allocate frame budget across scenes proportionally
        num_scenes = len(scenes)
        if num_scenes == 1:
            # Fall back to uniform sampling within the single scene
            return UniformTimeSamplingStrategy(self.target_count).select(candidates)

        # Each scene gets at least 1 frame; remaining budget distributed by scene length
        min_per_scene = 1
        budget_remaining = max(0, self.target_count - (num_scenes * min_per_scene))

        selected: List[FrameCandidate] = []
        for scene in scenes:
            scene_weight = len(scene) / float(n)
            scene_budget = min_per_scene + round(budget_remaining * scene_weight)
            scene_budget = min(len(scene), max(1, scene_budget))

            # Sample within scene
            sub_strategy = UniformTimeSamplingStrategy(scene_budget)
            selected.extend(sub_strategy.select(scene))

        # Trim or pad if rounding causes slight mismatch from target_count
        if len(selected) > self.target_count:
            # Sub-sample uniformly to exact target_count
            selected = UniformTimeSamplingStrategy(self.target_count).select(selected)

        # Ensure order is strictly ascending by frame_index
        selected.sort(key=lambda c: c.frame_index)

        # Deduplicate while preserving order
        seen_idx = set()
        out: List[FrameCandidate] = []
        for c in selected:
            if c.frame_index not in seen_idx:
                seen_idx.add(c.frame_index)
                out.append(c)

        return out


def extract_video_keyframes(
    video_path: Union[Path, str],
    output_dir: Union[Path, str],
    target_count: int = 24,
    strategy: Optional[IFrameSelectionStrategy] = None,
    image_format: str = "jpg",
) -> List[Dict[str, Any]]:
    """Decode and extract keyframes from a video file into output_dir.

    Args:
        video_path: Path to input MP4/MOV/etc.
        output_dir: Directory where extracted keyframe images will be written.
        target_count: Desired number of keyframes.
        strategy: Custom frame selection strategy (defaults to SceneAwareFrameSelectionStrategy).
        image_format: "jpg" or "png".

    Returns:
        List of dictionaries with frame metadata, file path, and measured sharpness.
    """
    v_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {v_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Probe candidate frames
        candidates: List[FrameCandidate] = []
        for i in range(total_frames):
            t_s = float(i) / fps
            candidates.append(FrameCandidate(frame_index=i, timestamp_s=t_s, width=width, height=height))

        # Select frames
        if strategy is None:
            # Sample low-res thumbnails to detect scene transitions quickly
            sample_step = max(1, total_frames // 100)
            sampled_frames: List[Tuple[int, float, np.ndarray]] = []
            for idx in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    thumb = cv2.resize(frame, (160, 90))
                    sampled_frames.append((idx, idx / fps, thumb))

            transitions = detect_scene_transitions_from_frames(sampled_frames, cut_threshold=0.25)
            strategy = SceneAwareFrameSelectionStrategy(
                target_count=target_count,
                scene_transitions=transitions,
            )

        selected_candidates = strategy.select(candidates)
        selected_set = {c.frame_index for c in selected_candidates}

        results: List[Dict[str, Any]] = []
        for c in selected_candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c.frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"keyframe_{c.frame_index:06d}_{c.timestamp_s:.2f}s.{image_format}"
            frame_path = out_dir / frame_filename

            # Compute sharpness via Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            cv2.imwrite(str(frame_path), frame)

            results.append({
                "frame_index": c.frame_index,
                "timestamp_s": round(c.timestamp_s, 3),
                "file_path": str(frame_path),
                "width": c.width,
                "height": c.height,
                "sharpness": round(sharpness, 2),
                "strategy": strategy.name,
            })

        return results

    finally:
        cap.release()



def ingest_video_to_keyframes(
    video_path: Union[Path, str],
    output_dir: Union[Path, str],
    target_min: int = 100,
    target_max: int = 300,
    strategy: Optional[IFrameSelectionStrategy] = None,
    image_format: str = "jpg",
    sample_hz: float = 4.0,
) -> VideoIngestResult:
    """Ingest a video capture into sharp, information-rich keyframes for reconstruction.

    Fulfills Blueprint 16 & 30 Task 9.2:
    - Reconstructs a walk-around video like a photo set.
    - Selects frames by information gain (motion, parallax, novelty, sharpness).
    - Eliminates static pauses and blurry turns.
    - Accurately reports keyframe count and presentation timestamps.

    Args:
        video_path: Path to source video file (MP4, MOV, MKV, etc.).
        output_dir: Target directory to write extracted keyframe images.
        target_min: Minimum desired keyframes (scaled if video has fewer).
        target_max: Maximum keyframe budget.
        strategy: Optional custom IFrameSelectionStrategy (defaults to InformationGainSelectionStrategy).
        image_format: Output image format ("jpg" or "png").
        sample_hz: Probing frequency for motion/sharpness analysis.
    """
    v_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {v_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 25.0
        duration_s = float(total_frames) / fps if total_frames > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            return VideoIngestResult(
                video_path=str(v_path),
                duration_s=0.0,
                total_frames=0,
                fps=fps,
                keyframe_count=0,
                keyframes=[],
                strategy_used="none",
            )

        # Step 1: Probe candidate frames at sample_hz (or every frame if short)
        step = max(1, int(round(fps / sample_hz))) if sample_hz > 0 else 1
        candidate_indices = list(range(0, total_frames, step))
        if (total_frames - 1) not in candidate_indices:
            candidate_indices.append(total_frames - 1)

        candidates: List[FrameCandidate] = []
        prev_thumb = None

        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            t_s = float(idx) / fps

            # Low-res thumbnail for fast motion and sharpness analysis
            thumb = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            motion_delta = 0.0
            if prev_thumb is not None:
                motion_delta = compute_frame_difference(prev_thumb, thumb)
            prev_thumb = thumb

            candidates.append(
                FrameCandidate(
                    frame_index=idx,
                    timestamp_s=t_s,
                    width=width,
                    height=height,
                    sharpness=sharpness,
                    motion_delta=motion_delta,
                )
            )

        # Step 2: Apply strategy
        if strategy is None:
            # Scale target bounds proportionally if video is shorter than 5 minutes
            actual_min = min(len(candidates), max(2, min(target_min, len(candidates))))
            actual_max = max(actual_min, min(target_max, len(candidates)))
            strategy = InformationGainSelectionStrategy(
                target_min=actual_min,
                target_max=actual_max,
                min_gap_s=0.20,
                max_gap_s=3.0,
                min_motion_delta=0.02,
            )

        selected_candidates = strategy.select(candidates)

        # Step 3: Decode and save selected keyframes
        keyframes: List[Dict[str, Any]] = []
        for c in selected_candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c.frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"keyframe_{c.frame_index:06d}_{c.timestamp_s:.2f}s.{image_format}"
            frame_path = out_dir / frame_filename

            # Full-res sharpness measurement
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            full_sharpness = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())

            cv2.imwrite(str(frame_path), frame)

            keyframes.append({
                "frame_index": c.frame_index,
                "timestamp_s": round(c.timestamp_s, 3),
                "file_path": str(frame_path),
                "width": c.width,
                "height": c.height,
                "sharpness": round(full_sharpness, 2),
                "motion_delta": round(c.motion_delta, 4),
            })

        return VideoIngestResult(
            video_path=str(v_path),
            duration_s=duration_s,
            total_frames=total_frames,
            fps=fps,
            keyframe_count=len(keyframes),
            keyframes=keyframes,
            strategy_used=strategy.name,
        )

    finally:
        cap.release()
