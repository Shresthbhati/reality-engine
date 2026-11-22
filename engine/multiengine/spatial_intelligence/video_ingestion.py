"""Video Ingestion, Temporal Coverage, Shot Boundary, and Image Quality Assessment.

Provides production-ready video ingestion and frame sampling for Reality Engine:
- Video metadata extraction (fps, frame count, resolution, duration).
- Temporal frame sampling strategies (UNIFORM, KEYFRAME_SHARPNESS, SHOT_BOUNDARY).
- Shot boundary / scene change detection via inter-frame visual divergence.
- Comprehensive image quality assessment (Laplacian blur variance, mean luma, clipping).
- Frame and sequence provenance tracking (video_id, frame_index, timestamp_sec, shot_id).
- Graceful pure-numpy fallbacks if OpenCV is unavailable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class TemporalSamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    KEYFRAME_SHARPNESS = "keyframe_sharpness"
    SHOT_BOUNDARY = "shot_boundary"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata of an ingested video source."""

    source_id: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 3),
        }


@dataclass(frozen=True)
class ImageQualityReport:
    """Quantitative image quality evaluation."""

    blur_laplacian_var: float
    mean_luma: float
    clipped_high_fraction: float
    clipped_low_fraction: float
    quality_score: float  # [0.0, 1.0]
    verdict: str  # "ok", "warn", "will_be_dropped"
    issues: tuple[str, ...] = ()
    is_usable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_laplacian_var": round(self.blur_laplacian_var, 2),
            "mean_luma": round(self.mean_luma, 1),
            "clipped_high_fraction": round(self.clipped_high_fraction, 4),
            "clipped_low_fraction": round(self.clipped_low_fraction, 4),
            "quality_score": round(self.quality_score, 4),
            "verdict": self.verdict,
            "issues": list(self.issues),
            "is_usable": self.is_usable,
        }


@dataclass(frozen=True)
class FrameProvenance:
    """Temporal and spatial provenance of an extracted video frame or image."""

    source_video_id: str
    frame_index: int
    timestamp_sec: float
    shot_id: int
    quality: ImageQualityReport
    sampling_strategy: str = "uniform"
    sampling_reason: str = "selected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_video_id": self.source_video_id,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "shot_id": self.shot_id,
            "quality": self.quality.to_dict(),
            "sampling_strategy": self.sampling_strategy,
            "sampling_reason": self.sampling_reason,
        }


@dataclass(frozen=True)
class SampledFrame:
    """Sampled video frame ready for perception pipeline ingestion."""

    frame_id: str
    image_rgb: np.ndarray
    provenance: FrameProvenance

    @property
    def width(self) -> int:
        return self.image_rgb.shape[1]

    @property
    def height(self) -> int:
        return self.image_rgb.shape[0]


# ============================================================================
# Image Quality Assessment
# ============================================================================

def compute_image_quality(
    image_rgb_or_gray: np.ndarray,
    blur_warn_threshold: float = 80.0,
    blur_drop_threshold: float = 20.0,
    min_mean_luma: float = 30.0,
    max_mean_luma: float = 225.0,
    max_clip_fraction: float = 0.35,
) -> ImageQualityReport:
    """Computes Laplacian blur variance, luma, clipping, and composite quality score.

    Works with RGB or grayscale numpy arrays. If cv2 is unavailable, uses numpy
    discrete Laplacian convolution.
    """
    if image_rgb_or_gray.ndim == 3:
        if image_rgb_or_gray.shape[2] == 3:
            gray = (
                0.299 * image_rgb_or_gray[..., 0] +
                0.587 * image_rgb_or_gray[..., 1] +
                0.114 * image_rgb_or_gray[..., 2]
            ).astype(np.float32)
        else:
            gray = image_rgb_or_gray[..., 0].astype(np.float32)
    elif image_rgb_or_gray.ndim == 2:
        gray = image_rgb_or_gray.astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got ndim={image_rgb_or_gray.ndim}")

    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return ImageQualityReport(
            blur_laplacian_var=0.0,
            mean_luma=0.0,
            clipped_high_fraction=0.0,
            clipped_low_fraction=0.0,
            quality_score=0.0,
            verdict="will_be_dropped",
            issues=("image_too_small",),
            is_usable=False,
        )

    # Resize to analysis resolution (max 640px) for consistent Laplacian scale
    max_side = max(h, w)
    if max_side > 640:
        scale = 640.0 / max_side
        new_w, new_h = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
        if cv2 is not None:
            gray_scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            step_y = max(1, h // new_h)
            step_x = max(1, w // new_w)
            gray_scaled = gray[::step_y, ::step_x][:new_h, :new_w]
    else:
        gray_scaled = gray

    # Laplacian variance
    if cv2 is not None:
        try:
            lap_var = float(cv2.Laplacian(gray_scaled.astype(np.float64), cv2.CV_64F).var())
        except Exception:
            lap = (
                gray_scaled[:-2, 1:-1] +
                gray_scaled[2:, 1:-1] +
                gray_scaled[1:-1, :-2] +
                gray_scaled[1:-1, 2:] -
                4.0 * gray_scaled[1:-1, 1:-1]
            )
            lap_var = float(np.var(lap))
    else:
        # Pure numpy Laplacian 3x3 kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap = (
            gray_scaled[:-2, 1:-1] +
            gray_scaled[2:, 1:-1] +
            gray_scaled[1:-1, :-2] +
            gray_scaled[1:-1, 2:] -
            4.0 * gray_scaled[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))

    # Luma and clipping
    mean_luma = float(np.mean(gray))
    total_pixels = float(h * w)
    clip_high = float(np.sum(gray >= 250.0)) / total_pixels
    clip_low = float(np.sum(gray <= 5.0)) / total_pixels
    total_clip = clip_high + clip_low

    issues: list[str] = []
    verdict = "ok"

    if lap_var < blur_drop_threshold:
        verdict = "will_be_dropped"
        issues.append(f"severe_motion_blur_lap_{lap_var:.1f}")
    elif lap_var < blur_warn_threshold:
        verdict = "warn"
        issues.append(f"mild_blur_lap_{lap_var:.1f}")

    if mean_luma < min_mean_luma:
        if verdict != "will_be_dropped":
            verdict = "will_be_dropped" if mean_luma < 15.0 else "warn"
        issues.append(f"underexposed_luma_{mean_luma:.1f}")
    elif mean_luma > max_mean_luma:
        if verdict != "will_be_dropped":
            verdict = "will_be_dropped" if mean_luma > 245.0 else "warn"
        issues.append(f"overexposed_luma_{mean_luma:.1f}")

    if total_clip > max_clip_fraction:
        verdict = "will_be_dropped"
        issues.append(f"excessive_clipping_{total_clip:.2%}")

    # Compute continuous quality score in [0.0, 1.0]
    blur_score = min(1.0, math.sqrt(max(0.0, lap_var)) / 25.0)
    luma_dist = abs(mean_luma - 128.0) / 128.0
    luma_score = max(0.0, 1.0 - luma_dist)
    clip_score = max(0.0, 1.0 - (total_clip / 0.50))
    quality_score = float(np.clip(0.50 * blur_score + 0.30 * luma_score + 0.20 * clip_score, 0.0, 1.0))

    is_usable = verdict != "will_be_dropped"

    return ImageQualityReport(
        blur_laplacian_var=lap_var,
        mean_luma=mean_luma,
        clipped_high_fraction=clip_high,
        clipped_low_fraction=clip_low,
        quality_score=quality_score,
        verdict=verdict,
        issues=tuple(issues),
        is_usable=is_usable,
    )


# ============================================================================
# Shot Boundary Detection
# ============================================================================

def compute_interframe_divergence(frame_a_rgb: np.ndarray, frame_b_rgb: np.ndarray) -> float:
    """Computes color histogram divergence between two frames to detect shot boundaries."""
    # Resize to small 64x64 thumbnail for fast comparison
    ha, wa = frame_a_rgb.shape[:2]
    hb, wb = frame_b_rgb.shape[:2]

    step_ya, step_xa = max(1, ha // 64), max(1, wa // 64)
    step_yb, step_xb = max(1, hb // 64), max(1, wb // 64)

    sub_a = frame_a_rgb[::step_ya, ::step_xa][:64, :64].astype(np.float32)
    sub_b = frame_b_rgb[::step_yb, ::step_xb][:64, :64].astype(np.float32)

    min_h = min(sub_a.shape[0], sub_b.shape[0])
    min_w = min(sub_a.shape[1], sub_b.shape[1])
    sub_a = sub_a[:min_h, :min_w]
    sub_b = sub_b[:min_h, :min_w]

    # Mean absolute difference in RGB
    diff = np.mean(np.abs(sub_a - sub_b))
    return float(diff)


# ============================================================================
# Video Ingestion Pipeline
# ============================================================================

@dataclass(frozen=True)
class SamplingExecutionReport:
    """Telemetry report of video sampling execution."""

    strategy: str
    total_frames_evaluated: int
    frames_sampled: int
    frames_dropped_by_reason: dict[str, int]
    shots_detected: int
    average_quality_score: float
    mean_interframe_divergence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_frames_evaluated": self.total_frames_evaluated,
            "frames_sampled": self.frames_sampled,
            "frames_dropped_by_reason": dict(self.frames_dropped_by_reason),
            "shots_detected": self.shots_detected,
            "average_quality_score": round(self.average_quality_score, 4),
            "mean_interframe_divergence": round(self.mean_interframe_divergence, 2),
        }


class VideoIngestionPipeline:
    """Production video ingestion engine with selectable temporal sampling and quality gating."""

    def __init__(
        self,
        target_fps: float = 2.0,
        max_frame_budget: int = 120,
        strategy: TemporalSamplingStrategy = TemporalSamplingStrategy.ADAPTIVE,
        min_quality_score: float = 0.20,
        shot_divergence_threshold: float = 40.0,
    ) -> None:
        self.target_fps = target_fps
        self.max_frame_budget = max_frame_budget
        self.strategy = strategy
        self.min_quality_score = min_quality_score
        self.shot_divergence_threshold = shot_divergence_threshold
        self.last_report: SamplingExecutionReport | None = None

    def _sample_uniform(
        self,
        frames_data: list[tuple[int, float, np.ndarray, ImageQualityReport]],
        source_id: str,
        fps: float,
    ) -> tuple[list[SampledFrame], dict[str, int], int]:
        stride = max(1, int(round(fps / max(0.1, self.target_fps))))
        dropped: dict[str, int] = defaultdict(int)
        sampled: list[SampledFrame] = []
        shot_id = 0
        last_kept_rgb: np.ndarray | None = None

        for idx, t_sec, img, qr in frames_data:
            if idx % stride != 0:
                dropped["temporal_stride_skip"] += 1
                continue

            if not qr.is_usable or qr.quality_score < self.min_quality_score:
                dropped["quality_gating"] += 1
                continue

            if last_kept_rgb is not None:
                div = compute_interframe_divergence(last_kept_rgb, img)
                if div > self.shot_divergence_threshold:
                    shot_id += 1

            prov = FrameProvenance(
                source_video_id=source_id,
                frame_index=idx,
                timestamp_sec=t_sec,
                shot_id=shot_id,
                quality=qr,
                sampling_strategy="uniform",
                sampling_reason=f"stride_{stride}",
            )
            sampled.append(SampledFrame(
                frame_id=f"{source_id}_f{idx:06d}",
                image_rgb=img,
                provenance=prov,
            ))
            last_kept_rgb = img
            if len(sampled) >= self.max_frame_budget:
                dropped["budget_exceeded"] += max(0, len(frames_data) - idx - 1)
                break

        return sampled, dropped, shot_id + 1

    def _sample_keyframe_sharpness(
        self,
        frames_data: list[tuple[int, float, np.ndarray, ImageQualityReport]],
        source_id: str,
        fps: float,
    ) -> tuple[list[SampledFrame], dict[str, int], int]:
        window_size = max(1, int(round(fps / max(0.1, self.target_fps))))
        dropped: dict[str, int] = defaultdict(int)
        sampled: list[SampledFrame] = []
        shot_id = 0
        last_kept_rgb: np.ndarray | None = None

        # Group into sliding non-overlapping windows of size window_size
        for w_start in range(0, len(frames_data), window_size):
            window = frames_data[w_start:w_start + window_size]
            usable_in_window = [
                (idx, t, img, qr) for idx, t, img, qr in window
                if qr.is_usable and qr.quality_score >= self.min_quality_score
            ]
            if not usable_in_window:
                dropped["window_all_unusable"] += len(window)
                continue

            # Pick the single sharpest frame in the window
            best_cand = max(
                usable_in_window,
                key=lambda item: (item[3].quality_score, item[3].blur_laplacian_var),
            )
            dropped["suppressed_by_sharper_neighbor"] += (len(usable_in_window) - 1)
            dropped["quality_gating"] += (len(window) - len(usable_in_window))

            idx, t_sec, img, qr = best_cand
            if last_kept_rgb is not None:
                div = compute_interframe_divergence(last_kept_rgb, img)
                if div > self.shot_divergence_threshold:
                    shot_id += 1

            prov = FrameProvenance(
                source_video_id=source_id,
                frame_index=idx,
                timestamp_sec=t_sec,
                shot_id=shot_id,
                quality=qr,
                sampling_strategy="keyframe_sharpness",
                sampling_reason=f"peak_sharpness_score_{qr.quality_score:.2f}",
            )
            sampled.append(SampledFrame(
                frame_id=f"{source_id}_f{idx:06d}",
                image_rgb=img,
                provenance=prov,
            ))
            last_kept_rgb = img
            if len(sampled) >= self.max_frame_budget:
                break

        return sampled, dropped, shot_id + 1

    def _sample_shot_boundary(
        self,
        frames_data: list[tuple[int, float, np.ndarray, ImageQualityReport]],
        source_id: str,
        fps: float,
    ) -> tuple[list[SampledFrame], dict[str, int], int]:
        dropped: dict[str, int] = defaultdict(int)
        sampled: list[SampledFrame] = []

        if not frames_data:
            return sampled, dropped, 0

        # Segment frames into shots based on divergence
        shots: list[list[tuple[int, float, np.ndarray, ImageQualityReport]]] = []
        current_shot: list[tuple[int, float, np.ndarray, ImageQualityReport]] = [frames_data[0]]

        for i in range(1, len(frames_data)):
            div = compute_interframe_divergence(frames_data[i - 1][2], frames_data[i][2])
            if div > self.shot_divergence_threshold:
                shots.append(current_shot)
                current_shot = [frames_data[i]]
            else:
                current_shot.append(frames_data[i])
        if current_shot:
            shots.append(current_shot)

        # For each shot, extract entry, central sharpest, and exit keyframes
        target_stride = max(1, int(round(fps / max(0.1, self.target_fps))))
        for s_idx, shot in enumerate(shots):
            usable = [f for f in shot if f[3].is_usable and f[3].quality_score >= self.min_quality_score]
            if not usable:
                dropped["shot_all_unusable"] += len(shot)
                continue

            dropped["quality_gating"] += (len(shot) - len(usable))

            selected_indices: set[int] = set()

            # 1. Entry frame (first usable)
            entry_f = usable[0]
            selected_indices.add(entry_f[0])

            # 2. Central peak sharpness frame
            best_f = max(usable, key=lambda f: (f[3].quality_score, f[3].blur_laplacian_var))
            selected_indices.add(best_f[0])

            # 3. Exit frame if shot is long enough (>= 4 frames)
            if len(usable) >= 4:
                exit_f = usable[-1]
                selected_indices.add(exit_f[0])

            # 4. If shot is very long, add intermediate frames spaced by target_stride
            if len(usable) > target_stride * 2:
                for step in range(target_stride, len(usable) - 1, target_stride):
                    selected_indices.add(usable[step][0])

            # Build SampledFrame for selected items in chronological order
            shot_sampled = [f for f in usable if f[0] in selected_indices]
            dropped["suppressed_within_shot"] += (len(usable) - len(shot_sampled))

            for idx, t_sec, img, qr in shot_sampled:
                reason = "shot_entry" if idx == entry_f[0] else ("shot_exit" if idx == usable[-1][0] else "shot_keyframe")
                prov = FrameProvenance(
                    source_video_id=source_id,
                    frame_index=idx,
                    timestamp_sec=t_sec,
                    shot_id=s_idx,
                    quality=qr,
                    sampling_strategy="shot_boundary",
                    sampling_reason=reason,
                )
                sampled.append(SampledFrame(
                    frame_id=f"{source_id}_f{idx:06d}",
                    image_rgb=img,
                    provenance=prov,
                ))
                if len(sampled) >= self.max_frame_budget:
                    break
            if len(sampled) >= self.max_frame_budget:
                break

        return sampled, dropped, len(shots)

    def _sample_adaptive(
        self,
        frames_data: list[tuple[int, float, np.ndarray, ImageQualityReport]],
        source_id: str,
        fps: float,
    ) -> tuple[list[SampledFrame], dict[str, int], int]:
        base_stride = max(1, int(round(fps / max(0.1, self.target_fps))))
        dropped: dict[str, int] = defaultdict(int)
        sampled: list[SampledFrame] = []
        shot_id = 0
        last_kept_rgb: np.ndarray | None = None
        last_kept_idx = -9999

        for idx, t_sec, img, qr in frames_data:
            if not qr.is_usable or qr.quality_score < self.min_quality_score:
                dropped["quality_gating"] += 1
                continue

            if last_kept_rgb is None:
                # Always accept first usable frame
                prov = FrameProvenance(
                    source_video_id=source_id,
                    frame_index=idx,
                    timestamp_sec=t_sec,
                    shot_id=shot_id,
                    quality=qr,
                    sampling_strategy="adaptive",
                    sampling_reason="initial_frame",
                )
                sampled.append(SampledFrame(
                    frame_id=f"{source_id}_f{idx:06d}",
                    image_rgb=img,
                    provenance=prov,
                ))
                last_kept_rgb = img
                last_kept_idx = idx
                continue

            div = compute_interframe_divergence(last_kept_rgb, img)
            frames_since_last = idx - last_kept_idx

            if div > self.shot_divergence_threshold:
                shot_id += 1
                reason = "shot_transition"
                accept = True
            elif div >= 20.0 and frames_since_last >= max(1, base_stride // 2):
                # Rapid motion / panning: sample more densely
                reason = f"high_motion_div_{div:.1f}"
                accept = True
            elif div >= 8.0 and frames_since_last >= base_stride:
                # Standard motion: sample at target stride
                reason = f"normal_motion_div_{div:.1f}"
                accept = True
            elif div < 8.0 and frames_since_last >= base_stride * 2:
                # Stationary camera: throttle sampling to prevent redundant views
                reason = f"stationary_timeout_div_{div:.1f}"
                accept = True
            else:
                dropped["redundant_or_substride"] += 1
                accept = False

            if accept:
                prov = FrameProvenance(
                    source_video_id=source_id,
                    frame_index=idx,
                    timestamp_sec=t_sec,
                    shot_id=shot_id,
                    quality=qr,
                    sampling_strategy="adaptive",
                    sampling_reason=reason,
                )
                sampled.append(SampledFrame(
                    frame_id=f"{source_id}_f{idx:06d}",
                    image_rgb=img,
                    provenance=prov,
                ))
                last_kept_rgb = img
                last_kept_idx = idx

            if len(sampled) >= self.max_frame_budget:
                break

        return sampled, dropped, shot_id + 1

    def ingest_image_sequence(
        self,
        images_rgb: Sequence[np.ndarray],
        source_id: str = "image_seq",
        fps: float = 2.0,
    ) -> list[SampledFrame]:
        """Ingests an ordered sequence of images with selected temporal sampling strategy."""
        frames_data = [
            (idx, float(idx) / max(0.1, fps), img, compute_image_quality(img))
            for idx, img in enumerate(images_rgb)
        ]

        if self.strategy == TemporalSamplingStrategy.UNIFORM:
            sampled, dropped, num_shots = self._sample_uniform(frames_data, source_id, fps)
        elif self.strategy == TemporalSamplingStrategy.KEYFRAME_SHARPNESS:
            sampled, dropped, num_shots = self._sample_keyframe_sharpness(frames_data, source_id, fps)
        elif self.strategy == TemporalSamplingStrategy.SHOT_BOUNDARY:
            sampled, dropped, num_shots = self._sample_shot_boundary(frames_data, source_id, fps)
        else:  # ADAPTIVE
            sampled, dropped, num_shots = self._sample_adaptive(frames_data, source_id, fps)

        # Compute telemetry
        avg_q = float(np.mean([s.provenance.quality.quality_score for s in sampled])) if sampled else 0.0
        divs: list[float] = []
        for i in range(1, len(sampled)):
            divs.append(compute_interframe_divergence(sampled[i - 1].image_rgb, sampled[i].image_rgb))
        mean_div = float(np.mean(divs)) if divs else 0.0

        self.last_report = SamplingExecutionReport(
            strategy=self.strategy.value if isinstance(self.strategy, TemporalSamplingStrategy) else str(self.strategy),
            total_frames_evaluated=len(frames_data),
            frames_sampled=len(sampled),
            frames_dropped_by_reason=dropped,
            shots_detected=num_shots,
            average_quality_score=avg_q,
            mean_interframe_divergence=mean_div,
        )
        return sampled

    def ingest_video_file(
        self,
        video_path: str | Path,
        video_id: str | None = None,
    ) -> tuple[VideoMetadata, list[SampledFrame]]:
        """Extracts and samples frames from a video file with selectable strategy."""
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vid_id = video_id or path.stem

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to decode video files directly.")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = float(total_frames / fps) if fps > 0 else 0.0

            metadata = VideoMetadata(
                source_id=vid_id,
                fps=fps,
                frame_count=total_frames,
                width=width,
                height=height,
                duration_sec=duration,
            )

            # Read all frames (or subsampled pre-scan for very long videos)
            raw_frames: list[tuple[int, float, np.ndarray, ImageQualityReport]] = []
            frame_idx = 0
            # Pre-stride if video is massive (> 1000 frames) to maintain performance
            read_stride = max(1, int(round(fps / (self.target_fps * 4.0)))) if total_frames > 600 else 1

            while cap.isOpened() and frame_idx < total_frames:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break

                if frame_idx % read_stride == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    qr = compute_image_quality(rgb)
                    t_sec = float(frame_idx) / max(0.1, fps)
                    raw_frames.append((frame_idx, t_sec, rgb, qr))

                frame_idx += 1

            # Dispatch through sampling strategy
            if self.strategy == TemporalSamplingStrategy.UNIFORM:
                sampled, dropped, num_shots = self._sample_uniform(raw_frames, vid_id, fps)
            elif self.strategy == TemporalSamplingStrategy.KEYFRAME_SHARPNESS:
                sampled, dropped, num_shots = self._sample_keyframe_sharpness(raw_frames, vid_id, fps)
            elif self.strategy == TemporalSamplingStrategy.SHOT_BOUNDARY:
                sampled, dropped, num_shots = self._sample_shot_boundary(raw_frames, vid_id, fps)
            else:  # ADAPTIVE
                sampled, dropped, num_shots = self._sample_adaptive(raw_frames, vid_id, fps)

            avg_q = float(np.mean([s.provenance.quality.quality_score for s in sampled])) if sampled else 0.0
            divs: list[float] = []
            for i in range(1, len(sampled)):
                divs.append(compute_interframe_divergence(sampled[i - 1].image_rgb, sampled[i].image_rgb))
            mean_div = float(np.mean(divs)) if divs else 0.0

            self.last_report = SamplingExecutionReport(
                strategy=self.strategy.value if isinstance(self.strategy, TemporalSamplingStrategy) else str(self.strategy),
                total_frames_evaluated=len(raw_frames),
                frames_sampled=len(sampled),
                frames_dropped_by_reason=dropped,
                shots_detected=num_shots,
                average_quality_score=avg_q,
                mean_interframe_divergence=mean_div,
            )
            return metadata, sampled
        finally:
            cap.release()
