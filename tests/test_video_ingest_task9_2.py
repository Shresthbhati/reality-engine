"""Unit tests for Task 9.2: Video keyframe ingestion with information-gain selection."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from evidence.frames import (
    FrameCandidate,
    FrameSelectionError,
    InformationGainSelectionStrategy,
)
from evidence.video import (
    VideoIngestResult,
    ingest_video_to_keyframes,
)


def test_information_gain_strategy_validation():
    with pytest.raises(FrameSelectionError):
        InformationGainSelectionStrategy(target_min=1)
    with pytest.raises(FrameSelectionError):
        InformationGainSelectionStrategy(target_min=10, target_max=5)


def test_information_gain_strategy_endpoints_and_deduplication():
    # 200 candidates over 10 seconds (20 fps)
    candidates = [
        FrameCandidate(
            frame_index=i,
            timestamp_s=i * 0.05,
            width=640,
            height=480,
            sharpness=100.0 + (i % 10) * 5.0,
            motion_delta=0.08 if i % 4 == 0 else 0.01,
        )
        for i in range(200)
    ]

    strategy = InformationGainSelectionStrategy(
        target_min=10,
        target_max=30,
        min_gap_s=0.20,
        min_motion_delta=0.03,
    )
    selected = strategy.select(candidates)

    # First and last frames must always be selected
    assert selected[0].frame_index == 0
    assert selected[-1].frame_index == 199

    # Must be monotonically increasing in index and timestamp
    for k in range(len(selected) - 1):
        assert selected[k + 1].frame_index > selected[k].frame_index
        assert selected[k + 1].timestamp_s > selected[k].timestamp_s

    # Budget constraints
    assert 10 <= len(selected) <= 30


def test_information_gain_prefers_sharp_over_blurry():
    # 5 frames close together: candidate 2 is sharp, candidates 0,1,3,4 are blurry
    candidates = [
        FrameCandidate(0, 0.0, 640, 480, sharpness=10.0, motion_delta=0.1),
        FrameCandidate(1, 0.05, 640, 480, sharpness=20.0, motion_delta=0.1),
        FrameCandidate(2, 0.10, 640, 480, sharpness=250.0, motion_delta=0.1),
        FrameCandidate(3, 0.15, 640, 480, sharpness=15.0, motion_delta=0.1),
        FrameCandidate(4, 0.20, 640, 480, sharpness=12.0, motion_delta=0.1),
        FrameCandidate(5, 1.00, 640, 480, sharpness=200.0, motion_delta=0.1),
    ]

    strategy = InformationGainSelectionStrategy(
        target_min=2,
        target_max=4,
        min_gap_s=0.25,
        min_motion_delta=0.02,
    )
    selected = strategy.select(candidates)
    kept_indices = [c.frame_index for c in selected]

    # Must include candidate 2 (sharpest) in the first window, rather than blurred candidate 1
    assert 2 in kept_indices
    assert selected[0].frame_index == 0
    assert selected[-1].frame_index == 5


def test_information_gain_skips_stationary_near_duplicates():
    # Sequence with 20 stationary frames (motion_delta near 0) followed by 10 moving frames
    candidates = []
    # 0 to 19: stationary
    for i in range(20):
        candidates.append(
            FrameCandidate(i, i * 0.05, 640, 480, sharpness=100.0, motion_delta=0.002)
        )
    # 20 to 35: moving
    for i in range(20, 36):
        candidates.append(
            FrameCandidate(i, i * 0.05, 640, 480, sharpness=120.0, motion_delta=0.12)
        )

    strategy = InformationGainSelectionStrategy(
        target_min=4,
        target_max=12,
        min_gap_s=0.15,
        max_gap_s=2.0,
        min_motion_delta=0.05,
    )
    selected = strategy.select(candidates)
    stationary_selected = [c for c in selected if c.frame_index < 20]
    moving_selected = [c for c in selected if c.frame_index >= 20]

    # Most of the selected frames should come from the moving segment
    assert len(moving_selected) >= len(stationary_selected)
    assert selected[0].frame_index == 0
    assert selected[-1].frame_index == 35


def test_ingest_video_to_keyframes_end_to_end(tmp_path):
    vid_path = tmp_path / "walkaround.mp4"
    out_dir = tmp_path / "keyframes"

    fps = 20
    w, h = 320, 240
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()

    # Generate 60 frames (3 seconds):
    # Frames 0-20: stationary scene
    # Frames 21-45: panning textured scene
    # Frames 46-59: stationary scene
    base_img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(base_img, (100, 100), 40, (0, 255, 0), -1)

    for i in range(60):
        if 20 <= i < 45:
            shift = (i - 20) * 4
            frame = np.roll(base_img, shift, axis=1)
        else:
            frame = base_img.copy()
        writer.write(frame)
    writer.release()

    res = ingest_video_to_keyframes(
        video_path=vid_path,
        output_dir=out_dir,
        target_min=4,
        target_max=15,
        sample_hz=10.0,
    )

    assert isinstance(res, VideoIngestResult)
    assert res.total_frames == 60
    assert res.fps == pytest.approx(20.0, abs=1.0)
    assert res.duration_s == pytest.approx(3.0, abs=0.2)
    assert 4 <= res.keyframe_count <= 15
    assert len(res.keyframes) == res.keyframe_count
    assert "information_gain" in res.strategy_used

    # Check that output files actually exist and are decodable
    for kf in res.keyframes:
        kf_path = Path(kf["file_path"])
        assert kf_path.is_file()
        assert kf_path.stat().st_size > 0
        img = cv2.imread(str(kf_path))
        assert img is not None
        assert img.shape == (h, w, 3)

    summary = res.summary()
    assert f"Extracted {res.keyframe_count} keyframes from 60 frames" in summary
