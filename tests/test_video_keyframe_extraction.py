"""Unit tests for video keyframe extraction and scene transition detection (Master Development Plan Tasks 2.2 & 2.3)."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from evidence.frames import FrameCandidate
from evidence.video import (
    SceneAwareFrameSelectionStrategy,
    SceneTransition,
    compute_frame_difference,
    detect_scene_transitions_from_frames,
    extract_video_keyframes,
)


def test_compute_frame_difference_identical_and_dissimilar():
    img_black = np.zeros((64, 64, 3), dtype=np.uint8)
    img_white = np.ones((64, 64, 3), dtype=np.uint8) * 255

    # Identical frames -> 0.0 distance
    assert compute_frame_difference(img_black, img_black) == 0.0

    # Max dissimilar frames -> high distance >= 0.8
    assert compute_frame_difference(img_black, img_white) >= 0.8


def test_detect_scene_transitions():
    # 5 frames of Scene 1 (black), then 5 frames of Scene 2 (white)
    frames = []
    for i in range(5):
        frames.append((i, i * 0.1, np.zeros((32, 32, 3), dtype=np.uint8)))
    for i in range(5, 10):
        frames.append((i, i * 0.1, np.ones((32, 32, 3), dtype=np.uint8) * 255))

    transitions = detect_scene_transitions_from_frames(frames, cut_threshold=0.30)
    assert len(transitions) == 9

    # The cut occurred at frame 5
    cuts = [t for t in transitions if t.is_cut]
    assert len(cuts) == 1
    assert cuts[0].frame_index == 5
    assert cuts[0].delta_score >= 0.8


def test_scene_aware_frame_selection_strategy():
    # 50 total candidates with a cut at frame 25
    candidates = [
        FrameCandidate(frame_index=i, timestamp_s=i * 0.04, width=640, height=480)
        for i in range(50)
    ]
    cut_transition = SceneTransition(frame_index=25, timestamp_s=1.0, delta_score=0.85, is_cut=True)

    strategy = SceneAwareFrameSelectionStrategy(
        target_count=6,
        scene_transitions=[cut_transition],
    )

    selected = strategy.select(candidates)
    assert len(selected) == 6

    # Both scenes must have representation
    scene1_selected = [c for c in selected if c.frame_index < 25]
    scene2_selected = [c for c in selected if c.frame_index >= 25]

    assert len(scene1_selected) >= 2
    assert len(scene2_selected) >= 2

    # First and last frames must be included
    assert selected[0].frame_index == 0
    assert selected[-1].frame_index == 49


def test_extract_video_keyframes_end_to_end(tmp_path: Path):
    video_file = tmp_path / "test_synth_video.mp4"
    output_dir = tmp_path / "extracted_keyframes"

    # Generate a 2-second synthetic video at 10 fps (20 frames)
    # Frames 0-9: Red gradient (Scene 1)
    # Frames 10-19: Blue gradient (Scene 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (128, 128))

    for i in range(10):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # Red
        out.write(frame)

    for i in range(10, 20):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # Blue
        out.write(frame)

    out.release()

    # Extract 6 keyframes
    extracted = extract_video_keyframes(
        video_path=video_file,
        output_dir=output_dir,
        target_count=6,
    )

    assert len(extracted) == 6
    for item in extracted:
        assert Path(item["file_path"]).exists()
        assert Path(item["file_path"]).stat().st_size > 0
        assert "sharpness" in item
        assert item["width"] == 128
        assert item["height"] == 128
