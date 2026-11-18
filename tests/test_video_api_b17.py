import cv2
import numpy as np
import pytest
from pathlib import Path
from starlette.testclient import TestClient

from apps.api.main import app
import apps.api.jobs as jobrunner


def _make_synth_video(path: Path, distinct_views: int = 2, repeats: int = 1, frame_w: int = 160, frame_h: int = 120):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (frame_w, frame_h))
    assert writer.isOpened()
    for view_idx in range(distinct_views):
        img = np.full((frame_h, frame_w, 3), (view_idx * 70 + 30) % 255, dtype=np.uint8)
        cv2.rectangle(img, (10 + view_idx * 20, 10), (50 + view_idx * 20, 60), (255, 255, 255), -1)
        for _ in range(repeats):
            writer.write(img)
    writer.release()


def test_video_upload_response_counters_and_two_frame_clip(tmp_path, monkeypatch):
    # Mock enqueue_job so background reconstruction does not run slow SfM
    monkeypatch.setattr(jobrunner, "enqueue_job", lambda db, job_type, entity_type, entity_id, payload=None, team_id=None: type("J", (), {"id": "mock-job-1"})())

    # 1. Test 2-frame clip keeps 2
    two_path = tmp_path / "two_frames.mp4"
    _make_synth_video(two_path, distinct_views=2, repeats=1)

    with TestClient(app) as client:
        r = client.post("/api/reconstructions", files=[("files", ("two_frames.mp4", two_path.read_bytes(), "video/mp4"))])
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["frames_seen"] == 2
        assert body["frames_kept"] == 2
        assert body["duplicates_dropped"] == 0
        kinds = [e["type"] for e in body["evidence"]]
        assert kinds.count("video") == 1
        assert kinds.count("photo") == 2

    # 2. Test 18 frames (6 views repeated 3x)
    eighteen_path = tmp_path / "eighteen_frames.mp4"
    _make_synth_video(eighteen_path, distinct_views=6, repeats=3)

    with TestClient(app) as client:
        r = client.post("/api/reconstructions", files=[("files", ("eighteen_frames.mp4", eighteen_path.read_bytes(), "video/mp4"))])
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["frames_seen"] == 18
        assert body["frames_kept"] == 6
        assert body["duplicates_dropped"] == 12
        kinds = [e["type"] for e in body["evidence"]]
        assert kinds.count("video") == 1
        assert kinds.count("photo") == 6
