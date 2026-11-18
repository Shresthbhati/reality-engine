"""Tests for engine/pipeline/stages/artifacts.py and DurableJobCheckpointStore."""

from __future__ import annotations

import threading

import numpy as np

from apps.api.jobs import DurableJobCheckpointStore
from engine.pipeline.stages.artifacts import (
    get_array,
    get_bytes,
    get_json,
    get_npz,
    put_array,
    put_bytes,
    put_json,
    put_npz,
)
from engine.pipeline.stages.base import StageResult


def test_artifacts_bytes_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    raw = b"reconstruction-point-cloud-bytes-data-12345"
    sha = put_bytes(raw)
    assert len(sha) == 64
    assert get_bytes(sha) == raw


def test_artifacts_json_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    payload = {"entity_count": 42, "roles": ["floor", "support_surface", "wall"], "score": 0.99}
    sha = put_json(payload)
    assert len(sha) == 64
    assert get_json(sha) == payload


def test_artifacts_array_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    sha = put_array(arr)
    loaded = get_array(sha, mmap=False)
    np.testing.assert_array_equal(arr, loaded)

    loaded_mmap = get_array(sha, mmap=True)
    np.testing.assert_array_equal(arr, loaded_mmap)


def test_artifacts_npz_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    sha = put_npz(pts=pts, colors=colors)
    npz = get_npz(sha)
    assert "pts" in npz
    assert "colors" in npz
    np.testing.assert_array_equal(pts, npz["pts"])
    np.testing.assert_array_equal(colors, npz["colors"])


def test_durable_job_checkpoint_store_resumes_succeeded_stage():
    class _MockStore:
        def __init__(self):
            self.records = []

        def lookup(self, job_id, name, inputs_hash):
            if name == "depth" and inputs_hash == "hash1":
                return StageResult(
                    name="depth",
                    status="succeeded",
                    inputs_hash=inputs_hash,
                    outputs={"depth_map": "sha1"},
                    metrics={},
                    warnings=[],
                    error=None,
                    started_at=None,
                    ended_at=None,
                    peak_rss_mb=None,
                )
            return None

        def record(self, job_id, result):
            self.records.append((job_id, result))

    mock = _MockStore()
    abandoned = threading.Event()
    durable = DurableJobCheckpointStore(mock, abandoned)

    # Coarse segments cannot resume without full artifact restoration
    assert durable.lookup("job_1", "reconstruct", "hash1") is None

    # Fine stages with valid outputs can resume
    res = durable.lookup("job_1", "depth", "hash1")
    assert res is not None
    assert res.name == "depth"
    assert res.status == "succeeded"

    # Abandoned event blocks lookup
    abandoned.set()
    assert durable.lookup("job_1", "depth", "hash1") is None
