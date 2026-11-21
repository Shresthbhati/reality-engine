"""Depth cache: reused only when engine + exact checkpoint hash + image bytes all match; a damaged entry is a miss; an all-cached engine is
never loaded. Synthetic depth (proves the caching rules, not any model)."""

from __future__ import annotations

import os

import numpy as np

from apps.api import evidence_world_stage as ew
from engine.multiengine.depth_cache import DepthCache, file_sha256
from tests.test_lift_stage_e2e import DX, _camera, _Depth, _inputs, _per_view, _points, photos  # noqa: F401

SHA = "ab" * 32


def test_the_key_needs_real_digests_so_unknown_weights_are_never_cached():
    img = "cd" * 32
    assert DepthCache.key("e", None, img) is None and DepthCache.key("e", "not-a-hash", img) is None and DepthCache.key("e", SHA, "short") is None
    assert DepthCache.key("e", SHA, img) == f"e|{SHA}|{img}"
    assert DepthCache.key("e", SHA, img) != DepthCache.key("other-engine", SHA, img) != DepthCache.key("e", "ef" * 32, img)


def test_round_trip_and_a_damaged_entry_is_a_miss_not_a_wrong_map(tmp_path):
    c = DepthCache(tmp_path / "c")
    a = np.random.default_rng(0).uniform(0.1, 3, (6, 8)).astype(np.float32)
    k = DepthCache.key("e", SHA, "cd" * 32)
    assert c.get(k) is None
    c.put(k, a)
    assert np.array_equal(c.get(k), a)
    blob = next((tmp_path / "c" / "artifacts").rglob("*"))
    while blob.is_dir():
        blob = next(blob.iterdir())
    blob.write_bytes(b"corrupted")  # tamper with the stored bytes
    assert c.get(k) is None
    nan = a.copy()
    nan[0, 0] = np.nan
    c.put(k, nan)
    assert c.get(k) is None  # a non-finite map is never served


class _CountingDepth(_Depth):
    def __init__(self, arr):
        super().__init__(arr)
        self.loaded = 0

    def load(self, device):
        self.loaded += 1


def _run(tmp_path, photo_paths, depth, job):
    cams = {p: _camera(DX[os.path.basename(p)]) for p in photo_paths}
    return ew.run_for_job(team_id="t1", world_id="w1", job_id=job, sources=photo_paths, base=tmp_path / "ws", flow_inputs=_inputs(depth),
                          depth_engines=["fake_depth"], cameras=cams, points3d=_points())


def test_a_second_job_on_the_same_images_and_weights_never_loads_the_model_and_gets_identical_depth(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.setattr("engine.multiengine.metric3d_producers.checkpoint_sha256", lambda path: SHA)
    d = _CountingDepth(_per_view())
    first = _run(tmp_path, photos, d, "job-1")
    assert d.loaded == 1 and first["depth"][0]["cache"] == {"hits": 0, "misses": 2, "disabled_reason": None}
    second = _run(tmp_path, photos, d, "job-2")
    assert d.loaded == 1  # NOT loaded again
    rep = second["depth"][0]
    assert rep["cache"]["hits"] == 2 and rep["cache"]["misses"] == 0 and rep["accepted"] == 2 and rep["status"] == "succeeded"
    assert second["lifting"]["lifted"] == first["lifting"]["lifted"] == 2  # the cached depth feeds the same geometry
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    def _depth(job):
        m = EvidenceMatrix(tmp_path / "ws" / "evidence-world" / "t1" / "w1" / job)
        return sorted(m.artifacts.get(m.context(o.observation_id).artifact_refs["raw_output"]) for o in m.store.query(quantity="depth_relative"))

    assert _depth("job-1") == _depth("job-2")  # byte-identical raw maps


def test_without_a_checkpoint_hash_nothing_is_cached_and_the_report_says_why(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.setattr("engine.multiengine.metric3d_producers.checkpoint_sha256", lambda path: None)
    d = _CountingDepth(_per_view())
    _run(tmp_path, photos, d, "job-1")
    second = _run(tmp_path, photos, d, "job-2")
    assert d.loaded == 2 and second["depth"][0]["cache"]["hits"] == 0 and "weights are known" in second["depth"][0]["cache"]["disabled_reason"]


def test_a_changed_image_is_a_miss_even_with_the_same_weights(tmp_path, photos, monkeypatch):  # noqa: F811
    monkeypatch.setattr("engine.multiengine.metric3d_producers.checkpoint_sha256", lambda path: SHA)
    d = _CountingDepth(_per_view())
    _run(tmp_path, photos, d, "job-1")
    from PIL import Image

    Image.new("RGB", (800, 600), (10, 200, 30)).save(photos[1])  # same path, different bytes
    assert file_sha256(photos[1]) != file_sha256(photos[0])
    second = _run(tmp_path, photos, d, "job-2")
    assert second["depth"][0]["cache"]["hits"] == 1 and second["depth"][0]["cache"]["misses"] == 1 and d.loaded == 2
