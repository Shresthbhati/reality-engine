"""The persistent COLMAP session: its lifecycle is what stops a rejected candidate from
becoming the base of the next incremental registration. Model-free (no COLMAP needed)."""

from __future__ import annotations

import json

from reconstruction.colmap_session import ColmapSession

SIG = {"robust_sift": True, "guided_matching": False, "trusted_intrinsics": None}


def _fill(staging, images, registered, *, incremental_ok=True, signature=SIG):
    """What the backend leaves in staging after a successful run."""
    (staging / "database.db").write_bytes(b"db:" + ",".join(images).encode())
    (staging / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (staging / "sparse" / "0" / "points3D.bin").write_bytes(b"pts")
    (staging / "images").mkdir(exist_ok=True)
    return {"signature": signature, "images": {n: "ev-" + n for n in images}, "registered": list(registered),
            "model_dir": "sparse/0", "incremental_ok": incremental_ok}


def _run(sess, images, registered, **kw):
    ws, prior = sess.begin(SIG, images)
    sess.write_manifest(_fill(ws, images, registered, **kw))
    return ws, prior


def test_first_run_has_no_prior_and_nothing_is_committed_until_commit(tmp_path):
    s = ColmapSession(tmp_path)
    ws, prior = _run(s, ["a", "b", "c"], ["a", "b", "c"])
    assert prior is None and s.committed_manifest() is None          # staging is not authority
    assert s.commit() is True
    m = s.committed_manifest()
    assert set(m["images"]) == {"a", "b", "c"} and not s.staging.exists()


def test_next_run_starts_from_a_copy_of_the_committed_state(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    ws, prior = s.begin(SIG, ["a", "b", "c"])
    assert prior is not None and prior["registered"] == ["a", "b"]
    assert (ws / "database.db").read_bytes() == b"db:a,b" and (ws / "sparse" / "0" / "points3D.bin").exists()


def test_discard_leaves_the_committed_state_untouched_so_a_rejected_run_cannot_poison_the_next(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    _run(s, ["a", "b", "c", "d"], ["a", "b", "c", "d"])               # a candidate that gets REJECTED
    s.discard()
    assert set(s.committed_manifest()["images"]) == {"a", "b"}
    _, prior = s.begin(SIG, ["a", "b", "c", "d"])
    assert set(prior["images"]) == {"a", "b"}                          # next run builds on V(adopted), not the reject


def test_a_crashed_run_leaves_stale_staging_that_the_next_begin_throws_away(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    ws, _ = s.begin(SIG, ["a", "b", "c"])
    (ws / "database.db").write_bytes(b"half-written")                  # crash: no manifest written
    ws2, prior = s.begin(SIG, ["a", "b", "c"])
    assert prior is not None and (ws2 / "database.db").read_bytes() == b"db:a,b"
    assert s.commit() is False                                         # staging without a manifest is never committed
    assert set(s.committed_manifest()["images"]) == {"a", "b"}


def test_prior_state_is_refused_when_it_cannot_be_extended_faithfully(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    assert s.begin({**SIG, "robust_sift": False}, ["a", "b", "c"])[1] is None       # pipeline settings changed
    assert s.begin(SIG, ["a", "c"])[1] is None                                       # evidence was removed
    import shutil
    shutil.rmtree(s.current / "sparse")                                              # model missing on disk
    assert s.begin(SIG, ["a", "b", "c"])[1] is None


def test_a_merged_multi_model_result_is_never_extended(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b", "c"], ["a", "b", "c"], incremental_ok=False); s.commit()
    assert s.begin(SIG, ["a", "b", "c", "d"])[1] is None


def test_corrupt_or_foreign_manifest_is_ignored_not_trusted(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    (s.current / "manifest.json").write_text("{not json")
    assert s.committed_manifest() is None and s.begin(SIG, ["a", "b", "c"])[1] is None
    (s.current / "manifest.json").write_text(json.dumps({"version": 999, "signature": SIG}))
    assert s.committed_manifest() is None


def test_staging_never_carries_the_committed_manifest_so_a_crashed_run_is_not_committable(tmp_path):
    s = ColmapSession(tmp_path)
    _run(s, ["a", "b"], ["a", "b"]); s.commit()
    ws, prior = s.begin(SIG, ["a", "b", "c"])
    assert prior is not None and not (ws / "manifest.json").exists()
    assert s.commit() is False and set(s.committed_manifest()["images"]) == {"a", "b"}
