"""Failure injection against the REAL COLMAP session: the last valid model must survive everything.

A world's committed COLMAP state (``current/``) may only ever move forward when a version was adopted.
Here real COLMAP runs on South Building photographs and the process is "killed" (a BaseException that
nothing in the backend may swallow) before/after every step, or the committed state is corrupted.
After each failure:

    * ``current/`` is byte-identical to what was committed (nothing valid was deleted or rewritten)
    * the next run starts from that state and still produces a valid model
    * no evidence is lost, nothing pretends to have succeeded

A committed SIX-photo state is built once and copied per test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tests.integration.test_incremental_registration import A, B, C, SIX, items  # noqa: F401 -- harness + marks
from tests.integration.test_progressive_product_journey import pytestmark  # noqa: F401

from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.colmap_session import ColmapSession


class SimulatedCrash(BaseException):
    """The process died. BaseException: no `except Exception` in the code under test may absorb it."""


def digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@pytest.fixture(scope="module")
def committed_six(tmp_path_factory):
    root = tmp_path_factory.mktemp("committed") / "colmap"
    sess = ColmapSession(root)
    result = ColmapReconstructionBackend(session=sess).reconstruct(items(SIX))
    assert sess.commit() and len(result.camera_poses) == 6
    return root, len(result.camera_poses)


@pytest.fixture
def world(committed_six, tmp_path):
    root = tmp_path / "colmap"
    shutil.copytree(committed_six[0], root)
    return ColmapSession(root), digest(root / "current")


def crash_at(monkeypatch, step, when):
    real = ColmapReconstructionBackend._run_step

    def wrapper(self, s, args, env):
        if s == step and when == "before":
            raise SimulatedCrash(f"killed before {s}")
        real(self, s, args, env)
        if s == step and when == "after":
            raise SimulatedCrash(f"killed after {s}")

    monkeypatch.setattr(ColmapReconstructionBackend, "_run_step", wrapper)


@pytest.mark.parametrize("step,when", [
    ("feature_extractor", "before"),
    ("exhaustive_matcher", "after"),
    ("image_registrator", "before"),
    ("image_registrator", "after"),
    ("point_triangulator", "after"),
    ("bundle_adjuster", "after"),
])
def test_a_crash_at_any_incremental_step_leaves_the_committed_model_intact_and_the_next_run_works(
        world, monkeypatch, step, when):
    sess, before = world
    with monkeypatch.context() as m:
        crash_at(m, step, when)
        with pytest.raises(SimulatedCrash):
            ColmapReconstructionBackend(session=sess).reconstruct(items(SIX + C))
    assert digest(sess.current) == before, "the committed COLMAP state changed during a crashed run"
    assert sess.committed_manifest() is not None
    assert sess.staging.exists(), "crash leaves stale staging for the next begin() to discard"

    be = ColmapReconstructionBackend(session=sess)
    result = be.reconstruct(items(SIX + C))                      # 'restart': a fresh backend on the same root
    info = be.last_run_info
    assert info["prior_images"] == 6 and info["new_images"] == [f"ev-{p.stem}" for p in C], info
    assert len(result.camera_poses) >= 7 and info["registered_total"] == len(result.camera_poses)
    assert digest(sess.current) == before                        # still not committed: only an adopted version commits


def test_a_crash_during_the_full_rebuild_keeps_the_committed_model(tmp_path, monkeypatch):
    sess = ColmapSession(tmp_path / "colmap")
    ColmapReconstructionBackend(session=sess).reconstruct(items(A))
    assert sess.commit()
    before = digest(sess.current)
    with monkeypatch.context() as m:
        crash_at(m, "mapper", "before")          # the 3-photo prior cannot place B, so the full rebuild is reached
        with pytest.raises(SimulatedCrash):
            ColmapReconstructionBackend(session=sess).reconstruct(items(A + B))
    assert digest(sess.current) == before
    be = ColmapReconstructionBackend(session=sess)
    assert len(be.reconstruct(items(A + B)).camera_poses) == 6 and be.last_run_info["mode"] == "full"


def test_crash_after_the_run_but_before_commit_converges_on_the_same_result_when_retried(world):
    """Models 'WorldStore committed the version but the process died before ColmapSession.commit()'."""
    sess, before = world
    first = ColmapReconstructionBackend(session=sess)
    r1 = first.reconstruct(items(SIX + C))
    assert sess.staging.joinpath("manifest.json").is_file() and digest(sess.current) == before
    # process dies here: nothing committed. A new process retries the SAME evidence:
    again = ColmapReconstructionBackend(session=sess)
    r2 = again.reconstruct(items(SIX + C))
    assert again.last_run_info["prior_images"] == 6 and again.last_run_info["mode"] == first.last_run_info["mode"]
    assert {p.evidence_id for p in r2.camera_poses} == {p.evidence_id for p in r1.camera_poses}
    assert sess.commit()                                         # and now it can advance, exactly once
    assert digest(sess.current) != before and sess.committed_manifest()["registered"]


def test_crash_after_commit_before_job_completion_leaves_a_consistent_reusable_state(world):
    sess, before = world
    be = ColmapReconstructionBackend(session=sess)
    r = be.reconstruct(items(SIX + C))
    assert sess.commit()
    after = digest(sess.current)
    # process dies before the job records completion; on restart the same evidence changes nothing that was
    # established. Photographs that stayed unplaced are legitimately RETRIED (mode incremental/full); with none
    # waiting the committed model is reused as-is.
    be2 = ColmapReconstructionBackend(session=sess)
    r2 = be2.reconstruct(items(SIX + C))
    info = be2.last_run_info
    waiting = len(items(SIX + C)) - len(r.camera_poses)
    assert info["mode"] in (("reused",) if waiting == 0 else ("incremental", "full", "reused")), info
    assert info["new_images"] == [] and info["prior_images"] == 9
    assert {p.evidence_id for p in r.camera_poses} <= {p.evidence_id for p in r2.camera_poses}, "registration regressed"
    assert digest(sess.current) == after                       # and nothing is committed until a version is adopted


# ------------------------------------------------------------------- corruption


def test_missing_model_directory_is_refused_not_trusted_and_the_valid_files_are_not_deleted(world):
    sess, _ = world
    shutil.rmtree(sess.current / "sparse")
    kept = digest(sess.current)
    be = ColmapReconstructionBackend(session=sess)
    result = be.reconstruct(items(SIX + C))
    assert be.last_run_info["mode"] == "full" and "unusable" in be.last_run_info["reason"]
    assert len(result.camera_poses) >= 6
    assert digest(sess.current) == kept, "a refused prior must not be rewritten before a version is adopted"


def test_missing_database_is_refused(world):
    sess, _ = world
    (sess.current / "database.db").unlink()
    be = ColmapReconstructionBackend(session=sess)
    be.reconstruct(items(SIX + C))
    assert be.last_run_info["mode"] == "full" and "unusable" in be.last_run_info["reason"]


def test_invalid_manifest_is_refused(world):
    sess, _ = world
    (sess.current / "manifest.json").write_text("{ not json", encoding="utf-8")
    be = ColmapReconstructionBackend(session=sess)
    be.reconstruct(items(SIX + C))
    assert be.last_run_info["mode"] == "full"


def test_truncated_model_file_makes_incremental_fail_visibly_and_the_full_rebuild_takes_over(world):
    sess, _ = world
    victim = next(p for p in sorted((sess.current / "sparse" / "0").iterdir()) if p.stem == "points3D")
    victim.write_bytes(victim.read_bytes()[:12])
    be = ColmapReconstructionBackend(session=sess)
    result = be.reconstruct(items(SIX + C))
    info = be.last_run_info
    assert info["mode"] == "full" and len(result.camera_poses) >= 6, info
    assert "incremental_error" in info, "the failed incremental attempt must be on the record"


def test_stale_staging_from_a_dead_run_is_discarded_not_read(world):
    sess, before = world
    sess.staging.mkdir(parents=True)
    (sess.staging / "manifest.json").write_text('{"version": 1, "registered": ["ghost"], "images": {"ghost.jpg": "x"}}')
    (sess.staging / "junk.bin").write_bytes(b"\x00" * 64)
    be = ColmapReconstructionBackend(session=sess)
    be.reconstruct(items(SIX + C))
    assert "ghost" not in str(be.last_run_info) and be.last_run_info["prior_images"] == 6
    assert digest(sess.current) == before


def test_evidence_removed_since_the_commit_is_refused_as_a_prior(world):
    sess, _ = world
    be = ColmapReconstructionBackend(session=sess)
    be.reconstruct(items(SIX[:-1]))          # a photo the committed model contains is no longer evidence
    assert be.last_run_info["mode"] == "full"


# -------------------------------------------------------- world-level candidate arbitration


def _snap_of(result):
    poses = {p.evidence_id: [float(v) for v in p.position] for p in result.camera_poses}
    return {"registered_ids": list(poses), "input_ids": list(poses), "level": 2, "model_state": "PARTIAL",
            "points": len(result.points), "cameras": poses, "entities": [], "entity_types": {}, "provenance": {}}


class _Arbiter:
    """Records what the backend asks and returns a scripted choice."""

    def __init__(self, choice, prev):
        self.choice, self.prev, self.assessed = choice, prev, []

    def assess(self, result, tag):
        from engine.pipeline import candidate_selection as cs
        self.assessed.append(tag)
        return cs.assess(self.prev, _snap_of(result), None, tag)

    def choose(self, inc, full):
        return {"choice": self.choice, "deciding": "scripted", "why": f"scripted {self.choice}"}


def _prev_of(sess):
    reg = json.loads((sess.current / "manifest.json").read_text())["registered"]
    ids = ["ev-" + Path(n).stem for n in reg]
    return {"registered_ids": ids, "input_ids": ids, "level": 2, "model_state": "PARTIAL", "points": 1,
            "cameras": {}, "entities": [], "entity_types": {}, "provenance": {}}


def test_the_backend_defers_to_the_world_arbiter_not_to_camera_count(world):
    sess, _ = world
    be = ColmapReconstructionBackend(session=sess)
    be.candidate_arbiter = _Arbiter("full", _prev_of(sess))
    result = be.reconstruct(items(SIX + C))
    info = be.last_run_info
    if "arbitration" in info:
        assert info["mode"] == "full" and info["reason"] == "scripted full", info
        assert be.candidate_arbiter.assessed == ["incremental", "full"]
    else:
        # the incremental candidate was clean AND placed every photograph: no full rebuild was needed
        assert info["mode"] == "incremental" and be.candidate_arbiter.assessed == ["incremental"], info
    assert len(result.camera_poses) == info["registered_total"]


def test_a_failing_arbiter_falls_back_to_the_registration_count_rule_and_says_so(world):
    sess, _ = world

    class Broken:
        def assess(self, *a):
            raise RuntimeError("boom")

    be = ColmapReconstructionBackend(session=sess)
    be.candidate_arbiter = Broken()
    result = be.reconstruct(items(SIX + C))
    info = be.last_run_info
    assert "arbiter_error" in info and "boom" in info["arbiter_error"]
    assert info["mode"] in ("incremental", "full") and len(result.camera_poses) >= 7
