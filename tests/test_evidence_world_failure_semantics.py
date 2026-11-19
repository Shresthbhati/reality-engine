"""Failure semantics of the evidence-world stage: SUCCEEDED, PARTIAL, FAILED, CANCELLED and SKIPPED stay distinct, a persistence
failure is never an empty success, and re-running a job is idempotent. Engines are SYNTHETIC doubles; the stage glue, flow,
persistence and tenancy checks are the real code."""

from __future__ import annotations

import json

import pytest

from apps.api import evidence_world_stage as ew
from tests.test_multiengine_evidence_flow import _setup


@pytest.fixture()
def files(tmp_path):
    paths = []
    for name in ("img1.jpg", "img2.jpg"):
        p = tmp_path / "src" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xd9")  # presence matters to the stage; the synthetic engines do not decode pixels
        paths.append(str(p))
    return paths


def _run(tmp_path, files, *, fail=(), cancel=None, job="job-1", inputs=None, base=None):
    def default():
        reg, adapters = _setup(fail)
        return reg, adapters, None

    return ew.run_for_job(team_id="t1", world_id="w1", job_id=job, sources=files, should_cancel=cancel,
                          base=base or tmp_path / "ws", flow_inputs=inputs or default)


def test_all_engines_succeeding_is_complete(tmp_path, files):
    r = _run(tmp_path, files)
    assert r["status"] == "complete" and r["counts"]["observations"] > 0


def test_one_failing_engine_is_partial_not_complete_and_not_failed(tmp_path, files):
    r = _run(tmp_path, files, fail=("det_b",))
    assert r["status"] == "partial" and r["counts"]["observations"] > 0


def test_every_engine_failing_never_reports_complete(tmp_path, files):
    r = _run(tmp_path, files, fail=("det_a", "det_b"))
    assert r["status"] in ("partial", "failed") and r["status"] != "complete" and r["counts"]["observations"] == 0


def test_a_crash_in_the_flow_setup_is_failed_with_the_error(tmp_path, files):
    def boom():
        raise RuntimeError("model zoo exploded")

    r = _run(tmp_path, files, inputs=boom)
    assert r["status"] == "failed" and "model zoo exploded" in r["error"]


def test_operator_cancel_is_cancelled_and_distinct(tmp_path, files):
    r = _run(tmp_path, files, cancel=lambda: True)
    assert r["status"] == "cancelled"


def test_no_readable_evidence_is_skipped_not_an_empty_success(tmp_path):
    r = _run(tmp_path, ["virtual:room", str(tmp_path / "missing.jpg")])
    assert r["status"] == "skipped" and r["skipped_sources"] == 2 and "counts" not in r


def test_a_persistence_failure_is_failed_not_an_empty_success(tmp_path, files):
    blocker = tmp_path / "ws"
    blocker.write_text("a file where the evidence directory must go")  # makes the tenant directory impossible to create
    r = _run(tmp_path, files, base=blocker)
    assert r["status"] == "failed" and r["error"]


def test_rerunning_the_same_job_is_idempotent_and_keeps_the_raw_evidence(tmp_path, files):
    a = _run(tmp_path, files)
    snap = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1" / "snapshot.json"
    first = snap.read_text()
    b = _run(tmp_path, files)
    assert a["status"] == b["status"] == "complete" and b["duplicate_execution"] is True and "duplicate_execution" not in a
    assert b["counts"] == a["counts"]  # reported from the recorded result, not recomputed
    assert snap.read_text() == first  # the duplicate execution neither re-ran the engines nor touched the evidence
    assert json.loads(first)["counts"]["entities"] == a["counts"]["entities"]  # and did not double the world


@pytest.mark.parametrize("team,world,job", [("../x", "w", "j"), ("t", "w/../..", "j"), ("t", "w", ""), ("t", "w", "a b")])
def test_unsafe_ids_are_refused_before_any_path_is_built(tmp_path, files, team, world, job):
    with pytest.raises(ew.EvidenceWorldError):
        ew.run_for_job(team_id=team, world_id=world, job_id=job, sources=files, base=tmp_path / "ws")
