"""E209: the measured-room runner refuses to run on test backends or bad inputs and never scores failed jobs.

The API itself is mocked here (``httpx.MockTransport``): this checks the runner's plumbing and exit
codes. Whether the real API produces a scoreable world is what E210 measures.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from scripts.run_measured_room import main as run_main
from tests.test_ground_truth_schema import CAPTURE
from tests.test_measured_room_accuracy_report import _perfect_world_and_gt

PHOTO_BYTES = {"a.jpg": b"jpeg-a", "b.jpg": b"jpeg-b"}


@pytest.fixture(autouse=True)
def _real_backend(monkeypatch):
    monkeypatch.delenv("REALITY_TEST_BACKEND", raising=False)


def _measured_gt() -> dict:
    _, gt = _perfect_world_and_gt()
    gt = copy.deepcopy(gt)
    for o, wall in zip(gt["openings"], ("w-s", "w-n")):
        o["wall"] = wall
    gt["source"] = "measured"
    gt["capture"] = copy.deepcopy(CAPTURE)
    return gt


def _room_dir(tmp_path: Path, *, photos=True, ground_truth=True) -> Path:
    room = tmp_path / "room-a"
    (room / "photos").mkdir(parents=True)
    if photos:
        for name, data in PHOTO_BYTES.items():
            (room / "photos" / name).write_bytes(data)
    if ground_truth:
        (room / "ground_truth.json").write_text(json.dumps(_measured_gt()), encoding="utf-8")
    return room


def _args(room: Path, out: Path) -> list[str]:
    return ["--room-dir", str(room), "--api", "http://api.test", "--token", "tok", "--out", str(out), "--poll-s", "0"]


def _api(tmp_path: Path, *, job_status="succeeded", job_error=None, stages=True):
    world, _ = _perfect_world_and_gt()
    world_json = json.loads(json.dumps(world.to_dict()))  # what GET /api/worlds/{id}/worldir serves
    seen: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        path = request.url.path
        if path in ("/api/sessions", "/api/worlds"):
            return httpx.Response(200, json={"id": "s1" if path == "/api/sessions" else "w1"})
        if path == "/api/worlds/w1/attach/s1":
            return httpx.Response(200, json={})
        if path == "/api/uploads":
            return httpx.Response(201, json={})
        if path == "/api/sessions/s1/reconstruct":
            return httpx.Response(200, json={"job_id": "j1"})
        if path == "/api/jobs/j1":
            return httpx.Response(200, json={"id": "j1", "status": job_status, "error": job_error})
        if path == "/api/jobs/j1/stages":
            if not stages:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"items": [{"name": "s01_evidence", "status": "succeeded"}]})
        if path == "/api/worlds/w1/worldir":
            return httpx.Response(200, json=world_json)
        return httpx.Response(404, json={"detail": path})

    return httpx.MockTransport(handler), seen


# --- the two refusals the card specifies ---------------------------------------------------


def test_test_backend_env_var_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    assert run_main(_args(_room_dir(tmp_path), tmp_path / "out")) == 3
    assert "REALITY_TEST_BACKEND is set; this script measures the real engine only" in capsys.readouterr().err


def test_missing_ground_truth_exits_2(tmp_path):
    assert run_main(_args(_room_dir(tmp_path, ground_truth=False), tmp_path / "out")) == 2


# --- further input refusals ----------------------------------------------------------------


def test_invalid_ground_truth_and_no_photos_exit_2(tmp_path, capsys):
    room = _room_dir(tmp_path)
    gt = json.loads((room / "ground_truth.json").read_text(encoding="utf-8"))
    del gt["capture"]["consent_reference"]
    (room / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    assert run_main(_args(room, tmp_path / "out")) == 2
    assert "consent_reference" in capsys.readouterr().err

    empty = tmp_path / "other" / "room-b"
    (empty / "photos").mkdir(parents=True)
    (empty / "ground_truth.json").write_text(json.dumps(_measured_gt()), encoding="utf-8")
    assert run_main(_args(empty, tmp_path / "out")) == 2


# --- plumbing against a mocked API ---------------------------------------------------------


def test_successful_run_writes_the_scored_record(tmp_path):
    transport, seen = _api(tmp_path)
    out = tmp_path / "out"
    assert run_main(_args(_room_dir(tmp_path), out), transport=transport) == 0

    record_path = next(p for p in (out / "room-a").glob("*.json") if p.name.count(".") == 1)
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    assert rec["job_id"] == "j1" and rec["job_status"] == "succeeded" and rec["evidence_class"] == "measured"
    assert rec["stages"] == [{"name": "s01_evidence", "status": "succeeded"}]
    assert rec["git_sha"] and rec["wall_clock_s"] >= 0
    assert rec["photos_sha256"] == {n: hashlib.sha256(d).hexdigest() for n, d in PHOTO_BYTES.items()}
    assert rec["evaluation"]["meets_tolerances"] is True and rec["evaluation"]["rooms"]
    assert {a for _, _, a in seen} == {"Bearer tok"}
    assert sum(1 for m, p, _ in seen if p == "/api/uploads") == 2


def test_missing_stages_route_is_recorded_as_null_not_fabricated(tmp_path):
    transport, _ = _api(tmp_path, stages=False)
    out = tmp_path / "out"
    assert run_main(_args(_room_dir(tmp_path), out), transport=transport) == 0
    record_path = next(p for p in (out / "room-a").glob("*.json") if p.name.count(".") == 1)
    assert json.loads(record_path.read_text(encoding="utf-8"))["stages"] is None


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_or_cancelled_job_exits_4_and_is_never_scored(tmp_path, status, capsys):
    transport, seen = _api(tmp_path, job_status=status, job_error="colmap crashed")
    out = tmp_path / "out"
    assert run_main(_args(_room_dir(tmp_path), out), transport=transport) == 4
    files = sorted(p.name for p in (out / "room-a").iterdir())
    assert len(files) == 1 and files[0].endswith(".failed.json")
    failure = json.loads((out / "room-a" / files[0]).read_text(encoding="utf-8"))
    assert failure["job_status"] == status and failure["error"] == "colmap crashed"
    assert not any(p == "/api/worlds/w1/worldir" for _, p, _ in seen)  # the world was never even fetched
    assert "colmap crashed" in capsys.readouterr().err


def test_unreachable_api_exits_5(tmp_path, capsys):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert run_main(_args(_room_dir(tmp_path), tmp_path / "out"), transport=httpx.MockTransport(boom)) == 5
    assert "API error" in capsys.readouterr().err
