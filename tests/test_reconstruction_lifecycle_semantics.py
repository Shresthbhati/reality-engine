"""Reconstruction lifecycle semantics the UI depends on.

The Studio renders reconstruction state from `GET /api/jobs/{id}`, so the
contract it can trust is narrow and must be exact:

    queued -> running -> succeeded | partial | failed | cancelled

The invariants pinned here are the ones a UI bug would silently violate:

  * an enqueue acknowledgement (HTTP 200) is NOT a reconstruction success --
    the world must still have no version at that moment;
  * a job that times out can never grade `succeeded` or `partial`;
  * `partial` always carries the degradation reasons and a real version id;
  * a cancelled run adopts nothing;
  * a duplicate enqueue produces two independent, individually honest jobs;
  * a version id reported by a job actually exists in the world's lineage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from reconstruction.backend.interface import ReconstructionResult  # noqa: E402
from tests.test_cli_compile import _TwoViewBackend  # noqa: E402


@dataclass(frozen=True)
class _HangingBackend:
    """Backend that never returns in time: the job must time out, never
    succeed. The stall is kept short (just past the 1s test timeout) so the
    orphaned worker thread drains promptly between tests."""

    def reconstruct(self, evidence):
        time.sleep(5)
        return ReconstructionResult(points=[], camera_poses=[], registration_status="success")


@dataclass(frozen=True)
class _DegradedBackend(_TwoViewBackend):
    """Registers, but not cleanly: must grade `partial` with reasons."""

    def reconstruct(self, evidence):
        result = super().reconstruct(evidence)
        return ReconstructionResult(
            points=result.points,
            camera_poses=result.camera_poses,
            registration_status="partial",
        )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _setup(client, tmp_path, monkeypatch,
           spec="tests.test_cli_compile:_TwoViewBackend", n_photos=3):
    monkeypatch.setenv("REALITY_TEST_BACKEND", spec)
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "Lifecycle"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Lifecycle world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(n_photos):
        r = client.post(
            f"/api/uploads?session_id={sid}",
            files={"file": (f"f{i}.jpg", b"jpeg-" + f"{i}".encode() * 8, "image/jpeg")},
        )
        assert r.status_code == 201, r.text
    return sid, wid


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not settle in {timeout}s")


def test_enqueue_200_is_not_a_reconstruction_success(client, tmp_path, monkeypatch):
    sid, wid = _setup(client, tmp_path, monkeypatch)
    response = client.post(f"/api/sessions/{sid}/reconstruct")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"], "the enqueue acknowledgement must carry a trackable job id"

    # The acknowledgement must not smuggle a result.
    for forbidden in ("version_id", "success", "result", "entities", "worldir"):
        assert forbidden not in body, (
            f"enqueue response carries '{forbidden}': a 200 is an acknowledgement, not a result"
        )

    # And the world must genuinely have nothing yet.
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []
    _drain(client, body["job_id"])


def test_timeout_can_never_grade_success(client, tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")
    sid, wid = _setup(
        client,
        tmp_path,
        monkeypatch,
        spec="tests.test_reconstruction_lifecycle_semantics:_HangingBackend",
    )
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] == "failed", job
    assert job["status"] not in ("succeeded", "partial")
    assert "timed out" in (job.get("error") or "").lower()
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


def test_degraded_run_is_partial_and_carries_reasons(client, tmp_path, monkeypatch):
    sid, wid = _setup(
        client,
        tmp_path,
        monkeypatch,
        spec="tests.test_reconstruction_lifecycle_semantics:_DegradedBackend",
    )
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] == "partial", job
    payload = job.get("payload") or {}
    assert payload.get("version_id"), "a partial run still adopted a version; report it"
    assert payload.get("degraded"), "a partial run must say why it was degraded"
    assert any("registration" in reason for reason in payload["degraded"])
    assert payload["outcome"] == "partial"


def test_cancelled_run_adopts_nothing(client, tmp_path, monkeypatch):
    sid, wid = _setup(client, tmp_path, monkeypatch)
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code in (200, 202), cancel.text
    job = _wait(client, job_id)
    assert job["status"] == "cancelled", job
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


def test_duplicate_enqueue_is_rejected_not_fabricated(client, tmp_path, monkeypatch):
    """A second enqueue while one is in flight is a 409, not a second job
    and not a second version. The UI must be able to surface the conflict."""
    sid, wid = _setup(client, tmp_path, monkeypatch)
    first = client.post(f"/api/sessions/{sid}/reconstruct")
    assert first.status_code == 200
    first_id = first.json()["job_id"]

    duplicate = client.post(f"/api/sessions/{sid}/reconstruct")
    assert duplicate.status_code == 409, duplicate.text
    assert "already in progress" in (duplicate.json().get("detail") or "").lower()

    # Exactly one job exists for this session.
    jobs = client.get("/api/jobs").json()["items"]
    reconstruct_jobs = [
        j for j in jobs
        if j["entity_id"] == sid and j["type"] == "RECONSTRUCT_SESSION"
    ]
    assert [j["id"] for j in reconstruct_jobs] == [first_id]
    _drain(client, first_id)


def test_reported_version_exists_in_world_lineage(client, tmp_path, monkeypatch):
    """The success the UI shows must be a version the store actually holds."""
    sid, wid = _setup(client, tmp_path, monkeypatch)
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] == "succeeded", job.get("error")

    version_id = (job.get("payload") or {}).get("version_id")
    assert version_id, "a succeeded job must name the version it committed"

    lineage = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert version_id in {v["id"] for v in lineage}
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] == version_id
    # The version is real data, not an empty shell.
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200


# --------------------------------------------------------------------------
# Correction -> WorldIR -> WorldStore -> new version -> reload -> diff
# --------------------------------------------------------------------------

# Fields the commit contract refuses to touch: a correction may rename,
# reclassify, rescore or relabel, never rewrite identity, geometry,
# provenance or observations.
IMMUTABLE_FIELDS = (
    "id",
    "provenance",
    "type_id",
    "observations",
    "relationships",
    "geometry_ids",
    "custom_properties",
    "semantic_class",
)


def test_invalid_corrections_are_rejected_and_change_nothing(client, tmp_path, monkeypatch):
    sid, wid = _setup(client, tmp_path, monkeypatch)
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial"), job
    baseline = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    head = client.get(f"/api/worlds/{wid}").json()["current_version_id"]
    entity_id = _any_entity(client, wid)

    for field in IMMUTABLE_FIELDS:
        response = client.post(
            f"/api/worlds/{wid}/commit",
            json={
                "entity_id": entity_id,
                "changes": {field: "tampered"},
                "parent_version_id": head,
            },
        )
        assert response.status_code in (409, 413, 422), (
            f"commit accepted a write to immutable field '{field}': {response.text}"
        )

    # An empty changes object is not a correction either.
    empty = client.post(
        f"/api/worlds/{wid}/commit",
        json={"entity_id": entity_id, "changes": {}, "parent_version_id": head},
    )
    assert empty.status_code == 422, empty.text

    # Nothing moved: HEAD is identical and no version was minted.
    after = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert [v["id"] for v in after] == [v["id"] for v in baseline]
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] == head


def test_successful_correction_persists_and_is_diffable(client, tmp_path, monkeypatch):
    """v1 -> correction -> v2 -> diff, with v2 loading in a fresh read."""
    sid, wid = _setup(client, tmp_path, monkeypatch)
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial"), job

    v1 = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert v1, "a reconstruction must have produced a first version"
    head = client.get(f"/api/worlds/{wid}").json()["current_version_id"]
    entity_id = _any_entity(client, wid)

    before_entity = _entity(client, wid, entity_id)
    commit = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": entity_id,
            "changes": {"name": "CORRECTED_BY_OPERATOR", "semantic_labels": ["verified"]},
            "parent_version_id": head,
            "commit_message": "operator correction",
        },
    )
    assert commit.status_code == 200, commit.text
    body = commit.json()
    v2 = body.get("version_id")
    assert v2, "a successful commit must name the version it created"
    assert v2 != head, "a correction must create a new version, not rewrite the old one"

    # v2 is real, and it carries the change.
    worldir = client.get(f"/api/worlds/{wid}/worldir").json()
    assert worldir["entities"][entity_id]["name"] == "CORRECTED_BY_OPERATOR"
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2

    # The previous version is untouched and still readable.
    versions = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert head in {v["id"] for v in versions}
    diff = client.get(f"/api/worlds/{wid}/diff?base={head}&head={v2}")
    assert diff.status_code == 200, diff.text
    summary = diff.json()["summary"]
    assert summary.get("entities_modified", 0) >= 1, diff.json()

    # The pre-correction entity really did differ.
    assert before_entity.get("name") != "CORRECTED_BY_OPERATOR"


def test_stale_parent_commit_is_refused(client, tmp_path, monkeypatch):
    """Two operators on the same HEAD: the second must not silently win."""
    sid, wid = _setup(client, tmp_path, monkeypatch)
    _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    head = client.get(f"/api/worlds/{wid}").json()["current_version_id"]
    entity_id = _any_entity(client, wid)

    first = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": entity_id,
            "changes": {"name": "FIRST_CORRECTION"},
            "parent_version_id": head,
        },
    )
    assert first.status_code == 200, first.text

    # Same (now stale) parent again.
    second = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": entity_id,
            "changes": {"name": "SECOND_CORRECTION"},
            "parent_version_id": head,
        },
    )
    assert second.status_code == 409, second.text


def _entity(client, world_id: str, entity_id: str) -> dict:
    return client.get(f"/api/worlds/{world_id}/worldir").json()["entities"][entity_id]


def _drain(client: TestClient, job_id: str) -> None:
    """Wait out a job this test enqueued but does not assert on.

    The API's worker is a background task on the app; a test that leaves a
    reconstruction in flight hands a still-running worker to the next test's
    database, which wedges it. Draining (outcome ignored) keeps the module
    runnable as a whole.
    """
    try:
        _wait(client, job_id)
    except AssertionError:
        pass


def _any_entity(client, world_id: str) -> str:
    entities = client.get(f"/api/worlds/{world_id}/worldir").json()["entities"]
    assert entities, "a compiled world must contain entities to correct"
    return next(iter(entities))

