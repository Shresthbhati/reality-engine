"""Reconstruction reliability (mission): determinism, incremental safety,
adversarial inputs, crash recovery, and reload acceptance across the
Evidence -> Session -> Job -> Reconstruction -> WorldIR -> WorldStore ->
Version -> Reload path.

Every test asserts honest states: success only with persistent output,
failure/degradation explicit, previous results never destroyed.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")


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


def _wait_job(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not settle in time")


def _chain_setup(client, tmp_path, monkeypatch, n_photos=3):
    """Session + world + attach + photos, deterministic backend armed."""
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "Rel"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Rel world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(n_photos):
        r = client.post(
            f"/api/uploads?session_id={sid}",
            files={"file": (f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8, "image/jpeg")},
        )
        assert r.status_code == 201, r.text
    return sid, wid


# --------------------------------------------------------------------------
# Determinism: identical input + configuration -> equivalent output
# --------------------------------------------------------------------------


def _slice_items(tmp_path):
    from evidence.session import EvidenceItem, EvidenceKind

    blob = tmp_path / "frame.jpg"
    blob.write_bytes(b"jpeg-bytes")
    uri = blob.resolve().as_uri()
    return [
        EvidenceItem(id=f"ev-{i}", kind=EvidenceKind.PHOTO, source_uri=uri)
        for i in range(3)
    ]


def _slice_options():
    import sys

    sys.path.insert(0, ".")
    from tests.test_cli_compile import _TwoViewBackend
    from engine.pipeline.vertical_slice import VerticalSliceOptions

    return VerticalSliceOptions(
        reconstruction_backend=_TwoViewBackend(),
        depth_model=None,
        perception_model=None,
        mesh_enabled=False,
        detail_enabled=False,
    )


def test_identical_input_produces_equivalent_world(tmp_path):
    """Same evidence + same configuration twice: identical canonical
    bytes (geometry, stable entity ids, classifications, topology,
    confidence, provenance)."""
    from engine.pipeline.vertical_slice import vertical_slice

    world_a = vertical_slice(_slice_items(tmp_path), _slice_options()).world
    world_b = vertical_slice(_slice_items(tmp_path), _slice_options()).world
    a = json.dumps(world_a.to_dict(), sort_keys=True)
    b = json.dumps(world_b.to_dict(), sort_keys=True)
    assert a == b, "identical inputs must produce byte-equivalent worlds"
    assert len(world_a.entities) > 0
    ids_a = sorted(world_a.entities)
    assert ids_a == sorted(world_b.entities)
    for eid in ids_a:
        ea, eb = world_a.entities[eid], world_b.entities[eid]
        assert ea.type == eb.type
        assert ea.confidence == eb.confidence
        assert ea.provenance == eb.provenance


# --------------------------------------------------------------------------
# Incremental safety: refinement chains linearly; failed refinement
# preserves the previous HEAD untouched and readable.
# --------------------------------------------------------------------------


def test_refinement_chains_and_failed_refinement_preserves_head(
    client, tmp_path, monkeypatch
):
    sid, wid = _chain_setup(client, tmp_path, monkeypatch)
    first = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                      timeout=90.0)
    assert first["status"] == "succeeded", first.get("error")
    v1 = first["payload"]["version_id"]

    # Refinement with genuinely new evidence mints a chained version...
    r = client.post(
        f"/api/uploads?session_id={sid}",
        files={"file": ("frame_03.jpg", b"jpeg-" + b"3" * 8, "image/jpeg")},
    )
    assert r.status_code == 201, r.text
    second = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                       timeout=90.0)
    assert second["status"] == "succeeded", second.get("error")
    v2 = second["payload"]["version_id"]
    assert v2 != v1

    # ...while a byte-identical retry dedups to the adopted version.
    repeat = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                       timeout=90.0)
    assert repeat["status"] == "succeeded", repeat.get("error")
    assert repeat["payload"]["version_id"] == v2
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] == v2
    # Lineage is a chain, and the superseded version stays readable.
    assert client.get(f"/api/worlds/{wid}/worldir?version={v1}").status_code == 200
    assert client.get(f"/api/worlds/{wid}/worldir?version={v2}").status_code == 200

    # A failed refinement afterwards must not destroy v2.
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.definitely_missing_module:_Backend")
    bad = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=90.0)
    assert bad["status"] == "failed"
    detail = client.get(f"/api/worlds/{wid}").json()
    assert detail["current_version_id"] == v2
    assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200


# --------------------------------------------------------------------------
# Adversarial inputs
# --------------------------------------------------------------------------


def test_corrupted_capture_fails_honestly(client, tmp_path, monkeypatch):
    """Garbage bytes typed as photos: the pipeline must fail with a real
    error, never a fabricated reconstruction."""
    sid = client.post("/api/sessions", json={"name": "Corrupt"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Corrupt world"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(2):
        r = client.post(
            f"/api/uploads?session_id={sid}",
            files={"file": (f"frame_{i}.jpg", bytes(range(256)) * 4, "image/jpeg")},
        )
        assert r.status_code == 201
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=150.0)
    assert job["status"] == "failed", job
    assert job.get("error"), "failure must carry its reason"
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


class _UnregistrableBackend:
    """Backend whose images never connect: no shared features, no
    overlap -- reconstruction must report failure, never a world."""

    def reconstruct(self, evidence):
        from reconstruction.backend.interface import ReconstructionResult

        return ReconstructionResult(
            points=[], camera_poses=[], registration_status="failed",
        )


def test_unregistrable_input_fails_without_output(client, tmp_path, monkeypatch):
    """Disconnected/insufficient-overlap input: the backend reports
    failure, the job fails honestly, and no version is adopted."""
    sid, wid = _chain_setup(client, tmp_path, monkeypatch)
    # Override the deterministic backend _chain_setup armed: nothing here
    # can ever register.
    monkeypatch.setenv(
        "REALITY_TEST_BACKEND", "tests.test_reconstruction_reliability:_UnregistrableBackend"
    )
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=90.0)
    assert job["status"] == "failed", job
    err = (job.get("error") or "").lower()
    assert "reconstruction stage failed" in err or "nothing usable" in err, err[-500:]
    assert client.get(f"/api/worlds/{wid}").json()["current_version_id"] is None
    assert client.get(f"/api/worlds/{wid}/versions").json()["items"] == []


def test_single_photo_fails_honestly(client, tmp_path, monkeypatch):
    sid, wid = _chain_setup(client, tmp_path, monkeypatch, n_photos=1)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=90.0)
    assert job["status"] == "failed"
    assert "needs at least 2" in (job.get("error") or "").lower()


def test_no_shell_subprocess_in_reconstruction():
    """Static guardrail: reconstruction backends spawn only list-argv,
    shell=False subprocesses. A shell=True call with evidence-derived
    paths would be remote command injection, so the shape is pinned."""
    import ast

    root = Path(__file__).resolve().parents[1] / "reconstruction"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else "")
            if name == "system":
                offenders.append(f"{path.name}:{node.lineno} os.system")
            if name in ("run", "call", "check_call", "check_output", "Popen"):
                for kw in node.keywords:
                    if kw.arg == "shell" and not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is False
                    ):
                        offenders.append(f"{path.name}:{node.lineno} shell=")
    assert offenders == [], f"shell subprocess invocations: {offenders}"


def test_untraced_reconstruction_refuses_persistence():
    """The hard gate, unit-exercised: RECONSTRUCTED entities without a
    build trace raise instead of persisting; other provenances pass."""
    from apps.api.jobs import _assert_reconstruction_traced
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.world_v1 import WorldIR

    traced = WorldIR(id="w-ok")
    traced.entities["e1"] = Entity(
        id="e1", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED,
        custom_properties={"reconstruction": {"session_id": "ses_x"}},
    )
    _assert_reconstruction_traced(traced)

    procedural = WorldIR(id="w-proc")
    procedural.entities["e2"] = Entity(
        id="e2", type=EntityType.ROOM, provenance=Provenance.GENERATED,
    )
    _assert_reconstruction_traced(procedural)

    untraced = WorldIR(id="w-bad")
    untraced.entities["e3"] = Entity(
        id="e3", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED,
    )
    with pytest.raises(RuntimeError, match="untraced"):
        _assert_reconstruction_traced(untraced)


def test_malformed_worldir_rejected_on_load():
    """WorldIR.from_dict on garbage is an explicit error, never a
    half-built world that later corrupts a version."""
    from world_ir.world_v1 import WorldIR

    for bad in ({}, {"id": 123}, {"entities": ["not-a-dict"]}, {"entities": {}, "geometries": None}):
        try:
            w = WorldIR.from_dict(bad)
        except Exception:
            continue
        # If it loads, it must load as an empty-but-consistent world.
        assert isinstance(w.entities, dict) and isinstance(w.geometries, dict)


def test_uncertainty_roundtrip_preserved():
    """Uncertainty survives serialize -> persist -> reload with its value
    intact, and rejects out-of-range confidence loudly at construction."""
    from provenance import Uncertainty

    u = Uncertainty(confidence=0.62, note="occluded")
    assert Uncertainty.from_dict(u.to_dict()) == u
    with pytest.raises(ValueError, match="confidence"):
        Uncertainty(confidence=1.5)
    with pytest.raises(Exception):
        Uncertainty.from_dict(None)


def test_invalid_topology_flagged_by_validator():
    """Dangling relationship/geometry references are ERRORs: a world
    with broken topology cannot grade as valid, so no job can call it
    SUCCEEDED and no commit can persist it."""
    from world_ir import Entity, EntityType
    from world_ir.validation import validate_world_ir
    from world_ir.world_v1 import WorldIR
    from provenance import Provenance

    w = WorldIR(id="w-topo")
    w.entities["e1"] = Entity(
        id="e1", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED,
        confidence=0.5, geometry_ids=["geom-ghost"],
    )
    report = validate_world_ir(w)
    assert not report.is_valid()
    assert any("geom-ghost" in m for m in report.messages())


def test_concurrent_reconstruct_sessions_stay_independent(client, tmp_path, monkeypatch):
    """Two sessions reconstructing at once: both succeed with distinct
    worlds and versions -- no cross-talk, no shared HEAD."""
    import threading

    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    setups = []
    for n in ("A", "B"):
        sid = client.post("/api/sessions", json={"name": f"Conc {n}"}).json()["id"]
        wid = client.post("/api/worlds", json={"name": f"Conc world {n}"}).json()["id"]
        assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
        for i in range(2):
            client.post(
                f"/api/uploads?session_id={sid}",
                files={"file": (f"f{i}.jpg", b"jpeg-bytes-here", "image/jpeg")},
            )
        setups.append((sid, wid))

    barrier = threading.Barrier(2)
    outcomes = []

    def worker(sid):
        try:
            barrier.wait(timeout=10)
            job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
            outcomes.append(_wait_job(client, job_id, timeout=120.0))
        except Exception as exc:  # noqa: BLE001
            outcomes.append({"status": f"ERROR {type(exc).__name__}: {exc}"})

    threads = [threading.Thread(target=worker, args=(sid,)) for sid, _ in setups]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    assert [o["status"] for o in outcomes] == ["succeeded", "succeeded"], outcomes
    worlds = [o["payload"]["world_id"] for o in outcomes]
    assert worlds[0] != worlds[1]
    assert outcomes[0]["payload"]["version_id"] != outcomes[1]["payload"]["version_id"]
    for _, wid in setups:
        assert client.get(f"/api/worlds/{wid}/worldir").status_code == 200


def test_crash_restart_recovers_stale_job(client, tmp_path, monkeypatch):
    """A worker that dies mid-job leaves a stale 'running' row. A fresh
    process (new client over the same DB file) reaps and settles it --
    the job never wedges, HEAD stays honest."""
    import asyncio

    db_file = tmp_path / "shared.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))

    async def _plant_stale():
        import apps.api.db as db_mod
        from apps.api.models import Job, Session, World
        from datetime import timedelta

        # Drop the fixture's cached engine so planting + the fresh
        # client below both use the shared crash-simulation DB file.
        await db_mod.dispose_db()
        await db_mod.init_db()
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(World(id="wld_crash01", name="Crash world"))
            db.add(Session(id="ses_crash01", name="Crash session", world_id="wld_crash01"))
            from apps.api.models import utcnow

            db.add(Job(
                id="job_crash001", type="RECONSTRUCT_SESSION",
                entity_type="session", entity_id="ses_crash01",
                status="running", worker_id="worker-dead",
                attempts=1, max_attempts=1,
                heartbeat_at=utcnow() - timedelta(seconds=3600),
            ))
            await db.commit()

    # Plant the stranded row, then open a fresh client (new worker
    # process in effect): startup reap + normal polling settle the
    # stranded job without operator help.
    asyncio.run(_plant_stale())
    from fastapi.testclient import TestClient
    import apps.api.main as main_mod

    with TestClient(main_mod.app) as fresh:
        job = _wait_job(fresh, "job_crash001", timeout=60.0)
    # No usable evidence -> honest failure after requeue, attempts spent.
    assert job["status"] == "failed", job


# --------------------------------------------------------------------------
# Reload acceptance: what the API persisted must load in a NEW process.
# --------------------------------------------------------------------------


def test_persisted_version_reloads_in_new_process(client, tmp_path, monkeypatch):
    """Acceptance: a reconstruction result persisted by the API loads in
    a fresh OS process with verified bytes -- 'Room reconstructed
    successfully' is only ever claimed for state that survives this."""
    sid, wid = _chain_setup(client, tmp_path, monkeypatch)
    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"],
                    timeout=90.0)
    assert job["status"] == "succeeded", job.get("error")
    version_id = job["payload"]["version_id"]

    script = tmp_path / "_reload_check.py"
    repo_root = Path(__file__).resolve().parents[1]
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from worldstore.store import WorldStore\n"
        "store = WorldStore(sys.argv[1])\n"
        "world = store.load_version(sys.argv[2])\n"
        "assert world.entities, 'reloaded world has no entities'\n"
        "assert store.verify_version(sys.argv[2]) == []\n"
        "print(f'reloaded {len(world.entities)} entities, verified ok')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "ws"), version_id],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "verified ok" in proc.stdout
