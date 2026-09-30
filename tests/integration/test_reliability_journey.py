"""Product-path reliability: kill things, restart things, break stages -- the last valid world survives.

Real engines, real South Building photographs, the same API/worker/WorldStore the Studio uses.
V1 (SIX photos) is built once and its whole app root (database + WorldStore + artifacts + COLMAP state) is
copied per test, so every scenario starts from the same real V1.

Invariants asserted after EVERY injected failure:
  * HEAD is still V1 and V1's WorldIR reads back unchanged
  * the committed COLMAP state is the one V1 was built from (never advanced by a failed/rejected run)
  * every uploaded photograph is still known to the world (nothing lost)
  * nothing reports success that did not happen
and then that the same world recovers and moves forward once the fault is gone.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest

from tests.integration.test_continuous_evidence import _session_manifest
from tests.integration.test_progressive_product_journey import (  # noqa: F401 -- harness + marks
    IMGS,
    _post,
    _settle,
    _worldir,
    app_client,
    pytestmark,
)

SIX, C, D = IMGS[15:21], IMGS[21:24], IMGS[24:27]


class SimulatedCrash(BaseException):
    """The process died: nothing in the code under test may absorb it."""


@pytest.fixture(scope="module")
def v1_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v1")
    with app_client(root) as c:
        wid = _post(c, SIX)["world_id"]
        st = _settle(c, wid)
        assert st["model"]["images_registered"] == 6 and len(st["versions"]) == 1
        (root / "wid.json").write_text(json.dumps({"wid": wid, "v1": st["model"]["version_id"]}))
    return root


@pytest.fixture
def root(v1_root, tmp_path):
    shutil.copytree(v1_root, tmp_path / "app")
    return tmp_path / "app"


def _ids(root):
    return json.loads((root / "wid.json").read_text())


def _assert_head_is_v1(c, root, st, *, photos):
    ids = _ids(root)
    assert st["model"]["version_id"] == ids["v1"] and len(st["versions"]) == 1, "HEAD moved"
    assert _worldir(c, ids["wid"], ids["v1"])["entities"], "V1 no longer readable"
    assert len(_session_manifest(root, ids["wid"])["images"]) == 6, "COLMAP state advanced without an adopted version"
    assert len(st["evidence"]) == photos, "evidence lost"


@pytest.mark.parametrize("stage,target", [
    ("WorldIR creation", "engine.pipeline.vertical_slice.compile_reconstruction_to_world"),
    ("world_delta", "engine.pipeline.world_delta.compute_delta"),
    ("candidate acceptance", "engine.pipeline.world_delta.decide"),
])
def test_a_crash_in_a_late_stage_leaves_V1_and_the_colmap_state_untouched_then_the_world_recovers(
        root, monkeypatch, stage, target):
    import importlib

    module, _, attr = target.rpartition(".")
    mod = importlib.import_module(module)

    def boom(*a, **k):
        raise RuntimeError(f"simulated crash in {stage}")

    ids = _ids(root)
    with app_client(root) as c:
        monkeypatch.setattr(mod, attr, boom)
        _post(c, C, world_id=ids["wid"])
        st = _settle(c, ids["wid"])
        print(f"[{stage}] state={st['state']} last_run={st.get('last_run')}")
        assert st["state"] in ("FAILED", "PARTIALLY_COMPLETE")
        assert not (st.get("last_run") or {}).get("adopted"), "a crashed run must not claim adoption"
        _assert_head_is_v1(c, root, st, photos=9)
        monkeypatch.undo()

        _post(c, D, world_id=ids["wid"])                       # fault gone, more evidence: the world moves forward
        st2 = _settle(c, ids["wid"])
        assert len(st2["versions"]) == 2 and st2["model"]["images_used"] == 12, st2["model"]
        assert (st2["last_run"] or {}).get("adopted") is True
        assert len(_session_manifest(root, ids["wid"])["images"]) == 12


def test_crash_after_worldstore_adoption_but_before_colmap_commit_does_not_diverge(root, monkeypatch):
    """The worst ordering: the version IS adopted, then the process dies before the COLMAP state advances.
    The next run must converge (the base is simply one version behind), never double-count or corrupt."""
    from reconstruction.colmap_session import ColmapSession

    ids = _ids(root)
    with app_client(root) as c:
        real_commit = ColmapSession.commit

        def die(self):
            raise RuntimeError("fault between WorldStore commit and COLMAP commit")

        monkeypatch.setattr(ColmapSession, "commit", die)
        _post(c, C, world_id=ids["wid"])
        st = _settle(c, ids["wid"])                      # every retry hits the same fault
        assert len(st["versions"]) == 2, ("adopted exactly once, retries must not duplicate versions", st["versions"])
        v2 = st["model"]["version_id"]
        assert len(_session_manifest(root, ids["wid"])["images"]) == 6      # COLMAP base one version behind
    monkeypatch.setattr(ColmapSession, "commit", real_commit)

    with app_client(root) as c2:                                             # process restart
        st = c2.get(f"/api/worlds/{ids['wid']}/status").json()
        assert st["model"]["version_id"] == v2 and _worldir(c2, ids["wid"], v2)["entities"]
        _post(c2, D, world_id=ids["wid"])
        st2 = _settle(c2, ids["wid"])
        print("[converged]", st2["model"]["images_used"], st2["model"]["images_registered"], st2["last_run"])
        assert st2["model"]["images_used"] == 12
        assert st2["model"]["images_registered"] >= st["model"]["images_registered"], "registration collapsed"
        assert len(_session_manifest(root, ids["wid"])["images"]) == 12, "COLMAP base failed to catch up"


class _Server:
    """The real API + worker in its own process, so it can be really killed (TerminateProcess)."""

    def __init__(self, root, port):
        import os
        import subprocess
        import sys

        env = dict(os.environ, DATABASE_URL=f"sqlite+aiosqlite:///{(root / 'app.db').as_posix()}",
                   STORAGE_ROOT=str(root / "artifacts"), WORLDSTORE_ROOT=str(root / "ws"),
                   JOB_STALE_AFTER_SECONDS="30", PYTHONPATH=os.getcwd())
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "apps.api.main:app", "--port", str(port)],
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import httpx

        self.http = httpx.Client(base_url=self.url, timeout=60)
        for _ in range(120):
            try:
                if self.http.get("/api/worlds").status_code < 500:
                    return
            except Exception:
                time.sleep(1)
        raise AssertionError("server did not start")

    def kill(self):
        import subprocess

        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)   # incl. COLMAP children
        self.proc.wait(timeout=30)

    def status(self, wid):
        return self.http.get(f"/api/worlds/{wid}/status").json()

    def post(self, paths, wid):
        r = self.http.post("/api/reconstructions", data={"world_id": wid},
                           files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths])
        assert r.status_code == 202, r.text
        return r.json()

    def settle(self, wid, timeout=1200):
        end = time.time() + timeout
        while time.time() < end:
            st = self.status(wid)
            if not st["in_progress"]:
                return st
            time.sleep(2)
        raise AssertionError("never settled")


@pytest.mark.parametrize("kill_after_s", [8, 30])
def test_the_whole_process_killed_mid_reconstruction_then_restarted_keeps_V1_and_the_world_continues(root, kill_after_s):
    """A hard kill (process tree, COLMAP children included) while a rebuild is in flight."""
    ids = _ids(root)
    srv = _Server(root, 8765)
    try:
        srv.post(C, ids["wid"])
        time.sleep(kill_after_s)
        mid = srv.status(ids["wid"])
        print(f"[kill at {kill_after_s}s] in_progress={mid['in_progress']} state={mid['state']}")
        assert mid["in_progress"], "the job finished before the kill; the test would prove nothing"
    finally:
        srv.kill()

    srv2 = _Server(root, 8766)
    try:
        st = srv2.status(ids["wid"])
        # Exactly one of two states -- never a mixed one:
        #   A: the kill landed before adoption   -> HEAD is V1, COLMAP base untouched (6 photos)
        #   B: the kill landed after adoption    -> HEAD is a valid V2; the COLMAP base may lag one version (6)
        #      or have caught up (9), but never claims photos the authoritative World does not have
        n_versions = len(st["versions"])
        assert n_versions in (1, 2), st["versions"]
        if n_versions == 1:
            assert st["model"]["version_id"] == ids["v1"], "HEAD changed without a new version"
        else:
            assert st["model"]["version_id"] != ids["v1"]
        assert srv2.http.get(f"/api/worlds/{ids['wid']}/worldir",
                             params={"version": st["model"]["version_id"]}).json()["entities"], "HEAD unreadable"
        assert len(_session_manifest(root, ids["wid"])["images"]) in ((6,) if n_versions == 1 else (6, 9)),             "COLMAP state disagrees with the authoritative World"
        print(f"[state after restart] {'A: V1 still HEAD' if n_versions == 1 else 'B: V2 adopted before the kill'}")
        assert len(st["evidence"]) == 9, "the upload was lost"
        st = srv2.settle(ids["wid"], timeout=1500)                 # the stranded job is reaped and resumed
        print("[after restart]", st["state"], len(st["versions"]), st["model"]["images_used"])
        assert st["model"]["images_used"] == 9 and len(st["versions"]) in (1, 2)
        for v in st["versions"]:
            assert srv2.http.get(f"/api/worlds/{ids['wid']}/worldir", params={"version": v["id"]}).json()["entities"]
        srv2.post(D, ids["wid"])                                    # and the same world keeps improving
        st = srv2.settle(ids["wid"], timeout=1500)
        nums = [v["number"] for v in st["versions"]]
        assert nums == list(range(1, len(nums) + 1)) and st["model"]["images_used"] == 12
    finally:
        srv2.kill()


def test_two_uploads_racing_on_one_world_serialize_without_corrupting_versions_or_evidence(root):
    ids = _ids(root)
    with app_client(root) as c:
        r1 = _post(c, C, world_id=ids["wid"])
        r2 = _post(c, D, world_id=ids["wid"])              # arrives while the first is still running
        assert r1["world_id"] == r2["world_id"] == ids["wid"]
        st = _settle(c, ids["wid"])
        nums = [v["number"] for v in st["versions"]]
        print("[race]", nums, st["model"]["images_used"], st["model"]["images_registered"])
        assert nums == list(range(1, len(nums) + 1)), "version numbering corrupted"
        assert st["model"]["images_used"] == 12 and len(st["evidence"]) == 12
        parents = {v["id"]: v.get("parent_version_id") for v in st["versions"]}
        assert sum(1 for p in parents.values() if p is None) == 1, "lineage broken"
        for v in st["versions"]:
            assert _worldir(c, ids["wid"], v["id"])["entities"]
        assert len(_session_manifest(root, ids["wid"])["images"]) == 12


# ------------------------------------------------- remaining durable boundaries (validation .. read-back)


@pytest.mark.parametrize("stage,target", [
    ("WorldIR validation", "world_ir.validation.validate_world_ir"),
    ("conflict reconciliation", "engine.pipeline.world_delta.reconcile_conflicts"),
    ("WorldStore commit (before it writes)", "apps.api.worldstore_service.commit_version"),
])
def test_a_crash_before_the_worldstore_commit_never_advances_head_or_colmap_state(root, monkeypatch, stage, target):
    import importlib

    module, _, attr = target.rpartition(".")
    mod = importlib.import_module(module)

    async def _aboom(*a, **k):
        raise RuntimeError(f"simulated crash: {stage}")

    def _boom(*a, **k):
        raise RuntimeError(f"simulated crash: {stage}")

    ids = _ids(root)
    with app_client(root) as c:
        monkeypatch.setattr(mod, attr, _aboom if attr == "commit_version" else _boom)
        _post(c, C, world_id=ids["wid"])
        st = _settle(c, ids["wid"])
        assert not (st.get("last_run") or {}).get("adopted")
        _assert_head_is_v1(c, root, st, photos=9)
        monkeypatch.undo()
        _post(c, D, world_id=ids["wid"])
        st2 = _settle(c, ids["wid"])
        assert st2["model"]["images_used"] == 12 and len(st2["versions"]) == 2


@pytest.mark.parametrize("stage", ["crash right after the WorldStore commit", "read-back verification fails"])
def test_a_fault_after_the_commit_leaves_a_valid_immutable_head_and_the_world_converges(root, monkeypatch, stage):
    import apps.api.worldstore_service as wss

    ids = _ids(root)
    real_commit = wss.commit_version

    async def commit_then_die(*a, **k):
        row = await real_commit(*a, **k)
        raise RuntimeError("simulated crash after commit")

    with app_client(root) as c:
        if stage.startswith("crash"):
            monkeypatch.setattr(wss, "commit_version", commit_then_die)
        else:
            monkeypatch.setattr(wss.get_store().__class__, "verify_version",
                                lambda self, vid: [{"reason": "simulated hash mismatch"}])
        _post(c, C, world_id=ids["wid"])
        st = _settle(c, ids["wid"])
        print(f"[{stage}] state={st['state']} versions={len(st['versions'])}")
        assert not (st["state"] == "COMPLETE" and (st.get("last_run") or {}).get("adopted") is None)
        for v in st["versions"]:                                   # whatever exists is valid and readable
            assert _worldir(c, ids["wid"], v["id"])["entities"]
        nums = [v["number"] for v in st["versions"]]
        assert nums == list(range(1, len(nums) + 1)) and len(st["evidence"]) == 9
        monkeypatch.undo()
        _post(c, D, world_id=ids["wid"])
        st2 = _settle(c, ids["wid"])
        assert st2["model"]["images_used"] == 12
        nums = [v["number"] for v in st2["versions"]]
        assert nums == list(range(1, len(nums) + 1))
        assert sum(1 for v in st2["versions"] if v.get("parent_version_id") is None) == 1
        assert len(_session_manifest(root, ids["wid"])["images"]) == 12, "COLMAP base failed to converge"


# ------------------------------------------------------------------------------------- rollback


def test_rollback_moves_head_only_keeps_history_and_evidence_and_the_next_version_chains_onto_the_target(root):
    """V1 -> V2 -> V3 -> rollback to V1 -> new evidence -> V4 whose parent is V1 (the chosen rollback model).
    History is immutable, evidence is never dropped, the COLMAP base of the rolled-back-from head is set aside."""
    import json as _json

    X = IMGS[27:30]
    ids = _ids(root)
    wid, v1 = ids["wid"], ids["v1"]
    with app_client(root) as c:
        _post(c, C, world_id=wid); st2 = _settle(c, wid)
        _post(c, D, world_id=wid); st3 = _settle(c, wid)
        v2, v3 = st2["model"]["version_id"], st3["model"]["version_id"]
        assert len({v1, v2, v3}) == 3 and [v["number"] for v in st3["versions"]] == [1, 2, 3]
        before = {v: _json.dumps(_worldir(c, wid, v), sort_keys=True) for v in (v1, v2, v3)}

        r = c.post(f"/api/worlds/{wid}/rollback", json={"version_id": v1})
        assert r.status_code == 200, r.text
        st = c.get(f"/api/worlds/{wid}/status").json()
        assert st["model"]["version_id"] == v1 and len(st["versions"]) == 3, "rollback must not delete or add versions"
        assert _json.dumps(_worldir(c, wid), sort_keys=True) == before[v1], "HEAD content is not V1"
        for v in (v1, v2, v3):
            assert _json.dumps(_worldir(c, wid, v), sort_keys=True) == before[v], f"{v} was mutated by rollback"
        assert len(st["evidence"]) == 12, "evidence was dropped by rollback"
        not_in_model = [e for e in st["evidence"] if not e["in_current_model"]]
        assert len(not_in_model) == 6, "photos only later versions used must show as outside the current model"
        assert not (root / "ws" / "colmap-sessions" / wid / "current").exists(), "stale COLMAP base still authoritative"
        assert any(p.name.startswith("superseded-by-rollback") for p in (root / "ws" / "colmap-sessions" / wid).iterdir())

        # unknown / foreign version is refused, HEAD untouched
        assert c.post(f"/api/worlds/{wid}/rollback", json={"version_id": "nope"}).status_code == 422
        assert c.get(f"/api/worlds/{wid}/status").json()["model"]["version_id"] == v1

        _post(c, X, world_id=wid)
        st4 = _settle(c, wid)
        heads = [v for v in st4["versions"] if v["is_current"]]
        assert len(st4["versions"]) == 4 and len(heads) == 1
        assert heads[0]["parent_version_id"] == v1, "the version after a rollback must chain onto the rolled-back HEAD"
        assert len(st4["evidence"]) == 15 and (st4["last_run"] or {}).get("adopted") is True
        for v in (v1, v2, v3):
            assert _json.dumps(_worldir(c, wid, v), sort_keys=True) == before[v]


# ------------------------------------------------------------------------------ user-supplied metric scale


def test_a_user_measured_distance_metricizes_the_same_world_into_a_new_version_and_says_so(root):
    """V1 is RELATIVE. The user supplies a measured distance between two of its photographs: the SAME world gets
    a new version whose scale is metric, with the source on record. Invalid references change nothing."""
    ids = _ids(root)
    wid = ids["wid"]
    with app_client(root) as c:
        st1 = c.get(f"/api/worlds/{wid}/status").json()
        assert st1["model"]["scale"]["state"] != "metric", "V1 must not claim meters"
        registered = [e["id"] for e in st1["evidence"] if e["registered"]]
        a, b = registered[0], registered[-1]
        # A physically consistent "measurement": the separation of those two cameras in the world's own frame.
        # (An arbitrary number such as 1.2 m would shrink a building to 0.4 m; the fixed-metre plane tolerances
        # then find no walls and the adoption gate rightly rejects that candidate.)
        import math
        import sqlite3

        con = sqlite3.connect(root / "app.db")
        cams = json.loads(con.execute("select report from world_versions where id=?", (ids["v1"],)).fetchone()[0])["cameras"]
        con.close()
        sep = math.dist(cams[a], cams[b])
        assert sep > 0.5, cams

        # refused: same photo twice, a stranger, a non-distance -- and nothing was queued or stored
        assert c.post(f"/api/worlds/{wid}/scale", json={"evidence_id_a": a, "evidence_id_b": a, "distance_m": 1.0}).status_code == 422
        assert c.post(f"/api/worlds/{wid}/scale", json={"evidence_id_a": a, "evidence_id_b": "nope", "distance_m": 1.0}).status_code == 422
        assert c.post(f"/api/worlds/{wid}/scale", json={"evidence_id_a": a, "evidence_id_b": b, "distance_m": -1.0}).status_code == 422
        assert len(c.get(f"/api/worlds/{wid}/status").json()["versions"]) == 1

        r = c.post(f"/api/worlds/{wid}/scale", json={"evidence_id_a": a, "evidence_id_b": b, "distance_m": sep,
                                                     "method": "tape measure"})
        assert r.status_code == 202, r.text
        st2 = _settle(c, wid)
        m = st2["model"]
        print("[scale]", m["scale"], m["changes"][:2])
        assert len(st2["versions"]) == 2 and m["version_id"] != ids["v1"], "metricizing must create a new version"
        assert [v for v in st2["versions"] if v["is_current"]][0]["parent_version_id"] == ids["v1"]
        assert m["scale"]["state"] == "metric" and m["scale"]["meters_per_unit"] > 0
        assert m["scale"]["references"][0]["distance_m"] == pytest.approx(sep) and m["scale"]["references"][0]["method"] == "tape measure"
        assert any("Scale changed" in t and "metric" in t for t in m["changes"]), m["changes"]
        assert st2["model"]["images_used"] == 6 and len(st2["evidence"]) == 6          # same world, same evidence
        assert _worldir(c, wid, ids["v1"])["entities"] and _worldir(c, wid)["entities"]
        assert m["identity"] is not None                                                # identity decisions recorded
