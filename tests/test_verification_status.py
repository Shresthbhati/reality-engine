"""The verification status model (engine.core.verification): its rules, its consistency with the ledger, the API and CLI
that report it, and the 3DCityDB availability probe (AVAILABLE / UNAVAILABLE / FAILED).

The point of the model is that a synthetic result is never worded as a real-world one and a capability never claims
more than it has evidence for. These tests make that a checked property, not a convention.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from engine.core import verification as v
from engine.core.verification import Capability, Verification

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------- the rules

def test_every_capability_cites_evidence_files_that_exist():
    for c in v.CAPABILITIES:
        assert c.evidence, f"{c.id} claims a level with no evidence"
        for f in c.evidence:
            assert (ROOT / f).is_file(), f"{c.id}: evidence file {f} does not exist"


def test_a_synthetic_result_is_never_worded_as_real_and_real_names_its_data():
    for c in v.CAPABILITIES:
        if c.level == Verification.REAL_WORLD_VERIFIED:
            assert c.real_data, c.id
        else:
            assert not c.real_data, f"{c.id} names real data but is only {c.level.value}"
        if c.level == Verification.SYNTHETICALLY_VERIFIED:
            assert "SYNTHETIC" in c.label.upper() or "FAKE" in c.label.upper(), c.label
            assert "REAL-WORLD VERIFIED" not in c.label


def test_the_construction_rules_refuse_dishonest_records():
    with pytest.raises(ValueError, match="must name the real data"):
        Capability("x", "x", Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED")
    with pytest.raises(ValueError, match="only REAL-WORLD"):
        Capability("x", "x", Verification.SYNTHETICALLY_VERIFIED, "SYNTHETIC X", real_data="a camera")
    with pytest.raises(ValueError, match="worded as synthetic"):
        Capability("x", "x", Verification.SYNTHETICALLY_VERIFIED, "VERIFIED")
    with pytest.raises(ValueError, match="unknown scope"):
        Capability("x", "x", Verification.IMPLEMENTED, "IMPLEMENTED", scope="maybe")


def test_the_four_named_blockers_carry_the_mandated_labels():
    cap = v.by_id()
    assert cap["rgbd"].status_line == "SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING"
    assert cap["vio"].status_line == ("SYNTHETIC VIO PIPELINE VERIFIED -- REAL BACKEND / HARDWARE VERIFICATION "
                                       "PENDING")
    assert cap["indoor_architecture"].label == "SYNTHETIC INDOOR VERIFIED"
    assert cap["indoor_architecture"].level == Verification.SYNTHETICALLY_VERIFIED
    assert "FAKE" in cap["citydb"].label and cap["citydb"].level != Verification.REAL_WORLD_VERIFIED
    assert cap["citydb"].external_pending


def test_nothing_that_needs_absent_hardware_or_services_is_marked_real_world():
    for cid in ("rgbd", "vio", "indoor_architecture", "citydb", "export_ecosystem"):
        assert v.by_id()[cid].level != Verification.REAL_WORLD_VERIFIED, cid
        assert v.by_id()[cid].external_pending, cid


def test_ecosystem_exporters_are_scoped_out_of_the_prototype():
    assert v.by_id()["export_ecosystem"].scope == "optional"
    assert v.by_id()["export_core"].scope == "core"


# ------------------------------------------------------------------------------------------ the ledger agrees

def test_capabilities_yaml_verification_block_matches_the_registry():
    doc = yaml.safe_load((ROOT / ".agent" / "CAPABILITIES.yaml").read_text(encoding="utf-8"))
    block = doc["verification"]
    assert set(block) == set(v.by_id()), "ledger and registry list different capabilities"
    for cid, c in v.by_id().items():
        assert block[cid]["level"] == c.level.value, cid
        assert block[cid]["status"] == c.status_line, cid
        assert block[cid]["scope"] == c.scope, cid


def test_task_ledger_states_do_not_contradict_the_registry():
    tasks = {t["id"]: t for t in yaml.safe_load((ROOT / ".agent" / "TASKS.yaml").read_text(encoding="utf-8"))["tasks"]}
    for c in v.CAPABILITIES:
        for tid in c.ledger:
            assert tid in tasks, f"{c.id} points at ledger task {tid}, which does not exist"
            status = tasks[tid]["status"]
            assert status in ("DONE", "PARTIAL", "BLOCKED", "OPTIONAL_DEFERRED"), (tid, status)
            if c.level != Verification.REAL_WORLD_VERIFIED and c.external_pending and c.scope == "core":
                # a core capability with external verification pending may not be presented as unconditionally done
                assert status != "DONE" or "pending" in json.dumps(tasks[tid]).lower() or \
                    "unverified" in json.dumps(tasks[tid]).lower(), (tid, status)


# ----------------------------------------------------------------------------------------- API and CLI report it

def test_api_reports_the_registry_and_a_runtime_probe(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "art"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        body = c.get("/api/system/verification").json()
    assert body["levels"] == ["IMPLEMENTED", "SYNTHETICALLY VERIFIED", "REAL-WORLD VERIFIED"]
    assert body["external_pending_label"] == "EXTERNAL VERIFICATION PENDING"
    assert {c["id"] for c in body["capabilities"]} == set(v.by_id())
    assert sum(body["tally"].values()) == len(v.CAPABILITIES)
    rgbd = next(c for c in body["capabilities"] if c["id"] == "rgbd")
    assert rgbd["external_verification_pending"] is True and rgbd["level"] == "SYNTHETICALLY VERIFIED"
    assert {r["status"] for r in body["runtime"]} <= {"AVAILABLE", "UNAVAILABLE", "FAILED"}
    assert any(r["name"].startswith("3dcitydb") for r in body["runtime"])


def test_cli_prints_every_capability_with_its_pending_items(capsys):
    from apps.cli.main import main

    assert main(["verification"]) == 0
    out = capsys.readouterr().out
    for c in v.CAPABILITIES:
        assert c.id in out
    assert "SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING" in out
    assert main(["verification", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tally"]


# ---------------------------------------------------------------------------------- 3DCityDB availability probe

def _completed(rc=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr=err)


def test_citydb_probe_unavailable_when_the_tool_is_not_installed(tmp_path):
    from exporters.citydb import UNAVAILABLE, probe_citydb

    p = probe_citydb(env={"PATH": str(tmp_path)})
    assert p.status == UNAVAILABLE and p.tool is None and "not found" in p.detail


def test_citydb_probe_available_when_the_cli_runs_but_says_the_database_was_not_contacted(tmp_path):
    from exporters.citydb import AVAILABLE, probe_citydb

    fake = tmp_path / "citydb"
    fake.write_text("")
    p = probe_citydb(tool=str(fake), runner=lambda *a, **k: _completed(0, "citydb-tool 1.2.3\n"))
    assert p.status == AVAILABLE and p.version == "citydb-tool 1.2.3"
    assert "database behind it was not contacted" in p.detail, "AVAILABLE must not imply a real database was verified"


@pytest.mark.parametrize("runner,needle", [
    (lambda *a, **k: _completed(2, "", "java.lang.UnsupportedClassVersionError"), "exited 2"),
    (lambda *a, **k: _completed(0, "", ""), "exited 0"),
    (lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("citydb", 1)), "did not finish"),
    (lambda *a, **k: (_ for _ in ()).throw(OSError("exec format error")), "could not start"),
])
def test_citydb_probe_failed_when_the_tool_is_present_but_does_not_run(tmp_path, runner, needle):
    from exporters.citydb import FAILED, probe_citydb

    fake = tmp_path / "citydb"
    fake.write_text("")
    p = probe_citydb(tool=str(fake), runner=runner)
    assert p.status == FAILED and needle in p.detail and p.tool == str(fake)


def test_runtime_availability_reports_three_valued_statuses_and_never_raises():
    rows = v.runtime_availability()
    assert {r.status for r in rows} <= {"AVAILABLE", "UNAVAILABLE", "FAILED"}
    assert {"colmap", "3dcitydb (citydb CLI)"} <= {r.name for r in rows}
