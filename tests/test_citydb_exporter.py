"""3DCityDB export (exporters/citydb): the invocation contract against a FAKE `citydb` CLI, and one env-gated test
against a REAL database.

Honesty boundary of this file: the TestFakeCli tests prove that the exporter builds the vendor's documented command,
keeps the password off the command line, is idempotent by construction (--import-mode=delete), reports failures and
a missing tool as such, and verifies by reading back. They do NOT prove that 3DCityDB accepts the data: that is
TestRealDatabase, which runs only when a real `citydb` tool and a real database are configured
(CITYDB_TOOL / CITYDB_TEST_HOST / CITYDB_TEST_DB / CITYDB_TEST_USER / CITYDB_PASSWORD) and is otherwise SKIPPED --
in this repository's CI and on the development machine it is skipped, so the real-database verification is BLOCKED,
not passed.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from exporters.citydb import (
    STATUS_FAILED, STATUS_IMPORTED, STATUS_NOTHING, STATUS_PREVIEWED, STATUS_UNAVAILABLE,
    CityDbTarget, export_to_3dcitydb, find_citydb_tool,
)
from tests.test_cityjson_exporter import _world
from world_ir.world_v1 import WorldIR

_FAKE_CITYDB = textwrap.dedent("""\
    import json, os, sys
    from pathlib import Path

    args = sys.argv[1:]
    db = Path(os.environ["FAKE_DB"])            # the fake "database": objectid -> attributes
    log = Path(os.environ["FAKE_LOG"])
    with open(log, "a") as f:
        f.write(json.dumps({"args": args, "env": {k: v for k, v in os.environ.items() if k.startswith("CITYDB_")}}) + "\\n")

    def opt(name):
        for a in args:
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return None

    if args == ["--version"]:
        print("citydb-tool 1.3.0 (fake)")
    elif args[:2] == ["import", "cityjson"]:
        if os.environ.get("FAKE_FAIL"):
            sys.stderr.write("ERROR: feature 'x' has an invalid geometry\\n")
            sys.exit(2)
        if "--preview" in args:
            print("preview: nothing imported"); sys.exit(0)
        src = json.loads(Path(args[-1]).read_text())
        store = json.loads(db.read_text()) if db.exists() else {}
        mode = opt("--import-mode")
        for oid, obj in src["CityObjects"].items():
            if mode == "delete" or oid not in store:
                store[oid] = {"type": obj["type"], "lineage": opt("--lineage")}
        db.write_text(json.dumps(store))
        print("imported %d features" % len(src["CityObjects"]))
    elif args[:2] == ["export", "cityjson"]:
        out = args[args.index("-o") + 1]
        store = json.loads(db.read_text()) if db.exists() else {}
        if os.environ.get("FAKE_DROP_ONE") and store:
            store.pop(sorted(store)[0])
        Path(out).write_text(json.dumps({"type": "CityJSON", "version": "2.0", "CityObjects": store}))
    else:
        sys.stderr.write("unknown command\\n"); sys.exit(1)
""")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A platform-native `citydb` executable running the fake above; returns (tool_path, db_path, log_path)."""
    script = tmp_path / "fake_citydb.py"
    script.write_text(_FAKE_CITYDB)
    db, log = tmp_path / "db.json", tmp_path / "calls.jsonl"
    log.write_text("")
    if os.name == "nt":
        tool = tmp_path / "citydb.bat"
        tool.write_text(f'@echo off\r\n@"{sys.executable}" "{script}" %*\r\n')
    else:
        tool = tmp_path / "citydb"
        tool.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_DB", str(db))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CITYDB_PASSWORD", "s3cret-pw")
    return str(tool), db, log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines() if line]


TARGET = CityDbTarget(host="db.example", database="citydb_test", username="re_user", port=5544, schema="cityv5")


class TestFakeCli:
    def test_imports_with_the_documented_command_and_connection_in_the_environment(self, fake):
        tool, db, log = fake
        rep = export_to_3dcitydb(_world(), TARGET, tool=tool)
        assert rep.status == STATUS_IMPORTED and rep.tool_version.startswith("citydb-tool 1.3.0")
        imp = [c for c in _calls(log) if c["args"][:2] == ["import", "cityjson"]][0]
        assert "--import-mode=delete" in imp["args"] and "--fail-fast" in imp["args"]
        assert any(a.startswith("--lineage=reality-engine:") for a in imp["args"])
        assert imp["env"] == {"CITYDB_HOST": "db.example", "CITYDB_PORT": "5544", "CITYDB_NAME": "citydb_test",
                              "CITYDB_SCHEMA": "cityv5", "CITYDB_USERNAME": "re_user", "CITYDB_PASSWORD": "s3cret-pw"}
        assert set(json.loads(db.read_text())) == set(rep.objects) and rep.objects

    def test_the_password_is_never_on_the_command_line_or_in_the_report(self, fake):
        tool, _, log = fake
        rep = export_to_3dcitydb(_world(), TARGET, tool=tool)
        assert "s3cret-pw" not in json.dumps(rep.to_dict())
        assert all("s3cret-pw" not in a for c in _calls(log) for a in c["args"])
        assert not any(a in ("-p", "--db-password") or a.startswith("--db-password") for a in rep.command)

    def test_exporting_twice_is_idempotent_the_same_objects_not_duplicates(self, fake):
        tool, db, _ = fake
        first = export_to_3dcitydb(_world(), TARGET, tool=tool)
        before = json.loads(db.read_text())
        second = export_to_3dcitydb(_world(), TARGET, tool=tool)
        assert json.loads(db.read_text()).keys() == before.keys()
        assert first.cityjson_sha256 == second.cityjson_sha256        # deterministic payload

    def test_readback_verifies_every_object_and_detects_a_missing_one(self, fake, monkeypatch):
        tool, _, _ = fake
        ok = export_to_3dcitydb(_world(), TARGET, tool=tool, verify=True)
        assert ok.verified is True and ok.missing_after_readback == ()
        monkeypatch.setenv("FAKE_DROP_ONE", "1")
        bad = export_to_3dcitydb(_world(), TARGET, tool=tool, verify=True)
        assert bad.status == STATUS_IMPORTED and bad.verified is False and len(bad.missing_after_readback) == 1

    def test_a_failing_import_is_reported_with_the_tools_message_and_leaves_the_database_alone(self, fake, monkeypatch):
        tool, db, _ = fake
        monkeypatch.setenv("FAKE_FAIL", "1")
        rep = export_to_3dcitydb(_world(), TARGET, tool=tool, verify=True)
        assert rep.status == STATUS_FAILED and "invalid geometry" in rep.detail and "exited 2" in rep.detail
        assert rep.verified is None and not db.exists()

    def test_preview_runs_the_tool_but_changes_nothing_and_is_reported_as_a_preview(self, fake):
        tool, db, log = fake
        rep = export_to_3dcitydb(_world(), TARGET, tool=tool, preview=True, verify=True)
        assert rep.status == STATUS_PREVIEWED and "--preview" in rep.command and not db.exists()
        assert rep.verified is None                                   # nothing was imported, so nothing to read back

    def test_a_timeout_is_a_failure_that_says_rerunning_is_safe(self, fake):
        tool, _, _ = fake

        def slow(cmd, **kw):
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "citydb-tool 1.3.0", "")
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

        rep = export_to_3dcitydb(_world(), TARGET, tool=tool, runner=slow, timeout_s=3)
        assert rep.status == STATUS_FAILED and "timed out" in rep.detail and "idempotent" in rep.detail

    def test_a_tool_that_cannot_start_is_a_failure(self, fake):
        tool, _, _ = fake

        def boom(cmd, **kw):
            raise OSError("exec format error")

        rep = export_to_3dcitydb(_world(), TARGET, tool=tool, runner=boom)
        assert rep.status == STATUS_FAILED and "could not start" in rep.detail


class TestHonestRefusals:
    def test_without_the_tool_nothing_runs_and_no_file_is_presented_as_an_export(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CITYDB_TOOL", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))                      # an empty directory: no `citydb` anywhere
        ran = []
        rep = export_to_3dcitydb(_world(), TARGET, runner=lambda *a, **k: ran.append(a))
        assert rep.status == STATUS_UNAVAILABLE and not ran and rep.command == ()
        assert "no file is presented as a 3DCityDB export" in rep.detail

    def test_a_named_tool_that_does_not_exist_is_unavailable_not_a_path_search(self, tmp_path):
        assert find_citydb_tool(str(tmp_path / "missing-citydb"), env={"PATH": str(tmp_path)}) is None

    def test_a_world_with_nothing_exportable_never_contacts_the_database(self, fake):
        tool, _, log = fake
        rep = export_to_3dcitydb(WorldIR(), TARGET, tool=tool)
        assert rep.status == STATUS_NOTHING and _calls(log) == []

    def test_skipped_entities_are_reported_with_reasons_not_dropped(self, fake):
        tool, _, _ = fake
        world = _world()
        from world_ir.schema_v1 import Entity, EntityType
        world.entities["no-geometry"] = Entity(id="no-geometry", type=EntityType.WALL)
        rep = export_to_3dcitydb(world, TARGET, tool=tool)
        assert "no-geometry" in rep.skipped and len(rep.skipped) == len(rep.skip_reasons)
        assert "no-geometry" not in rep.objects


_REAL = (os.environ.get("CITYDB_TOOL") and os.environ.get("CITYDB_TEST_HOST") and os.environ.get("CITYDB_TEST_DB")
         and os.environ.get("CITYDB_TEST_USER"))


@pytest.mark.skipif(not _REAL, reason="BLOCKED: needs a real 3DCityDB v5 + citydb-tool "
                    "(CITYDB_TOOL, CITYDB_TEST_HOST, CITYDB_TEST_DB, CITYDB_TEST_USER, CITYDB_PASSWORD) -- not available "
                    "on this machine; the exporter's behaviour against a real database is UNVERIFIED")
class TestRealDatabase:
    def test_world_round_trips_through_a_real_3dcitydb_idempotently(self):
        target = CityDbTarget(host=os.environ["CITYDB_TEST_HOST"], database=os.environ["CITYDB_TEST_DB"],
                              username=os.environ["CITYDB_TEST_USER"],
                              port=int(os.environ.get("CITYDB_TEST_PORT", "5432")),
                              schema=os.environ.get("CITYDB_TEST_SCHEMA", "citydb"))
        world = _world()
        first = export_to_3dcitydb(world, target, verify=True)
        assert first.status == STATUS_IMPORTED and first.verified is True, first.to_dict()
        second = export_to_3dcitydb(world, target, verify=True)
        assert second.status == STATUS_IMPORTED and second.verified is True, second.to_dict()
