"""3DCityDB v5 export: load a WorldIR into a REAL 3D City Database through the vendor's own importer.

3DCityDB is a database target (PostgreSQL/PostGIS schema ``citydb``), not a file format. A CityGML or CityJSON file
is NOT a 3DCityDB export, and nothing in this module pretends otherwise. The database schema is owned by the
3DCityDB project and changes between versions, so this module does NOT write SQL into it; it serializes the world
with the repo's CityJSON exporter and hands the file to the official ``citydb`` command-line tool
(``citydb import cityjson``, https://docs.3dcitydb.org/latest/citydb-tool/import-cityjson/), which owns schema
compatibility, geometry mapping, transactions and the ``objectid`` bookkeeping.

What this module owns:
  - tool discovery (explicit path, ``CITYDB_TOOL``, then ``citydb`` on PATH) -- absent tool => status UNAVAILABLE and
    NOTHING is run (no fallback file is presented as an export);
  - connection details from a typed target; the password travels ONLY in the child environment (``CITYDB_PASSWORD``),
    never on the command line, never in the report;
  - idempotent writes: ``--import-mode=delete`` replaces any feature with the same ``objectid`` before importing, so
    exporting the same world twice leaves the same features (objectids are the WorldIR entity ids, stable);
  - provenance: ``--lineage`` and ``--reason-for-update`` record world id / version in 3DCityDB's own metadata columns;
  - error recovery: a non-zero exit, a timeout, or an unreadable tool is reported as FAILED with the tool's own
    message; ``--fail-fast`` makes the import stop at the first bad feature instead of half-loading silently;
  - read-back verification (optional): ``citydb export cityjson`` and a check that every expected id is present.

What it does NOT own and does not claim: the coordinate reference system. The CityJSON written here is in the
world's own (local, metric) frame; which CRS the database interprets it in is the database's setting, and geo-
referencing the world is the caller's job. Real-database behaviour is verified only by the env-gated integration
test (tests/test_citydb_exporter.py::TestRealDatabase); everything else in that file runs against a fake CLI and
proves the invocation contract, not 3DCityDB itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

from exporters.cityjson import export_to_cityjson_with_report

STATUS_IMPORTED = "imported"
STATUS_PREVIEWED = "previewed"        # --preview: processed as if importing, database untouched
STATUS_UNAVAILABLE = "unavailable"    # no citydb tool: nothing was run
STATUS_NOTHING = "nothing_to_export"  # the world has no exportable geometry: no database call was made
STATUS_FAILED = "failed"

#: Environment variables citydb-tool reads for the connection (documented: docs.3dcitydb.org citydb-tool/database).
_ENV_KEYS = ("CITYDB_HOST", "CITYDB_PORT", "CITYDB_NAME", "CITYDB_SCHEMA", "CITYDB_USERNAME", "CITYDB_PASSWORD")


@dataclass(frozen=True)
class CityDbTarget:
    """Where to write. The password is deliberately NOT a field: it is read from ``password_env`` at call time."""

    host: str
    database: str
    username: str
    port: int = 5432
    schema: str = "citydb"
    password_env: str = "CITYDB_PASSWORD"

    def child_env(self, base: Mapping[str, str]) -> dict:
        env = dict(base)
        env.update(CITYDB_HOST=self.host, CITYDB_PORT=str(self.port), CITYDB_NAME=self.database,
                   CITYDB_SCHEMA=self.schema, CITYDB_USERNAME=self.username)
        password = base.get(self.password_env)
        if password is not None:
            env["CITYDB_PASSWORD"] = password
        else:
            env.pop("CITYDB_PASSWORD", None)
        return env


@dataclass(frozen=True)
class CityDbExportReport:
    status: str
    world_id: str
    world_version: int
    #: CityJSON object ids sent to the database (WorldIR entity ids)
    objects: Tuple[str, ...] = ()
    #: entities that could not be exported, with reasons (from the CityJSON exporter; never silently dropped)
    skipped: Tuple[str, ...] = ()
    skip_reasons: Tuple[str, ...] = ()
    #: the exact command that ran, password-free by construction
    command: Tuple[str, ...] = ()
    tool: Optional[str] = None
    tool_version: Optional[str] = None
    import_mode: str = "delete"
    #: None = verification not requested / not possible; True = every object read back; False = something missing
    verified: Optional[bool] = None
    missing_after_readback: Tuple[str, ...] = ()
    detail: str = ""
    cityjson_sha256: str = ""
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def find_citydb_tool(explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Path of the ``citydb`` executable, or None. Order: explicit argument, ``CITYDB_TOOL``, PATH."""
    env = os.environ if env is None else env
    for candidate in (explicit, env.get("CITYDB_TOOL")):
        if candidate:
            return candidate if Path(candidate).is_file() else None
    return shutil.which("citydb", path=env.get("PATH"))


Runner = Callable[..., "subprocess.CompletedProcess"]


def _tool_version(tool: str, env: Mapping[str, str], runner: Runner) -> Optional[str]:
    try:
        out = runner([tool, "--version"], capture_output=True, text=True, timeout=60, env=dict(env), shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr or "").strip()
    return text.splitlines()[0] if out.returncode == 0 and text else None


def _tail(text: Optional[str], n: int = 800) -> str:
    return (text or "").strip()[-n:]


def export_to_3dcitydb(
    world,
    target: CityDbTarget,
    *,
    tool: Optional[str] = None,
    preview: bool = False,
    verify: bool = False,
    timeout_s: float = 1800.0,
    env: Optional[Mapping[str, str]] = None,
    runner: Runner = subprocess.run,
) -> CityDbExportReport:
    """Load ``world`` into the 3DCityDB at ``target``. Never raises for environment problems: returns the report."""
    env = dict(os.environ if env is None else env)
    cityjson, cj_report = export_to_cityjson_with_report(world)
    base = dict(world_id=world.id, world_version=world.version, objects=tuple(cj_report.entities_exported),
                skipped=tuple(cj_report.entities_skipped), skip_reasons=tuple(cj_report.skip_reasons),
                cityjson_sha256=cj_report.content_hash)
    if not cj_report.entities_exported:
        return CityDbExportReport(status=STATUS_NOTHING, detail="no entity has exportable geometry; the database "
                                  "was not contacted", **{**base, "objects": ()})

    exe = find_citydb_tool(tool, env)
    if exe is None:
        return CityDbExportReport(
            status=STATUS_UNAVAILABLE,
            detail="the 3DCityDB `citydb` command-line tool was not found (pass tool=..., set CITYDB_TOOL, or put "
                   "`citydb` on PATH); nothing was run and no file is presented as a 3DCityDB export", **base)

    child_env = target.child_env(env)
    version = _tool_version(exe, child_env, runner)
    with tempfile.TemporaryDirectory(prefix="re_citydb_") as tmp:
        src = Path(tmp) / f"{world.id}.json"
        src.write_text(json.dumps(cityjson, sort_keys=True), encoding="utf-8")
        command = [
            exe, "import", "cityjson",
            "--import-mode=delete",          # idempotent: same objectid is replaced, not duplicated
            "--fail-fast",
            f"--lineage=reality-engine:{world.id}:v{world.version}",
            f"--reason-for-update=Reality Engine world {world.id} version {world.version}",
        ]
        if preview:
            command.append("--preview")
        command.append(str(src))
        try:
            proc = runner(command, capture_output=True, text=True, timeout=timeout_s, env=child_env, shell=False)
        except subprocess.TimeoutExpired:
            return CityDbExportReport(status=STATUS_FAILED, command=tuple(command), tool=exe, tool_version=version,
                                      detail=f"citydb import timed out after {timeout_s:g}s; the database state is "
                                             "whatever the tool committed -- re-running is safe (idempotent)", **base)
        except OSError as exc:
            return CityDbExportReport(status=STATUS_FAILED, command=tuple(command), tool=exe, tool_version=version,
                                      detail=f"could not start the citydb tool: {exc}", **base)
        if proc.returncode != 0:
            return CityDbExportReport(
                status=STATUS_FAILED, command=tuple(command), tool=exe, tool_version=version,
                detail=f"citydb import exited {proc.returncode}: {_tail(proc.stderr) or _tail(proc.stdout)}", **base)

        verified, missing, note = None, (), ()
        if verify and not preview:
            verified, missing, vnote = _read_back(exe, child_env, base["objects"], timeout_s, runner)
            note = (vnote,) if vnote else ()
        return CityDbExportReport(
            status=STATUS_PREVIEWED if preview else STATUS_IMPORTED, command=tuple(command), tool=exe,
            tool_version=version, verified=verified, missing_after_readback=tuple(missing), notes=note,
            detail=_tail(proc.stdout, 400), **base)


def _read_back(exe: str, child_env: Mapping[str, str], expected: Sequence[str], timeout_s: float, runner: Runner):
    """``citydb export cityjson`` and check every expected object id is present. Returns (verified, missing, note)."""
    with tempfile.TemporaryDirectory(prefix="re_citydb_rb_") as tmp:
        out_path = Path(tmp) / "readback.json"
        cmd = [exe, "export", "cityjson", "--no-json-lines", "-o", str(out_path)]
        try:
            proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s, env=dict(child_env), shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, (), f"read-back could not run: {exc}"
        if proc.returncode != 0 or not out_path.is_file():
            return None, (), f"read-back export exited {proc.returncode}: {_tail(proc.stderr)}"
        try:
            present = set(json.loads(out_path.read_text(encoding="utf-8")).get("CityObjects", {}))
        except (ValueError, OSError) as exc:
            return None, (), f"read-back file unreadable: {exc}"
    missing = tuple(i for i in expected if i not in present)
    return (not missing), missing, ""
