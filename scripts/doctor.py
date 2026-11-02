"""Environment doctor (blueprint task 0.2): ``python scripts/doctor.py [--json]``.

Exit 1 only when a required check FAILs (python, DB writable). Missing GPU / COLMAP / node are WARN.
COLMAP is located the way the project does it: ``shutil.which(os.environ.get("REALITY_COLMAP", "colmap"))``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 10)
SQLITE_PREFIX = "sqlite+aiosqlite:///"


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, shell=False)
        return ((out.stdout or "") + (out.stderr or "")).strip().splitlines()[0] if (out.stdout or out.stderr) else ""
    except Exception as exc:  # noqa: BLE001 - a doctor must report, not crash
        return f"error: {exc}"


def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return ("OK" if ok else "FAIL"), f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else f" (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")


def check_venv():
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    return ("OK" if active else "WARN"), (f"active ({sys.prefix})" if active else "no virtualenv active")


def check_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - missing or broken install
        return "WARN", f"not importable ({type(exc).__name__}: {exc})", None
    cuda = bool(torch.cuda.is_available())
    return ("OK" if cuda else "WARN"), f"{torch.__version__}, cuda available: {cuda}", torch


def check_gpu(torch):
    if torch is None or not torch.cuda.is_available():
        return "WARN", "none"
    p = torch.cuda.get_device_properties(0)
    return "OK", f"{p.name}, {p.total_memory / 2**30:.1f} GiB VRAM"


def check_colmap():
    binary = os.environ.get("REALITY_COLMAP", "colmap")
    path = shutil.which(binary)
    if path is None:
        return "WARN", f"{binary!r} not on PATH (set REALITY_COLMAP to its path)"
    first = _run([path, "help"])  # first line is "COLMAP 3.x -- Structure-from-Motion ..."
    return "OK", f"{path} ({first})"


def check_tool(name: str):
    path = shutil.which(name)
    if path is None:
        return "WARN", "not on PATH"
    return "OK", f"{_run([path, '--version'])} ({path})"


def check_db():
    try:
        from apps.api.db import database_url
        url = database_url()
    except Exception:  # noqa: BLE001 - sqlalchemy missing: same default as apps/api/db.py
        url = os.environ.get("DATABASE_URL", SQLITE_PREFIX + "./data/app.db")
    if not url.startswith(SQLITE_PREFIX):
        return "WARN", f"{url.split('@')[-1]} (non-sqlite, writability not checked)"
    raw = url[len(SQLITE_PREFIX):].split("?", 1)[0]
    if raw == ":memory:":
        return "OK", "in-memory"
    path = Path(raw).expanduser().resolve()
    probe = path.parent
    while not probe.exists() and probe != probe.parent:  # the API creates missing dirs; test the nearest existing one
        probe = probe.parent
    try:
        with tempfile.NamedTemporaryFile(dir=probe):
            pass
    except OSError as exc:
        return "FAIL", f"{path} not writable ({exc})"
    return "OK", f"{path} writable"


def run_checks() -> list[dict]:
    sys.path.insert(0, str(REPO))
    torch_status, torch_detail, torch = check_torch()
    rows = [
        ("python", *check_python()),
        ("venv", *check_venv()),
        ("torch", torch_status, torch_detail),
        ("gpu", *check_gpu(torch)),
        ("colmap", *check_colmap()),
        ("node", *check_tool("node")),
        ("npm", *check_tool("npm")),
        ("db", *check_db()),
    ]
    return [{"check": n, "status": s, "detail": d} for n, s, d in rows]


def main(argv: list[str] | None = None) -> int:
    rows = run_checks()
    if "--json" in (sys.argv[1:] if argv is None else argv):
        print(json.dumps({"ok": not any(r["status"] == "FAIL" for r in rows), "checks": rows}, indent=2))
    else:
        for r in rows:
            print(f"[{r['status']:<4}] {r['check']:<7} {r['detail']}")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
