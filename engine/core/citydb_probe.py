"""Probe for the 3DCityDB CLI tool.

Core-layer probe: reports whether the ``citydb`` command-line tool is
available on this machine. Never contacts a database; never raises.

The core layer owns this probe so that `engine.core.verification` can
report runtime availability without importing the `exporters` layer
(which would violate the import boundary: core must not import above it).
The real export implementation stays in `exporters.citydb`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class CityDbProbe:
    status: str  # AVAILABLE | UNAVAILABLE | FAILED
    tool: Optional[str] = None
    version: Optional[str] = None
    detail: str = ""


STATUS_AVAILABLE = "AVAILABLE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_FAILED = "FAILED"


def find_citydb_tool(tool: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Locate the citydb executable: explicit path, CITYDB_TOOL env, then PATH."""
    if tool:
        return tool if os.path.exists(tool) else None
    if env is None:
        env = os.environ
    from_env = env.get("CITYDB_TOOL")
    if from_env:
        return from_env if os.path.exists(from_env) else None
    return shutil.which("citydb")


def probe_citydb(
    tool: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    runner: Runner = subprocess.run,
    timeout_s: float = 30.0,
) -> CityDbProbe:
    """Probe only: finds the tool and asks it for its version.

    Never contacts a database and never raises.
    """
    env = dict(os.environ if env is None else env)
    exe = find_citydb_tool(tool, env)
    if exe is None:
        return CityDbProbe(
            STATUS_UNAVAILABLE,
            detail="the 3DCityDB `citydb` command-line tool was not found (pass tool=..., "
            "set CITYDB_TOOL, or put `citydb` on PATH)",
        )
    try:
        out = runner(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return CityDbProbe(
            STATUS_FAILED, tool=exe, detail=f"`citydb --version` did not finish within {timeout_s:g}s"
        )
    except OSError as exc:
        return CityDbProbe(STATUS_FAILED, tool=exe, detail=f"could not start the citydb tool: {exc}")
    text = (out.stdout or out.stderr or "").strip()
    if out.returncode != 0 or not text:
        return CityDbProbe(
            STATUS_FAILED,
            tool=exe,
            detail=f"`citydb --version` exited {out.returncode}: {text[-300:]}",
        )
    return CityDbProbe(
        STATUS_AVAILABLE,
        tool=exe,
        version=text.splitlines()[0],
        detail="the CLI runs; the database behind it was not contacted",
    )
