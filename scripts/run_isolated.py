"""Run a command in an isolated child process and say WHAT KIND of failure it was.

    python scripts/run_isolated.py --timeout 900 --report out.json -- python -m pytest tests/integration/test_release_gate.py

A native crash (COLMAP/OpenCV/CUDA abort, access violation, segfault) must never be reported as "an assertion failed", and
a hang must never leave a child holding the GPU. This wrapper:

* runs the command in its own process group / session so the WHOLE tree can be killed (``taskkill /T`` on Windows,
  ``killpg`` on POSIX) when the timeout expires -- the real process dies, not just a Python wrapper;
* classifies the outcome: ``passed`` | ``test_failure`` | ``usage_error`` | ``timeout`` | ``native_crash`` | ``killed``;
* records exit code (and the signal name / NTSTATUS name), duration, python/platform and the last 40 lines of output.

pytest exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal error, 4 usage error, 5 nothing collected. Anything else
(negative on POSIX = killed by signal; 0xC0000005-style NTSTATUS on Windows) is a crash of the process, not of a test.
Exit status of THIS script: 0 passed, 1 test_failure, 2 anything that is not a test verdict (timeout / crash / usage).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time

#: Windows NTSTATUS exit codes that mean the process itself crashed
_NTSTATUS = {
    0xC0000005: "STATUS_ACCESS_VIOLATION", 0xC0000409: "STATUS_STACK_BUFFER_OVERRUN (fast-fail / abort)",
    0xC00000FD: "STATUS_STACK_OVERFLOW", 0xC0000374: "STATUS_HEAP_CORRUPTION", 0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO", 0xC0000142: "STATUS_DLL_INIT_FAILED",
}
_VERDICTS = {2: "interrupted", 3: "internal_error", 4: "usage_error", 5: "no_tests"}


#: Printed by CPython's faulthandler ONLY when the interpreter dies of a fatal signal / abort / access violation. (The
#: "Windows fatal exception:" line is NOT used: faulthandler prints it even for access violations that ctypes then turns
#: into an ordinary OSError, which is a failing test, not a crash.)
_FATAL_MARKER = "Fatal Python error:"


def classify(returncode: int | None, timed_out: bool, output: str = "") -> tuple[str, str]:
    """(outcome, detail) for a finished child. Pure function: unit-tested for every branch.

    ``output`` matters because exit codes alone are ambiguous: with faulthandler enabled (pytest enables it) a REAL native
    crash on Windows exits with code 3, which is also pytest's "internal error" code.
    """
    if timed_out:
        return "timeout", "exceeded the wall-clock limit; the whole process tree was killed"
    assert returncode is not None
    if returncode not in (0, 1) and _FATAL_MARKER in output:
        line = next((ln for ln in output.splitlines() if _FATAL_MARKER in ln), _FATAL_MARKER)
        return "native_crash", f"{line.strip()} (exit {returncode}; faulthandler traceback follows in the output)"
    if returncode < 0:  # POSIX: killed by signal N
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return ("killed" if name in ("SIGKILL", "SIGTERM") else "native_crash"), name
    unsigned = returncode & 0xFFFFFFFF
    if unsigned in _NTSTATUS or unsigned >= 0xC0000000:
        return "native_crash", _NTSTATUS.get(unsigned, f"NTSTATUS 0x{unsigned:08X}")
    if returncode in (132, 134, 136, 139):  # shell-wrapped SIGILL / SIGABRT / SIGFPE / SIGSEGV
        return "native_crash", f"exit {returncode} (128 + signal {returncode - 128})"
    if returncode == 0:
        return "passed", ""
    if returncode == 1:
        return "test_failure", "pytest reported failing tests"
    return _VERDICTS.get(returncode, "nonzero_exit"), f"exit {returncode}"


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
        proc.kill()


def run(cmd: list[str], timeout: float, env: dict | None = None) -> dict:
    kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True, "errors": "replace",
                    "env": {**os.environ, "PYTHONFAULTHANDLER": "1", **(env or {})}}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    started = time.monotonic()
    proc = subprocess.Popen(cmd, **kwargs)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        out, _ = proc.communicate()
    outcome, detail = classify(proc.returncode, timed_out, out)
    return {
        "command": cmd, "outcome": outcome, "detail": detail, "returncode": proc.returncode,
        "returncode_hex": f"0x{proc.returncode & 0xFFFFFFFF:08X}" if proc.returncode is not None else None,
        "seconds": round(time.monotonic() - started, 2), "timeout_s": timeout,
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "faulthandler_native_trace": "Windows fatal exception" in out or "Fatal Python error" in out,
        "output_tail": out.splitlines()[-40:],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock seconds before the process TREE is killed")
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("give a command after --")
    res = run(cmd, args.timeout)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
    print(f"[run_isolated] outcome={res['outcome']} {res['detail']} rc={res['returncode']} ({res['returncode_hex']}) "
          f"in {res['seconds']}s")
    if res["outcome"] != "passed":
        print("\n".join(res["output_tail"]))
    return {"passed": 0, "test_failure": 1}.get(res["outcome"], 2)


if __name__ == "__main__":
    sys.exit(main())
