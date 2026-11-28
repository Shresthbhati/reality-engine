"""The CI tooling must itself be trustworthy: crash vs failure vs timeout classification, process-tree kill, JUnit
diagnostics, and the declared-requirements marker (skip by default, FAIL under RE_STRICT_REQUIREMENTS=1)."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iso = _load("run_isolated")
diag = _load("ci_diagnostics")


@pytest.mark.parametrize("rc,timed_out,outcome", [
    (0, False, "passed"), (1, False, "test_failure"), (2, False, "interrupted"), (4, False, "usage_error"),
    (5, False, "no_tests"), (0xC0000005, False, "native_crash"), (0xC0000409, False, "native_crash"),
    (-int(signal.SIGTERM), False, "killed"), (139, False, "native_crash"), (7, False, "nonzero_exit"), (0, True, "timeout"),
])
def test_classify_distinguishes_a_crash_from_a_failing_test(rc, timed_out, outcome):
    assert iso.classify(rc, timed_out)[0] == outcome


@pytest.mark.skipif(os.name == "nt", reason="negative return codes (death by signal) exist on POSIX only")
def test_classify_names_posix_crash_signals():
    assert iso.classify(-int(signal.SIGSEGV), False) == ("native_crash", "SIGSEGV")


def test_a_real_native_style_crash_is_reported_as_a_crash_not_a_failure():
    code = "import faulthandler; faulthandler._sigsegv()"   # a REAL fatal signal / access violation: the interpreter dies
    res = iso.run([sys.executable, "-c", code], timeout=60)
    assert res["outcome"] == "native_crash", res
    assert iso.main(["--timeout", "60", "--", sys.executable, "-c", code]) == 2   # not a test verdict


def test_on_windows_a_real_crash_exits_3_which_is_pytests_internal_error_code_so_the_marker_decides():
    """Measured: faulthandler makes a real crash exit with 3 on Windows. Exit code alone would call it an internal error."""
    assert iso.classify(3, False, "")[0] == "internal_error"
    assert iso.classify(3, False, "Fatal Python error: Segmentation fault\n")[0] == "native_crash"
    # a Python-level exception that merely contains the words is not a crash (exit 1 is a failing test)
    assert iso.classify(1, False, "Fatal Python error: in a log line")[0] == "test_failure"


@pytest.mark.skipif(os.name != "nt", reason="ctypes turns an access violation into OSError only on Windows (POSIX: real SIGSEGV)")
def test_an_access_violation_that_ctypes_converts_to_an_exception_is_a_failure_not_a_crash():
    res = iso.run([sys.executable, "-c", "import ctypes; ctypes.string_at(0)"], timeout=60)
    assert res["outcome"] == "test_failure", res


def test_a_failing_test_run_is_a_test_failure_and_a_clean_run_passes():
    assert iso.run([sys.executable, "-c", "raise SystemExit(1)"], 60)["outcome"] == "test_failure"
    assert iso.run([sys.executable, "-c", "pass"], 60)["outcome"] == "passed"


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_timeout_kills_the_whole_process_tree(tmp_path):
    """A hung child AND its own grandchild must both die: the timeout kills the real process tree, not just the wrapper."""
    pidfile = tmp_path / "grandchild.pid"
    script = textwrap.dedent(f"""
        import subprocess, sys, time
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pidfile)!r}, "w").write(str(g.pid))
        time.sleep(120)
    """)
    res = iso.run([sys.executable, "-c", script], timeout=4)
    assert res["outcome"] == "timeout" and res["seconds"] < 60, res
    pid = int(pidfile.read_text())
    deadline = time.time() + 15
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    assert not _alive(pid), "the grandchild survived the timeout: it would keep holding the GPU / temp space"


def test_diagnostics_summarise_junit_with_slowest_tests_skip_reasons_and_shard_balance(tmp_path):
    def junit(path, cases):
        path.write_text(f"<testsuites><testsuite name='pytest' time='{len(cases)}'>{''.join(cases)}</testsuite></testsuites>")

    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    junit(a, ["<testcase classname='tests.test_x.TestY' name='t1' time='3.5'/>",
              "<testcase classname='tests.test_x' name='t2' time='0.1'><skipped message='requires: torch'/></testcase>"])
    junit(b, ["<testcase classname='tests.integration.test_z' name='t3' time='1.0'><failure message='x'/></testcase>"])
    out = tmp_path / "d.json"
    assert diag.main([str(a), str(b), "--out", str(out), "--markdown", str(tmp_path / "d.md")]) == 0
    d = json.loads(out.read_text())
    assert d["totals"] == {"total": 3, "passed": 1, "skipped": 1, "failed": 1}
    assert d["inputs"][0]["slowest_tests"][0] == {"test": "tests/test_x.py::t1", "seconds": 3.5}
    assert d["inputs"][1]["slowest_files"][0]["file"] == "tests/integration/test_z.py"
    assert d["inputs"][0]["skip_reasons"] == [{"reason": "requires: torch", "count": 1}]
    assert d["shard_balance"]["max_over_min"] == 2.0   # walls 2 and 1


# ----------------------------------------------------------------------------- the declared-requirements marker


class _Item:
    def __init__(self, *caps):
        self._caps = caps

    def iter_markers(self, name):
        class M:
            args = self._caps
        return [M()] if name == "requires" and self._caps else []


def _root_conftest():
    spec = importlib.util.spec_from_file_location("root_conftest_under_test", ROOT / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_missing_requirement_skips_by_default_and_names_what_is_missing(monkeypatch):
    monkeypatch.delenv("RE_STRICT_REQUIREMENTS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="not_a_real_module_xyz"):
        _root_conftest().pytest_runtest_setup(_Item("not_a_real_module_xyz"))


def test_a_missing_requirement_FAILS_in_the_strict_lane(monkeypatch):
    monkeypatch.setenv("RE_STRICT_REQUIREMENTS", "1")
    with pytest.raises(pytest.fail.Exception, match="not a skip"):
        _root_conftest().pytest_runtest_setup(_Item("not_a_real_module_xyz"))


def test_a_present_requirement_neither_skips_nor_fails(monkeypatch):
    monkeypatch.setenv("RE_STRICT_REQUIREMENTS", "1")
    _root_conftest().pytest_runtest_setup(_Item("json", "numpy"))   # present everywhere; returns normally


def test_colmap_and_dataset_probes_do_not_pretend(monkeypatch):
    from tests import _capabilities as caps

    monkeypatch.setenv("RE_COLMAP_BIN", "colmap-that-does-not-exist-xyz")
    assert caps.probe("colmap")[0] is False
    assert caps.probe("dataset:definitely_not_a_dataset")[0] is False
    assert caps.probe("hf_cache:nobody/nothing")[0] is False
