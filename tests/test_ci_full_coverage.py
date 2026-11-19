"""CI must execute EVERY test file, not a hand-kept allowlist, and must not let skips pass as passes.

Guards against the failure found in the cycle-3 audit: 619 test files existed and ci.yml named 93 of them, so 526 never
ran in CI (including the three named in the audit). These tests fail if a test file can fall out of CI again.
"""
from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARDS = 8


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def in_repo(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_the_shards_partition_every_test_file_exactly_once(in_repo):
    sh = _load("ci_shard_tests")
    files = sh.all_test_files()
    parts = sh.shards(files, SHARDS)
    flat = [f for p in parts for f in p]
    assert sorted(flat) == files and len(flat) == len(set(flat))
    assert len(files) > 400, "test discovery found suspiciously few files"
    assert all(parts), "an empty shard wastes a runner and hides a discovery problem"


def test_shard_sizes_are_balanced(in_repo):
    sh = _load("ci_shard_tests")
    loads = [sum(Path(f).stat().st_size for f in p) for p in sh.shards(sh.all_test_files(), SHARDS)]
    assert max(loads) < 1.5 * min(loads), f"shards are badly unbalanced: {loads}"


@pytest.mark.parametrize("name", ["tests/test_real_specialist_executors.py", "tests/test_ifc_roundtrip.py",
                                  "tests/test_robust_depth_fusion.py", "tests/test_room_mesh.py"])
def test_files_that_were_missing_from_ci_are_now_in_a_shard(in_repo, name):
    sh = _load("ci_shard_tests")
    assert name in [f for p in sh.shards(sh.all_test_files(), SHARDS) for f in p]


def test_the_workflow_runs_every_shard_and_the_skip_gate():
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    full = wf["jobs"]["full"]
    assert full["strategy"]["matrix"]["shard"] == list(range(1, SHARDS + 1)), "the workflow must run every shard"
    assert full["strategy"].get("fail-fast") is False, "one failing shard must not hide the others"
    steps = "\n".join(str(s.get("run", "")) for s in full["steps"])
    assert f"--of {SHARDS}" in steps and "ci_shard_tests.py" in steps
    assert "ci_skip_report.py" in steps, "skips must be reported and budgeted, not silently counted as passes"
    assert "|| true" not in steps and "--exit-zero" not in steps
    assert "not slow" in steps, "slow tests run in the scheduled job; everything else runs per shard"


def test_the_skip_report_fails_a_mostly_skipped_run_and_an_empty_run(tmp_path):
    rep = _load("ci_skip_report")

    def xml(passed: int, skipped: int, failed: int = 0) -> str:
        cases = "".join(f"<testcase classname='t' name='p{i}'/>" for i in range(passed))
        cases += "".join(f"<testcase classname='t' name='s{i}'><skipped message='needs torch'/></testcase>" for i in range(skipped))
        cases += "".join(f"<testcase classname='t' name='f{i}'><failure message='x'/></testcase>" for i in range(failed))
        return textwrap.dedent(f"<testsuites><testsuite name='pytest'>{cases}</testsuite></testsuites>")

    ok, mostly_skipped, empty = tmp_path / "ok.xml", tmp_path / "skipped.xml", tmp_path / "empty.xml"
    ok.write_text(xml(9, 1))
    mostly_skipped.write_text(xml(2, 8))
    empty.write_text(xml(0, 0))
    assert rep.main([str(ok), "--max-skip-share", "0.30"]) == 0
    assert rep.main([str(mostly_skipped), "--max-skip-share", "0.30"]) == 1
    assert rep.main([str(empty)]) == 1
    s = rep.summarize(str(mostly_skipped))
    assert s["counts"]["skipped"] == 8 and s["reasons"]["needs torch"] == 8
