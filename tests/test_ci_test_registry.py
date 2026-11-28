import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ci_test_registry import build_registry  # noqa: E402

reg = build_registry(ROOT / ".github/workflows/ci.yml", ROOT / "tests")


def test_core_contains_golden_path():
    assert "tests/test_golden_path_api_to_glb.py" in reg["core"]


def test_unlisted_contains_gaussian_splats():
    assert "tests/test_gaussian_splats.py" in reg["unlisted"]


def test_no_overlap():
    assert set(reg["core"]) & set(reg["unlisted"]) == set()


def test_doc_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/ci_test_registry.py"), "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_registry_has_exactly_the_files_on_disk_and_every_entry_is_complete():
    on_disk = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("test_*.py"))
    assert [e["file"] for e in reg["entries"]] == on_disk
    for e in reg["entries"]:
        assert e["owner"] != "unowned" and e["category"] and 1 <= e["shard"] <= 8 and e["lanes"], e["file"]


def test_registry_shards_equal_what_ci_runs():
    """The registry's shard column must equal the manifest CI actually executes (`ci_shard_tests.py --shard N`)."""
    for shard in range(1, 9):
        out = subprocess.run([sys.executable, str(ROOT / "scripts/ci_shard_tests.py"), "--shard", str(shard), "--of", "8"],
                             cwd=ROOT, capture_output=True)
        assert b"\r" not in out.stdout, "shard manifest must be LF-only on every OS"
        listed = out.stdout.decode().split("\n")[:-1]
        assert listed == [e["file"] for e in reg["entries"] if e["shard"] == shard], f"shard {shard} differs from the registry"


def test_registry_is_identical_across_hash_seeds():
    """The area/owner columns must not depend on set-iteration order (PYTHONHASHSEED)."""
    import os

    results = []
    for seed in ("1", "2"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/ci_test_registry.py"), "--check"], cwd=ROOT,
                           capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed})
        results.append((r.returncode, r.stderr))
    assert results == [(0, ""), (0, "")], results


def test_fixtures_that_start_with_test_are_not_counted_as_tests():
    """tests/test_security_audit.py defines a ``test_client`` FIXTURE; pytest does not collect it, so the registry must not count it."""
    import ast

    tree = ast.parse((ROOT / "tests/test_security_audit.py").read_text(encoding="utf-8"))
    def is_test(n):
        return (isinstance(n, ast.FunctionDef) and n.name.startswith("test")
                and not any("fixture" in ast.unparse(d) for d in n.decorator_list))

    real = sum(is_test(n) for n in tree.body) + sum(
        is_test(m) for c in tree.body if isinstance(c, ast.ClassDef) and c.name.startswith("Test") for m in c.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "test_client" for n in tree.body)   # the fixture is really there
    entry = next(e for e in reg["entries"] if e["file"] == "tests/test_security_audit.py")
    assert entry["tests"] == real
