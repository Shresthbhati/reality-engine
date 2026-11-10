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
    r = subprocess.run([sys.executable, str(ROOT / "scripts/ci_test_registry.py"), "--check"], cwd=ROOT)
    assert r.returncode == 0
