"""B12 regression: the deterministic-matching flag must be one the installed COLMAP accepts.
COLMAP 4.x renamed SiftMatching.* to FeatureMatching.*; the old name made exhaustive_matcher exit 1 on every
capture of <= 60 photos."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "reconstruction" / "backend" / "colmap_backend.py"


def test_backend_emits_no_sift_matching_flags():
    assert "--SiftMatching." not in SRC.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("colmap") is None, reason="colmap binary not installed")
def test_emitted_matcher_flags_exist_in_installed_colmap():
    r = subprocess.run(["colmap", "exhaustive_matcher", "-h"], capture_output=True, text=True)
    help_text = r.stdout + r.stderr  # colmap prints help to stderr on some builds
    for flag in set(re.findall(r'"(--FeatureMatching\.[a-z_]+)"', SRC.read_text(encoding="utf-8"))):
        assert flag in help_text, f"{flag} not accepted by installed colmap"
