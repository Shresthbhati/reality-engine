"""Tests for the third-party license registry (.agent/LICENSES.yaml)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LICENSES = ROOT / ".agent" / "LICENSES.yaml"


def test_licenses_file_parses():
    assert LICENSES.is_file()
    data = yaml.safe_load(LICENSES.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "declared_runtime" in data
    assert "external_binaries" in data


def test_declared_runtime_has_required_fields():
    data = yaml.safe_load(LICENSES.read_text(encoding="utf-8"))
    for entry in data["declared_runtime"]:
        assert "package" in entry
        assert "declared_license" in entry
        assert "verify" in entry


def test_integration_paths_resolve():
    # Paths mentioned in integration/usage notes must point at real files.
    data = yaml.safe_load(LICENSES.read_text(encoding="utf-8"))
    paths = set()
    for entry in data.get("external_binaries", []):
        text = str(entry.get("integration", ""))
        for m in re.finditer(r"[\w./\\-]+\.py", text):
            paths.add(m.group(0))
    for entry in data.get("declared_runtime", []):
        text = str(entry.get("usage", ""))
        for m in re.finditer(r"[\w./\\-]+\.py", text):
            paths.add(m.group(0))
    for rel in paths:
        # normalize separators
        p = ROOT / rel.replace("\\", "/")
        assert p.is_file(), f"LICENSES.yaml references missing file: {rel}"