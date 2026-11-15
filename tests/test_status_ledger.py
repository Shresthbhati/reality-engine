"""Status ledger (T010): docs/status.yaml is the single source; docs/STATUS.md is generated."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_status  # noqa: E402

ITEMS = yaml.safe_load((ROOT / "docs" / "status.yaml").read_text(encoding="utf-8"))


def test_enums_valid():
    assert ITEMS, "ledger is empty"
    assert render_status.validate(ITEMS) == []
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids))


def test_field_validated_requires_dataset():
    for item in ITEMS:
        if item["status"] == "FIELD-VALIDATED":
            assert str(item["evidence"]).startswith("benchmarks/"), item["id"]
    bad = [{"id": "x", "capability": "c", "status": "FIELD-VALIDATED", "gate": "G2",
            "evidence": "tests/foo.py", "tests": [], "updated": "2026-11-10", "next": "n"}]
    assert render_status.validate(bad), "validator must reject FIELD-VALIDATED without benchmarks/ evidence"


def test_doc_is_current():
    assert render_status.main(["--check"]) == 0
