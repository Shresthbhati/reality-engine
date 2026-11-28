"""The static-analysis ratchet must refuse new debt anywhere, let debt shrink, and refuse to raise its own baseline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ci_static_analysis_gate", ROOT / "scripts" / "ci_static_analysis_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

BASE = {"ruff_total": 10, "mypy_total": 4, "ruff_by_file": {"a.py": 6, "b.py": 4}, "mypy_by_file": {"a.py": 4}}


def cur(ruff_by_file, mypy_by_file):
    return {"ruff_total": sum(ruff_by_file.values()), "mypy_total": sum(mypy_by_file.values()),
            "ruff_by_file": ruff_by_file, "mypy_by_file": mypy_by_file}


def test_unchanged_and_reduced_debt_passes():
    assert gate.compare(BASE, cur({"a.py": 6, "b.py": 4}, {"a.py": 4})) == []
    assert gate.compare(BASE, cur({"a.py": 2, "b.py": 0}, {})) == []


def test_new_debt_in_a_new_file_fails_even_when_the_total_goes_down():
    """Fixing a.py must not pay for a brand-new violation in c.py: a file absent from the baseline has a budget of 0."""
    problems = gate.compare(BASE, cur({"a.py": 0, "b.py": 0, "c.py": 1}, {}))
    assert len(problems) == 1 and "c.py" in problems[0] and "budget 0" in problems[0]


def test_a_file_that_got_worse_fails_and_a_higher_total_fails():
    worse = gate.compare(BASE, cur({"a.py": 7, "b.py": 4}, {"a.py": 4}))
    assert any("Ruff violations increased" in p for p in worse) and any("a.py has 7" in p for p in worse)
    assert any("Mypy errors increased" in p for p in gate.compare(BASE, cur({"a.py": 6, "b.py": 4}, {"a.py": 5})))


def test_the_baseline_cannot_be_raised_without_an_explicit_flag():
    assert gate.raised_totals(BASE, cur({"a.py": 11}, {})) == ["ruff_total would rise 10 -> 11"]
    assert gate.raised_totals(BASE, cur({"a.py": 3}, {})) == []


def test_the_committed_baseline_is_per_file_and_consistent():
    base = json.loads((ROOT / "docs" / "static_analysis_baseline.json").read_text(encoding="utf-8"))
    assert sum(base["ruff_by_file"].values()) == base["ruff_total"] == sum(base["ruff_by_code"].values())
    assert sum(base["mypy_by_file"].values()) == base["mypy_total"]
