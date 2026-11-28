"""Static-analysis RATCHET (Ruff + Mypy): no new debt anywhere, existing debt can only go down.

    python scripts/ci_static_analysis_gate.py --fail-on-regression   # CI: fail if the total OR any single file got worse
    python scripts/ci_static_analysis_gate.py --write-baseline       # record a LOWER baseline (refuses to raise it)
    python scripts/ci_static_analysis_gate.py --write-baseline --allow-increase   # explicit, reviewed acceptance only

The baseline (docs/static_analysis_baseline.json) stores totals, Ruff counts by rule, and Ruff / Mypy counts PER FILE.
A change fails if the total grew, or if any file has more violations than its baseline (a file absent from the baseline
has a budget of 0, so new code starts clean). Fixing debt in one file never pays for new debt in another. There are no
directory-wide suppressions: nothing is excluded, only counted.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "docs" / "static_analysis_baseline.json"
TARGET_DIRS = [
    "apps/", "engine/", "world_ir/", "evidence/", "reconstruction/", "exporters/", "sdk/", "worldstore/",
    "registration/", "trajectories/", "uncertainty/", "provenance/", "events/", "perception/",
]
MYPY_ENTRYPOINT = "apps/cli/main.py"  # mypy follows imports from here, so the count covers everything the CLI reaches
_MYPY_LINE = re.compile(r"^(?P<file>[^:\n]+):\d+(?::\d+)?: error:")


def _norm(path: str) -> str:
    p = Path(path)
    try:
        p = p.resolve().relative_to(ROOT)
    except ValueError:
        pass
    return p.as_posix()


def run_ruff() -> tuple[int, dict[str, int], dict[str, int]]:
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", *TARGET_DIRS, "--output-format", "json"],
                          cwd=str(ROOT), capture_output=True, text=True)
    try:
        items = json.loads(proc.stdout)
    except Exception:
        items = []
    by_code: collections.Counter[str] = collections.Counter(i.get("code", "UNKNOWN") for i in items)
    by_file: collections.Counter[str] = collections.Counter(_norm(i["filename"]) for i in items if "filename" in i)
    return len(items), dict(by_code), dict(by_file)


def run_mypy() -> tuple[int, dict[str, int]]:
    proc = subprocess.run([sys.executable, "-m", "mypy", MYPY_ENTRYPOINT, "--ignore-missing-imports", "--no-error-summary"],
                          cwd=str(ROOT), capture_output=True, text=True)
    by_file: collections.Counter[str] = collections.Counter()
    for line in proc.stdout.splitlines():
        m = _MYPY_LINE.match(line)
        if m:
            by_file[_norm(m.group("file"))] += 1
    return sum(by_file.values()), dict(by_file)


def measure() -> dict:
    ruff_total, ruff_by_code, ruff_by_file = run_ruff()
    mypy_total, mypy_by_file = run_mypy()
    return {"ruff_total": ruff_total, "ruff_by_code": ruff_by_code, "ruff_by_file": ruff_by_file,
            "mypy_total": mypy_total, "mypy_by_file": mypy_by_file}


def compare(baseline: dict, current: dict) -> list[str]:
    """Human-readable regressions (empty list = ratchet holds)."""
    problems: list[str] = []
    for key, label in (("ruff_total", "Ruff violations"), ("mypy_total", "Mypy errors")):
        if current[key] > baseline.get(key, 0):
            problems.append(f"{label} increased: {current[key]} (baseline {baseline.get(key, 0)}, +{current[key] - baseline.get(key, 0)})")
    for key, label in (("ruff_by_file", "Ruff"), ("mypy_by_file", "Mypy")):
        base = baseline.get(key)
        if base is None:  # legacy baseline without per-file data: totals-only until it is rewritten
            continue
        worse = sorted(((n - base.get(f, 0), f) for f, n in current[key].items() if n > base.get(f, 0)), reverse=True)
        problems += [f"{label}: {f} has {current[key][f]} (budget {base.get(f, 0)}, +{d})" for d, f in worse[:25]]
        if len(worse) > 25:
            problems.append(f"{label}: ...and {len(worse) - 25} more files over budget")
    return problems


def raised_totals(old: dict, new: dict) -> list[str]:
    return [f"{k} would rise {old.get(k, 0)} -> {new[k]}" for k in ("ruff_total", "mypy_total") if new[k] > old.get(k, 0)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write-baseline", action="store_true", help="record the current counts as the baseline (lower only)")
    ap.add_argument("--allow-increase", action="store_true", help="with --write-baseline: accept a HIGHER baseline (review!)")
    ap.add_argument("--fail-on-regression", action="store_true", help="fail if the total or any file is worse than baseline")
    args = ap.parse_args(argv)

    print("Running static analysis ratchet...")
    cur = measure()
    print(f"Ruff violations: {cur['ruff_total']} in {len(cur['ruff_by_file'])} files")
    print(f"Mypy errors (via {MYPY_ENTRYPOINT}): {cur['mypy_total']} in {len(cur['mypy_by_file'])} files")
    old = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None

    if args.write_baseline:
        if old and not args.allow_increase and (raised := raised_totals(old, cur)):
            print("REFUSED: the baseline may only go down (" + "; ".join(raised) + "). Fix the debt, or pass --allow-increase "
                  "after review.", file=sys.stderr)
            return 1
        BASELINE_PATH.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote baseline to {BASELINE_PATH}")
        return 0

    if args.fail_on_regression:
        if old is None:
            print(f"Error: Baseline file {BASELINE_PATH} does not exist. Run with --write-baseline first.", file=sys.stderr)
            return 1
        problems = compare(old, cur)
        if problems:
            print("FAILED: Static analysis regressions detected:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"SUCCESS: ratchet holds (Ruff {cur['ruff_total']}/{old.get('ruff_total')}, Mypy {cur['mypy_total']}/{old.get('mypy_total')}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
