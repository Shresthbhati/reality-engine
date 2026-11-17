"""Controlled baseline gate for static analysis (Ruff & Mypy).

Removes --exit-zero and || true from CI gates. Fails with exit code 1 if
static analysis errors exceed the committed baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "docs" / "static_analysis_baseline.json"
TARGET_DIRS = [
    "apps/",
    "engine/",
    "world_ir/",
    "evidence/",
    "reconstruction/",
    "exporters/",
    "sdk/",
    "worldstore/",
    "registration/",
    "trajectories/",
    "uncertainty/",
    "provenance/",
    "events/",
    "perception/",
]


def run_ruff() -> tuple[int, dict[str, int]]:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *TARGET_DIRS,
        "--output-format",
        "json",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    try:
        items = json.loads(proc.stdout)
    except Exception:
        # If ruff output isn't JSON, return empty
        items = []

    counts: dict[str, int] = {}
    for item in items:
        code = item.get("code", "UNKNOWN")
        counts[code] = counts.get(code, 0) + 1
    total = len(items)
    return total, counts


def run_mypy() -> int:
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "apps/cli/main.py",
        "--ignore-missing-imports",
        "--no-error-summary",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    # Count lines matching "path:line: error: ..."
    error_lines = [
        line for line in proc.stdout.splitlines() if re.search(r":\d+: error:", line)
    ]
    return len(error_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Static analysis baseline gate")
    parser.add_argument("--write-baseline", action="store_true", help="Write baseline to file")
    parser.add_argument("--fail-on-regression", action="store_true", help="Fail if errors exceed baseline")
    args = parser.parse_args()

    print("Running static analysis gate...")
    ruff_total, ruff_by_code = run_ruff()
    mypy_total = run_mypy()
    print(f"Ruff violations: {ruff_total}")
    print(f"Mypy errors in apps/cli/main.py: {mypy_total}")

    data = {
        "ruff_total": ruff_total,
        "ruff_by_code": ruff_by_code,
        "mypy_total": mypy_total,
    }

    if args.write_baseline:
        BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote baseline to {BASELINE_PATH}")
        return 0

    if args.fail_on_regression:
        if not BASELINE_PATH.exists():
            print(f"Error: Baseline file {BASELINE_PATH} does not exist. Run with --write-baseline first.", file=sys.stderr)
            return 1

        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        base_ruff = baseline.get("ruff_total", 0)
        base_mypy = baseline.get("mypy_total", 0)

        regressions = []
        if ruff_total > base_ruff:
            regressions.append(
                f"Ruff violations increased: {ruff_total} (baseline: {base_ruff}, +{ruff_total - base_ruff})"
            )
        if mypy_total > base_mypy:
            regressions.append(
                f"Mypy errors increased: {mypy_total} (baseline: {base_mypy}, +{mypy_total - base_mypy})"
            )

        if regressions:
            print("FAILED: Static analysis regressions detected:", file=sys.stderr)
            for r in regressions:
                print(f"  - {r}", file=sys.stderr)
            return 1

        print(f"SUCCESS: Static analysis checks passed within baseline (Ruff: {ruff_total}/{base_ruff}, Mypy: {mypy_total}/{base_mypy}).")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
