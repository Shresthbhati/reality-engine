"""Render docs/STATUS.md from docs/status.yaml (T010).

``python scripts/render_status.py``          rewrite docs/STATUS.md
``python scripts/render_status.py --check``  exit 1 if the table in STATUS.md has drifted

The ``Baseline:`` line records the commit the file was rendered at; it is not part of the drift check
(otherwise every commit would make the file stale).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "status.yaml"
OUT = ROOT / "docs" / "STATUS.md"

STATUSES = ("MISSING", "FOUNDATION", "PARTIAL", "UNWIRED", "IMPL-UNVERIFIED",
            "CONTRACT-VERIFIED", "FIELD-VALIDATED", "DEFERRED")
GATES = ("G0", "G1", "G2", "G3", "G4")
FIELDS = ("id", "capability", "status", "gate", "evidence", "tests", "updated", "next")
NEEDS_NO_EVIDENCE = ("MISSING", "FOUNDATION")  # everything above FOUNDATION must cite evidence


def validate(items: list[dict]) -> list[str]:
    errors: list[str] = []
    for item in items:
        name = item.get("id", "?")
        for f in FIELDS:
            if f not in item:
                errors.append(f"{name}: missing field {f}")
        if item.get("status") not in STATUSES:
            errors.append(f"{name}: bad status {item.get('status')!r}")
        if item.get("gate") not in GATES:
            errors.append(f"{name}: bad gate {item.get('gate')!r}")
        evidence = str(item.get("evidence") or "").strip()
        if item.get("status") not in NEEDS_NO_EVIDENCE and not evidence:
            errors.append(f"{name}: status {item.get('status')} requires evidence")
        if item.get("status") == "FIELD-VALIDATED" and not evidence.startswith("benchmarks/"):
            errors.append(f"{name}: FIELD-VALIDATED evidence must start with benchmarks/")
    return errors


def _cell(value) -> str:
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def table(items: list[dict]) -> str:
    rows = ["| ID | Capability | Status | Gate | Evidence | Tests | Updated | Next |",
            "|---|---|---|---|---|---|---|---|"]
    rows += ["| " + " | ".join(_cell(i[f]) for f in FIELDS) + " |" for i in items]
    return "\n".join(rows) + "\n"


def _baseline() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    items = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    errors = validate(items)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    body = table(items)
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if not current.endswith(body):
            print("docs/STATUS.md is stale: run python scripts/render_status.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(
        "# Status\n\nGenerated from `docs/status.yaml` by `scripts/render_status.py`. Do not edit by hand.\n\n"
        f"Baseline: {_baseline()}\n\n{body}",
        encoding="utf-8", newline="\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
