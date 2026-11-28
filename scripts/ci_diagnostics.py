"""Machine-readable CI diagnostics from pytest JUnit XML (one file per shard, or one for a whole run).

    python scripts/ci_diagnostics.py junit1.xml junit2.xml ... --out ci_diagnostics.json [--markdown report.md]

Reports, per input and overall: totals (passed/failed/errors/skipped), wall time, the slowest tests and test FILES, skip
reasons, and shard balance (max/min and the spread against the mean). Pure parsing: no network, no pytest import.
GitHub step summary is appended when GITHUB_STEP_SUMMARY is set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import xml.etree.ElementTree as ET  # noqa: S405 -- parses only JUnit files this repo's own pytest run just wrote


def parse(path: str) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    per_file: collections.Counter[str] = collections.Counter()
    tests: list[tuple[float, str]] = []
    for case in (c for s in suites for c in s.iter("testcase")):
        t = float(case.get("time") or 0.0)
        cls, name = case.get("classname") or "", case.get("name") or ""
        # classname is the dotted module path (tests.test_x.TestY); the file is the first `test_*` component
        parts = cls.split(".")
        fi = next((i for i, p in enumerate(parts) if p.startswith("test_")), None)
        file = "/".join(parts[: fi + 1]) + ".py" if fi is not None else (cls or "?")
        counts["total"] += 1
        per_file[file] += t
        tests.append((t, f"{file}::{name}"))
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif (sk := case.find("skipped")) is not None:
            counts["skipped"] += 1
            reasons[(sk.get("message") or sk.get("type") or "no reason")[:140]] += 1
        else:
            counts["passed"] += 1
    wall = sum(float(s.get("time") or 0.0) for s in suites)
    return {
        "source": os.path.basename(path), "counts": dict(counts), "wall_seconds": round(wall, 2),
        "sum_test_seconds": round(sum(per_file.values()), 2),
        "slowest_tests": [{"test": n, "seconds": round(t, 2)} for t, n in sorted(tests, reverse=True)[:25]],
        "slowest_files": [{"file": f, "seconds": round(t, 2)} for f, t in per_file.most_common(15)],
        "skip_reasons": [{"reason": r, "count": n} for r, n in reasons.most_common(12)],
    }


def combine(reports: list[dict]) -> dict:
    counts: collections.Counter[str] = collections.Counter()
    for r in reports:
        counts.update(r["counts"])
    walls = [r["wall_seconds"] for r in reports]
    mean = sum(walls) / len(walls) if walls else 0.0
    return {
        "inputs": reports, "totals": dict(counts), "total_wall_seconds": round(sum(walls), 2),
        "shard_balance": {
            "wall_seconds": walls, "max": max(walls, default=0), "min": min(walls, default=0),
            "max_over_min": round(max(walls) / min(walls), 2) if walls and min(walls) > 0 else None,
            "max_over_mean": round(max(walls) / mean, 2) if mean else None,
        },
    }


def markdown(d: dict) -> str:
    t, b = d["totals"], d["shard_balance"]
    lines = ["## CI runtime and reliability", "",
             f"tests {t.get('total', 0)} | passed {t.get('passed', 0)} | failed {t.get('failed', 0)} | errors {t.get('errors', 0)} "
             f"| skipped {t.get('skipped', 0)} | total wall {d['total_wall_seconds']}s",
             f"shard wall seconds {b['wall_seconds']} | max/min {b['max_over_min']} | max/mean {b['max_over_mean']}", ""]
    slow: list[tuple[float, str]] = [(x["seconds"], x["test"]) for r in d["inputs"] for x in r["slowest_tests"]]
    lines += ["### slowest tests", "", *[f"- {s}s `{n}`" for s, n in sorted(slow, reverse=True)[:15]], ""]
    reasons: collections.Counter[str] = collections.Counter()
    for r in d["inputs"]:
        for x in r["skip_reasons"]:
            reasons[x["reason"]] += x["count"]
    lines += ["### skip reasons", "", *[f"- x{n}: {r}" for r, n in reasons.most_common(10)], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("junit", nargs="+")
    ap.add_argument("--out", help="write the JSON diagnostics here")
    ap.add_argument("--markdown", help="write a markdown summary here")
    args = ap.parse_args(argv)
    d = combine([parse(p) for p in args.junit])
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, indent=1, sort_keys=True)
    md = markdown(d)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
    if (summary := os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
