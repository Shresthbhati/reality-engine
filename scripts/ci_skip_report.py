"""Report a pytest JUnit XML honestly: passed, failed, ERRORED and SKIPPED are different things.

    python scripts/ci_skip_report.py junit.xml --max-skip-share 0.30

A skipped test did not verify anything. This prints the counts and the most common skip reasons (and writes them to the
GitHub step summary), and FAILS when no test ran at all or when the skipped share exceeds the budget -- so a missing
dependency or marker cannot turn most of a shard into a green run that proved little.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import xml.etree.ElementTree as ET  # noqa: S405 -- parses only the JUnit file this job's own pytest just wrote (trusted)


def summarize(path: str) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    for case in (c for s in suites for c in s.iter("testcase")):
        counts["total"] += 1
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif (skip := case.find("skipped")) is not None:
            counts["skipped"] += 1
            reasons[(skip.get("message") or skip.get("type") or "no reason given")[:140]] += 1
        else:
            counts["passed"] += 1
    return {"counts": counts, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("junit")
    ap.add_argument("--max-skip-share", type=float, default=0.30)
    args = ap.parse_args(argv)
    s = summarize(args.junit)
    c, total = s["counts"], s["counts"]["total"]
    lines = [f"tests: {total} | passed {c['passed']} | failed {c['failed']} | errors {c['errors']} | SKIPPED {c['skipped']}"]
    lines += [f"  skipped x{n}: {why}" for why, n in s["reasons"].most_common(8)]
    verdict = 0
    if total == 0:
        lines.append("FAIL: no test ran")
        verdict = 1
    elif c["skipped"] / total > args.max_skip_share:
        lines.append(f"FAIL: {c['skipped'] / total:.0%} of the tests were skipped (budget {args.max_skip_share:.0%}); "
                     "skipped tests verify nothing")
        verdict = 1
    text = "\n".join(lines)
    print(text)
    if (summary := os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(summary, "a", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
