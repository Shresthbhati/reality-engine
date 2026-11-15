"""Reality Engine benchmark CLI.

  python scripts/run_reality_benchmark.py                       # the 'awaiting execution' dashboard (nothing measured)
  python scripts/run_reality_benchmark.py --results DIR         # score scenario result files (*.json) from DIR
  python scripts/run_reality_benchmark.py --emit-manifests DIR  # write the 24 empty scenario manifests to DIR
  add --json for machine-readable output

A result file is {"scenario_id": "001", "ground_truth_available": true, "metric_results": [...]} with metric results
in the schema shape (benchmarks/reality_benchmark/result.schema.json). Exit code: 1 on any FAIL, else 0."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.reality_benchmark.manifests import manifest_template        # noqa: E402
from benchmarks.reality_benchmark.registry import SCENARIOS                  # noqa: E402
from benchmarks.reality_benchmark.scorer import aggregate, blank_results, score_scenario   # noqa: E402


def _scorecards(results_dir):
    cards = []
    if results_dir is None:
        for s in SCENARIOS:
            cards.append(score_scenario(s.id, blank_results(s, ground_truth_available=False), ground_truth_available=False))
        return cards
    for f in sorted(pathlib.Path(results_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cards.append(score_scenario(d["scenario_id"], d["metric_results"], bool(d.get("ground_truth_available", False))))
    return cards


def _git(*args) -> str:
    import subprocess

    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never a reason to lose the report
        return ""


def build_report(agg: dict, cards: list) -> dict:
    """The machine-readable release report: what was measured, on which commit, when. Nothing is added that was not scored."""
    import time

    return {"report_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "engine_commit": _git("rev-parse", "HEAD") or None, "working_tree_dirty": bool(_git("status", "--porcelain")),
            "any_metric_measured": any(c["measured"] for c in agg["categories"].values()),
            "aggregate": agg, "scorecards": cards}


def history_line(report: dict) -> dict:
    """One compact, append-only line per release: enough to see accuracy move between commits."""
    agg = report["aggregate"]
    return {"generated_at": report["generated_at"], "engine_commit": report["engine_commit"], "dirty": report["working_tree_dirty"],
            "verdict": agg["verdict"], "scenarios": agg["scenarios"], "gates": agg["gates"],
            "category_scores": {cid: {"score": c["score"], "coverage": c["coverage"]} for cid, c in agg["categories"].items()}}


def _fmt(v):
    return "not measured" if v is None else f"{v:.0%}"


def _print_dashboard(agg, cards):
    print(f"Reality Engine benchmark - {agg['scenarios']} scenario(s) - overall: {agg['verdict']}"
          f"  {agg['scenarios_by_verdict']}")
    print("\nCritical gates (independent of the weighted score)")
    for gid, st in agg["gates"].items():
        print(f"  {gid:34s} {st}")
    print("\nCategory                                  weight  measured/metrics  score")
    for cid, c in agg["categories"].items():
        print(f"  {cid} {c['name']:38s} {c['weight']:5.1f}  {c['measured']:>5}/{c['metrics']:<6}       {_fmt(c['score'])}")
    if not any(c["measured"] for c in agg["categories"].values()):
        print("\nAwaiting execution: no metric has been measured. Nothing above is a result.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="directory of scenario result files")
    ap.add_argument("--emit-manifests", metavar="DIR", help="write empty manifests for all scenarios to DIR")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--report", metavar="FILE", help="write the machine-readable release report (JSON) to FILE")
    ap.add_argument("--history", metavar="FILE", help="append one compact line for this run to FILE (JSONL accuracy history)")
    args = ap.parse_args(argv)
    if args.emit_manifests:
        out = pathlib.Path(args.emit_manifests)
        out.mkdir(parents=True, exist_ok=True)
        for s in SCENARIOS:
            (out / f"{s.id}_{s.slug}.json").write_text(json.dumps(manifest_template(s), indent=2), encoding="utf-8")
        print(f"wrote {len(SCENARIOS)} manifests to {out}")
        return 0
    cards = _scorecards(args.results)
    agg = aggregate(cards)
    if args.report or args.history:
        report = build_report(agg, cards)
        if args.report:
            p = pathlib.Path(args.report)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=1), encoding="utf-8")
        if args.history:
            h = pathlib.Path(args.history)
            h.parent.mkdir(parents=True, exist_ok=True)
            with h.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(history_line(report)) + "\n")
    if args.json:
        print(json.dumps({"aggregate": agg, "scorecards": cards}, indent=2))
    else:
        _print_dashboard(agg, cards)
    return 1 if agg["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
