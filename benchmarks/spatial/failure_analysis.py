"""Classify every association failure of the compiler against ground truth (no ground truth is altered).

For each (scene, seed) it matches compiled entities to the true ones and reports:
  over_split      one true wall explained by >1 compiled walls (with each fragment's views, length and why the
                  gate rejected it: recomputed with the production gate against the main cluster)
  merged          one compiled wall holding observations of >1 true wall
  missed          a true wall with no compiled counterpart
  false_openings  compiled openings beyond the true count, or whose host differs from the true host
Run:  python -m benchmarks.spatial.failure_analysis [--seeds 20] [--out path]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.spatial.run_spatial_benchmark import ROOT
from benchmarks.spatial.synth import SimConfig, all_scenes, simulate_observations
from reconstruction.spatial.association import AssociationConfig, _gate, fit_line
from reconstruction.spatial.compiler import compile_spatial
from reconstruction.spatial.model import Kind


def analyse(seeds: int, cfg: AssociationConfig = AssociationConfig()) -> dict:
    rows, tally = [], Counter()
    for sc in all_scenes():
        for seed in range(seeds):
            obs = simulate_observations(sc, SimConfig(seed=seed))
            by_ev = {o.evidence_id: o for o in obs}
            res = compile_spatial(obs, storeys=sc.storeys, assoc_cfg=cfg)
            walls = [e for e in res.association.entities if e.kind is Kind.WALL]
            truth_of = lambda ev: ev.split("#")[1]  # noqa: E731
            groups = defaultdict(list)           # true wall id -> compiled entities touching it
            merged = []
            for e in walls:
                ids = {truth_of(x) for x in e.evidence_ids}
                if len(ids) > 1:
                    merged.append({"entity": e.entity_id, "true_walls": sorted(ids)})
                for t in ids:
                    groups[t].append(e)
            split = []
            for t, ents in groups.items():
                if len(ents) > 1:
                    ents = sorted(ents, key=lambda e: -len(e.evidence_ids))
                    main = [by_ev[x] for x in ents[0].evidence_ids]
                    for frag in ents[1:]:
                        why = []
                        for x in frag.evidence_ids:
                            o = by_ev[x]
                            g = _gate(o, fit_line(main), main, cfg)
                            why.append(g.reason + f" (z_off {g.z_off:.1f}, z_ang {g.z_ang:.1f}, sigma {o.sigma_m:.3f}, len {o.length:.2f})")
                        split.append({"true_wall": t, "fragment_views": len(frag.image_ids), "reasons": why})
            missed = [w.id for w in sc.walls if w.id not in groups]
            ops = [e for e in res.association.entities if e.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING)]
            bad_ops = []
            true_ops = {o.id for o in sc.openings}
            for e in ops:
                tids = {truth_of(x) for x in e.evidence_ids}
                if len(ops) > len(true_ops) and len(e.image_ids) < 2:
                    bad_ops.append({"entity": e.entity_id, "kind": "single_view_fragment", "true": sorted(tids), "host": e.host_id})
                elif len(tids) > 1:
                    bad_ops.append({"entity": e.entity_id, "kind": "merged_openings", "true": sorted(tids)})
            fails = len(split) + len(merged) + len(bad_ops)
            tally.update(over_split=len(split), merged=len(merged), false_openings=len(bad_ops), missed_walls=len(missed))
            rows.append({"scene": sc.name, "seed": seed, "over_split": split, "merged": merged, "false_openings": bad_ops,
                         "missed_walls": missed, "failures": fails})
    n_fail = sum(1 for r in rows if r["failures"])
    return {"runs": len(rows), "runs_with_association_failure": n_fail, "totals": dict(tally), "rows": [r for r in rows if r["failures"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "spatial_failure_analysis.json"))
    a = ap.parse_args(argv)
    r = analyse(a.seeds)
    Path(a.out).write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    print(r["runs"], "runs;", r["runs_with_association_failure"], "with a failure;", r["totals"])
    for row in r["rows"][:40]:
        for s in row["over_split"]:
            print(row["scene"], row["seed"], "SPLIT", s["true_wall"], s["fragment_views"], "view(s):", s["reasons"][0][:110])
        for m in row["merged"]:
            print(row["scene"], row["seed"], "MERGE", m)
        for o in row["false_openings"]:
            print(row["scene"], row["seed"], "OPENING", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
