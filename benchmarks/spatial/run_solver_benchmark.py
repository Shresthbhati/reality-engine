"""Dense vs sparse multi-view optimization, with accuracy parity checked on the SAME inputs.

For each view count it runs both solvers on identical simulated inputs (evaluation seeds >= 100) and reports: wall time,
peak traced memory, Jacobian-construction time, solver iterations (function evaluations), final objective, convergence
failures, pose error, and the geometry that results from compiling the optimized observations (wall offset against the
true walls, symmetric chamfer). Parity is the acceptance rule: a faster solver must not change the geometry.

Run:  python -m benchmarks.spatial.run_solver_benchmark [--seeds 4] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np

from benchmarks.spatial.run_multiview_benchmark import chamfer
from benchmarks.spatial.run_spatial_benchmark import ROOT, match
from benchmarks.spatial.synth import (
    PoseSim,
    SimConfig,
    corridor_with_repeated_rooms,
    irregular_non_manhattan,
    simulate_multiview,
    two_rooms_shared_wall,
)
from reconstruction.spatial.association import line_through, signed
from reconstruction.spatial.compiler import compile_spatial
from reconstruction.spatial.model import Kind
from reconstruction.spatial.multiview import MultiViewConfig, optimize_multiview

VIEWS = (2, 4, 6, 8, 12, 16, 24, 48, 96)
SCENES = (corridor_with_repeated_rooms, irregular_non_manhattan, two_rooms_shared_wall)


def _offset_error(res, sc) -> float | None:
    ents = [e for e in res.compiled_entities() if e.kind is Kind.WALL]
    mapping, _, _ = match(ents, sc)
    byid = {e.entity_id: e for e in ents}
    errs = []
    for eid, gid in mapping.items():
        g = sc.wall(gid)
        f = line_through(g.a, g.b)
        e = byid[eid]
        errs.append(abs((signed(f, e.a)[1] + signed(f, e.b)[1]) / 2))
    return float(np.mean(errs)) if errs else None


def one(sc, n_views: int, seed: int, solver: str) -> dict:
    local, prior, true = simulate_multiview(sc, SimConfig(n_images=n_views, seed=seed), PoseSim())
    tracemalloc.start()
    mv = optimize_multiview(local, prior, MultiViewConfig(solver=solver))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    res = compile_spatial(mv.observations, storeys=sc.storeys)
    ch = chamfer(res.compiled_entities(), sc)
    pt = [math.hypot(p.tx - true[k].tx, p.ty - true[k].ty) for k, p in mv.poses.items()]
    return {"seconds": mv.seconds, "jac_seconds": mv.jac_seconds, "peak_mb": peak / 1e6, "nfev": mv.nfev, "cost": mv.cost_after,
            "converged": mv.converged, "wall_offset": _offset_error(res, sc), "chamfer": ch["chamfer"],
            "pose_t": float(np.mean(pt)), "n_params": mv.n_params, "n_residuals": mv.n_residuals, "nnz": mv.jac_nnz}


def run(seeds: int, views=VIEWS) -> dict:
    table = {}
    for n in views:
        row = {"views": n, "runs": 0}
        agg = {"dense": [], "sparse": []}
        for mk in SCENES:
            sc = mk()
            for s in range(seeds):
                for solver in ("dense", "sparse"):
                    agg[solver].append(one(sc, n, 100 + s, solver))
        row["runs"] = len(agg["dense"])
        for solver, rows in agg.items():
            f = lambda k: [r[k] for r in rows if r[k] is not None]  # noqa: E731
            row[solver] = {
                "seconds_mean": float(np.mean(f("seconds"))), "seconds_p90": float(np.quantile(f("seconds"), 0.9)),
                "jac_seconds_mean": float(np.mean(f("jac_seconds"))), "peak_mb_mean": float(np.mean(f("peak_mb"))),
                "nfev_mean": float(np.mean(f("nfev"))), "cost_mean": float(np.mean(f("cost"))),
                "convergence_failures": sum(1 for r in rows if not r["converged"]),
                "wall_offset_mean": float(np.mean(f("wall_offset"))) if f("wall_offset") else None,
                "chamfer_mean": float(np.mean(f("chamfer"))) if f("chamfer") else None,
                "pose_translation_error_mean": float(np.mean(f("pose_t"))),
                "n_params_mean": float(np.mean(f("n_params"))), "jac_nnz_mean": float(np.mean(f("nnz")))}
        d, s_ = agg["dense"], agg["sparse"]
        row["parity"] = {
            "max_abs_cost_diff": float(max(abs(a["cost"] - b["cost"]) for a, b in zip(d, s_))),
            "max_abs_pose_error_diff_m": float(max(abs(a["pose_t"] - b["pose_t"]) for a, b in zip(d, s_))),
            "max_abs_wall_offset_diff_m": float(max(abs((a["wall_offset"] or 0) - (b["wall_offset"] or 0)) for a, b in zip(d, s_)))}
        table[str(n)] = row
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "spatial_solver_benchmark.json"))
    a = ap.parse_args(argv)
    t = time.time()
    table = run(a.seeds)
    out = {"table": table, "seeds_per_scene": a.seeds, "scenes": [mk().name for mk in SCENES], "seed_split": "evaluation seeds 100..",
           "wall_clock_s": round(time.time() - t, 1),
           "limits": ["simulated inputs; timing is for this machine only", "association cost is included in both solvers' seconds and is "
                      "identical between them; the dense/sparse difference is the Jacobian + linear algebra share"]}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"{'views':>5} {'runs':>4} | {'dense s':>8} {'sparse s':>8} | {'dense MB':>8} {'sparse MB':>9} | {'nfev d/s':>9} | conv-fail d/s | parity cost / pose m / offset m")
    for n, r in table.items():
        d, s_, p = r["dense"], r["sparse"], r["parity"]
        print(f"{n:>5} {r['runs']:>4} | {d['seconds_mean']:8.3f} {s_['seconds_mean']:8.3f} | {d['peak_mb_mean']:8.2f} {s_['peak_mb_mean']:9.2f} | "
              f"{d['nfev_mean']:4.0f}/{s_['nfev_mean']:<4.0f} | {d['convergence_failures']}/{s_['convergence_failures']} | "
              f"{p['max_abs_cost_diff']:.2e} / {p['max_abs_pose_error_diff_m']:.2e} / {p['max_abs_wall_offset_diff_m']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
