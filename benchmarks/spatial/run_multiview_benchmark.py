"""Multi-view benchmark: existing pipeline vs. joint optimization vs. optimization + spatial constraints.

Arms (same observations, same ground truth, same scoring as run_spatial_benchmark):
  existing          prior poses applied as given -> compile_spatial (association + constraints), no optimization
  optimized         optimize_multiview -> compile_spatial with junction constraints OFF
  opt+constraints   optimize_multiview -> compile_spatial with junction constraints ON
Extra metrics: symmetric chamfer distance of reconstructed vs true wall geometry (plan view, 5 cm samples; one-sided
parts report accuracy est->GT and completeness GT->est separately), camera pose error before/after, calibration of the
reported pose sigma, global validation state, wall-clock and peak memory.

Splits: development seeds 0-11 were used to look at failures; THIS benchmark evaluates on seeds >= 100 and changes no
parameter. Observations are simulated over known truth, so nothing here is a real-world accuracy claim.

Run:  python -m benchmarks.spatial.run_multiview_benchmark [--seeds 6] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from benchmarks.spatial.run_spatial_benchmark import ROOT, _pool, score
from benchmarks.spatial.synth import (
    GTScene,
    PoseSim,
    SimConfig,
    all_scenes,
    corridor_with_repeated_rooms,
    simulate_multiview,
)
from reconstruction.spatial.compiler import compile_spatial
from reconstruction.spatial.model import Kind
from reconstruction.spatial.multiview import apply_pose, optimize_multiview
from reconstruction.spatial.validation import classify_reconstruction

ARMS = ("existing", "optimized", "opt+constraints")


def _samples(a, b, step=0.05):
    n = max(2, int(math.dist(a, b) / step) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def chamfer(ents, scene: GTScene) -> dict[str, float]:
    est = [e for e in ents if e.kind is Kind.WALL]
    out = {}
    for storey in scene.storeys:
        G = np.vstack([_samples(w.a, w.b) for w in scene.walls if w.storey == storey])
        E = [_samples(e.a, e.b) for e in est if e.storey == storey]
        if not E:
            out[storey] = None
            continue
        E = np.vstack(E)
        acc = cKDTree(G).query(E)[0].mean()
        comp = cKDTree(E).query(G)[0].mean()
        out[storey] = (float(acc), float(comp))
    vals = [v for v in out.values() if v]
    if not vals:
        return {"acc": None, "comp": None, "chamfer": None}
    acc, comp = float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
    return {"acc": acc, "comp": comp, "chamfer": acc + comp}


def pose_errors(poses, true):
    t, r = [], []
    for k, p in poses.items():
        T = true[k]
        t.append(math.hypot(p[1] - T.tx, p[2] - T.ty))
        r.append(abs((p[0] - T.theta + math.pi) % (2 * math.pi) - math.pi))
    return t, r


def run_case(sc: GTScene, cfg: SimConfig, ps: PoseSim) -> dict[str, dict]:
    local, prior, true = simulate_multiview(sc, cfg, ps)
    out: dict[str, dict] = {}
    mv = None
    for arm in ARMS:
        t0 = time.perf_counter()
        if arm == "existing":
            obs = [apply_pose(o, prior[o.image_id].theta, prior[o.image_id].tx, prior[o.image_id].ty) for o in local]
            res = compile_spatial(obs, storeys=sc.storeys)
            poses = {k: (p.theta, p.tx, p.ty) for k, p in prior.items()}
            zs = []
        else:
            if mv is None:
                mv = optimize_multiview(local, prior)
            res = compile_spatial(mv.observations, storeys=sc.storeys, impose_junctions=(arm == "opt+constraints"))
            poses = {k: (p.theta, p.tx, p.ty) for k, p in mv.poses.items()}
            zs = []
            for k, p in mv.poses.items():
                sg = p.sigma()
                T = true[k]
                if sg:
                    zs.append((abs((p.theta - T.theta + math.pi) % (2 * math.pi) - math.pi) / sg[0],
                               math.hypot(p.tx - T.tx, p.ty - T.ty) / sg[1]))
        dt = time.perf_counter() - t0
        ents = res.compiled_entities()
        s = score(ents, res.constraints.junctions, sc, [o for o in local], arm != "existing" or True)
        te, re_ = pose_errors(poses, true)
        gv = classify_reconstruction(res.entities, res.topology, res.constraints.junctions, optimization=mv if arm != "existing" else None)
        s.update({"chamfer": chamfer(ents, sc), "pose_t": te, "pose_r": [math.degrees(x) for x in re_], "pose_z": zs,
                  "state": gv.state.value, "seconds": dt})
        out[arm] = s
    return out


def _agg(rows: list[dict]) -> dict:
    d = _pool(rows)
    ch = [r["chamfer"] for r in rows if r["chamfer"]["chamfer"] is not None]
    d["chamfer_total_m"] = float(np.mean([c["chamfer"] for c in ch])) if ch else None
    d["chamfer_accuracy_m"] = float(np.mean([c["acc"] for c in ch])) if ch else None
    d["chamfer_completeness_m"] = float(np.mean([c["comp"] for c in ch])) if ch else None
    pt = [x for r in rows for x in r["pose_t"]]
    pr = [x for r in rows for x in r["pose_r"]]
    d["pose_translation_error_m"] = {"mean": float(np.mean(pt)), "p90": float(np.quantile(pt, 0.9))} if pt else None
    d["pose_rotation_error_deg"] = {"mean": float(np.mean(pr)), "p90": float(np.quantile(pr, 0.9))} if pr else None
    zs = [z for r in rows for z in r["pose_z"]]
    if zs:
        zt = np.array([z[1] for z in zs])
        zr = np.array([z[0] for z in zs])
        d["pose_sigma_calibration"] = {"rotation_within_1_2_3_sigma": [float(np.mean(zr <= k)) for k in (1, 2, 3)],
                                       "translation_within_1_2_3_sigma(rayleigh 0.39/0.86/0.99)": [float(np.mean(zt <= k)) for k in (1, 2, 3)]}
    d["validation_states"] = dict(Counter(r["state"] for r in rows))
    d["seconds_mean"] = float(np.mean([r["seconds"] for r in rows]))
    return d


CONDITIONS = {
    "single_view": (lambda sc: SimConfig(n_images=1), PoseSim()),
    "two_views": (lambda sc: SimConfig(n_images=2), PoseSim()),
    "three_views": (lambda sc: SimConfig(n_images=3), PoseSim()),
    "six_views": (lambda sc: SimConfig(n_images=6), PoseSim()),
    "twelve_views": (lambda sc: SimConfig(n_images=12), PoseSim()),
    "poor_coverage": (lambda sc: SimConfig(n_images=4, p_see_wall=0.4), PoseSim()),
    "heavy_occlusion": (lambda sc: SimConfig(n_images=6, p_truncate=0.95, min_coverage=0.3), PoseSim()),
    "large_pose_noise": (lambda sc: SimConfig(n_images=6), PoseSim(sigma_theta=0.05, sigma_t=0.30)),
    "one_bad_pose_0.6m": (lambda sc: SimConfig(n_images=6), PoseSim(bad_image=2, bad_t=0.6)),
    "overconfident_priors_x0.4": (lambda sc: SimConfig(n_images=6), PoseSim(declared_scale=0.4)),
}


def run(seeds: int, scenes=None) -> dict:
    scenes = scenes or all_scenes()
    res: dict[str, dict] = {}
    for cname, (mk, ps) in CONDITIONS.items():
        rows = {a: [] for a in ARMS}
        for sc in scenes:
            for s in range(seeds):
                cfg = mk(sc)
                cfg.seed = 100 + s            # evaluation split: disjoint from the development seeds 0-11
                for arm, r in run_case(sc, cfg, ps).items():
                    rows[arm].append(r)
        res[cname] = {a: _agg(v) for a, v in rows.items()}
    return res


def profile_scaling(sizes=(2, 4, 8, 16, 32, 64)) -> list[dict]:
    sc = corridor_with_repeated_rooms()
    out = []
    for n in sizes:
        local, prior, _ = simulate_multiview(sc, SimConfig(n_images=n, seed=7), PoseSim())
        tracemalloc.start()
        t0 = time.perf_counter()
        mv = optimize_multiview(local, prior)
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        t1 = time.perf_counter()
        warm = optimize_multiview(local, prior, warm_start=mv)
        out.append({"images": n, "observations": len(local), "seconds": round(dt, 3), "nfev": mv.nfev,
                    "peak_mem_mb": round(peak / 1e6, 1), "converged": mv.converged, "clusters": mv.n_clusters,
                    "warm_restart_seconds": round(time.perf_counter() - t1, 3), "warm_restart_nfev": warm.nfev})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "spatial_multiview_benchmark.json"))
    a = ap.parse_args(argv)
    t = time.time()
    result = {"conditions": run(a.seeds), "profile": profile_scaling(), "seeds": a.seeds, "seed_split": "evaluation seeds 100..",
              "limits": ["simulated observations and simulated pose errors over known truth; no real image set with an independent "
                         "3D ground truth is available to this repo's benchmark harness, so no real-world claim is made",
                         "pose priors are plan-view (yaw + horizontal translation); scale and tilt are not optimised"]}
    result["wall_clock_s"] = round(time.time() - t, 1)
    Path(a.out).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    f = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
    for c, v in result["conditions"].items():
        for arm in ARMS:
            m = v[arm]
            print(f"{c[:26]:26s} {arm:16s} off {f(m['wall_offset_error_m']['mean'])} chamfer {f(m['chamfer_total_m'])} "
                  f"ent/gt {f(m['entities_per_gt_wall'])} dup {f(m['duplicate_rate'])} jR {f(m['junction_recall'])} "
                  f"rP/R {f(m['room_precision'])}/{f(m['room_recall'])} poseT {f((m['pose_translation_error_m'] or {}).get('mean'))} "
                  f"poseR {f((m['pose_rotation_error_deg'] or {}).get('mean'))} {m['seconds_mean']:.2f}s")
    print("profile", result["profile"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
