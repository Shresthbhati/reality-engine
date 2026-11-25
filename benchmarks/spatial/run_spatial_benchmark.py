"""Spatial compiler benchmark: compiler vs. two baselines, against ground truth, across failure conditions.

Baselines (both implemented here, both deliberately the kind of thing the compiler replaces):
  raw    every observation becomes its own entity (no association at all);
  naive  pairwise greedy merge with FIXED thresholds (5 deg, 0.15 m, small gap), mean end points, no sigma,
         no junction snapping, junctions by plain proximity (0.25 m) -- the legacy union-find style.
Metrics (per condition, pooled over scenes x seeds): geometric error (wall offset / angle / observed-end),
dimensional error (length of fully-bounded walls, room area, opening width), topological accuracy (junction
pair precision/recall, room precision/recall), association accuracy (pairwise precision/recall of
observation->wall), duplicate rate, uncertainty calibration (coverage of 1/2/3 sigma), evidence traceability.

All observations are SIMULATED over known truth (see synth.py): the numbers show the compiler's behaviour under
the stated noise model, not perception accuracy on real photographs.

Run:  python -m benchmarks.spatial.run_spatial_benchmark [--seeds 10] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.spatial.synth import (
    GTScene,
    SimConfig,
    all_scenes,
    from_surveyed,
    simulate_observations,
)
from reconstruction.spatial.association import line_through, signed
from reconstruction.spatial.compiler import compile_spatial, inspect_spatial_world, to_worldir
from reconstruction.spatial.model import Kind, SpatialEntity, StructuralObservation, angle_diff
from reconstruction.spatial.topology import _faces, build_graph
from world_ir.observations import DepthSource

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------- baselines

def _entity_from(obs: Sequence[StructuralObservation], eid: str) -> SpatialEntity:
    pts = np.array([p for o in obs for p in (o.p0, o.p1)], float)
    c = pts.mean(0)
    _, vec = np.linalg.eigh(np.cov((pts - c).T)) if len(pts) > 2 else (None, np.array([[0, 1], [1, 0]]))
    if len(pts) == 2:
        d = pts[1] - pts[0]
        u = d / np.linalg.norm(d)
    else:
        u = vec[:, 1]
    t = (pts - c) @ u
    a, b = c + t.min() * u, c + t.max() * u
    from reconstruction.spatial.model import UNKNOWN_BELIEF
    return SpatialEntity(eid, obs[0].kind, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), None, None,
                         tuple(o.evidence_id for o in obs), tuple(sorted({o.image_id for o in obs})),
                         {k: UNKNOWN_BELIEF for k in ("existence", "position", "dimensions", "orientation", "geometry", "topology", "semantic")},
                         end_a_observed=True, end_b_observed=True, storey=obs[0].storey)


def baseline_raw(observations) -> list[SpatialEntity]:
    return [_entity_from([o], f"raw-{i}") for i, o in enumerate(observations) if o.kind is Kind.WALL]


def baseline_naive(observations) -> list[SpatialEntity]:
    walls = [o for o in observations if o.kind is Kind.WALL]
    parent = list(range(len(walls)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in combinations(range(len(walls)), 2):
        a, b = walls[i], walls[j]
        if a.storey != b.storey:
            continue
        fit = line_through(a.p0, a.p1)
        ang = angle_diff(fit.theta, line_through(b.p0, b.p1).theta)
        d = max(abs(signed(fit, b.p0)[1]), abs(signed(fit, b.p1)[1]))
        t = sorted([signed(fit, b.p0)[0], signed(fit, b.p1)[0]])
        gap = max(0.0, t[0] - fit.t_hi, fit.t_lo - t[1])
        if ang < math.radians(5) and d < 0.15 and gap < 0.3:
            parent[find(i)] = find(j)
    groups: dict[int, list[StructuralObservation]] = defaultdict(list)
    for i, o in enumerate(walls):
        groups[find(i)].append(o)
    return [_entity_from(g, f"naive-{k}") for k, g in sorted(groups.items())]


def proximity_junctions(ents: Sequence[SpatialEntity], tol: float = 0.25):
    out = []
    for i, wi in enumerate(ents):
        for which, p in (("a", wi.a), ("b", wi.b)):
            for j, wj in enumerate(ents):
                if i == j or wi.storey != wj.storey:
                    continue
                f = line_through(wj.a, wj.b)
                t, d = signed(f, p)
                if abs(d) <= tol and f.t_lo - tol <= t <= f.t_hi + tol and angle_diff(wi.angle % math.pi, wj.angle % math.pi) > math.radians(10):
                    out.append((wi.entity_id, which, wj.entity_id, "t"))
    return out


def rooms_from(ents, junctions):
    node_xy, edges, *_ = build_graph(ents, junctions)
    faces, _ = _faces(node_xy, edges)
    return [(sorted(set(w)), 0.5 * abs(sum(node_xy[c[i]][0] * node_xy[c[(i + 1) % len(c)]][1] - node_xy[c[(i + 1) % len(c)]][0] * node_xy[c[i]][1] for i in range(len(c)))))
            for c, w in faces if len(c) >= 3]


# ----------------------------------------------------------------------------------------------- evaluation

def match(est: Sequence[SpatialEntity], scene: GTScene) -> tuple[dict[str, str], list[str], list[str]]:
    """est wall id -> gt wall id (Hungarian on offset+angle within storey). Returns (map, duplicates, spurious)."""
    gts = scene.walls
    big = 1e6
    C = np.full((len(est), len(gts)), big)
    for i, e in enumerate(est):
        for j, g in enumerate(gts):
            if e.storey != g.storey:
                continue
            f = line_through(g.a, g.b)
            ang = angle_diff(e.angle % math.pi, f.theta)
            d = (abs(signed(f, e.a)[1]) + abs(signed(f, e.b)[1])) / 2
            lo, hi = sorted((signed(f, e.a)[0], signed(f, e.b)[0]))
            ov = min(hi, f.t_hi) - max(lo, f.t_lo)
            if ang < math.radians(20) and d < 0.5 and ov > 0.3 * min(e.length, g.a and math.dist(g.a, g.b)):
                C[i, j] = d + ang
    r, c = linear_sum_assignment(C)
    mapping = {est[i].entity_id: gts[j].id for i, j in zip(r, c) if C[i, j] < big}
    dups, spur = [], []
    for i, e in enumerate(est):
        if e.entity_id in mapping:
            continue
        (dups if C[i].min() < big else spur).append(e.entity_id)
    return mapping, dups, spur


def gt_junctions(scene: GTScene):
    pairs = set()
    for i, wi in enumerate(scene.walls):
        for wj in scene.walls:
            if wi is wj or wi.storey != wj.storey:
                continue
            f = line_through(wj.a, wj.b)
            for p in (wi.a, wi.b):
                t, d = signed(f, p)
                if abs(d) < 1e-6 and f.t_lo - 1e-6 <= t <= f.t_hi + 1e-6 and angle_diff(line_through(wi.a, wi.b).theta, f.theta) > 1e-3:
                    pairs.add(frozenset((wi.id, wj.id)))
    return pairs


def score(ents: Sequence[SpatialEntity], junctions, scene: GTScene, observations, with_sigma: bool) -> dict:
    ests = [e for e in ents if e.kind is Kind.WALL]
    mapping, dups, spur = match(ests, scene)
    byid = {e.entity_id: e for e in ests}
    gtd = {g.id: g for g in scene.walls}
    off, ang, endv, lens, z_off, z_end = [], [], [], [], [], []
    lb_ok = lb_n = 0
    for eid, gid in mapping.items():
        e, g = byid[eid], gtd[gid]
        f = line_through(g.a, g.b)
        d = (signed(f, e.a)[1] + signed(f, e.b)[1]) / 2
        off.append(abs(d))
        ang.append(math.degrees(angle_diff(e.angle % math.pi, f.theta)))
        if with_sigma and e.beliefs["position"].sigma:
            z_off.append(abs(d) / e.beliefs["position"].sigma)
        for p, seen, sg in ((e.a, e.end_a_observed, e.end_sigma("a")), (e.b, e.end_b_observed, e.end_sigma("b"))):
            if seen:
                err = min(math.dist(p, g.a), math.dist(p, g.b))
                endv.append(err)
                if with_sigma and sg:
                    z_end.append(err / sg)
        if e.end_a_observed and e.end_b_observed:
            lens.append(abs(e.length - math.dist(g.a, g.b)))
        else:
            lb_n += 1
            lb_ok += int(e.length <= math.dist(g.a, g.b) + 0.15)
    # junctions
    gj = gt_junctions(scene)
    ej = set()
    for wi, _, wj, _ in junctions:
        if wi in mapping and wj in mapping and mapping[wi] != mapping[wj]:
            ej.add(frozenset((mapping[wi], mapping[wj])))
    tp = len(gj & ej)
    # rooms
    gt_rooms = {frozenset(v) for v in scene.rooms.values()}
    est_rooms = []
    for wids, area in rooms_from(ests, junctions):
        est_rooms.append((frozenset(mapping[w] for w in wids if w in mapping), area, len(wids) == len([w for w in wids if w in mapping])))
    rtp = sum(1 for s, _, ok in est_rooms if ok and s in gt_rooms)
    # association pairwise
    est_of = {}
    for e in ests:
        for ev in e.evidence_ids:
            est_of[ev] = e.entity_id
    walls_obs = [o for o in observations if o.kind is Kind.WALL]
    truth = {o.evidence_id: o.evidence_id.split("#")[1] for o in walls_obs}
    tpp = fpp = fnp = 0
    for a, b in combinations([o.evidence_id for o in walls_obs], 2):
        same_t, same_e = truth[a] == truth[b], est_of.get(a) is not None and est_of.get(a) == est_of.get(b)
        tpp += same_t and same_e
        fpp += (not same_t) and same_e
        fnp += same_t and not same_e
    return {"n_est": len(ests), "n_gt": len(scene.walls), "matched": len(mapping), "duplicates": len(dups), "spurious": len(spur),
            "off": off, "ang": ang, "endv": endv, "lens": lens, "z_off": z_off, "z_end": z_end, "lb_ok": lb_ok, "lb_n": lb_n,
            "j_tp": tp, "j_est": len(ej), "j_gt": len(gj), "r_tp": rtp, "r_est": len(est_rooms), "r_gt": len(gt_rooms),
            "pp_tp": tpp, "pp_fp": fpp, "pp_fn": fnp}


def _pool(rows: list[dict]) -> dict:
    def cat(k):
        return [x for r in rows for x in r[k]]

    def s(k):
        return sum(r[k] for r in rows)

    def stat(v):
        return {"n": len(v), "mean": float(np.mean(v)) if v else None, "p90": float(np.quantile(v, 0.9)) if v else None}

    prec = lambda tp, n: tp / n if n else None  # noqa: E731
    zo, ze = cat("z_off"), cat("z_end")
    return {
        "wall_offset_error_m": stat(cat("off")), "wall_angle_error_deg": stat(cat("ang")),
        "observed_end_error_m": stat(cat("endv")), "length_error_fully_bounded_m": stat(cat("lens")),
        "wall_recall": prec(s("matched"), s("n_gt")), "duplicate_rate": prec(s("duplicates"), s("n_gt")),
        "spurious_per_scene": s("spurious") / len(rows), "entities_per_gt_wall": prec(s("n_est"), s("n_gt")),
        "junction_precision": prec(s("j_tp"), s("j_est")), "junction_recall": prec(s("j_tp"), s("j_gt")),
        "room_precision": prec(s("r_tp"), s("r_est")), "room_recall": prec(s("r_tp"), s("r_gt")),
        "association_pair_precision": prec(s("pp_tp"), s("pp_tp") + s("pp_fp")),
        "association_pair_recall": prec(s("pp_tp"), s("pp_tp") + s("pp_fn")),
        "lower_bound_valid_fraction": prec(s("lb_ok"), s("lb_n")),
        "calibration_offset_within_1_2_3_sigma": [float(np.mean(np.array(zo) <= k)) for k in (1, 2, 3)] if zo else None,
        "calibration_end_within_1_2_3_sigma": [float(np.mean(np.array(ze) <= k)) for k in (1, 2, 3)] if ze else None,
    }


CONDITIONS = {
    "nominal": lambda sc: SimConfig(),
    "sparse_views": lambda sc: SimConfig(n_images=3),
    "heavy_occlusion": lambda sc: SimConfig(p_truncate=0.95, min_coverage=0.3),
    "missing_wall": lambda sc: SimConfig(drop_walls=(sc.walls[-1].id,)),
    "gross_outliers_10pct": lambda sc: SimConfig(outlier_rate=0.10),
    "one_biased_image_0.4m": lambda sc: SimConfig(biased_image=2, bias_m=0.4),
    "monocular_relative_depth": lambda sc: SimConfig(depth=DepthSource.RELATIVE),
    "miscalibrated_sigma_x2": lambda sc: SimConfig(noise_scale=2.0),
}


def run(seeds: int, scenes: Sequence[GTScene] | None = None) -> dict:
    scenes = scenes or all_scenes()
    out: dict[str, dict] = {}
    trace_ok = trace_n = 0
    for cname, mk in CONDITIONS.items():
        rows = {"compiler": [], "baseline_raw": [], "baseline_naive": []}
        status = defaultdict(int)
        for sc in scenes:
            for seed in range(seeds):
                cfg = mk(sc)
                cfg.seed = seed
                obs = simulate_observations(sc, cfg)
                res = compile_spatial(obs, storeys=sc.storeys)
                ents = res.compiled_entities()
                rows["compiler"].append(score(ents, res.constraints.junctions, sc, obs, True))
                raw, naive = baseline_raw(obs), baseline_naive(obs)
                rows["baseline_raw"].append(score(raw, proximity_junctions(raw), sc, obs, False))
                rows["baseline_naive"].append(score(naive, proximity_junctions(naive), sc, obs, False))
                status[res.topology.status()] += 1
                if cname == "nominal" and seed < 2:   # traceability through a real world round trip
                    w = to_worldir(res)
                    ins = inspect_spatial_world(w)
                    trace_n += len([e for e in w.entities.values() if (e.custom_properties or {}).get("evidence_ids")])
                    bad = {e for e, _ in ins["unresolved_evidence"]}
                    trace_ok += len([e for e in w.entities.values() if (e.custom_properties or {}).get("evidence_ids") and e.id not in bad])
        out[cname] = {k: _pool(v) for k, v in rows.items()}
        out[cname]["compiler_topology_status_counts"] = dict(status)
    out["_traceability"] = {"entities_checked": trace_n, "entities_with_all_evidence_resolvable": trace_ok}
    return out


def surveyed_scenes() -> list[GTScene]:
    d = ROOT / "benchmarks" / "measured_rooms"
    return [from_surveyed(p) for p in sorted(d.glob("room_0*/ground_truth.json"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "spatial_compiler_benchmark.json"))
    a = ap.parse_args(argv)
    t = time.time()
    result = {"synthetic_scenes": run(a.seeds), "surveyed_ground_truth_rooms": run(a.seeds, surveyed_scenes()),
              "seeds": a.seeds, "wall_clock_s": None,
              "limits": ["observations are simulated over known truth; this measures the compiler under the stated noise "
                         "model, not perception accuracy on real photographs",
                         "surveyed rooms contribute real laser-measured ground-truth plans, with simulated observations",
                         "thresholds are starting points and were not tuned on these results"]}
    result["wall_clock_s"] = round(time.time() - t, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    for group in ("synthetic_scenes", "surveyed_ground_truth_rooms"):
        print(f"== {group}")
        for cname, d in result[group].items():
            if cname.startswith("_"):
                continue
            for who in ("compiler", "baseline_naive", "baseline_raw"):
                m = d[who]
                f = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
                print(f"{cname:28s} {who:15s} off {f(m['wall_offset_error_m']['mean'])} ang {f(m['wall_angle_error_deg']['mean'])} "
                      f"ent/gt {f(m['entities_per_gt_wall'])} dup {f(m['duplicate_rate'])} jP {f(m['junction_precision'])} "
                      f"jR {f(m['junction_recall'])} roomP {f(m['room_precision'])} roomR {f(m['room_recall'])} "
                      f"asoc P/R {f(m['association_pair_precision'])}/{f(m['association_pair_recall'])}")
    print("traceability", result["synthetic_scenes"]["_traceability"], "wall clock", result["wall_clock_s"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
