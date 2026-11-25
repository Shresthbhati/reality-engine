"""Reconstruction failure diagnostics against ground truth: a taxonomy of 20 categories, observation-level evidence per failure.

Every record carries: scene, seed, condition, category, severity, entity ids, observation ids, residual, outlier score,
supporting evidence, root cause, resolution status, `excluded` and `tentative` flags. Resolution statuses (never 'resolved'
because a residual vanished):
  unresolved          the error is still present in the COMPILED world
  excluded_tentative  the geometry is kept out of the compiled world (single view) but stays in the record
  surfaced            kept in the world AS a conflict / lower bound / ambiguity (not hidden, not asserted)
  mitigated           the observation is wrong but the FINAL compiled geometry is within tolerance; the cause persists
Ground truth is only used to LABEL failures; nothing here changes a result or a ground-truth value.

Categories: incorrect_association, over_segmentation, under_segmentation, duplicate_structural_geometry, opening_attachment_error,
geometric_outlier, incorrect_plane_estimation, incorrect_coordinate_transform, camera_pose_inconsistency, optimization_divergence,
topology_inconsistency, contradictory_hypotheses, incorrect_storey_assignment, insufficient_viewpoint_coverage, insufficient_evidence,
occlusion_lower_bound, scale_ambiguity, unstable_identity (separate reprocessing probe), incorrect_wall_correspondence, depth_inconsistency
(no depth input in this contract: reported as not evaluable).

Run:  python -m benchmarks.spatial.failure_analysis [--seeds 12] [--out dir]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.spatial.run_spatial_benchmark import ROOT, match
from benchmarks.spatial.synth import PoseSim, SimConfig, all_scenes, simulate_multiview
from reconstruction.spatial.association import line_through, signed
from reconstruction.spatial.compiler import compile_multiview
from reconstruction.spatial.model import Kind, Status
from reconstruction.spatial.topology import Severity
from world_ir.observations import DepthSource

CATEGORIES = (
    "incorrect_association", "over_segmentation", "under_segmentation", "duplicate_structural_geometry", "opening_attachment_error",
    "geometric_outlier", "incorrect_plane_estimation", "incorrect_coordinate_transform", "camera_pose_inconsistency",
    "optimization_divergence", "topology_inconsistency", "contradictory_hypotheses", "incorrect_storey_assignment",
    "insufficient_viewpoint_coverage", "insufficient_evidence", "occlusion_lower_bound", "scale_ambiguity", "unstable_identity",
    "incorrect_wall_correspondence", "depth_inconsistency")

CONDITIONS = {
    "nominal": (lambda sc: SimConfig(n_images=6), PoseSim()),
    "sparse_views": (lambda sc: SimConfig(n_images=3), PoseSim()),
    "heavy_occlusion": (lambda sc: SimConfig(n_images=6, p_truncate=0.95, min_coverage=0.3), PoseSim()),
    "missing_wall": (lambda sc: SimConfig(n_images=6, drop_walls=(sc.walls[-1].id,)), PoseSim()),
    "outlier_fragments": (lambda sc: SimConfig(n_images=6, outlier_rate=0.10), PoseSim()),
    "bad_pose": (lambda sc: SimConfig(n_images=6), PoseSim(bad_image=2, bad_t=0.6)),
    "relative_depth": (lambda sc: SimConfig(n_images=6, depth=DepthSource.RELATIVE), PoseSim()),
}


def _truth(ev: str) -> str:
    return ev.split("#")[1]


def _perp(w, p):
    return abs(signed(line_through(w.a, w.b), p)[1])


def analyse_case(sc, seed: int, cname: str, cfg, ps) -> list[dict]:
    cfg.seed = 100 + seed
    local, prior, true = simulate_multiview(sc, cfg, ps)
    res, mv = compile_multiview(local, prior, storeys=sc.storeys)
    gt = {w.id: w for w in sc.walls}
    gt_ops = {o.id: o for o in sc.openings}
    obs = {o.evidence_id: o for o in mv.observations}
    base = {"scene": sc.name, "seed": seed, "condition": cname}
    out: list[dict] = []

    def add(cat, sev, root, resolution, *, entities=(), observations=(), residual=None, score=None, evidence=None, excluded=False, tentative=False):
        out.append({**base, "category": cat, "severity": sev, "entity_ids": sorted(entities), "observation_ids": sorted(observations),
                    "residual": residual, "outlier_score": score, "supporting_evidence": evidence or {}, "root_cause": root,
                    "resolution": resolution, "excluded": excluded, "tentative": tentative})

    walls = [e for e in res.association.entities if e.kind is Kind.WALL]
    comp = {e.entity_id for e in res.compiled_entities()}
    by_gt = defaultdict(list)
    for e in walls:
        truths = Counter(_truth(x) for x in e.evidence_ids)
        major = truths.most_common(1)[0][0]
        tent = e.entity_id not in comp
        if len(truths) > 1:
            add("under_segmentation", "high", "gate admitted observations of different true walls", "excluded_tentative" if tent else "unresolved",
                entities=[e.entity_id], observations=[x for x in e.evidence_ids if _truth(x) != major],
                evidence={"true_walls": dict(truths), "views": len(e.image_ids)}, excluded=tent, tentative=tent)
            for x in e.evidence_ids:
                if _truth(x) != major:
                    add("incorrect_association", "high", f"observation of {_truth(x)} sits in an entity that is mostly {major}",
                        "excluded_tentative" if tent else "unresolved", entities=[e.entity_id], observations=[x],
                        residual=max(_perp(gt[_truth(x)], p) for p in (obs[x].p0, obs[x].p1)) if _truth(x) in gt else None,
                        excluded=tent, tentative=tent)
        by_gt[major].append(e)
    for g, ents in by_gt.items():
        if len(ents) > 1:
            ents = sorted(ents, key=lambda e: -len(e.evidence_ids))
            for frag in ents[1:]:
                tent = frag.entity_id not in comp
                cat = "over_segmentation" if tent or len(frag.image_ids) < 2 else "duplicate_structural_geometry"
                add(cat, "medium" if tent else "high", "fragment rejected by the fit-based gate (statistical tail or pose-induced offset)",
                    "excluded_tentative" if tent else "unresolved", entities=[frag.entity_id, ents[0].entity_id], observations=list(frag.evidence_ids),
                    evidence={"true_wall": g, "fragment_views": len(frag.image_ids), "main_views": len(ents[0].image_ids)}, excluded=tent, tentative=tent)
    mapping, dups, spur = match([e for e in walls if e.entity_id in comp], sc)
    estmap = {e.entity_id: e for e in walls}
    final_err = {}
    for eid, g in mapping.items():
        e = estmap[eid]
        f = line_through(gt[g].a, gt[g].b)
        d = abs((signed(f, e.a)[1] + signed(f, e.b)[1]) / 2)
        final_err[g] = d
        sg = e.beliefs["position"].sigma or 0.0
        if d > max(4 * sg, 0.15):
            add("incorrect_plane_estimation", "high", "wall line is farther from truth than its own sigma allows (sigma too small or biased inputs)",
                "unresolved", entities=[eid], observations=e.evidence_ids, residual=d, score=d / sg if sg else None,
                evidence={"true_wall": g, "position_sigma_m": sg, "views": len(e.image_ids)})
        if e.storey != gt[g].storey:
            add("incorrect_storey_assignment", "high", "entity storey differs from the true wall's", "unresolved", entities=[eid], residual=None)
        if any(b.status is Status.CONFLICTING for b in e.beliefs.values()):
            add("contradictory_hypotheses", "medium", "conflicting evidence kept as alternatives", "surfaced", entities=[eid],
                evidence={"alternatives": e.alternatives[:3]})
        if not (e.end_a_observed and e.end_b_observed):
            add("occlusion_lower_bound", "info", "an end was never observed; length is a lower bound", "surfaced", entities=[eid])
    # observation-level outliers (truth label vs the post-optimization world observation)
    for ev, o in obs.items():
        if o.kind is not Kind.WALL or _truth(ev) not in gt:
            continue
        z = max(_perp(gt[_truth(ev)], p) for p in (o.p0, o.p1)) / o.sigma_m
        if z > 4.0:
            g = _truth(ev)
            detected = ev in mv.outliers
            ent = next((e for e in walls if ev in e.evidence_ids), None)
            tent = ent is not None and ent.entity_id not in comp
            fe = final_err.get(g)
            add("geometric_outlier", "medium", "observation farther than 4 sigma from the true wall (gross lift / pose error)",
                "excluded_tentative" if tent else ("mitigated" if fe is not None and fe < 0.1 else "unresolved"),
                entities=[ent.entity_id] if ent else [], observations=[ev], residual=z * o.sigma_m, score=z,
                evidence={"flagged_by_optimizer": detected, "multiview_residual_z": mv.residual_z.get(ev), "final_wall_error_m": fe},
                excluded=tent, tentative=tent)
    # openings
    for e in res.association.entities:
        if e.kind not in (Kind.DOOR, Kind.WINDOW, Kind.OPENING):
            continue
        tids = {_truth(x) for x in e.evidence_ids}
        tent = e.entity_id not in comp
        if len(tids) > 1:
            add("opening_attachment_error", "high", "observations of different true openings were merged", "unresolved", entities=[e.entity_id],
                observations=e.evidence_ids, evidence={"true_openings": sorted(tids)}, excluded=tent, tentative=tent)
            continue
        t = next(iter(tids))
        if t in gt_ops and e.host_id in estmap:
            host_truth = Counter(_truth(x) for x in estmap[e.host_id].evidence_ids).most_common(1)[0][0]
            if host_truth != gt_ops[t].host:
                add("opening_attachment_error", "high", f"attached to {host_truth}, true host is {gt_ops[t].host}", "unresolved",
                    entities=[e.entity_id, e.host_id], observations=e.evidence_ids, excluded=tent, tentative=tent)
        elif t in gt_ops and e.host_id is None:
            add("opening_attachment_error", "low", "host unknown or ambiguous (kept unattached)", "surfaced", entities=[e.entity_id],
                observations=e.evidence_ids, evidence={"host_alternatives": list(e.host_alternatives)}, excluded=tent, tentative=tent)
    # coverage / missing evidence
    views = defaultdict(set)
    for ev, o in obs.items():
        if _truth(ev) in gt:
            views[_truth(ev)].add(o.image_id)
    matched = set(mapping.values())
    for gid in gt:
        n = len(views.get(gid, ()))
        if gid not in matched:
            if n == 0:
                add("insufficient_evidence", "info", "no view observed this wall: nothing was reconstructed (and nothing fabricated)", "unresolved",
                    evidence={"true_wall": gid, "views": 0})
            elif n < 2:
                add("insufficient_viewpoint_coverage", "medium", "seen from one view only: tentative, not compiled", "excluded_tentative",
                    evidence={"true_wall": gid, "views": n}, excluded=True, tentative=True)
            else:
                add("incorrect_wall_correspondence", "high", "wall observed from several views but no compiled entity matches it geometrically", "unresolved",
                    evidence={"true_wall": gid, "views": n})
    # poses / optimizer
    for k, p in mv.poses.items():
        sg = p.sigma()
        err = math.hypot(p.tx - true[k].tx, p.ty - true[k].ty)
        if sg and err > 4 * sg[1]:
            add("camera_pose_inconsistency", "high", "pose error exceeds 4x its reported sigma", "unresolved", observations=[x for x, o in obs.items() if o.image_id == k],
                residual=err, score=err / sg[1], evidence={"image": k, "pose_sigma_m": sg[1]})
        if err > 0.3:
            add("incorrect_coordinate_transform", "high", "image pose is more than 0.3 m from truth after optimization", "unresolved",
                observations=[x for x, o in obs.items() if o.image_id == k], residual=err, evidence={"image": k})
    if (not mv.converged) or mv.cost_after > mv.cost_before:
        add("optimization_divergence", "high", "optimizer did not converge or increased its objective", "unresolved",
            evidence={"diagnostics": mv.diagnostics, "cost_before": mv.cost_before, "cost_after": mv.cost_after})
    for i in res.topology.issues:
        if i.severity in (Severity.STRUCTURAL_ERROR, Severity.CONTRADICTION):
            add("topology_inconsistency", "high" if i.severity is Severity.STRUCTURAL_ERROR else "medium", i.code, "surfaced",
                entities=list(i.entities), observations=list(i.evidence_ids)[:20], evidence={"message": i.message})
    if cfg.depth is DepthSource.RELATIVE:
        add("scale_ambiguity", "medium", "relative-depth inputs: sigma is ~25% of range, metric precision is not supported", "surfaced",
            evidence={"median_observation_sigma_m": sorted(o.sigma_m for o in local)[len(local) // 2]})
    return out


def identity_stability(seeds: int) -> dict:
    """Reprocessing probe: add one image to the evidence and ask whether established walls keep their entity id."""
    kept = changed = 0
    for sc in all_scenes():
        for s in range(seeds):
            r = []
            for n in (6, 7):
                local, prior, _ = simulate_multiview(sc, SimConfig(n_images=n, seed=100 + s), PoseSim())
                res, _ = compile_multiview(local, prior, storeys=sc.storeys)
                r.append([e for e in res.compiled_entities() if e.kind is Kind.WALL])
            for e in r[0]:
                near = min(r[1], key=lambda x: math.dist(x.a, e.a) + math.dist(x.b, e.b), default=None)
                if near is None or math.dist(near.a, e.a) + math.dist(near.b, e.b) > 0.6:
                    continue
                kept += near.entity_id == e.entity_id
                changed += near.entity_id != e.entity_id
    n = kept + changed
    return {"compared": n, "id_kept": kept, "id_changed": changed, "change_rate": (changed / n) if n else None,
            "root_cause": "entity ids are a hash of the supporting evidence set, so adding any evidence changes the id",
            "resolution": "unresolved"}


def run(seeds: int) -> dict:
    failures, cases = [], 0
    for cname, (mk, ps) in CONDITIONS.items():
        for sc in all_scenes():
            for s in range(seeds):
                cases += 1
                failures += analyse_case(sc, s, cname, mk(sc), ps)
    by_cat = Counter(f["category"] for f in failures)
    by_res = Counter(f["resolution"] for f in failures)
    by_cond = defaultdict(Counter)
    for f in failures:
        by_cond[f["condition"]][f["category"]] += 1
    actionable = [f for f in failures if f["severity"] in ("high", "medium") and f["resolution"] == "unresolved"]
    return {"cases": cases, "seeds_per_scene": seeds, "seed_split": "evaluation seeds 100..", "taxonomy": list(CATEGORIES),
            "counts_by_category": dict(by_cat), "counts_by_resolution": dict(by_res),
            "counts_by_condition": {k: dict(v) for k, v in by_cond.items()}, "unresolved_high_or_medium": len(actionable),
            "identity_stability": identity_stability(max(2, seeds // 3)),
            "not_evaluable": {"depth_inconsistency": "spatial_inputs.v1 carries no depth channel; nothing to compare"},
            "failures": failures}


def summary_text(r: dict) -> str:
    L = [f"Failure analysis: {r['cases']} cases ({r['seeds_per_scene']} seeds x 7 scenes x {len(CONDITIONS)} conditions), {len(r['failures'])} records",
         "", "By category:"] + [f"  {c:34s} {n:5d}" for c, n in sorted(r["counts_by_category"].items(), key=lambda kv: -kv[1])]
    L += ["", "By resolution: " + ", ".join(f"{k}={v}" for k, v in sorted(r["counts_by_resolution"].items())),
          f"Unresolved high/medium severity: {r['unresolved_high_or_medium']}",
          f"Identity stability on reprocessing: {r['identity_stability']['id_changed']}/{r['identity_stability']['compared']} established walls changed id "
          f"({r['identity_stability']['root_cause']})", "", "By condition (top categories):"]
    for c, cnt in r["counts_by_condition"].items():
        L.append(f"  {c:18s} " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    a = ap.parse_args(argv)
    r = run(a.seeds)
    out = Path(a.out)
    (out / "spatial_failure_analysis.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
    (out / "spatial_failure_summary.txt").write_text(summary_text(r), encoding="utf-8")
    print(summary_text(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
