"""Pipeline-facing orchestration of the global spatial reconstruction: a deterministic JSON-in / JSON-out contract.

Input contract `spatial_inputs.v1` (producer: the perception side; this module never invents any of it):
    {"schema": "spatial_inputs.v1", "scale_state": "metric", "frame": "world", "storeys": {"L0": 0.0},
     "poses": [{"image_id", "theta", "tx", "ty", "sigma_theta"?, "sigma_t"?}],
     "observations": [StructuralObservation.to_dict() with frame_id == image_id]}
Every refusal is STRUCTURED (`code`, `message`) and nothing is guessed:
    BAD_SCHEMA / MISSING_FIELD / NON_FINITE / DUPLICATE_EVIDENCE_ID / NO_POSE_FOR_FRAME / FRAME_MISMATCH   -> status "failed"
    SCALE_NOT_METRIC (scale_state != "metric": metres must not be assumed)                                  -> status "refused"
    unchanged input hash vs the previous run                                                               -> status "unchanged"
Output `facts` is JSON-serialisable: status, input hash, optimization summary (solver, cost, iterations, outliers),
global validation state + flags, topology status, completeness axes, per-dimension epistemic-status counts, sigma
budget presence, tentative exclusions with their evidence ids, and warnings (e.g. implausible wall lengths for a
metric claim). The world is persisted through WorldStore with the previous version as parent, so lineage is preserved
and an earlier version is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .compiler import compile_multiview, identity_report, to_worldir
from .model import DIMENSIONS, Kind, StructuralObservation
from .multiview import ImagePose, MultiViewConfig, attach_optimization
from .validation import classify_reconstruction

SCHEMA = "spatial_inputs.v1"
HASH_KEY = "spatial_input_hash"


class SpatialInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def input_hash(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def parse_inputs(d: dict):
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise SpatialInputError("BAD_SCHEMA", f"expected schema {SCHEMA!r}")
    for k in ("scale_state", "poses", "observations"):
        if k not in d:
            raise SpatialInputError("MISSING_FIELD", f"missing {k!r}")
    poses: dict[str, ImagePose] = {}
    for p in d["poses"]:
        if not _finite(p.get("theta"), p.get("tx"), p.get("ty")) or any(
                p.get(s) is not None and not (_finite(p[s]) and p[s] > 0) for s in ("sigma_theta", "sigma_t")):
            raise SpatialInputError("NON_FINITE", f"pose {p.get('image_id')!r} has a non-finite value or a non-positive sigma")
        poses[p["image_id"]] = ImagePose(p["image_id"], p["theta"], p["tx"], p["ty"], p.get("sigma_theta"), p.get("sigma_t"))
    obs = []
    for o in d["observations"]:
        try:
            ob = StructuralObservation.from_dict(o)
        except (KeyError, ValueError, TypeError) as exc:
            raise SpatialInputError("MISSING_FIELD", f"observation {o.get('evidence_id')!r}: {exc}") from exc
        if not _finite(*ob.p0, *ob.p1, ob.sigma_m):
            raise SpatialInputError("NON_FINITE", f"observation {ob.evidence_id!r} has a non-finite coordinate")
        if ob.frame_id != ob.image_id:
            raise SpatialInputError("FRAME_MISMATCH", f"observation {ob.evidence_id!r}: frame_id {ob.frame_id!r} != image_id {ob.image_id!r}")
        if ob.image_id not in poses:
            raise SpatialInputError("NO_POSE_FOR_FRAME", f"no pose for image {ob.image_id!r} (observation {ob.evidence_id!r})")
        obs.append(ob)
    ids = [o.evidence_id for o in obs]
    if len(set(ids)) != len(ids):
        dup = [k for k, c in Counter(ids).items() if c > 1][:3]
        raise SpatialInputError("DUPLICATE_EVIDENCE_ID", f"duplicate evidence ids {dup}")
    return obs, poses, dict(d.get("storeys") or {}), d["scale_state"]


@dataclass
class SpatialStageOutput:
    status: str                       # succeeded | degraded | failed | refused | unchanged | insufficient
    facts: dict[str, Any]
    world: Any = None
    version_id: str | None = None
    errors: list[dict] = field(default_factory=list)


def latest_input_hash(store, parent: str | None) -> str | None:
    if store is None or parent is None:
        return None
    return (store.load_version(parent).metadata or {}).get(HASH_KEY)


def run_spatial_global(inputs: dict, *, store=None, parent: str | None = None, solver: str = "auto",
                       source_session_ids: list[str] | None = None) -> SpatialStageOutput:
    import time as _t
    _T0 = _t.perf_counter()
    h = input_hash(inputs) if isinstance(inputs, dict) else None
    base = {"schema": SCHEMA, "input_hash": h}
    try:
        obs, poses, storeys, scale = parse_inputs(inputs)
    except SpatialInputError as e:
        return SpatialStageOutput("failed", {**base, "status": "failed", "error": {"code": e.code, "message": e.message}},
                                  errors=[{"code": e.code, "message": e.message}])
    if scale != "metric":
        msg = f"scale_state={scale!r}: metric scale is never assumed, so metre-valued optimization was not run"
        return SpatialStageOutput("refused", {**base, "status": "refused", "error": {"code": "SCALE_NOT_METRIC", "message": msg}},
                                  errors=[{"code": "SCALE_NOT_METRIC", "message": msg}])
    # user exclusions persisted on the parent version ("this observation / this entity's evidence is wrong"): applied before optimisation,
    # the excluded observations stay in the world's record, and the input hash includes them so adding one triggers reprocessing
    pw = store.load_version(parent) if (store is not None and parent is not None) else None
    excl_records = list((pw.metadata or {}).get("spatial_exclusions") or []) if pw is not None else []
    excluded_obs: list = []
    excl_report = {"applied": [], "unresolved": []}
    if excl_records:
        ids = {o.evidence_id for o in obs}
        dropped: set = set()
        for r in excl_records:
            ev = {r["evidence_id"]} if r.get("evidence_id") else (
                set((pw.entities[r["entity_id"]].custom_properties or {}).get("evidence_ids") or ())
                if r.get("entity_id") in pw.entities else set())
            hit = ev & ids
            excl_report["applied" if hit else "unresolved"].append({"record": r, "observations_dropped": len(hit)})
            dropped |= hit
        excluded_obs = [o.to_dict() for o in obs if o.evidence_id in dropped]
        obs = [o for o in obs if o.evidence_id not in dropped]
        h = input_hash({"inputs": h, "exclusions": excl_records})
        base["input_hash"] = h
    walls_seen = [o for o in obs if o.kind is Kind.WALL]
    if not obs:
        return SpatialStageOutput("insufficient", {**base, "status": "insufficient", "code": "NO_EVIDENCE",
                                                   "reason": "the producer supplied poses but no observations: nothing to reconstruct (no geometry fabricated)"})
    if len({o.image_id for o in walls_seen}) < 2:
        return SpatialStageOutput("insufficient", {**base, "status": "insufficient", "code": "INSUFFICIENT_OBSERVATIONS", "n_observations": len(obs),
                                                   "n_images_with_walls": len({o.image_id for o in walls_seen}),
                                                   "reason": "fewer than 2 views of structure: no cross-view constraint exists, so no joint optimization or canonical geometry is produced"})
    if latest_input_hash(store, parent) == h:
        return SpatialStageOutput("unchanged", {**base, "status": "unchanged", "version_id": parent, "reason": "identical input hash"},
                                  version_id=parent)
    _T1 = _t.perf_counter()
    res, mv = compile_multiview(obs, poses, storeys=storeys, mv_cfg=MultiViewConfig(solver=solver))
    _T2 = _t.perf_counter()
    gv = classify_reconstruction(res.entities, res.topology, res.constraints.junctions, optimization=mv)
    warnings: list[str] = []
    lengths = sorted(e.length for e in res.entities if e.kind is Kind.WALL and len(e.image_ids) >= 2)
    if lengths and not (0.3 <= lengths[len(lengths) // 2] <= 100.0):
        warnings.append(f"median wall length {lengths[len(lengths) // 2]:.2f} m is implausible for a metric claim: check the scale")
    if not mv.converged:
        warnings.append("optimizer did not converge")
    if mv.outliers:
        warnings.append(f"{len(mv.outliers)} observation(s) rejected as multi-view outliers (kept in the record, see facts.outliers)")
    comp = res.compiled_entities()
    statuses = {d: dict(Counter(e.beliefs[d].status.value for e in comp)) for d in DIMENSIONS}
    id_map: dict[str, str] = {}
    ident = None
    if pw is not None:
        ident = identity_report(res, pw, previous_history=(pw.metadata or {}).get("spatial_identity", {}).get("history"))
        id_map = ident.mapping
    world = to_worldir(res, name="spatial-global", id_map=id_map, frame=inputs.get("frame") or "world", scale_state=scale)
    attach_optimization(world, mv)
    world.metadata["scale"] = {"state": "metric", "basis": "spatial_inputs.v1 scale_state=metric (declared by the producer)"}
    carry = {}
    if pw is not None:                       # user corrections and decisions outrank automatic results and survive reprocessing
        from world_ir.corrections import carry_corrections, carry_decisions
        try:
            if (pw.metadata or {}).get("user_corrections"):
                carry["corrections"] = carry_corrections(world, pw.metadata["user_corrections"])
            if (pw.metadata or {}).get("user_decisions"):
                carry["decisions"] = carry_decisions(world, pw.metadata["user_decisions"])
        except Exception as exc:  # noqa: BLE001 - the rebuilt world stands; the failure is visible in the facts
            carry["error"] = f"{type(exc).__name__}: {exc}"
    if excl_records:
        world.metadata["spatial_exclusions"] = excl_records
        world.metadata["spatial_excluded_evidence"] = excluded_obs      # original observations are never lost
    facts = {
        **base, "status": "degraded" if warnings else "succeeded", "scale_state": scale,
        "n_images": len(poses), "n_observations": len(obs),
        "optimization": {"solver": mv.solver, "cost_before": mv.cost_before, "cost_after": mv.cost_after, "nfev": mv.nfev,
                         "rounds": mv.rounds, "converged": mv.converged, "seconds": mv.seconds, "jac_seconds": mv.jac_seconds,
                         "n_params": mv.n_params, "n_residuals": mv.n_residuals, "n_clusters": mv.n_clusters},
        "outliers": [{"evidence_id": ev, "max_abs_z": mv.residual_z.get(ev)} for ev in mv.outliers],
        "validation": gv.to_dict(), "topology_status": res.topology.status(),
        "completeness": res.completeness.to_dict(),
        "entity_epistemic_status": statuses,
        "identity_carried": len(id_map),
        "user_exclusions": excl_report if excl_records else None,
        "user_carry": {k: {"applied": len(v.get("applied", [])), "skipped": len(v.get("skipped", []))} if isinstance(v, dict) else v for k, v in carry.items()} or None,
        "sigma_budget_entities": sum(1 for e in comp if e.sigma_parts),
        "tentative_excluded": [{"entity_id": e.entity_id, "evidence_ids": list(e.evidence_ids), "views": len(e.image_ids)}
                               for e in res.entities if e not in comp],
        "conflicts": [c.to_dict() for c in res.association.conflicts] + [c.to_dict() for c in res.constraints.contradictions],
        "warnings": warnings,
        "quality": _quality(res, comp, gv, mv, len(obs)),
        "association_cache": {**mv.assoc_cache} if getattr(mv, "assoc_cache", None) else None,
    }
    _T3 = _t.perf_counter()
    world.metadata[HASH_KEY] = h
    if ident is not None:
        world.metadata["spatial_identity"] = ident.to_dict()
        facts["identity"] = dict(Counter(d["decision"] for d in ident.decisions))
    world.metadata["spatial_facts"] = json.loads(json.dumps(facts, default=str))
    vid = None
    if store is not None:
        vid = store.save_version(world, parent=parent, source_session_ids=source_session_ids).version_id
        facts["version_id"] = vid
    facts["timings_s"] = {"validate_inputs": _T1 - _T0, "optimize_and_compile": _T2 - _T1, "to_worldir_and_identity": _T3 - _T2,
                          "persist": _t.perf_counter() - _T3, "optimizer_total": mv.seconds, "jacobian_assembly": mv.jac_seconds,
                          "final_association": res.timings_s.get("association"),
                          "constraints_topology_completeness": res.timings_s.get("constraints_topology_completeness"),
                          "peak_memory": "not measured per job (see benchmarks/results/spatial_solver_benchmark*.json)"}
    return SpatialStageOutput(facts["status"], facts, world, vid)


def _quality(res, comp, gv, mv, n_obs: int) -> dict:
    """Quality as SEPARATE axes. No aggregate score. Accuracy against ground truth is not evaluable inside a job (no truth exists), which
    is stated, not defaulted: only internally measurable quantities are reported here."""
    import numpy as np

    from .constraints import CKind
    from .model import Kind as K
    walls = [e for e in comp if e.kind is K.WALL]
    sig = [e.beliefs["position"].sigma for e in walls if e.beliefs["position"].sigma is not None]
    by = lambda kind: dict(Counter(c.status.value for c in res.constraints.constraints if c.kind is kind))  # noqa: E731
    top = res.completeness.topological
    direct = sum(1 for e in comp for b in e.beliefs.values() if b.status.value == "directly_observed")
    inferred = sum(1 for e in comp for b in e.beliefs.values() if b.status.value in ("geometrically_inferred", "semantically_inferred"))
    q = lambda v, p: float(np.quantile(v, p)) if v else None  # noqa: E731
    return {
        "geometric": {"accuracy_vs_ground_truth": "not evaluable in a job: no ground truth", "position_sigma_m": {"p50": q(sig, .5), "p90": q(sig, .9), "max": max(sig) if sig else None},
                      "walls_with_lower_bound_extent": len(res.completeness.surface_completeness["walls_with_lower_bound_extent"])},
        "structural": {"wall_continuity": by(CKind.GEOMETRIC_CONTINUITY), "opening_attachment": by(CKind.OPENING_ATTACHMENT),
                       "opening_dimensions": by(CKind.OPENING_DIMENSIONS), "rooms_enclosed": top["rooms_enclosed"],
                       "junction_fraction": top["junction_fraction"], "wall_components": len(gv.components)},
        "evidence": {"observations": n_obs, "observation_use_fraction": res.completeness.evidence["observation_use_fraction"],
                     "entities": len(comp), "min_views_per_entity": min((len(e.image_ids) for e in comp), default=None),
                     "directly_observed_beliefs": direct, "inferred_beliefs": inferred,
                     "entities_with_conflicting_beliefs": sum(1 for e in comp if any(b.status.value == "conflicting" for b in e.beliefs.values()))},
        "topological": {"structural_errors": gv.checks["structural_errors"], "contradictions": gv.checks["contradictions"],
                        "incomplete_issues": gv.checks["incomplete_issues"], "state": gv.state.value},
        "processing": {"solver": mv.solver, "converged": mv.converged, "rounds": mv.rounds, "nfev": mv.nfev, "seconds": mv.seconds,
                       "diagnostics": mv.diagnostics},
    }
