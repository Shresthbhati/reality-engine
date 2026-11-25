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

from .compiler import compile_multiview, to_worldir
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
    status: str                       # succeeded | degraded | failed | refused | unchanged
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
    if latest_input_hash(store, parent) == h:
        return SpatialStageOutput("unchanged", {**base, "status": "unchanged", "version_id": parent, "reason": "identical input hash"},
                                  version_id=parent)
    res, mv = compile_multiview(obs, poses, storeys=storeys, mv_cfg=MultiViewConfig(solver=solver))
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
    world = to_worldir(res, name="spatial-global")
    attach_optimization(world, mv)
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
        "sigma_budget_entities": sum(1 for e in comp if e.sigma_parts),
        "tentative_excluded": [{"entity_id": e.entity_id, "evidence_ids": list(e.evidence_ids), "views": len(e.image_ids)}
                               for e in res.entities if e not in comp],
        "conflicts": [c.to_dict() for c in res.association.conflicts] + [c.to_dict() for c in res.constraints.contradictions],
        "warnings": warnings,
    }
    world.metadata[HASH_KEY] = h
    world.metadata["spatial_facts"] = json.loads(json.dumps(facts, default=str))
    vid = None
    if store is not None:
        vid = store.save_version(world, parent=parent, source_session_ids=source_session_ids).version_id
        facts["version_id"] = vid
    return SpatialStageOutput(facts["status"], facts, world, vid)
