"""Canonical entities with provenance back to raw observations, and ``explain()``.

A canonical entity is NOT a new source of truth: it is a *view* over associated raw observations. Every value it carries is the
output of ``consensus.resolve_estimates`` (correlation-aware, no voting, no averaging of incompatible data) and every value keeps the
observation ids that support it, the ones that were excluded and why, and what is still uncertain.

Observation convention (documented, not assumed): ``predicted["value"]`` is a list of floats; ``uncertainty["covariance"]`` is an
optional matrix; the ``dimensions`` quantity is ``[w, d, h]`` in metres, the axis names ``config/ontology.yaml`` uses.

``explain`` answers the three questions the programme asks of every entity: which observations support its position, which
measurements determine its dimensions, and what remains uncertain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.multiengine.alignment import FrameGraph, FrameMismatch
from engine.multiengine.association import AssociatedEntity, AssociationResult, Decision
from engine.multiengine.consensus import (
    ConsensusError,
    Estimate,
    LabelClaim,
    reconcile_labels,
    resolve_estimates,
)
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.registry import EngineRegistry, RegistryError

POSITION = "position"
DIMENSIONS = "dimensions"
DIM_AXES = ("w", "d", "h")


@dataclass(frozen=True)
class Determination:
    quantity: str
    status: str  # "resolved" | "conflict" | "insufficient" | "unmeasured" | "not_determinable"
    value: tuple[float, ...] | None
    frame: str
    method: str = ""
    supporting: tuple[str, ...] = ()  # observation ids behind the value
    excluded: tuple[tuple[str, str], ...] = ()  # (observation id, why it was not used) -- raw rows are never dropped
    verification: str = ""
    uncertainty: Mapping[str, Any] | None = None
    independence: str = ""
    notes: tuple[str, ...] = ()
    # the competing independent estimates when status == "conflict": kept, never merged or discarded
    candidates: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class LabelDetermination:
    """A categorical attribute (e.g. material) reconciled across engines WITHOUT voting; competing labels are all kept."""

    name: str
    status: str  # LabelResult.status: agreed_unverified | resolved_by_evidence | conflict
    label: str | None
    candidates: tuple[str, ...]
    supporting: tuple[str, ...]  # observation ids behind the label claims
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalEntity:
    entity_id: str
    ontology_path: str
    ontology_version: str
    class_status: str  # LabelResult.status, or "unlabelled"
    class_candidates: tuple[str, ...]
    members: tuple[str, ...]
    determinations: Mapping[str, Determination]
    decisions: tuple[Decision, ...] = ()  # association decisions that formed (or refused to extend) this entity
    notes: tuple[str, ...] = field(default_factory=tuple)
    parent_entity_id: str | None = None  # set when this entity is a PART of another canonical entity
    attributes: Mapping[str, LabelDetermination] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ building
def _group_of(registry: EngineRegistry | None, engine_id: str) -> str:
    if registry is None:
        return engine_id
    try:
        return registry.get(engine_id).family or engine_id
    except RegistryError:
        return engine_id


def _path_scale(graph: FrameGraph, src: str, dst: str) -> float:
    pts = graph.transform_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), src, dst)
    return float(np.linalg.norm(pts[1] - pts[0]))


def determine(
    quantity: str,
    member_ids: Sequence[str],
    matrix: EvidenceMatrix,
    *,
    target_frame: str,
    tolerance: float | None = None,
    registry: EngineRegistry | None = None,
    graph: FrameGraph | None = None,
    assume_independent_groups: bool = False,
) -> Determination:
    """Resolve one quantity of one entity from its member observations; every excluded observation is listed with its reason."""
    estimates: list[Estimate] = []
    excluded: list[tuple[str, str]] = []
    notes: list[str] = []
    for oid in member_ids:
        o = matrix.store.get(oid)
        if o.quantity != quantity:
            continue
        if matrix.status(oid) == "rejected":
            excluded.append((oid, "rejected in the evidence matrix"))
            continue
        value = o.predicted.get("value")
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, (int, float)) for v in value):
            excluded.append((oid, "predicted['value'] is not a numeric vector"))
            continue
        vec = np.asarray(value, dtype=float)
        cov = (o.uncertainty or {}).get("covariance")
        cov_arr = None if cov is None else np.asarray(cov, dtype=float)
        if o.coordinate_system != target_frame:
            if graph is None:
                excluded.append(
                    (oid, f"frame {o.coordinate_system!r} differs from {target_frame!r} and no frame graph was given")
                )
                continue
            try:
                if quantity == POSITION:
                    graph.require_comparable(o.coordinate_system, target_frame)
                    vec = graph.transform_points(vec, o.coordinate_system, target_frame)[0]
                else:  # lengths: need a metric path, and scale by the path's scale
                    graph.require_comparable(o.coordinate_system, target_frame, metric=True)
                    vec = vec * _path_scale(graph, o.coordinate_system, target_frame)
            except FrameMismatch as exc:
                excluded.append((oid, str(exc)))
                continue
            if cov_arr is not None:
                cov_arr = None
                notes.append(f"covariance of {oid} dropped: it is not propagated through a frame transform")
        estimates.append(
            Estimate(
                engine_id=o.engine_id,
                observation_id=oid,
                group=_group_of(registry, o.engine_id),
                frame=target_frame,
                value=tuple(float(x) for x in vec),
                covariance=None if cov_arr is None else tuple(tuple(float(x) for x in r) for r in cov_arr),
            )
        )
    if not estimates:
        return Determination(
            quantity, "unmeasured", None, target_frame, excluded=tuple(excluded),
            notes=tuple(notes) + (f"no usable observation of {quantity!r} among the entity's {len(member_ids)} members",),
        )
    try:
        res = resolve_estimates(estimates, tolerance=tolerance, assume_independent_groups=assume_independent_groups)
    except ConsensusError as exc:
        return Determination(
            quantity, "not_determinable", None, target_frame, supporting=(), excluded=tuple(excluded),
            notes=tuple(notes) + (f"consensus refused: {exc}",),
        )
    excluded.extend((oid, f"outlier: {o.reason}") for o in res.outliers for oid in o.observation_ids)
    return Determination(
        quantity=quantity,
        status=res.status,
        value=None if res.value is None else tuple(float(x) for x in res.value),
        frame=target_frame,
        method=res.method,
        supporting=tuple(res.supporting),
        excluded=tuple(excluded),
        verification=res.verification,
        uncertainty=res.uncertainty,
        independence=res.independence,
        notes=tuple(notes) + tuple(res.notes),
        candidates=tuple(
            {"group": g.group, "value": [float(x) for x in g.value], "observation_ids": list(g.members)}
            for g in res.groups
        )
        if res.status == "conflict"
        else (),
    )


def build_canonical_entity(
    entity: AssociatedEntity,
    matrix: EvidenceMatrix,
    *,
    target_frame: str,
    tolerances: Mapping[str, float] | None = None,
    labels: Mapping[str, str] | None = None,
    registry: EngineRegistry | None = None,
    graph: FrameGraph | None = None,
    ontology: Any | None = None,
    association: AssociationResult | None = None,
    quantities: Sequence[str] = (POSITION, DIMENSIONS),
    assume_independent_groups: bool = False,
    parent_entity_id: str | None = None,
    attribute_quantities: Sequence[str] = ("material",),
) -> CanonicalEntity:
    from perception.ontology import FALLBACK_PATH, get_default_ontology

    onto = ontology or get_default_ontology()
    tolerances = tolerances or {}
    class_status, class_candidates, path = "unlabelled", (), FALLBACK_PATH
    notes: list[str] = []
    if labels:
        claims = [
            LabelClaim(
                matrix.store.get(oid).engine_id,
                oid,
                _group_of(registry, matrix.store.get(oid).engine_id),
                onto.resolve_or_unknown(labels[oid]).path,
            )
            for oid in entity.members
            if oid in labels
        ]
        if claims:
            lr = reconcile_labels(claims)
            class_status, class_candidates = lr.status, lr.candidates
            if lr.label is not None:
                path = lr.label
            else:
                notes.append(f"class unresolved: sources disagree between {list(lr.candidates)}; all candidates are kept")
    dets = {
        q: determine(
            q, entity.members, matrix, target_frame=target_frame, tolerance=tolerances.get(q),
            registry=registry, graph=graph, assume_independent_groups=assume_independent_groups,
        )
        for q in quantities
    }
    attrs: dict[str, LabelDetermination] = {}
    for name in attribute_quantities:
        claims, supporting = [], []
        for oid in entity.members:
            o = matrix.store.get(oid)
            lab = o.predicted.get("label") if o.quantity == name else None
            if isinstance(lab, str) and lab and matrix.status(oid) != "rejected":
                claims.append(LabelClaim(o.engine_id, oid, _group_of(registry, o.engine_id), lab.strip().lower()))
                supporting.append(oid)
        if claims:
            lr = reconcile_labels(claims)
            attrs[name] = LabelDetermination(
                name, lr.status, lr.label, lr.candidates, tuple(sorted(supporting)), tuple(lr.notes)
            )
    decisions: tuple[Decision, ...] = ()
    if association is not None:
        seen: dict[tuple[str, str, str, str], Decision] = {}
        for m in entity.members:
            for d in association.decisions_for(m):
                seen.setdefault((d.a, d.b, d.outcome, d.reason), d)
        decisions = tuple(seen.values())
    return CanonicalEntity(
        entity_id=entity.entity_key, ontology_path=path, ontology_version=getattr(onto, "version", ""),
        class_status=class_status, class_candidates=class_candidates, members=entity.members,
        determinations=dets, decisions=decisions, notes=tuple(notes),
        parent_entity_id=parent_entity_id, attributes=attrs,
    )


# ------------------------------------------------------------------------------------------------ explain
def _obs_row(matrix: EvidenceMatrix, oid: str) -> dict[str, Any]:
    o = matrix.store.get(oid)
    c = matrix.context(oid)
    return {
        "observation_id": oid, "engine_id": o.engine_id, "model_version": o.model_version, "source_id": o.source_id,
        "frame": o.coordinate_system, "image_region": list(o.image_region) if o.image_region else None,
        "raw_confidence": o.confidence, "calibrated_confidence": c.calibrated_confidence,
        "fallback": dict(o.fallback) if o.fallback else None, "run_id": c.run_id, "checkpoint_id": c.checkpoint_id,
        "status": matrix.status(oid),
    }


def explain(entity: CanonicalEntity, matrix: EvidenceMatrix, *, ontology: Any | None = None) -> dict[str, Any]:
    """Structured answer to: what supports this entity's position, what determines its dimensions, what is uncertain."""
    from perception.ontology import get_default_ontology

    onto = ontology or get_default_ontology()
    out: dict[str, Any] = {
        "entity_id": entity.entity_id, "ontology_path": entity.ontology_path,
        "class_status": entity.class_status, "members": list(entity.members),
    }
    uncertain: list[str] = []
    for q, d in entity.determinations.items():
        out[q] = {
            "status": d.status, "value": None if d.value is None else list(d.value), "frame": d.frame, "method": d.method,
            "verification": d.verification, "independence": d.independence,
            "supported_by": [_obs_row(matrix, i) for i in d.supporting],
            "excluded": [{"observation_id": i, "reason": r} for i, r in d.excluded],
            "notes": list(d.notes),
        }
        if d.status == "unmeasured":
            uncertain.append(f"{q}: not measured by any usable observation")
        elif d.status in ("conflict", "insufficient", "not_determinable"):
            uncertain.append(f"{q}: {d.status} -- {'; '.join(d.notes) or 'no further detail'}")
        else:
            if d.verification == "unverified":
                uncertain.append(f"{q}: agreement is not verified against the original image evidence")
            if d.uncertainty is None:
                uncertain.append(f"{q}: no uncertainty could be derived from the engines' stated covariances")
            if len(d.supporting) == 1:
                uncertain.append(f"{q}: supported by a single observation ({d.supporting[0]})")
        for oid in d.supporting:
            row = _obs_row(matrix, oid)
            if row["calibrated_confidence"] is None:
                uncertain.append(f"{q}: confidence of {oid} ({row['engine_id']}) is uncalibrated")
            if row["fallback"]:
                uncertain.append(f"{q}: {oid} came from a fallback run ({row['fallback']})")
        if d.excluded:
            uncertain.append(f"{q}: {len(d.excluded)} observation(s) were excluded; see 'excluded' for each reason")
    dims = entity.determinations.get(DIMENSIONS)
    if dims and dims.status == "resolved" and dims.value is not None:
        node = onto.get(entity.ontology_path)
        typical = getattr(node, "typical_dims_m", None) or {}
        for axis, v in zip(DIM_AXES, dims.value, strict=False):
            lo_hi = typical.get(axis)
            if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
                uncertain.append(
                    f"dimensions.{axis}={v:.3f} m is outside the ontology's typical range {list(lo_hi)} for "
                    f"{entity.ontology_path} (a prior only; the measurement is kept)"
                )
    if entity.class_status == "agreed_unverified":
        uncertain.append("class: engines agree, but agreement is not verified against the image evidence")
    elif entity.class_status in ("conflict", "unlabelled"):
        uncertain.append(f"class: {entity.class_status}; candidates {list(entity.class_candidates)}")
    uncertain.extend(entity.notes)
    refused = [d for d in entity.decisions if d.outcome == "refused" and d.reason != "already_same_entity"]
    if refused:
        uncertain.append(f"identity: {len(refused)} candidate merge(s) were refused ({sorted({d.reason for d in refused})})")
    out["uncertain"] = list(dict.fromkeys(uncertain))
    return out


def render_explanation(exp: Mapping[str, Any]) -> str:
    lines = [
        f"{exp['entity_id']}: {exp['ontology_path']} (class {exp['class_status']}), {len(exp['members'])} observation(s)"
    ]
    for q in (POSITION, DIMENSIONS):
        block = exp.get(q)
        if not block:
            continue
        val = "n/a" if block["value"] is None else ", ".join(f"{x:.3f}" for x in block["value"])
        lines.append(f"{q}: {block['status']} [{val}] in {block['frame']} via {block['method'] or 'n/a'}")
        for r in block["supported_by"]:
            lines.append(f"  supported by {r['observation_id']} ({r['engine_id']} {r['model_version']}, image {r['source_id']})")
        for e in block["excluded"]:
            lines.append(f"  excluded {e['observation_id']}: {e['reason']}")
    lines.append("uncertain:")
    if exp["uncertain"]:
        lines.extend(f"  - {u}" for u in exp["uncertain"])
    else:
        lines.append("  (nothing flagged)")
    return "\n".join(lines)
