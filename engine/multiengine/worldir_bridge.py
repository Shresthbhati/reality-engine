"""Canonical entity -> WorldIR bridge.

Converts ``CanonicalEntity`` (a consensus view over raw observations) into WorldIR ``Entity`` / ``Geometry`` / ``Material`` records
without inventing anything:

* geometry exists only when BOTH position and dimensions were resolved (an axis-aligned box; orientation is not modelled);
* a material exists only when the material label was resolved; competing labels / competing estimates stay in the evidence block;
* confidence is NEVER defaulted into a measured-looking value: ``Entity.confidence`` is ``None`` (the agreed Optional scalar) with no
  ``confidence_basis``, the per-dimension ``confidence_vector`` stays ``None``, and the raw model confidence is recorded SEPARATELY on
  each referenced observation next to the calibrated one (or ``None``). Geometry/Material/Relationship/Observation confidences are
  not Optional in WorldIR, so those keep WorldIR's documented 0.5 "not measured" sentinel (``NOT_MEASURED``) and say so in metadata;
* a part is its own entity linked ``PART_OF`` its whole; ontology-declared parts that were not observed are listed as "not observed"
  -- their absence is never claimed;
* ontology ``attach``/``hosts`` rules are PRIORS and produce no relationship; only an evidence-backed ``SpatialRelation`` does.

Deterministic: every id derives from the canonical entity id, collections are sorted, nothing uses a random id or the clock, so the
same input converts to byte-identical output. Reversible where practical: ``canonical_summary`` reads the evidence block back and
``trace`` re-checks every referenced observation against the Evidence Matrix.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.canonical import DIMENSIONS, POSITION, CanonicalEntity, Determination
from engine.multiengine.evidence_matrix import EvidenceMatrix
from provenance import Provenance, Uncertainty
from world_ir.schema_v1 import (
    ConfidenceVector,
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Observation,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.statement_state import StatementState
from world_ir.v2_extensions import ExternalReference
from world_ir.world_v1 import WorldIR

BRIDGE_VERSION = "evidence-world-bridge/1"
NOT_MEASURED: float | None = None  # None represents unmeasured confidence; never a fabricated 0.5 sentinel
_UNCERTAINTY_NOTE = (
    "confidence not measured: no calibration dataset exists for these engines; raw model confidences are recorded "
    "per observation, uncalibrated"
)


class BridgeError(ValueError):
    pass


@dataclass(frozen=True)
class SpatialRelation:
    """An EVIDENCE-backed relation between two canonical entities (e.g. support found from contact of resolved geometry)."""

    source_entity_id: str
    kind: RelationshipKind
    target_entity_id: str
    basis: str  # what established it; required, an unexplained relation is not accepted
    observation_ids: tuple[str, ...] = ()


@dataclass
class BridgeResult:
    entities: dict[str, Entity] = field(default_factory=dict)
    geometries: dict[str, Geometry] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def apply_to(self, world: WorldIR) -> None:
        """Merge into a world. An id that already exists with DIFFERENT content is refused: nothing is silently overwritten."""
        pairs = (
            (world.entities, self.entities),
            (world.geometries, self.geometries),
            (world.materials, self.materials),
        )
        for table, items in pairs:
            for key, new in items.items():
                old = table.get(key)
                if old is not None and old.to_dict() != new.to_dict():
                    raise BridgeError(f"{key} already exists in the world with different content; refusing to overwrite it")
        for table, items in pairs:
            # copies: the world must not alias this result, or editing the world would silently edit the bridge output
            table.update({k: copy.deepcopy(v) for k, v in items.items()})


def _default(o: Any) -> Any:
    if isinstance(o, (tuple, set, frozenset)):
        return list(o)
    if isinstance(o, Mapping):
        return dict(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=_default))


def _ontology_parts(path: str) -> tuple[str, str | None]:
    segs = [s for s in path.replace(".", "/").split("/") if s]
    return (segs[0] if segs else "unknown"), (segs[1] if len(segs) > 2 else None)


def _entity_type(node: Any, group: str) -> EntityType:
    """The ontology's declared ``world_type`` first; when it does not name a WorldIR type (furniture classes declare 'unknown'),
    the path's top-level group if that IS a WorldIR type (``furniture``); otherwise UNKNOWN -- never a guess."""
    for candidate in ((str(node.world_type),) if node is not None else ()) + (group,):
        try:
            t = EntityType(candidate)
        except ValueError:
            continue
        if t is not EntityType.UNKNOWN:
            return t
    return EntityType.UNKNOWN


def _det_block(d: Determination) -> dict[str, Any]:
    return {
        "status": d.status, "value": None if d.value is None else list(d.value), "frame": d.frame, "method": d.method,
        "verification": d.verification, "supporting": sorted(d.supporting),
        "excluded": [{"observation_id": i, "reason": r} for i, r in sorted(d.excluded)],
        "uncertainty": d.uncertainty, "independence": d.independence, "notes": list(d.notes),
        "candidates": [dict(c) for c in d.candidates],
    }


def _world_observation(matrix: EvidenceMatrix, oid: str, roles: list[str]) -> Observation:
    row = matrix.row(oid)
    o = row["observation"]
    ctx = row["context"]
    return Observation(
        id=oid,
        sensor_type="camera",
        timestamp=0.0,  # the capture time is not in the evidence record; 0.0 is WorldIR's "unset", not a time
        frame_id=o["coordinate_system"],
        data_uri=o["source_id"],
        data_hash="",  # recorded-never-computed: the source image hash is not known to the matrix
        metadata=_jsonable({
            "engine_id": o["engine_id"], "model_version": o["model_version"], "task": o["task"], "quantity": o["quantity"],
            "image_region": o["image_region"], "run_id": ctx["run_id"], "architecture_id": ctx["architecture_id"],
            "checkpoint_id": ctx["checkpoint_id"], "camera_ref": ctx["camera_ref"],
            "raw_confidence": row["raw_confidence"], "calibrated_confidence": row["calibrated_confidence"],
            "calibration_basis": ctx["calibration_basis"], "calibration": ctx["calibration"],
            "validation": ctx["validation"],
            "matrix_status": row["status"], "fallback": o["fallback"], "roles": sorted(roles),
            "artifact_refs": ctx["artifact_refs"],
        }),
        # the WorldIR field means "measured confidence": only an ESTABLISHED calibrated value may fill it
        confidence=row["calibrated_confidence"] if row["calibrated_confidence"] is not None else NOT_MEASURED,
    )


def convert(
    canonicals: Sequence[CanonicalEntity],
    matrix: EvidenceMatrix,
    *,
    ontology: Any | None = None,
    relations: Sequence[SpatialRelation] = (),
) -> BridgeResult:
    from perception.ontology import get_default_ontology

    onto = ontology or get_default_ontology()
    out = BridgeResult()
    by_id = {c.entity_id: c for c in canonicals}
    if len(by_id) != len(canonicals):
        raise BridgeError("duplicate canonical entity ids in one conversion batch")
    children: dict[str, list[CanonicalEntity]] = {}
    for c in canonicals:
        if c.parent_entity_id:
            children.setdefault(c.parent_entity_id, []).append(c)

    for ce in sorted(canonicals, key=lambda c: c.entity_id):
        node = onto.get(ce.ontology_path)
        group, sub = _ontology_parts(ce.ontology_path)
        eid = f"ent-{ce.entity_id}"
        pos, dims = ce.determinations.get(POSITION), ce.determinations.get(DIMENSIONS)

        roles: dict[str, list[str]] = {m: [] for m in ce.members}
        for q, d in ce.determinations.items():
            for oid in d.supporting:
                roles.setdefault(oid, []).append(f"supports:{q}")
            for oid, why in d.excluded:
                roles.setdefault(oid, []).append(f"excluded:{q}: {why}")
        for a in ce.attributes.values():
            for oid in a.supporting:
                roles.setdefault(oid, []).append(f"claims:{a.name}")
        observations = [_world_observation(matrix, oid, roles[oid]) for oid in sorted(ce.members)]

        refs: list[ExternalReference] = []
        for src in sorted({o.data_uri for o in observations}):
            refs.append(ExternalReference("image", src, "supporting_evidence", provenance=Provenance.OBSERVED))
        arts = sorted({u for o in observations for u in (o.metadata.get("artifact_refs") or {}).values()})
        for uri in arts:
            refs.append(ExternalReference(
                "model_output", uri, "raw_model_output", checksum="sha256:" + uri.removeprefix("artifact://"),
                provenance=Provenance.ESTIMATED,
            ))

        transform = None
        if pos and pos.status == "resolved" and pos.value is not None and len(pos.value) == 3:
            x, y, z = pos.value
            transform = {"position": {"x": x, "y": y, "z": z}, "frame_id": pos.frame}

        geometry_ids: list[str] = []
        if (
            pos and dims and pos.status == "resolved" and dims.status == "resolved"
            and pos.value is not None and dims.value is not None and len(pos.value) == len(dims.value) == 3
            and pos.frame == dims.frame
        ):
            lo = [p - d / 2 for p, d in zip(pos.value, dims.value, strict=True)]
            hi = [p + d / 2 for p, d in zip(pos.value, dims.value, strict=True)]
            gid = f"geom-{ce.entity_id}"
            out.geometries[gid] = Geometry(
                id=gid, type=GeometryType.BOX, lod_level=0,
                bounds_min=Vector3(*lo), bounds_max=Vector3(*hi), provenance=Provenance.ESTIMATED,
                confidence=NOT_MEASURED,
                quality_metrics=_jsonable({
                    "confidence_state": "not_measured", "frame_id": pos.frame, "bridge": BRIDGE_VERSION,
                    "model": "axis-aligned box centred on the resolved position with the resolved [w, d, h]; "
                             "orientation and component shape are not modelled by this box",
                    "derived_from": {"position": sorted(pos.supporting), "dimensions": sorted(dims.supporting)},
                    # propagated, evidence-derived uncertainty; None = none could be derived (never a stand-in value)
                    "uncertainty": {"position": pos.uncertainty, "dimensions": dims.uncertainty},
                }),
            )
            geometry_ids.append(gid)
        else:
            out.notes.append(
                f"{eid}: no geometry created (position {pos.status if pos else 'absent'}, "
                f"dimensions {dims.status if dims else 'absent'}); none is fabricated"
            )

        material_ids: list[str] = []
        competing: list[dict[str, Any]] = []
        for name, a in sorted(ce.attributes.items()):
            if name == "material" and a.label is not None:
                mid = "mat-" + hashlib.sha256(f"{ce.entity_id}|{a.label}".encode()).hexdigest()[:16]
                out.materials[mid] = Material(
                    id=mid, name=a.label, class_name=a.label, provenance=Provenance.ESTIMATED, confidence=NOT_MEASURED,
                )
                material_ids.append(mid)
            elif a.label is None:
                competing.append({
                    "quantity": name, "candidates": [{"label": c} for c in a.candidates], "supporting": list(a.supporting),
                })
        for q, d in sorted(ce.determinations.items()):
            if d.status == "conflict":
                competing.append({"quantity": q, "candidates": [dict(c) for c in d.candidates]})
        if ce.class_status == "conflict":
            competing.append({"quantity": "class", "candidates": [{"label": c} for c in ce.class_candidates]})

        rels: list[Relationship] = []
        if ce.parent_entity_id:
            parent = by_id.get(ce.parent_entity_id)
            listed = None
            if parent is not None:
                pnode = onto.get(parent.ontology_path)
                pcomps = tuple(pnode.components) if pnode else ()
                resolved = {onto.get(c).path if onto.get(c) else c for c in pcomps}
                listed = ce.ontology_path in resolved or (node.name if node else "") in pcomps
            else:
                out.notes.append(
                    f"{eid}: parent {ce.parent_entity_id} is not in this batch; PART_OF is recorded without a target entity"
                )
            rels.append(Relationship(
                RelationshipKind.PART_OF, f"ent-{ce.parent_entity_id}", confidence=NOT_MEASURED,
                provenance=Provenance.ESTIMATED,
                metadata={"basis": "cross-view association marked this detection as a part of the parent detection",
                          "confidence_state": "not_measured", "listed_in_parent_ontology_components": listed},
            ))
        for r in relations:
            if r.source_entity_id != ce.entity_id:
                continue
            if not r.basis:
                raise BridgeError("a SpatialRelation must state its basis")
            if r.target_entity_id not in by_id:
                raise BridgeError(f"relation target {r.target_entity_id!r} is not a canonical entity in this batch")
            rels.append(Relationship(
                r.kind, f"ent-{r.target_entity_id}", confidence=NOT_MEASURED, provenance=Provenance.ESTIMATED,
                metadata={"basis": r.basis, "observation_ids": sorted(r.observation_ids), "confidence_state": "not_measured"},
            ))
        rels.sort(key=lambda r: (r.kind.value, r.target_id))

        unobserved: list[str] = []
        if node is not None and node.components:
            seen = {c.ontology_path for c in children.get(ce.entity_id, [])}
            seen_leaf = {(onto.get(p).name if onto.get(p) else p) for p in seen}
            for comp in node.components:
                cn = onto.get(comp)
                if (cn.path if cn else comp) not in seen and comp not in seen_leaf:
                    unobserved.append(comp)

        evidence = _jsonable({
            "bridge_version": BRIDGE_VERSION, "canonical_entity_id": ce.entity_id,
            "ontology": {"path": ce.ontology_path, "version": ce.ontology_version, "category": group, "subcategory": sub},
            "class_status": ce.class_status, "class_candidates": list(ce.class_candidates),
            "members": sorted(ce.members),
            "determinations": {q: _det_block(d) for q, d in sorted(ce.determinations.items())},
            "attributes": {
                n: {"status": a.status, "label": a.label, "candidates": list(a.candidates),
                    "supporting": list(a.supporting), "notes": list(a.notes)}
                for n, a in sorted(ce.attributes.items())
            },
            "competing_hypotheses": competing,
            "unobserved_parts": sorted(unobserved),
            "unobserved_parts_note": "listed parts were NOT observed in the evidence; their absence is not claimed"
            if unobserved else "",
            "association_decisions": [
                {"a": d.a, "b": d.b, "score": d.score, "outcome": d.outcome, "reason": d.reason, "basis": dict(d.basis)}
                for d in sorted(ce.decisions, key=lambda d: (d.a, d.b, d.outcome, d.reason))
            ],
            "parent_entity_id": ce.parent_entity_id, "notes": list(ce.notes),
        })
        labels = [ce.ontology_path] + [x for x in (group, sub) if x]
        ce_conf = getattr(ce, "confidence", None)
        conf_basis = getattr(ce, "confidence_basis", None)
        out.entities[eid] = Entity(
            id=eid, type=_entity_type(node, group), name=(node.label if node else ce.ontology_path),
            transform=transform, geometry_ids=geometry_ids, material_ids=material_ids, relationships=rels,
            semantic_labels=list(dict.fromkeys(labels)), observations=observations,
            provenance=Provenance.ESTIMATED, confidence=ce_conf, confidence_basis=conf_basis,
            confidence_vector=ConfidenceVector(),
            uncertainty=Uncertainty(confidence=0.0, note=_UNCERTAINTY_NOTE, calibrated=False),
            custom_properties={"evidence": evidence}, statement_state=StatementState.PREDICTED,
            external_references=tuple(refs),
        )
    return out


# ------------------------------------------------------------------------------------------------ reverse / trace
def canonical_summary(entity: Entity) -> dict[str, Any]:
    """The canonical identity and determinations as recorded in the entity's evidence block (the reversible part)."""
    ev = entity.custom_properties.get("evidence")
    if not isinstance(ev, dict) or ev.get("bridge_version") is None:
        raise BridgeError(f"{entity.id} was not produced by the evidence-world bridge")
    return {
        k: ev[k]
        for k in (
            "canonical_entity_id", "ontology", "class_status", "class_candidates", "members",
            "determinations", "attributes", "parent_entity_id",
        )
    }


def trace(entity: Entity, matrix: EvidenceMatrix) -> list[str]:
    """Every problem found when re-checking the entity against the Evidence Matrix: a missing/changed observation, an image it
    does not cite, an artifact reference that no longer verifies. Empty = fully traceable to the original evidence."""
    problems: list[str] = []
    summary = canonical_summary(entity)
    if sorted(o.id for o in entity.observations) != sorted(summary["members"]):
        problems.append("entity observations do not match the recorded canonical members")
    for o in entity.observations:
        try:
            row = matrix.row(o.id)
        except Exception as exc:  # noqa: BLE001 -- a missing row is the finding
            problems.append(f"observation {o.id}: {exc}")
            continue
        if row["observation"]["source_id"] != o.data_uri:
            problems.append(
                f"observation {o.id}: cites image {o.data_uri!r} but the matrix says {row['observation']['source_id']!r}"
            )
        if o.metadata.get("raw_confidence") != row["raw_confidence"]:
            problems.append(f"observation {o.id}: raw confidence differs from the matrix row")
    cited = {r.uri for r in entity.external_references if r.kind == "image"}
    if cited != {o.data_uri for o in entity.observations}:
        problems.append("external image references do not match the observations' source images")
    for r in entity.external_references:
        if r.kind == "model_output" and not matrix.has_artifact(r.uri):
            problems.append(f"model output {r.uri} is missing from the artifact store")
    problems.extend(f"matrix: {p}" for p in matrix.verify())
    return problems
