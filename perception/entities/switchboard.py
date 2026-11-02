"""A switchboard is a faceplate carrying individually identified modules -- never one generic box.

Input is what was OBSERVED (plate pose + the modules seen on it, from the small-object pass and multi-view association).
Output is canonical WorldIR entities: the plate, one entity per module, ``part_of`` / ``attached_to`` relationships, the
module count and arrangement, orientation, host-wall relationship and per-entity evidence. Rules:

  * unknown stays unknown: no plate size => the dimension is ``None`` and state ``UNKNOWN`` (a typical size is never
    substituted); a module list that is not declared complete is a LOWER BOUND, not "the" count;
  * a module outside the plate, or of an implausible size for its class, is reported, not silently kept;
  * dimensions are RECONSTRUCTED from photographs; they are in metres only when the world scale is metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.entities import taxonomy
from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Observation, Relationship, RelationshipKind

Vec = Tuple[float, float, float]


@dataclass(frozen=True)
class ModuleObservation:
    class_id: str                                   # e.g. electrical.switchboard.module.socket
    center: Vec                                     # world position of the module face centre
    size_m: Optional[Tuple[float, float]] = None    # (width, height) on the plate, when measured
    confidence: float = 0.5
    evidence_ids: Tuple[str, ...] = ()
    attributes: Dict[str, object] = field(default_factory=dict)   # e.g. {"earth_pin": "unknown", "gang": 1}


@dataclass(frozen=True)
class PlateObservation:
    center: Vec
    normal: Vec                                     # points OUT of the wall, toward the room
    up: Vec = (0.0, 1.0, 0.0)
    size_m: Optional[Tuple[float, float]] = None    # (width, height); None = not measured
    depth_m: Optional[float] = None                 # protrusion from the wall; None = not measured
    confidence: float = 0.5
    evidence_ids: Tuple[str, ...] = ()
    host_wall_id: Optional[str] = None


@dataclass
class SwitchboardBuild:
    entities: List[Entity]
    rejected: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def plate(self) -> Entity:
        return self.entities[0]


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("zero-length axis")
    return a / n


def _dim(value: Optional[float], metric: bool, basis: str) -> dict:
    if value is None:
        return {"value": None, "unit": "m" if metric else "model_units", "state": "UNKNOWN", "basis": "not measured"}
    return {"value": float(value), "unit": "m" if metric else "model_units", "state": "RECONSTRUCTED", "basis": basis}


def _observations(ids: Sequence[str], conf: float) -> List[Observation]:
    return [Observation(id=f"obs-{e}", sensor_type="camera", frame_id=e, confidence=float(conf)) for e in ids]


def _classes(class_id: str) -> List[str]:
    return [class_id, *taxonomy.ancestors(class_id)]


def build_switchboard(plate: PlateObservation, modules: Sequence[ModuleObservation], *, entity_id: str,
                      metric: bool = False, complete: bool = False, tolerance_m: float = 0.01) -> SwitchboardBuild:
    """``complete`` = the caller verified every module on the plate was seen; otherwise the count is a lower bound."""
    n, u = _unit(plate.normal), _unit(plate.up)
    right = _unit(np.cross(u, n))
    u = np.cross(n, right)                              # re-orthogonalise 'up' against the normal
    origin = np.asarray(plate.center, dtype=float)
    cls = taxonomy.get("electrical.switchboard")
    out = SwitchboardBuild(entities=[])

    def uv(m: ModuleObservation) -> Tuple[float, float]:
        rel = np.asarray(m.center, dtype=float) - origin
        return float(rel @ right), float(rel @ u)

    placed: List[Tuple[str, ModuleObservation, float, float]] = []
    for m in sorted(modules, key=lambda m: (-round(uv(m)[1], 3), round(uv(m)[0], 3))):
        try:
            mc = taxonomy.get(m.class_id)
        except KeyError:
            out.rejected.append({"class_id": m.class_id, "reason": "class is not in the entity taxonomy"})
            continue
        if not m.class_id.startswith("electrical.switchboard.module"):
            out.rejected.append({"class_id": m.class_id, "reason": "not a switchboard module class"})
            continue
        if m.size_m is not None and not taxonomy.size_plausible(m.class_id, max(m.size_m)):
            out.rejected.append({"class_id": m.class_id, "reason": f"size {m.size_m} m implausible for {mc.label}",
                                 "evidence_ids": list(m.evidence_ids)})
            continue
        a, b = uv(m)
        if plate.size_m is not None and (abs(a) > plate.size_m[0] / 2 + tolerance_m
                                         or abs(b) > plate.size_m[1] / 2 + tolerance_m):
            out.warnings.append(f"{m.class_id} at ({a:.3f},{b:.3f}) lies outside the {plate.size_m} plate: kept, flagged")
        placed.append((f"{entity_id}/m{len(placed) + 1:02d}", m, a, b))

    # arrangement: rows top-to-bottom, modules left-to-right within a row (rows merge when within tolerance)
    rows: List[List[str]] = []
    row_v: List[float] = []
    for mid, _, _, b in placed:
        for k, rv in enumerate(row_v):
            if abs(rv - b) <= max(tolerance_m, 0.012):
                rows[k].append(mid)
                break
        else:
            rows.append([mid])
            row_v.append(b)
    col = {mid: a for mid, _, a, _ in placed}
    rows = [sorted(r, key=lambda mid: col[mid]) for r in rows]

    frame = {"origin": [float(x) for x in origin], "right": [float(x) for x in right], "up": [float(x) for x in u],
             "normal": [float(x) for x in n]}
    out.entities.append(Entity(
        id=entity_id, type=EntityType.STRUCTURE, name=cls.label, semantic_labels=_classes(cls.id),
        component_ids=[mid for mid, *_ in placed], provenance=Provenance.RECONSTRUCTED, confidence=float(plate.confidence),
        observations=_observations(plate.evidence_ids, plate.confidence),
        relationships=([Relationship(RelationshipKind.ATTACHED_TO, plate.host_wall_id, confidence=plate.confidence,
                                     provenance=Provenance.INFERRED, metadata={"relation": "mounted_on"})] if plate.host_wall_id else []),
        custom_properties={
            "entity_class": cls.id, "kind": "switchboard", "geometry_strategy": "hybrid",
            "parametric": {"shape": "plate", "frame": frame},
            "dimensions": {"width": _dim(plate.size_m[0] if plate.size_m else None, metric, "photo measurement"),
                           "height": _dim(plate.size_m[1] if plate.size_m else None, metric, "photo measurement"),
                           "protrusion": _dim(plate.depth_m, metric, "photo measurement")},
            "module_count": len(placed), "module_count_state": "complete" if complete else "lower_bound",
            "arrangement": {"rows": rows, "n_rows": len(rows), "columns_per_row": [len(r) for r in rows]},
            "host_wall_id": plate.host_wall_id,
            "evidence": {"source_images": list(plate.evidence_ids), "classification_confidence": plate.confidence},
            "unknown_attributes": [k for k, v in (("size", plate.size_m), ("protrusion", plate.depth_m),
                                                  ("host_wall", plate.host_wall_id)) if v is None],
            "reconstruction_status": "partial" if not complete or plate.size_m is None else "reconstructed",
        }))
    for mid, m, a, b in placed:
        mc = taxonomy.get(m.class_id)
        known = {"width": m.size_m[0] if m.size_m else None, "height": m.size_m[1] if m.size_m else None}
        out.entities.append(Entity(
            id=mid, type=EntityType.STRUCTURE, name=mc.label, semantic_labels=_classes(mc.id),
            provenance=Provenance.RECONSTRUCTED, confidence=float(m.confidence),
            observations=_observations(m.evidence_ids, m.confidence),
            relationships=[Relationship(RelationshipKind.PART_OF, entity_id, confidence=m.confidence, provenance=Provenance.INFERRED,
                                        metadata={"relation": "module_of"})],
            custom_properties={
                "entity_class": mc.id, "kind": mc.id.rsplit(".", 1)[-1], "geometry_strategy": mc.geometry,
                "parametric": {"shape": "module", "frame": frame, "plate_uv": [a, b]},
                "dimensions": {k: _dim(v, metric, "photo measurement") for k, v in known.items()},
                "attributes": dict(m.attributes),
                "evidence": {"source_images": list(m.evidence_ids), "classification_confidence": m.confidence},
                "unknown_attributes": [k for k, v in known.items() if v is None],
                "reconstruction_status": "reconstructed" if all(v is not None for v in known.values()) else "partial",
            }))
    return out
