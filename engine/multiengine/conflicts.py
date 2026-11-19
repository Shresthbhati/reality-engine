"""Geometric and semantic conflict detection across canonical entities.

``consensus`` resolves disagreement about ONE quantity of ONE entity; ``association`` refuses implausible merges. This module
finds disagreement BETWEEN entities that no single-quantity check can see:

* interpenetration: two resolved boxes overlap by more than a stated tolerance and the ontology does not declare the contact
  legitimate (a lamp hosted by a table, a chair resting on a floor);
* semantic: association refused a merge because the labels were incompatible (reported, not resolved).

Nothing is repaired or removed here. A conflict is a record; ``record_conflicts`` links the supporting observations with
``conflicts_with`` in the evidence store so the disagreement stays visible next to the raw evidence.

Limits stated up front: boxes are AXIS-ALIGNED and centred on the resolved position (rotation is not modelled), so a rotated
object can be reported as larger than it is. ``tolerance_m`` has NO default: how much overlap is measurement noise is a benchmark
decision, not a constant to guess.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.association import AssociationResult
from engine.multiengine.canonical import DIMENSIONS, POSITION, CanonicalEntity
from engine.multiengine.evidence_store import EvidenceStore

ContactAllowed = Callable[["Box", "Box"], bool]


@dataclass(frozen=True)
class Box:
    entity_id: str
    ontology_path: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass(frozen=True)
class Conflict:
    kind: str  # "interpenetration" | "semantic_label"
    entity_ids: tuple[str, ...]
    detail: str
    magnitude: float | None = None  # overlap depth in metres for interpenetration
    basis: Mapping[str, Any] = field(default_factory=dict)


def box_from_entity(ce: CanonicalEntity) -> Box | None:
    """Axis-aligned box centred on the resolved position with the resolved [w, d, h] as x/y/z extents; None unless BOTH are resolved."""
    pos, dims = ce.determinations.get(POSITION), ce.determinations.get(DIMENSIONS)
    if not pos or not dims or pos.status != "resolved" or dims.status != "resolved":
        return None
    if pos.value is None or dims.value is None or len(pos.value) != 3 or len(dims.value) != 3:
        return None
    if pos.frame != dims.frame:
        return None
    lo = tuple(p - d / 2 for p, d in zip(pos.value, dims.value, strict=True))
    hi = tuple(p + d / 2 for p, d in zip(pos.value, dims.value, strict=True))
    return Box(ce.entity_id, ce.ontology_path, lo, hi)  # type: ignore[arg-type]


def ontology_contact_allowed(ontology: Any | None = None) -> ContactAllowed:
    """Overlap/contact is legitimate when either class declares the other as a host or attachment target
    (``hosts`` / ``attach`` in config/ontology.yaml), including via an ancestor class."""
    from perception.ontology import get_default_ontology

    onto = ontology or get_default_ontology()

    def lineage(path: str) -> set[str]:
        return {path, *onto.ancestors(path)}

    def declares(a: Box, b: Box) -> bool:
        node = onto.get(a.ontology_path)
        if node is None:
            return False
        targets = set(node.hosts) | {v for att in node.attach for v in att.values()}
        return bool(targets & lineage(b.ontology_path))

    return lambda a, b: declares(a, b) or declares(b, a)


def find_overlaps(
    boxes: Sequence[Box], *, tolerance_m: float, contact_allowed: ContactAllowed | None = None
) -> list[Conflict]:
    if tolerance_m < 0:
        raise ValueError("tolerance_m must be >= 0")
    out: list[Conflict] = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            depths = [min(a.hi[k], b.hi[k]) - max(a.lo[k], b.lo[k]) for k in range(3)]
            if min(depths) <= 0:
                continue
            depth = min(depths)  # the smallest axis overlap: the minimum translation that would separate them
            if depth <= tolerance_m:
                continue
            if contact_allowed is not None and contact_allowed(a, b):
                continue
            out.append(
                Conflict(
                    "interpenetration", (a.entity_id, b.entity_id),
                    f"{a.ontology_path} and {b.ontology_path} overlap by {depth:.3f} m (> tolerance {tolerance_m} m) "
                    "and the ontology does not declare this contact",
                    depth, {"model": "axis-aligned boxes at resolved position/dimensions; rotation not modelled"},
                )
            )
    return out


def semantic_conflicts(result: AssociationResult) -> list[Conflict]:
    return [
        Conflict("semantic_label", (d.a, d.b), f"association refused merging {d.a} and {d.b}: incompatible labels",
                 d.score, dict(d.basis))
        for d in result.refused("label_conflict")
    ]


def record_conflicts(
    store: EvidenceStore, conflicts: Sequence[Conflict], supporting: Mapping[str, Sequence[str]]
) -> int:
    """Link, with ``conflicts_with``, the first supporting observation of each side of an interpenetration; semantic conflicts
    already name observation ids. Returns how many new relations were written."""
    n = 0
    for c in conflicts:
        if c.kind == "semantic_label":
            a, b = c.entity_ids
        else:
            sa, sb = (supporting.get(e) for e in c.entity_ids)
            if not sa or not sb:
                continue
            a, b = sa[0], sb[0]
        n += int(store.link(a, b, "conflicts_with"))
    return n
