"""Universal entity taxonomy (Blueprint 17 / BUILD_SPEC B03).

Thin shim that delegates to the canonical ontology (`config/ontology.yaml` via `perception.ontology`).
Preserves backward-compatible interfaces for `get()`, `all_classes()`, `children()`,
`ancestors()`, `class_for_label()`, and `size_plausible()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from perception.ontology import FALLBACK_PATH, OntologyNode, get_default_ontology

CATEGORIES: Dict[str, str] = {
    "building": "Buildings and structural architecture",
    "room": "Rooms and interior spaces",
    "envelope": "Walls, floors and ceilings",
    "opening": "Doors, windows and openings",
    "structural": "Columns, beams and structural components",
    "circulation": "Staircases, railings and landings",
    "electrical": "Electrical infrastructure",
    "plumbing": "Plumbing and sanitary fixtures",
    "hvac": "HVAC and mechanical systems",
    "furniture": "Furniture and appliances",
    "finish": "Interior finishes and decorative details",
    "exterior": "Exterior building components",
    "road": "Roads, footpaths and kerbs",
    "vegetation": "Trees, plants and landscape features",
    "street": "Lampposts and street furniture",
    "utility": "Utility infrastructure",
    "signage": "Signs, poles and public infrastructure",
    "misc": "Miscellaneous objects and unknown entities",
}

GEOMETRY = ("parametric", "procedural", "mesh", "pointcloud", "volumetric", "hybrid")


@dataclass(frozen=True)
class EntityClass:
    id: str
    label: str
    category: str
    parent: Optional[str]
    world_type: str                      # an existing world_ir EntityType value
    geometry: str                        # preferred representation (GEOMETRY)
    size_m: Tuple[float, float]          # plausible largest dimension (min, max), metres
    small: bool = False                  # needs the high-detail inspection pass
    prompts: Tuple[str, ...] = ()        # open-vocabulary detector queries
    hosts: Tuple[str, ...] = ()          # class ids it is normally mounted on / inside


#: class id -> support level.
SUPPORT: Dict[str, str] = {
    "electrical.switchboard": "reconstruction+persistence (synthetic only; no learned detector installed)",
    "electrical.switchboard.module.socket": "reconstruction+persistence (synthetic only)",
    "electrical.switchboard.module.switch": "reconstruction+persistence (synthetic only)",
}


def _node_to_entity_class(node: OntologyNode) -> EntityClass:
    cat = node.category
    if not cat:
        first = node.path.split(".")[0] if "." in node.path else node.path.split("/")[0]
        cat = first if first in CATEGORIES else "misc"
    geom = node.geometry if node.geometry in GEOMETRY else "hybrid"
    wt = node.world_type if node.world_type != "unknown" else "structure"
    return EntityClass(
        id=node.path,
        label=node.label,
        category=cat,
        parent=node.parent,
        world_type=wt,
        geometry=geom,
        size_m=node.size_m,
        small=node.small,
        prompts=node.prompts,
        hosts=node.hosts,
    )


def get(class_id: str) -> EntityClass:
    onto = get_default_ontology()
    node = onto.get(class_id)
    if node is None:
        raise KeyError(class_id)
    return _node_to_entity_class(node)


def all_classes() -> List[EntityClass]:
    onto = get_default_ontology()
    res = []
    seen = set()
    for node in onto.nodes.values():
        if node.category in CATEGORIES or any(node.path.startswith(c + ".") or node.path == c for c in CATEGORIES):
            ec = _node_to_entity_class(node)
            if ec.id not in seen:
                seen.add(ec.id)
                res.append(ec)
    return res


def small_classes() -> List[EntityClass]:
    return [c for c in all_classes() if c.small]


def children(class_id: str) -> List[EntityClass]:
    onto = get_default_ontology()
    node = onto.get(class_id)
    if not node:
        return []
    out = []
    seen = set()
    for n in onto.nodes.values():
        if (n.parent == node.path or n.parent == class_id) and n.path not in seen:
            seen.add(n.path)
            out.append(_node_to_entity_class(n))
    return out


def ancestors(class_id: str) -> List[str]:
    onto = get_default_ontology()
    node = onto.get(class_id)
    if not node:
        return []
    return onto.ancestors(node.path)


def support_level(class_id: str) -> str:
    return SUPPORT.get(class_id, "registered")


def class_for_label(label: str) -> Optional[EntityClass]:
    """Map a raw detector label to a class by its prompts (exact, case-insensitive)."""
    key = label.strip().lower()
    # 1. Search all registered classes first
    for c in all_classes():
        if key == c.label.lower() or key in [p.lower() for p in c.prompts]:
            return c
    # 2. Try ontology resolution
    onto = get_default_ontology()
    node = onto.resolve_or_unknown(label)
    if node.path != FALLBACK_PATH:
        return _node_to_entity_class(node)
    return None


def size_plausible(class_id: str, largest_dimension_m: float) -> bool:
    ec = get(class_id)
    lo, hi = ec.size_m
    return lo * 0.5 <= largest_dimension_m <= hi * 1.5
