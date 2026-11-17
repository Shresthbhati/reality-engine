"""Universal entity taxonomy (a shim over ``config/ontology.yaml``; the data lives there since B03): stable semantic ids, parents, geometry strategy, size priors, detection prompts.

An id is a dotted path (``electrical.switchboard.module.socket``) and NEVER changes once released; labels may.
``parent`` is the primary classification only: other relations (hosted_by, part_of, adjacent_to) are WorldIR
relationships, so nothing is forced into one rigid tree. ``world_type`` maps onto the EXISTING WorldIR ``EntityType``
(no enum change): fixtures and furniture are ``structure`` and identified by ``custom_properties["entity_class"]``.

A class listed here is *registered*, not *supported*: support means detection, localisation, reconstruction,
persistence and verification all work, and is tracked in ``SUPPORT`` below (default ``registered``).
Size ranges are plausibility priors in metres (min, max of the largest dimension), used to reject absurd detections;
they are never written to a world as measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from perception.ontology import load_ontology

CATEGORIES: Dict[str, str] = {
    "building": "Buildings and structural architecture", "room": "Rooms and interior spaces",
    "envelope": "Walls, floors and ceilings", "opening": "Doors, windows and openings",
    "structural": "Columns, beams and structural components", "circulation": "Staircases, railings and landings",
    "electrical": "Electrical infrastructure", "plumbing": "Plumbing and sanitary fixtures",
    "hvac": "HVAC and mechanical systems", "furniture": "Furniture and appliances",
    "finish": "Interior finishes and decorative details", "exterior": "Exterior building components",
    "road": "Roads, footpaths and kerbs", "vegetation": "Trees, plants and landscape features",
    "street": "Lampposts and street furniture", "utility": "Utility infrastructure",
    "signage": "Signs, poles and public infrastructure", "misc": "Miscellaneous objects and unknown entities",
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


_cache: Tuple[object, Dict[str, EntityClass]] = (None, {})


def _R() -> Dict[str, EntityClass]:
    """Registered classes, read from the ontology (config/ontology.yaml): the nodes that carry a ``world_type``.

    Id = ontology path with "/" -> "."; hosts = the node's ``hosted_by`` attach targets.
    """
    global _cache
    onto = load_ontology()
    if _cache[0] is not onto:
        reg: Dict[str, EntityClass] = {}
        for n in onto.nodes.values():
            if n.world_type is None:
                continue
            cid = n.path.replace("/", ".")
            reg[cid] = EntityClass(
                cid, n.label, n.group, n.parent.replace("/", ".") if n.parent else None, n.world_type, n.geometry,
                n.size_m, n.small, n.prompts,
                tuple(t.replace("/", ".") for a in n.attach for k, t in a.items() if k == "hosted_by"))
        _cache = (onto, reg)
    return _cache[1]


#: class id -> support level. NOTHING is "supported" until detection, localisation, reconstruction, persistence and
#: verification all work for it (see docs/entities/ENTITY_PERCEPTION_PLAN.md). Updated only with test evidence.
SUPPORT: Dict[str, str] = {
    "electrical.switchboard": "reconstruction+persistence (synthetic only; no learned detector installed)",
    "electrical.switchboard.module.socket": "reconstruction+persistence (synthetic only)",
    "electrical.switchboard.module.switch": "reconstruction+persistence (synthetic only)",
}


def get(class_id: str) -> EntityClass:
    return _R()[class_id]


def all_classes() -> List[EntityClass]:
    return list(_R().values())


def small_classes() -> List[EntityClass]:
    return [c for c in _R().values() if c.small]


def children(class_id: str) -> List[EntityClass]:
    return [c for c in _R().values() if c.parent == class_id]


def ancestors(class_id: str) -> List[str]:
    reg = _R()
    out, cur = [], reg[class_id].parent
    while cur:
        out.append(cur)
        cur = reg[cur].parent
    return out


def support_level(class_id: str) -> str:
    return SUPPORT.get(class_id, "registered")


def class_for_label(label: str) -> Optional[EntityClass]:
    """Map a raw detector label to a class by its prompts (exact, case-insensitive); None when it is not in the registry."""
    key = label.strip().lower()
    for c in _R().values():
        if key == c.label.lower() or key in c.prompts:
            return c
    return None


def size_plausible(class_id: str, largest_dimension_m: float) -> bool:
    lo, hi = _R()[class_id].size_m
    return lo * 0.5 <= largest_dimension_m <= hi * 1.5
