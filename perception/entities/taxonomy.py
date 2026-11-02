"""Universal entity taxonomy: stable semantic ids, parents, geometry strategy, size priors, detection prompts.

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
from typing import Dict, Iterable, List, Optional, Tuple

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


_R: Dict[str, EntityClass] = {}


def _c(id: str, label: str, parent: Optional[str], world_type: str, geometry: str, size: Tuple[float, float],
       small: bool = False, prompts: Iterable[str] = (), hosts: Iterable[str] = ()) -> None:
    category = id.split(".")[0]
    assert category in CATEGORIES and geometry in GEOMETRY and id not in _R, id
    assert parent is None or parent in _R, (id, parent)
    _R[id] = EntityClass(id, label, category, parent, world_type, geometry, size, small, tuple(prompts) or (label.lower(),),
                         tuple(hosts))


# ---- 1-2 buildings, rooms ---------------------------------------------------------------------------------------------
_c("building", "Building", None, "building", "hybrid", (3, 500))
_c("building.storey", "Storey", "building", "storey", "parametric", (2, 100))
_c("room", "Room", None, "room", "parametric", (1.2, 15))
_c("room.corridor", "Corridor", "room", "corridor", "parametric", (1, 60))
_c("room.kitchen", "Kitchen", "room", "room", "parametric", (1.5, 12))
_c("room.bathroom", "Bathroom", "room", "room", "parametric", (1.2, 8))
_c("room.bedroom", "Bedroom", "room", "room", "parametric", (2, 10))
_c("room.living", "Living room", "room", "room", "parametric", (2.5, 15))
# ---- 3 envelope -----------------------------------------------------------------------------------------------------
_c("envelope.wall", "Wall", None, "wall", "parametric", (0.3, 30))
_c("envelope.floor", "Floor", None, "floor", "parametric", (1, 30))
_c("envelope.ceiling", "Ceiling", None, "ceiling", "parametric", (1, 30))
_c("envelope.skirting", "Skirting board", "envelope.wall", "structure", "procedural", (0.5, 20), prompts=("skirting board",))
_c("envelope.cornice", "Cornice / coving", "envelope.ceiling", "structure", "procedural", (0.5, 20))
# ---- 4 openings ---------------------------------------------------------------------------------------------------
_c("opening.door", "Door", None, "door", "parametric", (0.5, 3.2), prompts=("door",), hosts=("envelope.wall",))
_c("opening.door.leaf", "Door leaf", "opening.door", "structure", "parametric", (0.5, 3.0))
_c("opening.door.frame", "Door frame", "opening.door", "structure", "parametric", (0.5, 3.2))
_c("opening.door.handle", "Door handle", "opening.door", "structure", "hybrid", (0.05, 0.35), True,
   ("door handle", "door knob", "lever handle"), ("opening.door.leaf",))
_c("opening.door.hinge", "Door hinge", "opening.door", "structure", "parametric", (0.03, 0.2), True, ("door hinge",),
   ("opening.door.frame",))
_c("opening.door.lock", "Door lock / latch", "opening.door", "structure", "parametric", (0.02, 0.3), True,
   ("door lock", "keyhole", "latch"), ("opening.door.leaf",))
_c("opening.window", "Window", None, "window", "parametric", (0.2, 4), prompts=("window",), hosts=("envelope.wall",))
_c("opening.window.frame", "Window frame", "opening.window", "structure", "parametric", (0.2, 4))
_c("opening.window.glazing", "Glazing", "opening.window", "structure", "parametric", (0.1, 4))
_c("opening.window.mullion", "Window subdivision (mullion/transom)", "opening.window", "structure", "parametric", (0.1, 3))
_c("opening.window.handle", "Window handle", "opening.window", "structure", "hybrid", (0.04, 0.25), True,
   ("window handle", "window latch"), ("opening.window.frame",))
_c("opening.opening", "Generic opening", None, "opening", "parametric", (0.2, 6))
# ---- 5 structural ---------------------------------------------------------------------------------------------------
_c("structural.column", "Column", None, "column", "parametric", (0.1, 12), prompts=("column", "pillar"))
_c("structural.beam", "Beam", None, "beam", "parametric", (0.2, 15), prompts=("exposed beam",))
_c("structural.arch", "Arch", None, "arch", "procedural", (0.5, 12), prompts=("arch", "archway"))
# ---- 6 circulation --------------------------------------------------------------------------------------------------
_c("circulation.stairs", "Staircase", None, "stairs", "procedural", (1, 12), prompts=("staircase", "stairs"))
_c("circulation.stairs.tread", "Stair tread", "circulation.stairs", "structure", "parametric", (0.15, 2), True)
_c("circulation.stairs.riser", "Stair riser", "circulation.stairs", "structure", "parametric", (0.05, 2), True)
_c("circulation.stairs.stringer", "Stair stringer / support", "circulation.stairs", "structure", "mesh", (0.5, 8))
_c("circulation.stairs.landing", "Landing", "circulation.stairs", "structure", "parametric", (0.6, 5))
_c("circulation.railing", "Railing / balustrade", None, "structure", "procedural", (0.3, 30),
   prompts=("railing", "balustrade", "glass railing", "handrail"))
_c("circulation.railing.post", "Baluster / post", "circulation.railing", "structure", "parametric", (0.4, 1.4), True)
_c("circulation.railing.glass", "Glass panel", "circulation.railing", "structure", "parametric", (0.3, 3))
# ---- 7 electrical (small-object critical) -------------------------------------------------------------------------------
_c("electrical.switchboard", "Switchboard / faceplate", None, "structure", "hybrid", (0.05, 1.2), True,
   ("switchboard", "switch plate", "wall socket plate", "electrical faceplate"), ("envelope.wall",))
_c("electrical.switchboard.module", "Switchboard module", "electrical.switchboard", "structure", "parametric", (0.02, 0.15),
   True, hosts=("electrical.switchboard",))
_c("electrical.switchboard.module.socket", "Power socket (plug point)", "electrical.switchboard.module", "structure",
   "parametric", (0.03, 0.12), True, ("wall socket", "power outlet", "plug point", "electrical socket"),
   ("electrical.switchboard",))
_c("electrical.switchboard.module.switch", "Switch", "electrical.switchboard.module", "structure", "parametric", (0.02, 0.1),
   True, ("light switch", "wall switch", "rocker switch", "push button"), ("electrical.switchboard",))
_c("electrical.switchboard.module.dimmer", "Dimmer / rotary control", "electrical.switchboard.module", "structure",
   "parametric", (0.02, 0.1), True, ("dimmer switch", "rotary knob"), ("electrical.switchboard",))
_c("electrical.switchboard.module.usb", "USB port", "electrical.switchboard.module", "structure", "parametric", (0.01, 0.05),
   True, ("usb port", "usb charging socket"), ("electrical.switchboard",))
_c("electrical.switchboard.module.data", "Data / TV / phone port", "electrical.switchboard.module", "structure",
   "parametric", (0.015, 0.06), True, ("ethernet port", "tv socket", "telephone socket"), ("electrical.switchboard",))
_c("electrical.switchboard.module.blank", "Blank plate", "electrical.switchboard.module", "structure", "parametric",
   (0.02, 0.1), True, hosts=("electrical.switchboard",))
_c("electrical.junction_box", "Junction box", None, "structure", "parametric", (0.04, 0.3), True, ("junction box",),
   ("envelope.wall", "envelope.ceiling"))
_c("electrical.distribution_board", "Distribution board / fuse box", None, "structure", "hybrid", (0.2, 1.2), False,
   ("distribution board", "fuse box", "circuit breaker panel"), ("envelope.wall",))
_c("electrical.conduit", "Conduit / trunking", None, "structure", "procedural", (0.1, 30), True, ("conduit", "cable trunking"))
_c("electrical.wiring", "Visible wiring", None, "structure", "procedural", (0.1, 30), True, ("exposed wire", "cable"))
_c("electrical.light_fitting", "Light fitting", None, "structure", "hybrid", (0.05, 1.5), False,
   ("ceiling light", "pendant light", "downlight", "wall lamp"), ("envelope.ceiling", "envelope.wall"))
_c("electrical.fan", "Ceiling fan", None, "structure", "hybrid", (0.3, 1.6), False, ("ceiling fan",), ("envelope.ceiling",))
# ---- 8 plumbing ----------------------------------------------------------------------------------------------------------
_c("plumbing.pipe", "Pipe", None, "structure", "procedural", (0.1, 30), True, ("pipe",))
_c("plumbing.tap", "Tap / faucet", None, "structure", "hybrid", (0.05, 0.4), True, ("tap", "faucet", "shower head"))
_c("plumbing.sink", "Sink / basin", None, "structure", "hybrid", (0.3, 1.5), False, ("sink", "wash basin"))
_c("plumbing.toilet", "Toilet", None, "structure", "mesh", (0.4, 0.9), False, ("toilet",))
_c("plumbing.bath", "Bathtub / shower", None, "structure", "hybrid", (0.7, 2.2), False, ("bathtub", "shower enclosure"))
_c("plumbing.drain", "Drain", None, "structure", "parametric", (0.05, 0.5), True, ("floor drain",))
# ---- 9 hvac --------------------------------------------------------------------------------------------------------------
_c("hvac.vent", "Vent / grille", None, "structure", "parametric", (0.1, 1), True, ("air vent", "ventilation grille"))
_c("hvac.ac_unit", "Air-conditioning unit", None, "structure", "hybrid", (0.3, 2), False, ("air conditioner",))
_c("hvac.radiator", "Radiator", None, "structure", "hybrid", (0.3, 2.5), False, ("radiator",))
_c("hvac.duct", "Duct", None, "structure", "procedural", (0.2, 30), False, ("duct",))
_c("hvac.thermostat", "Thermostat", None, "structure", "parametric", (0.05, 0.2), True, ("thermostat",), ("envelope.wall",))
# ---- 10 furniture & appliances -------------------------------------------------------------------------------------------
for _id, _lbl, _sz in (("sofa", "Sofa", (1.2, 4)), ("chair", "Chair", (0.3, 1.2)), ("table", "Table", (0.4, 4)),
                       ("bed", "Bed", (1.2, 2.4)), ("wardrobe", "Wardrobe / cabinet", (0.4, 3)),
                       ("shelf", "Shelf / bookcase", (0.3, 3)), ("tv_unit", "TV unit", (0.5, 3)),
                       ("desk", "Desk", (0.6, 2.5)), ("rug", "Rug / carpet", (0.5, 5)),
                       ("tv", "Television", (0.4, 2.5)), ("refrigerator", "Refrigerator", (0.5, 2.2)),
                       ("oven", "Oven / hob", (0.4, 1.2)), ("washing_machine", "Washing machine", (0.5, 1)),
                       ("microwave", "Microwave", (0.3, 0.7))):
    _c(f"furniture.{_id}", _lbl, None, "structure", "hybrid", _sz, prompts=(_lbl.lower().split(" /")[0],))
# ---- 11 finishes ---------------------------------------------------------------------------------------------------------
_c("finish.curtain", "Curtain / blind", None, "structure", "mesh", (0.3, 4), prompts=("curtain", "window blind"))
_c("finish.picture", "Picture / framed art", None, "structure", "parametric", (0.1, 3), prompts=("picture frame", "wall art"),
   hosts=("envelope.wall",))
_c("finish.mirror", "Mirror", None, "structure", "parametric", (0.1, 2.5), prompts=("mirror",), hosts=("envelope.wall",))
_c("finish.tiling", "Tiled area", None, "structure", "procedural", (0.3, 30), prompts=("wall tiles",))
_c("finish.moulding", "Decorative moulding / ornament", None, "structure", "mesh", (0.05, 30))
# ---- 12 exterior ---------------------------------------------------------------------------------------------------------
_c("exterior.roof", "Roof", None, "roof", "procedural", (2, 60), prompts=("roof",))
_c("exterior.balcony", "Balcony / terrace", None, "structure", "procedural", (0.8, 12), prompts=("balcony",))
_c("exterior.gate", "Gate", None, "structure", "parametric", (0.6, 8), prompts=("gate",))
_c("exterior.fence", "Fence / boundary wall", None, "structure", "procedural", (1, 200), prompts=("fence", "boundary wall"))
_c("exterior.chimney", "Chimney", None, "structure", "parametric", (0.3, 4), prompts=("chimney",))
_c("exterior.gutter", "Gutter / downpipe", None, "structure", "procedural", (0.5, 30), True, ("downpipe", "gutter"))
# ---- 13 roads -----------------------------------------------------------------------------------------------------------
_c("road.carriageway", "Carriageway", None, "road", "parametric", (3, 5000), prompts=("road",))
_c("road.footpath", "Footpath / sidewalk", None, "sidewalk", "parametric", (1, 5000), prompts=("footpath", "sidewalk"))
_c("road.kerb", "Kerb", None, "curb", "parametric", (1, 5000), prompts=("kerb", "curb"))
_c("road.marking", "Road marking", None, "structure", "procedural", (0.1, 100), prompts=("zebra crossing", "road marking"))
_c("road.drain", "Road drain / manhole", None, "structure", "parametric", (0.3, 1.2), True, ("manhole cover", "storm drain"))
# ---- 14 vegetation ------------------------------------------------------------------------------------------------------
_c("vegetation.tree", "Tree", None, "vegetation", "procedural", (1.5, 45), prompts=("tree",))
_c("vegetation.tree.trunk", "Trunk", "vegetation.tree", "vegetation", "parametric", (0.5, 20))
_c("vegetation.tree.canopy", "Canopy", "vegetation.tree", "vegetation", "volumetric", (1, 30))
_c("vegetation.shrub", "Shrub / bush", None, "vegetation", "volumetric", (0.3, 5), prompts=("shrub", "bush", "hedge"))
_c("vegetation.plant", "Potted / indoor plant", None, "vegetation", "volumetric", (0.1, 3), prompts=("potted plant", "houseplant"))
_c("vegetation.flowerbed", "Flowerbed", None, "vegetation", "procedural", (0.5, 100), prompts=("flower bed",))
_c("vegetation.lawn", "Lawn / garden", None, "vegetation", "procedural", (1, 2000), prompts=("lawn", "garden"))
# ---- 15 street furniture ------------------------------------------------------------------------------------------------
_c("street.lamppost", "Lamppost", None, "structure", "procedural", (2, 15), prompts=("street lamp", "lamp post"))
_c("street.lamppost.shaft", "Lamppost shaft", "street.lamppost", "structure", "procedural", (2, 14))
_c("street.lamppost.fixture", "Lamp head / fixture", "street.lamppost", "structure", "hybrid", (0.2, 2), True)
_c("street.bench", "Bench", None, "structure", "hybrid", (0.8, 3), prompts=("park bench",))
_c("street.bin", "Dustbin", None, "structure", "hybrid", (0.3, 1.5), prompts=("dustbin", "trash can"))
_c("street.postbox", "Post box", None, "structure", "hybrid", (0.3, 1.8), prompts=("post box",))
_c("street.barricade", "Barricade / bollard", None, "structure", "hybrid", (0.3, 3), prompts=("barricade", "bollard", "traffic barrier"))
# ---- 16 utility ---------------------------------------------------------------------------------------------------------
_c("utility.meter", "Meter", None, "structure", "parametric", (0.1, 0.6), True, ("electricity meter", "gas meter"))
_c("utility.transformer", "Transformer / cabinet", None, "structure", "hybrid", (0.5, 4), prompts=("transformer",))
_c("utility.hydrant", "Fire hydrant", None, "structure", "hybrid", (0.3, 1.2), prompts=("fire hydrant",))
_c("utility.cable_run", "Overhead cable", None, "structure", "procedural", (2, 500), prompts=("overhead cable",))
# ---- 17 signage ---------------------------------------------------------------------------------------------------------
_c("signage.sign", "Sign", None, "structure", "parametric", (0.1, 6), prompts=("sign", "signboard"))
_c("signage.pole", "Pole", None, "structure", "procedural", (1.5, 20), prompts=("utility pole", "signal pole"))
_c("signage.traffic_light", "Traffic signal", None, "structure", "hybrid", (0.5, 8), prompts=("traffic light",))
# ---- 18 misc ------------------------------------------------------------------------------------------------------------
_c("misc.object", "Unclassified object", None, "unknown", "hybrid", (0.01, 50))
_c("misc.unknown", "Unknown entity", None, "unknown", "pointcloud", (0.01, 500))

#: class id -> support level. NOTHING is "supported" until detection, localisation, reconstruction, persistence and
#: verification all work for it (see docs/entities/ENTITY_PERCEPTION_PLAN.md). Updated only with test evidence.
SUPPORT: Dict[str, str] = {
    "electrical.switchboard": "reconstruction+persistence (synthetic only; no learned detector installed)",
    "electrical.switchboard.module.socket": "reconstruction+persistence (synthetic only)",
    "electrical.switchboard.module.switch": "reconstruction+persistence (synthetic only)",
}


def get(class_id: str) -> EntityClass:
    return _R[class_id]


def all_classes() -> List[EntityClass]:
    return list(_R.values())


def small_classes() -> List[EntityClass]:
    return [c for c in _R.values() if c.small]


def children(class_id: str) -> List[EntityClass]:
    return [c for c in _R.values() if c.parent == class_id]


def ancestors(class_id: str) -> List[str]:
    out, cur = [], _R[class_id].parent
    while cur:
        out.append(cur)
        cur = _R[cur].parent
    return out


def support_level(class_id: str) -> str:
    return SUPPORT.get(class_id, "registered")


def class_for_label(label: str) -> Optional[EntityClass]:
    """Map a raw detector label to a class by its prompts (exact, case-insensitive); None when it is not in the registry."""
    key = label.strip().lower()
    for c in _R.values():
        if key == c.label.lower() or key in c.prompts:
            return c
    return None


def size_plausible(class_id: str, largest_dimension_m: float) -> bool:
    lo, hi = _R[class_id].size_m
    return lo * 0.5 <= largest_dimension_m <= hi * 1.5
