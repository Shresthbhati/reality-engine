"""Extensible Entity Taxonomy for Interior and Exterior Perception (Phase 1).

Defines fine-grained architectural, infrastructural, and furnishing entities with:
- Canonical taxonomy IDs and hierarchies
- Part-parent structural relationships
- Plausible physical dimension bounds (min/max in metres)
- Mounting host semantics (e.g., sockets mounted on walls)
- Common confusion classes and open-vocabulary detection aliases
- Semantic compatibility evaluation across multi-view observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    UNIVERSAL = "universal"


class GeometryType(str, Enum):
    BOX = "box"
    PLANE = "plane"
    CYLINDER = "cylinder"
    POINT_PROXY = "point_proxy"
    MESH = "mesh"


class DimensionMeasurability(str, Enum):
    DIRECT = "direct"  # Can be physically measured directly with clear boundaries
    BOUNDED_PROXY = "bounded_proxy"  # Bounded by bounding box / extent proxy
    SURFACE_ONLY = "surface_only"  # Planar extent (width/height without depth)
    UNMEASURED = "unmeasured"  # Unclear physical boundary (e.g., foliage)


@dataclass(frozen=True)
class TaxonomyClass:
    """Canonical specification for an entity class in the recognition system."""

    taxonomy_id: str
    name: str
    domain: Domain
    category: str
    parent_class: str | None = None
    aliases: tuple[str, ...] = ()
    expected_geometry: GeometryType = GeometryType.BOX
    min_dimensions_m: tuple[float, float, float] = (0.01, 0.01, 0.01)  # (w, d, h)
    max_dimensions_m: tuple[float, float, float] = (100.0, 100.0, 100.0)
    can_contain_parts: bool = False
    potential_parts: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()  # Classes this entity is typically mounted on
    dimension_measurability: DimensionMeasurability = DimensionMeasurability.DIRECT
    common_confusion_classes: tuple[str, ...] = ()
    min_detector_confidence: float = 0.15
    is_structural: bool = False


# ============================================================================
# CANONICAL TAXONOMY CATALOGUE
# ============================================================================

_TAXONOMY_REGISTRY: dict[str, TaxonomyClass] = {}


def _register(cls: TaxonomyClass) -> TaxonomyClass:
    _TAXONOMY_REGISTRY[cls.taxonomy_id] = cls
    return cls


# --- INTERIOR SURFACES & STRUCTURAL ---
WALL = _register(TaxonomyClass(
    taxonomy_id="interior.surface.wall",
    name="wall",
    domain=Domain.INTERIOR,
    category="surface",
    aliases=("wall", "interior wall", "drywall", "partition wall"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.5, 0.05, 1.5),
    max_dimensions_m=(30.0, 0.8, 10.0),
    can_contain_parts=True,
    potential_parts=("door", "window", "socket", "switch", "lamp", "baseboard"),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    common_confusion_classes=("wardrobe", "door", "curtain"),
    min_detector_confidence=0.10,
    is_structural=True,
))

FLOOR = _register(TaxonomyClass(
    taxonomy_id="interior.surface.floor",
    name="floor",
    domain=Domain.INTERIOR,
    category="surface",
    aliases=("floor", "flooring", "hardwood floor", "tiled floor", "carpet"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.0, 1.0, 0.01),
    max_dimensions_m=(50.0, 50.0, 0.5),
    can_contain_parts=True,
    potential_parts=("table", "chair", "bed", "couch", "rug"),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    min_detector_confidence=0.10,
    is_structural=True,
))

CEILING = _register(TaxonomyClass(
    taxonomy_id="interior.surface.ceiling",
    name="ceiling",
    domain=Domain.INTERIOR,
    category="surface",
    aliases=("ceiling", "roof interior", "false ceiling", "soffit"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.0, 1.0, 0.01),
    max_dimensions_m=(50.0, 50.0, 0.5),
    can_contain_parts=True,
    potential_parts=("ceiling_light", "ceiling_fan", "vent", "smoke_detector"),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    min_detector_confidence=0.10,
    is_structural=True,
))

COLUMN = _register(TaxonomyClass(
    taxonomy_id="interior.structural.column",
    name="column",
    domain=Domain.UNIVERSAL,
    category="structural",
    aliases=("column", "pillar", "structural post"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.2, 0.2, 1.5),
    max_dimensions_m=(2.0, 2.0, 15.0),
    dimension_measurability=DimensionMeasurability.DIRECT,
    is_structural=True,
))

BEAM = _register(TaxonomyClass(
    taxonomy_id="interior.structural.beam",
    name="beam",
    domain=Domain.UNIVERSAL,
    category="structural",
    aliases=("beam", "girder", "joist"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.15, 1.0, 0.15),
    max_dimensions_m=(1.5, 30.0, 1.5),
    is_structural=True,
))

# --- INTERIOR OPENINGS & HARDWARE ---
DOOR = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door",
    name="door",
    domain=Domain.INTERIOR,
    category="opening",
    aliases=("door", "interior door", "wooden door", "entryway"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.6, 0.03, 1.8),
    max_dimensions_m=(2.4, 0.25, 3.2),
    can_contain_parts=True,
    potential_parts=("door_frame", "door_handle", "door_hinge", "door_lock"),
    hosts=("wall",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("window", "wardrobe", "cabinet"),
    min_detector_confidence=0.15,
))

DOOR_FRAME = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_frame",
    name="door_frame",
    domain=Domain.INTERIOR,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("door frame", "door casing", "door jamb"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.65, 0.05, 1.85),
    max_dimensions_m=(2.6, 0.35, 3.3),
    hosts=("wall",),
))

DOOR_HANDLE = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_handle",
    name="door_handle",
    domain=Domain.INTERIOR,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("door handle", "doorknob", "door lever", "handle"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.04, 0.04, 0.04),
    max_dimensions_m=(0.30, 0.15, 0.30),
    hosts=("door",),
    min_detector_confidence=0.20,
))

WINDOW = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window",
    name="window",
    domain=Domain.UNIVERSAL,
    category="opening",
    aliases=("window", "glass window", "casement window", "sliding window"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.03, 0.4),
    max_dimensions_m=(4.0, 0.35, 3.0),
    can_contain_parts=True,
    potential_parts=("window_frame", "window_glass", "window_handle", "window_sill"),
    hosts=("wall", "building_facade"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("door", "mirror", "picture_frame"),
    min_detector_confidence=0.15,
))

WINDOW_FRAME = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_frame",
    name="window_frame",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window frame", "window sash", "window casing"),
    expected_geometry=GeometryType.BOX,
    hosts=("wall",),
))

# --- ELECTRICAL & CONTROLS ---
SOCKET = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.socket",
    name="socket",
    domain=Domain.INTERIOR,
    category="electrical",
    aliases=("socket", "electrical socket", "power outlet", "wall outlet", "plug point"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.05, 0.01, 0.05),
    max_dimensions_m=(0.25, 0.08, 0.25),
    hosts=("wall", "floor", "desk"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("switch", "junction_box"),
    min_detector_confidence=0.15,
))

SWITCH = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.switch",
    name="switch",
    domain=Domain.INTERIOR,
    category="electrical",
    aliases=("switch", "light switch", "wall switch", "toggle switch"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.05, 0.01, 0.05),
    max_dimensions_m=(0.25, 0.08, 0.25),
    hosts=("wall",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("socket", "thermostat"),
    min_detector_confidence=0.15,
))

SWITCHBOARD_PLATE = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.switchboard_plate",
    name="switchboard_plate",
    domain=Domain.INTERIOR,
    category="electrical",
    aliases=("switchboard", "switch plate", "gang plate", "modular plate"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.08, 0.01, 0.08),
    max_dimensions_m=(0.40, 0.08, 0.25),
    can_contain_parts=True,
    potential_parts=("socket", "switch", "control_switch"),
    hosts=("wall",),
))

OUTLET_PLATE = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.outlet_plate",
    name="outlet_plate",
    domain=Domain.INTERIOR,
    category="electrical",
    aliases=("outlet plate", "wall plate", "faceplate", "cover plate"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.06, 0.005, 0.06),
    max_dimensions_m=(0.35, 0.03, 0.20),
    can_contain_parts=True,
    potential_parts=("socket", "switch", "control_switch"),
    hosts=("wall", "desk"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

CONTROL_SWITCH = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.control_switch",
    name="control_switch",
    domain=Domain.INTERIOR,
    category="electrical",
    parent_class="interior.electrical.switchboard_plate",
    aliases=("control switch", "rocker switch", "push button", "dimmer switch"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.01, 0.02),
    max_dimensions_m=(0.15, 0.05, 0.15),
    hosts=("wall", "switchboard_plate", "outlet_plate"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

# --- LIGHTING & HVAC ---
LIGHT_CEILING = _register(TaxonomyClass(
    taxonomy_id="interior.lighting.ceiling_light",
    name="ceiling_light",
    domain=Domain.INTERIOR,
    category="lighting",
    aliases=("ceiling light", "downlight", "recessed light", "flush mount light", "luminaire"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.08, 0.08, 0.02),
    max_dimensions_m=(1.5, 1.5, 0.8),
    hosts=("ceiling",),
))

LAMP = _register(TaxonomyClass(
    taxonomy_id="interior.lighting.lamp",
    name="lamp",
    domain=Domain.INTERIOR,
    category="lighting",
    aliases=("lamp", "table lamp", "desk lamp", "bedside lamp", "floor lamp"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.15, 0.15, 0.25),
    max_dimensions_m=(0.8, 0.8, 2.0),
    can_contain_parts=True,
    potential_parts=("diffuser", "light_housing", "mounting_hardware"),
    hosts=("table", "desk", "nightstand", "floor"),
    common_confusion_classes=("vase", "speaker"),
))

LIGHT_HOUSING = _register(TaxonomyClass(
    taxonomy_id="interior.lighting.light_housing",
    name="light_housing",
    domain=Domain.INTERIOR,
    category="lighting",
    aliases=("light housing", "luminaire body", "fixture body", "lamp body"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.08, 0.08, 0.05),
    max_dimensions_m=(1.8, 1.8, 1.2),
    hosts=("ceiling", "wall", "lamp"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

DIFFUSER = _register(TaxonomyClass(
    taxonomy_id="interior.lighting.diffuser",
    name="diffuser",
    domain=Domain.INTERIOR,
    category="lighting",
    aliases=("diffuser", "light diffuser", "lamp shade", "shade", "glass shade"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.08, 0.08, 0.02),
    max_dimensions_m=(1.2, 1.2, 0.8),
    hosts=("lamp", "ceiling_light", "light_housing"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

MOUNTING_HARDWARE = _register(TaxonomyClass(
    taxonomy_id="interior.hardware.mounting_hardware",
    name="mounting_hardware",
    domain=Domain.UNIVERSAL,
    category="hardware",
    aliases=("mounting hardware", "mounting bracket", "canopy", "ceiling rose"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.04, 0.02, 0.02),
    max_dimensions_m=(0.40, 0.20, 0.25),
    hosts=("ceiling", "wall"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

CEILING_FAN = _register(TaxonomyClass(
    taxonomy_id="interior.hvac.ceiling_fan",
    name="ceiling_fan",
    domain=Domain.INTERIOR,
    category="hvac",
    aliases=("ceiling fan", "fan"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.8, 0.8, 0.2),
    max_dimensions_m=(1.8, 1.8, 0.8),
    hosts=("ceiling",),
))

VENT = _register(TaxonomyClass(
    taxonomy_id="interior.hvac.vent",
    name="vent",
    domain=Domain.INTERIOR,
    category="hvac",
    aliases=("vent", "air vent", "vent grille", "diffuser", "exhaust vent"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.1, 0.02, 0.1),
    max_dimensions_m=(1.2, 0.15, 0.6),
    hosts=("ceiling", "wall"),
))

RADIATOR = _register(TaxonomyClass(
    taxonomy_id="interior.hvac.radiator",
    name="radiator",
    domain=Domain.INTERIOR,
    category="hvac",
    aliases=("radiator", "heater", "baseboard heater"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.08, 0.3),
    max_dimensions_m=(2.5, 0.25, 1.0),
    hosts=("wall", "floor"),
))

# --- PLUMBING & SANITARY ---
SINK = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.sink",
    name="sink",
    domain=Domain.INTERIOR,
    category="plumbing",
    aliases=("sink", "washbasin", "bathroom sink", "kitchen sink"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.35, 0.30, 0.20),
    max_dimensions_m=(1.5, 0.70, 0.90),
    can_contain_parts=True,
    potential_parts=("faucet", "drain"),
    hosts=("counter", "cabinet", "wall"),
))

FAUCET = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.faucet",
    name="faucet",
    domain=Domain.INTERIOR,
    category="plumbing",
    parent_class="interior.plumbing.sink",
    aliases=("faucet", "tap", "mixer tap", "water tap"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.04, 0.10, 0.10),
    max_dimensions_m=(0.25, 0.35, 0.50),
    can_contain_parts=True,
    potential_parts=("faucet_handle", "spout"),
    hosts=("sink", "counter"),
))

BASIN = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.basin",
    name="basin",
    domain=Domain.INTERIOR,
    category="plumbing",
    parent_class="interior.plumbing.sink",
    aliases=("basin", "sink basin", "washbasin", "sink bowl"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.25, 0.25, 0.12),
    max_dimensions_m=(1.20, 0.80, 0.50),
    hosts=("sink",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

SPOUT = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.spout",
    name="spout",
    domain=Domain.INTERIOR,
    category="plumbing",
    parent_class="interior.plumbing.faucet",
    aliases=("spout", "tap spout", "faucet spout"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.02, 0.05, 0.05),
    max_dimensions_m=(0.10, 0.35, 0.35),
    hosts=("faucet", "sink"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

TOILET = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.toilet",
    name="toilet",
    domain=Domain.INTERIOR,
    category="plumbing",
    aliases=("toilet", "water closet", "commode", "wc"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.35, 0.50, 0.40),
    max_dimensions_m=(0.60, 0.85, 0.95),
    hosts=("floor", "wall"),
))

# --- FURNITURE & APPLIANCES ---
TABLE = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.table",
    name="table",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("table", "dining table", "coffee table", "side table", "conference table"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.4, 0.3),
    max_dimensions_m=(4.0, 2.0, 1.1),
    can_contain_parts=True,
    potential_parts=("table_leg", "table_top"),
    hosts=("floor",),
    common_confusion_classes=("desk", "counter", "bench"),
))

DESK = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.desk",
    name="desk",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("desk", "workstation", "office desk", "computer desk"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.7, 0.5, 0.65),
    max_dimensions_m=(2.5, 1.2, 0.9),
    hosts=("floor",),
    common_confusion_classes=("table", "counter"),
))

CHAIR = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.chair",
    name="chair",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("chair", "dining chair", "office chair", "desk chair", "stool"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.35, 0.35, 0.45),
    max_dimensions_m=(0.90, 0.90, 1.30),
    hosts=("floor",),
    common_confusion_classes=("armchair", "stool"),
))

SOFA = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.sofa",
    name="sofa",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("sofa", "couch", "lounge", "settee", "davenport"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.8, 0.7, 0.6),
    max_dimensions_m=(3.5, 1.8, 1.1),
    hosts=("floor",),
    common_confusion_classes=("bed", "bench"),
))

BED = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.bed",
    name="bed",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("bed", "double bed", "single bed", "king bed", "mattress"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.9, 1.8, 0.3),
    max_dimensions_m=(2.2, 2.3, 1.4),
    hosts=("floor",),
    common_confusion_classes=("sofa",),
))

CABINET = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.cabinet",
    name="cabinet",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("cabinet", "cupboard", "credenza", "sideboard", "kitchen cabinet"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.3, 0.5),
    max_dimensions_m=(3.0, 0.8, 2.4),
    hosts=("floor", "wall"),
    common_confusion_classes=("wardrobe", "shelf"),
))

WARDROBE = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.wardrobe",
    name="wardrobe",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("wardrobe", "closet", "armoire"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.6, 0.45, 1.5),
    max_dimensions_m=(3.5, 0.8, 2.6),
    hosts=("floor",),
    common_confusion_classes=("cabinet", "door", "wall"),
))

REFRIGERATOR = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.refrigerator",
    name="refrigerator",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("refrigerator", "fridge", "freezer"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.5, 0.5, 0.8),
    max_dimensions_m=(1.2, 0.9, 2.1),
    hosts=("floor",),
    common_confusion_classes=("cabinet",),
))

TELEVISION = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.television",
    name="television",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("television", "tv", "flat screen tv", "monitor", "display"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.03, 0.25),
    max_dimensions_m=(2.2, 0.25, 1.4),
    hosts=("wall", "table", "cabinet"),
    common_confusion_classes=("picture_frame", "blackboard"),
))

# --- ARCHITECTURAL DETAILS & MOULDINGS ---
BASEBOARD = _register(TaxonomyClass(
    taxonomy_id="interior.detail.baseboard",
    name="baseboard",
    domain=Domain.INTERIOR,
    category="detail",
    aliases=("baseboard", "skirting", "skirting board", "mopboard"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.5, 0.01, 0.05),
    max_dimensions_m=(20.0, 0.03, 0.25),
    hosts=("wall", "floor"),
    is_structural=False,
))

MOULDING = _register(TaxonomyClass(
    taxonomy_id="interior.detail.moulding",
    name="moulding",
    domain=Domain.INTERIOR,
    category="detail",
    aliases=("moulding", "molding", "crown moulding", "coving", "cornice moulding", "wall trim"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.5, 0.01, 0.03),
    max_dimensions_m=(20.0, 0.20, 0.35),
    hosts=("wall", "ceiling"),
    is_structural=False,
))

# --- DOORS & HARDWARE ---
DOOR_HINGE = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_hinge",
    name="door_hinge",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("hinge", "door hinge", "butt hinge"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.01, 0.04),
    max_dimensions_m=(0.15, 0.05, 0.20),
    hosts=("door_frame", "door"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

DOOR_LOCK = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_lock",
    name="door_lock",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("door lock", "lock", "deadbolt", "latch", "strike plate"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.03, 0.02, 0.05),
    max_dimensions_m=(0.20, 0.10, 0.25),
    hosts=("door", "door_frame"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

DOOR_PANEL = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_panel",
    name="door_panel",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("door panel", "panel", "leaf", "door leaf"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.50, 0.02, 1.70),
    max_dimensions_m=(2.20, 0.15, 3.00),
    hosts=("door", "door_frame"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

DOOR_GLAZING = _register(TaxonomyClass(
    taxonomy_id="interior.opening.door_glazing",
    name="door_glazing",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.door",
    aliases=("door glazing", "door glass", "door pane", "vision panel"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.15, 0.003, 0.15),
    max_dimensions_m=(2.0, 0.05, 2.5),
    hosts=("door", "door_panel"),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    min_detector_confidence=0.15,
))

# --- WINDOWS & GLAZING ---
WINDOW_GLAZING = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_glazing",
    name="window_glazing",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window glazing", "glazing", "window pane", "window glass", "glass pane"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.2, 0.003, 0.2),
    max_dimensions_m=(5.0, 0.05, 4.0),
    hosts=("window_frame",),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
))

WINDOW_SILL = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_sill",
    name="window_sill",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window sill", "window ledge", "sill"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.3, 0.08, 0.02),
    max_dimensions_m=(5.0, 0.40, 0.15),
    hosts=("wall", "window_frame"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

WINDOW_HANDLE = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_handle",
    name="window_handle",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window handle", "window crank", "window latch", "sash lock"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.02, 0.05),
    max_dimensions_m=(0.25, 0.12, 0.25),
    hosts=("window_frame", "window_sash", "window"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

WINDOW_SASH = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_sash",
    name="window_sash",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window sash", "movable frame", "sash", "casement"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.30, 0.02, 0.30),
    max_dimensions_m=(3.0, 0.15, 2.5),
    hosts=("window_frame", "window"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

WINDOW_MULLION = _register(TaxonomyClass(
    taxonomy_id="interior.opening.window_mullion",
    name="window_mullion",
    domain=Domain.UNIVERSAL,
    category="opening",
    parent_class="interior.opening.window",
    aliases=("window mullion", "mullion", "transom bar", "window divider", "muntin"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.02, 0.30),
    max_dimensions_m=(0.15, 0.15, 3.5),
    hosts=("window_frame", "window"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

# --- ELECTRICAL & CONTROLS ---
ELECTRICAL_PANEL = _register(TaxonomyClass(
    taxonomy_id="interior.electrical.panel",
    name="electrical_panel",
    domain=Domain.UNIVERSAL,
    category="electrical",
    aliases=("electrical panel", "breaker box", "distribution board", "fuse box", "panelboard"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.25, 0.08, 0.30),
    max_dimensions_m=(1.2, 0.30, 2.0),
    hosts=("wall",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("cabinet", "junction_box"),
))

# --- HVAC UNITS & EXPANDED PLUMBING ---
SPLIT_AC = _register(TaxonomyClass(
    taxonomy_id="interior.hvac.split_ac",
    name="split_ac",
    domain=Domain.INTERIOR,
    category="hvac",
    aliases=("split ac", "air conditioner", "ac unit", "ductless mini split", "aircon"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.60, 0.18, 0.25),
    max_dimensions_m=(1.40, 0.35, 0.45),
    hosts=("wall",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("vent",),
))

PIPE = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.pipe",
    name="pipe",
    domain=Domain.UNIVERSAL,
    category="plumbing",
    aliases=("pipe", "plumbing pipe", "water pipe", "drain pipe", "conduit"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.02, 0.02, 0.20),
    max_dimensions_m=(0.50, 0.50, 10.0),
    hosts=("wall", "ceiling", "floor"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

DRAIN_INTERIOR = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.drain",
    name="drain",
    domain=Domain.INTERIOR,
    category="plumbing",
    aliases=("drain", "floor drain", "shower drain", "sink drain"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.05, 0.05, 0.01),
    max_dimensions_m=(0.30, 0.30, 0.05),
    hosts=("floor", "sink"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

SHOWER = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.shower",
    name="shower",
    domain=Domain.INTERIOR,
    category="plumbing",
    aliases=("shower", "shower head", "shower cabin", "shower stall", "shower enclosure"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.70, 0.70, 1.80),
    max_dimensions_m=(1.80, 1.80, 2.40),
    hosts=("floor", "wall"),
    is_structural=False,
))

BATHTUB = _register(TaxonomyClass(
    taxonomy_id="interior.plumbing.bathtub",
    name="bathtub",
    domain=Domain.INTERIOR,
    category="plumbing",
    aliases=("bathtub", "bath", "tub", "soaking tub"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(1.20, 0.65, 0.40),
    max_dimensions_m=(2.10, 1.10, 0.75),
    hosts=("floor",),
    is_structural=False,
))

# --- FURNITURE COMPONENTS & STORAGE ---
SHELF = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.shelf",
    name="shelf",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("shelf", "bookshelf", "wall shelf", "floating shelf", "shelving unit"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.30, 0.15, 0.15),
    max_dimensions_m=(4.0, 0.60, 2.5),
    hosts=("wall", "floor"),
    common_confusion_classes=("cabinet", "bookcase"),
))

DRAWER = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.drawer",
    name="drawer",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("drawer", "sliding drawer", "desk drawer"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.20, 0.25, 0.08),
    max_dimensions_m=(1.20, 0.75, 0.45),
    hosts=("desk", "cabinet", "table"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

FURNITURE_HANDLE = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.handle",
    name="furniture_handle",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("furniture handle", "drawer handle", "cabinet knob", "drawer pull", "pull handle"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.01, 0.01),
    max_dimensions_m=(0.40, 0.06, 0.05),
    hosts=("cabinet", "drawer", "wardrobe"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

FURNITURE_LEG = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.leg",
    name="furniture_leg",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("furniture leg", "table leg", "chair leg", "sofa leg", "desk leg"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.02, 0.02, 0.05),
    max_dimensions_m=(0.15, 0.15, 1.0),
    hosts=("table", "chair", "sofa", "desk"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

COUNTERTOP = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.countertop",
    name="countertop",
    domain=Domain.INTERIOR,
    category="furniture",
    aliases=("countertop", "counter", "worktop", "kitchen counter", "bar counter"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.50, 0.45, 0.02),
    max_dimensions_m=(6.0, 1.20, 0.10),
    hosts=("cabinet", "floor"),
    is_structural=False,
))

TABLE_TOP = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.table_top",
    name="table_top",
    domain=Domain.INTERIOR,
    category="furniture",
    parent_class="interior.furniture.table",
    aliases=("table top", "tabletop", "desk top", "surface"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.4, 0.4, 0.01),
    max_dimensions_m=(4.0, 2.0, 0.15),
    hosts=("table", "desk"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

CHAIR_SEAT = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.chair_seat",
    name="chair_seat",
    domain=Domain.INTERIOR,
    category="furniture",
    parent_class="interior.furniture.chair",
    aliases=("chair seat", "seat cushion", "seat"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.3, 0.3, 0.02),
    max_dimensions_m=(0.8, 0.8, 0.20),
    hosts=("chair",),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

CHAIR_BACKREST = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.chair_backrest",
    name="chair_backrest",
    domain=Domain.INTERIOR,
    category="furniture",
    parent_class="interior.furniture.chair",
    aliases=("chair backrest", "backrest", "chair back"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.25, 0.02, 0.20),
    max_dimensions_m=(0.75, 0.20, 0.90),
    hosts=("chair",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

CHAIR_ARMREST = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.chair_armrest",
    name="chair_armrest",
    domain=Domain.INTERIOR,
    category="furniture",
    parent_class="interior.furniture.chair",
    aliases=("chair armrest", "armrest", "chair arm"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.15, 0.03, 0.05),
    max_dimensions_m=(0.45, 0.15, 0.35),
    hosts=("chair",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

DRAWER_HANDLE = _register(TaxonomyClass(
    taxonomy_id="interior.furniture.drawer_handle",
    name="drawer_handle",
    domain=Domain.INTERIOR,
    category="furniture",
    parent_class="interior.furniture.drawer",
    aliases=("drawer handle", "drawer pull", "cabinet pull", "pull handle"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.02, 0.01, 0.01),
    max_dimensions_m=(0.30, 0.08, 0.08),
    hosts=("drawer", "cabinet"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    min_detector_confidence=0.15,
))

# --- KITCHEN & HOME APPLIANCES ---
MICROWAVE = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.microwave",
    name="microwave",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("microwave", "microwave oven"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.40, 0.30, 0.25),
    max_dimensions_m=(0.70, 0.55, 0.45),
    hosts=("countertop", "shelf", "cabinet"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

OVEN = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.oven",
    name="oven",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("oven", "stove", "range", "built-in oven"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.55, 0.50, 0.50),
    max_dimensions_m=(1.20, 0.75, 1.10),
    hosts=("floor", "cabinet"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

DISHWASHER = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.dishwasher",
    name="dishwasher",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("dishwasher", "dish washer"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.55, 0.55, 0.75),
    max_dimensions_m=(0.70, 0.65, 0.90),
    hosts=("floor", "cabinet"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

WASHING_MACHINE = _register(TaxonomyClass(
    taxonomy_id="interior.appliance.washing_machine",
    name="washing_machine",
    domain=Domain.INTERIOR,
    category="appliance",
    aliases=("washing machine", "washer", "dryer", "laundry washer"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.55, 0.50, 0.75),
    max_dimensions_m=(0.75, 0.75, 1.05),
    hosts=("floor",),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

# --- DECOR & SOFT FURNISHINGS ---
ARTWORK = _register(TaxonomyClass(
    taxonomy_id="interior.decor.artwork",
    name="artwork",
    domain=Domain.INTERIOR,
    category="decor",
    aliases=("artwork", "painting", "picture frame", "framed art", "poster", "canvas", "wall art"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.15, 0.01, 0.15),
    max_dimensions_m=(4.0, 0.10, 3.0),
    hosts=("wall",),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("television", "mirror", "window"),
))

MIRROR = _register(TaxonomyClass(
    taxonomy_id="interior.decor.mirror",
    name="mirror",
    domain=Domain.INTERIOR,
    category="decor",
    aliases=("mirror", "wall mirror", "vanity mirror", "full length mirror"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.20, 0.01, 0.20),
    max_dimensions_m=(3.0, 0.10, 2.5),
    hosts=("wall", "wardrobe"),
    dimension_measurability=DimensionMeasurability.DIRECT,
    common_confusion_classes=("artwork", "window"),
))

CURTAIN = _register(TaxonomyClass(
    taxonomy_id="interior.decor.curtain",
    name="curtain",
    domain=Domain.INTERIOR,
    category="decor",
    aliases=("curtain", "curtains", "drapes", "drapery", "window blind", "roller blind"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.50, 0.02, 0.80),
    max_dimensions_m=(6.0, 0.20, 4.0),
    hosts=("wall", "window", "ceiling"),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    common_confusion_classes=("window", "wall"),
))

CARPET = _register(TaxonomyClass(
    taxonomy_id="interior.decor.carpet",
    name="carpet",
    domain=Domain.INTERIOR,
    category="decor",
    aliases=("carpet", "rug", "area rug", "mat", "floor runner"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.40, 0.40, 0.005),
    max_dimensions_m=(10.0, 10.0, 0.03),
    hosts=("floor",),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    common_confusion_classes=("floor",),
))

PLANT_INDOOR = _register(TaxonomyClass(
    taxonomy_id="interior.decor.plant",
    name="plant",
    domain=Domain.UNIVERSAL,
    category="decor",
    aliases=("indoor plant", "potted plant", "houseplant", "flowerpot", "vase with plant"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.10, 0.10, 0.15),
    max_dimensions_m=(1.5, 1.5, 2.5),
    hosts=("floor", "table", "desk", "window_sill"),
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
))

# --- EXPANDED EXTERIOR ARCHITECTURE & CIRCULATION ---
BUILDING = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.building",
    name="building",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("building", "building structure", "house", "tower", "edifice", "multistorey building"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(3.0, 3.0, 3.0),
    max_dimensions_m=(300.0, 300.0, 300.0),
    can_contain_parts=True,
    potential_parts=("building_facade", "roof", "balcony", "window", "door"),
    is_structural=True,
))

COLUMN_EXTERIOR = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.column",
    name="column_exterior",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("exterior column", "portico column", "pillar", "structural post"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.20, 0.20, 2.0),
    max_dimensions_m=(2.5, 2.5, 30.0),
    hosts=("ground", "footpath"),
    is_structural=True,
))

CORNICE = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.cornice",
    name="cornice",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("cornice", "eaves", "parapet cornice", "fascia", "roof overhang"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(1.0, 0.10, 0.10),
    max_dimensions_m=(100.0, 1.5, 1.5),
    hosts=("building_facade", "roof"),
    is_structural=True,
))

STAIRCASING = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.staircase",
    name="staircase",
    domain=Domain.UNIVERSAL,
    category="architecture",
    aliases=("staircase", "stairs", "steps", "outdoor stairs", "entrance stoop", "stairway"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.8, 0.6, 0.3),
    max_dimensions_m=(15.0, 30.0, 20.0),
    hosts=("ground", "building_facade", "floor"),
    is_structural=True,
))

LANE = _register(TaxonomyClass(
    taxonomy_id="exterior.circulation.lane",
    name="lane",
    domain=Domain.EXTERIOR,
    category="circulation",
    aliases=("traffic lane", "bus lane", "bike lane", "cycle track", "marked lane"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.5, 5.0, 0.01),
    max_dimensions_m=(5.0, 500.0, 0.10),
    hosts=("road",),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
))

PAVING = _register(TaxonomyClass(
    taxonomy_id="exterior.circulation.paving",
    name="paving",
    domain=Domain.EXTERIOR,
    category="circulation",
    aliases=("paving", "block paving", "cobblestone", "stone paving", "pavers", "plaza"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.0, 1.0, 0.02),
    max_dimensions_m=(100.0, 100.0, 0.20),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    is_structural=True,
))

# --- EXPANDED VEGETATION COMPONENTS ---
TREE_TRUNK = _register(TaxonomyClass(
    taxonomy_id="exterior.vegetation.trunk",
    name="tree_trunk",
    domain=Domain.EXTERIOR,
    category="vegetation",
    parent_class="exterior.vegetation.tree",
    aliases=("tree trunk", "trunk", "bole", "tree stem"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.10, 0.10, 1.0),
    max_dimensions_m=(4.0, 4.0, 40.0),
    hosts=("ground",),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

TREE_BRANCH = _register(TaxonomyClass(
    taxonomy_id="exterior.vegetation.branch",
    name="tree_branch",
    domain=Domain.EXTERIOR,
    category="vegetation",
    parent_class="exterior.vegetation.tree",
    aliases=("tree branch", "bough", "limb", "branches"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.05, 0.05, 0.5),
    max_dimensions_m=(1.5, 1.5, 15.0),
    hosts=("tree_trunk",),
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
))

TREE_CANOPY = _register(TaxonomyClass(
    taxonomy_id="exterior.vegetation.canopy",
    name="tree_canopy",
    domain=Domain.EXTERIOR,
    category="vegetation",
    parent_class="exterior.vegetation.tree",
    aliases=("tree canopy", "canopy", "crown", "foliage", "leaves"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(1.0, 1.0, 1.0),
    max_dimensions_m=(35.0, 35.0, 35.0),
    hosts=("tree_trunk",),
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
))

# --- STREET FURNITURE & UTILITY INFRASTRUCTURE ---
BOLLARD = _register(TaxonomyClass(
    taxonomy_id="exterior.street.bollard",
    name="bollard",
    domain=Domain.EXTERIOR,
    category="street",
    aliases=("bollard", "traffic bollard", "post bollard", "security post"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.10, 0.10, 0.6),
    max_dimensions_m=(0.40, 0.40, 1.2),
    hosts=("footpath", "road", "paving"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

BOUNDARY_WALL = _register(TaxonomyClass(
    taxonomy_id="exterior.perimeter.wall",
    name="boundary_wall",
    domain=Domain.EXTERIOR,
    category="perimeter",
    aliases=("boundary wall", "perimeter wall", "retaining wall", "garden wall", "stone wall"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.0, 0.10, 0.5),
    max_dimensions_m=(100.0, 1.0, 5.0),
    hosts=("ground",),
    is_structural=True,
))

MANHOLE = _register(TaxonomyClass(
    taxonomy_id="exterior.utility.manhole",
    name="manhole",
    domain=Domain.EXTERIOR,
    category="utility",
    aliases=("manhole", "manhole cover", "inspection chamber", "sewer grate", "utility cover"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.30, 0.30, 0.02),
    max_dimensions_m=(1.2, 1.2, 0.10),
    hosts=("road", "footpath", "paving"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

FIRE_HYDRANT = _register(TaxonomyClass(
    taxonomy_id="exterior.utility.fire_hydrant",
    name="fire_hydrant",
    domain=Domain.EXTERIOR,
    category="utility",
    aliases=("fire hydrant", "hydrant", "water hydrant"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.20, 0.20, 0.50),
    max_dimensions_m=(0.60, 0.60, 1.10),
    hosts=("footpath", "ground"),
    dimension_measurability=DimensionMeasurability.DIRECT,
))

# --- EXTERIOR FAÇADES & BUILDINGS ---
BUILDING_FACADE = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.building_facade",
    name="building_facade",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("building facade", "facade", "facade wall", "exterior wall", "storefront"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(2.0, 0.2, 2.5),
    max_dimensions_m=(200.0, 5.0, 150.0),
    can_contain_parts=True,
    potential_parts=("window", "door", "balcony", "canopy", "signboard"),
    is_structural=True,
))

ROOF = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.roof",
    name="roof",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("roof", "rooftop", "pitched roof", "flat roof"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(2.0, 2.0, 0.2),
    max_dimensions_m=(100.0, 100.0, 15.0),
    is_structural=True,
))

BALCONY = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.balcony",
    name="balcony",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("balcony", "terrace", "deck"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(1.0, 0.8, 0.9),
    max_dimensions_m=(15.0, 3.5, 1.4),
    hosts=("building_facade",),
    is_structural=True,
))

# --- EXTERIOR CIRCULATION & PAVING ---
ROAD = _register(TaxonomyClass(
    taxonomy_id="exterior.circulation.road",
    name="road",
    domain=Domain.EXTERIOR,
    category="circulation",
    aliases=("road", "street", "carriageway", "asphalt road", "highway"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(2.5, 3.0, 0.05),
    max_dimensions_m=(200.0, 50.0, 0.5),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    is_structural=True,
))

FOOTPATH = _register(TaxonomyClass(
    taxonomy_id="exterior.circulation.footpath",
    name="footpath",
    domain=Domain.EXTERIOR,
    category="circulation",
    aliases=("footpath", "sidewalk", "pavement", "walkway", "pedestrian path"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.8, 1.5, 0.05),
    max_dimensions_m=(100.0, 10.0, 0.3),
    dimension_measurability=DimensionMeasurability.SURFACE_ONLY,
    is_structural=True,
))

KERB = _register(TaxonomyClass(
    taxonomy_id="exterior.circulation.kerb",
    name="kerb",
    domain=Domain.EXTERIOR,
    category="circulation",
    aliases=("kerb", "curb", "kerbstone", "curbstone"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.5, 0.1, 0.05),
    max_dimensions_m=(100.0, 0.3, 0.35),
    is_structural=True,
))

# --- VEGETATION & LANDSCAPE ---
TREE = _register(TaxonomyClass(
    taxonomy_id="exterior.vegetation.tree",
    name="tree",
    domain=Domain.EXTERIOR,
    category="vegetation",
    aliases=("tree", "deciduous tree", "coniferous tree", "palm tree", "large tree"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.5, 0.5, 1.5),
    max_dimensions_m=(25.0, 25.0, 50.0),
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
    common_confusion_classes=("shrub", "pole"),
))

SHRUB = _register(TaxonomyClass(
    taxonomy_id="exterior.vegetation.shrub",
    name="shrub",
    domain=Domain.EXTERIOR,
    category="vegetation",
    aliases=("shrub", "bush", "hedge", "plant"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.2, 0.2, 0.2),
    max_dimensions_m=(15.0, 2.0, 3.5),
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
    common_confusion_classes=("tree",),
))

# --- STREET FURNITURE & SIGNS ---
LAMPPOST = _register(TaxonomyClass(
    taxonomy_id="exterior.street.lamppost",
    name="lamppost",
    domain=Domain.EXTERIOR,
    category="street",
    aliases=("lamppost", "street light", "streetlamp", "light pole"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.15, 0.15, 2.5),
    max_dimensions_m=(1.5, 1.5, 15.0),
    hosts=("footpath", "ground"),
    common_confusion_classes=("utility_pole", "traffic_sign"),
))

ROAD_SIGN = _register(TaxonomyClass(
    taxonomy_id="exterior.street.road_sign",
    name="road_sign",
    domain=Domain.EXTERIOR,
    category="street",
    aliases=("road sign", "traffic sign", "street sign", "stop sign"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.3, 0.05, 1.2),
    max_dimensions_m=(2.5, 0.2, 4.0),
    hosts=("footpath", "ground"),
))

BENCH_OUTDOOR = _register(TaxonomyClass(
    taxonomy_id="exterior.street.bench",
    name="bench",
    domain=Domain.EXTERIOR,
    category="street",
    aliases=("bench", "park bench", "street bench"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.8, 0.4, 0.4),
    max_dimensions_m=(3.0, 0.9, 1.0),
    hosts=("footpath", "ground"),
))

TRASH_BIN = _register(TaxonomyClass(
    taxonomy_id="exterior.street.trash_bin",
    name="trash_bin",
    domain=Domain.UNIVERSAL,
    category="street",
    aliases=("trash bin", "garbage bin", "litter bin", "dustbin", "waste container"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.25, 0.25, 0.35),
    max_dimensions_m=(1.2, 1.2, 1.5),
    hosts=("footpath", "floor"),
))

# --- PERIMETER & UTILITY ---
FENCE = _register(TaxonomyClass(
    taxonomy_id="exterior.perimeter.fence",
    name="fence",
    domain=Domain.EXTERIOR,
    category="perimeter",
    aliases=("fence", "railing", "picket fence", "chain link fence", "metal fence"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(1.0, 0.02, 0.6),
    max_dimensions_m=(100.0, 0.3, 3.5),
    hosts=("ground",),
))

GATE = _register(TaxonomyClass(
    taxonomy_id="exterior.perimeter.gate",
    name="gate",
    domain=Domain.EXTERIOR,
    category="perimeter",
    aliases=("gate", "entry gate", "driveway gate", "wicket gate"),
    expected_geometry=GeometryType.BOX,
    min_dimensions_m=(0.8, 0.05, 0.9),
    max_dimensions_m=(8.0, 0.5, 3.5),
    hosts=("fence", "boundary_wall"),
))

UTILITY_POLE = _register(TaxonomyClass(
    taxonomy_id="exterior.utility.utility_pole",
    name="utility_pole",
    domain=Domain.EXTERIOR,
    category="utility",
    aliases=("utility pole", "telegraph pole", "power pole", "telephone pole"),
    expected_geometry=GeometryType.CYLINDER,
    min_dimensions_m=(0.15, 0.15, 4.0),
    max_dimensions_m=(0.8, 0.8, 18.0),
    hosts=("footpath", "ground"),
))

STORM_DRAIN = _register(TaxonomyClass(
    taxonomy_id="exterior.utility.storm_drain",
    name="storm_drain",
    domain=Domain.EXTERIOR,
    category="utility",
    aliases=("drain", "storm drain", "drain grate", "gully", "catch basin"),
    expected_geometry=GeometryType.PLANE,
    min_dimensions_m=(0.2, 0.2, 0.02),
    max_dimensions_m=(1.5, 1.0, 0.1),
    hosts=("road", "kerb"),
))

FALLBACK_UNKNOWN = TaxonomyClass(
    taxonomy_id="universal.misc.unknown_object",
    name="unknown_object",
    domain=Domain.UNIVERSAL,
    category="misc",
    aliases=("object", "thing", "unknown"),
    expected_geometry=GeometryType.BOX,
    dimension_measurability=DimensionMeasurability.BOUNDED_PROXY,
)


# ============================================================================
# RESOLUTION AND COMPATIBILITY API
# ============================================================================

_ALIAS_MAP: dict[str, TaxonomyClass] = {}


def _build_alias_map() -> None:
    _ALIAS_MAP.clear()
    # 1. Register aliases first
    for tax_cls in _TAXONOMY_REGISTRY.values():
        for alias in tax_cls.aliases:
            alias_lower = alias.lower().strip()
            _ALIAS_MAP[alias_lower] = tax_cls
            _ALIAS_MAP[alias_lower.replace("_", " ")] = tax_cls
    # 2. Canonical taxonomy_id and primary name have strict precedence over aliases
    for tax_cls in _TAXONOMY_REGISTRY.values():
        name_lower = tax_cls.name.lower().strip()
        _ALIAS_MAP[name_lower] = tax_cls
        _ALIAS_MAP[name_lower.replace("_", " ")] = tax_cls
        _ALIAS_MAP[tax_cls.taxonomy_id.lower().strip()] = tax_cls


_build_alias_map()


def resolve_taxonomy_class(raw_label: str) -> TaxonomyClass:
    """Resolves an open-vocabulary detection string or taxonomy ID to its canonical TaxonomyClass."""
    clean = str(raw_label).lower().strip().replace("-", " ").replace("_", " ")
    if clean in _ALIAS_MAP:
        return _ALIAS_MAP[clean]

    # Substring search for common tokens
    for alias, tax_cls in _ALIAS_MAP.items():
        if alias in clean or clean in alias:
            return tax_cls

    return FALLBACK_UNKNOWN


def are_semantically_compatible(class_a: str, class_b: str) -> bool:
    """Evaluates whether two class names can physically denote the same real-world object.

    Handles:
    - Exact alias identity (e.g. 'electrical socket' == 'socket')
    - Same canonical taxonomy class
    - Parent-part containment (e.g. 'door' and 'door_frame', 'sink' and 'faucet')
    - Plausible visual label overlap (e.g. 'desk' and 'table')
    """
    node_a = resolve_taxonomy_class(class_a)
    node_b = resolve_taxonomy_class(class_b)

    if node_a.taxonomy_id == node_b.taxonomy_id:
        return True

    # Check parent-part relationships
    if node_a.parent_class == node_b.taxonomy_id or node_b.parent_class == node_a.taxonomy_id:
        return True

    if node_b.name in node_a.potential_parts or node_a.name in node_b.potential_parts:
        return True

    # Known acceptable semantic confusions across viewpoints
    if node_b.name in node_a.common_confusion_classes or node_a.name in node_b.common_confusion_classes:
        return True

    return False


def get_all_taxonomy_classes() -> Sequence[TaxonomyClass]:
    """Returns all registered taxonomy classes in canonical order."""
    return tuple(_TAXONOMY_REGISTRY.values())


def get_detection_queries_by_domain(domain: Domain | None = None) -> Mapping[str, tuple[str, ...]]:
    """Returns grouped open-vocabulary detection queries suitable for detectors (e.g. OWL-ViT, Grounding DINO)."""
    groups: dict[str, list[str]] = {}
    for tax_cls in _TAXONOMY_REGISTRY.values():
        if domain is not None and tax_cls.domain not in (domain, Domain.UNIVERSAL):
            continue
        cat = tax_cls.category
        if cat not in groups:
            groups[cat] = []
        for a in (tax_cls.name, *tax_cls.aliases[:2]):
            if a not in groups[cat]:
                groups[cat].append(a)
    return {k: tuple(v) for k, v in groups.items()}


def get_hierarchical_path(taxonomy_id: str) -> list[str]:
    """Returns the dotted hierarchy path segments for a given taxonomy ID (e.g. ['interior', 'opening', 'door'])."""
    return [segment.strip() for segment in taxonomy_id.split(".") if segment.strip()]


def get_ancestor_classes(taxonomy_id: str) -> list[TaxonomyClass]:
    """Returns the sequence of parent TaxonomyClasses from immediate parent to root."""
    ancestors: list[TaxonomyClass] = []
    current_id: str | None = taxonomy_id
    visited = set()
    while current_id and current_id in _TAXONOMY_REGISTRY and current_id not in visited:
        visited.add(current_id)
        current = _TAXONOMY_REGISTRY[current_id]
        if current.parent_class and current.parent_class in _TAXONOMY_REGISTRY:
            parent = _TAXONOMY_REGISTRY[current.parent_class]
            ancestors.append(parent)
            current_id = parent.taxonomy_id
        else:
            break
    return ancestors


def get_subparts_for_class(taxonomy_id_or_name: str) -> tuple[TaxonomyClass, ...]:
    """Returns all registered TaxonomyClasses that designate parts, sub-components or hosted objects of this class."""
    parent_cls = resolve_taxonomy_class(taxonomy_id_or_name)
    subparts: list[TaxonomyClass] = []
    seen: set[str] = set()

    broad_surfaces = {"floor", "ceiling", "ground", "road", "footpath", "paving"}

    for cls in _TAXONOMY_REGISTRY.values():
        if cls.taxonomy_id == parent_cls.taxonomy_id:
            continue

        # 1. Direct parent class relationship
        is_child = (cls.parent_class == parent_cls.taxonomy_id or cls.parent_class == parent_cls.name)

        # 2. Host relationship for local objects/surfaces
        is_hosted = False
        if parent_cls.name not in broad_surfaces:
            parent_keys = {parent_cls.name.lower(), parent_cls.taxonomy_id.lower(), *(a.lower() for a in parent_cls.aliases)}
            if any(str(h).lower() in parent_keys for h in cls.hosts):
                is_hosted = True

        # 3. Explicit potential parts declared on the parent
        is_potential = (
            cls.name in parent_cls.potential_parts or
            cls.taxonomy_id in parent_cls.potential_parts or
            any(a in parent_cls.potential_parts for a in cls.aliases)
        )

        if is_child or is_hosted or is_potential:
            if cls.taxonomy_id not in seen:
                seen.add(cls.taxonomy_id)
                subparts.append(cls)

    return tuple(subparts)


def get_domain_prompts(domain: Domain) -> tuple[str, ...]:
    """Generates comprehensive, non-redundant detection prompt queries for an entire domain."""
    seen = set()
    prompts: list[str] = []
    for tax_cls in _TAXONOMY_REGISTRY.values():
        if tax_cls.domain in (domain, Domain.UNIVERSAL):
            for candidate in (tax_cls.name, *tax_cls.aliases[:2]):
                if candidate not in seen:
                    seen.add(candidate)
                    prompts.append(candidate)
    return tuple(prompts)

