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
    potential_parts=("socket", "switch"),
    hosts=("wall",),
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
    hosts=("table", "desk", "nightstand", "floor"),
    common_confusion_classes=("vase", "speaker"),
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
    hosts=("sink", "counter"),
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

# --- ARCHITECTURAL DETAILS ---
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

# --- EXTERIOR FAÇADES & BUILDINGS ---
BUILDING_FACADE = _register(TaxonomyClass(
    taxonomy_id="exterior.architecture.building_facade",
    name="building_facade",
    domain=Domain.EXTERIOR,
    category="architecture",
    aliases=("building", "facade", "building facade", "exterior wall", "storefront"),
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
    for tax_cls in _TAXONOMY_REGISTRY.values():
        _ALIAS_MAP[tax_cls.name.lower()] = tax_cls
        _ALIAS_MAP[tax_cls.taxonomy_id.lower()] = tax_cls
        for alias in tax_cls.aliases:
            _ALIAS_MAP[alias.lower().strip()] = tax_cls


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
