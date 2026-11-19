"""Universal Furniture, Appliance, Fixture, Architectural, and Urban Object Ontology.

Defines the universal object and part hierarchy for Reality Engine, bridging
low-level perception detections (SAM, Grounding DINO, Florence-2) with
high-level spatial and semantic relationships in WorldIR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObjectCategory(str, Enum):
    FURNITURE = "furniture"
    APPLIANCE = "appliance"
    FIXTURE = "fixture"
    BUILDING_ELEMENT = "building_element"
    EVERYDAY_OBJECT = "everyday_object"
    URBAN_INFRASTRUCTURE = "urban_infrastructure"


class SpatialRelationType(str, Enum):
    PART_OF = "part_of"
    SUPPORTS = "supports"
    ATTACHED_TO = "attached_to"
    CONTAINED_IN = "contained_in"
    ADJACENT_TO = "adjacent_to"
    HOSTED_BY = "hosted_by"
    CONNECTS = "connects"


@dataclass(frozen=True)
class OntologyPart:
    """A subcomponent or subpart belonging to a parent object."""

    part_name: str
    parent_class: str
    description: str
    typical_relative_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # normalized (dx, dy, dz)
    can_have_subparts: bool = False


@dataclass
class OntologyClass:
    """A semantic object class in the universal reconstruction ontology."""

    class_name: str
    category: ObjectCategory
    synonyms: list[str] = field(default_factory=list)
    allowed_parts: list[str] = field(default_factory=list)
    expected_relations: list[SpatialRelationType] = field(default_factory=list)
    typical_dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (w, d, h)
    default_material: str = "generic"
    is_container: bool = False
    is_structural: bool = False


class UniversalObjectOntology:
    """Registry and query engine for universal objects, part decompositions, and spatial relations."""

    def __init__(self) -> None:
        self._classes: dict[str, OntologyClass] = {}
        self._synonym_map: dict[str, str] = {}
        self._parts: dict[tuple[str, str], OntologyPart] = {}
        self._populate_standard_ontology()

    def _populate_standard_ontology(self) -> None:
        # 1. Furniture
        self.register_class(
            OntologyClass(
                class_name="chair",
                category=ObjectCategory.FURNITURE,
                synonyms=["armchair", "office chair", "dining chair", "stool"],
                allowed_parts=["seat", "backrest", "legs", "armrests", "cushion", "base"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.55, 0.55, 0.90),
                default_material="wood_or_fabric",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="table",
                category=ObjectCategory.FURNITURE,
                synonyms=["desk", "dining table", "coffee table", "workstation"],
                allowed_parts=["top", "legs", "drawer", "shelf", "frame"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(1.40, 0.80, 0.75),
                default_material="wood",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="sofa",
                category=ObjectCategory.FURNITURE,
                synonyms=["couch", "settee", "lounge"],
                allowed_parts=["seat_cushion", "back_cushion", "armrests", "base", "legs"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(2.10, 0.95, 0.85),
                default_material="fabric_or_leather",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="bed",
                category=ObjectCategory.FURNITURE,
                synonyms=["double bed", "single bed", "bunk bed"],
                allowed_parts=["mattress", "headboard", "frame", "legs", "pillow", "blanket"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(1.60, 2.05, 0.60),
                default_material="wood_and_fabric",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="cabinet",
                category=ObjectCategory.FURNITURE,
                synonyms=["wardrobe", "cupboard", "credenza", "closet"],
                allowed_parts=["door", "handle", "shelf", "drawer", "carcass"],
                expected_relations=[SpatialRelationType.CONTAINED_IN, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.90, 0.50, 1.80),
                default_material="wood",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="bookshelf",
                category=ObjectCategory.FURNITURE,
                synonyms=["bookcase", "shelving unit"],
                allowed_parts=["shelf", "side_panel", "back_panel", "base"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.80, 0.35, 1.90),
                default_material="wood",
                is_container=True,
            )
        )

        # 2. Appliances
        self.register_class(
            OntologyClass(
                class_name="refrigerator",
                category=ObjectCategory.APPLIANCE,
                synonyms=["fridge", "freezer"],
                allowed_parts=["body", "door", "handle", "freezer_door", "vent_grille"],
                expected_relations=[SpatialRelationType.ADJACENT_TO, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.85, 0.75, 1.75),
                default_material="metal",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="oven",
                category=ObjectCategory.APPLIANCE,
                synonyms=["stove", "range", "cooktop"],
                allowed_parts=["housing", "door", "handle", "window", "control_knobs", "display", "burner"],
                expected_relations=[SpatialRelationType.HOSTED_BY, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.75, 0.65, 0.90),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="microwave",
                category=ObjectCategory.APPLIANCE,
                synonyms=["microwave oven"],
                allowed_parts=["housing", "door", "handle", "window", "control_panel", "turntable"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.55, 0.40, 0.35),
                default_material="metal_and_glass",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="dishwasher",
                category=ObjectCategory.APPLIANCE,
                synonyms=["dish washer"],
                allowed_parts=["door", "handle", "control_panel", "rack"],
                expected_relations=[SpatialRelationType.HOSTED_BY, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.60, 0.60, 0.85),
                default_material="metal",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="washing_machine",
                category=ObjectCategory.APPLIANCE,
                synonyms=["washer", "dryer", "laundry machine"],
                allowed_parts=["housing", "drum_door", "detergent_drawer", "dial", "display"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.60, 0.65, 0.85),
                default_material="metal",
            )
        )

        # 3. Fixtures (Electrical, Plumbing, HVAC)
        self.register_class(
            OntologyClass(
                class_name="electrical_socket",
                category=ObjectCategory.FIXTURE,
                synonyms=["wall outlet", "power point", "socket"],
                allowed_parts=["faceplate", "receptacle", "switch_button"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.08, 0.03, 0.08),
                default_material="plastic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="light_switch",
                category=ObjectCategory.FIXTURE,
                synonyms=["wall switch", "toggle switch"],
                allowed_parts=["faceplate", "toggle_rocker"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.08, 0.03, 0.08),
                default_material="plastic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="lighting_fixture",
                category=ObjectCategory.FIXTURE,
                synonyms=["ceiling lamp", "pendant light", "chandelier", "sconce"],
                allowed_parts=["mount", "canopy", "lamp_shade", "bulb_socket", "stem"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.40, 0.40, 0.50),
                default_material="metal_and_glass",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="sink",
                category=ObjectCategory.FIXTURE,
                synonyms=["washbasin", "kitchen sink", "bathroom basin"],
                allowed_parts=["basin", "faucet", "drain", "countertop_mount"],
                expected_relations=[SpatialRelationType.HOSTED_BY, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.60, 0.50, 0.30),
                default_material="ceramic_or_stainless_steel",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="toilet",
                category=ObjectCategory.FIXTURE,
                synonyms=["water closet", "commode"],
                allowed_parts=["bowl", "tank", "seat", "lid", "flush_handle"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.50, 0.70, 0.80),
                default_material="ceramic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="hvac_vent",
                category=ObjectCategory.FIXTURE,
                synonyms=["air register", "diffuser", "exhaust grille"],
                allowed_parts=["grille", "damper", "frame"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.30, 0.05, 0.30),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="radiator",
                category=ObjectCategory.FIXTURE,
                synonyms=["heater", "convector", "heating radiator"],
                allowed_parts=["fins", "valve", "pipe", "bracket"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(1.00, 0.15, 0.60),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="thermostat",
                category=ObjectCategory.FIXTURE,
                synonyms=["temperature controller", "climate control"],
                allowed_parts=["display", "dial", "faceplate"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.12, 0.04, 0.12),
                default_material="plastic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="faucet",
                category=ObjectCategory.FIXTURE,
                synonyms=["tap", "water tap", "mixer"],
                allowed_parts=["spout", "handle", "aerator", "base"],
                expected_relations=[SpatialRelationType.ATTACHED_TO, SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.20, 0.15, 0.30),
                default_material="chrome_metal",
            )
        )

        # 4. Building Elements (Structural & Openings)
        self.register_class(
            OntologyClass(
                class_name="wall",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["partition", "interior wall", "exterior wall", "drywall"],
                allowed_parts=["core", "drywall_layer", "baseboard", "cornice"],
                expected_relations=[SpatialRelationType.CONNECTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(4.00, 0.15, 2.70),
                default_material="concrete_or_gypsum",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="floor",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["floor slab", "ground floor", "deck"],
                allowed_parts=["slab", "screed", "finish_layer"],
                expected_relations=[SpatialRelationType.SUPPORTS],
                typical_dimensions_m=(5.00, 5.00, 0.20),
                default_material="concrete_or_timber",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="ceiling",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["soffit", "suspended ceiling"],
                allowed_parts=["slab_underside", "suspension_grid", "tile"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(5.00, 5.00, 0.10),
                default_material="plasterboard",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="door",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["entry door", "interior door"],
                allowed_parts=["frame", "leaf", "handle", "hinges", "lock"],
                expected_relations=[SpatialRelationType.HOSTED_BY, SpatialRelationType.CONNECTS],
                typical_dimensions_m=(0.90, 0.10, 2.10),
                default_material="wood_or_metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="window",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["casement window", "glazing"],
                allowed_parts=["frame", "sash", "pane", "sill", "latch"],
                expected_relations=[SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(1.20, 0.15, 1.40),
                default_material="glass_and_aluminum",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="stairs",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["staircase", "steps"],
                allowed_parts=["tread", "riser", "stringer", "nosing", "handrail", "baluster"],
                expected_relations=[SpatialRelationType.CONNECTS, SpatialRelationType.SUPPORTS],
                typical_dimensions_m=(1.00, 3.50, 2.80),
                default_material="concrete_or_wood",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="column",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["pillar", "post", "support column"],
                allowed_parts=["capital", "shaft", "base"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.CONNECTS],
                typical_dimensions_m=(0.40, 0.40, 2.70),
                default_material="concrete_or_steel",
                is_structural=True,
            )
        )

        # 5. Everyday Objects
        self.register_class(
            OntologyClass(
                class_name="laptop",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["notebook computer", "portable pc"],
                allowed_parts=["screen_lid", "keyboard_base", "trackpad"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.CONTAINED_IN],
                typical_dimensions_m=(0.35, 0.25, 0.02),
                default_material="aluminum_or_plastic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="monitor",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["display", "screen", "tv", "television"],
                allowed_parts=["display_panel", "stand", "bezel"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.60, 0.15, 0.40),
                default_material="plastic_and_glass",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="bottle",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["water bottle", "flask"],
                allowed_parts=["body", "cap", "neck"],
                expected_relations=[SpatialRelationType.CONTAINED_IN],
                typical_dimensions_m=(0.08, 0.08, 0.25),
                default_material="glass_or_plastic",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="mug",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["coffee mug", "cup", "teacup"],
                allowed_parts=["handle", "rim", "body"],
                expected_relations=[SpatialRelationType.CONTAINED_IN, SpatialRelationType.SUPPORTS],
                typical_dimensions_m=(0.12, 0.09, 0.10),
                default_material="ceramic",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="book",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["tome", "hardcover", "paperback"],
                allowed_parts=["cover", "spine", "pages"],
                expected_relations=[SpatialRelationType.CONTAINED_IN, SpatialRelationType.SUPPORTS],
                typical_dimensions_m=(0.22, 0.15, 0.03),
                default_material="paper",
            )
        )

        # 6. Urban Infrastructure
        self.register_class(
            OntologyClass(
                class_name="building_facade",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["exterior facade", "frontage", "elevation"],
                allowed_parts=["ground_floor", "upper_floors", "roofline", "parapet"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(15.0, 1.0, 20.0),
                default_material="brick_masonry_or_glass",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="road",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["street", "avenue", "carriageway"],
                allowed_parts=["lane", "intersection", "marking", "curb"],
                expected_relations=[SpatialRelationType.CONNECTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(7.0, 50.0, 0.3),
                default_material="asphalt",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="footpath",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["pavement", "sidewalk", "walkway"],
                allowed_parts=["paving", "curb", "tactile_paving"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(2.5, 50.0, 0.15),
                default_material="concrete_pavers",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="street_lamp",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["lamp post", "street light"],
                allowed_parts=["pole", "luminaire", "base", "bracket"],
                expected_relations=[SpatialRelationType.ATTACHED_TO, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.50, 0.50, 6.00),
                default_material="steel",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="traffic_light",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["signal light", "stoplight"],
                allowed_parts=["pole", "signal_head", "visor", "bracket"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.60, 0.40, 4.50),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="tree",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["street tree", "deciduous tree", "conifer"],
                allowed_parts=["trunk", "crown", "branches", "foliage"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(4.0, 4.0, 8.0),
                default_material="vegetation",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="bush",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["shrub", "hedge"],
                allowed_parts=["branches", "foliage"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(1.5, 1.5, 1.2),
                default_material="vegetation",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="fire_hydrant",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["hydrant", "water hydrant"],
                allowed_parts=["body", "nozzle_cap", "operating_nut"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.40, 0.40, 0.85),
                default_material="cast_iron",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="traffic_sign",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["road sign", "street sign", "stop sign"],
                allowed_parts=["post", "signboard", "mount"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.60, 0.10, 2.40),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="bench",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["park bench", "street bench"],
                allowed_parts=["seat", "backrest", "legs", "armrests"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(1.80, 0.65, 0.85),
                default_material="wood_or_metal",
            )
        )
        # 7. Additional MEP, Electrical & Plumbing Fixtures
        self.register_class(
            OntologyClass(
                class_name="fuse_box",
                category=ObjectCategory.FIXTURE,
                synonyms=["breaker panel", "electrical panel", "distribution board"],
                allowed_parts=["enclosure", "door", "breaker_switch", "bus_bar"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.40, 0.15, 0.60),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="conduit",
                category=ObjectCategory.FIXTURE,
                synonyms=["electrical conduit", "cable raceway", "wire duct"],
                allowed_parts=["pipe_body", "connector", "hanger"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.05, 2.00, 0.05),
                default_material="pvc_or_metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="junction_box",
                category=ObjectCategory.FIXTURE,
                synonyms=["electrical junction box", "terminal box"],
                allowed_parts=["box_cover", "mounting_screw"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.12, 0.08, 0.12),
                default_material="plastic_or_metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="smoke_detector",
                category=ObjectCategory.FIXTURE,
                synonyms=["fire alarm", "smoke alarm"],
                allowed_parts=["sensor_housing", "test_button", "indicator_led"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.14, 0.05, 0.14),
                default_material="plastic",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="water_meter",
                category=ObjectCategory.FIXTURE,
                synonyms=["flow meter", "water flow meter"],
                allowed_parts=["meter_dial", "pipe_inlet", "pipe_outlet"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.20, 0.15, 0.20),
                default_material="brass_or_metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="shutoff_valve",
                category=ObjectCategory.FIXTURE,
                synonyms=["ball valve", "stopcock", "gate valve"],
                allowed_parts=["valve_handle", "valve_body"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.10, 0.08, 0.12),
                default_material="brass",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="plumbing_pipe",
                category=ObjectCategory.FIXTURE,
                synonyms=["water pipe", "drain pipe"],
                allowed_parts=["pipe_segment", "elbow_joint", "tee_joint"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.06, 2.50, 0.06),
                default_material="copper_or_pvc",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="hvac_duct",
                category=ObjectCategory.FIXTURE,
                synonyms=["air duct", "ventilation duct"],
                allowed_parts=["duct_trunk", "duct_hanger", "damper"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.40, 2.00, 0.30),
                default_material="galvanized_steel",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="air_diffuser",
                category=ObjectCategory.FIXTURE,
                synonyms=["vent grille", "ac vent", "air register"],
                allowed_parts=["louver_slats", "frame"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.30, 0.05, 0.30),
                default_material="aluminum",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="drain",
                category=ObjectCategory.FIXTURE,
                synonyms=["floor drain", "waste drain"],
                allowed_parts=["grate", "strainer"],
                expected_relations=[SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.15, 0.15, 0.05),
                default_material="chrome_or_pvc",
            )
        )

        # 8. Architectural Details & Infrastructure
        self.register_class(
            OntologyClass(
                class_name="beam",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["structural beam", "ceiling beam", "joist", "girder"],
                allowed_parts=["flange", "web"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.30, 4.00, 0.40),
                default_material="steel_or_timber",
                is_structural=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="door_handle",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["doorknob", "door lever", "lever handle"],
                allowed_parts=["escutcheon", "lever"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.15, 0.08, 0.06),
                default_material="metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="door_frame",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["door casing", "door jamb"],
                allowed_parts=["jamb", "header", "casing"],
                expected_relations=[SpatialRelationType.HOSTED_BY, SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.95, 0.12, 2.15),
                default_material="wood_or_metal",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="skirting_board",
                category=ObjectCategory.BUILDING_ELEMENT,
                synonyms=["baseboard", "mopboard"],
                allowed_parts=["profile"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.02, 3.00, 0.10),
                default_material="wood",
            )
        )

        # 9. Street & Outdoor Infrastructure
        self.register_class(
            OntologyClass(
                class_name="bollard",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["traffic bollard", "security post"],
                allowed_parts=["post_body", "cap", "reflector"],
                expected_relations=[SpatialRelationType.ATTACHED_TO],
                typical_dimensions_m=(0.20, 0.20, 0.90),
                default_material="steel_or_concrete",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="manhole_cover",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["sewer cover", "drain cover", "utility hole"],
                allowed_parts=["cover_plate", "frame_ring"],
                expected_relations=[SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.65, 0.65, 0.08),
                default_material="cast_iron",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="trash_can",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["garbage bin", "waste bin", "litter bin"],
                allowed_parts=["bin_body", "lid", "liner"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.45, 0.45, 0.85),
                default_material="metal_or_plastic",
                is_container=True,
            )
        )
        self.register_class(
            OntologyClass(
                class_name="curb",
                category=ObjectCategory.URBAN_INFRASTRUCTURE,
                synonyms=["road curb", "pavement kerb"],
                allowed_parts=["curb_stone"],
                expected_relations=[SpatialRelationType.ADJACENT_TO],
                typical_dimensions_m=(0.15, 1.00, 0.25),
                default_material="concrete_or_granite",
            )
        )

        # 10. Everyday Tech & Personal Objects
        self.register_class(
            OntologyClass(
                class_name="smartphone",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["mobile phone", "cell phone", "iphone", "android phone"],
                allowed_parts=["screen", "body", "camera_bump"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.075, 0.01, 0.15),
                default_material="glass_and_aluminum",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="laptop",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["notebook", "macbook", "laptop computer"],
                allowed_parts=["screen", "keyboard", "trackpad", "lid"],
                expected_relations=[SpatialRelationType.SUPPORTS, SpatialRelationType.HOSTED_BY],
                typical_dimensions_m=(0.32, 0.22, 0.02),
                default_material="aluminum",
            )
        )
        self.register_class(
            OntologyClass(
                class_name="water_bottle",
                category=ObjectCategory.EVERYDAY_OBJECT,
                synonyms=["flask", "drink bottle"],
                allowed_parts=["bottle_body", "cap"],
                expected_relations=[SpatialRelationType.SUPPORTS],
                typical_dimensions_m=(0.08, 0.08, 0.25),
                default_material="stainless_steel_or_plastic",
                is_container=True,
            )
        )

    def register_class(self, cls: OntologyClass) -> None:
        name = cls.class_name.lower()
        self._classes[name] = cls
        self._synonym_map[name] = name
        for syn in cls.synonyms:
            self._synonym_map[syn.lower()] = name

        for part_name in cls.allowed_parts:
            part = OntologyPart(part_name=part_name, parent_class=name, description=f"{part_name} of {name}")
            self._parts[(name, part_name)] = part

    def resolve_class_name(self, label: str) -> str | None:
        """Resolves raw detection string or synonym to canonical ontology class name."""
        cleaned = label.strip().lower()
        return self._synonym_map.get(cleaned)

    def get_class(self, class_name: str) -> OntologyClass | None:
        canonical = self.resolve_class_name(class_name)
        if not canonical:
            return None
        return self._classes.get(canonical)

    def is_valid_part(self, parent_class: str, part_name: str) -> bool:
        canonical = self.resolve_class_name(parent_class)
        if not canonical:
            return False
        p_name = part_name.lower().strip()
        if p_name.startswith(f"{canonical}_"):
            p_name = p_name[len(canonical) + 1:]
        if (canonical, p_name) in self._parts:
            return True
        # Check singular/plural
        if p_name.endswith("s") and (canonical, p_name[:-1]) in self._parts:
            return True
        if not p_name.endswith("s") and (canonical, p_name + "s") in self._parts:
            return True
        return False

    def get_parts(self, parent_class: str) -> list[OntologyPart]:
        canonical = self.resolve_class_name(parent_class)
        if not canonical:
            return []
        cls = self._classes[canonical]
        return [self._parts[(canonical, p)] for p in cls.allowed_parts if (canonical, p) in self._parts]

    def validate_spatial_relation(
        self,
        source_class: str,
        target_class: str,
        relation: SpatialRelationType,
    ) -> tuple[bool, str]:
        """Validates physical plausibility of a spatial relation between two classes."""
        if relation == SpatialRelationType.PART_OF:
            if not self.is_valid_part(target_class, source_class):
                return False, f"'{source_class}' is not an established part of '{target_class}'"
            return True, "Valid part_of relation"

        src = self.get_class(source_class)
        tgt = self.get_class(target_class)
        if not src or not tgt:
            return True, "Unknown class, passing unvalidated"

        if relation == SpatialRelationType.SUPPORTS:
            # Floors, tables, shelves, desks support objects
            valid_supporters = {"floor", "table", "chair", "bed", "bookshelf", "road", "footpath"}
            if src.class_name not in valid_supporters and not src.is_structural:
                return False, f"'{source_class}' does not typically support '{target_class}'"

        if relation == SpatialRelationType.CONTAINED_IN:
            if not tgt.is_container:
                return False, f"Target '{target_class}' is not a container"

        if relation == SpatialRelationType.ATTACHED_TO:
            # Wall, ceiling, post are valid attach targets
            valid_targets = {"wall", "ceiling", "floor", "street_lamp", "building_facade"}
            if tgt.class_name not in valid_targets and not tgt.is_structural:
                return False, f"'{source_class}' attached to non-surface '{target_class}'"

        return True, "Valid spatial relation"


_GLOBAL_ONTOLOGY: UniversalObjectOntology | None = None


def get_universal_ontology() -> UniversalObjectOntology:
    global _GLOBAL_ONTOLOGY
    if _GLOBAL_ONTOLOGY is None:
        _GLOBAL_ONTOLOGY = UniversalObjectOntology()
    return _GLOBAL_ONTOLOGY
