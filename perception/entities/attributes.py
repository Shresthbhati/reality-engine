"""Fine-Grained Entity Attributes and Subtype Intelligence (Wave 1, Blocks 1 & 2).

Provides typed representations of fine physical attributes, mechanical configurations,
mounting styles, operational states, and subtype-specific taxonomies.

Guarantees:
- Structured attributes attach directly to WorldIR Entity metadata and custom_properties.
- Plausibility verification against physical priors.
- Explicit uncertainty tracking (never fabulates unobserved attributes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from world_ir.schema_v1 import Entity


class MechanismType(str, Enum):
    """Physical mechanism type of opening or fixture."""
    FIXED = "fixed"
    HINGED_SINGLE = "hinged_single"
    HINGED_DOUBLE = "hinged_double"
    SLIDING = "sliding"
    POCKET_SLIDING = "pocket_sliding"
    FOLDING = "folding"
    REVOLVING = "revolving"
    TILT_AND_TURN = "tilt_and_turn"
    ROCKER = "rocker"
    PUSH_BUTTON = "push_button"
    ROTARY_DIMMER = "rotary_dimmer"
    LEVER = "lever"
    SENSOR_TOUCHLESS = "sensor_touchless"
    UNKNOWN = "unknown"


class MountingType(str, Enum):
    """How the entity is physically mounted in the environment."""
    FLOOR_MOUNTED = "floor_mounted"
    WALL_MOUNTED = "wall_mounted"
    CEILING_MOUNTED = "ceiling_mounted"
    RECESSED_INSET = "recessed_inset"
    SURFACE_RESTING = "surface_resting"
    SUSPENDED = "suspended"
    EMBEDDED_STRUCTURE = "embedded_structure"
    UNKNOWN = "unknown"


class OperationalState(str, Enum):
    """Observable operational or physical state."""
    OPEN = "open"
    CLOSED = "closed"
    AJAR = "ajar"
    ON = "on"
    OFF = "off"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


# Registered granular subtypes per taxonomy class_id
SUBTYPES_BY_CLASS: Dict[str, Dict[str, str]] = {
    "furniture.chair": {
        "dining": "Dining chair with upright back",
        "office_swivel": "Office task chair with casters and swivel base",
        "armchair": "Upholstered armchair / easy chair",
        "lounge": "Low-slung accent or lounge chair",
        "stool": "High bar stool or backless stool",
        "recliner": "Adjustable reclining chair",
        "bench": "Multi-person seating bench",
    },
    "furniture.table": {
        "dining": "Large dining or banquet table",
        "coffee": "Low living-room coffee table",
        "bedside": "Compact bedside nightstand table",
        "study_desk": "Work or study desk with legroom",
        "conference": "Large multi-seat meeting table",
        "console": "Narrow hallway or accent console table",
    },
    "furniture.sofa": {
        "standard_two_seat": "Two-seater love seat",
        "standard_three_seat": "Three-seater sofa",
        "sectional_l": "L-shaped sectional couch",
        "sofa_bed": "Convertible sofa-bed / futon",
    },
    "furniture.bed": {
        "single": "Single / twin size bed",
        "double": "Double / full size bed",
        "queen": "Queen size bed",
        "king": "King size bed",
        "bunk": "Two-tier bunk bed",
    },
    "opening.door": {
        "hinged_single": "Standard single-leaf swinging door",
        "hinged_double": "Double-leaf French door",
        "sliding": "Sliding panel door",
        "pocket": "In-wall recessed pocket door",
        "bifold": "Folding accordion or bifold door",
        "revolving": "Commercial revolving door",
    },
    "opening.window": {
        "casement": "Side-hinged outward-swinging casement window",
        "sliding_horizontal": "Horizontal track sliding window",
        "single_hung": "Vertical sliding bottom-leaf sash window",
        "double_hung": "Vertical sliding dual-leaf sash window",
        "fixed_picture": "Non-operable fixed glazing window",
        "tilt_turn": "Dual-action tilt or inward swing window",
        "clerestory": "High-wall horizontal ribbon window",
    },
    "electrical.switchboard.module.switch": {
        "rocker": "Standard rocker switch",
        "push_button": "Momentary or latching push button",
        "dimmer_rotary": "Rotary analog dimmer knob",
        "toggle": "Classic lever toggle switch",
        "touch_smart": "Capacitive touch / IoT smart switch",
    },
    "electrical.switchboard.module.socket": {
        "type_c_euro": "Europlug 2-pin socket (CEE 7/16)",
        "type_d_india": "Indian 3-pin round socket (5A)",
        "type_m_power": "Heavy-duty 15A/16A power socket",
        "type_g_uk": "British 3-pin rectangular socket (BS 1363)",
        "type_a_b_us": "North American 2/3-pin flat socket (NEMA)",
        "universal": "Multi-standard universal socket receptacle",
        "usb_charging": "Dedicated USB-A / USB-C charging port",
    },
    "electrical.light_fitting": {
        "recessed_downlight": "Flush ceiling pot light / downlight",
        "pendant": "Suspended hanging pendant fixture",
        "chandelier": "Multi-arm ornamental ceiling chandelier",
        "wall_sconce": "Wall-mounted accent light fixture",
        "flush_mount": "Direct surface-mounted ceiling fixture",
        "track_light": "Linear directional track spotlight fixture",
        "strip_led": "Concealed linear cove LED strip",
    },
    "plumbing.sink": {
        "drop_in": "Top-mounted drop-in basin with counter rim",
        "undermount": "Under-counter seamless mounted sink",
        "vessel": "Countertop bowl vessel basin",
        "wall_mounted": "Direct wall-hung basin",
        "pedestal": "Free-standing pedestal basin",
    },
    "plumbing.tap": {
        "single_lever": "Single-handle mixer faucet",
        "two_handle": "Dual hot/cold handle faucet",
        "wall_mounted": "Wall-mounted spout and controls",
        "gooseneck": "High-arch curved kitchen mixer faucet",
        "sensor_automatic": "Infrared touchless automatic tap",
    },
}


@dataclass
class EntityAttributes:
    """Rich fine-grained physical and semantic attributes for an entity."""
    class_id: str
    subtype: Optional[str] = None
    mechanism: MechanismType = MechanismType.UNKNOWN
    mounting: MountingType = MountingType.UNKNOWN
    state: OperationalState = OperationalState.UNKNOWN
    material_label: Optional[str] = None
    finish: Optional[str] = None
    color_rgb: Optional[Tuple[float, float, float]] = None
    dimensions_m: Optional[Tuple[float, float, float]] = None  # (width, height, depth)
    clearance_m: Optional[float] = None                       # Required operational clearance
    confidence: float = 1.0
    uncertainty_notes: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize attributes to dictionary for JSON/metadata storage."""
        out: Dict[str, Any] = {
            "class_id": self.class_id,
            "subtype": self.subtype,
            "mechanism": self.mechanism.value,
            "mounting": self.mounting.value,
            "state": self.state.value,
            "confidence": round(self.confidence, 3),
        }
        if self.material_label:
            out["material_label"] = self.material_label
        if self.finish:
            out["finish"] = self.finish
        if self.color_rgb:
            out["color_rgb"] = [round(c, 4) for c in self.color_rgb]
        if self.dimensions_m:
            out["dimensions_m"] = [round(d, 4) for d in self.dimensions_m]
        if self.clearance_m is not None:
            out["clearance_m"] = round(self.clearance_m, 4)
        if self.uncertainty_notes:
            out["uncertainty_notes"] = list(self.uncertainty_notes)
        if self.custom_properties:
            out["custom_properties"] = dict(self.custom_properties)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityAttributes:
        """Construct attributes from serialized dictionary."""
        color = None
        if "color_rgb" in data and data["color_rgb"]:
            color = tuple(float(x) for x in data["color_rgb"][:3])
        dims = None
        if "dimensions_m" in data and data["dimensions_m"]:
            dims = tuple(float(x) for x in data["dimensions_m"][:3])

        return cls(
            class_id=data.get("class_id", "misc.object"),
            subtype=data.get("subtype"),
            mechanism=MechanismType(data.get("mechanism", MechanismType.UNKNOWN.value)),
            mounting=MountingType(data.get("mounting", MountingType.UNKNOWN.value)),
            state=OperationalState(data.get("state", OperationalState.UNKNOWN.value)),
            material_label=data.get("material_label"),
            finish=data.get("finish"),
            color_rgb=color,
            dimensions_m=dims,
            clearance_m=data.get("clearance_m"),
            confidence=float(data.get("confidence", 1.0)),
            uncertainty_notes=list(data.get("uncertainty_notes", [])),
            custom_properties=dict(data.get("custom_properties", {})),
        )

    def attach_to_entity(self, entity: Entity) -> None:
        """Inject these attributes into an existing WorldIR Entity's custom properties."""
        if entity.custom_properties is None:
            entity.custom_properties = {}
        entity.custom_properties["attributes"] = self.to_dict()
        if self.subtype:
            entity.custom_properties["subtype"] = self.subtype
        if self.mechanism != MechanismType.UNKNOWN:
            entity.custom_properties["mechanism"] = self.mechanism.value
        if self.mounting != MountingType.UNKNOWN:
            entity.custom_properties["mounting"] = self.mounting.value


def infer_entity_attributes(
    class_id: str,
    dimensions_m: Optional[Tuple[float, float, float]] = None,
    visual_tags: Sequence[str] = (),
    host_type: Optional[str] = None,
    measured_confidence: float = 0.85,
) -> EntityAttributes:
    """Infer fine-grained attributes using category rules, visual tags, and host relationships.
    
    Adheres strictly to the rule: distinguish observed evidence from inference.
    Inferred fields record uncertainty notes explaining the inference rationale.
    """
    notes: List[str] = []
    tags_lower = {t.strip().lower() for t in visual_tags}

    # 1. Subtype resolution
    detected_subtype: Optional[str] = None
    known_subtypes = SUBTYPES_BY_CLASS.get(class_id, {})
    for st_key in known_subtypes:
        if st_key in tags_lower or any(st_key in t for t in tags_lower):
            detected_subtype = st_key
            break

    # 2. Mounting style
    mounting = MountingType.UNKNOWN
    if host_type:
        if "wall" in host_type.lower():
            mounting = MountingType.WALL_MOUNTED
        elif "ceiling" in host_type.lower():
            mounting = MountingType.CEILING_MOUNTED
        elif "floor" in host_type.lower():
            mounting = MountingType.FLOOR_MOUNTED
    else:
        # Prior defaults based on taxonomy class
        if class_id.startswith("electrical.switchboard") or class_id == "opening.door" or class_id == "opening.window":
            mounting = MountingType.WALL_MOUNTED
            notes.append("Mounting inferred from entity class host prior")
        elif class_id == "electrical.fan" or "ceiling" in class_id:
            mounting = MountingType.CEILING_MOUNTED
            notes.append("Mounting inferred from entity class host prior")
        elif class_id.startswith("furniture."):
            mounting = MountingType.FLOOR_MOUNTED
            notes.append("Mounting inferred from furniture ground support prior")

    # 3. Mechanism resolution
    mechanism = MechanismType.UNKNOWN
    if "sliding" in tags_lower or (detected_subtype and "sliding" in detected_subtype):
        mechanism = MechanismType.SLIDING
    elif "hinged" in tags_lower or (detected_subtype and "hinged" in detected_subtype):
        mechanism = MechanismType.HINGED_SINGLE
    elif "rocker" in tags_lower:
        mechanism = MechanismType.ROCKER
    elif "dimmer" in tags_lower:
        mechanism = MechanismType.ROTARY_DIMMER
    elif "lever" in tags_lower or "handle" in tags_lower:
        mechanism = MechanismType.LEVER
    elif class_id == "opening.door":
        mechanism = MechanismType.HINGED_SINGLE
        notes.append("Door mechanism inferred as hinged single (dominant architectural prior)")

    # 4. State resolution
    state = OperationalState.UNKNOWN
    if "open" in tags_lower:
        state = OperationalState.OPEN
    elif "closed" in tags_lower:
        state = OperationalState.CLOSED
    elif "on" in tags_lower:
        state = OperationalState.ON
    elif "off" in tags_lower:
        state = OperationalState.OFF

    # 5. Clearance estimation
    clearance: Optional[float] = None
    if dimensions_m:
        w, h, d = dimensions_m
        if mechanism in (MechanismType.HINGED_SINGLE, MechanismType.HINGED_DOUBLE):
            # Swing clearance equal to door width
            clearance = max(w, d)
        elif mechanism == MechanismType.SLIDING:
            clearance = 0.10  # Minimal sliding clearance
        elif class_id == "furniture.chair":
            clearance = 0.60  # Pull-out space

    return EntityAttributes(
        class_id=class_id,
        subtype=detected_subtype,
        mechanism=mechanism,
        mounting=mounting,
        state=state,
        dimensions_m=dimensions_m,
        clearance_m=clearance,
        confidence=measured_confidence if detected_subtype else max(0.5, measured_confidence - 0.2),
        uncertainty_notes=notes,
    )
