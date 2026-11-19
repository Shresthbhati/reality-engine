"""Extensible Architectural Attribute Taxonomy and Visual Recognition Contracts.

Defines fine-grained architectural attribute taxonomy across 11 key categories:
1. Door handles and locks
2. Window types and hardware
3. Electrical sockets and plug points
4. Switches and switch configurations
5. Lighting fixtures
6. Plumbing fixtures and fittings
7. HVAC components
8. Furniture construction
9. Materials and surface finishes
10. Structural elements
11. Small architectural details

Enforces strict separation between visual model inferences and heuristic priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttributeCategory(str, Enum):
    """Broad architectural attribute categories."""

    DOOR_HARDWARE = "door_hardware"
    WINDOW_HARDWARE = "window_hardware"
    ELECTRICAL_SOCKETS = "electrical_sockets"
    ELECTRICAL_SWITCHES = "electrical_switches"
    LIGHTING_FIXTURES = "lighting_fixtures"
    PLUMBING_FIXTURES = "plumbing_fixtures"
    HVAC_COMPONENTS = "hvac_components"
    FURNITURE_CONSTRUCTION = "furniture_construction"
    MATERIALS_FINISHES = "materials_finishes"
    STRUCTURAL_ELEMENTS = "structural_elements"
    ARCHITECTURAL_DETAILS = "architectural_details"


@dataclass(frozen=True)
class AttributeDefinition:
    """Specification of an architectural attribute in the taxonomy."""

    name: str
    category: AttributeCategory
    allowed_values: tuple[str, ...]
    description: str
    default_unknown: str = "unknown"
    visual_inspection_hints: str = ""

    def is_valid_value(self, val: str) -> bool:
        """Check whether a predicted value is permitted by this definition."""
        return val in self.allowed_values or val == self.default_unknown


@dataclass(frozen=True)
class VisualAttributePrediction:
    """Strongly-typed container for attribute inference with full cryptographic provenance."""

    attribute_name: str
    value: str
    confidence: float
    basis: str
    is_visual_inference: bool
    source_evidence_id: str | None = None
    crop_bbox_px: tuple[int, int, int, int] | None = None
    model_name: str | None = None
    checkpoint_hash: str | None = None
    model_version: str | None = None
    inference_config: dict[str, Any] = field(default_factory=dict)
    is_calibrated: bool = False
    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "p": round(self.confidence, 4),
            "basis": self.basis,
            "is_visual_inference": self.is_visual_inference,
            "source_evidence_id": self.source_evidence_id,
            "crop_bbox_px": list(self.crop_bbox_px) if self.crop_bbox_px else None,
            "model_name": self.model_name,
            "checkpoint_hash": self.checkpoint_hash,
            "model_version": self.model_version,
            "inference_config": self.inference_config,
            "is_calibrated": self.is_calibrated,
            "refusal_reason": self.refusal_reason,
        }


class AttributeTaxonomy:
    """Canonical registry and validator for Reality Engine attribute taxonomy."""

    _DEFINITIONS: dict[str, AttributeDefinition] = {
        # 1. Door hardware
        "handle_type": AttributeDefinition(
            name="handle_type",
            category=AttributeCategory.DOOR_HARDWARE,
            allowed_values=(
                "lever",
                "knob",
                "pull_handle",
                "flush_pull",
                "panic_bar",
                "thumbturn",
            ),
            description="Type of door or cabinet handle mechanism",
            visual_inspection_hints="Look for elongated horizontal arm (lever), spherical knob, or long vertical bar (pull).",
        ),
        "lock_type": AttributeDefinition(
            name="lock_type",
            category=AttributeCategory.DOOR_HARDWARE,
            allowed_values=(
                "mortise_lock",
                "cylinder_deadbolt",
                "tubular_latch",
                "card_reader",
                "keypad_lock",
                "rim_latch",
                "cam_lock",
            ),
            description="Type of door locking mechanism",
        ),
        # 2. Window hardware
        "window_operation": AttributeDefinition(
            name="window_operation",
            category=AttributeCategory.WINDOW_HARDWARE,
            allowed_values=(
                "casement",
                "double_hung_sash",
                "sliding",
                "tilt_and_turn",
                "awning",
                "fixed_picture",
            ),
            description="Operating mode of window sash",
        ),
        "window_hardware": AttributeDefinition(
            name="window_hardware",
            category=AttributeCategory.WINDOW_HARDWARE,
            allowed_values=(
                "cam_handle",
                "sash_lock",
                "folding_crank",
                "espagnolette_bolt",
                "friction_stay",
            ),
            description="Visible window locking and opening hardware",
        ),
        # 3. Electrical sockets
        "socket_type": AttributeDefinition(
            name="socket_type",
            category=AttributeCategory.ELECTRICAL_SOCKETS,
            allowed_values=(
                "nema_5_15_duplex",
                "cee_7_3_schuko",
                "bs_1363",
                "type_c_europlug",
                "gfci_receptacle",
                "usb_integrated_receptacle",
            ),
            description="Type and standard of electrical receptacle",
        ),
        # 4. Electrical switches
        "switch_configuration": AttributeDefinition(
            name="switch_configuration",
            category=AttributeCategory.ELECTRICAL_SWITCHES,
            allowed_values=(
                "single_gang_rocker",
                "double_gang_rocker",
                "traditional_toggle",
                "push_button",
                "rotary_dimmer",
                "slide_dimmer",
                "occupancy_sensor_switch",
            ),
            description="Configuration of light switch control",
        ),
        # 5. Lighting fixtures
        "lighting_type": AttributeDefinition(
            name="lighting_type",
            category=AttributeCategory.LIGHTING_FIXTURES,
            allowed_values=(
                "recessed_can",
                "surface_flush_mount",
                "pendant_light",
                "track_lighting_head",
                "wall_sconce",
                "linear_strip_diffuser",
                "chandelier",
            ),
            description="Luminaire mounting and form factor",
        ),
        # 6. Plumbing fixtures
        "faucet_type": AttributeDefinition(
            name="faucet_type",
            category=AttributeCategory.PLUMBING_FIXTURES,
            allowed_values=(
                "single_lever_mixer",
                "dual_handle_faucet",
                "wall_mounted_spout",
                "gooseneck_commercial",
                "sensor_activated",
            ),
            description="Style and control mechanism of water faucet",
        ),
        # 7. HVAC components
        "hvac_terminal": AttributeDefinition(
            name="hvac_terminal",
            category=AttributeCategory.HVAC_COMPONENTS,
            allowed_values=(
                "linear_slot_diffuser",
                "egg_crate_return_grille",
                "four_way_ceiling_diffuser",
                "ductless_mini_split_indoor",
                "digital_thermostat",
                "baseboard_convector",
            ),
            description="Visible HVAC distribution or control terminal",
        ),
        # 8. Furniture construction
        "furniture_joinery": AttributeDefinition(
            name="furniture_joinery",
            category=AttributeCategory.FURNITURE_CONSTRUCTION,
            allowed_values=(
                "solid_hardwood",
                "engineered_veneer",
                "tubular_steel_frame",
                "molded_plywood",
                "upholstered_textile",
                "tempered_glass_insert",
                "cast_iron_base",
            ),
            description="Primary structural construction of furniture item",
        ),
        # 9. Materials & finishes
        "material_finish": AttributeDefinition(
            name="material_finish",
            category=AttributeCategory.MATERIALS_FINISHES,
            allowed_values=(
                "matte_latex_paint",
                "satin_acrylic_paint",
                "brushed_stainless_steel",
                "polished_chrome",
                "anodized_aluminum",
                "oiled_natural_oak",
                "polished_carrara_marble",
                "glazed_ceramic_tile",
                "matte_porcelain_tile",
            ),
            description="Surface finish and material treatment",
        ),
        # 10. Structural elements
        "structural_profile": AttributeDefinition(
            name="structural_profile",
            category=AttributeCategory.STRUCTURAL_ELEMENTS,
            allowed_values=(
                "cast_in_place_rc_column",
                "structural_steel_i_beam",
                "dimensional_timber_stud",
                "engineered_lvl_joist",
                "reinforced_concrete_shear_wall",
                "hollow_concrete_block_masonry",
            ),
            description="Structural cross-section and load-bearing profile",
        ),
        # 11. Small architectural details
        "trim_profile": AttributeDefinition(
            name="trim_profile",
            category=AttributeCategory.ARCHITECTURAL_DETAILS,
            allowed_values=(
                "profiled_baseboard",
                "coved_crown_moulding",
                "bullnose_window_sill",
                "beveled_door_threshold",
                "shadowline_reveal",
                "corner_bead_trim",
            ),
            description="Architectural millwork or boundary reveal profile",
        ),
    }

    @classmethod
    def all_categories(cls) -> set[AttributeCategory]:
        """Return all categories present in the taxonomy."""
        return set(AttributeCategory)

    @classmethod
    def get_category_attributes(cls, category: AttributeCategory) -> list[AttributeDefinition]:
        """Return all attribute definitions under a category."""
        return [defn for defn in cls._DEFINITIONS.values() if defn.category == category]

    @classmethod
    def get_definition(cls, attribute_name: str) -> AttributeDefinition | None:
        """Look up definition for attribute name."""
        return cls._DEFINITIONS.get(attribute_name)

    @classmethod
    def register_definition(cls, definition: AttributeDefinition) -> None:
        """Register a new attribute definition dynamically."""
        cls._DEFINITIONS[definition.name] = definition

    @classmethod
    def all_definitions(cls) -> dict[str, AttributeDefinition]:
        """Return copy of all attribute definitions."""
        return dict(cls._DEFINITIONS)

    @classmethod
    def validate_prediction(cls, attribute_name: str, value: str) -> bool:
        """Check whether value is valid for the given attribute."""
        defn = cls._DEFINITIONS.get(attribute_name)
        if defn is None:
            return True  # Open taxonomy for uncatalogued attributes
        return defn.is_valid_value(value)
