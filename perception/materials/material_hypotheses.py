"""M3 & Task 6.10: ranked material HYPOTHESES from a surface's measured colour and texture.

A colour is weak evidence of a material: warm brown does not prove wood, and off-white does not prove paint. So this module
only ever produces hypotheses -- ``state="ESTIMATED"``, ``asserted=False``, confidence capped at ``CONFIDENCE_CAP`` -- and
nothing at all when the surface colour was not measured (``not_measured``/missing: no hypothesis is invented). Each carries
the evidence it came from. The result is stored beside, never in place of, the measured colour.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from perception.materials.classifier import (
    CONFIDENCE_CAP_COLOR_ONLY,
    CONFIDENCE_CAP_WITH_TEXTURE,
    MaterialEvidence,
    classify_material,
)

STATE = "ESTIMATED"
CONFIDENCE_CAP = CONFIDENCE_CAP_COLOR_ONLY
NOTE = "hypothesis from measured color and texture statistics; not verified by physical contact"
UNIFORM_DELTA_E = 6.0


def hypothesize_materials(
    color_rgb: Optional[Sequence[float]],
    kind: str,
    *,
    spread_delta_e: Optional[float] = None,
    color_state: str = "measured_approximate",
    texture_roughness: Optional[float] = None,
    specular_sheen: Optional[float] = None,
    is_translucent: bool = False,
) -> List[dict]:
    """Ranked (best first) hypotheses for a surface; ``[]`` when the colour was not measured."""
    if color_rgb is None or color_state == "not_measured":
        return []

    ev = MaterialEvidence(
        color_rgb=color_rgb,
        spread_delta_e=spread_delta_e,
        texture_roughness=texture_roughness,
        specular_sheen=specular_sheen,
        surface_kind=kind,
        is_translucent=is_translucent,
        color_state=color_state,
    )
    classifications = classify_material(ev)
    return [c.to_dict() for c in classifications]


def attach_material_hypotheses(entity) -> Optional[List[dict]]:
    """Store ``custom_properties['material_hypotheses']`` for an entity whose appearance was measured from photos."""
    cp = entity.custom_properties
    app = cp.get("appearance") or {}
    if cp.get("color_source") != "image_projected" or not cp.get("mean_color_rgb"):
        return None
    kind = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
    hyps = hypothesize_materials(
        cp["mean_color_rgb"],
        kind,
        spread_delta_e=app.get("spread_delta_e"),
        color_state=app.get("color_state") or cp.get("color_state") or "measured_approximate",
        texture_roughness=app.get("texture_roughness"),
        specular_sheen=app.get("specular_sheen"),
    )
    if hyps:
        cp["material_hypotheses"] = hyps
    return hyps
