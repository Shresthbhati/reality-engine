"""M3: ranked material HYPOTHESES from a surface's measured colour and uniformity.

A colour is weak evidence of a material: warm brown does not prove wood, and off-white does not prove paint. So this module
only ever produces hypotheses -- ``state="ESTIMATED"``, ``asserted=False``, confidence capped at ``CONFIDENCE_CAP`` -- and
nothing at all when the surface colour was not measured (``not_measured``/missing: no hypothesis is invented). Each carries
the evidence it came from. A real classifier (``docs/future/perception/MATERIAL_PERCEPTION.md``) may later raise confidence;
this does not pretend to be one. The result is stored beside, never in place of, the measured colour.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from perception.materials.surface_appearance import linear_to_lab, srgb_to_linear

STATE = "ESTIMATED"
CONFIDENCE_CAP = 0.4
NOTE = "hypothesis from measured colour and uniformity only; not verified by a material classifier"
#: median inlier colour spread (CIE76) below which a surface reads as one flat finish
UNIFORM_DELTA_E = 6.0


def _lch(rgb) -> tuple:
    lab = linear_to_lab(srgb_to_linear(np.asarray(rgb, float)))
    return float(lab[0]), float(math.hypot(lab[1], lab[2])), float(math.degrees(math.atan2(lab[2], lab[1])) % 360.0)


def _hyp(material: str, score: float, basis: str, evidence: dict) -> dict:
    score = max(0.0, min(1.0, score))
    return {"material": material, "score": round(score, 3), "confidence": round(min(CONFIDENCE_CAP, score * 0.5), 3),
            "state": STATE, "asserted": False, "basis": basis, "evidence": evidence, "note": NOTE}


def hypothesize_materials(color_rgb: Optional[Sequence[float]], kind: str, *, spread_delta_e: Optional[float] = None,
                          color_state: str = "measured_approximate") -> List[dict]:
    """Ranked (best first) hypotheses for a wall/floor/ceiling; ``[]`` when the colour was not measured."""
    if color_rgb is None or color_state == "not_measured":
        return []
    L, C, h = _lch(color_rgb)
    uniform = spread_delta_e is not None and spread_delta_e <= UNIFORM_DELTA_E
    ev = {"color_rgb": [int(round(v)) for v in color_rgb], "L": round(L, 1), "chroma": round(C, 1), "hue_deg": round(h, 0),
          "spread_delta_e": None if spread_delta_e is None else round(float(spread_delta_e), 2)}
    out: List[dict] = []
    if kind in ("wall", "ceiling"):
        # most interior walls/ceilings are painted; a flat, light, low-chroma finish is the typical case
        score = 0.35 + (0.25 if uniform else 0.0) + (0.15 if L >= 60 and C < 25 else 0.0)
        out.append(_hyp("painted_wall" if kind == "wall" else "painted_ceiling", score,
                        "flat uniform finish" if uniform else "finish uniformity unknown or textured", ev))
    elif kind == "floor":
        warm = 15.0 <= h <= 85.0 and 12.0 <= C <= 55.0 and 20.0 <= L <= 75.0
        grey = C < 12.0 and 25.0 <= L <= 80.0
        out.append(_hyp("wood_floor", 0.3 + (0.35 if warm else 0.0), "warm brown hue range" if warm else "colour is not wood-like", ev))
        out.append(_hyp("concrete_or_tile_floor", 0.25 + (0.3 if grey else 0.0) + (0.1 if grey and uniform else 0.0),
                        "neutral grey" if grey else "colour is not neutral grey", ev))
    out.sort(key=lambda d: -d["score"])
    return out


def attach_material_hypotheses(entity) -> Optional[List[dict]]:
    """Store ``custom_properties['material_hypotheses']`` for an entity whose appearance was measured from photos."""
    cp = entity.custom_properties
    app = cp.get("appearance") or {}
    if cp.get("color_source") != "image_projected" or not cp.get("mean_color_rgb"):
        return None
    kind = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
    hyps = hypothesize_materials(cp["mean_color_rgb"], kind, spread_delta_e=app.get("spread_delta_e"),
                                 color_state=app.get("color_state") or cp.get("color_state") or "measured_approximate")
    if hyps:
        cp["material_hypotheses"] = hyps
    return hyps
