"""Surface finishes as a design layer (plan T-4b.2): wall paint, floor material.

A finish is stored on the wall / floor / ceiling entity as ``custom_properties["design_finish"]`` and NOTHING else changes: the
measured colour (``mean_color_rgb``), the photo texture and the geometry are the survey and are never edited here. The viewer
draws the finish in its normal view and ignores it in the evidence view; the as-built exports do not read it (tested).

The colours are approximate on-screen swatches, not matched to any paint brand or flooring product.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from provenance import Provenance
from world_ir import EntityType
from world_ir.design_edits import DesignEditError, _log
from world_ir.edit_carry import record_edit

SURFACE_TYPES = (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)
MAX_TARGETS = 200


@dataclass(frozen=True)
class Swatch:
    id: str
    name: str
    rgb: tuple


FLOOR_MATERIALS: Dict[str, Swatch] = {s.id: s for s in (
    Swatch("oak", "Oak boards", (196, 154, 108)), Swatch("light-oak", "Light oak", (222, 196, 150)),
    Swatch("walnut", "Walnut", (93, 64, 46)), Swatch("concrete", "Polished concrete", (150, 150, 148)),
    Swatch("white-tile", "White tile", (232, 232, 228)), Swatch("grey-tile", "Grey tile", (128, 130, 134)),
    Swatch("carpet-grey", "Grey carpet", (110, 110, 114)), Swatch("slate", "Black slate", (52, 54, 58)),
)}

PAINTS: Dict[str, Swatch] = {s.id: s for s in (
    Swatch("warm-white", "Warm white", (240, 235, 225)), Swatch("soft-grey", "Soft grey", (200, 200, 198)),
    Swatch("sage", "Sage", (178, 190, 168)), Swatch("terracotta", "Terracotta", (190, 110, 80)),
    Swatch("navy", "Navy", (40, 52, 84)), Swatch("charcoal", "Charcoal", (60, 62, 66)),
)}


def finish_options() -> dict:
    row = lambda s: {"id": s.id, "name": s.name, "rgb": list(s.rgb)}   # noqa: E731
    return {"floor_materials": [row(s) for s in FLOOR_MATERIALS.values()], "paints": [row(s) for s in PAINTS.values()],
            "note": "Approximate on-screen colours, not matched to any paint or flooring product."}


def _rgb(value: Any) -> List[int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in value)):
        raise DesignEditError("colour_rgb must be three whole numbers from 0 to 255")
    return [int(v) for v in value]


def _targets(world, entity_id: str, also: Optional[Iterable[str]]) -> List[Any]:
    ids = [entity_id] + [i for i in (also or []) if i != entity_id]
    if len(ids) > MAX_TARGETS:
        raise DesignEditError(f"too many surfaces in one edit (max {MAX_TARGETS})")
    out = []
    for eid in ids:
        e = world.entities.get(eid)
        if e is None:
            raise DesignEditError(f"entity '{eid}' not found")
        if e.type not in SURFACE_TYPES:
            raise DesignEditError(f"'{eid}' is a {getattr(e.type, 'value', e.type)}; finishes apply to walls, floors and ceilings")
        out.append(e)
    return out


def set_finish(world, entity_id: str, colour_rgb: Any = None, material: Any = None, paint: Any = None,
               also_entity_ids: Optional[Iterable[str]] = None) -> dict:
    """Give the surface(s) a flat colour (``colour_rgb``), a named paint, or - floors only - a named material."""
    given = [x for x in (colour_rgb, material, paint) if x is not None]
    if len(given) != 1:
        raise DesignEditError("give exactly one of colour_rgb, paint or material")
    targets = _targets(world, entity_id, also_entity_ids)
    if material is not None:
        swatch = FLOOR_MATERIALS.get(material) if isinstance(material, str) else None
        if swatch is None:
            raise DesignEditError(f"unknown floor material {material!r}; see GET /api/finishes")
        if any(t.type != EntityType.FLOOR for t in targets):
            raise DesignEditError("floor materials apply to floors only; use a paint or colour for walls and ceilings")
        finish = {"kind": "material", "material": swatch.id, "name": swatch.name, "colour_rgb": list(swatch.rgb)}
    elif paint is not None:
        swatch = PAINTS.get(paint) if isinstance(paint, str) else None
        if swatch is None:
            raise DesignEditError(f"unknown paint {paint!r}; see GET /api/finishes")
        finish = {"kind": "paint", "paint": swatch.id, "name": swatch.name, "colour_rgb": list(swatch.rgb)}
    else:
        rgb = _rgb(colour_rgb)
        finish = {"kind": "colour", "name": "Custom colour", "colour_rgb": rgb}
    finish["provenance"] = Provenance.GENERATED.value
    for t in targets:
        cp = dict(t.custom_properties or {})
        cp["design_finish"] = dict(finish)
        t.custom_properties = cp
        record_edit(world, t, "custom_properties.design_finish", dict(finish))
    return _log(world, "finish", entity_id, {"finish": finish, "entity_ids": [t.id for t in targets]})


def clear_finish(world, entity_id: str, also_entity_ids: Optional[Iterable[str]] = None) -> dict:
    """Back to the survey's own colour and texture."""
    targets = _targets(world, entity_id, also_entity_ids)
    for t in targets:
        cp = dict(t.custom_properties or {})
        cp.pop("design_finish", None)
        t.custom_properties = cp
        record_edit(world, t, "custom_properties.design_finish", None)
    return _log(world, "clear_finish", entity_id, {"entity_ids": [t.id for t in targets]})
