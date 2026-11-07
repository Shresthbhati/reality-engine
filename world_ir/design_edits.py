"""Design edits to furniture (plan T-4b.1): move, rotate, delete, replace from the catalog.

Pure functions on a WorldIR. Each edit is also stored as a command in ``world.metadata["design_commands"]`` with provenance
GENERATED (a design decision, not an observation), and the position/rotation/properties it changes are recorded for
``world_ir.edit_carry`` so a later rebuild from the photos replays them. Deleting is the exception: a deleted piece is gone from
this version, but a rebuild that detects the same object again will bring it back (flagged in the plan).

Only FURNITURE entities can be edited here; walls, floors and openings are the survey and are never touched by a design edit.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace as _replace
from typing import Any, Dict

from provenance import Provenance
from world_ir import EntityType
from world_ir.edit_carry import record_edit
from world_ir.furniture_catalog import CATALOG
from world_ir.measure import dimension_from_metadata, scale_state_of
from world_ir.schema_v1 import GeometryType, Vector3

MAX_COMMANDS = 500
COORD_LIMIT_M = 1000.0


class DesignEditError(ValueError):
    """The edit was refused; nothing was changed."""


def _num(value: Any, name: str, limit: float = COORD_LIMIT_M) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
        raise DesignEditError(f"{name} must be a finite number within +/-{limit:g}")
    return float(value)


def _furniture(world, entity_id: str):
    e = world.entities.get(entity_id)
    if e is None:
        raise DesignEditError(f"entity '{entity_id}' not found")
    if e.type != EntityType.FURNITURE:
        raise DesignEditError(f"'{entity_id}' is a {getattr(e.type, 'value', e.type)}, not furniture; design edits only change furniture")
    return e


def _dims(entity) -> tuple[float, float, float] | None:
    d = (entity.custom_properties or {}).get("dimensions") or {}
    try:
        return tuple(float(d[k]["value"]) for k in ("length_m", "width_m", "height_m"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def _refit_boxes(world, entity) -> None:
    """Keep the entity's BOX geometry (what exports and spatial queries read) in step with its position, size and turn."""
    pos = (entity.transform or {}).get("position")
    dims = _dims(entity)
    if not pos or dims is None:
        return
    length, width, height = dims
    yaw = float((entity.custom_properties or {}).get("yaw_rad") or 0.0)
    hx = abs(math.cos(yaw)) * length / 2 + abs(math.sin(yaw)) * width / 2
    hz = abs(math.sin(yaw)) * length / 2 + abs(math.cos(yaw)) * width / 2
    for gid in entity.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.type == GeometryType.BOX:
            world.geometries[gid] = _replace(
                g, bounds_min=Vector3(pos["x"] - hx, pos["y"], pos["z"] - hz), bounds_max=Vector3(pos["x"] + hx, pos["y"] + height, pos["z"] + hz))


def _log(world, op: str, entity_id: str, args: Dict[str, Any]) -> dict:
    cmd = {"op": op, "entity_id": entity_id, "args": args, "provenance": Provenance.GENERATED.value, "at": time.time()}
    world.metadata["design_commands"] = ((world.metadata.get("design_commands") or []) + [cmd])[-MAX_COMMANDS:]
    return cmd


def _yaw_quaternion(yaw: float) -> Dict[str, float]:
    return {"w": math.cos(yaw / 2), "x": 0.0, "y": math.sin(yaw / 2), "z": 0.0}


def move(world, entity_id: str, x: Any, z: Any, y: Any = None) -> dict:
    e = _furniture(world, entity_id)
    pos = dict((e.transform or {}).get("position") or {"x": 0.0, "y": 0.0, "z": 0.0})
    pos.update(x=_num(x, "x"), z=_num(z, "z"))
    if y is not None:
        pos["y"] = _num(y, "y")
    e.transform = {**(e.transform or {}), "position": pos}
    _refit_boxes(world, e)
    record_edit(world, e, "transform", e.transform)
    return _log(world, "move", entity_id, {"x": pos["x"], "z": pos["z"], "y": pos["y"]})


def rotate(world, entity_id: str, yaw_rad: Any) -> dict:
    e = _furniture(world, entity_id)
    yaw = _num(yaw_rad, "yaw_rad", limit=4 * math.pi)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))                       # normalised to (-pi, pi]
    e.transform = {**(e.transform or {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}), "rotation": _yaw_quaternion(yaw)}
    cp = dict(e.custom_properties or {})
    cp["yaw_rad"], cp["yaw_basis"] = yaw, "user"
    e.custom_properties = cp
    _refit_boxes(world, e)
    record_edit(world, e, "transform", e.transform)
    record_edit(world, e, "custom_properties.yaw_rad", yaw)
    record_edit(world, e, "custom_properties.yaw_basis", "user")
    return _log(world, "rotate", entity_id, {"yaw_rad": yaw})


def delete(world, entity_id: str) -> dict:
    e = _furniture(world, entity_id)
    own = set(e.geometry_ids)
    for other in world.entities.values():
        if other.id != entity_id:
            own -= set(other.geometry_ids)                               # a geometry another entity still uses stays
    for gid in own:
        world.geometries.pop(gid, None)
    del world.entities[entity_id]
    for other in world.entities.values():
        other.relationships = [r for r in other.relationships if r.target_id != entity_id]
        other.component_ids = [c for c in other.component_ids if c != entity_id]
    world.metadata["user_edits"] = [r for r in (world.metadata.get("user_edits") or []) if r.get("entity_id") != entity_id]
    return _log(world, "delete", entity_id, {"geometries_removed": sorted(own)})


def replace(world, entity_id: str, catalog_id: Any) -> dict:
    e = _furniture(world, entity_id)
    item = CATALOG.get(catalog_id) if isinstance(catalog_id, str) else None
    if item is None:
        raise DesignEditError(f"unknown catalog item {catalog_id!r}; see GET /api/furniture-catalog")
    if scale_state_of(world.metadata) != "metric":
        raise DesignEditError("This world has no real-world scale yet, so a catalog size in metres would be meaningless. "
                              "Use 'Set scale' with a known length first, then swap the piece.")
    basis = f"catalog default for '{item.name}' (nominal, not measured)"
    dims = {key: dimension_from_metadata(value, world.metadata, state="ESTIMATED", basis=basis)
            for key, value in (("length_m", item.length_m), ("width_m", item.width_m), ("height_m", item.height_m))}
    old = (e.custom_properties or {}).get("entity_class")
    cp = dict(e.custom_properties or {})
    for stale in ("mean_color_rgb", "colour_basis", "color_basis", "flags"):    # measured from the OLD object's pixels
        cp.pop(stale, None)
    cp.update(category=item.category, entity_class=f"furniture.{item.category}", dimensions=dims,
              classification_basis="user", catalog_id=item.id)
    e.custom_properties, e.name, e.provenance = cp, item.name, Provenance.GENERATED
    _refit_boxes(world, e)
    for field, value in (("name", item.name), ("entity_class", cp["entity_class"]), ("custom_properties.category", item.category),
                         ("custom_properties.dimensions", dims), ("custom_properties.catalog_id", item.id)):
        record_edit(world, e, field, value)
    return _log(world, "replace", entity_id, {"catalog_id": item.id, "was": old})


def apply(world, entity_id: str, op: Dict[str, Any]) -> dict:
    """Run one command object ``{"op": "move"|"rotate"|"delete"|"replace", ...}``; raises DesignEditError (world unchanged on refusal)."""
    kind = op.get("op")
    if kind == "move":
        return move(world, entity_id, op.get("x"), op.get("z"), op.get("y"))
    if kind == "rotate":
        return rotate(world, entity_id, op.get("yaw_rad"))
    if kind == "delete":
        return delete(world, entity_id)
    if kind == "replace":
        return replace(world, entity_id, op.get("catalog_id"))
    raise DesignEditError("op must be one of move, rotate, delete, replace")
