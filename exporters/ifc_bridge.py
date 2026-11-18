"""WorldIR -> IFC bridge (P8-03: "WorldIR -> IFC via IfcOpenShell
(adapter, license recorded); do not rebuild BIM from scratch").

Uses the REAL ifcopenshell 0.8.x API: a proper IFC4 project (IfcProject
-> IfcSite -> IfcBuilding -> IfcBuildingStorey spatial hierarchy) and
one semantic IFC class per WorldIR architectural entity -- IfcWall,
IfcSlab (floor/ceiling), IfcColumn, IfcBeam, IfcDoor, IfcWindow,
IfcOpeningElement, IfcRoof, IfcBuildingElementProxy for anything else
with bounds. Each element gets real IfcExtrudedAreaSolid geometry from
its measured bounds_min/bounds_max (the same AABB every other exporter
consumes) and a Name carrying "<worldir_id> <name>" so the reverse path
(IFC -> WorldIR, later) can trace elements back to their evidence.

Openings and Voiding (Epic E04-2):
  - Doors, windows, and opening elements hosted by an IfcWall are carved
    using negative boolean voids via IfcOpeningElement + IfcRelVoidsElement.
  - Doors and windows filling the opening are connected to their opening
    via IfcRelFillsElement.
  - Clashes between solid host walls and embedded doors/windows are
    completely eliminated in BIM/IFC tools.

Honesty rules:

  - Entities without real bounds are skipped and REPORTED in the
    returned report, never invented.
  - A world with no exportable entities at all raises IFCBridgeError
    -- an empty BIM file would look like success while meaning nothing.
  - If ifcopenshell is not installed, importing this module raises
    ImportError with instructions; nothing degrades to fake output.
  - DETERMINISTIC: ifcopenshell normally mints random (uuid4) GlobalIds
    and stamps the header with the wall-clock time; this bridge
    overrides every GlobalId with a uuid5 derived from the WorldIR id +
    element identity and pins the header time_stamp to the epoch, so
    the same world -> byte-identical IFC (matching the determinism
    contract of the other exporters).

License: ifcopenshell is LGPL-3.0; recorded per the ledger's
LICENSES.yaml convention (see the ledger sync in this change).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR

#: WorldIR EntityType -> IFC4 element class. Everything not listed
#: exports as IfcBuildingElementProxy (the honest IFC answer for "a
#: real element whose BIM class is unknown").
_ENTITY_IFC_CLASS = {
    EntityType.WALL.value: "IfcWall",
    EntityType.FLOOR.value: "IfcSlab",
    EntityType.CEILING.value: "IfcSlab",
    EntityType.ROOF.value: "IfcRoof",
    EntityType.COLUMN.value: "IfcColumn",
    EntityType.BEAM.value: "IfcBeam",
    EntityType.DOOR.value: "IfcDoor",
    EntityType.WINDOW.value: "IfcWindow",
    EntityType.OPENING.value: "IfcOpeningElement",
    EntityType.STAIRS.value: "IfcStair",
    EntityType.BUILDING.value: "IfcBuilding",
    EntityType.ROOM.value: "IfcSpace",
    EntityType.FURNITURE.value: "IfcFurnishingElement",
    EntityType.STOREY.value: "IfcBuildingStorey",
    EntityType.LEVEL.value: "IfcBuildingStorey",
    EntityType.CORRIDOR.value: "IfcSpace",
}

_IMPORT_ERROR_MESSAGE = (
    "ifcopenshell is required for the IFC bridge "
    "(pip install ifcopenshell, LGPL-3.0); refusing to fake BIM output"
)


#: Pinned IFC header time_stamp. ifcopenshell stamps the wall-clock
#: time at file creation, which would break the repo's determinism
#: contract (same world -> byte-identical export); the epoch is a
#: fixed, honest "no authoring-time information" value.
_FIXED_HEADER_TIMESTAMP = "1970-01-01T00:00:00"


class IFCBridgeError(ValueError):
    pass


def _deterministic_guid(seed: str) -> str:
    """A valid 22-char IFC GlobalId derived deterministically from
    `seed` (uuid5 -> ifcopenshell's base64 compression). Same seed ->
    same GUID, forever; distinct seeds -> distinct GUIDs with the same
    collision resistance as uuid4."""
    import uuid

    import ifcopenshell.guid

    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"reality-engine://ifc/{seed}").hex
    return ifcopenshell.guid.compress(digest)


def _normalize_unit_order(model) -> None:
    """Sort the project's IfcUnitAssignment.Units deterministically.

    ifcopenshell's unit.assign_unit collects units in a Python `set`
    of entity instances whose iteration order is id-hash based, i.e.
    it varies between processes; serializing that list as-is breaks
    byte determinism. Sorting by (class, UnitType, Name) gives a
    stable, schema-meaningful order."""
    for project in model.by_type("IfcProject"):
        assignment = project.UnitsInContext
        if assignment is None:
            continue
        units = sorted(
            assignment.Units,
            key=lambda u: (
                u.is_a(),
                str(getattr(u, "UnitType", "")),
                str(getattr(u, "Name", "")),
            ),
        )
        assignment.Units = units


def _reassign_relationship_guids(model, world_id: str) -> None:
    """Give every relationship object (IfcRelAggregates from
    root.create_entity, IfcRelContainedInSpatialStructure from
    spatial.assign_container, IfcRelVoidsElement and IfcRelFillsElement
    from feature operations) a deterministic GlobalId. These are
    created implicitly by the API with random GUIDs; their seeds use
    the (already deterministic) GlobalIds of the objects they relate,
    so the assignment order and values are stable."""
    # Deterministic GUIDs for all property sets
    for pset in sorted(model.by_type("IfcPropertySet"), key=lambda p: (str(getattr(p, "Name", "")), p.id())):
        pset.GlobalId = _deterministic_guid(f"{world_id}/pset/{pset.Name}/{pset.id()}")

    rels = (
        list(model.by_type("IfcRelAggregates"))
        + list(model.by_type("IfcRelContainedInSpatialStructure"))
        + list(model.by_type("IfcRelVoidsElement"))
        + list(model.by_type("IfcRelFillsElement"))
        + list(model.by_type("IfcRelDefinesByProperties"))
    )

    def _key(rel):
        relating = (
            getattr(rel, "RelatingObject", None)
            or getattr(rel, "RelatingStructure", None)
            or getattr(rel, "RelatingBuildingElement", None)
            or getattr(rel, "RelatingOpeningElement", None)
            or getattr(rel, "RelatingPropertyDefinition", None)
        )
        related = (
            getattr(rel, "RelatedObjects", None)
            or getattr(rel, "RelatedElements", None)
            or getattr(rel, "RelatedOpeningElement", None)
            or getattr(rel, "RelatedBuildingElement", None)
            or ()
        )
        if not isinstance(related, (list, tuple, set)) and related is not None:
            related = (related,)
        relating_id = relating.GlobalId if relating else ""
        related_ids = tuple(sorted(o.GlobalId for o in related if o is not None))
        return (rel.is_a(), relating_id, related_ids)

    for rel in sorted(rels, key=_key):
        cls, relating, related = _key(rel)
        rel_str = "+".join(related)
        rel.GlobalId = _deterministic_guid(
            f"{world_id}/rel/{cls}/{relating}/{rel_str}")
        if getattr(rel, "RelatedObjects", None):
            rel.RelatedObjects = sorted(
                rel.RelatedObjects, key=lambda o: o.GlobalId)
        elif getattr(rel, "RelatedElements", None):
            rel.RelatedElements = sorted(
                rel.RelatedElements, key=lambda o: o.GlobalId)


def _bounds_extent(geom) -> Optional[Tuple[float, float, float]]:
    """Real AABB extent (dx, dy, dz) from measured bounds, or None."""
    if geom.bounds_min is None or geom.bounds_max is None:
        return None
    dx = geom.bounds_max.x - geom.bounds_min.x
    dy = geom.bounds_max.y - geom.bounds_min.y
    dz = geom.bounds_max.z - geom.bounds_min.z
    if dx <= 0 or dy <= 0 or dz <= 0:
        # Degenerate bounds: floor at 1 cm like the other exporters so
        # a zero-thickness plane still produces a usable solid.
        return (max(dx, 0.01), max(dy, 0.01), max(dz, 0.01))
    return (dx, dy, dz)


def _get_primary_geom(world, entity: Entity) -> Optional[Geometry]:
    """Return the first geometry with valid bounds attached to the entity."""
    for gid in entity.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            return g
    return None


def _find_host_wall_id(world, entity: Entity, wall_ids: Sequence[str]) -> Optional[str]:
    """Find the host wall entity ID for a door, window, or opening."""
    # 1. in_wall custom property
    in_wall = entity.custom_properties.get("in_wall")
    if isinstance(in_wall, dict) and in_wall.get("wall_id") in wall_ids:
        return in_wall["wall_id"]

    # 2. explicit relationships (PART_OF or ATTACHED_TO)
    for rel in entity.relationships:
        if rel.target_id in wall_ids and rel.kind in (RelationshipKind.PART_OF, RelationshipKind.ATTACHED_TO):
            return rel.target_id

    # 3. wall slab openings list or CONTAINS relationship on candidate walls
    for wid in wall_ids:
        wall = world.entities.get(wid)
        if not wall:
            continue
        ws = wall.custom_properties.get("wall_slab")
        if isinstance(ws, dict):
            for op in ws.get("openings", []):
                if isinstance(op, dict) and op.get("entity_id") == entity.id:
                    return wid
        for rel in wall.relationships:
            if rel.target_id == entity.id and rel.kind == RelationshipKind.CONTAINS:
                return wid

    # 4. spatial bounding box overlap check (within 15 cm tolerance)
    e_geom = _get_primary_geom(world, entity)
    if e_geom and e_geom.bounds_min and e_geom.bounds_max:
        eb_min, eb_max = e_geom.bounds_min, e_geom.bounds_max
        best_wall = None
        min_dist = float("inf")
        for wid in wall_ids:
            w_geom = _get_primary_geom(world, world.entities[wid])
            if not w_geom or not w_geom.bounds_min or not w_geom.bounds_max:
                continue
            wb_min, wb_max = w_geom.bounds_min, w_geom.bounds_max
            overlap_x = min(eb_max.x, wb_max.x) - max(eb_min.x, wb_min.x)
            overlap_y = min(eb_max.y, wb_max.y) - max(eb_min.y, wb_min.y)
            overlap_z = min(eb_max.z, wb_max.z) - max(eb_min.z, wb_min.z)
            if overlap_x >= -0.15 and overlap_y >= -0.15 and overlap_z >= -0.15:
                ec = ((eb_min.x + eb_max.x) / 2, (eb_min.y + eb_max.y) / 2, (eb_min.z + eb_max.z) / 2)
                wc = ((wb_min.x + wb_max.x) / 2, (wb_min.y + wb_max.y) / 2, (wb_min.z + wb_max.z) / 2)
                d = sum((a - b) ** 2 for a, b in zip(ec, wc))
                if d < min_dist:
                    min_dist = d
                    best_wall = wid
        if best_wall is not None:
            return best_wall

    return None


def _get_storey_elevation(world: WorldIR, entity: Entity) -> float:
    """Extract measured elevation for a storey entity."""
    cp = getattr(entity, "custom_properties", None) or {}
    for key in ("elevation_m", "floor_height_m", "elevation", "height_m"):
        val = cp.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    geom = _get_primary_geom(world, entity)
    if geom and geom.bounds_min is not None:
        return float(geom.bounds_min.z)
    return 0.0


def _resolve_target_storey(
    world: WorldIR,
    entity: Entity,
    storeys_by_id: dict[str, tuple[Any, float]],
    default_storey: Any,
) -> Any:
    """Find the matching IfcBuildingStorey for an entity."""
    # 1. Direct relationship to storey
    for rel in getattr(entity, "relationships", ()):
        if rel.target_id in storeys_by_id and rel.kind in (
            RelationshipKind.PART_OF,
            RelationshipKind.INSIDE,
            RelationshipKind.ATTACHED_TO,
            "part_of",
            "inside",
        ):
            return storeys_by_id[rel.target_id][0]

    # 2. Parent ID is a storey
    parent_id = getattr(entity, "parent_id", None)
    if parent_id and parent_id in storeys_by_id:
        return storeys_by_id[parent_id][0]

    # 3. custom_properties["storey_id"]
    cp = getattr(entity, "custom_properties", None) or {}
    s_id = cp.get("storey_id")
    if s_id and s_id in storeys_by_id:
        return storeys_by_id[s_id][0]

    # 4. Indirect through a parent room
    for rel in getattr(entity, "relationships", ()):
        if rel.kind in (RelationshipKind.INSIDE, RelationshipKind.PART_OF, "inside", "part_of"):
            target_ent = world.entities.get(rel.target_id)
            if target_ent and target_ent.type == EntityType.ROOM:
                for r_rel in getattr(target_ent, "relationships", ()):
                    if r_rel.target_id in storeys_by_id:
                        return storeys_by_id[r_rel.target_id][0]
                room_s_id = (getattr(target_ent, "custom_properties", None) or {}).get("storey_id")
                if room_s_id and room_s_id in storeys_by_id:
                    return storeys_by_id[room_s_id][0]

    # 5. By elevation matching if multiple storeys exist
    real_storeys = {k: v for k, v in storeys_by_id.items() if k != "_default"}
    if len(real_storeys) > 1:
        geom = _get_primary_geom(world, entity)
        if geom and geom.bounds_min is not None:
            z = geom.bounds_min.z
            candidate = None
            cand_elev = -float("inf")
            for _sid, (s_elem, elev) in real_storeys.items():
                if elev <= z + 0.15 and elev >= cand_elev:
                    cand_elev = elev
                    candidate = s_elem
            if candidate is not None:
                return candidate

    return default_storey


def _make_opening_solid(builder, wall_geom: Geometry, opening_geom: Geometry):
    """Construct an extruded solid for an opening cutting completely through wall thickness."""
    wbmin, wbmax = wall_geom.bounds_min, wall_geom.bounds_max
    obmin, obmax = opening_geom.bounds_min, opening_geom.bounds_max
    wdx = wbmax.x - wbmin.x
    wdy = wbmax.y - wbmin.y
    wdz = wbmax.z - wbmin.z
    odx = max(obmax.x - obmin.x, 0.05)
    ody = max(obmax.y - obmin.y, 0.05)
    odz = max(obmax.z - obmin.z, 0.05)

    if wdx >= wdy:
        # Wall lies mostly along X; thickness is along Y
        y_min = wbmin.y - 0.02
        y_max = wbmax.y + 0.02
        cut_dy = max(y_max - y_min, ody + 0.04)
        cut_pos_y = y_min
        cut_dx = min(odx, wdx)
        cut_pos_x = obmin.x
        cut_dz = min(odz, wdz)
        cut_pos_z = max(wbmin.z, obmin.z)
    else:
        # Wall lies mostly along Y; thickness is along X
        x_min = wbmin.x - 0.02
        x_max = wbmax.x + 0.02
        cut_dx = max(x_max - x_min, odx + 0.04)
        cut_pos_x = x_min
        cut_dy = min(ody, wdy)
        cut_pos_y = obmin.y
        cut_dz = min(odz, wdz)
        cut_pos_z = max(wbmin.z, obmin.z)

    profile = builder.rectangle(
        size=(cut_dx, cut_dy),
        position=(cut_pos_x, cut_pos_y),
    )
    return builder.extrude(profile, magnitude=cut_dz, position=(0.0, 0.0, cut_pos_z))


def _make_slab_opening_solid(builder, ws: dict, op: dict, wall_geom: Geometry):
    """Construct an opening cutout solid from wall_slab parameters and in-wall opening rect."""
    offset = float(op.get("offset_m", 0.0))
    sill = float(op.get("sill_m", 0.0))
    width = float(op.get("width_m", 0.9))
    height = float(op.get("height_m", 2.1))

    start = ws.get("start_xyz_m")
    d = ws.get("direction")
    u = ws.get("up")

    if start and d and u:
        import numpy as np

        p0 = np.array(start, float) + offset * np.array(d, float) + sill * np.array(u, float)
        p1 = p0 + width * np.array(d, float) + height * np.array(u, float)
        obmin = (min(p0[0], p1[0]), min(p0[1], p1[1]), min(p0[2], p1[2]))
        obmax = (max(p0[0], p1[0]), max(p0[1], p1[1]), max(p0[2], p1[2]))

        wbmin, wbmax = wall_geom.bounds_min, wall_geom.bounds_max
        wdx = wbmax.x - wbmin.x
        wdy = wbmax.y - wbmin.y
        wdz = wbmax.z - wbmin.z

        odx = max(obmax[0] - obmin[0], 0.05)
        ody = max(obmax[1] - obmin[1], 0.05)
        odz = max(obmax[2] - obmin[2], 0.05)

        if wdx >= wdy:
            cut_dy = wdy + 0.04
            cut_pos_y = (wbmin.y + wbmax.y) / 2.0
            cut_dx = min(odx, wdx)
            cut_pos_x = (obmin[0] + obmax[0]) / 2.0
            cut_dz = min(odz, wdz)
            cut_pos_z = max(wbmin.z, obmin[2])
        else:
            cut_dx = wdx + 0.04
            cut_pos_x = (wbmin.x + wbmax.x) / 2.0
            cut_dy = min(ody, wdy)
            cut_pos_y = (obmin[1] + obmax[1]) / 2.0
            cut_dz = min(odz, wdz)
            cut_pos_z = max(wbmin.z, obmin[2])

        profile = builder.rectangle(size=(cut_dx, cut_dy), position=(cut_pos_x, cut_pos_y))
        return builder.extrude(profile, magnitude=cut_dz, position=(0.0, 0.0, cut_pos_z))
    else:
        return _make_opening_solid(builder, wall_geom, wall_geom)



def _resolve_ifc_class(entity: Entity) -> str:
    entity_class = entity.custom_properties.get("entity_class", "")
    if isinstance(entity_class, str) and entity_class:
        ec = entity_class.lower()
        if ec.startswith("electrical.socket") or ec.startswith("electrical.plug"):
            return "IfcOutlet"
        if ec.startswith("electrical.switch"):
            return "IfcSwitchingDevice"
        if ec.startswith("electrical.light"):
            return "IfcLightFixture"
        if ec.startswith("plumbing"):
            return "IfcSanitaryTerminal"
        if ec.startswith("furniture"):
            return "IfcFurnishingElement"
    t_val = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
    if t_val in _ENTITY_IFC_CLASS:
        return _ENTITY_IFC_CLASS[t_val]

    # Map ontology paths (e.g. storage/almirah, doors/hinged_door, electrical/socket_outlet)
    path = t_val.lower().replace(".", "/")
    if path.startswith("storage") or path.startswith("furniture") or path.startswith("seating"):
        return "IfcFurnishingElement"
    if path.startswith("doors") or "/door" in path:
        return "IfcDoor"
    if path.startswith("windows") or "/window" in path:
        return "IfcWindow"
    if path.startswith("structure/wall") or "/wall" in path:
        return "IfcWall"
    if path.startswith("structure/floor") or "/floor" in path or path.startswith("structure/ceiling") or "/ceiling" in path:
        return "IfcSlab"
    if path.startswith("structure/column") or "/column" in path:
        return "IfcColumn"
    if path.startswith("structure/beam") or "/beam" in path:
        return "IfcBeam"
    if path.startswith("structure/stair") or "/stair" in path:
        return "IfcStair"
    if "socket" in path or "plug" in path:
        return "IfcOutlet"
    if "switchboard" in path or "distribution_board" in path:
        return "IfcDistributionBoard"
    if "switch" in path:
        return "IfcSwitchingDevice"
    if "light" in path:
        return "IfcLightFixture"
    if any(k in path for k in ("tap", "basin", "wc", "sink", "toilet")):
        return "IfcSanitaryTerminal"

    return "IfcBuildingElementProxy"


def _enrich_element_property_sets(ifc_api, model, element, entity: Entity, extent: Tuple[float, float, float]) -> None:
    """Enrich IFC element with buildingSMART standard property sets (Epic E10-1)."""
    ifc_class = element.is_a()
    name = entity.name or entity.id
    props = entity.custom_properties

    # 1. Pset_WallCommon
    if ifc_class == "IfcWall":
        p_wall = ifc_api.run("pset.add_pset", model, product=element, name="Pset_WallCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_wall, properties={
            "Reference": name,
            "LoadBearing": bool(props.get("load_bearing", True)),
            "IsExternal": bool(props.get("is_external", False)),
            "ExtendToStructure": True,
            "FireRating": str(props.get("fire_rating", "Unrated")),
            "GrossSideArea": float(round(max(extent[0], extent[1]) * extent[2], 3)),
            "NominalThickness": float(round(min(extent[0], extent[1]), 3)),
        })

    # 2. Pset_SpaceCommon (Rooms)
    elif ifc_class == "IfcSpace":
        p_space = ifc_api.run("pset.add_pset", model, product=element, name="Pset_SpaceCommon")
        area = float(round(extent[0] * extent[1], 3))
        volume = float(round(extent[0] * extent[1] * extent[2], 3))
        ifc_api.run("pset.edit_pset", model, pset=p_space, properties={
            "Reference": name,
            "IsExternal": bool(props.get("is_external", False)),
            "GrossFloorArea": area,
            "NetFloorArea": area,
            "GrossVolume": volume,
            "Height": float(round(extent[2], 3)),
        })

    # 3. Pset_DoorCommon
    elif ifc_class == "IfcDoor":
        p_door = ifc_api.run("pset.add_pset", model, product=element, name="Pset_DoorCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_door, properties={
            "Reference": name,
            "IsExternal": bool(props.get("is_external", False)),
            "FireRating": str(props.get("fire_rating", "Unrated")),
            "AcousticRating": str(props.get("acoustic_rating", "Unrated")),
            "SecurityRating": str(props.get("security_rating", "Unrated")),
        })

    # 4. Pset_WindowCommon
    elif ifc_class == "IfcWindow":
        p_win = ifc_api.run("pset.add_pset", model, product=element, name="Pset_WindowCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_win, properties={
            "Reference": name,
            "IsExternal": bool(props.get("is_external", True)),
            "FireRating": str(props.get("fire_rating", "Unrated")),
            "AcousticRating": str(props.get("acoustic_rating", "Unrated")),
            "GlazingArea": float(round(max(extent[0], extent[1]) * extent[2], 3)),
        })

    # 5. Pset_SlabCommon
    elif ifc_class == "IfcSlab":
        p_slab = ifc_api.run("pset.add_pset", model, product=element, name="Pset_SlabCommon")
        is_floor = (getattr(entity.type, "value", entity.type) == EntityType.FLOOR.value)
        ifc_api.run("pset.edit_pset", model, pset=p_slab, properties={
            "Reference": name,
            "LoadBearing": bool(props.get("load_bearing", is_floor)),
            "IsExternal": bool(props.get("is_external", False)),
            "GrossArea": float(round(extent[0] * extent[1], 3)),
            "Thickness": float(round(extent[2], 3)),
        })

    # 6. Pset_ColumnCommon
    elif ifc_class == "IfcColumn":
        p_col = ifc_api.run("pset.add_pset", model, product=element, name="Pset_ColumnCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_col, properties={
            "Reference": name,
            "LoadBearing": True,
            "IsExternal": bool(props.get("is_external", False)),
        })

    # 7. Pset_BeamCommon
    elif ifc_class == "IfcBeam":
        p_beam = ifc_api.run("pset.add_pset", model, product=element, name="Pset_BeamCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_beam, properties={
            "Reference": name,
            "LoadBearing": True,
            "IsExternal": bool(props.get("is_external", False)),
        })

    # 8. Pset_OutletCommon
    elif ifc_class == "IfcOutlet":
        p_out = ifc_api.run("pset.add_pset", model, product=element, name="Pset_OutletCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_out, properties={
            "Reference": name,
            "NominalVoltage": float(props.get("voltage", 230.0)),
            "NumberOfGangs": int(props.get("gangs", 1)),
        })

    # 9. Pset_SwitchCommon
    elif ifc_class == "IfcSwitchingDevice":
        p_sw = ifc_api.run("pset.add_pset", model, product=element, name="Pset_SwitchCommon")
        ifc_api.run("pset.edit_pset", model, pset=p_sw, properties={
            "Reference": name,
            "SwitchFunction": str(props.get("function", "LIGHTING")),
            "NumberOfGangs": int(props.get("gangs", 1)),
        })

    # 10. Pset_SanitaryTerminalTypeCommon
    elif ifc_class == "IfcSanitaryTerminal":
        p_san = ifc_api.run("pset.add_pset", model, product=element, name="Pset_SanitaryTerminalTypeCommon")
        term_type = "SINK" if "sink" in (entity.name or entity.id).lower() else "TOILET"
        ifc_api.run("pset.edit_pset", model, pset=p_san, properties={
            "Reference": name,
            "SanitaryTerminalType": str(props.get("terminal_type", term_type)),
        })

    # 11. RealityEngine_Metadata
    meta = ifc_api.run("pset.add_pset", model, product=element, name="RealityEngine_Metadata")
    ifc_api.run("pset.edit_pset", model, pset=meta, properties={
        "EntityId": str(entity.id),
        "EntityType": str(getattr(entity.type, "value", str(entity.type))),
        "Provenance": str(getattr(entity.provenance, "value", str(entity.provenance))),
        "Confidence": float(round(entity.confidence, 3)),
    })

    # 12. Pset_RealityEngine (BUILD_SPEC B19 / RE-ASSET-000002)
    # Carries attributes with Basis and Confidence.
    # Unknown attributes export as absent with Basis = "unknown", never a default.
    pset_re = ifc_api.run("pset.add_pset", model, product=element, name="Pset_RealityEngine")
    props_re: dict[str, Any] = {
        "EntityId": str(entity.id),
        "EntityType": str(getattr(entity.type, "value", str(entity.type))),
        "Basis": str(getattr(entity.provenance, "value", str(entity.provenance))),
        "Confidence": float(round(entity.confidence, 3)),
    }
    attrs = (getattr(entity, "extra_fields", None) or {}).get("attributes")
    if not attrs and isinstance(getattr(entity, "custom_properties", None), dict):
        attrs = entity.custom_properties.get("attributes")

    if isinstance(attrs, dict):
        for attr_name, attr_data in attrs.items():
            safe_name = "".join(c for c in str(attr_name) if c.isalnum() or c in ("_", "-")) or "Attribute"
            if isinstance(attr_data, dict):
                val = attr_data.get("value")
                basis = str(attr_data.get("basis") or ("visual" if val is not None else "unknown"))
                conf = float(attr_data.get("p") if attr_data.get("p") is not None else attr_data.get("confidence", 1.0 if val is not None else 0.0))
                if val is not None and basis != "unknown":
                    props_re[safe_name] = str(val)
                props_re[f"{safe_name}_Basis"] = basis
                props_re[f"{safe_name}_Confidence"] = float(round(conf, 3))
            elif attr_data is not None:
                props_re[safe_name] = str(attr_data)
                props_re[f"{safe_name}_Basis"] = "user_stated"
                props_re[f"{safe_name}_Confidence"] = 1.0
            else:
                props_re[f"{safe_name}_Basis"] = "unknown"
                props_re[f"{safe_name}_Confidence"] = 0.0

    ifc_api.run("pset.edit_pset", model, pset=pset_re, properties=props_re)


def export_world_to_ifc(world, path: str | Path, schema: str = "IFC4") -> Dict[str, Any]:
    """Write `world`'s architectural entities to a real IFC4 or IFC2X3 file.

    Returns a report: {"skipped": {entity_id: reason}, "written":
    [entity_id, ...]}. Raises IFCBridgeError when nothing was
    exportable (an empty BIM file is not a success).
    """
    try:
        import ifcopenshell
        import ifcopenshell.api as ifc_api
        import ifcopenshell.util.shape_builder as shape_builder
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR_MESSAGE) from exc

    schema_ver = str(schema).upper()
    if schema_ver not in ("IFC4", "IFC2X3"):
        raise IFCBridgeError(f"unsupported IFC schema version {schema!r}; expected 'IFC4' or 'IFC2X3'")

    model = ifc_api.run("project.create_file", version=schema_ver)
    if schema_ver == "IFC2X3":
        # IFC2X3 schema requires explicit OwnerHistory for root entities
        p = ifc_api.run("owner.add_person", model, family_name="Engineer")
        o = ifc_api.run("owner.add_organisation", model, name="RealityEngine")
        ifc_api.run("owner.add_person_and_organisation", model, person=p, organisation=o)
        ifc_api.run("owner.add_application", model)

    project = ifc_api.run("root.create_entity", model, ifc_class="IfcProject",
                          name=f"Reality Engine {world.id}")
    ifc_api.run("unit.assign_unit", model)
    model_ctx = ifc_api.run("context.add_context", model,
                            context_type="Model")
    body_ctx = ifc_api.run(
        "context.add_context", model, context_type="Model",
        context_identifier="Body", parent=model_ctx,
        target_view="MODEL_VIEW")
    site = ifc_api.run("root.create_entity", model, ifc_class="IfcSite",
                       name="Reality Engine Site")
    building = ifc_api.run("root.create_entity", model,
                           ifc_class="IfcBuilding", name="Reality Engine Building")
    # Determinism: replace every random GlobalId the API minted with a
    # uuid5-derived one seeded by the world id + the element's role.
    project.GlobalId = _deterministic_guid(f"{world.id}/project")
    site.GlobalId = _deterministic_guid(f"{world.id}/site")
    building.GlobalId = _deterministic_guid(f"{world.id}/building")

    # Discover storey entities (EntityType.STOREY, EntityType.LEVEL, or is_storey)
    storey_entities = [
        e for e in world.entities.values()
        if e.type in (EntityType.STOREY, EntityType.LEVEL)
        or (isinstance(getattr(e, "custom_properties", None), dict) and e.custom_properties.get("is_storey"))
    ]
    created_storeys: Dict[str, Tuple[Any, float]] = {}
    if storey_entities:
        for idx, s_ent in enumerate(sorted(storey_entities, key=lambda s: (_get_storey_elevation(world, s), s.id))):
            s_name = s_ent.name or f"Storey {idx}"
            s_elem = ifc_api.run("root.create_entity", model,
                                 ifc_class="IfcBuildingStorey", name=s_name)
            s_elem.GlobalId = _deterministic_guid(f"{world.id}/storey/{s_ent.id}")
            elev = _get_storey_elevation(world, s_ent)
            s_elem.Elevation = float(elev)
            created_storeys[s_ent.id] = (s_elem, elev)
        default_storey = list(created_storeys.values())[0][0]
    else:
        storey = ifc_api.run("root.create_entity", model,
                             ifc_class="IfcBuildingStorey", name="L0")
        storey.GlobalId = _deterministic_guid(f"{world.id}/storey")
        default_storey = storey
        created_storeys["_default"] = (storey, 0.0)

    builder = shape_builder.ShapeBuilder(model)
    written: list = []
    skipped: Dict[str, str] = {}

    created_products: Dict[str, Any] = {}
    wall_entities: Dict[str, Entity] = {}
    entity_geoms: Dict[str, Geometry] = {}

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        ifc_class = _resolve_ifc_class(entity)
        if ifc_class in ("IfcBuilding", "IfcBuildingStorey"):
            skipped[entity_id] = (
                f"semantic container class {ifc_class} (spatial hierarchy container)"
            )
            continue
        geometry = _get_primary_geom(world, entity)
        if geometry is None:
            if not entity.geometry_ids:
                skipped[entity_id] = "no geometry attached"
            else:
                skipped[entity_id] = "geometry has no real bounds_min/bounds_max"
            continue
        extent = _bounds_extent(geometry)
        if extent is None:
            skipped[entity_id] = (
                "geometry has no real bounds_min/bounds_max")
            continue

        name = f"{entity.id} {entity.name}".strip()
        element = ifc_api.run("root.create_entity", model,
                              ifc_class=ifc_class, name=name)
        element.GlobalId = _deterministic_guid(f"{world.id}/entity/{entity.id}")

        target_storey = _resolve_target_storey(world, entity, created_storeys, default_storey)

        if ifc_class == "IfcSpace":
            ifc_api.run("aggregate.assign_object", model,
                        products=[element], relating_object=target_storey)
        elif ifc_class == "IfcOpeningElement":
            # Opening elements are hosted by IfcWall via IfcRelVoidsElement,
            # not directly contained in IfcBuildingStorey
            pass
        else:
            is_part = False
            for rel in getattr(entity, "relationships", ()):
                kind = getattr(rel, "kind", None)
                if (kind in (RelationshipKind.PART_OF, "part_of") or str(kind).endswith("PART_OF")) and rel.target_id in world.entities:
                    is_part = True
                    break
            if not is_part and getattr(entity, "parent_id", None) and entity.parent_id in world.entities:
                is_part = True

            if not is_part:
                ifc_api.run("spatial.assign_container", model,
                            products=[element], relating_structure=target_storey)

        dx, dy, dz = extent
        bmin = geometry.bounds_min

        # Non-convex polygon support (P6-01 / P8-03): If the entity or geometry defines
        # boundary_polygon_2d or floor_polygon_2d, extrude that exact 2D closed polygon
        # instead of a bounding box.
        polygon_2d = None
        if isinstance(getattr(entity, "custom_properties", None), dict):
            polygon_2d = (
                entity.custom_properties.get("floor_polygon_2d")
                or entity.custom_properties.get("boundary_polygon_2d")
                or entity.custom_properties.get("footprint")
            )
        if polygon_2d is None and isinstance(getattr(geometry, "custom_properties", None), dict):
            polygon_2d = (
                geometry.custom_properties.get("floor_polygon_2d")
                or geometry.custom_properties.get("boundary_polygon_2d")
                or geometry.custom_properties.get("footprint")
            )

        if polygon_2d and len(polygon_2d) >= 3:
            pts_2d = [(float(p[0]), float(p[1])) for p in polygon_2d]
            if (
                len(pts_2d) > 2
                and math.isclose(pts_2d[0][0], pts_2d[-1][0], abs_tol=1e-5)
                and math.isclose(pts_2d[0][1], pts_2d[-1][1], abs_tol=1e-5)
            ):
                pts_2d = pts_2d[:-1]
            curve = builder.polyline(pts_2d, closed=True)
            profile = builder.profile(curve)
            solid = builder.extrude(profile, magnitude=dz,
                                    position=(0.0, 0.0, bmin.z))
        else:
            profile = builder.rectangle(
                size=(dx, dy),
                position=(bmin.x, bmin.y),
            )
            solid = builder.extrude(profile, magnitude=dz,
                                    position=(0.0, 0.0, bmin.z))
        representation = builder.get_representation(context=body_ctx,
                                                    items=solid)
        ifc_api.run("geometry.assign_representation", model,
                    product=element, representation=representation)
        _enrich_element_property_sets(ifc_api, model, element, entity, extent)
        written.append(entity_id)
        created_products[entity_id] = element
        entity_geoms[entity_id] = geometry
        if entity.type == EntityType.WALL:
            wall_entities[entity_id] = entity

    # Connect openings and fillings (doors / windows) to their host walls
    voided_openings = set()
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if entity.type not in (EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING):
            continue
        if entity_id not in created_products:
            continue

        host_wall_id = _find_host_wall_id(world, entity, sorted(wall_entities.keys()))
        if not host_wall_id or host_wall_id not in created_products:
            continue

        host_wall_elem = created_products[host_wall_id]
        host_wall_geom = entity_geoms[host_wall_id]
        elem = created_products[entity_id]

        if entity.type == EntityType.OPENING:
            if entity_id not in voided_openings:
                ifc_api.run("feature.add_feature", model,
                            feature=elem, element=host_wall_elem)
                voided_openings.add(entity_id)
        else:
            op_key = f"{entity_id}-opening"
            if op_key not in voided_openings:
                opening_elem = ifc_api.run("root.create_entity", model,
                                           ifc_class="IfcOpeningElement",
                                           name=f"opening-{entity.id}")
                opening_elem.GlobalId = _deterministic_guid(f"{world.id}/opening/{entity.id}")

                e_geom = entity_geoms[entity_id]
                cut_solid = _make_opening_solid(builder, host_wall_geom, e_geom)
                rep = builder.get_representation(context=body_ctx, items=cut_solid)
                ifc_api.run("geometry.assign_representation", model,
                            product=opening_elem, representation=rep)

                ifc_api.run("feature.add_feature", model,
                            feature=opening_elem, element=host_wall_elem)
                ifc_api.run("feature.add_filling", model,
                            opening=opening_elem, element=elem)
                voided_openings.add(op_key)
                written.append(f"opening-{entity.id}")

    for wall_id, wall_ent in sorted(wall_entities.items()):
        ws = wall_ent.custom_properties.get("wall_slab")
        if not isinstance(ws, dict):
            continue
        slab_openings = ws.get("openings", [])
        if not isinstance(slab_openings, list):
            continue
        host_wall_elem = created_products.get(wall_id)
        host_wall_geom = entity_geoms.get(wall_id)
        if not host_wall_elem or not host_wall_geom:
            continue

        for idx, op in enumerate(slab_openings):
            if not isinstance(op, dict):
                continue
            op_ent_id = op.get("entity_id")
            if op_ent_id and (op_ent_id in voided_openings or f"{op_ent_id}-opening" in voided_openings):
                continue
            op_key = f"{wall_id}-slab-op-{idx}"
            if op_key in voided_openings:
                continue

            opening_elem = ifc_api.run("root.create_entity", model,
                                       ifc_class="IfcOpeningElement",
                                       name=f"opening-{wall_id}-{idx + 1:02d}")
            opening_elem.GlobalId = _deterministic_guid(f"{world.id}/opening/{wall_id}/{idx}")

            cut_solid = _make_slab_opening_solid(builder, ws, op, host_wall_geom)
            rep = builder.get_representation(context=body_ctx, items=cut_solid)
            ifc_api.run("geometry.assign_representation", model,
                        product=opening_elem, representation=rep)
            ifc_api.run("feature.add_feature", model,
                        feature=opening_elem, element=host_wall_elem)
            voided_openings.add(op_key)
            written.append(f"opening-{wall_id}-{idx + 1:02d}")

    # Aggregate parts into whole entities via IfcRelAggregates (BUILD_SPEC B19)
    for entity_id in written:
        entity = world.entities.get(entity_id)
        if not entity:
            continue
        parent_id = getattr(entity, "parent_id", None)
        if not parent_id:
            for rel in getattr(entity, "relationships", ()):
                kind = getattr(rel, "kind", None)
                if (kind in (RelationshipKind.PART_OF, "part_of") or str(kind).endswith("PART_OF")) and rel.target_id in created_products:
                    parent_id = rel.target_id
                    break
        if parent_id and parent_id in created_products and entity_id in created_products:
            parent_elem = created_products[parent_id]
            elem = created_products[entity_id]
            ifc_api.run("aggregate.assign_object", model, products=[elem], relating_object=parent_elem)
        elif parent_id and parent_id not in created_products and entity_id in created_products:
            # Fallback if parent wasn't exportable (e.g. lacked bounds)
            elem = created_products[entity_id]
            target_s = _resolve_target_storey(world, entity, created_storeys, default_storey)
            ifc_api.run("spatial.assign_container", model, products=[elem], relating_structure=target_s)

    if not written:
        raise IFCBridgeError(
            "no entities were exportable to IFC -- the world has no "
            "BOX/PLANE geometry with real bounds; refusing to write an "
            "empty BIM file")

    _reassign_relationship_guids(model, world.id)
    _normalize_unit_order(model)
    model.header.file_name.time_stamp = _FIXED_HEADER_TIMESTAMP

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))
    return {"skipped": skipped, "written": written}


def _extract_ifc_geometry(elem, geom_id: str) -> tuple[Optional[Geometry], dict[str, Any]]:
    rep = getattr(elem, "Representation", None)
    if not rep:
        return None, {}

    for r in getattr(rep, "Representations", ()):
        for item in getattr(r, "Items", ()):
            if item.is_a("IfcExtrudedAreaSolid"):
                depth = float(item.Depth)
                pos = getattr(item, "Position", None)
                pos_coord = pos.Location.Coordinates if pos and hasattr(pos, "Location") else (0.0, 0.0, 0.0)
                z_min = float(pos_coord[2])
                z_max = z_min + depth

                swept = getattr(item, "SweptArea", None)
                if not swept:
                    continue

                pts_2d: list[tuple[float, float]] = []
                if swept.is_a("IfcArbitraryClosedProfileDef"):
                    curve = getattr(swept, "OuterCurve", None)
                    if curve and hasattr(curve, "Points") and hasattr(curve.Points, "CoordList"):
                        pts_2d = [tuple(float(c) for c in pt) for pt in curve.Points.CoordList]
                    elif curve and hasattr(curve, "Points"):
                        pts_2d = [tuple(float(c) for c in pt.Coordinates) for pt in curve.Points]
                elif swept.is_a("IfcRectangleProfileDef"):
                    dx = float(swept.XDim)
                    dy = float(swept.YDim)
                    spos = swept.Position.Location.Coordinates if hasattr(swept, "Position") and hasattr(swept.Position, "Location") else (0.0, 0.0)
                    cx, cy = float(spos[0]), float(spos[1])
                    pts_2d = [
                        (cx - dx / 2.0, cy - dy / 2.0),
                        (cx + dx / 2.0, cy - dy / 2.0),
                        (cx + dx / 2.0, cy + dy / 2.0),
                        (cx - dx / 2.0, cy + dy / 2.0),
                    ]

                if pts_2d:
                    xs = [p[0] for p in pts_2d]
                    ys = [p[1] for p in pts_2d]
                    geom = Geometry(
                        id=geom_id,
                        type=GeometryType.BOX if len(pts_2d) == 4 else GeometryType.PLANE,
                        bounds_min=Vector3(min(xs), min(ys), z_min),
                        bounds_max=Vector3(max(xs), max(ys), z_max),
                    )
                    return geom, {"boundary_polygon_2d": pts_2d}

            elif item.is_a("IfcTriangulatedFaceSet"):
                coords = [tuple(float(c) for c in pt) for pt in item.Coordinates.CoordList]
                faces = [tuple(int(idx) - 1 for idx in face) for face in item.CoordIndex]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                zs = [c[2] for c in coords]
                geom = Geometry(
                    id=geom_id,
                    type=GeometryType.MESH,
                    vertex_count=len(coords),
                    triangle_count=len(faces),
                    bounds_min=Vector3(min(xs), min(ys), min(zs)),
                    bounds_max=Vector3(max(xs), max(ys), max(zs)),
                )
                return geom, {"vertices": coords, "faces": faces}

    return None, {}


def import_ifc_to_world(path: str | Path) -> WorldIR:
    """Import an IFC4 or IFC2X3 file into a WorldIR model with zero geometric drift.

    Restores:
      - World identifier from IfcProject.
      - Spatial hierarchy (IfcBuildingStorey -> EntityType.STOREY).
      - Semantic architectural entities (IfcWall, IfcSlab, IfcColumn, IfcBeam,
        IfcDoor, IfcWindow, IfcOpeningElement, IfcStair, IfcSpace,
        IfcFurnishingElement, IfcBuildingElementProxy).
      - Geometry bounds and 2D polygonal footprints / 3D meshes without drift.
      - Relationships and containment structures.
      - Property sets (Pset_RealityEngine, RealityEngine_Metadata).
    """
    try:
        import ifcopenshell
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR_MESSAGE) from exc

    f = ifcopenshell.open(str(path))
    projects = f.by_type("IfcProject")
    world_id = "imported-ifc"
    if projects:
        proj_name = projects[0].Name or ""
        if proj_name.startswith("Reality Engine "):
            world_id = proj_name.removeprefix("Reality Engine ").strip()
        elif proj_name:
            world_id = proj_name

    world = WorldIR(id=world_id)

    # 1. Discover IfcBuildingStorey entities
    storey_map: dict[int, str] = {}  # ifc entity id -> worldir entity id
    for s_elem in f.by_type("IfcBuildingStorey"):
        s_name = s_elem.Name or f"Storey_{s_elem.id()}"
        s_id = s_elem.GlobalId or f"storey-{s_elem.id()}"
        if " " in s_name:
            parts = s_name.split(" ", 1)
            if parts[0].startswith("storey-") or parts[0].startswith("level-"):
                s_id = parts[0]
                s_name = parts[1]

        elev = float(getattr(s_elem, "Elevation", 0.0) or 0.0)
        s_ent = Entity(
            id=s_id,
            name=s_name,
            type=EntityType.STOREY,
            custom_properties={"elevation_m": elev, "is_storey": True},
        )
        world.entities[s_id] = s_ent
        storey_map[s_elem.id()] = s_id

    # 2. Invert _ENTITY_IFC_CLASS mapping
    class_to_type = {
        "IfcWall": EntityType.WALL,
        "IfcColumn": EntityType.COLUMN,
        "IfcBeam": EntityType.BEAM,
        "IfcDoor": EntityType.DOOR,
        "IfcWindow": EntityType.WINDOW,
        "IfcOpeningElement": EntityType.OPENING,
        "IfcStair": EntityType.STAIRS,
        "IfcStairFlight": EntityType.STAIRS,
        "IfcSpace": EntityType.ROOM,
        "IfcRoof": EntityType.ROOF,
        "IfcFurnishingElement": EntityType.FURNITURE,
    }

    # 3. Process products
    for elem in f.by_type("IfcProduct"):
        if elem.is_a("IfcProject") or elem.is_a("IfcSite") or elem.is_a("IfcBuilding") or elem.is_a("IfcBuildingStorey"):
            continue

        raw_name = elem.Name or f"element-{elem.id()}"
        ent_id = f"ent-{elem.id()}"
        ent_name = raw_name

        if " " in raw_name:
            cand_id, cand_name = raw_name.split(" ", 1)
            if cand_id and not cand_id.startswith("Reality"):
                ent_id = cand_id
                ent_name = cand_name

        custom_props: dict[str, Any] = {}
        provenance = Provenance.OBSERVED
        confidence = 1.0
        override_type = None

        for rel in getattr(elem, "IsDefinedBy", ()):
            pset = getattr(rel, "RelatingPropertyDefinition", None)
            if not pset or not pset.is_a("IfcPropertySet"):
                continue
            pset_name = pset.Name or ""
            props = {
                p.Name: getattr(p.NominalValue, "wrappedValue", None)
                for p in getattr(pset, "HasProperties", ())
            }

            if pset_name in ("RealityEngine_Metadata", "Pset_RealityEngine"):
                if "EntityId" in props and props["EntityId"]:
                    ent_id = str(props["EntityId"])
                if "EntityType" in props and props["EntityType"]:
                    try:
                        override_type = EntityType(props["EntityType"])
                    except ValueError:
                        pass
                if "Provenance" in props and props["Provenance"]:
                    try:
                        provenance = Provenance(props["Provenance"])
                    except ValueError:
                        pass
                elif "Basis" in props and props["Basis"]:
                    try:
                        provenance = Provenance(props["Basis"])
                    except ValueError:
                        pass
                if "Confidence" in props and props["Confidence"] is not None:
                    try:
                        confidence = float(props["Confidence"])
                    except (ValueError, TypeError):
                        pass
            else:
                custom_props[pset_name] = props

        if override_type:
            ent_type = override_type
        elif elem.is_a("IfcSlab"):
            ptype = str(getattr(elem, "PredefinedType", "")).upper()
            if "ROOF" in ptype or "ROOF" in ent_name.upper():
                ent_type = EntityType.ROOF
            elif "CEILING" in ptype or "CEILING" in ent_name.upper():
                ent_type = EntityType.CEILING
            else:
                ent_type = EntityType.FLOOR
        elif elem.is_a() in class_to_type:
            ent_type = class_to_type[elem.is_a()]
        else:
            ent_type = EntityType.STRUCTURE

        geom, extra_props = _extract_ifc_geometry(elem, f"geom-{ent_id}")
        geometry_ids = []
        if geom is not None:
            world.geometries[geom.id] = geom
            geometry_ids.append(geom.id)
            custom_props.update(extra_props)

        relationships = []
        for rel in getattr(elem, "ContainedInStructure", ()):
            struct = getattr(rel, "RelatingStructure", None)
            if struct and struct.id() in storey_map:
                relationships.append(Relationship(
                    target_id=storey_map[struct.id()],
                    kind=RelationshipKind.INSIDE,
                ))

        for rel in getattr(elem, "Decomposes", ()):
            parent = getattr(rel, "RelatingObject", None)
            if parent and parent.id() in storey_map:
                relationships.append(Relationship(
                    target_id=storey_map[parent.id()],
                    kind=RelationshipKind.INSIDE,
                ))

        entity = Entity(
            id=ent_id,
            name=ent_name,
            type=ent_type,
            geometry_ids=geometry_ids,
            relationships=relationships,
            provenance=provenance,
            confidence=confidence,
            custom_properties=custom_props,
        )
        world.entities[ent_id] = entity

    return world
