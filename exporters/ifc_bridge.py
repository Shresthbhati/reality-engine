"""WorldIR -> IFC bridge (P8-03: "WorldIR -> IFC via IfcOpenShell
(adapter, license recorded); do not rebuild BIM from scratch").

Uses the REAL ifcopenshell 0.8.x API: a proper IFC4 project (IfcProject
-> IfcSite -> IfcBuilding -> IfcBuildingStorey spatial hierarchy) and
one semantic IFC class per WorldIR architectural entity -- IfcWall,
IfcSlab (floor/ceiling), IfcColumn, IfcBeam, IfcDoor, IfcWindow,
IfcRoof, IfcBuildingElementProxy for anything else with bounds. Each
element gets real IfcExtrudedAreaSolid geometry from its measured
bounds_min/bounds_max (the same AABB every other exporter consumes)
and a Name carrying "<worldir_id> <name>" so the reverse path (IFC ->
WorldIR, later) can trace elements back to their evidence.

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

from pathlib import Path
from typing import Dict, Optional, Tuple

from world_ir.schema_v1 import EntityType, GeometryType

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
    EntityType.STAIRS.value: "IfcStair",
    EntityType.BUILDING.value: "IfcBuilding",
    EntityType.ROOM.value: "IfcSpace",
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
    spatial.assign_container) a deterministic GlobalId. These are
    created implicitly by the API with random GUIDs; their seeds use
    the (already deterministic) GlobalIds of the objects they relate,
    so the assignment order and values are stable."""
    rels = list(model.by_type("IfcRelAggregates")) + list(
        model.by_type("IfcRelContainedInSpatialStructure"))

    def _key(rel):
        # IfcRelAggregates uses RelatingObject/RelatedObjects;
        # IfcRelContainedInSpatialStructure uses RelatingStructure/
        # RelatedElements -- same relationship concept, different
        # attribute names in the IFC4 schema.
        relating = getattr(rel, "RelatingObject", None) or getattr(
            rel, "RelatingStructure", None)
        related = getattr(rel, "RelatedObjects", None) or getattr(
            rel, "RelatedElements", None) or ()
        relating_id = relating.GlobalId if relating else ""
        related_ids = tuple(sorted(o.GlobalId for o in related))
        return (rel.is_a(), relating_id, related_ids)

    for rel in sorted(rels, key=_key):
        cls, relating, related = _key(rel)
        rel.GlobalId = _deterministic_guid(
            f"{world_id}/rel/{cls}/{relating}/{'+'.join(related)}")
        # The API accumulates RelatedElements/RelatedObjects in a hash
        # set of swig proxies (id-address order -- varies between
        # runs); rewrite the list sorted by the (now deterministic)
        # GlobalIds so serialization is stable.
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


def export_world_to_ifc(world, path: str | Path) -> Dict[str, Dict[str, str]]:
    """Write `world`'s architectural entities to a real IFC4 file.

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

    model = ifc_api.run("project.create_file", version="IFC4")
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
    storey = ifc_api.run("root.create_entity", model,
                         ifc_class="IfcBuildingStorey", name="L0")
    # Determinism: replace every random GlobalId the API minted with a
    # uuid5-derived one seeded by the world id + the element's role.
    project.GlobalId = _deterministic_guid(f"{world.id}/project")
    site.GlobalId = _deterministic_guid(f"{world.id}/site")
    building.GlobalId = _deterministic_guid(f"{world.id}/building")
    storey.GlobalId = _deterministic_guid(f"{world.id}/storey")

    builder = shape_builder.ShapeBuilder(model)
    written: list = []
    skipped: Dict[str, str] = {}

    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        ifc_class = _ENTITY_IFC_CLASS.get(
            entity.type, "IfcBuildingElementProxy")
        # The semantic container classes (IfcBuilding/IfcSpace) need
        # their own spatial aggregation; elements with geometry are
        # this bridge's scope.
        if ifc_class in {"IfcBuilding", "IfcSpace"}:
            skipped[entity_id] = (
                f"semantic container class {ifc_class} (spatial "
                "aggregation not built in this bridge pass)")
            continue
        geometry = None
        for gid in entity.geometry_ids:
            g = world.geometries.get(gid)
            if g is not None and g.type in {GeometryType.BOX,
                                            GeometryType.PLANE}:
                geometry = g
                break
        if geometry is None:
            skipped[entity_id] = "no BOX/PLANE geometry attached"
            continue
        extent = _bounds_extent(geometry)
        if extent is None:
            skipped[entity_id] = (
                "geometry has no real bounds_min/bounds_max")
            continue

        name = f"{entity.id} {entity.name}".strip()
        element = ifc_api.run("root.create_entity", model,
                              ifc_class=ifc_class, name=name)
        # Deterministic identity: the GUID is a pure function of the
        # WorldIR entity id, so re-exporting the same world re-mints
        # the same IFC GlobalIds (traceable across exports).
        element.GlobalId = _deterministic_guid(f"{world.id}/entity/{entity.id}")
        ifc_api.run("spatial.assign_container", model,
                    products=[element], relating_structure=storey)

        dx, dy, dz = extent
        # The solid sits at the bounds' minimum corner: profile drawn
        # centered on the footprint, extruded upward from z_min.
        bmin = geometry.bounds_min
        profile = builder.rectangle(
            size=(dx, dy),
            position=(bmin.x + dx / 2.0, bmin.y + dy / 2.0),
        )
        solid = builder.extrude(profile, magnitude=dz,
                                position=(0.0, 0.0, bmin.z))
        representation = builder.get_representation(context=body_ctx,
                                                    items=solid)
        ifc_api.run("geometry.assign_representation", model,
                    product=element, representation=representation)
        written.append(entity_id)

    if not written:
        raise IFCBridgeError(
            "no entities were exportable to IFC -- the world has no "
            "BOX/PLANE geometry with real bounds; refusing to write an "
            "empty BIM file")

    # Relationships were minted with random GUIDs by the API calls
    # above; reseed them deterministically, normalize the unit list
    # order (assign_unit uses a hash set), and pin the header
    # time_stamp, before serializing.
    _reassign_relationship_guids(model, world.id)
    _normalize_unit_order(model)
    model.header.file_name.time_stamp = _FIXED_HEADER_TIMESTAMP

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))
    return {"skipped": skipped, "written": written}
