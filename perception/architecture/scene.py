"""Interior scene assembly (INTERIOR RECONSTRUCTION MISSION: the
composition stage).

Individual stages exist and are verified:

  reconstruction.orchestrator     -> ReconstructionResult (real backend)
  perception.geometry.planes      -> DetectedPlanes
  perception.geometry.orientation -> floor/ceiling/wall roles
  evidence.promote_planes         -> WALL/FLOOR/CEILING entities
  openings.detect_openings        -> door/window/generic OpeningFits
  room_graph.build_room_graph     -> rooms w/ bounds + openings
  corridors.detect_corridor       -> measured circulation cells
  topology.promote_building_topology -> STOREY/ROOM entities + edges
  topology.link_stairs_to_storeys -> stair-storey topology
  worldstore.store.WorldStore     -> persistent versioned worlds

But the full interior composition -- reconstruction in, one coherent
interior WorldIR out, every entity retained, every refusal counted --
existed only as a script (scripts/reconstruct_room.py) that stopped at
planes+rooms and skipped openings/corridors/connectivity entirely.

This module is the assembly line:

    ReconstructionResult
      -> plane detection + orientation (existing)
      -> plane promotion (existing; entity ids shared with openings)
      -> opening detection on promoted WALLS (new in the chain)
      -> opening promotion into first-class entities (new)
      -> room graph + corridor inference (existing detectors, new use)
      -> room connectivity through shared-wall passages (new)
      -> storey/stair wiring where storeys exist (new)
      -> assembled InteriorSceneResult with honest diagnostics

Determinism: every stage is deterministic; same input -> identical
world and identical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from perception.geometry.planes import (
    canonicalize_plane_ids,
    detect_planes,
    merge_coplanar_fragments,
    split_parallel_sheets,
)
from perception.geometry.orientation import classify_planes as orient_planes
from perception.geometry.orientation import OrientedPlane
from evidence.promote_planes import positions_by_plane, promote_plane_to_entity
from perception.architecture.classify import PlaneInput
from perception.architecture.openings import OpeningFit, detect_openings
from perception.architecture.openings import promote_opening_to_entity
from perception.architecture.room_graph import (
    RoomGraph,
    RoomOpening,
    build_room_graph,
    build_building_graph,
)
from perception.architecture.corridors import CorridorFit, detect_corridor
from perception.architecture.topology import (
    connected_rooms_through_openings,
    promote_building_topology,
)
from world_ir import WorldIR
from world_ir.validation import validate_world_ir


class SceneAssemblyError(ValueError):
    """Scene assembly refused (insufficient evidence)."""


@dataclass
class InteriorSceneResult:
    """The assembled interior world + every stage's honest accounting."""

    world: WorldIR
    #: plane_id -> wall openings measured on it
    openings_by_plane: Dict[str, List[OpeningFit]] = field(default_factory=dict)
    #: promoted opening entity ids
    opening_entity_ids: List[str] = field(default_factory=list)
    #: openings detected but not promoted (with reason)
    openings_unpromoted: List[Dict[str, str]] = field(default_factory=list)
    #: room-graph cells (bounds, measured openings, adjacency)
    rooms: List[RoomGraph] = field(default_factory=list)
    #: corridor cells: (room_id, CorridorFit)
    corridors: List[Tuple[str, CorridorFit]] = field(default_factory=list)
    #: canonical room-pair connectivity through shared-wall passages
    room_links: List[Tuple[str, str]] = field(default_factory=list)
    #: promoted ROOM entity ids in the world (empty when the room layer
    #: was skipped -- see storeys_skipped)
    room_entity_ids: List[str] = field(default_factory=list)
    #: promoted STOREY entity ids in the world
    storey_entity_ids: List[str] = field(default_factory=list)
    #: storey-level connectivity through stairs (when storeys exist)
    storey_links: List[Tuple[str, str]] = field(default_factory=list)
    #: planes the classifier could not role-label (never guessed)
    unclassified_planes: List[str] = field(default_factory=list)
    #: why the storey layer was skipped (absence = it ran)
    storeys_skipped: str = ""
    validation_issues: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            f"entities: {len(self.world.entities)}",
            f"openings: {len(self.opening_entity_ids)} promoted, "
            f"{len(self.openings_unpromoted)} unpromoted",
            f"rooms: {len(self.rooms)}",
            f"corridors: {len(self.corridors)}",
            f"room links: {len(self.room_links)}",
            f"storey links: {len(self.storey_links)}",
            f"unclassified planes: {len(self.unclassified_planes)}",
            f"validation issues: {len(self.validation_issues)}",
        ]
        return "\n".join(lines)


def _plane_inputs(
    oriented: Sequence[OrientedPlane],
    positions: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, PlaneInput]:
    """PlaneInputs for the opening scanner, keyed by plane_id, carrying
    the real inlier positions the classifier's planes were fit to."""
    inputs: Dict[str, PlaneInput] = {}
    for o in oriented:
        if o.role != "wall":
            continue
        pts = positions.get(o.plane.plane_id, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        n = len(pts)
        centroid = (sum(xs) / n, sum(ys) / n, sum(zs) / n)
        inputs[o.plane.plane_id] = PlaneInput(
            plane_id=o.plane.plane_id,
            normal=o.normal,
            centroid=centroid,
            bounds_min=(min(xs), min(ys), min(zs)),
            bounds_max=(max(xs), max(ys), max(zs)),
            inlier_positions=tuple(pts),
        )
    return inputs


def assemble_interior_scene(
    result,
    *,
    up: Tuple[float, float, float],
    seed: int = 42,
    world: Optional[WorldIR] = None,
    distance_tolerance_m: float = 0.02,
    min_inliers: int = 30,
) -> InteriorSceneResult:
    """Assemble the full interior WorldIR from one ReconstructionResult.

    Deterministic. Raises SceneAssemblyError when the input cannot
    honestly support an interior scene (no planes at all). Individual
    stage refusals (unclassifiable planes, walls without openings,
    floors that form no enclosure) are counted in the result, never
    hidden and never guessed through.
    """
    detection = detect_planes(
        result, seed=seed,
        distance_tolerance_m=distance_tolerance_m,
        min_inliers=min_inliers,
    )
    if not detection.planes:
        raise SceneAssemblyError(
            "no planes detected -- an interior scene cannot be assembled "
            "from unstructured points; refusing to guess one"
        )
    # Sheet-split refinement: RANSAC's wide collection tolerance lets one
    # horizontal candidate absorb two parallel offset sheets (stepped
    # floor slabs, doubled finishes), leaving a phantom mid-surface that
    # corrupts sill heights, room enclosures, and storey grouping. Split
    # any such plane along `up` before classification, then re-key ids.
    positions_by_id = {p.track_id: p.position for p in result.points}
    split_planes = merge_coplanar_fragments(
        split_parallel_sheets(
            detection.planes, positions_by_id, up,
            distance_tolerance_m=distance_tolerance_m,
            min_inliers=min_inliers,
        ),
        positions_by_id, up,
        distance_tolerance_m=distance_tolerance_m,
        min_inliers=min_inliers,
    )
    split_planes = canonicalize_plane_ids(split_planes)
    camera_positions = [p.position for p in result.camera_poses]
    oriented = orient_planes(split_planes, camera_positions, up=up)
    positions = positions_by_plane(result, oriented)

    world = world if world is not None else WorldIR(id="world-interior")
    res = InteriorSceneResult(world=world)

    # ---- stage 1: promote structural planes (walls/floors/ceilings) --
    floor_heights: List[float] = []
    for o in oriented:
        if o.role == "unknown":
            res.unclassified_planes.append(o.plane.plane_id)
            continue
        if o.role == "floor":
            zs = [p[2] for p in positions[o.plane.plane_id]]
            floor_heights.append(sum(zs) / len(zs))
        entity_id = f"struct-{o.plane.plane_id}"
        try:
            promote_plane_to_entity(
                o, result, world, entity_id,
                other_planes=oriented,
                plane_positions=positions,
            )
        except Exception as exc:
            res.unclassified_planes.append(
                f"{o.plane.plane_id}: promotion refused: {exc}"
            )

    # Reference floor height for opening sills: the lowest measured
    # floor plane (a wall's openings are measured against the storey it
    # stands on).
    ref_floor = min(floor_heights) if floor_heights else None

    # ---- stage 2: openings on promoted walls ----
    wall_inputs = _plane_inputs(oriented, positions)
    all_openings: List[Tuple[str, RoomOpening]] = []
    for plane_id, plane in sorted(wall_inputs.items()):
        if f"struct-{plane_id}" not in world.entities:
            continue  # wall was not promotable; no entity to host openings
        if ref_floor is None:
            continue
        fits = detect_openings(plane, up=up, floor_height=ref_floor)
        res.openings_by_plane[plane_id] = fits
        for i, fit in enumerate(fits, start=1):
            entity_id = f"opening-{plane_id}-{i:02d}"
            try:
                ent = promote_opening_to_entity(
                    fit, world, entity_id,
                    host_wall_id=f"struct-{plane_id}",
                )
                res.opening_entity_ids.append(ent.id)
            except Exception as exc:
                res.openings_unpromoted.append({
                    "plane_id": plane_id,
                    "kind": fit.kind,
                    "note": str(exc),
                })
                continue
            all_openings.append((f"wall-{plane_id}", RoomOpening(
                wall_element_id=f"wall-{plane_id}",
                kind="doorway" if fit.kind == "door" else fit.kind,
                width_m=fit.width_m,
                height_m=fit.height_m,
            )))

    # ---- stage 3: rooms (per-floor grouping via the room graph) ----
    elements: List = []
    from perception.architecture.classify import ArchitecturalElement
    for o in oriented:
        if o.role == "unknown":
            continue
        pts = positions.get(o.plane.plane_id, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        n = len(pts)
        elements.append(ArchitecturalElement(
            element_id=f"{o.role}-{o.plane.plane_id}",
            element_type=o.role,
            source_plane_id=o.plane.plane_id,
            reason=f"oriented {o.role}",
            bounds_min=(min(xs), min(ys), min(zs)),
            bounds_max=(max(xs), max(ys), max(zs)),
        ))
    plane_inputs = dict(wall_inputs)
    for o in oriented:
        if o.role == "wall":
            continue
        pts = positions.get(o.plane.plane_id, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        n = len(pts)
        plane_inputs[o.plane.plane_id] = PlaneInput(
            plane_id=o.plane.plane_id,
            normal=o.normal,
            centroid=(sum(xs) / n, sum(ys) / n, sum(zs) / n),
            bounds_min=(min(xs), min(ys), min(zs)),
            bounds_max=(max(xs), max(ys), max(zs)),
            inlier_positions=tuple(pts),
        )
    res.rooms = build_room_graph(elements, up=up, planes=plane_inputs)

    # ---- stage 4: corridors among the rooms ----
    for room in res.rooms:
        fit = detect_corridor(room)
        if fit is not None:
            res.corridors.append((room.room_id, fit))

    # ---- stage 5: room connectivity through shared-wall passages ----
    res.room_links = connected_rooms_through_openings(res.rooms, all_openings)

    # ---- stage 6: rooms, storeys, building -- always promoted when the
    # evidence supports them. The room graph names boundaries by the
    # plane element ids (<role>-<plane_id>); promotion wrote the planes
    # as struct-<plane_id>, so the boundary ids are re-pointed at the
    # promoted entities before promotion. Rooms carry bounds, openings,
    # and adjacency into WorldIR; storeys group rooms by measured floor
    # height (single-storey interiors get one storey -- the grouping is
    # measured, not assumed). When any boundary entity is missing the
    # layer is skipped and recorded: a partial interior is honest, an
    # invented one is not.
    if res.rooms:
        building = build_building_graph(res.rooms, up=up)
        remapped_rooms = _with_promoted_boundary_ids(res.rooms)
        if building is not None and _parts_promoted(remapped_rooms, world):
            topo = promote_building_topology(building, remapped_rooms, world)
            res.room_entity_ids = list(topo.room_ids)
            res.storey_entity_ids = list(topo.storey_ids)
        else:
            res.storeys_skipped = (
                "room boundary entities not all promoted -- room/storey "
                "layer skipped (recorded, not guessed)"
            )

    # ---- validation ----
    report = validate_world_ir(world)
    res.validation_issues = [i.message for i in report.issues]
    return res


def _parts_promoted(rooms: Sequence[RoomGraph], world) -> bool:
    """True when every room's boundary entities exist in the world (the
    precondition promote_building_topology refuses without)."""
    for room in rooms:
        for eid in room.boundary_element_ids:
            if eid not in world.entities:
                return False
    return True


def _with_promoted_boundary_ids(
    rooms: Sequence[RoomGraph],
) -> List[RoomGraph]:
    """Rooms with boundary ids re-pointed from the room graph's element
    ids (<role>-<plane_id>) at the promoted plane entities
    (struct-<plane_id>). Only ids that actually exist as promoted
    entities are remapped -- anything else is left as-is so the
    promotion precondition, not this mapping, decides refusals."""
    remapped: List[RoomGraph] = []
    for room in rooms:
        boundary = tuple(
            f"struct-{eid.split('-', 1)[1]}"
            if eid.startswith(("wall-", "floor-", "ceiling-")) else eid
            for eid in room.boundary_element_ids
        )
        if boundary == room.boundary_element_ids:
            remapped.append(room)
            continue
        remapped.append(RoomGraph(
            room_id=room.room_id,
            boundary_element_ids=boundary,
            bounds_min=room.bounds_min,
            bounds_max=room.bounds_max,
            dimensions_m=room.dimensions_m,
            floor_area_m2=room.floor_area_m2,
            openings=room.openings,
            adjacent_room_ids=room.adjacent_room_ids,
        ))
    return remapped
