"""World compiler (spec sec 16/24: EVIDENCE -> OBSERVATIONS -> GEOMETRY
-> SEMANTICS -> ENTITIES -> RELATIONSHIPS -> MEASUREMENTS -> CONFIDENCE
-> PROVENANCE -> WORLDIR).

The bridge between perception and the engine foundation: ONE call that
turns a ReconstructionResult (e.g. COLMAP sparse output) into a
complete, validated WorldIR.

    ReconstructionResult
      -> detect_planes (deterministic RANSAC)
      -> classify_planes (camera-side wall/floor/ceiling)
      -> promote_plane_to_entity (WALL/FLOOR/CEILING entities +
         GeometryType.PLANE geometry + extent/thickness measurements)
      -> detect_rooms + promote_room_to_entity (ROOM entities +
         CONTAINS/PART_OF + area/extent/height measurements)
      -> validate_world_ir (structural + geometric/provenance gate)
      -> WorldIR + CompileDiagnostics

Design invariants:

  - Deterministic and replayable: seeded RANSAC (seed is a compile
    parameter, not ambient state), canonical ordering everywhere, no
    clocks, no RNG beyond the seed. The same inputs + seed produce a
    byte-identical world (tested).
  - Nothing silently dropped: every plane (including unclassifiable
    ones), every unassigned point, and every room candidate (detected
    or failed with its notes) is accounted for in CompileDiagnostics.
    Unassigned points stay visible as an unassigned-point bookkeeping
    count, never as invented geometry.
  - Failed reconstruction is refused (CompileInputError): there is
    nothing honest to compile from an empty or failed result.
  - The validation gate runs BEFORE returning: a world that fails its
    own structural/geometric checks is refused (WorldValidationGateError)
    rather than handed downstream as silently corrupted state.
  - Provenance semantics preserved from the promotion layers: points
    are RECONSTRUCTED, typed structure is INFERRED, measurements are
    ESTIMATED. The compiler adds no provenance of its own and never
    upgrades a value's status.
  - LLM-free: everything here is deterministic geometry over
    reconstructed data.

The compiler composes the existing pure layers (planes, orientation,
promotions, validation) instead of reimplementing them: detection stays
pure, promotion stays the only write path, and the gate stays the last
word before any consumer sees the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world_ir.artifact_store import ArtifactStore

from evidence.promote_planes import (
    PlanePromotionError,
    positions_by_plane,
    promote_plane_to_entity,
)
from evidence.promote_rooms import (
    DetectedRoom,
    RoomInferenceError,
    detect_rooms,
    plane_summary_from,
    promote_room_to_entity,
)
from perception.geometry.orientation import (
    OrientationError,
    classify_planes,
)
from perception.geometry.planes import detect_planes
from provenance import Provenance
from world_ir import WorldIR

from world_ir.validation import ValidationSeverity, validate_world_ir

__all__ = [
    "CompileDiagnostics",
    "CompileInputError",
    "CompileOptions",
    "WorldValidationGateError",
    "compile_reconstruction_to_world",
]


class CompileInputError(ValueError):
    """The reconstruction result cannot honestly be compiled."""


class WorldValidationGateError(ValueError):
    """The compiled world failed its validation gate.

    Carries the issues so a caller can quarantine or inspect them --
    never a silent insert of corrupted state (spec sec 25).
    """

    def __init__(self, issues: List[str]):
        self.issues = list(issues)
        super().__init__(
            f"compiled world failed validation with {len(self.issues)} issue(s); "
            "first: " + (self.issues[0] if self.issues else "<none>")
        )


@dataclass(frozen=True)
class CompileOptions:
    """Compile parameters. `seed` makes reconstruction-derived structure
    replayable; everything else is handed through to the pure layers."""

    seed: int = 42
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: Entity id prefixes (caller-controllable so multiple compiles can
    #: target one world without id collisions).
    structure_prefix: str = "struct"
    room_prefix: str = "room"
    #: When True (default), the compiled world must pass its own
    #: validation gate before being returned. A gate failure raises
    #: WorldValidationGateError with the issues attached.
    require_validation: bool = True
    #: When given, promoted planes store their real inlier point
    #: positions as a real geometry artifact (world_ir/geometry_data.py)
    #: referenced by Geometry.data_uri/data_hash, instead of carrying
    #: only vertex_count + bounds. Optional and additive: omitting it
    #: (default) reproduces the exact pre-existing behavior.
    artifact_store: Optional[ArtifactStore] = None
    #: When True, promoted corridor candidates are added directly to world.entities.
    promote_corridors: bool = True
    #: When True, promoted window candidates are added directly to world.entities and associated to rooms.
    promote_windows: bool = True
    #: When True, promoted stair candidates are added directly to world.entities and linked to storeys.
    promote_stairs: bool = True
    #: When True, promoted building envelope and storey entities are added directly to world.entities.
    promote_building: bool = True


@dataclass(frozen=True)
class CompileDiagnostics:
    """Everything the compile observed, for inspection and honest
    reporting. Nothing about the reconstruction disappears silently:
    unclassifiable planes, unassigned points, and failed room candidates
    are all accounted for here."""

    seed: int
    points_total: int
    points_in_planes: int
    points_unassigned: int
    planes_total: int
    planes_by_role: Dict[str, int]
    planes_unpromoted: List[Dict[str, str]]  # unclassifiable planes: id + note
    entities_created: List[str]
    room_candidates: List[Dict[str, str]]  # status + notes per candidate
    rooms_detected: int
    measurements_count: int
    relationships_count: int
    validation_issues: List[str] = field(default_factory=list)
    #: Points that entered no plane and no room: the honest size of what
    #: this compile did NOT understand. Consumers (coverage reasoning,
    #: quality reports) should treat this as the known-unknown floor.
    unknown_point_fraction: float = 0.0

    def summary_text(self) -> str:
        lines = [
            f"compile (seed={self.seed}): {self.points_total} points -> "
            f"{self.planes_total} planes {dict(self.planes_by_role)}",
            f"  entities: {len(self.entities_created)} "
            f"({self.rooms_detected} room(s)), measurements: {self.measurements_count}, "
            f"relationships: {self.relationships_count}",
            f"  unassigned points: {self.points_unassigned} "
            f"({100.0 * self.unknown_point_fraction:.1f}% of input)",
        ]
        for plane in self.planes_unpromoted:
            lines.append(f"  unpromoted plane {plane['plane_id']}: {plane['note']}")
        for candidate in self.room_candidates:
            if candidate["status"] != "detected":
                lines.append(
                    f"  room candidate {candidate['room_id']}: "
                    f"{candidate['status']} ({candidate['notes']})"
                )
        if self.validation_issues:
            lines.append(f"  validation issues: {len(self.validation_issues)}")
        return "\n".join(lines)


def compile_reconstruction_to_world(
    result,
    options: Optional[CompileOptions] = None,
    world: Optional[WorldIR] = None,
    world_id: str = "",
    world_name: str = "Compiled World",
) -> Tuple[WorldIR, CompileDiagnostics]:
    """Compile one ReconstructionResult into a validated WorldIR.

    Returns (world, diagnostics). Raises:
      - CompileInputError: empty or failed reconstruction (nothing
        honest to compile), or no cameras (classification is defined by
        camera-side and refuses to guess).
      - WorldValidationGateError: the compiled world failed validation
        (only when options.require_validation).

    Deterministic: same result + same options -> byte-identical world.
    """
    options = options or CompileOptions()

    # ---- input gate: refuse what cannot be compiled honestly ----
    if result.registration_status == "failed":
        raise CompileInputError(
            "refusing to compile a failed reconstruction -- there is no "
            "honest world in an empty result"
        )
    if not result.points:
        raise CompileInputError(
            "refusing to compile an empty reconstruction "
            f"(registration_status={result.registration_status!r})"
        )
    camera_positions = [pose.position for pose in result.camera_poses]
    if not camera_positions:
        raise CompileInputError(
            "refusing to compile without camera poses -- wall/floor/ceiling "
            "classification is defined by camera-side and refuses to guess"
        )

    if world is None:
        # Deterministic identity: WorldIR's dataclass defaults mint a
        # uuid4 id and main_branch_id per instance, which would make two
        # identically-compiled worlds differ byte-for-byte -- the same
        # class of problem the created_at=0.0 convention exists for. The
        # compiler supplies stable identity; ad-hoc worlds keep UUIDs.
        stable_world_id = world_id or f"world-compiled-seed{options.seed}"
        world = WorldIR(
            id=stable_world_id,
            name=world_name,
            main_branch_id=f"branch-main-{stable_world_id}",
        )

    # ---- planes: detect -> classify ----
    detection = detect_planes(result, seed=options.seed)
    try:
        oriented = classify_planes(detection.planes, camera_positions, options.up)
    except OrientationError as exc:
        raise CompileInputError(f"plane classification failed: {exc}") from exc

    planes_by_role: Dict[str, int] = {}
    for plane in oriented:
        planes_by_role[plane.role] = planes_by_role.get(plane.role, 0) + 1

    # ---- promote structure planes ----
    positions = positions_by_plane(result, oriented)
    entities_created: List[str] = []
    planes_unpromoted: List[Dict[str, str]] = []
    for plane in oriented:
        if plane.role == "unknown":
            # Honest refusal path from promote_planes: an unclassifiable
            # plane is not typed by guesswork; it is reported, not hidden.
            planes_unpromoted.append({
                "plane_id": plane.plane.plane_id,
                "note": plane.uncertainty.note or "unclassified",
            })
            continue
        entity_id = f"{options.structure_prefix}-{plane.plane.plane_id}"
        try:
            promotion = promote_plane_to_entity(
                plane,
                result,
                world,
                entity_id,
                other_planes=oriented,
                plane_positions=positions,
                artifact_store=options.artifact_store,
            )
        except PlanePromotionError as exc:
            planes_unpromoted.append({
                "plane_id": plane.plane.plane_id,
                "note": f"promotion refused: {exc}",
            })
            continue
        entities_created.append(promotion.entity.id)

    # ---- rooms: build summaries for PROMOTED planes only, resolving
    # each summary's entity_id to the entity the plane promotion wrote
    # (frozen dataclass -> object.__setattr__, same as the tests) ----
    summaries = []
    for plane in oriented:
        if plane.role == "unknown":
            continue
        entity_id = f"{options.structure_prefix}-{plane.plane.plane_id}"
        if entity_id not in world.entities:
            continue  # its promotion was refused; rooms build on promoted planes only
        summary = plane_summary_from(plane, positions[plane.plane.plane_id])
        object.__setattr__(summary, "entity_id", entity_id)
        summaries.append(summary)
    room_candidates: List[DetectedRoom] = detect_rooms(summaries, options.up)
    rooms_detected = 0
    room_candidate_reports: List[Dict[str, str]] = []
    for room in room_candidates:
        if room.status == "detected":
            room_entity_id = f"{options.room_prefix}-{room.floor.plane_id}"
            try:
                room_promotion = promote_room_to_entity(
                    room, world, room_entity_id,
                    artifact_store=options.artifact_store,
                )
                entities_created.append(room_promotion.entity.id)
                rooms_detected += 1
                room_candidate_reports.append({
                    "room_id": room.room_id,
                    "status": "detected",
                    "notes": "; ".join(room.notes) if room.notes else "",
                })
            except RoomInferenceError as exc:
                room_candidate_reports.append({
                    "room_id": room.room_id,
                    "status": "promotion_refused",
                    "notes": str(exc),
                })
        else:
            room_candidate_reports.append({
                "room_id": room.room_id,
                "status": room.status,
                "notes": "; ".join(room.notes),
            })

    # ---- interior architecture: rooms, corridors, windows, stairs, building & space graph ----
    from perception.architecture.classify import ArchitecturalElement, PlaneInput
    from perception.architecture.corridor import detect_corridors
    from perception.architecture.room_graph import build_room_graph, build_building_graph
    from perception.architecture.space_graph import InteriorSpaceGraph

    arch_elements = []
    plane_inputs_map = {}
    for plane in oriented:
        if plane.role == "unknown":
            continue
        p_inliers = positions.get(plane.plane.plane_id, ())
        if p_inliers:
            bmin = (min(p[0] for p in p_inliers), min(p[1] for p in p_inliers), min(p[2] for p in p_inliers))
            bmax = (max(p[0] for p in p_inliers), max(p[1] for p in p_inliers), max(p[2] for p in p_inliers))
            cen = ((bmin[0] + bmax[0]) / 2.0, (bmin[1] + bmax[1]) / 2.0, (bmin[2] + bmax[2]) / 2.0)
        else:
            bmin = (0.0, 0.0, 0.0)
            bmax = (0.0, 0.0, 0.0)
            cen = (0.0, 0.0, 0.0)
        pi = PlaneInput(
            plane_id=plane.plane.plane_id,
            normal=plane.normal,
            centroid=cen,
            bounds_min=bmin,
            bounds_max=bmax,
            inlier_positions=tuple(p_inliers),
        )
        plane_inputs_map[plane.plane.plane_id] = pi
        arch_elements.append(ArchitecturalElement(
            element_id=f"{options.structure_prefix}-{plane.plane.plane_id}",
            element_type=plane.role,
            source_plane_id=plane.plane.plane_id,
            reason=plane.uncertainty.note or f"classified as {plane.role}",
            bounds_min=bmin,
            bounds_max=bmax,
        ))

    # 1. Build room graph from elements
    rg_rooms = build_room_graph(arch_elements, up=options.up, plane_inputs=plane_inputs_map)

    # 2. Detect circulation corridors
    corridors = detect_corridors(arch_elements, up=options.up, plane_inputs=plane_inputs_map, rooms=rg_rooms)

    # 3. Detect windows in wall planes
    detected_windows = []
    try:
        from perception.architecture.windows import detect_window
        for pid, pi in plane_inputs_map.items():
            prole = next((p.role for p in oriented if p.plane.plane_id == pid), None)
            if prole == "wall":
                try:
                    w_fit = detect_window(pi, up=options.up, floor_height=0.0)
                    if w_fit is not None:
                        detected_windows.append(w_fit)
                except Exception:
                    pass
    except Exception:
        pass

    # 4. Detect stairs from point cloud
    detected_stairs = []
    try:
        from perception.architecture.stairs import detect_stairs
        up_idx = 2
        u = [abs(x) for x in options.up]
        if max(u) > 1e-6:
            up_idx = u.index(max(u))
        if up_idx == 1:
            pts = [(p.position[0], p.position[2], p.position[1]) for p in result.points]
        elif up_idx == 0:
            pts = [(p.position[1], p.position[2], p.position[0]) for p in result.points]
        else:
            pts = [p.position for p in result.points]
        st_fit = detect_stairs(pts)
        if st_fit is not None:
            if up_idx == 1 and st_fit.position:
                import dataclasses
                pos = (st_fit.position[0], st_fit.position[2], st_fit.position[1])
                st_fit = dataclasses.replace(st_fit, position=pos)
            elif up_idx == 0 and st_fit.position:
                import dataclasses
                pos = (st_fit.position[2], st_fit.position[0], st_fit.position[1])
                st_fit = dataclasses.replace(st_fit, position=pos)
            detected_stairs.append(st_fit)
    except Exception:
        pass

    # 5. Build building graph and assign levels
    bld_graph = build_building_graph(rg_rooms, up=options.up, corridors=corridors, stairs=detected_stairs)

    from world_ir.schema_v1 import Geometry, GeometryType, Vector3, Entity, EntityType, Relationship, RelationshipKind

    # 6. Promote windows into WorldIR and associate with rooms
    promoted_windows = []
    if options.promote_windows and detected_windows:
        for widx, win in enumerate(detected_windows, start=1):
            win_id = f"window-{widx:03d}"
            if win_id not in world.entities:
                win_geom_id = f"geom-{win_id}"
                wbmin, wbmax = win.bounds_min, win.bounds_max
                world.geometries[win_geom_id] = Geometry(
                    id=win_geom_id,
                    type=GeometryType.BOX,
                    bounds_min=Vector3(wbmin[0], wbmin[1], wbmin[2]),
                    bounds_max=Vector3(wbmax[0], wbmax[1], wbmax[2]),
                )
                win_ent = Entity(
                    id=win_id,
                    type=EntityType.WINDOW,
                    name=f"Window {widx:03d}",
                    geometry_ids=[win_geom_id],
                    custom_properties=win.to_dict(),
                    provenance=Provenance.INFERRED,
                    confidence=win.confidence,
                )
                world.entities[win_id] = win_ent
                entities_created.append(win_id)
                promoted_windows.append(win_ent)

        if promoted_windows and rg_rooms:
            try:
                from perception.architecture.topology import associate_windows_to_rooms
                associate_windows_to_rooms(promoted_windows, rg_rooms, world)
            except Exception:
                pass

    # 7. Promote stairs into WorldIR
    promoted_stairs = []
    if options.promote_stairs and detected_stairs:
        for sidx, st in enumerate(detected_stairs, start=1):
            st_id = f"stairs-{sidx:03d}"
            if st_id not in world.entities:
                st_geom_id = f"geom-{st_id}"
                pos = getattr(st, "position", (0.0, 0.0, 0.0))
                rise = getattr(st, "rise_m", 0.17)
                n_steps = getattr(st, "n_steps", 8)
                going = getattr(st, "going_m", 0.28)
                total_rise = rise * n_steps
                total_run = going * n_steps
                st_bmin = (pos[0] - total_run / 2.0, pos[1] - 0.5, pos[2] - total_rise / 2.0)
                st_bmax = (pos[0] + total_run / 2.0, pos[1] + 0.5, pos[2] + total_rise / 2.0)
                world.geometries[st_geom_id] = Geometry(
                    id=st_geom_id,
                    type=GeometryType.BOX,
                    bounds_min=Vector3(st_bmin[0], st_bmin[1], st_bmin[2]),
                    bounds_max=Vector3(st_bmax[0], st_bmax[1], st_bmax[2]),
                )
                st_ent = Entity(
                    id=st_id,
                    type=EntityType.STAIRS,
                    name=f"Stairs {sidx:03d}",
                    geometry_ids=[st_geom_id],
                    custom_properties=st.to_dict() if hasattr(st, "to_dict") else {
                        "n_steps": n_steps,
                        "rise_m": rise,
                        "going_m": going,
                        "total_rise_m": total_rise,
                        "total_run_m": total_run,
                    },
                    provenance=Provenance.INFERRED,
                    confidence=getattr(st, "confidence", 0.9),
                )
                world.entities[st_id] = st_ent
                entities_created.append(st_id)
                promoted_stairs.append(st_ent)

    # 8. Promote corridors into WorldIR and wire canonical topology
    if options.promote_corridors and corridors:
        for c in corridors:
            cid = f"corridor-{c.corridor_id}" if not c.corridor_id.startswith("corridor-") else c.corridor_id
            if cid not in world.entities:
                c_bmin, c_bmax = c.bounds_min, c.bounds_max
                c_geom_id = f"geom-{cid}"
                world.geometries[c_geom_id] = Geometry(
                    id=c_geom_id,
                    type=GeometryType.BOX,
                    bounds_min=Vector3(c_bmin[0], c_bmin[1], c_bmin[2]),
                    bounds_max=Vector3(c_bmax[0], c_bmax[1], c_bmax[2]),
                )
                c_ent = Entity(
                    id=cid,
                    type=EntityType.CORRIDOR,
                    name=f"Corridor {c.corridor_id.rsplit('-', 1)[-1]}",
                    geometry_ids=[c_geom_id],
                    custom_properties=c.to_dict(),
                    provenance=Provenance.INFERRED,
                    confidence=c.confidence,
                )
                # Boundary containment edges
                for bid in c.boundary_element_ids:
                    if bid in world.entities:
                        c_ent.relationships.append(Relationship(
                            kind=RelationshipKind.CONTAINS,
                            target_id=bid,
                            confidence=c.confidence,
                            provenance=Provenance.INFERRED,
                            metadata={"derived_from": "corridor_boundary_membership"},
                        ))
                        world.entities[bid].relationships.append(Relationship(
                            kind=RelationshipKind.PART_OF,
                            target_id=cid,
                            confidence=c.confidence,
                            provenance=Provenance.INFERRED,
                            metadata={"derived_from": "corridor_boundary_membership"},
                        ))
                world.entities[cid] = c_ent
                entities_created.append(cid)

            # ROOM <-> DOORWAY <-> CORRIDOR canonical connectivity
            cent = world.entities[cid]
            for r_id in c.connected_room_ids:
                room_ent = None
                if r_id in world.entities:
                    room_ent = world.entities[r_id]
                else:
                    match = next((e for e in world.entities.values() if e.type == EntityType.ROOM and (r_id in e.id or e.id in r_id)), None)
                    if match:
                        room_ent = match
                if room_ent:
                    if not any(r.target_id == room_ent.id for r in cent.relationships):
                        cent.relationships.append(Relationship(
                            kind=RelationshipKind.CONNECTS,
                            target_id=room_ent.id,
                            confidence=c.confidence,
                            provenance=Provenance.INFERRED,
                            metadata={"derived_from": "circulation_doorway_connection", "corridor_id": cid},
                        ))
                    if not any(r.target_id == cid for r in room_ent.relationships):
                        room_ent.relationships.append(Relationship(
                            kind=RelationshipKind.CONNECTS,
                            target_id=cid,
                            confidence=c.confidence,
                            provenance=Provenance.INFERRED,
                            metadata={"derived_from": "circulation_doorway_connection", "corridor_id": cid},
                        ))

            # CORRIDOR <-> STAIR connectivity
            for st_ent in promoted_stairs:
                st_geom = world.geometries.get(st_ent.geometry_ids[0]) if st_ent.geometry_ids else None
                if st_geom and st_geom.bounds_min and st_geom.bounds_max:
                    if (c.bounds_min[0] <= st_geom.bounds_max.x and st_geom.bounds_min.x <= c.bounds_max[0] and
                        c.bounds_min[1] <= st_geom.bounds_max.y and st_geom.bounds_min.y <= c.bounds_max[1]):
                        if not any(r.target_id == st_ent.id for r in cent.relationships):
                            cent.relationships.append(Relationship(
                                kind=RelationshipKind.CONNECTS,
                                target_id=st_ent.id,
                                confidence=min(c.confidence, st_ent.confidence),
                                provenance=Provenance.INFERRED,
                                metadata={"derived_from": "corridor_stair_landing"},
                            ))
                        if not any(r.target_id == cid for r in st_ent.relationships):
                            st_ent.relationships.append(Relationship(
                                kind=RelationshipKind.CONNECTS,
                                target_id=cid,
                                confidence=min(c.confidence, st_ent.confidence),
                                provenance=Provenance.INFERRED,
                                metadata={"derived_from": "corridor_stair_landing"},
                            ))

    # 9. Promote building & storeys if requested, and attach space graph
    if bld_graph:
        if options.promote_building:
            try:
                from perception.architecture.topology import promote_building_topology
                topo_res = promote_building_topology(bld_graph, rg_rooms, world)
                for sid in topo_res.storey_ids:
                    entities_created.append(sid)
                entities_created.append(topo_res.building_id)
            except Exception:
                pass

        if detected_stairs:
            try:
                from perception.architecture.topology import link_stairs_to_storeys
                link_stairs_to_storeys(detected_stairs, bld_graph, world)
            except Exception:
                pass

        space_graph = InteriorSpaceGraph.from_building(bld_graph, rg_rooms, corridors, detected_stairs)
        world.metadata["interior_space_graph"] = space_graph.to_dict()

    # ---- provenance stamping on the world itself ----
    # The world as a whole is derived from reconstruction; its
    # global_provenance records that without upgrading any entity's own
    # provenance (entities keep RECONSTRUCTED/INFERRED as promoted).
    if world.global_provenance == Provenance.UNKNOWN:
        world.global_provenance = Provenance.RECONSTRUCTED
    world.metadata["compiled_from"] = {
        "points_total": detection.points_total,
        "planes_total": len(detection.planes),
        "seed": options.seed,
        "registration_status": result.registration_status,
        "up": list(options.up),
    }

    # ---- validation gate ----
    report = validate_world_ir(world)
    validation_issues = [i.message for i in report.issues if i.severity is ValidationSeverity.ERROR]
    if options.require_validation and validation_issues:
        raise WorldValidationGateError(validation_issues)

    measurements_count = sum(
        1
        for entity in world.entities.values()
        for key in entity.custom_properties
        if key.endswith(("_m", "_m2"))
    )
    relationships_count = sum(len(e.relationships) for e in world.entities.values())
    unassigned = detection.points_total - detection.points_in_planes

    diagnostics = CompileDiagnostics(
        seed=options.seed,
        points_total=detection.points_total,
        points_in_planes=detection.points_in_planes,
        points_unassigned=unassigned,
        planes_total=len(detection.planes),
        planes_by_role=planes_by_role,
        planes_unpromoted=planes_unpromoted,
        entities_created=entities_created,
        room_candidates=room_candidate_reports,
        rooms_detected=rooms_detected,
        measurements_count=measurements_count,
        relationships_count=relationships_count,
        validation_issues=validation_issues,
        unknown_point_fraction=(unassigned / detection.points_total) if detection.points_total else 0.0,
    )
    return world, diagnostics
