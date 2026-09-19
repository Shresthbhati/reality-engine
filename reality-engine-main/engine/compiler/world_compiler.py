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
                room_promotion = promote_room_to_entity(room, world, room_entity_id)
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
