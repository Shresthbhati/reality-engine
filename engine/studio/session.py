"""StudioSession: ties viewport, inspector, outliner, selection and the
command pipeline together (spec §27 - Studio foundation).

This is the "Studio" object a UI would hold one of per open world. It owns
no truth of its own -- reads go through Inspector/Outliner, UI-local state
lives in Selection/Viewport, and every edit goes through
WorldCommandProcessor (spec §7's validated mutation path: typed command ->
validation -> permission -> WorldAPI -> event -> new version). Studio never
mutates `world` directly; `move_selected`/`create_entity`/`delete_selected`
are thin conveniences that build the right typed Command and hand it to the
processor, so a Studio edit is indistinguishable from any other command
caller -- same validation, same permission check, same event log entry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from world_ir import WorldIR

from engine.commands import (
    AddRelationshipCommand,
    CommandResult,
    CompileWorldCommand,
    CreateEntityCommand,
    DeleteEntityCommand,
    SetEntityTransformCommand,
    WorldCommandProcessor,
)
from engine.commands.permissions import PermissionPolicy
from engine.geometry.adjacency import infer_geometric_relationships
from engine.inspector.inspector import Inspector
from engine.physics.math3 import Vec3
from engine.render.viewport import Camera, Viewport
from events.bus import EventBus

from .outliner import Outliner
from .selection import Selection

_DEFAULT_CAMERA = Camera(position=Vec3(0.0, -10.0, 5.0), forward=Vec3(0.0, 1.0, -0.3))


class StudioSession:
    def __init__(
        self,
        world: WorldIR,
        camera: Optional[Camera] = None,
        event_bus: Optional[EventBus] = None,
        permission_policy: Optional[PermissionPolicy] = None,
        actor_id: Optional[str] = None,
    ):
        self.world = world
        self.inspector = Inspector(world)
        self.outliner = Outliner(world)
        self.selection = Selection()
        self.viewport = Viewport(camera or _DEFAULT_CAMERA)
        self.event_bus = event_bus or EventBus()
        self.commands = WorldCommandProcessor(world, self.event_bus, permission_policy=permission_policy)
        self.actor_id = actor_id

    def select(self, entity_id: str, *, additive: bool = False) -> None:
        if entity_id not in self.world.entities:
            raise KeyError(f"no such entity: {entity_id!r}")
        self.selection.select(entity_id, additive=additive)

    # ---- editing: every mutation goes through WorldCommandProcessor ----

    def create_entity(self, entity_id: str, entity_type: str, name: str = "") -> CommandResult:
        return self.commands.execute(
            CreateEntityCommand(entity_id=entity_id, entity_type=entity_type, name=name, actor_id=self.actor_id)
        )

    def move_selected(self, position: Tuple[float, float, float]) -> CommandResult:
        """Move the active selection. Raises ValueError if nothing is
        selected -- there is no implicit target to guess at."""
        active = self.selection.active
        if active is None:
            raise ValueError("move_selected: no entity is selected")
        return self.commands.execute(
            SetEntityTransformCommand(entity_id=active, position=position, actor_id=self.actor_id)
        )

    def delete_selected(self) -> CommandResult:
        """Delete the active selection, then drop it from Selection too --
        a deleted entity id has no business remaining "selected"."""
        active = self.selection.active
        if active is None:
            raise ValueError("delete_selected: no entity is selected")
        result = self.commands.execute(DeleteEntityCommand(entity_id=active, actor_id=self.actor_id))
        self.selection.deselect(active)
        return result

    def compile_reconstruction(self, result, compile_options=None) -> CommandResult:
        """Compile a ReconstructionResult into this session's world through
        the command pipeline: validated, permission-checked, event-logged,
        transactional (a gate failure leaves the world untouched -- see
        CompileWorldCommand). Diagnostics of the last successful compile
        are available via self.commands.last_compile_diagnostics.
        """
        return self.commands.execute(
            CompileWorldCommand(result=result, compile_options=compile_options, actor_id=self.actor_id)
        )

    def reconstruct_and_compile(self, evidence, orchestrator, compile_options=None):
        """Evidence -> real reconstruction backend -> compiled world, in one
        call. Before this method, `reconstruction/orchestrator.py`'s real
        backend selection/execution (COLMAP availability probing, fallback,
        per-attempt diagnostics -- see that module's docstring) had no path
        into a Studio session; a caller had to run the orchestrator itself,
        then separately remember to call compile_reconstruction() with its
        result. This is that missing glue.

        `orchestrator` is a `reconstruction.orchestrator.ReconstructionOrchestrator`
        (an explicit parameter, not a session-owned singleton -- which
        backend chain to use is a caller decision, not something Studio
        should default silently). Raises
        `reconstruction.orchestrator.ReconstructionOrchestrationError` if
        every candidate backend declines/fails (never a silently empty
        world); returns `(ReconstructionRun, CommandResult)` so a caller can
        inspect both the raw reconstruction diagnostics (which backend ran,
        reprojection stats, attempt log) and the compile outcome (gate
        pass/fail, diagnostics) from one call.
        """
        run = orchestrator.run(evidence)
        command_result = self.compile_reconstruction(run.result, compile_options=compile_options)
        return run, command_result

    def infer_and_add_adjacency(self, adjacency_margin: float = 0.0) -> List[CommandResult]:
        """Close the loop between the pure geometric-adjacency query and the
        validated mutation pipeline: run infer_geometric_relationships()
        against the current world, then execute an AddRelationshipCommand
        for each candidate pair so it actually becomes a real, event-logged
        Relationship on the source entity (provenance defaults to
        INFERRED -- see AddRelationshipCommand)."""
        results = []
        for source_id, target_id, kind in infer_geometric_relationships(self.world, adjacency_margin):
            results.append(self.commands.execute(
                AddRelationshipCommand(source_entity_id=source_id, target_entity_id=target_id, kind=kind, actor_id=self.actor_id)
            ))
        return results

    def active_entity_summary(self):
        """Full resolved Inspector view of whatever's currently active, or None."""
        active = self.selection.active
        return self.inspector.inspect_entity(active) if active else None

    def provenance_panel(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Everything a Studio provenance/evidence panel needs for one entity,
        in one call, instead of a UI stitching together four Inspector calls
        itself. Purely an aggregation of existing Inspector queries -- no new
        data, no new truth.
        """
        entity = self.inspector.get_entity(entity_id)
        if entity is None:
            return None

        materials = self.inspector.get_materials(entity_id)
        return {
            "entity_id": entity_id,
            "provenance": entity.provenance.value,
            "confidence": entity.confidence,
            "uncertainty": entity.uncertainty.to_dict(),
            "is_canonical": entity.provenance.value not in ("GENERATED", "UNKNOWN", "CONFLICT"),
            "observations": [o.to_dict() for o in self.inspector.get_evidence(entity_id)],
            "materials": [
                {"id": m.id, "name": m.name, "provenance": m.provenance.value, "confidence": m.confidence}
                for m in materials
            ],
            "measurements": self.inspector.get_measurements(entity_id),
        }

    def visible_entities(self) -> List[dict]:
        """Depth-sorted render list for the current camera, built from every
        entity that carries a transform (untransformed entities have no
        position to cull against and are omitted, not guessed at)."""
        candidates = []
        for entity in self.world.entities.values():
            if not entity.transform:
                continue
            pos = entity.transform.get("position") or entity.transform.get("translation")
            if pos is None:
                continue
            candidates.append((entity.id, Vec3(pos["x"], pos["y"], pos["z"]), entity.transform.get("radius", 0.0)))
        return self.viewport.cull(candidates)
