"""StudioSession: ties viewport, inspector, outliner and selection together
(spec §27 - Studio foundation).

This is the "Studio" object a UI would hold one of per open world -- it owns
no new truth, only read access (Inspector/Outliner) plus UI-local state
(Selection, Viewport camera). No transform/edit tools live here: mutating
WorldIR needs the validated command pipeline described in spec §7
(intent -> parsing -> validation -> permission -> WorldAPI), which does not
exist in this repo yet -- adding editing here would be building it
un-validated, exactly what §7 forbids. That is real, separate, undone work,
not part of this increment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from world_ir import WorldIR

from engine.inspector.inspector import Inspector
from engine.physics.math3 import Vec3
from engine.render.viewport import Camera, Viewport

from .outliner import Outliner
from .selection import Selection

_DEFAULT_CAMERA = Camera(position=Vec3(0.0, -10.0, 5.0), forward=Vec3(0.0, 1.0, -0.3))


class StudioSession:
    def __init__(self, world: WorldIR, camera: Optional[Camera] = None):
        self.world = world
        self.inspector = Inspector(world)
        self.outliner = Outliner(world)
        self.selection = Selection()
        self.viewport = Viewport(camera or _DEFAULT_CAMERA)

    def select(self, entity_id: str, *, additive: bool = False) -> None:
        if entity_id not in self.world.entities:
            raise KeyError(f"no such entity: {entity_id!r}")
        self.selection.select(entity_id, additive=additive)

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
