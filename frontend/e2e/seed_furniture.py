"""E2E helper: commit a metric room (floor + wall) with ONE sofa as the head version of an existing world.

Same reason as seed_shell.py: no public API creates a reconstructed room with detected furniture without running the whole
pipeline, so the design e2e writes one through the service the API uses (apps.api.worldstore_service.commit_version).
Usage (cwd = repo root, DATABASE_URL / STORAGE_ROOT / WORLDSTORE_ROOT = the e2e sandbox):
    python frontend/e2e/seed_furniture.py <world_id>
"""

from __future__ import annotations

import asyncio
import sys

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.schema_v1 import Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

from apps.api import worldstore_service as svc
from apps.api.db import get_sessionmaker


def _dim(v: float) -> dict:
    return {"value": v, "unit": "m", "state": "ESTIMATED", "basis": "e2e seed"}


def room() -> WorldIR:
    wir = WorldIR(id="x")
    wir.metadata["scale"] = {"state": "metric"}
    boxes = {"floor": ((0, 0, 0), (6, 0.1, 5)), "wall-n": ((0, 0, 0), (6, 2.6, 0.2))}
    for name, (lo, hi) in boxes.items():
        gid = f"g-{name}"
        wir.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi),
                                       provenance=Provenance.RECONSTRUCTED)
        wir.entities[name] = Entity(id=name, type=EntityType.FLOOR if name == "floor" else EntityType.WALL, name=name,
                                    geometry_ids=[gid], provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    wir.geometries["g-sofa"] = Geometry(id="g-sofa", type=GeometryType.BOX, bounds_min=Vector3(2, 0, 1.5), bounds_max=Vector3(4, 0.85, 2.5),
                                        provenance=Provenance.ESTIMATED)
    wir.entities["sofa"] = Entity(
        id="sofa", type=EntityType.FURNITURE, name="Sofa", transform={"position": {"x": 3.0, "y": 0.0, "z": 2.0}},
        geometry_ids=["g-sofa"], semantic_labels=["furniture_candidate"], provenance=Provenance.INFERRED, confidence=0.6,
        custom_properties={"category": "sofa", "entity_class": "furniture.sofa", "yaw_rad": 0.0,
                           "dimensions": {"length_m": _dim(2.0), "width_m": _dim(1.0), "height_m": _dim(0.85)}})
    return wir


async def main(world_id: str) -> None:
    async with get_sessionmaker()() as db:
        wir = room()
        wir.id = world_id
        await svc.commit_version(db, world_id=world_id, world=wir, parent=None)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
