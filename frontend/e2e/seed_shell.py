"""E2E helper: commit a RECONSTRUCTED floor + wall shell as the head version of an existing world.

No public API creates a reconstructed shell without running the whole reconstruction pipeline, so the
completion e2e writes one through the same service the API uses (apps.api.worldstore_service.commit_version).
Usage (cwd = repo root, DATABASE_URL / STORAGE_ROOT / WORLDSTORE_ROOT = the e2e sandbox):
    python frontend/e2e/seed_shell.py <world_id> [width] [depth]
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


def shell(width: float, depth: float) -> WorldIR:
    wir = WorldIR(id="x")
    boxes = {"floor": ((0, 0, 0), (width, 0.1, depth)), "wall-n": ((0, 0, 0), (width, 2.6, 0.2))}
    for name, (lo, hi) in boxes.items():
        gid = f"g-{name}"
        wir.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo),
                                       bounds_max=Vector3(*hi), provenance=Provenance.RECONSTRUCTED)
        wir.entities[name] = Entity(id=name, type=EntityType.FLOOR if name == "floor" else EntityType.WALL,
                                    name=name, geometry_ids=[gid], provenance=Provenance.RECONSTRUCTED,
                                    confidence=0.8)
    return wir


async def main(world_id: str, width: float, depth: float) -> None:
    async with get_sessionmaker()() as db:
        wir = shell(width, depth)
        wir.id = world_id
        await svc.commit_version(db, world_id=world_id, world=wir, parent=None)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.0,
                     float(sys.argv[3]) if len(sys.argv) > 3 else 5.0))
