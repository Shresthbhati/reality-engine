#!/usr/bin/env python3
"""Seed a dev/verify database with SYNTHETIC demo worlds so the UI can be checked in a browser.

    python scripts/seed_demo_worlds.py        # from the repo root; targets the verify DB unless env says otherwise

Worlds are compiled from the repo's synthetic fixtures (never real captures) with the same entry the fidelity
scorecard uses, then persisted through the app's own path: a World row + ``worldstore_service.commit_version``.
Idempotent: a world whose name already exists is skipped.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/verify_app.db")
os.environ.setdefault("STORAGE_ROOT", "./data/verify_artifacts")
os.environ.setdefault("WORLDSTORE_ROOT", "./data/verify_worldstore")
Path("data").mkdir(exist_ok=True)

from sqlalchemy import select  # noqa: E402

from apps.api import worldstore_service  # noqa: E402
from apps.api.db import get_sessionmaker, init_db  # noqa: E402
from apps.api.models import World, new_id  # noqa: E402


def _fidelity():
    sys.path.insert(0, str(ROOT / "scripts"))
    from room_fidelity_report import build_and_compile
    return build_and_compile(colors="albedo", voxel_m=0.08)[2]


def _fidelity_coarse():
    sys.path.insert(0, str(ROOT / "scripts"))
    from room_fidelity_report import build_and_compile
    return build_and_compile(colors="albedo", voxel_m=0.12)[2]


# (name, description, builder). The L-room fixture (synthetic/lroom.py) needs COLMAP on rendered images, so it is
# not "cheap"; a coarser-voxel compile of the same fidelity room stands in as the second world, named for what it is.
DEMOS = [
    ("Demo room (synthetic)", "Synthetic fidelity room (5x4x2.6 m, door+window, furniture). Not a real capture.", _fidelity),
    ("Demo room, coarse voxels (synthetic)", "Same synthetic fidelity room compiled at 0.12 m voxels. Not a real capture.",
     _fidelity_coarse),
]


async def main() -> None:
    await init_db()
    Session = get_sessionmaker()
    for name, desc, build in DEMOS:
        async with Session() as db:
            if (await db.execute(select(World).where(World.name == name))).scalars().first():
                print(f"skip (exists): {name}")
                continue
            world = build()
            w = World(id=new_id("wld"), name=name, description=desc)
            db.add(w)
            await db.commit()
            row = await worldstore_service.commit_version(db, world_id=w.id, world=world, parent=None)
            print(f"created: {w.id}  version={row.id}  {name}")


if __name__ == "__main__":
    asyncio.run(main())
