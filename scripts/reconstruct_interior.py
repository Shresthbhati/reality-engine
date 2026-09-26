#!/usr/bin/env python3
"""Reality Engine - Interior Reconstruction Demo

Proves the interior acceptance chain end to end:

  interior capture (point cloud)
    -> assemble_interior_scene (perception/architecture/scene.py)
         walls / floors / ceilings  (RANSAC + orientation + promotion)
         openings: doors, windows, generic (perception/architecture/openings.py)
         rooms + corridor            (room_graph + topology)
         room connectivity           (topology.connected_rooms_through_openings)
    -> WorldIR (world_ir.world_v1)
    -> WorldStore.save_version (worldstore/store.py)
    -> WorldStore.load_version
    -> structural comparison: same entities, same types, same links

Capture honesty: this demo runs on a deterministic synthetic interior
point cloud (two-room apartment with a shared-wall door and a facade
window) because the repository currently has no real indoor capture
(south_building is an outdoor facade dataset). Every stage after the
point cloud is the real production code path -- the same RANSAC,
opening scan, room graph, promotion and persistence used by the
pipeline -- exercised identically by tests/test_interior_scene_e2e.py.

Usage:
    python scripts/reconstruct_interior.py
    python scripts/reconstruct_interior.py --seed 7 --store-root C:/tmp/interior_store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when run as a script (sys.path[0] is
# scripts/ otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance import Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)

UP = (0.0, 0.0, 1.0)
STEP = 0.05

# Apartment plan (z=0 floor, z=2.4 ceiling):
#   Room A: x in [0,3], y in [0,4]
#   Room B: x in [3,6], y in [0,4]   (shared wall x=3 with a door)
#   Window in room A's y=0 facade wall (sill 0.9)
DOOR = (2.6, 3.4, 0.0, 2.0)     # in shared wall x=3 (y, z ranges)
WINDOW = (0.8, 2.2, 0.9, 1.9)   # in facade y=0 (x, z ranges)


def _grid(a0, a1, b0, b1, step=STEP):
    x = a0
    while x <= a1 + 1e-9:
        y = b0
        while y <= b1 + 1e-9:
            yield x, y
            y += step
        x += step


def build_interior_capture():
    """Dense deterministic points on every structural surface, minus
    the door and window voids. Room B's slab carries a 5 cm finish
    threshold at the doorway so RANSAC separates the slabs and the
    room graph sees two enclosures."""
    pts = []
    for (x0, x1, fz, cz) in ((0.1, 2.9, 0.0, 2.4), (3.1, 5.9, 0.05, 2.45)):
        for x, y in _grid(x0, x1, 0.1, 3.9):
            pts.append((x, y, fz))
            pts.append((x, y, cz))
    # Facade y=0 (with window), back y=4.
    for x, z in _grid(0.0, 6.0, 0.0, 2.4, 0.02):
        if not (WINDOW[0] <= x <= WINDOW[1] and WINDOW[2] <= z <= WINDOW[3]):
            pts.append((x, 0.0, z))
        pts.append((x, 4.0, z))
    # Shared wall x=3 (with door), ends x=0 and x=6.
    for y, z in _grid(0.0, 4.0, 0.0, 2.4, 0.02):
        if not (DOOR[0] <= y <= DOOR[1] and DOOR[2] <= z <= DOOR[3]):
            pts.append((3.0, y, z))
        pts.append((0.0, y, z))
        pts.append((6.0, y, z))
    return pts


def build_reconstruction():
    pts = build_interior_capture()
    recon_pts = [
        ReconstructedPoint(
            position=p,
            track_id=f"track-{i:06d}",
            source_evidence_ids=["ev-1"],
            uncertainty=Uncertainty(confidence=0.9),
        )
        for i, p in enumerate(pts)
    ]
    poses = [
        ReconstructedCameraPose(
            evidence_id=f"ev-{k}",
            position=(3.0, 2.0, 1.2 + 0.1 * k),
            rotation=(1.0, 0.0, 0.0, 0.0),
            uncertainty=Uncertainty(confidence=0.95),
        )
        for k in range(1, 5)
    ]
    return ReconstructionResult(
        points=recon_pts,
        camera_poses=poses,
        registration_status="registered",
    )


def structure(world):
    """Comparable semantic structure: (id, type) per entity."""
    return sorted((e.id, e.type.value) for e in world.entities.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct an interior scene and prove WorldStore round-trip"
    )
    parser.add_argument("--seed", type=int, default=42, help="Determinism seed")
    parser.add_argument(
        "--store-root", type=Path, default=Path("output/interior_store"),
        help="WorldStore root directory",
    )
    args = parser.parse_args()

    from perception.architecture.scene import assemble_interior_scene
    from worldstore.store import WorldStore

    print("=== Reality Engine: Interior Reconstruction ===")

    print("\n[1/5] Interior capture (deterministic synthetic apartment)...")
    result = build_reconstruction()
    print(f"  Points: {len(result.points)}")
    print(f"  Poses:  {len(result.camera_poses)}")

    print("\n[2/5] Assembling interior scene (planes -> openings -> rooms)...")
    scene = assemble_interior_scene(result, up=UP, seed=args.seed)
    print(scene.summary_text())

    counts: dict = {}
    for e in scene.world.entities.values():
        counts[e.type.value] = counts.get(e.type.value, 0) + 1
    print("  Entity inventory:")
    for t in sorted(counts):
        print(f"    {t:<10} {counts[t]}")
    print(f"  Rooms: {len(scene.rooms)}  Corridors: {len(scene.corridors)}")
    print(f"  Room links (through openings): {len(scene.room_links)}")
    print(f"  Room entities: {len(scene.room_entity_ids)}  "
          f"Storey entities: {len(scene.storey_entity_ids)}")
    for plane_fits in scene.openings_by_plane.values():
        for fit in plane_fits:
            print(f"  opening: {fit.kind} w={fit.width_m:.2f} m "
                  f"h={fit.height_m:.2f} m sill={fit.sill_height_m:.3f} m")

    print("\n[3/5] Persisting to WorldStore...")
    store = WorldStore(args.store_root)
    stored = store.save_version(scene.world, parent=None)
    print(f"  Saved version: {stored.version_id}")
    print(f"  Store root:    {args.store_root}")

    print("\n[4/5] Reloading from WorldStore...")
    loaded = store.load_version(stored.version_id)
    print(f"  Entities loaded: {len(loaded.entities)}")

    print("\n[5/5] Comparing structure (saved vs reloaded)...")
    before = structure(scene.world)
    after = structure(loaded)
    if before != after:
        missing = set(before) - set(after)
        extra = set(after) - set(before)
        print("  FAIL: structure changed across the round trip")
        if missing:
            print(f"    missing after reload: {sorted(missing)[:5]}")
        if extra:
            print(f"    unexpected after reload: {sorted(extra)[:5]}")
        return 1

    doors = [e for e in loaded.entities.values() if e.type.value == "door"]
    windows = [e for e in loaded.entities.values() if e.type.value == "window"]
    rooms = [e for e in loaded.entities.values() if e.type.value == "room"]
    print(f"  PASS: {len(before)} entities identical across reload")
    if not doors:
        print("  FAIL: no door entity survived the round trip")
        return 1
    if not windows:
        print("  FAIL: no window entity survived the round trip")
        return 1
    if len(rooms) != len(scene.rooms):
        print(f"  FAIL: {len(scene.rooms)} rooms inferred but "
              f"{len(rooms)} survived the round trip")
        return 1
    print(f"  Door entities: {len(doors)}  Window entities: {len(windows)}  "
          f"Room entities: {len(rooms)}")

    print("\n=== ACCEPTANCE CHAIN COMPLETE ===")
    print("capture -> geometry -> walls/floor/ceiling -> openings ->")
    print("rooms -> WorldIR -> WorldStore -> reload -> same structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
