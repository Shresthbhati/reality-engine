"""End-to-end interior scene assembly (INTERIOR RECONSTRUCTION MISSION
acceptance chain).

REAL synthetic geometry: a deterministic two-room apartment point
cloud (shared wall with a door, window on the facade wall, corridor-
width passage) is assembled into a ReconstructionResult the same way
the canned-backend pipeline does, then:

    assemble_interior_scene
      -> structural planes promoted
      -> door + window openings detected and promoted as entities
      -> rooms inferred with bounds
      -> room connectivity through the shared-wall door
      -> WorldIR validation
      -> WorldStore.save_version -> load_version -> same structure

The point cloud is geometry, not a mock: planes are detected by the
real RANSAC, openings by the real coverage scan, rooms by the real
graph. The ReconstructionResult is constructed directly (that is what
a backend returns); the reconstruction ORCHESTRATOR is separately
covered by its own suite.
"""

from __future__ import annotations

import pytest

from provenance import Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)

UP = (0.0, 0.0, 1.0)

# Apartment plan (z=0 floor, z=2.4 ceiling):
#   Room A: x in [0,3], y in [0,4]
#   Room B: x in [3,6], y in [0,4]   (shared wall x=3 with a door)
#   Window in room A's y=0 facade wall (sill 0.9)
STEP = 0.05


def _grid(a0, a1, b0, b1, step=STEP):
    na = int(round((a1 - a0) / step))
    nb = int(round((b1 - b0) / step))
    for i in range(na + 1):
        for j in range(nb + 1):
            yield a0 + i * step, b0 + j * step


def _interior_points():
    """Dense deterministic points on every structural surface, minus
    the door and window voids."""
    pts = []

    door = (2.6, 3.4, 0.0, 2.0)     # in shared wall x=3 (y, z ranges)
    window = (0.8, 2.2, 0.9, 1.9)   # in facade y=0 (x, z ranges)

    # Floor/ceiling: two planes each. Room B's slab carries a 5 cm
    # finish threshold at the doorway (real construction: adjacent
    # slabs are coplanar in CAD but measurably distinct planes), so
    # RANSAC separates them and the room graph sees two enclosures.
    for (x0, x1, fz, cz) in ((0.1, 2.9, 0.0, 2.4), (3.1, 5.9, 0.05, 2.45)):
        for x, y in _grid(x0, x1, 0.1, 3.9):
            pts.append((x, y, fz))
            pts.append((x, y, cz))
    # Facade y=0 (with window), back y=4.
    for x, z in _grid(0.0, 6.0, 0.0, 2.4, 0.02):
        if not (window[0] <= x <= window[1] and window[2] <= z <= window[3]):
            pts.append((x, 0.0, z))
        pts.append((x, 4.0, z))
    # Shared wall x=3 (with door), ends x=0 and x=6.
    for y, z in _grid(0.0, 4.0, 0.0, 2.4, 0.02):
        if not (door[0] <= y <= door[1] and door[2] <= z <= door[3]):
            pts.append((3.0, y, z))
        pts.append((0.0, y, z))
        pts.append((6.0, y, z))
    return pts


def _reconstruction():
    pts = _interior_points()
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


class TestInteriorSceneAssembly:
    def test_full_interior_chain(self):
        from perception.architecture.scene import assemble_interior_scene

        result = _reconstruction()
        scene = assemble_interior_scene(result, up=UP, seed=42)

        # Structural planes became entities.
        types = [e.type.value for e in scene.world.entities.values()] \
            if hasattr(scene.world.entities, "values") else \
            [e.type.value for e in scene.world.entities]
        assert "wall" in types and "floor" in types and "ceiling" in types

        # Openings: the door (shared wall) + window (facade).
        opening_types = [t for t in types if t in ("door", "window", "opening")]
        assert "door" in opening_types
        assert "window" in opening_types

        # Rooms inferred: exactly two enclosures (the 5 cm slab step
        # separates them), each covering the full ~3x4 m room footprint
        # -- not one degenerate cell spanning the whole apartment (the
        # pre-fix bug), not fragmented sub-cells (the fragment bug).
        assert len(scene.rooms) == 2
        areas = sorted(r.floor_area_m2 for r in scene.rooms)
        assert all(9.0 < a < 15.0 for a in areas), areas

        # Rooms are promoted into WorldIR (survive WorldStore round trips)
        # and so is the measured storey layer.
        assert sorted(scene.room_entity_ids) == sorted(
            eid for eid in scene.world.entities
            if scene.world.entities[eid].type.value == "room"
        )
        assert len(scene.room_entity_ids) == 2
        assert len(scene.storey_entity_ids) >= 1

        # Room connectivity through the shared-wall door.
        assert scene.room_links == [("room-001", "room-002")]

        # Opening geometry is honest: the door meets the floor (sill 0),
        # the window sits at its real sill height -- a negative sill means
        # the floor reference was contaminated (the merged-slab bug).
        sills = {}
        for plane_fits in scene.openings_by_plane.values():
            for fit in plane_fits:
                sills[fit.kind] = fit.sill_height_m
        assert sills.get("door", None) == pytest.approx(0.0, abs=1e-6)
        assert sills.get("window", None) == pytest.approx(0.9, abs=1e-6)

        # The world passes its own validation -- clean, not merely
        # "no relationship complaints".
        assert scene.validation_issues == []

    def test_deterministic(self):
        from perception.architecture.scene import assemble_interior_scene

        a = assemble_interior_scene(_reconstruction(), up=UP, seed=42)
        b = assemble_interior_scene(_reconstruction(), up=UP, seed=42)
        # world_v1 WorldIR stores entities as a plain dict keyed by id.
        ids_a = sorted(a.world.entities)
        ids_b = sorted(b.world.entities)
        assert ids_a == ids_b
        assert a.room_links == b.room_links
        assert [o for l in (a.opening_entity_ids,) for o in l] == \
               [o for l in (b.opening_entity_ids,) for o in l]

    def test_empty_reconstruction_refuses(self):
        from perception.architecture.scene import (
            SceneAssemblyError,
            assemble_interior_scene,
        )

        empty = ReconstructionResult(
            points=[],
            camera_poses=[],
            registration_status="failed",
        )
        with pytest.raises(SceneAssemblyError):
            assemble_interior_scene(empty, up=UP, seed=42)


class TestWorldStoreRoundTrip:
    def test_save_reload_same_structure(self, tmp_path):
        from perception.architecture.scene import assemble_interior_scene
        from worldstore.store import WorldStore

        scene = assemble_interior_scene(_reconstruction(), up=UP, seed=42)
        store = WorldStore(tmp_path / "store")
        stored = store.save_version(scene.world, parent=None)
        loaded = store.load_version(stored.version_id)

        # Same semantic structure after the round trip.
        def structure(w):
            return sorted(
                (e.id, e.type.value)
                for e in w.entities.values()
            )
        assert structure(loaded) == structure(scene.world)

        # Entity count and openings survive.
        assert len(loaded.entities) == len(scene.world.entities)
        doors = [
            e for e in loaded.entities.values()
            if e.type.value == "door"
        ]
        windows = [
            e for e in loaded.entities.values()
            if e.type.value == "window"
        ]
        rooms = [
            e for e in loaded.entities.values()
            if e.type.value == "room"
        ]
        assert doors and windows

        # Semantic facts -- room bounds, measured opening dimensions --
        # survive the round trip exactly (the acceptance test is "same
        # semantic structure", not just the same id list).
        assert len(rooms) == 2
        live_by_id = {e.id: e for e in scene.world.entities.values()}
        for ent in loaded.entities.values():
            assert ent.custom_properties == live_by_id[ent.id].custom_properties
            assert ent.relationships == live_by_id[ent.id].relationships
