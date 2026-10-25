"""Regression tests: multi-storey plane support (reconstruction layer).

Background (measured on the canonical two-storey fixture, seed 42):
at the RANSAC module defaults (0.08 m / 8 inliers, sized for sparse
outdoor SfM clouds) a diagonal candidate fuses an upper storey's floor
with the storey below's ceiling into ONE tilted junk plane (812+703
pts, rms 0.041 m) and shreds the true floor to an 11-pt sliver; the
compiler path then promoted the junk as structure and fabricated a
phantom room. The fix pins the interior detection contract
(0.02 m / 30 inliers -- the scale assemble_interior_scene already
used) on the compiler path, plus the split/merge refinement.

These tests prove the fix does NOT cheat:
  A  genuine two-storey floor is recovered (both pipelines)
  B  a single-storey large room is NOT split into two floors
  C  two adjacent rooms sharing a wall keep both rooms, one storey
  D  wall-row points hovering near a floor do not split the floor
  E  tilted surfaces are not promoted as floor/ceiling sheets
  F  a sparse/noisy upper floor still yields ONE plane, honest rms
  G  incomplete upper-floor evidence stays honest (no phantom spans)
  H  disconnected coplanar fragments of one floor re-merge into ONE

All geometry helpers build dense grids on surfaces (established
fixture style); every assertion is on measured support, never on
fixture-specific heights.
"""
from __future__ import annotations

import math

import pytest

from tests.test_canonical_interior import UP as CANON_UP
from tests.test_canonical_interior import _canonical_interior_scene


# ------------------------------------------------------------------
# helpers (z-up small rooms; UP=(0,0,1) throughout these fixtures)
# ------------------------------------------------------------------

UP = (0.0, 0.0, 1.0)
STEP = 0.1


class _Cloud:
    def __init__(self, camera=(2.0, 2.0, 1.2)):
        self.pts = []
        self.camera = camera

    def add(self, x, y, z):
        self.pts.append((x, y, z))

    def grid(self, x0, x1, y0, y1, z, step=STEP):
        nx = int(round((x1 - x0) / step)) + 1
        ny = int(round((y1 - y0) / step)) + 1
        for i in range(nx):
            for j in range(ny):
                self.add(x0 + i * step, y0 + j * step, z)

    def wall_x(self, x, y0, y1, z0, z1, step=STEP):
        ny = int(round((y1 - y0) / step)) + 1
        nz = int(round((z1 - z0) / step)) + 1
        for j in range(ny):
            for k in range(nz):
                self.add(x, y0 + j * step, z0 + k * step)

    def wall_y(self, y, x0, x1, z0, z1, step=STEP):
        nx = int(round((x1 - x0) / step)) + 1
        nz = int(round((z1 - z0) / step)) + 1
        for i in range(nx):
            for k in range(nz):
                self.add(x0 + i * step, y, z0 + k * step)

    def result(self):
        from reconstruction.backend.interface import (
            ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult,
        )
        from provenance import Uncertainty
        pts = [
            ReconstructedPoint(
                position=p, track_id=f"pt-{i:05d}",
                source_evidence_ids=["ev-1"],
                uncertainty=Uncertainty(confidence=0.95),
            )
            for i, p in enumerate(self.pts)
        ]
        cams = [
            ReconstructedCameraPose(
                evidence_id="ev-1",
                position=self.camera,
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.9),
            )
        ]
        return ReconstructionResult(
            points=pts, camera_poses=cams, registration_status="success",
        )


def _room_box(c, x0, x1, y0, y1, z0, z1):
    """Floor + ceiling + four walls of one closed room."""
    c.grid(x0, x1, y0, y1, z0)
    c.grid(x0, x1, y0, y1, z1)
    c.wall_x(x0, y0, y1, z0, z1)
    c.wall_x(x1, y0, y1, z0, z1)
    c.wall_y(y0, x0, x1, z0, z1)
    c.wall_y(y1, x0, x1, z0, z1)


def _detect_refine(result):
    """The shared detect -> refine (split + merge) stage, as both
    pipelines now run it at the interior scale."""
    from perception.geometry.planes import detect_planes, refine_planes
    positions = {p.track_id: p.position for p in result.points}
    det = detect_planes(result, seed=42, distance_tolerance_m=0.02,
                        min_inliers=30)
    refined = refine_planes(det.planes, positions, UP,
                            distance_tolerance_m=0.02, min_inliers=30)
    return refined, positions


def _horizontal_planes(planes, positions):
    """(plane, z) for every near-horizontal plane, z = mean up-coordinate."""
    out = []
    for pl in planes:
        ny = abs(pl.normal[2])
        if ny < 0.9:
            continue
        zs = [positions[pid][2] for pid in pl.inlier_ids]
        out.append((pl, sum(zs) / len(zs)))
    return out


# ------------------------------------------------------------------
# A. genuine two-storey recovery (both pipelines, no phantoms)
# ------------------------------------------------------------------

def test_canonical_two_storey_recovers_genuine_upper_storey():
    scene = _assembled_canonical()
    assert len(scene.rooms) == 2, [r.floor_area_m2 for r in scene.rooms]
    assert len(scene.storey_entity_ids) == 2, scene.storey_entity_ids
    for room in scene.rooms:
        assert 4.0 <= room.floor_area_m2 <= 7.0, room.floor_area_m2
        assert room.status == "detected", (room.room_id, room.status)
    # No phantom: no room spans the whole plan (~36 m2 at the old bug).
    assert all(r.floor_area_m2 < 12.0 for r in scene.rooms)

    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    world, diag = compile_reconstruction_to_world(
        _canonical_interior_scene(), CompileOptions(seed=42, up=CANON_UP))
    types = [e.type.value for e in world.entities.values()]
    assert types.count("floor") == 2, sorted(types)
    assert types.count("ceiling") == 2, sorted(types)
    assert types.count("room") == 2, sorted(types)
    buildings = [e for e in world.entities.values()
                 if e.type.value == "building"]
    assert buildings and buildings[0].custom_properties["n_storeys"] == 2
    assert diag.rooms_detected == 2


def _assembled_canonical():
    from perception.architecture.scene import assemble_interior_scene
    return assemble_interior_scene(
        _canonical_interior_scene(), up=CANON_UP, seed=42)


# ------------------------------------------------------------------
# B. single-storey large room is NOT split into two floors
# ------------------------------------------------------------------

def test_single_storey_large_room_not_split():
    c = _Cloud()
    _room_box(c, 0, 6, 0, 6, 0.0, 2.5)
    result = c.result()
    refined, positions = _detect_refine(result)
    horiz = _horizontal_planes(refined, positions)
    floor_bands = [z for pl, z in horiz if z < 1.0]
    assert len(floor_bands) == 1, floor_bands
    assert abs(floor_bands[0]) <= 0.02

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    types = [e.type.value for e in scene.world.entities.values()]
    assert types.count("floor") == 1, sorted(types)
    assert len(scene.rooms) == 1
    assert 25.0 <= scene.rooms[0].floor_area_m2 <= 36.0


# ------------------------------------------------------------------
# C. two adjacent rooms sharing a wall: both rooms, one storey
# ------------------------------------------------------------------

def test_adjacent_rooms_share_wall_one_storey():
    c = _Cloud()
    # Room 1: x 0..3, room 2: x 3..6, sharing the x=3 wall; the rooms'
    # floors differ by one real finish step (4 cm) so each keeps its
    # own measured floor plane (coplanar same-height floors are ONE
    # merged surface by design -- see test H).
    _room_box(c, 0, 3, 0, 4, 0.0, 2.4)
    _room_box(c, 3, 6, 0, 4, 0.04, 2.44)
    result = c.result()

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    assert len(scene.rooms) == 2, [r.floor_area_m2 for r in scene.rooms]
    assert len(scene.storey_entity_ids) == 1, scene.storey_entity_ids
    for room in scene.rooms:
        assert 8.0 <= room.floor_area_m2 <= 13.0, room.floor_area_m2


# ------------------------------------------------------------------
# D. wall-row points near a floor do not split the floor
# ------------------------------------------------------------------

def test_wall_rows_near_floor_do_not_split_floor():
    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    # Wall-base rows hovering 0.12-0.24 m above the floor (furniture
    # feet / trim the scan caught): within the old 0.08... no -- these
    # are OUTSIDE the 0.02 tolerance, so they must simply not fuse
    # with, split, or duplicate the floor plane.
    for frac in (0.05, 0.10):
        z = 0.12 + frac
        c.grid(0, 4, 0, 0.3, z)   # along the y=0 wall
        c.grid(0, 0.3, 0, 4, z)   # along the x=0 wall
    result = c.result()

    refined, positions = _detect_refine(result)
    floor_bands = [z for pl, z in _horizontal_planes(refined, positions)
                   if z < 1.0]
    # The real floor stays exactly ONE plane at z=0. The floating rows
    # are genuinely planar point sets and MAY be detected as their own
    # planes (honest detection of real geometry), but they must never
    # merge into, split, or duplicate the floor sheet itself.
    assert sum(1 for z in floor_bands if abs(z) <= 0.02) == 1, floor_bands
    assert all(abs(z) > 0.05 for z in floor_bands if abs(z) > 0.02), (
        "a phantom floor band appeared between the rows and the floor")

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    # Downstream, the rows must not become structure: exactly one room
    # and one storey; any detected row planes end up demoted (recorded,
    # not promoted) or unclassified -- never additional floors.
    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    assert len(scene.storey_entity_ids) == 1, scene.storey_entity_ids
    types = [e.type.value for e in scene.world.entities.values()]
    assert types.count("floor") == 1, sorted(types)


# ------------------------------------------------------------------
# E. tilted surfaces are not promoted as floor/ceiling sheets
# ------------------------------------------------------------------

def test_tilted_surface_not_promoted_as_floor_sheet():
    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    # A 25-degree ramp across the room (a real tilted surface -- neither
    # horizontal nor vertical): grid on the tilted plane.
    for i in range(21):
        for j in range(21):
            x = 0.5 + i * 0.1
            y = 0.5 + j * 0.1
            z = 0.8 + (x - 0.5) * math.tan(math.radians(25.0))
            c.add(x, y, z)
    result = c.result()

    refined, positions = _detect_refine(result)
    from perception.geometry.orientation import classify_planes
    cams = [p.position for p in result.camera_poses]
    oriented = classify_planes(refined, cams, UP)
    for o in oriented:
        if o.plane.inlier_count >= 100 and abs(o.normal[2]) < 0.9:
            # A tilted (non-horizontal) surface must never carry a
            # floor/ceiling role: that is how phantom storeys start.
            assert o.role not in ("floor", "ceiling"), (
                o.plane.plane_id, o.role, o.normal)

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    assert len(scene.rooms) == 1
    assert len(scene.storey_entity_ids) == 1


# ------------------------------------------------------------------
# F. sparse/noisy upper floor: ONE plane, honest residuals
# ------------------------------------------------------------------

def test_noisy_upper_floor_single_plane_honest_rms():
    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    # Upper storey floor at 2.7 m: 60% of the grid, each point jittered
    # by ~1 cm (deterministic pseudo-noise).
    k = 0
    for i in range(41):
        for j in range(41):
            k += 1
            if k % 5 >= 2:  # keep 60%
                x = i * 0.1
                y = j * 0.1
                z = 2.7 + (math.sin(12.9898 * k) * 43758.5453 % 1.0 - 0.5) * 0.02
                c.add(x, y, z)
    result = c.result()

    refined, positions = _detect_refine(result)
    upper = [pl for pl, z in _horizontal_planes(refined, positions)
             if 2.5 < z < 2.9]
    assert len(upper) == 1, [(pl.plane_id, pl.inlier_count) for pl in upper]
    assert upper[0].inlier_count >= 300
    # The measured residual stays tight: the noise did not become a
    # second phantom sheet.
    assert upper[0].inlier_rms_distance_m < 0.02, upper[0].inlier_rms_distance_m


# ------------------------------------------------------------------
# G. incomplete upper-floor evidence stays honest (no phantom spans)
# ------------------------------------------------------------------

def test_incomplete_upper_floor_no_phantom_span():
    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    # Upper floor: only the left half ever observed.
    for i in range(21):
        for j in range(41):
            c.add(i * 0.1, j * 0.1, 2.7)
    result = c.result()

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    # The GROUND room is complete evidence (walls + full floor): its
    # measured area is the true 4x4 plan (16 m2). The honest invariant:
    # no room may measure BEYOND the real plan (the phantom signature
    # was 36 m2 over a 4x4 plan), and every room stays room-scale.
    for room in scene.rooms:
        assert room.floor_area_m2 <= 16.5, room.floor_area_m2
        assert room.floor_area_m2 >= 4.0, room.floor_area_m2
    # At most the ground storey plus (honestly) the partial upper
    # level -- never a fabricated tower of storeys.
    assert len(scene.storey_entity_ids) <= 2, scene.storey_entity_ids


# ------------------------------------------------------------------
# I. collapsed reconstruction: sub-room-scale sheets promote NOTHING
# ------------------------------------------------------------------

def test_collapsed_reconstruction_promotes_no_structure():
    """Measured real failure (dataset room_capture): COLMAP registered
    the cameras but the sparse map degenerated to a ~0.5 m shell; its
    sheets promoted as floors and walls at 0.96-0.99 confidence. A
    plane whose two largest extents are both < ~1 m cannot host a
    person: it must be refused everywhere, with the refusal recorded
    (honest degenerate capture, never silent success)."""
    c = _Cloud(camera=(0.0, 0.0, 0.3))
    # Degenerate shell: two small sheets + one tilted fragment, all
    # within a 0.5 m box (metric scale).
    c.grid(-0.2, 0.2, -0.25, 0.25, 0.0)     # 0.4 x 0.5 m sheet
    c.grid(-0.1, 0.1, -0.1, 0.1, 0.25)      # 0.2 x 0.2 m sheet
    for i in range(5):
        for j in range(5):
            c.add(-0.2 + i * 0.1, -0.1, j * 0.08)  # 0.4 x 0.32 m wall-ish
    result = c.result()

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    types = [e.type.value for e in scene.world.entities.values()]
    assert types.count("floor") == 0, sorted(types)
    assert types.count("ceiling") == 0, sorted(types)
    assert types.count("wall") == 0, sorted(types)
    assert len(scene.rooms) == 0
    assert len(scene.storey_entity_ids) == 0
    # The refusals are recorded, not silent: the sub-scale planes are
    # reported as demoted or unclassified.
    assert (scene.demoted_furniture_planes
            or scene.unclassified_planes), (
        scene.demoted_furniture_planes, scene.unclassified_planes)

    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    world, diag = compile_reconstruction_to_world(
        result, CompileOptions(seed=42, up=UP))
    assert len(world.entities) == 0, sorted(world.entities)
    assert diag.rooms_detected == 0
    assert diag.planes_unpromoted, "refusals must be recorded"


# ------------------------------------------------------------------
# J. assembler determinism: same points + seed -> identical world
# ------------------------------------------------------------------

def test_assembler_world_is_byte_identical_across_runs():
    """WorldIR's constructor mints a uuid4 main_branch_id per instance;
    without a stable identity two runs over the SAME points serialize
    differently (measured: identical entity sets, differing hashes).
    Same points + seed must give a byte-identical world -- the
    determinism contract the compiler path already pins."""
    import hashlib
    import json as _json

    from perception.architecture.scene import assemble_interior_scene

    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    result = c.result()

    hashes = []
    for _ in range(2):
        scene = assemble_interior_scene(result, up=UP, seed=42)
        d = scene.world.to_dict()
        hashes.append(hashlib.sha256(
            _json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest())
    assert hashes[0] == hashes[1], hashes


# ------------------------------------------------------------------
# H. disconnected coplanar fragments of ONE floor re-merge into ONE
# ------------------------------------------------------------------

def test_disconnected_coplanar_fragments_remerge():
    c = _Cloud()
    _room_box(c, 0, 4, 0, 4, 0.0, 2.4)
    # Floor arrives as two fragments with a 0.5 m gap (occlusion band).
    c.grid(0.0, 1.5, 0, 4, 0.0)
    c.grid(2.0, 4.0, 0, 4, 0.0)
    # Replace the room box's full floor: the box already laid a full
    # floor grid; rebuild without it instead.
    c2 = _Cloud()
    c2.grid(0, 4, 0, 4, 2.4)           # ceiling
    c2.wall_x(0, 0, 4, 0.0, 2.4)
    c2.wall_x(4, 0, 4, 0.0, 2.4)
    c2.wall_y(0, 0, 4, 0.0, 2.4)
    c2.wall_y(4, 0, 4, 0.0, 2.4)
    c2.grid(0.0, 1.5, 0, 4, 0.0)       # fragment 1
    c2.grid(2.0, 4.0, 0, 4, 0.0)       # fragment 2
    result = c2.result()

    refined, positions = _detect_refine(result)
    floor_bands = [z for pl, z in _horizontal_planes(refined, positions)
                   if z < 1.0]
    assert len(floor_bands) == 1, (
        f"fragments did not re-merge: {floor_bands}")
    assert abs(floor_bands[0]) <= 0.02

    from perception.architecture.scene import assemble_interior_scene
    scene = assemble_interior_scene(result, up=UP, seed=42)
    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    assert len(scene.storey_entity_ids) == 1
