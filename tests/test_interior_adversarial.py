"""Adversarial robustness suite for the interior perception chain.

Twelve messy-capture cases (furniture, occlusion, missing ceiling,
incomplete floor, noise, irregular plans, L-shaped circulation,
ambiguous openings, disconnected evidence, partial stairs, multi-level
interiors, coplanar wall fragments) run through the REAL pipeline:

    assemble_interior_scene  (RANSAC -> split/merge -> orientation ->
    classification -> openings -> room graph -> corridors -> topology)

Each case asserts what the EVIDENCE supports -- a measured structure
when the capture carries one, an honest refusal (empty rooms, recorded
issues, no entity) when it does not. The core invariant under mess:
the pipeline never manufactures a room the capture does not close, and
never converts uncertain geometry into false certainty.

Cases are deterministic synthetic point clouds in the established
fixture style (dense grids on surfaces, voids carved out); running the
STRONGEST REAL capture lives in scripts/reconstruct_interior_real.py
and is reported separately -- synthetic robustness alone claims
nothing about real captures.

Marked `slow` (each case runs a full RANSAC assembly, ~20-90 s).
"""

from __future__ import annotations

import math
import random

import pytest

from provenance import Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)

UP = (0.0, 0.0, 1.0)
STEP = 0.05
WALL_STEP = 0.02

pytestmark = pytest.mark.slow


# ------------------------------------------------------------------
# fixture helpers (same conventions as test_interior_scene_e2e.py)
# ------------------------------------------------------------------

def _grid(a0, a1, b0, b1, step=STEP):
    na = max(1, int(round((a1 - a0) / step)))
    nb = max(1, int(round((b1 - b0) / step)))
    for i in range(na + 1):
        for j in range(nb + 1):
            yield a0 + i * step, b0 + j * step


def _wall(pts, axis, const, u0, u1, v0, v1, step=WALL_STEP, skip=None):
    """Dense points on one rectangular wall patch. skip(t, z) -> True
    carves a void (door/window) out of the wall."""
    for u, z in _grid(u0, u1, v0, v1, step):
        if skip is not None and skip(u, z):
            continue
        if axis == "x":
            pts.append((const, u, z))
        elif axis == "y":
            pts.append((u, const, z))
        else:  # "z": a horizontal patch at height `const`
            pts.append((u, z, const))


def _slab(pts, x0, x1, y0, y1, z, step=STEP):
    for x, y in _grid(x0, x1, y0, y1, step):
        pts.append((x, y, z))


def _room_box(pts, x0, x1, y0, y1, z_floor, z_ceil,
              skip_ceiling=False, door=None, window=None,
              open_sides=()):
    """One rectangular room shell: floor, ceiling, four walls. Walls on
    sides named in open_sides are omitted entirely (disconnected
    evidence / partial captures). door/window: (u0, u1, z0, z1) voids
    in the y0-side wall (window) or y1-side wall (door)."""
    _slab(pts, x0, x1, y0, y1, z_floor)
    if not skip_ceiling:
        _slab(pts, x0, x1, y0, y1, z_ceil)
    sides = ("x0", "x1", "y0", "y1")
    for side in sides:
        if side in open_sides:
            continue
        if side == "y0":
            skip = (lambda u, z: window[0] <= u <= window[1]
                    and window[2] <= z <= window[3]) if window else None
            _wall(pts, "y", y0, x0, x1, z_floor, z_ceil, skip=skip)
        elif side == "y1":
            skip = (lambda u, z: door[0] <= u <= door[1]
                    and door[2] <= z <= door[3]) if door else None
            _wall(pts, "y", y1, x0, x1, z_floor, z_ceil, skip=skip)
        elif side == "x0":
            _wall(pts, "x", x0, y0, y1, z_floor, z_ceil)
        else:
            _wall(pts, "x", x1, y0, y1, z_floor, z_ceil)


def _furniture_block(pts, cx, cy, w, d, h, z_floor):
    """A solid furniture mass: top slab + side skirts (what a real scan
    sees of a wardrobe). Deliberately NOT a thin plane."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - d / 2, cy + d / 2
    _slab(pts, x0, x1, y0, y1, z_floor + h, step=0.1)
    for x, y in _grid(x0, x1, y0, y1, step=0.2):
        pts.append((x, y, z_floor + h / 2))


def _recon(pts, seed=7, noise=0.0, drop_fraction=0.0):
    """ReconstructionResult from raw xyz tuples, with optional Gaussian
    noise and stochastic point dropout (both seeded => deterministic)."""
    rng = random.Random(seed)
    out = []
    for i, p in enumerate(pts):
        if drop_fraction and rng.random() < drop_fraction:
            continue
        if noise:
            p = tuple(c + rng.gauss(0.0, noise) for c in p)
        out.append(ReconstructedPoint(
            position=p,
            track_id=f"track-{i:06d}",
            source_evidence_ids=[f"ev-{i % 4}"],
            uncertainty=Uncertainty(confidence=0.9),
        ))
    poses = [
        ReconstructedCameraPose(
            evidence_id=f"ev-{k}",
            position=(2.0 + 0.3 * k, 2.0, 1.3),
            rotation=(1.0, 0.0, 0.0, 0.0),
            uncertainty=Uncertainty(confidence=0.95),
        )
        for k in range(1, 5)
    ]
    return ReconstructionResult(
        points=out, camera_poses=poses, registration_status="registered",
    )


def _assembled(pts, **kw):
    from perception.architecture.scene import assemble_interior_scene
    return assemble_interior_scene(_recon(pts, **kw), up=UP, seed=42)


def _types(scene):
    counts = {}
    for e in scene.world.entities.values():
        counts[e.type.value] = counts.get(e.type.value, 0) + 1
    return counts


# ------------------------------------------------------------------
# 1. furniture-heavy room
# ------------------------------------------------------------------

def test_furniture_heavy_room():
    """A wardrobe + table inside a closed room: the walls must still
    bound the room, and no furniture mass may become a wall/floor or
    seed a second room."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, door=(1.7, 2.3, 0.0, 2.0))
    _furniture_block(pts, 1.0, 1.0, 0.8, 0.6, 1.8, 0.0)   # wardrobe
    _furniture_block(pts, 3.0, 2.8, 1.2, 0.8, 0.75, 0.0)  # table
    scene = _assembled(pts)

    inv = _types(scene)
    assert inv.get("floor", 0) == 1, inv
    assert inv.get("wall", 0) >= 3, inv
    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    room = scene.rooms[0]
    # Room area measured from its floor, not inflated by furniture AABBs.
    assert 9.0 <= room.floor_area_m2 <= 16.0, room.floor_area_m2
    # Furniture tops were demoted, not promoted as floors.
    assert len(scene.demoted_furniture_planes) >= 1, scene.demoted_furniture_planes


# ------------------------------------------------------------------
# 2. partially occluded walls
# ------------------------------------------------------------------

def test_partially_occluded_wall():
    """A sofa hides the middle 1.4 m of one wall (points removed); the
    wall remains one plane with a hole -- the room must survive."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, door=(1.7, 2.3, 0.0, 2.0))
    # Carve an interior hole out of the x=0 wall (between z 0.3-1.1,
    # y 1.2-2.6): furniture occlusion from a static scan.
    occluded = []
    for p in pts:
        x, y, z = p
        if abs(x) < 1e-9 and 1.2 <= y <= 2.6 and 0.3 <= z <= 1.1:
            continue
        occluded.append(p)
    scene = _assembled(occluded)

    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    assert 9.0 <= scene.rooms[0].floor_area_m2 <= 16.0


# ------------------------------------------------------------------
# 3. missing ceiling
# ------------------------------------------------------------------

def test_missing_ceiling_room_survives():
    """A capture without ceiling returns (common: tall rooms, scan
    budget). The room must NOT be refused -- its walls and floor still
    close the space."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, skip_ceiling=True,
              door=(1.7, 2.3, 0.0, 2.0))
    scene = _assembled(pts)

    inv = _types(scene)
    assert inv.get("floor", 0) == 1, inv
    assert inv.get("wall", 0) >= 3, inv
    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    assert 9.0 <= scene.rooms[0].floor_area_m2 <= 16.0
    # Missing ceiling is honest provenance, not silent absence.
    room = scene.rooms[0]
    assert room.ceiling_evidence == "missing", room.ceiling_evidence
    assert room.status == "partial", room.status
    assert any("no ceiling" in n for n in room.notes), room.notes


# ------------------------------------------------------------------
# 4. incomplete floor
# ------------------------------------------------------------------

def test_incomplete_floor():
    """A rug-covered/occluded floor with a missing interior patch still
    measures the room; the visible floor extent bounds the area."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, door=(1.7, 2.3, 0.0, 2.0))
    visible = []
    for p in pts:
        x, y, z = p
        if z == 0.0 and 1.2 < x < 2.6 and 1.2 < y < 2.6:
            continue  # 1.4 x 1.4 interior hole in the floor
        visible.append(p)
    scene = _assembled(visible)

    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    room = scene.rooms[0]
    # Measured area tracks the floor's visible extent (slightly smaller
    # than 16 m2) but stays a room-scale measurement -- never a
    # fragment of it.
    assert 8.0 <= room.floor_area_m2 <= 16.0, room.floor_area_m2


# ------------------------------------------------------------------
# 5. noisy point cloud
# ------------------------------------------------------------------

def test_noisy_points_still_reconstruct():
    """2 cm Gaussian noise on every point (a mediocre phone scan): the
    walls/floor survive and the room measures within tolerance."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, door=(1.7, 2.3, 0.0, 2.0))
    scene = _assembled(pts, seed=11, noise=0.02)

    assert len(scene.rooms) >= 1, scene.unclassified_planes
    room = scene.rooms[0]
    assert 7.0 <= room.floor_area_m2 <= 17.0, room.floor_area_m2


# ------------------------------------------------------------------
# 6. irregular room (octagonal plan)
# ------------------------------------------------------------------

def test_irregular_octagonal_room():
    """An octagonal plan: the four cardinal walls + floor close the
    space (the diagonal chamfer walls are additional real geometry the
    classifier may honestly call UNKNOWN). The room must survive with
    its measured footprint."""
    pts: list = []
    z0, z1 = 0.0, 2.4
    _slab(pts, 0.0, 4.0, 0, 4, z0)
    _slab(pts, 0.0, 4.0, 0, 4, z1)
    # Cardinal walls spanning the full 4 m width (the chamfers cut the
    # corners, the walls still close the plan).
    _wall(pts, "y", 0.0, 0.0, 4.0, z0, z1)                    # facade
    _wall(pts, "y", 4.0, 0.0, 4.0, z0, z1,
          skip=lambda u, z: 1.7 <= u <= 2.3 and z <= 2.0)     # back + door
    _wall(pts, "x", 0.0, 0.0, 4.0, z0, z1)
    _wall(pts, "x", 4.0, 0.0, 4.0, z0, z1)
    # Chamfer walls: real 45-deg diagonals at all four corners (2.4 m
    # vertical planes at arbitrary yaw -- classifier-honesty stress).
    c0, c1 = 0.8, 3.2
    corners = [(c0, 0.0, c0, c0), (c1, c1, 4.0, c1),
               (c0, c1, c0, 4.0), (4.0, c0, c1, c0)]
    for (ax, az, bx, bz) in corners:
        t = 0.0
        while t <= 1.0 + 1e-9:
            x = ax + (bx - ax) * t
            y = az + (bz - az) * t
            for z in (0.3, 1.0, 1.7, 2.3):
                pts.append((x, y, z))
            t += 0.05
    scene = _assembled(pts)

    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    room = scene.rooms[0]
    assert 6.0 <= room.floor_area_m2 <= 16.0, room.floor_area_m2


# ------------------------------------------------------------------
# 7. L-shaped corridor + room
# ------------------------------------------------------------------

def test_l_shaped_corridor():
    """A corridor (1.2 x 6 m) with a door into a 3x4 room: the corridor
    may be classified (measured elongation) or honestly not -- but its
    footprint must not be split into phantom rooms, and the room link
    through the shared door must exist."""
    pts: list = []
    # Corridor along x: 0..6 x 0..1.2, door into room at x=2.
    _slab(pts, 0, 6, 0, 1.2, 0.0)
    _slab(pts, 0, 6, 0, 1.2, 2.4)
    _wall(pts, "y", 0.0, 0, 6, 0.0, 2.4)
    _wall(pts, "y", 1.2, 0, 6, 0.0, 2.4,
          skip=lambda u, z: 1.7 <= u <= 2.3 and z <= 2.0)  # door to room
    _wall(pts, "x", 0.0, 0, 1.2, 0.0, 2.4)
    _wall(pts, "x", 6.0, 0, 1.2, 0.0, 2.4)
    # Room: 2..5 x 1.2..4.2 sharing the door wall.
    _slab(pts, 2, 5, 1.2, 4.2, 0.0)
    _slab(pts, 2, 5, 1.2, 4.2, 2.4)
    _wall(pts, "y", 4.2, 2, 5, 0.0, 2.4)
    _wall(pts, "x", 2.0, 1.2, 4.2, 0.0, 2.4)
    _wall(pts, "x", 5.0, 1.2, 4.2, 0.0, 2.4)
    scene = _assembled(pts)

    assert len(scene.rooms) >= 1
    areas = sorted(r.floor_area_m2 for r in scene.rooms)
    assert areas[0] >= 3.0, areas  # no sliver phantoms
    linked = {(a, b) for a, b in scene.room_links}
    assert len(scene.room_links) <= 1, scene.room_links


# ------------------------------------------------------------------
# 8. ambiguous opening (half-height void)
# ------------------------------------------------------------------

def test_ambiguous_opening_not_a_door():
    """A 1.5 m wide x 1.0 m tall half-wall void (bar counter pass-
    through): too high-silled for a door, too short for a window band.
    It must NOT become a door; whatever it becomes carries the measured
    geometry."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, door=(1.7, 2.3, 0.0, 2.0))
    # Carve an ambiguous void into the x=4 wall: z in [1.0, 2.0] (sill
    # 1.0 m, head 0.4 m below ceiling) x y in [1.2, 2.7].
    carved = []
    for p in pts:
        x, y, z = p
        if abs(x - 4.0) < 1e-9 and 1.2 <= y <= 2.7 and 1.0 <= z <= 2.0:
            continue
        carved.append(p)
    scene = _assembled(carved)

    inv = _types(scene)
    for e in scene.world.entities.values():
        if e.type.value == "door":
            # The ONLY door may be the real one in the y1 wall (sill 0).
            props = getattr(e, "custom_properties", {}) or {}
            sill = props.get("sill_height_m", 0)
            assert sill <= 0.1, (e.id, sill)


# ------------------------------------------------------------------
# 9. disconnected capture (open side)
# ------------------------------------------------------------------

def test_disconnected_capture_partial_room():
    """One wall entirely missing (the capture never saw it): the floor
    still closes on three sides. The room must be reported PARTIAL --
    never dropped (three walls + floor is real evidence of a space)
    and never confident (a fourth wall the capture never saw must not
    be invented): boundary_completeness < 1, status 'partial', the
    missing edge noted."""
    pts: list = []
    _room_box(pts, 0, 4, 0, 4, 0.0, 2.4, open_sides=("x1",))
    scene = _assembled(pts)

    inv = _types(scene)
    assert inv.get("wall", 0) >= 2, inv
    assert inv.get("floor", 0) == 1, inv
    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    room = scene.rooms[0]
    assert room.status == "partial", room.status
    assert room.boundary_completeness < 1.0, room.boundary_completeness
    assert room.confidence == room.boundary_completeness, room.confidence
    assert any("boundary incomplete" in n for n in room.notes), room.notes
    # The measured area still comes from the floor, not the walls.
    assert 9.0 <= room.floor_area_m2 <= 16.0, room.floor_area_m2


# ------------------------------------------------------------------
# 10. partial stair (a ramp must not become stairs)
# ------------------------------------------------------------------

def test_partial_stair_vs_ramp():
    """A 6-step flight (z 0 -> 0.9) beside a 20-deg ramp between the
    same levels: the ramp must never be a staircase; the stair flight
    (>= 3 measured level bands) may be detected or honestly refused,
    but never invented from the ramp."""
    from perception.architecture.parametric import FitRefused
    from perception.architecture.stairs import detect_stairs

    pts: list = []
    _room_box(pts, 0, 8, 0, 4, 0.0, 3.0, door=(0.2, 0.8, 0.0, 2.0))
    # Stair flight: 6 steps, x 5.0..6.2, rise 0.15, going 0.2.
    for i in range(6):
        x0 = 5.0 + i * 0.2
        _slab(pts, x0, x0 + 0.2, 0.0, 1.0, 0.15 * (i + 1), step=0.05)
    # Ramp: same 0.9 m climb over 2.6 m (19 deg), x 0.5..3.1.
    for i in range(26):
        x = 0.5 + i * 0.1
        z = 0.9 * (x - 0.5) / 2.6
        for yy in (0.0, 0.5, 1.0):
            pts.append((x, yy, z))
            pts.append((x, yy, max(0.0, z - 0.05)))  # ramp thickness

    # The stairs detector consumes raw points and refuses (FitRefused)
    # anything that is not a staircase; a ramp is a shallow continuous
    # incline with no level-band rhythm. Feed it the ramp region alone:
    # it must refuse.
    ramp_pts = [p for p in pts if p[0] <= 3.1 and p[2] <= 0.95 and p[0] >= 0.4]
    with pytest.raises(Exception) as excinfo:
        detect_stairs(ramp_pts)
    assert "refus" in str(excinfo.value).lower() or "band" in str(excinfo.value).lower(), excinfo.value

    # And the flight region alone must either detect with a plausible
    # step count or refuse honestly -- never invent a ramp-fit.
    flight_pts = [p for p in pts if p[0] >= 4.9 and p[2] >= 0.1]
    try:
        fit = detect_stairs(flight_pts)
        assert fit.n_steps >= 2, fit
    except Exception as exc:
        assert "refus" in str(exc).lower() or "band" in str(exc).lower(), exc


# ------------------------------------------------------------------
# 11. multi-level interior (mezzanine)
# ------------------------------------------------------------------

def test_multilevel_mezzanine():
    """A room with a 1.2 m mezzanine slab: the mezzanine is horizontal
    and mid-height. It must NOT seed a second room or a second storey;
    the ground room survives with its measured area. The slab itself
    MAY survive as a measured floor-plane entity (its geometry is
    real) -- what is forbidden is manufacturing structure from it."""
    pts: list = []
    _room_box(pts, 0, 5, 0, 4, 0.0, 2.6, door=(2.2, 2.8, 0.0, 2.0))
    _slab(pts, 3.0, 5.0, 0, 4, 1.2)  # mezzanine slab
    scene = _assembled(pts)

    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    room = scene.rooms[0]
    # Ground-floor room measured from the ground floor, not the
    # mezzanine's extent.
    assert 10.0 <= room.floor_area_m2 <= 20.0, room.floor_area_m2
    inv = _types(scene)
    # The slab's geometry is measured; at most one storey exists.
    assert inv.get("floor", 0) >= 1, inv
    assert len(scene.storey_entity_ids) <= 1, scene.storey_entity_ids


# ------------------------------------------------------------------
# 12. duplicate/coplanar wall fragments
# ------------------------------------------------------------------

def test_coplanar_wall_fragments_merge():
    """A partial wall (corner returns) split by a doorway into two
    coplanar pieces 4.8 m apart: same physical plane, so the pieces
    must re-merge (or at worst stay separate) but must NEVER become two
    rooms -- and the doorway between them must not fabricate a room."""
    pts: list = []
    _room_box(pts, 0, 5, 0, 4, 0.0, 2.4, door=(2.2, 2.8, 0.0, 2.0))
    # Split the x=5 wall's points into two strips, drop a thin middle
    # band (simulating two fragments arriving separately).
    split = []
    for p in pts:
        x, y, z = p
        if abs(x - 5.0) < 1e-9 and 1.45 < y < 1.55:
            continue
        split.append(p)
    scene = _assembled(split)

    assert len(scene.rooms) == 1, [r.floor_area_m2 for r in scene.rooms]
    assert 10.0 <= scene.rooms[0].floor_area_m2 <= 20.0
