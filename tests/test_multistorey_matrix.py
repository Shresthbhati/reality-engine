"""Multi-storey segmentation matrix: supported storeys are preserved, unsupported ones stay unknown, nothing forces
a storey count and nothing fabricates a floor.

Every fixture is built from raw points with a KNOWN storey structure and pushed through the REAL detect -> refine ->
assemble chain. Assertions are on measured support (plane elevations, plan areas, entity types), never on
fixture-specific thresholds inside the engine. Complements tests/test_multistorey_plane_support.py (canonical
two-storey, single storey, adjacent rooms, tilted, noisy, incomplete upper floor, collapsed, determinism, fragments).
"""

from __future__ import annotations

import math

import pytest

from tests.test_multistorey_plane_support import UP, _Cloud, _detect_refine, _horizontal_planes

STEP = 0.12          # coarser than the sibling suite: same geometry, ~30% fewer points


class _MultiCam(_Cloud):
    """A cloud observed from one camera per storey (a single low camera cannot see upper floors)."""

    def __init__(self, cameras, ranges=None):
        super().__init__(camera=cameras[0])
        self.cameras = cameras
        self.ranges = ranges          # per camera: the (z_low, z_high) of the storey it stands in

    def _observer(self, cams, z):
        if self.ranges:
            inside = [c for c, (lo, hi) in zip(cams, self.ranges) if lo - 1e-6 <= z <= hi + 1e-6]
            if inside:
                return inside[0]
        return min(cams, key=lambda c: abs(c.position[2] - z))

    def result(self):
        """Each point is observed by the camera on ITS storey (nearest camera elevation): the visibility a real
        multi-level capture has, which is what decides whether a horizontal sheet is a floor or a ceiling."""
        from provenance import Uncertainty
        from reconstruction.backend.interface import (
            ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult,
        )

        cams = [ReconstructedCameraPose(evidence_id=f"ev-{i + 1}", position=c, rotation=(1.0, 0.0, 0.0, 0.0),
                                        uncertainty=Uncertainty(confidence=0.9))
                for i, c in enumerate(self.cameras)]
        pts = [ReconstructedPoint(position=p, track_id=f"pt-{i:05d}",
                                  source_evidence_ids=[self._observer(cams, p[2]).evidence_id],
                                  uncertainty=Uncertainty(confidence=0.95))
               for i, p in enumerate(self.pts)]
        return ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")


def _storey(c, x0, x1, y0, y1, z0, height, *, floor=True, ceiling=True):
    z1 = z0 + height
    if floor:
        c.grid(x0, x1, y0, y1, z0, STEP)
    if ceiling:
        c.grid(x0, x1, y0, y1, z1, STEP)
    c.wall_x(x0, y0, y1, z0, z1, STEP)
    c.wall_x(x1, y0, y1, z0, z1, STEP)
    c.wall_y(y0, x0, x1, z0, z1, STEP)
    c.wall_y(y1, x0, x1, z0, z1, STEP)


def _tower(heights, gap=0.3, plan=(0, 4, 0, 4)):
    """Stacked closed storeys; returns (cloud-with-cameras, floor elevations)."""
    x0, x1, y0, y1 = plan
    cams, z, floors, ranges = [], 0.0, [], []
    c = _MultiCam([(0, 0, 0)])
    for h in heights:
        floors.append(z)
        cams.append(((x0 + x1) / 2, (y0 + y1) / 2, z + h / 2))
        ranges.append((z, z + h))
        _storey(c, x0, x1, y0, y1, z, h)
        z += h + gap
    c.cameras, c.ranges = cams, ranges
    return c, floors


def _assemble(c):
    from perception.architecture.scene import assemble_interior_scene

    return assemble_interior_scene(c.result(), up=UP, seed=42)


def _types(scene):
    return [e.type.value for e in scene.world.entities.values()]


def _no_tilted_horizontal(c):
    """No near-horizontal candidate may be a plane that mixes storeys: every horizontal plane's inliers sit within
    the tolerance band of ONE elevation, and no plane straddles two floors."""
    refined, positions = _detect_refine(c.result())
    for pl, z in _horizontal_planes(refined, positions):
        zs = [positions[pid][2] for pid in pl.inlier_ids]
        assert max(zs) - min(zs) <= 0.06, (pl.plane_id, z, max(zs) - min(zs))
    return [z for _, z in _horizontal_planes(refined, positions)]


# 1-2 (one / two storey) are covered by the sibling suite; 3: three storeys ------------------------------------

def test_three_storeys_are_recovered_as_three_without_tilted_merges():
    c, floors = _tower([2.4, 2.4, 2.4])
    zs = _no_tilted_horizontal(c)
    for f in floors:
        assert any(abs(z - f) <= 0.03 for z in zs), (f, zs)
    scene = _assemble(c)
    types = _types(scene)
    assert types.count("floor") == 3 and types.count("ceiling") == 3, sorted(types)
    assert len(scene.rooms) == 3 and len(scene.storey_entity_ids) == 3, [r.floor_area_m2 for r in scene.rooms]
    assert all(12.0 <= r.floor_area_m2 <= 17.0 for r in scene.rooms)


# 4: different floor heights ---------------------------------------------------------------------------------

@pytest.mark.parametrize("heights", [(2.4, 3.6), (3.6, 2.4, 3.0), (2.2, 2.2)])
def test_storeys_of_different_heights_keep_their_own_elevations(heights):
    c, floors = _tower(list(heights), gap=0.35)
    zs = _no_tilted_horizontal(c)
    for f in floors:
        assert any(abs(z - f) <= 0.03 for z in zs), (f, zs)
    scene = _assemble(c)
    assert len(scene.storey_entity_ids) == len(heights), scene.storey_entity_ids
    assert _types(scene).count("floor") == len(heights)


# 5: partial upper-floor visibility --------------------------------------------------------------------------

def test_partial_upper_storey_is_kept_partial_and_never_completed():
    c, _ = _tower([2.4])
    for i in range(int(round(2.0 / STEP)) + 1):                        # upper floor: only the left half observed
        for j in range(int(round(4.0 / STEP)) + 1):
            c.add(i * STEP, j * STEP, 2.7)
    c.cameras = [(2, 2, 1.2)]
    scene = _assemble(c)
    assert all(r.floor_area_m2 <= 16.5 for r in scene.rooms)          # nothing measures beyond the real plan
    assert len(scene.storey_entity_ids) <= 2


# 6: sloped surfaces -----------------------------------------------------------------------------------------

def test_a_ramp_between_two_storeys_is_not_a_floor():
    c, floors = _tower([2.4, 2.4])
    for i in range(31):                                                # ramp from floor 1 (z=0) up toward floor 2
        for j in range(31):
            x, y = 0.4 + i * STEP * 0.9, 0.4 + j * STEP * 0.9
            c.add(x, y, 0.2 + (x - 0.4) * math.tan(math.radians(20.0)))
    scene = _assemble(c)
    assert len(scene.storey_entity_ids) == 2, scene.storey_entity_ids
    assert _types(scene).count("floor") == 2


# 7: noisy surfaces ------------------------------------------------------------------------------------------

def test_noisy_storeys_do_not_grow_phantom_storeys():
    c, floors = _tower([2.4, 2.4])
    k = 0
    noisy = _MultiCam(c.cameras, c.ranges)
    for x, y, z in c.pts:
        k += 1
        jitter = (math.sin(12.9898 * k) * 43758.5453 % 1.0 - 0.5) * 0.01
        noisy.add(x, y, z + jitter)
    scene = _assemble(noisy)
    assert len(scene.storey_entity_ids) == 2 and _types(scene).count("floor") == 2


# 8: missing floor evidence -----------------------------------------------------------------------------------

def test_a_storey_with_no_floor_evidence_gets_no_fabricated_floor():
    c = _MultiCam([(2, 2, 1.2)])
    _storey(c, 0, 4, 0, 4, 0.0, 2.4, floor=False)                      # walls + ceiling, floor never observed
    scene = _assemble(c)
    types = _types(scene)
    assert types.count("floor") == 0, sorted(types)
    assert len(scene.storey_entity_ids) <= 1
    for room in scene.rooms:                                           # an enclosure is allowed only as PARTIAL
        assert room.status != "detected", (room.room_id, room.status)


def test_only_the_observed_storeys_of_a_tower_are_reported():
    c, _ = _tower([2.4, 2.4, 2.4])
    upper = _MultiCam(c.cameras[:2], c.ranges[:2])                                   # rebuild without any storey-3 evidence
    for p in c.pts:
        if p[2] < 5.0:
            upper.add(*p)
    scene = _assemble(upper)
    assert len(scene.storey_entity_ids) == 2 and _types(scene).count("floor") == 2


# 9-10: disconnected / visually identical but spatially separate horizontal planes ----------------------------

def test_disjoint_slabs_at_the_same_elevation_stay_separate_rooms_of_one_storey():
    c = _MultiCam([(2, 2, 1.2), (12, 2, 1.2)])
    _storey(c, 0, 4, 0, 4, 0.0, 2.4)
    _storey(c, 10, 14, 0, 4, 0.0, 2.4)                                 # identical, 6 m away, same elevation
    scene = _assemble(c)
    assert len(scene.rooms) == 2 and len(scene.storey_entity_ids) == 1, (
        [r.floor_area_m2 for r in scene.rooms], scene.storey_entity_ids)
    assert all(12.0 <= r.floor_area_m2 <= 17.0 for r in scene.rooms)   # never one 4x14 slab


def test_identical_footprints_at_different_elevations_are_never_one_plane():
    c, floors = _tower([2.4, 2.4])
    zs = _no_tilted_horizontal(c)
    assert sum(1 for z in zs if abs(z - floors[0]) <= 0.03) == 1
    assert sum(1 for z in zs if abs(z - floors[1]) <= 0.03) == 1


def test_offset_upper_slab_over_a_lower_room_is_its_own_storey_not_a_tilted_bridge():
    c, _ = _tower([2.4])
    for i in range(int(round(4.0 / STEP)) + 1):                        # a slab above, shifted 2 m in plan
        for j in range(int(round(4.0 / STEP)) + 1):
            c.add(2.0 + i * STEP, j * STEP, 2.7)
    c.cameras = [(2, 2, 1.2)]
    zs = _no_tilted_horizontal(c)
    assert any(abs(z - 2.7) <= 0.03 for z in zs)
