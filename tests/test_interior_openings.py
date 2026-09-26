"""Interior opening discrimination + first-class Opening entities
(INTERIOR RECONSTRUCTION MISSION, requirement 4/5).

Existing detectors measure two SEPARATE facts about a wall's inlier
coverage:

  classify.detect_wall_opening  -> a floor-reaching gap ("doorway")
  windows.detect_window         -> a sill-bounded gap ("window")

But nothing (a) discriminates the two facts on ONE wall against each
other, (b) detects a GENERIC wall opening (a hole that is neither
door-shaped nor window-shaped -- e.g. a pass-through or a large wall
break), or (c) turns the result into a first-class WorldIR entity
carrying host wall, width/height, position, confidence, provenance.

This suite pins `perception/architecture/openings.py`:

  detect_openings(wall, up, floor_height) -> List[OpeningFit]
      where each OpeningFit.kind is one of
      "door" | "window" | "opening" (generic, measured but
      unclassifiable as door/window)
"""

from __future__ import annotations

import pytest

from perception.architecture.classify import PlaneInput
from perception.architecture.parametric import FitRefused
from perception.architecture.windows import WindowRefused

UP = (0.0, 0.0, 1.0)
FLOOR = 0.0


def _grid(a0, a1, b0, b1, step=0.02):
    na = int(round((a1 - a0) / step))
    nb = int(round((b1 - b0) / step))
    for i in range(na + 1):
        for j in range(nb + 1):
            yield a0 + i * step, b0 + j * step


def _wall(gaps, x_span=4.0, z_span=3.0, step=0.02):
    """Wall on the y=0 plane (normal +y) with rectangular gaps
    (x_lo, x_hi, z_lo, z_hi) removed from the coverage grid."""
    pts = []
    for x, z in _grid(0.0, x_span, 0.0, z_span, step):
        skip = False
        for gx0, gx1, gz0, gz1 in gaps:
            if gx0 <= x <= gx1 and gz0 <= z <= gz1:
                skip = True
                break
        if not skip:
            pts.append((x, 0.0, z))
    n = len(pts)
    centroid = tuple(sum(p[i] for p in pts) / n for i in range(3))
    bmin = tuple(min(p[i] for p in pts) for i in range(3))
    bmax = tuple(max(p[i] for p in pts) for i in range(3))
    return PlaneInput(
        plane_id="wall-y0", normal=(0.0, 1.0, 0.0), centroid=centroid,
        bounds_min=bmin, bounds_max=bmax, inlier_positions=tuple(pts),
    )


class TestOpeningDetection:
    def test_floor_reaching_gap_is_door(self):
        from perception.architecture.openings import detect_openings

        wall = _wall(gaps=((1.0, 1.6, 0.0, 2.0),))
        fits = detect_openings(wall, up=UP, floor_height=FLOOR)
        assert len(fits) == 1
        f = fits[0]
        assert f.kind == "door"
        assert f.width_m == pytest.approx(0.6, abs=0.06)
        assert f.height_m == pytest.approx(2.0, abs=0.06)
        # Sill of a door is the floor (measured, ~0).
        assert f.sill_height_m <= 0.06
        assert 0.0 < f.confidence <= 1.0
        assert f.wall_plane_id == "wall-y0"

    def test_sill_bounded_gap_is_window(self):
        from perception.architecture.openings import detect_openings

        wall = _wall(gaps=((1.4, 2.6, 0.9, 1.9),))
        fits = detect_openings(wall, up=UP, floor_height=FLOOR)
        assert len(fits) == 1
        assert fits[0].kind == "window"
        assert fits[0].sill_height_m == pytest.approx(0.9, abs=0.06)

    def test_generic_opening_measured_not_classified(self):
        from perception.architecture.openings import detect_openings

        # A 1.4 m tall, 1.0 m wide opening with sill 0.1 m: sill is too
        # low for a window (door-shaped bottom) but height 1.4 is too
        # short for a door band (>= ~1.8). Neither class claims it;
        # it is a measured GENERIC opening.
        wall = _wall(gaps=((1.0, 2.0, 0.1, 1.5),))
        fits = detect_openings(wall, up=UP, floor_height=FLOOR)
        assert len(fits) == 1
        f = fits[0]
        assert f.kind == "opening"
        assert f.width_m == pytest.approx(1.0, abs=0.06)
        assert f.height_m == pytest.approx(1.4, abs=0.06)
        assert 0.0 < f.confidence <= 1.0

    def test_two_openings_on_one_wall(self):
        from perception.architecture.openings import detect_openings

        wall = _wall(gaps=((0.5, 1.1, 0.0, 2.0), (2.2, 3.0, 0.9, 1.9)))
        fits = detect_openings(wall, up=UP, floor_height=FLOOR)
        kinds = sorted(f.kind for f in fits)
        assert kinds == ["door", "window"]

    def test_solid_wall_refuses(self):
        from perception.architecture.openings import detect_openings

        wall = _wall(gaps=())
        assert detect_openings(wall, up=UP, floor_height=FLOOR) == []

    def test_opening_without_inliers_refuses(self):
        from perception.architecture.openings import detect_openings

        wall = PlaneInput(
            plane_id="wall-empty", normal=(0.0, 1.0, 0.0),
            centroid=(2.0, 0.0, 1.5), bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(4.0, 0.0, 3.0), inlier_positions=(),
        )
        assert detect_openings(wall, up=UP, floor_height=FLOOR) == []

    def test_determinism(self):
        from perception.architecture.openings import detect_openings

        wall = _wall(gaps=((0.5, 1.1, 0.0, 2.0), (2.2, 3.0, 0.9, 1.9)))
        a = [f.to_dict() for f in detect_openings(wall, up=UP, floor_height=FLOOR)]
        b = [f.to_dict() for f in detect_openings(wall, up=UP, floor_height=FLOOR)]
        assert a == b


# ------------------------------------------------------------------
# First-class Opening entities in WorldIR (mission requirement 4)
# ------------------------------------------------------------------


class TestOpeningEntities:
    def _world(self):
        from world_ir.world_v1 import WorldIR

        return WorldIR(id="w-openings")

    def test_promote_opening_to_entity(self):
        from perception.architecture.openings import (
            OpeningFit,
            promote_opening_to_entity,
        )

        world = self._world()
        # Host wall must exist first (no dangling topology).
        from world_ir.schema_v1 import Entity, EntityType as ET

        world.entities["wall-wall-y0"] = Entity(
            id="wall-wall-y0", type=ET.WALL, confidence=0.9,
        )
        fit = OpeningFit(
            kind="door", wall_plane_id="wall-y0",
            position=(1.3, 0.0, 1.0), width_m=0.6, height_m=2.0,
            sill_height_m=0.0, bounds_min=(1.0, 0.0, 0.0),
            bounds_max=(1.6, 0.0, 2.0), n_points=40, confidence=0.9,
        )
        ent = promote_opening_to_entity(fit, world, "opening-001")
        assert ent.id == "opening-001"
        assert ent.type.value == "door"
        props = ent.custom_properties
        assert props["kind"] == "door"
        assert props["host_wall_id"] == "wall-wall-y0"
        assert props["width_m"] == pytest.approx(0.6)
        assert props["height_m"] == pytest.approx(2.0)
        assert props["sill_height_m"] == pytest.approx(0.0)
        # First-class edges both ways.
        assert any(
            r.kind.value == "part_of" and r.target_id == "wall-wall-y0"
            for r in ent.relationships
        )
        assert any(
            r.kind.value == "contains" and r.target_id == "opening-001"
            for r in world.entities["wall-wall-y0"].relationships
        )
        # Round-trips through serialization.
        from world_ir.world_v1 import WorldIR as _W
        world2 = _W.from_dict(world.to_dict())
        assert world2.entities["opening-001"].custom_properties["kind"] == "door"

    def test_generic_opening_uses_opening_type(self):
        from perception.architecture.openings import (
            OpeningFit,
            promote_opening_to_entity,
        )

        world = self._world()
        from world_ir.schema_v1 import Entity, EntityType as ET

        world.entities["wall-wall-y0"] = Entity(
            id="wall-wall-y0", type=ET.WALL, confidence=0.9,
        )
        fit = OpeningFit(
            kind="opening", wall_plane_id="wall-y0",
            position=(1.5, 0.0, 0.8), width_m=1.0, height_m=1.4,
            sill_height_m=0.1, bounds_min=(1.0, 0.0, 0.1),
            bounds_max=(2.0, 0.0, 1.5), n_points=35, confidence=0.8,
        )
        ent = promote_opening_to_entity(fit, world, "opening-002")
        assert ent.type.value == "opening"

    def test_refuses_without_host_wall(self):
        from perception.architecture.openings import (
            OpeningFit,
            promote_opening_to_entity,
        )

        world = self._world()
        fit = OpeningFit(
            kind="door", wall_plane_id="wall-y0",
            position=(1.3, 0.0, 1.0), width_m=0.6, height_m=2.0,
            sill_height_m=0.0, bounds_min=(1.0, 0.0, 0.0),
            bounds_max=(1.6, 0.0, 2.0), n_points=40, confidence=0.9,
        )
        with pytest.raises(Exception):
            promote_opening_to_entity(fit, world, "opening-003")
