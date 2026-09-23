"""Red-first tests for the last two declared-but-unwired architectural
classes (P7-03 breadth): WINDOW and ROOF.

Follows the established measured-detector pattern exactly (stairs
2026-09-21, columns/beams 2026-09-22): a fit over REAL point/plane
structure with honest *FitRefused refusals carrying the measured fact,
wired into the canonical build_component_observations -> union-find
resolution -> WorldIR promotion path. No labeling of arbitrary geometry
without evidence; no scene-tuned thresholds.

Window evidence: a window is an EDGE-BOUNDED OPENING in a wall plane's
own inlier coverage that does NOT reach the floor (an opening that does
is measured like a door and refused as a window). What is measured: the
gap's lateral width, vertical extent, sill height above the floor, and
the bounding ring's coverage (edge-boundedness).

Roof evidence: the TOPMOST horizontal-ish plane of a supplied exterior
plane set (same relative-position logic classify.py already uses for
floor=lowest / ceiling=highest), with measured lateral extents. A plane
without enough extent is a cap/skylight, not a roof; a tilted plane is
measured and refused (pitched roofs need multi-plane aggregation --
documented limitation, not a silent guess).
"""

import math

import pytest

from perception.architecture.classify import PlaneInput
from perception.architecture.parametric import FitRefused


def _wall_inliers_with_window(
    x_lo, x_hi, z_lo, z_hi, x_span=4.0, z_span=3.0, step=0.04
):
    """Wall points on the y=0 plane (normal +y), dense grid, with the
    rectangle [x_lo,x_hi] x [z_lo,z_hi] EMPTY (the opening)."""
    pts = []
    nx = int(round(x_span / step))
    nz = int(round(z_span / step))
    for i in range(nx + 1):
        for j in range(nz + 1):
            x = i * step
            z = j * step
            if x_lo <= x <= x_hi and z_lo <= z <= z_hi:
                continue
            pts.append((x, 0.0, z))
    return pts


def _wall_plane(inliers, x_span=4.0, z_span=3.0):
    return PlaneInput(
        plane_id="wall-y0",
        normal=(0.0, 1.0, 0.0),
        centroid=(x_span / 2.0, 0.0, z_span / 2.0),
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(x_span, 0.0, z_span),
        inlier_positions=tuple(inliers),
    )


UP = (0.0, 0.0, 1.0)
FLOOR = 0.0



class TestWindowDetector:
    """detect_window: measured edge-bounded opening in a wall plane."""

    def test_measures_real_window_opening(self):
        from perception.architecture.windows import detect_window

        # A 1.2 x 1.0 m opening with sill at 0.9 m in a 4x3 wall.
        inliers = _wall_inliers_with_window(1.4, 2.6, 0.9, 1.9)
        fit = detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        assert fit.kind == "window"
        assert fit.width_m == pytest.approx(1.2, abs=0.12)
        assert fit.height_m == pytest.approx(1.0, abs=0.12)
        # Measured sill above the floor.
        assert fit.sill_height_m == pytest.approx(0.9, abs=0.12)
        assert fit.n_points >= 0
        assert 0.0 < fit.confidence <= 1.0
        # Geometry bounds bracket the measured opening.
        assert fit.bounds_min[2] >= 0.0
        assert fit.bounds_max[2] > fit.bounds_min[2]
        # Evidence: the fit's own support ids are traceable.
        assert all(e.startswith("win-") for e in fit.evidence_ids)

    def test_floorside_opening_is_not_a_window(self):
        from perception.architecture.windows import detect_window

        # Opening reaching the floor (0.0 sill): measured as a door-class
        # gap, refused as a window with the measured sill in the message.
        inliers = _wall_inliers_with_window(1.4, 2.6, 0.0, 2.0)
        with pytest.raises(FitRefused) as exc:
            detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        assert "sill" in str(exc.value).lower()

    def test_solid_wall_refused(self):
        from perception.architecture.windows import detect_window

        inliers = _wall_inliers_with_window(2.0, 2.0, 1.0, 1.0)  # no gap
        with pytest.raises(FitRefused):
            detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)

    def test_oversized_gap_refused(self):
        from perception.architecture.windows import detect_window

        # 3.2 m wide opening: wider than any window band.
        inliers = _wall_inliers_with_window(0.4, 3.6, 0.8, 2.9)
        with pytest.raises(FitRefused):
            detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)

    def test_tiny_gap_refused(self):
        from perception.architecture.windows import detect_window

        inliers = _wall_inliers_with_window(1.9, 2.1, 1.2, 1.4)
        with pytest.raises(FitRefused):
            detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)

    def test_wall_without_inliers_refused(self):
        from perception.architecture.windows import detect_window

        with pytest.raises(FitRefused):
            detect_window(_wall_plane(()), up=UP, floor_height=FLOOR)

    def test_deterministic(self):
        from perception.architecture.windows import detect_window

        inliers = _wall_inliers_with_window(1.4, 2.6, 0.9, 1.9)
        a = detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        b = detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        assert a.to_dict() == b.to_dict()



class TestRoofDetector:
    """detect_roof: topmost horizontal-ish plane of an exterior set."""

    @staticmethod
    def _planes():
        wall_a = PlaneInput(
            plane_id="wall-a", normal=(1.0, 0.0, 0.0),
            centroid=(0.0, 2.0, 1.5), bounds_min=(-3.0, 0.0, 0.0),
            bounds_max=(0.0, 4.0, 3.0),
        )
        wall_b = PlaneInput(
            plane_id="wall-b", normal=(-1.0, 0.0, 0.0),
            centroid=(6.0, 2.0, 1.5), bounds_min=(6.0, 0.0, 0.0),
            bounds_max=(9.0, 4.0, 3.0),
        )
        roof = PlaneInput(
            plane_id="roof-top", normal=(0.0, 0.0, 1.0),
            centroid=(3.0, 2.0, 3.2), bounds_min=(0.0, 0.0, 3.2),
            bounds_max=(6.0, 4.0, 3.2),
        )
        return [wall_a, wall_b, roof]

    def test_topmost_horizontal_plane_is_roof(self):
        from perception.architecture.roofs import detect_roof

        fit = detect_roof(self._planes(), up=UP)
        assert fit.kind == "roof"
        assert fit.plane_id == "roof-top"
        # Measured lateral extents from the plane's own bounds.
        assert fit.width_m == pytest.approx(6.0)
        assert fit.depth_m == pytest.approx(4.0)
        assert fit.height_m == pytest.approx(3.2)
        assert 0.0 < fit.confidence <= 1.0
        assert fit.n_planes == 1

    def test_lowest_plane_is_floor_not_roof(self):
        from perception.architecture.roofs import detect_roof

        floor = PlaneInput(
            plane_id="floor", normal=(0.0, 0.0, 1.0),
            centroid=(3.0, 2.0, 0.0), bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(6.0, 4.0, 0.0),
        )
        planes = self._planes() + [floor]
        fit = detect_roof(planes, up=UP)
        assert fit.plane_id == "roof-top"

    def test_tiny_horizontal_plane_refused(self):
        from perception.architecture.roofs import detect_roof

        skylight = PlaneInput(
            plane_id="skylight", normal=(0.0, 0.0, 1.0),
            centroid=(3.0, 2.0, 3.2), bounds_min=(2.8, 1.8, 3.2),
            bounds_max=(3.2, 2.2, 3.2),
        )
        with pytest.raises(FitRefused) as exc:
            detect_roof([skylight], up=UP)
        assert "extent" in str(exc.value).lower()

    def test_tilted_plane_refused_not_guessed(self):
        from perception.architecture.roofs import detect_roof

        # 45-degree pitched plane: real roof geometry, but this detector
        # measures horizontal slabs only (documented limitation).
        n = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4))
        pitched = PlaneInput(
            plane_id="pitched", normal=n,
            centroid=(3.0, 2.0, 3.2), bounds_min=(0.0, 0.0, 2.0),
            bounds_max=(6.0, 4.0, 4.0),
        )
        with pytest.raises(FitRefused) as exc:
            detect_roof([pitched], up=UP)
        assert "horizontal" in str(exc.value).lower()

    def test_no_planes_refused(self):
        from perception.architecture.roofs import detect_roof

        with pytest.raises(FitRefused):
            detect_roof([], up=UP)

    def test_deterministic(self):
        from perception.architecture.roofs import detect_roof

        a = detect_roof(self._planes(), up=UP)
        b = detect_roof(self._planes(), up=UP)
        assert a.to_dict() == b.to_dict()



class TestCanonicalPathIntegration:
    """Both fits compose with the CANONICAL observation -> resolution ->
    promotion path (no parallel pipeline), and refused points never
    become entities."""

    def test_window_observation_to_entity(self):
        from perception.architecture.components import (
            build_component_observations,
            resolve_components,
        )
        from perception.architecture.promotion import promote_component_to_entity
        from perception.architecture.windows import detect_window
        from world_ir import WorldIR

        inliers = _wall_inliers_with_window(1.4, 2.6, 0.9, 1.9)
        fit = detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        obs = build_component_observations(
            [(("seg-w1",), fit, ("ev-1", "ev-2"))],
            up=UP,
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.arch_class == "window"
        assert o.accepted
        assert o.evidence_ids == ("ev-1", "ev-2")

        candidates = resolve_components(obs)
        assert len(candidates) == 1
        world = WorldIR(id="w-window-test")
        entity = promote_component_to_entity(candidates[0], world, "window-1")
        assert entity.type.value == "window"
        props = entity.custom_properties
        assert props["fit_kind"] == "window"
        assert props["evidence_ids"] == ["ev-1", "ev-2"]
        assert props["width_m"] == pytest.approx(fit.width_m)

    def test_roof_observation_to_entity(self):
        from perception.architecture.components import (
            build_component_observations,
            resolve_components,
        )
        from perception.architecture.promotion import promote_component_to_entity
        from perception.architecture.roofs import detect_roof
        from world_ir import WorldIR

        fit = detect_roof(TestRoofDetector._planes(), up=UP)
        obs = build_component_observations(
            [(("seg-r1",), fit, ("ev-r1",))],
            up=UP,
        )
        assert obs[0].arch_class == "roof"
        assert obs[0].accepted
        candidates = resolve_components(obs)
        assert len(candidates) == 1
        world = WorldIR(id="w-roof-test")
        entity = promote_component_to_entity(candidates[0], world, "roof-1")
        assert entity.type.value == "roof"
        props = entity.custom_properties
        assert props["fit_kind"] == "roof"
        assert props["width_m"] == pytest.approx(fit.width_m)
        assert props["evidence_ids"] == ["ev-r1"]

    def test_refused_points_never_become_entities(self):
        from perception.architecture.components import (
            build_component_observations,
            resolve_components,
        )
        from perception.architecture.windows import detect_window

        # A floor-reaching opening refuses; nothing enters the pipeline.
        inliers = _wall_inliers_with_window(1.4, 2.6, 0.0, 2.0)
        with pytest.raises(FitRefused):
            detect_window(_wall_plane(inliers), up=UP, floor_height=FLOOR)
        # No fit -> no observation -> no candidate -> no entity.
        obs = build_component_observations([], up=UP)
        assert resolve_components(obs) == []

