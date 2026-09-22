"""Measured stair detection (P7-03 architectural-perception expansion).

Detector contract (mirrors parametric.py's honesty rules):

  - StaircaseFit carries ONLY measured quantities: per-step elevation
    rhythm (rise), planimetric rhythm (going), span, n_steps, residual
    of the piecewise-flat fit, and the point count that supports it.
  - StairRefused is raised (never best-effort geometry) when the
    measured structure does not demonstrate stairs: fewer than two
    distinct level bands, irregular rhythm, steps that are too tall to
    be walkable risers, or too few points.
  - Deterministic: pure float math, no RNG, no clocks.
"""

from __future__ import annotations

import pytest

from perception.architecture.parametric import FitRefused  # noqa: F401
from perception.architecture.stairs import StaircaseFit, detect_stairs


def _tread_points(n_steps, rise=0.16, going=0.28, width=1.2, per_side=8,
                  jitter=0.0):
    """Synthetic staircase: flat treads climbing +z along +x.

    Each tread is a rectangular patch of points at a distinct height.
    Deterministic grid, no RNG (jitter is a fixed per-index offset).
    """
    pts = []
    for step in range(n_steps + 1):
        z = step * rise
        for i in range(per_side):
            for j in range(per_side):
                x = step * going + (i / (per_side - 1)) * going
                y = -(width / 2) + (j / (per_side - 1)) * width
                if jitter:
                    x += 0.001 * ((i + j) % 3)
                pts.append((x, y, z))
    return pts


class TestMeasuredDetection:
    def test_detects_multi_step_staircase(self):
        pts = _tread_points(5)
        fit = detect_stairs(pts)
        assert isinstance(fit, StaircaseFit)
        assert fit.n_steps == 5
        assert fit.rise_m == pytest.approx(0.16, abs=0.02)
        assert fit.going_m == pytest.approx(0.28, abs=0.03)
        # Span includes the top tread's own depth: treads run from
        # step*going to (step+1)*going, so the planimetric extent is
        # (n_steps + 1) * going, not n_steps * going.
        assert fit.span_m == pytest.approx((5 + 1) * 0.28, abs=0.05)
        assert fit.n_points == len(pts)

    def test_two_treads_is_the_minimum_staircase(self):
        fit = detect_stairs(_tread_points(2))
        assert fit.n_steps == 2

    def test_rhythm_is_measured_not_assumed(self):
        pts = _tread_points(4, rise=0.18, going=0.30)
        fit = detect_stairs(pts)
        assert fit.rise_m == pytest.approx(0.18, abs=0.02)
        assert fit.going_m == pytest.approx(0.30, abs=0.03)

    def test_fit_is_dict_serializable(self):
        d = detect_stairs(_tread_points(3)).to_dict()
        assert d["n_steps"] == 3
        assert d["kind"] == "stairs"

    def test_confidence_within_unit_interval(self):
        fit = detect_stairs(_tread_points(6))
        assert 0.0 < fit.confidence <= 1.0


class TestHonestRefusals:
    def test_flat_ground_refuses(self):
        pts = [(0.1 * i, 0.1 * j, 0.0) for i in range(10) for j in range(10)]
        with pytest.raises(FitRefused):
            detect_stairs(pts)

    def test_single_step_is_not_a_staircase(self):
        with pytest.raises(FitRefused):
            detect_stairs(_tread_points(1))

    def test_ramp_is_not_a_staircase(self):
        pts = []
        for i in range(40):
            for j in range(8):
                x = 0.05 * i
                y = -0.6 + 0.1 * j
                z = 0.03 * i  # smooth incline: no level bands
                pts.append((x, y, z))
        with pytest.raises(FitRefused):
            detect_stairs(pts)

    def test_irregular_rhythm_refuses(self):
        # Level bands at 0.00, 0.10, 0.50, 0.90: real height gaps but
        # not a consistent riser rhythm.
        pts = []
        for step, z in enumerate((0.0, 0.10, 0.50, 0.90)):
            for i in range(6):
                for j in range(6):
                    pts.append((0.3 * step + 0.02 * i,
                                -0.3 + 0.05 * j, z))
        with pytest.raises(FitRefused):
            detect_stairs(pts)

    def test_unclimbable_riser_height_refuses(self):
        # 1.0 m "risers": level rhythm, but no human staircase.
        pts = _tread_points(3, rise=1.0)
        with pytest.raises(FitRefused):
            detect_stairs(pts)

    def test_too_few_points_refuse(self):
        with pytest.raises(FitRefused):
            detect_stairs([(0.0, 0.0, 0.0), (0.3, 0.0, 0.16),
                           (0.6, 0.0, 0.32), (0.9, 0.0, 0.48)])


class TestComponentPipelineIntegration:
    """The detector composes with the CANONICAL observation ->
    resolution -> promotion path (no parallel stair pipeline)."""

    def test_observation_resolution_promotion_end_to_end(self):
        from perception.architecture.components import (
            build_component_observations,
            resolve_components,
        )
        from perception.architecture.promotion import (
            promote_component_to_entity,
        )
        from world_ir import WorldIR

        fit = detect_stairs(_tread_points(4))
        obs = build_component_observations(
            [(("seg-1",), fit, ("ev-1", "ev-2"))],
            up=(0.0, 0.0, 1.0),
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.arch_class == "stairs"
        assert o.accepted
        assert o.evidence_ids == ("ev-1", "ev-2")
        assert o.position == fit.position

        candidates = resolve_components(obs)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.arch_class == "stairs"
        assert cand.effective_confidence() == pytest.approx(
            fit.confidence
        )

        world = WorldIR(id="w-stair-test")
        entity = promote_component_to_entity(
            cand, world, "stairs-1"
        )
        assert world.entities["stairs-1"] is entity
        assert entity.confidence == pytest.approx(fit.confidence)
        props = entity.custom_properties
        assert props["fit_kind"] == "stairs"
        assert props["n_steps"] == 4
        assert props["evidence_ids"] == ["ev-1", "ev-2"]
        assert props["rise_m"] == pytest.approx(fit.rise_m)
