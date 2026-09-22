"""Measured class detectors for columns and beams (P1 perception
breadth, following the PR #86 stair-detector pattern).

The gap this closes: the registry DECLARED `column` (and
`cornice -> beam`) but no detector produced class-level evidence for
them. `parametric.fit_cylinder` is a generic geometry fit: it accepts
ANY cylindrical shell -- a tilted pipe, a horizontal silo -- and the
column-ness decision was left to a downstream gate that only rejects
the most tilted axes. A column is a VERTICAL structural member; that
measured fact belongs in the detection contract, refused honestly
(ColumnRefused) at measurement time, exactly as StairRefused refuses
non-rhythmic points.

The beam detector mirrors this: a beam is a LINEAR PRISMATIC member
-- elongated along ONE horizontal axis, compact in the other two,
with a measured cross-section. A flat wall patch is elongated in TWO
directions; a blob in none; a 45-degree ramp is linear but not a
horizontal structural member. None is a beam; the fit refuses
(BeamRefused) instead of producing best-effort geometry.

Discipline carried from stairs.py/parametric.py:
  - every quantity is MEASURED from the supporting points
  - refusals are facts (a *Refused exception), never degraded output
  - thresholds are module-level documented constants (wide windows,
    not scene-tuned), deterministic; no RNG
  - confidence is a documented monotone function of measured fit
    quality
"""

from __future__ import annotations

import math

import pytest

from perception.architecture.beams import BeamFit, BeamRefused, detect_beam
from perception.architecture.columns import (
    ColumnFit,
    ColumnRefused,
    detect_column,
)

UP = (0.0, 0.0, 1.0)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cylinder_points(axis, origin, radius, ts, angles):
    """Deterministic points on a cylinder surface (same construction
    as tests/test_parametric_fits.py)."""
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _norm(_cross(axis, helper))
    v = _cross(axis, u)
    pts = []
    for t in ts:
        for ang in angles:
            p = [origin[i] + axis[i] * t
                 + (u[i] * math.cos(ang) + v[i] * math.sin(ang)) * radius
                 for i in range(3)]
            pts.append(tuple(p))
    return pts


def _box_points(axis, origin, ts, us, ws, height_axis=UP):
    """Deterministic points on a rectangular prism's long faces:
    axis = long direction; us = offsets along height_axis; ws =
    offsets along the remaining lateral axis."""
    a = _norm(axis)
    u = _norm(height_axis)
    w = _norm(_cross(a, u))
    u_face = max(abs(v) for v in us)   # half height
    w_face = max(abs(v) for v in ws)   # half width
    pts = []
    for t in ts:
        # Side faces (w = +/- half width), spanning the height.
        for s in us:
            for wf in (+w_face, -w_face):
                p = [origin[i] + a[i] * t + u[i] * s + w[i] * wf
                     for i in range(3)]
                pts.append(tuple(p))
        # Top/bottom faces (u = +/- half height), spanning the width.
        for r in ws:
            for uf in (+u_face, -u_face):
                p = [origin[i] + a[i] * t + u[i] * uf + w[i] * r
                     for i in range(3)]
                pts.append(tuple(p))
    return pts


class TestColumnDetector:
    def test_vertical_column_known_answer(self):
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, -0.75, 0.0, 0.75, 1.5, 2.25],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        fit = detect_column(pts)
        assert isinstance(fit, ColumnFit)
        assert fit.radius_m == pytest.approx(0.3, abs=1e-6)
        assert _dot(fit.axis, UP) >= fit.min_axis_up_dot
        assert fit.height_m == pytest.approx(3.75, abs=1e-6)
        assert fit.n_points == len(pts)

    def test_carries_cylinder_fit_evidence(self):
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (2.0, 1.0, 0.0), 0.25,
            ts=[0.0, 1.0, 2.0], angles=[i * math.pi / 6 for i in range(10)],
        )
        fit = detect_column(pts)
        assert fit.cylinder is not None
        assert fit.cylinder.radius_m == pytest.approx(0.25, abs=1e-6)
        d = fit.to_dict()
        assert d["kind"] == "column"
        assert d["n_points"] == len(pts)

    def test_tilted_pipe_is_refused_not_classified(self):
        # 45-degree tilt: a real cylinder, but not a column. The
        # generic fit accepts it; the CLASS detector must refuse.
        tilt = _norm((1.0, 0.0, 1.0))
        pts = _cylinder_points(
            tilt, (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, -0.75, 0.0, 0.75, 1.5, 2.25],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        with pytest.raises(ColumnRefused):
            detect_column(pts)

    def test_horizontal_silo_is_refused(self):
        pts = _cylinder_points(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.8,
            ts=[-2.0, -1.0, 0.0, 1.0, 2.0],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        with pytest.raises(ColumnRefused) as exc:
            detect_column(pts)
        assert "vertical" in str(exc.value).lower()

    def test_flat_wall_patch_is_refused(self):
        pts = [(0.05 * i, 0.05 * j, 0.0)
               for i in range(12) for j in range(12)]
        with pytest.raises(ColumnRefused):
            detect_column(pts)

    def test_refusal_message_names_the_measured_fact(self):
        tilt = _norm((1.0, 0.0, 1.0))
        pts = _cylinder_points(
            tilt, (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, 0.0, 1.5], angles=[i * math.pi / 6 for i in range(10)],
        )
        with pytest.raises(ColumnRefused) as exc:
            detect_column(pts)
        # The refusal must carry the measured up-dot, not a bare no.
        assert "up-dot" in str(exc.value)

    def test_insufficient_points_refused(self):
        # 6 points < the generic fit's 8-point minimum: refusal comes
        # from the geometry layer, wrapped as ColumnRefused.
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[0.0], angles=[i * math.pi / 6 for i in range(6)],
        )
        with pytest.raises(ColumnRefused):
            detect_column(pts)

    def test_deterministic(self):
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.0, 0.0, 1.0, 2.0],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        f1 = detect_column(pts)
        f2 = detect_column(pts)
        assert f1.to_dict() == f2.to_dict()


class TestBeamDetector:
    def test_horizontal_beam_known_answer(self):
        # ~3 m long, 0.3 m tall, 0.2 m wide beam along x at z=3.
        ts = [i * 0.25 for i in range(13)]
        pts = _box_points(
            (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), ts,
            us=[s * 0.075 for s in (-2, -1, 1, 2)],
            ws=[s * 0.05 for s in (-2, 2)],
        )
        fit = detect_beam(pts)
        assert isinstance(fit, BeamFit)
        assert fit.length_m == pytest.approx(3.0, abs=0.05)
        assert fit.height_m == pytest.approx(0.3, abs=0.05)
        assert fit.width_m == pytest.approx(0.2, abs=0.05)
        assert abs(_dot(fit.axis, UP)) < 0.2  # horizontal member
        assert fit.n_points == len(pts)

    def test_carries_cross_section_evidence(self):
        ts = [i * 0.5 for i in range(7)]
        pts = _box_points(
            (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), ts,
            us=[s * 0.075 for s in (-2, -1, 1, 2)],
            ws=[s * 0.05 for s in (-2, 2)],
        )
        fit = detect_beam(pts)
        d = fit.to_dict()
        assert d["kind"] == "beam"
        assert d["aspect_ratio"] > 3.0  # measured, elongated member
        assert d["rms_residual_m"] < 0.05

    def test_vertical_post_is_refused_not_beam(self):
        # A vertical member may be many things; a beam is not one.
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, -0.75, 0.0, 0.75, 1.5, 2.25],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        with pytest.raises(BeamRefused) as exc:
            detect_beam(pts)
        assert "horizontal" in str(exc.value).lower()

    def test_wall_patch_two_direction_elongation_refused(self):
        # Elongated in TWO directions: a wall panel, not a beam.
        pts = [(0.25 * i, 0.0, 3.0 + 0.05 * j) for i in range(14)
               for j in range(8)]
        with pytest.raises(BeamRefused):
            detect_beam(pts)

    def test_blob_no_elongation_refused(self):
        # Roughly isotropic blob: nothing linear to measure.
        pts = []
        for i in range(200):
            a = i * 2.399963  # golden-angle spiral, deterministic
            r = 0.4 * math.sqrt(i / 200.0)
            pts.append((r * math.cos(a), r * math.sin(a),
                        0.02 * math.sin(5 * a)))
        with pytest.raises(BeamRefused):
            detect_beam(pts)

    def test_slanted_ramp_refused(self):
        # 45-degree slope: linear and elongated, but not a horizontal
        # structural member.
        ts = [i * 0.25 for i in range(13)]
        pts = _box_points(
            (1.0, 0.0, 1.0), (0.0, 0.0, 3.0), ts,
            us=[s * 0.075 for s in (-2, -1, 1, 2)],
            ws=[s * 0.05 for s in (-2, 2)],
        )
        with pytest.raises(BeamRefused) as exc:
            detect_beam(pts)
        assert "horizontal" in str(exc.value).lower()

    def test_insufficient_points_refused(self):
        with pytest.raises(BeamRefused):
            detect_beam([(0.0, 0.0, 3.0), (1.0, 0.0, 3.0),
                         (2.0, 0.0, 3.0)])

    def test_deterministic(self):
        ts = [i * 0.25 for i in range(13)]
        pts = _box_points(
            (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), ts,
            us=[s * 0.075 for s in (-2, -1, 1, 2)],
            ws=[s * 0.05 for s in (-2, 2)],
        )
        f1 = detect_beam(pts)
        f2 = detect_beam(pts)
        assert f1.to_dict() == f2.to_dict()


class TestComponentPipelineIntegration:
    """Both detectors compose with the CANONICAL observation ->
    resolution -> promotion path (no parallel pipeline; the same
    union-find entity resolution and WorldIR promotion as stairs)."""

    def _promote(self, fit, evidence_ids, entity_id):
        from perception.architecture.components import (
            build_component_observations,
            resolve_components,
        )
        from perception.architecture.promotion import (
            promote_component_to_entity,
        )
        from world_ir import WorldIR

        obs = build_component_observations(
            [(("seg-1",), fit, evidence_ids)],
            up=UP,
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.arch_class == fit.kind
        assert o.accepted
        assert o.evidence_ids == tuple(sorted(set(evidence_ids)))

        candidates = resolve_components(obs)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.arch_class == fit.kind

        world = WorldIR(id="w-col-beam-test")
        entity = promote_component_to_entity(cand, world, entity_id)
        assert world.entities[entity_id] is entity
        return entity

    def test_column_end_to_end(self):
        pts = _cylinder_points(
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, -0.75, 0.0, 0.75, 1.5, 2.25],
            angles=[i * math.pi / 8 for i in range(12)],
        )
        fit = detect_column(pts)
        entity = self._promote(fit, ("ev-1", "ev-2"), "col-1")
        assert entity.confidence == pytest.approx(fit.confidence)
        props = entity.custom_properties
        assert props["fit_kind"] == "column"
        assert props["radius_m"] == pytest.approx(0.3, abs=1e-6)
        assert props["height_m"] == pytest.approx(3.75, abs=1e-6)
        assert props["evidence_ids"] == ["ev-1", "ev-2"]

    def test_beam_end_to_end(self):
        ts = [i * 0.25 for i in range(13)]
        pts = _box_points(
            (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), ts,
            us=[s * 0.075 for s in (-2, -1, 1, 2)],
            ws=[s * 0.05 for s in (-2, 2)],
        )
        fit = detect_beam(pts)
        entity = self._promote(fit, ("ev-1",), "beam-1")
        assert entity.confidence == pytest.approx(fit.confidence)
        props = entity.custom_properties
        assert props["fit_kind"] == "beam"
        assert props["length_m"] == pytest.approx(3.0, abs=0.05)
        assert props["aspect_ratio"] > 3.0
        assert props["evidence_ids"] == ["ev-1"]

    def test_refused_points_never_become_entities(self):
        # A refused structure raises at detection time: there is no
        # observation to accept, no entity to promote.
        tilt = _norm((1.0, 0.0, 1.0))
        pts = _cylinder_points(
            tilt, (0.0, 0.0, 0.0), 0.3,
            ts=[-1.5, 0.0, 1.5], angles=[i * math.pi / 6 for i in range(10)],
        )
        with pytest.raises(ColumnRefused):
            detect_column(pts)
        with pytest.raises(BeamRefused):
            detect_beam(pts)
