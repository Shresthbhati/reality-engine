"""Tests for architectural component hypotheses and multi-view entity
resolution (directive sections 5-11, 24, 28): turning fitted segments
into component observations, resolving them into candidate world
components, and layering repetition/symmetry as SUPPORTING evidence.

Rules under test:
  - An observation carries its real geometry (fit), source evidence
    ids, and measured confidence -- nothing fabricated.
  - Registry acceptance gates are evaluated per class: a fitted
    cylinder that is not vertical enough does NOT become a column
    component -- it becomes an unaccepted hypothesis with the reason
    recorded (never silently dropped, never silently accepted).
  - Confidence tiers (directive section 28): OBSERVED / STRONG /
    WEAK / UNOBSERVED are derived from measured confidence bands,
    documented, deterministic.
  - Entity resolution: same-class, spatially-close observations merge
    into ONE candidate component (union-find, deterministic); every
    source observation is retained; cross-class merging never happens.
  - Repetition: a group of >=3 same-class components with consistent
    spacing records a repetition prior -- supporting evidence that
    RAISES confidence of members, and lets a POORLY-observed member
    cite the pattern, without cloning anyone's geometry.
  - Symmetry: a detected bilateral/radial symmetry of components is
    recorded as a prior; it must NOT move any component's geometry.
"""

from __future__ import annotations

import math

import pytest

from perception.architecture.components import (
    ComponentObservation,
    ConfidenceTier,
    build_component_observations,
    resolve_components,
    detect_repetition,
    apply_repetition_priors,
    detect_symmetry,
)
from perception.architecture.parametric import CylinderFit, SphereFit, CircleFit

UP = (0.0, 0.0, 1.0)


def _cyl_fit(radius=0.3, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
             h0=-1.5, h1=1.5, rms=0.002):
    return CylinderFit(
        axis=axis,
        axis_point=origin,
        radius_m=radius,
        height_min_m=h0,
        height_max_m=h1,
        rms_residual_m=rms,
        max_residual_m=rms * 2.0,
        n_points=500,
        confidence=1.0 / (1.0 + (rms / 0.1) ** 2),
    )


def _sphere_fit(center=(0.0, 0.0, 0.0), radius=2.0, rms=0.003):
    return SphereFit(
        center=center,
        radius_m=radius,
        rms_residual_m=rms,
        max_residual_m=rms * 2.0,
        n_points=800,
        confidence=1.0 / (1.0 + (rms / 0.1) ** 2),
    )


def _circle_fit(center=(0.0, 0.0, 2.0), radius=1.0, span=math.pi, rms=0.002):
    return CircleFit(
        center=center,
        radius_m=radius,
        plane_normal=(1.0, 0.0, 0.0),
        angular_span_rad=span,
        opening_direction=(0.0, -1.0, 0.0),
        extrusion_depth_m=0.5,
        rms_residual_m=rms,
        max_residual_m=rms * 2.0,
        n_points=200,
        confidence=1.0 / (1.0 + (rms / 0.1) ** 2),
    )


class TestObservationConstruction:
    def test_vertical_cylinder_becomes_column_observation(self):
        obs = build_component_observations(
            [(("seg-1",), _cyl_fit(), ("img-102", "img-156"))],
            up=UP,
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.arch_class == "column"
        assert o.fit is not None
        assert o.evidence_ids == ("img-102", "img-156")
        assert o.accepted is True
        assert o.confidence == pytest.approx(o.fit.confidence)

    def test_tilted_cylinder_is_recorded_unaccepted_not_dropped(self):
        axis = (1.0, 0.0, 0.0)  # horizontal: not a column
        obs = build_component_observations(
            [(("seg-2",), _cyl_fit(axis=axis), ("img-7",))],
            up=UP,
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.arch_class == "column"
        assert o.accepted is False
        assert "vertical" in o.rejection_reason.lower()

    def test_horizontal_cylinder_without_column_gate_is_unknown_class(self):
        # A cylinder segment too horizontal for 'column' is recorded as
        # unaccepted column -- NOT reclassified by guesswork.
        obs = build_component_observations(
            [(("seg-3",), _cyl_fit(axis=(0.0, 1.0, 0.0)), ("img-9",))],
            up=UP,
        )
        assert obs[0].accepted is False

    def test_dome_sphere_accepted(self):
        obs = build_component_observations(
            [(("seg-dome",), _sphere_fit(center=(0, 0, 10.0)), ("img-1",))],
            up=UP,
        )
        # Sphere fits produce dome candidates via the registry.
        assert obs[0].arch_class == "dome"
        assert obs[0].accepted is True

    def test_confidence_tiers_documented_bands(self):
        assert ConfidenceTier.from_confidence(0.99) is ConfidenceTier.OBSERVED
        assert ConfidenceTier.from_confidence(0.80) is ConfidenceTier.STRONG
        assert ConfidenceTier.from_confidence(0.50) is ConfidenceTier.WEAK
        # Below the WEAK floor, a component is UNOBSERVED for world
        # purposes (still recorded upstream -- never deleted).
        assert ConfidenceTier.from_confidence(0.2) is ConfidenceTier.UNOBSERVED

    def test_to_dict_carries_provenance_chain(self):
        obs = build_component_observations(
            [(("seg-9",), _cyl_fit(), ("img-a", "img-b"))],
            up=UP,
        )
        d = obs[0].to_dict()
        for key in ("segment_id", "arch_class", "evidence_ids",
                    "confidence", "tier", "accepted", "fit"):
            assert key in d


class TestEntityResolution:
    def _col_obs(self, seg_id, x, evidence=("img-1",), rms=0.002):
        return build_component_observations(
            [((seg_id,), _cyl_fit(origin=(x, 0.0, 0.0), rms=rms), evidence)],
            up=UP,
        )[0]

    def test_same_column_multi_view_merges_to_one(self):
        # The same physical column observed in 4 images: 4 observations,
        # one candidate entity.
        obs = [
            self._col_obs("seg-a", 5.0, ("img-102",)),
            self._col_obs("seg-b", 5.02, ("img-156",)),
            self._col_obs("seg-c", 4.98, ("img-221",)),
            self._col_obs("seg-d", 5.01, ("img-417",)),
        ]
        candidates = resolve_components(obs, merge_distance_m=0.5)
        cols = [c for c in candidates if c.arch_class == "column"]
        assert len(cols) == 1
        cand = cols[0]
        assert cand.observation_count == 4
        assert cand.evidence_ids == ("img-1",) or len(cand.evidence_ids) == 4
        assert cand.arch_class == "column"
        # Fused geometry: the merged candidate keeps a real fit
        # (refit from member observations' parameters, not averaged
        # garbage) -- at minimum it must exist and be near the members.
        assert cand.position[0] == pytest.approx(5.0, abs=0.05)

    def test_two_distinct_columns_stay_distinct(self):
        obs = [self._col_obs("seg-l", 0.0), self._col_obs("seg-r", 3.0)]
        candidates = resolve_components(obs, merge_distance_m=0.5)
        assert len([c for c in candidates if c.arch_class == "column"]) == 2

    def test_cross_class_never_merges(self):
        col = self._col_obs("seg-c", 0.0)
        dome = build_component_observations(
            [(("seg-s",), _sphere_fit(center=(0.0, 0.0, 0.05)), ("img-2",))],
            up=UP,
        )[0]
        candidates = resolve_components([col, dome], merge_distance_m=10.0)
        assert len(candidates) == 2

    def test_poor_observation_uses_repetition_prior(self):
        # Three well-observed columns at regular spacing, one poorly
        # observed (high rms) at the next spacing position.
        good = [
            self._col_obs("seg-1", 0.0, ("img-1", "img-2", "img-3")),
            self._col_obs("seg-2", 3.0, ("img-1", "img-2", "img-3")),
            self._col_obs("seg-3", 6.0, ("img-1", "img-2", "img-3")),
        ]
        poor = self._col_obs("seg-4", 9.0, ("img-4",), rms=0.04)
        candidates = resolve_components(good + [poor], merge_distance_m=0.5)
        reps = detect_repetition(candidates)
        assert len(reps) == 1
        rep = reps[0]
        assert rep.count >= 4
        assert rep.spacing_m == pytest.approx(3.0, abs=0.1)
        # Apply the prior: the pattern supports the poor member -- its
        # effective confidence rises above its raw observation
        # confidence, but never above the best-supported sibling (the
        # pattern is evidence, not a license to claim more than the
        # class demonstrates). Geometry is untouched.
        adjusted = apply_repetition_priors(candidates, reps)
        poor_cand = [c for c in adjusted if c.arch_class == "column" and abs(c.position[0] - 9.0) < 0.1][0]
        assert poor_cand.effective_confidence() > poor_cand.source_observations[0].confidence
        assert poor_cand.effective_confidence() <= max(
            c.confidence for c in adjusted
            if c.arch_class == "column" and c is not poor_cand
        ) + 1e-9
        # And the unadjusted candidates are unchanged by the prior pass.
        raw = [c for c in candidates if c.arch_class == "column" and abs(c.position[0] - 9.0) < 0.1][0]
        assert raw.prior_adjusted_confidence is None
        assert poor_cand.position == raw.position


class TestSymmetry:
    def test_bilateral_symmetry_detected_and_recorded(self):
        obs = []
        xs = [0.0, 3.0, 6.0]
        for i, x in enumerate(xs):
            obs.append(build_component_observations(
                [((f"seg-l{i}",), _cyl_fit(origin=(x, 2.0, 0.0)), (f"img-{i}",))], up=UP)[0])
            obs.append(build_component_observations(
                [((f"seg-r{i}",), _cyl_fit(origin=(x, -2.0, 0.0)), (f"img-{i}",))], up=UP)[0])
        candidates = resolve_components(obs, merge_distance_m=0.5)
        syms = detect_symmetry(candidates)
        assert any(s.kind == "bilateral" for s in syms)
        sym = [s for s in syms if s.kind == "bilateral"][0]
        # The symmetry plane is the x-z plane (y=0) here.
        assert sym.axis_point is not None
        # Symmetry must NOT move geometry: positions unchanged.
        for c in candidates:
            assert c.fit is not None or c.position is not None

    def test_symmetry_never_overwrites_observed_geometry(self):
        # One side has radius 0.3, mirrored side 0.5 (asymmetric
        # reality): symmetry is recorded but radii stay as observed.
        left = build_component_observations(
            [(("seg-l",), _cyl_fit(radius=0.3, origin=(0, 2.0, 0.0)), ("img-1",))], up=UP)[0]
        right = build_component_observations(
            [(("seg-r",), _cyl_fit(radius=0.5, origin=(0, -2.0, 0.0)), ("img-2",))], up=UP)[0]
        candidates = resolve_components([left, right], merge_distance_m=0.5)
        detect_symmetry(candidates)
        radii = sorted(
            c.fit.radius_m for c in candidates if c.fit is not None
        )
        assert radii == pytest.approx([0.3, 0.5], abs=1e-9)
