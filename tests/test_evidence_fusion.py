"""Tests for evidence fusion (reconstruction/fusion/fusion.py).

Fixtures are hand-computable: the spec's canonical conflict example
(LiDAR 3.17 m vs photogrammetry 3.22 m) and controlled synthetic
quantities where the fused value, precision, sigma multiple, and
agreement RMS are all closed-form.
"""

from __future__ import annotations

import math

import pytest

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion import (
    CONFLICT_SIGMA,
    FusableObservation,
    FusionError,
    fused_to_measurement,
    fuse_quantity,
)


def _obs(
    value: float,
    source: str,
    precision: float = 0.01,
    unit: Unit = Unit.METER,
    provenance: Provenance = Provenance.RECONSTRUCTED,
    confidence: float = 0.9,
    evidence_ids: tuple = (),
) -> FusableObservation:
    return FusableObservation(
        value=value,
        unit=unit,
        source=source,
        provenance=provenance,
        confidence=confidence,
        precision=precision,
        evidence_ids=evidence_ids,
    )


class TestConstruction:
    def test_conflict_state_exists_in_provenance_vocabulary(self):
        # The spec's sec 15 vocabulary the fusion layer depends on.
        assert Provenance.CONFLICT.value == "CONFLICT"

    def test_rejects_zero_precision(self):
        with pytest.raises(FusionError, match="precision must be > 0"):
            FusableObservation(
                value=1.0, unit=Unit.METER, source="perfect_sensor",
                provenance=Provenance.OBSERVED, confidence=1.0, precision=0.0,
            )

    def test_rejects_negative_precision(self):
        with pytest.raises(FusionError, match="precision must be > 0"):
            FusableObservation(
                value=1.0, unit=Unit.METER, source="s",
                provenance=Provenance.OBSERVED, confidence=0.9, precision=-1.0,
            )

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(FusionError, match="confidence"):
            FusableObservation(
                value=1.0, unit=Unit.METER, source="s",
                provenance=Provenance.OBSERVED, confidence=1.5, precision=0.1,
            )

    def test_rejects_empty_source(self):
        with pytest.raises(FusionError, match="source"):
            FusableObservation(
                value=1.0, unit=Unit.METER, source="",
                provenance=Provenance.OBSERVED, confidence=0.9, precision=0.1,
            )

    def test_rejects_non_finite_value(self):
        with pytest.raises(FusionError, match="finite"):
            FusableObservation(
                value=float("nan"), unit=Unit.METER, source="s",
                provenance=Provenance.OBSERVED, confidence=0.9, precision=0.1,
            )


class TestPassthrough:
    def test_single_observation_is_exact_passthrough(self):
        obs = _obs(3.18, "lidar", precision=0.005, confidence=0.95)
        fused = fuse_quantity([obs], quantity="room_width")
        assert fused.value == 3.18
        assert fused.precision == 0.005
        assert fused.confidence == 0.95
        assert fused.provenance is Provenance.RECONSTRUCTED
        assert fused.method == "passthrough"
        assert fused.conflicts == ()
        assert fused.agreement_rms is None
        assert fused.contributing == (obs,)
        assert not fused.has_conflicts()

    def test_empty_observations_raise(self):
        with pytest.raises(FusionError, match="no observations"):
            fuse_quantity([], quantity="room_width")

    def test_unit_mismatch_raises_and_never_fuses(self):
        with pytest.raises(FusionError, match="unit mismatch"):
            fuse_quantity(
                [_obs(3.0, "lidar", unit=Unit.METER),
                 _obs(4.0, "scale", unit=Unit.KILOGRAM)],
                quantity="mystery",
            )


class TestConflictDetection:
    def test_spec_example_conflict(self):
        """The exact scenario from the spec: LiDAR 3.17 vs photogrammetry
        3.22 with honest centimeter-scale uncertainties is a real conflict
        (7.07 combined sigma), not noise."""
        lidar = _obs(3.17, "lidar", precision=0.005, confidence=0.98)
        photo = _obs(3.22, "photogrammetry", precision=0.005, confidence=0.95)
        fused = fuse_quantity([lidar, photo], quantity="wall_distance")

        assert fused.has_conflicts()
        assert fused.provenance is Provenance.CONFLICT
        assert len(fused.conflicts) == 1
        c = fused.conflicts[0]
        assert c.source_a == "lidar"
        assert c.source_b == "photogrammetry"
        assert c.value_a == 3.17
        assert c.value_b == 3.22
        assert c.delta == pytest.approx(0.05)
        assert c.combined_precision == pytest.approx(math.sqrt(2) * 0.005)
        assert c.sigma_multiple == pytest.approx(0.05 / (math.sqrt(2) * 0.005))
        assert c.sigma_multiple > CONFLICT_SIGMA

    def test_conflict_fused_value_is_still_weighted_mean_of_all_sources(self):
        # Resolution is provenance + lowered confidence + preserved pairs --
        # NEVER discarding a source or picking a winner. Equal weights here.
        fused = fuse_quantity(
            [_obs(3.17, "lidar", precision=0.005),
             _obs(3.22, "photogrammetry", precision=0.005)],
            quantity="wall_distance",
        )
        assert fused.value == pytest.approx(3.195)
        assert fused.precision == pytest.approx(0.005 / math.sqrt(2))
        # Confidence drops to the weakest contributor.
        assert fused.confidence == pytest.approx(0.9)
        # Both raw observations survive verbatim.
        assert {o.source for o in fused.contributing} == {"lidar", "photogrammetry"}
        assert fused.contributing[0].value == 3.17
        assert fused.contributing[1].value == 3.22

    def test_same_delta_is_consistent_when_uncertainties_are_honest(self):
        # The SAME 5 cm delta with 2 cm-scale uncertainties is 1.77 sigma --
        # statistically consistent. Conflict is a function of claimed
        # uncertainty, not of the raw delta alone.
        fused = fuse_quantity(
            [_obs(3.17, "lidar", precision=0.02),
             _obs(3.22, "photogrammetry", precision=0.02)],
            quantity="wall_distance",
        )
        assert not fused.has_conflicts()
        assert fused.provenance is Provenance.ESTIMATED
        assert fused.conflicts == ()
        assert fused.agreement_rms == pytest.approx(0.05)

    def test_multiple_conflicting_pairs_all_preserved(self):
        # A: consistent with C, conflicts with B. Every conflicting pair
        # must survive -- not just the first found.
        fused = fuse_quantity(
            [_obs(1.00, "a", precision=0.01),
             _obs(1.50, "b", precision=0.01),
             _obs(1.00, "c", precision=0.01)],
            quantity="x",
        )
        assert fused.provenance is Provenance.CONFLICT
        assert len(fused.conflicts) == 2
        pair_sources = {(c.source_a, c.source_b) for c in fused.conflicts}
        # Canonical order sorts by value first: a(1.0), c(1.0), b(1.5) --
        # so b is always source_b (the higher value in each pair).
        assert pair_sources == {("a", "b"), ("c", "b")}
        assert fused.value == pytest.approx((1.0 + 1.5 + 1.0) / 3)
        assert fused.agreement_rms == pytest.approx(
            math.sqrt((0.0 + 0.25 + 0.25) / 3)
        )

    def test_conflicting_pair_ids_are_canonically_ordered(self):
        fused = fuse_quantity(
            [_obs(3.22, "photogrammetry", precision=0.005),
             _obs(3.17, "lidar", precision=0.005)],
            quantity="wall_distance",
        )
        # Canonical order is by (value, source, ...): 3.17 < 3.22.
        assert fused.conflicts[0].source_a == "lidar"
        assert fused.conflicts[0].source_b == "photogrammetry"


class TestConfidenceWeighting:
    def test_tighter_source_dominates(self):
        fused = fuse_quantity(
            [_obs(5.0, "tight", precision=0.01),
             _obs(6.0, "loose", precision=1.0)],
            quantity="distance",
        )
        # Weight ratio 1e4:1 -> fused within 1 cm of the tight source even
        # though the loose source claims 1.0 m away.
        assert fused.value == pytest.approx(5.0, abs=0.01)
        # 1.0 delta over combined sigma sqrt(0.0001+1) ~ 1.0 sigma: consistent.
        assert not fused.has_conflicts()
        assert fused.provenance is Provenance.ESTIMATED

    def test_three_consistent_observations_fuse_tighter(self):
        fused = fuse_quantity(
            [_obs(3.20, "a", precision=0.01),
             _obs(3.20, "b", precision=0.01),
             _obs(3.20, "c", precision=0.01)],
            quantity="height",
        )
        assert fused.value == pytest.approx(3.20)
        assert fused.precision == pytest.approx(0.01 / math.sqrt(3))
        assert fused.precision < 0.01
        assert fused.provenance is Provenance.ESTIMATED


class TestDeterminism:
    def test_input_order_never_changes_the_result(self):
        obs = [
            _obs(3.17, "lidar", precision=0.005),
            _obs(3.22, "photogrammetry", precision=0.005),
            _obs(3.19, "vggt", precision=0.008, confidence=0.93),
        ]
        baseline = fuse_quantity(obs, quantity="wall_distance")
        for permuted in (list(reversed(obs)), [obs[2], obs[0], obs[1]]):
            assert fuse_quantity(permuted, quantity="wall_distance") == baseline

    def test_same_inputs_repeatedly_are_identical(self):
        obs = [_obs(2.5, "s1", precision=0.02), _obs(2.7, "s2", precision=0.02)]
        first = fuse_quantity(obs, quantity="q")
        for _ in range(5):
            assert fuse_quantity(obs, quantity="q") == first


class TestWorldIRBridge:
    def test_fused_to_measurement_preserves_everything(self):
        fused = fuse_quantity(
            [_obs(3.17, "lidar", precision=0.005, confidence=0.98,
                  evidence_ids=("ev-1", "ev-2")),
             _obs(3.22, "photogrammetry", precision=0.005, confidence=0.95)],
            quantity="wall_distance",
        )
        m = fused_to_measurement(fused)
        assert m.value == pytest.approx(3.195)
        assert m.unit == "meter"
        assert m.precision == pytest.approx(0.005 / math.sqrt(2))
        assert m.provenance is Provenance.CONFLICT
        assert m.confidence == pytest.approx(0.95)

    def test_passthrough_keeps_observed_provenance_into_measurement(self):
        m = fused_to_measurement(fuse_quantity(
            [_obs(1.234, "total_station", provenance=Provenance.OBSERVED)],
            quantity="control_distance",
        ))
        assert m.provenance is Provenance.OBSERVED
        assert m.value == pytest.approx(1.234)
