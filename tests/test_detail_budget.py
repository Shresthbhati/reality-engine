"""Detail-budget derivation (universal-perception directive Phase 4,
sections 10-11): turning the MEASURED evidence-quality report
(perception/quality/assessment.py, P7-04) into a per-region statement
of the maximum detail level the evidence can defensibly support, with
a compute-allocation tier.

Contract under test:
  - The budget is DERIVED from measured facts only (GSD, view counts,
    coverage). Nothing is fabricated: no cameras/points -> honest
    "unsupported" region, never a guessed budget.
  - Tiers are documented monotone mappings: finer GSD + more views ->
    higher justified detail and higher compute allocation. The region
    the evidence supports at L4 gets max allocation; a coarse-GSD
    region gets survey-level allocation, NOT silently full detail.
  - Insufficient multi-view coverage CAPS the detail level even when
    GSD is fine (fine pixels, one view = unreliable detail).
  - `detail_budget_for` refuses (ValueError) when handed a report with
    no measured GSD AND observed points -- there is nothing to derive
    from; and the empty-scene report maps to an honest unsupported
    region with zero allocation, not an error.
  - Deterministic: same report -> byte-identical budget.
"""

from __future__ import annotations

import pytest

from perception.quality.assessment import EvidenceQualityReport
from perception.quality.detail_budget import detail_budget_for


def _report(gsd, tier, observed=1.0, view_counts=None, unprojectable=()):
    return EvidenceQualityReport(
        gsd_mm_per_px=gsd,
        detail_tier=tier,
        observed_fraction=observed,
        view_counts=dict({"p0": 4} if view_counts is None else view_counts),
        unprojectable_point_ids=tuple(unprojectable),
        overclaim_count=0,
    )


class TestDetailBudgetDerivation:
    def test_fine_gsd_multiview_yields_l4_budget(self):
        b = detail_budget_for(_report(3.0, "fine"))
        assert b.justified_level == "L4"
        assert b.max_gsd_mm_per_px == 3.0
        assert b.compute_tier in ("full", "high")
        assert b.basis == "measured"

    def test_medium_gsd_yields_l2(self):
        b = detail_budget_for(_report(15.0, "medium"))
        assert b.justified_level == "L2"

    def test_coarse_gsd_yields_l0(self):
        # Documented mapping: >100 mm/px is L0 (macro massing only);
        # 25-100 is L1 (structural surfaces).
        b = detail_budget_for(_report(150.0, "coarse"))
        assert b.justified_level == "L0"
        assert b.compute_tier == "survey"
        b1 = detail_budget_for(_report(60.0, "coarse"))
        assert b1.justified_level == "L1"
        assert b1.compute_tier == "light"

    def test_unsupported_tier_yields_l0_zero_allocation(self):
        # An unsupported report has NO observed points (empty view
        # counts) -- the assessor cannot produce views without GSD.
        b = detail_budget_for(_report(None, "unsupported", view_counts={}))
        assert b.justified_level == "L0"
        assert b.compute_tier == "none"
        assert b.basis == "unsupported"

    def test_fine_gsd_single_view_is_capped_by_coverage(self):
        # Fine pixels but only ONE view: detail beyond L2 is not
        # defensible without multi-view corroboration.
        b = detail_budget_for(
            _report(3.0, "fine", view_counts={"p0": 1})
        )
        assert b.justified_level in ("L0", "L1", "L2")
        assert b.coverage_capped is True

    def test_coarse_tier_cannot_claim_fine_level(self):
        # The derived level may never contradict the measured tier.
        b = detail_budget_for(_report(40.0, "coarse"))
        assert b.justified_level in ("L0", "L1")

    def test_refuses_report_with_no_measurable_evidence(self):
        # OBSERVED points (non-empty view counts) with GSD None means
        # the report was built outside the assessor's contract -- the
        # assessor refuses to guess intrinsics, so this shape cannot
        # come from it. Refuse rather than invent a budget.
        with pytest.raises(ValueError):
            detail_budget_for(
                _report(None, "unsupported", view_counts={"p0": 2})
            )

    def test_coverage_failure_maps_to_unsupported_not_error(self):
        # Points exist but no camera observed any: the assessor DOES
        # produce this (empty view counts, all unprojectable) -- an
        # honest zero-allocation budget, not a refusal.
        b = detail_budget_for(
            _report(None, "unsupported", observed=0.0,
                    view_counts={}, unprojectable=("p0", "p1"))
        )
        assert b.compute_tier == "none"
        assert b.basis == "unsupported"


class TestDeterminismAndSerialization:
    def test_deterministic(self):
        r = _report(4.0, "fine")
        b1 = detail_budget_for(r)
        b2 = detail_budget_for(r)
        assert b1.to_dict() == b2.to_dict()

    def test_to_dict_roundtrip_fields(self):
        b = detail_budget_for(_report(4.0, "fine", observed=0.9))
        d = b.to_dict()
        for key in ("justified_level", "compute_tier", "max_gsd_mm_per_px",
                    "coverage_capped", "basis"):
            assert key in d
