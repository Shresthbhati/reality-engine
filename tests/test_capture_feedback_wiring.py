"""Wiring test (connective-tissue rule: a capability that nothing
consumes is unfinished): recommend_capture's capture-feedback output
now carries the machine-readable scene budget derived by the
detail-budget layer, so the Capture application gets ONE artifact with
prose recommendations AND the derived justified-detail/compute tier."""

from __future__ import annotations

from perception.quality.assessment import EvidenceQualityReport, recommend_capture
from perception.quality.detail_budget import DetailBudget, detail_budget_for


def _report(gsd, tier, view_counts=None, unprojectable=()):
    return EvidenceQualityReport(
        gsd_mm_per_px=gsd,
        detail_tier=tier,
        observed_fraction=1.0 if gsd is not None else 0.0,
        view_counts=dict(view_counts if view_counts is not None else {"p0": 4}),
        unprojectable_point_ids=tuple(unprojectable),
        overclaim_count=0,
    )


class TestRecommendCaptureCarriesBudget:
    def test_recommendations_include_machine_readable_budget(self):
        recs = recommend_capture(_report(3.0, "fine"))
        budget = recs["scene_budget"]
        assert isinstance(budget, DetailBudget)
        assert budget.justified_level == "L4"
        # Prose list unchanged in kind (may be empty for good evidence).
        prose = [k for k in recs if k != "scene_budget"]
        assert all(isinstance(k, str) for k in prose)

    def test_budget_is_derived_not_invented(self):
        recs = recommend_capture(_report(60.0, "coarse"))
        assert recs["scene_budget"].to_dict() == detail_budget_for(
            _report(60.0, "coarse")
        ).to_dict()

    def test_unsupported_scene_gets_honest_budget(self):
        recs = recommend_capture(_report(None, "unsupported", view_counts={}))
        assert recs["scene_budget"].basis == "unsupported"
        assert recs["scene_budget"].compute_tier == "none"
