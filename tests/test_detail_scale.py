"""P7-05: the detail budget is scale-aware. A relative reconstruction (arbitrary units) never gets a millimetre GSD,
a fine/medium/coarse tier or a GSD level band; its voxel comes from the scene extent, not from "1.0 metre".

Synthetic scenes are used on purpose: the property under test is INVARIANCE under the arbitrary unit, which only a
scene we can rescale exactly can prove. Real-data calibration lives in docs/engineering/DETAIL_CALIBRATION.md.
"""

from __future__ import annotations

import math

import pytest

from perception.detail.pipeline import run_detail_pipeline
from perception.detail.voxel import (
    METRIC_DEFAULT_VOXEL_M,
    MIN_POINTS_FOR_EXTENT,
    RELATIVE_VOXEL_FRACTION,
    derive_voxel_size,
    robust_scene_extent,
)
from perception.quality.assessment import assess_evidence_quality, recommend_capture
from perception.quality.detail_budget import detail_budget_for
from reconstruction.backend.interface import ReconstructedPoint
from tests.test_evidence_quality import _camera, _result


def _scene(k: float):
    """A wavy patch seen by three cameras, every coordinate multiplied by ``k`` (the arbitrary unit)."""
    pts = []
    for i in range(30):
        for j in range(20):
            x, y = -0.8 + 1.6 * i / 29, -0.5 + 1.0 * j / 19
            z = 2.0 + 0.25 * math.sin(4.0 * x) * math.cos(3.0 * y)
            pts.append(ReconstructedPoint(position=(k * x, k * y, k * z), track_id=f"p{i}-{j}",
                                          source_evidence_ids=["c0", "c1", "c2"]))
    cams = [_camera(f"c{n}", (k * dx, 0.0, 0.0)) for n, dx in enumerate((-0.1, 0.0, 0.1))]
    return _result(pts, cams), cams


# --------------------------------------------------------------------------------------------- extent + voxel


def test_the_robust_extent_ignores_far_flung_tracks_a_bounding_box_would_swallow():
    pts = [ReconstructedPoint(position=(i % 10 * 0.1, i // 10 % 10 * 0.1, i // 100 * 0.1), track_id=f"a{i}",
                              source_evidence_ids=[]) for i in range(500)]
    clean = robust_scene_extent(pts)
    pts += [ReconstructedPoint(position=(500.0, 500.0, 500.0), track_id="far1", source_evidence_ids=[]),
            ReconstructedPoint(position=(-400.0, 0.0, 300.0), track_id="far2", source_evidence_ids=[])]
    assert robust_scene_extent(pts) == pytest.approx(clean, rel=0.15)        # two outliers in 502 points: ignored
    bbox = math.dist((-400, 0, 0), (500, 500, 500))
    assert bbox > 100 * clean                                                  # and a bbox would have been 100x off


def test_an_extent_is_not_invented_from_too_few_points():
    pts = [ReconstructedPoint(position=(i, 0, 0), track_id=f"a{i}", source_evidence_ids=[])
           for i in range(MIN_POINTS_FOR_EXTENT - 1)]
    assert robust_scene_extent(pts) is None
    with pytest.raises(ValueError, match="cannot be measured"):
        derive_voxel_size(pts, "relative")


def test_voxel_choice_by_scale_state():
    result, _ = _scene(1.0)
    assert derive_voxel_size(result.points, "metric").size == METRIC_DEFAULT_VOXEL_M
    assert derive_voxel_size(result.points, "metric").basis == "metric_default"
    assert derive_voxel_size(result.points, "relative", explicit=0.3).size == 0.3          # the caller knows better
    rel = derive_voxel_size(result.points, "relative")
    assert rel.basis == "scene_extent_fraction" and rel.size == pytest.approx(RELATIVE_VOXEL_FRACTION * rel.scene_extent)
    assert derive_voxel_size(result.points, "unknown").basis == "scene_extent_fraction"     # unknown is NOT metric
    with pytest.raises(ValueError):
        derive_voxel_size(result.points, "relative", explicit=-1.0)


@pytest.mark.parametrize("k", [0.001, 7.0, 1000.0])
def test_a_relative_voxel_scales_exactly_with_the_arbitrary_unit(k):
    base = derive_voxel_size(_scene(1.0)[0].points, "relative")
    scaled = derive_voxel_size(_scene(k)[0].points, "relative")
    assert scaled.size == pytest.approx(k * base.size, rel=1e-9)


# ---------------------------------------------------------------------------------- report and budget withhold mm


def test_a_metric_report_is_unchanged_and_a_relative_one_withholds_every_metric_statement():
    result, cams = _scene(1.0)
    metric = assess_evidence_quality(result, cams)                       # default: the units ARE metres
    rel = assess_evidence_quality(result, cams, scale_state="relative")
    assert metric.scale_state == "metric" and metric.gsd_mm_per_px is not None and metric.detail_tier != "scale_unavailable"
    assert metric.per_point_gsd_mm and metric.gsd_model_units_per_px is None
    assert rel.scale_state == "relative"
    assert rel.gsd_mm_per_px is None and rel.per_point_gsd_mm == {} and rel.detail_tier == "scale_unavailable"
    assert rel.gsd_model_units_per_px == pytest.approx(metric.gsd_mm_per_px / 1000.0)    # the scale-free measure survives
    assert rel.view_counts == metric.view_counts and rel.observed_fraction == metric.observed_fraction
    assert rel.to_dict()["scale_state"] == "relative"


def test_a_relative_budget_never_claims_a_metric_level_and_says_why():
    result, cams = _scene(1.0)
    metric_budget = detail_budget_for(assess_evidence_quality(result, cams))
    assert metric_budget.justified_level == "L4" and metric_budget.gsd_basis == "metric"     # 3 mm/px is fine detail
    b = detail_budget_for(assess_evidence_quality(result, cams, scale_state="relative"))
    assert (b.basis, b.gsd_basis, b.max_gsd_mm_per_px) == ("relative_scale", "unavailable", None)
    assert b.justified_level == "L2" and b.compute_tier == "standard" and not b.coverage_capped
    assert b.to_dict()["gsd_basis"] == "unavailable"


def test_a_relative_budget_is_capped_by_coverage_alone():
    pts = [ReconstructedPoint(position=(0.01 * i, 0.0, 2.0), track_id=f"s{i}", source_evidence_ids=["c0"])
           for i in range(60)]
    cams = [_camera("c0", (0.0, 0.0, 0.0))]
    b = detail_budget_for(assess_evidence_quality(_result(pts, cams), cams, scale_state="relative"))
    assert b.justified_level == "L1" and b.coverage_capped and b.gsd_basis == "unavailable"


def test_an_empty_relative_scene_is_an_honest_unsupported_budget():
    from reconstruction.backend.interface import ReconstructionResult

    empty = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
    b = detail_budget_for(assess_evidence_quality(empty, [], scale_state="relative"))
    assert (b.basis, b.compute_tier, b.justified_level) == ("unsupported", "none", "L0")


def test_capture_feedback_tells_the_operator_how_to_get_metric_detail():
    result, cams = _scene(1.0)
    recs = recommend_capture(assess_evidence_quality(result, cams, scale_state="relative"))
    assert "scale_unavailable" in recs and "measured distance" in recs["scale_unavailable"]
    assert "scale_unavailable" not in recommend_capture(assess_evidence_quality(result, cams))


# --------------------------------------------------------------- the whole chain is invariant under the unit


def _signature(report):
    return ([(c.cell_id, c.budget.justified_level, c.budget.compute_tier, c.is_detail, c.is_structure)
             for c in report.candidates],
            [(r.status, r.budget.justified_level) for r in report.rois])


@pytest.mark.parametrize("k", [0.001, 1000.0])
def test_the_relative_detail_chain_gives_the_same_answer_whatever_the_arbitrary_unit(k):
    base_result, base_cams = _scene(1.0)
    result, cams = _scene(k)
    a = run_detail_pipeline(base_result, base_cams, voxel_size=None, scale_state="relative", include_structure=True)
    b = run_detail_pipeline(result, cams, voxel_size=None, scale_state="relative", include_structure=True)
    assert a.candidates, "the scene must produce candidate cells or this proves nothing"
    assert _signature(a) == _signature(b)
    assert b.summary["voxel"]["basis"] == "scene_extent_fraction"
    assert b.summary["voxel"]["size"] == pytest.approx(k * a.summary["voxel"]["size"], rel=1e-9)
    assert b.quality.gsd_mm_per_px is None and b.quality.scale_state == "relative"


def test_whereas_a_metric_chain_legitimately_depends_on_scale():
    """The same shape at 1 m and at 1000 m is a different measurement in a metric world: GSD (mm/px) is real."""
    near = assess_evidence_quality(*_scene(1.0))
    far = assess_evidence_quality(*_scene(1000.0))
    assert detail_budget_for(near).justified_level == "L4" and detail_budget_for(far).justified_level == "L0"


# ---------------------------------------------------------------------------------------- the product boundary


class _World:
    def __init__(self):
        from world_ir import WorldIR

        self.inner = WorldIR()
        self.metadata = self.inner.metadata

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_the_vertical_slice_detail_stage_passes_the_worlds_real_scale_and_records_the_voxel_basis():
    from engine.pipeline.vertical_slice import VerticalSliceOptions, _detail_stage

    result, _ = _scene(37.0)                                        # arbitrary unit: 37 "something" per scene
    opts = VerticalSliceOptions(intrinsics=(640.0, 640.0, 320.0, 240.0), image_size=(640, 480))
    assert opts.detail_voxel_size_m is None                          # default derives, never assumes a metre
    rel = _detail_stage(result, _World(), opts, "relative")
    assert rel["status"] == "ran" and rel["summary"]["voxel"]["basis"] == "scene_extent_fraction"
    assert rel["quality"]["gsd_mm_per_px"] is None and rel["quality"]["scale_state"] == "relative"
    assert all(r["budget"]["gsd_basis"] == "unavailable" for r in rel["rois"])
    met = _detail_stage(result, _World(), opts, "metric")
    assert met["summary"]["voxel"]["basis"] == "metric_default" and met["quality"]["gsd_mm_per_px"] is not None
    unknown = _detail_stage(result, _World(), opts)                  # a caller that does not say is NOT metric
    assert unknown["summary"]["scale_state"] == "unknown" and unknown["quality"]["gsd_mm_per_px"] is None
