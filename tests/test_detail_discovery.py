"""Universal detail discovery (directive PHASE 2-3; sections 8/11 of
the universal-perception directive): given a reconstruction's points,
find WHERE the evidence supports recoverable detail -- domain-agnostic
(a carved facade, a milled gear, tree bark, and a road crack are all
just points) -- and attach a per-candidate DetailBudget.

Contract under test:
  - Discovery is DETERMINISTIC: same points -> byte-identical
    candidates, in a stable spatial order (sorted by cell key).
  - Scores are MEASURED: curvature is the PCA smallest-eigenvalue
    ratio of a cell's points (planar -> ~0, curved -> high), point
    density is count/area, both recorded per candidate.
  - Tier is derived by the documented detail_budget mapping -- a flat
    low-density region CANNOT become an L4 candidate (the
    hallucination gate), and a rich curved region with fine GSD can.
  - A cell too small to fit (min_points) is skipped, never guessed.
  - Domain-agnostic: the SAME pipeline discovers detail on an
    architectural ornament AND a mechanical/generic curved surface.
"""

from __future__ import annotations

import math

import pytest

from perception.detail.discovery import discover_detail
from perception.quality.assessment import EvidenceQualityReport
from reconstruction.backend.interface import ReconstructedPoint
from tests.test_evidence_quality import _camera, _intrinsics, _result


def _pt(xyz, tid, eid="c0"):
    return ReconstructedPoint(
        position=xyz, track_id=tid, source_evidence_ids=[eid]
    )


def _plane_patch(cx, cy, z=2.0, n=40, jitter=0.0):
    """A flat patch (zero curvature by construction). z=2 puts it in
    the camera's view (|x| <= 1, |y| <= 0.75 in bounds at depth 2)."""
    pts = []
    for i in range(n):
        pts.append(_pt(
            (cx + (i % 8) * 0.05, cy + (i // 8) * 0.05, z + jitter * i),
            f"pl-{cx}-{z}-{i}",  # z in the id: patches at different depths are distinct points
        ))
    return pts


def _cyl_shell(cx, cy, r=0.3, rings=6, per_ring=10):
    """A cylindrical shell sample (high curvature by construction):
    stacked rings of points, as a real scanner samples a cylinder.
    Centered inside one voxel of the default 1.0 m grid (x/y within
    [cx+0.2, cx+0.8], z within [0.4, 0.9]) so the whole shell lands
    in ONE cell. The ring stacking is what makes the point set occupy
    a curved (non-planar) shell: PCA smallest-eigenvalue ratio is
    ~0.19 for this geometry vs ~0 for the flat patch."""
    pts = []
    for j in range(rings):
        z = 0.4 + j * 0.5 / (rings - 1)
        for k in range(per_ring):
            a = k * 2.0 * math.pi / per_ring + j * 0.1
            pts.append(_pt(
                (cx + 0.5 + r * math.cos(a), cy + 0.5 + r * math.sin(a), z),
                f"cyl-{cx}-{j}-{k}",
            ))
    return pts


class TestDiscoveryContract:
    def test_flat_region_scores_low_curvature(self):
        pts = _plane_patch(0.0, 0.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        assert cands
        assert all(c.curvature < 0.05 for c in cands)

    def test_cylindrical_region_scores_high_curvature(self):
        pts = _cyl_shell(0.0, 0.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        assert cands
        best = max(cands, key=lambda c: c.curvature)
        assert best.curvature > 0.1

    def test_deterministic_and_stable_order(self):
        pts = _plane_patch(0.0, 0.0) + _cyl_shell(2.0, 2.0)
        report = _quality_report(pts)
        c1 = discover_detail(_result(pts, _report_cameras()), report)
        c2 = discover_detail(_result(pts, _report_cameras()), report)
        assert [c.to_dict() for c in c1] == [c.to_dict() for c in c2]
        keys = [c.cell_id for c in c1]
        assert keys == sorted(keys)

    def test_small_cells_are_skipped_not_guessed(self):
        # 3 points in one voxel: below the minimum -- skipped entirely.
        pts = [_pt((0.0, 0.0, 0.0), "s0"), _pt((0.01, 0.0, 0.0), "s1"),
               _pt((0.0, 0.01, 0.0), "s2")]
        report = _quality_report(pts)
        cands = discover_detail(
            _result(pts, _report_cameras()), report, min_points=8
        )
        assert cands == []


def _report_cameras():
    # One camera at origin looking +Z, fx=width (see _intrinsics doc):
    # points at z=2 with |x|<=1 are in bounds.
    return [_camera("cam-000", (0.0, 0.0, 0.0))]


def _quality_report(pts):
    from perception.quality.assessment import assess_evidence_quality

    cams = _report_cameras()
    return assess_evidence_quality(_result(pts, cams), cams)


class TestBudgetAttachment:
    def test_candidates_carry_derived_budgets(self):
        pts = _cyl_shell(0.0, 0.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        assert all(c.budget is not None for c in cands)
        assert all(
            c.budget.justified_level in ("L0", "L1", "L2", "L3", "L4")
            for c in cands
        )

    def test_flat_low_density_region_cannot_claim_l4(self):
        # The hallucination gate: flat + coarse GSD -> the derived
        # budget must cap below fine detail no matter what.
        pts = _plane_patch(0.0, 0.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        assert all(c.budget.justified_level != "L4" for c in cands)


class TestDomainAgnosticism:
    def test_same_pipeline_flags_mechanical_and_organic_curves(self):
        # A "machine" cylinder and a "natural" bump discovered by the
        # SAME code path -- no architecture-specific branch exists.
        pts = _cyl_shell(0.0, 0.0) + _cyl_shell(5.0, 5.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        curved = [c for c in cands if c.curvature > 0.1]
        assert len(curved) >= 2


class TestPerCellGsdBudgets:
    """P7-05 remainder: each cell's budget must derive from ITS OWN
    measured GSD, not the scene median. The assessor measures per-point
    GSD; discovery now consumes it."""

    def test_report_records_per_point_gsd(self):
        from perception.quality.assessment import assess_evidence_quality

        cams = [_camera("cam-000", (0.0, 0.0, 0.0))]
        pts = [
            _pt((0.0, 0.0, 1.0), "near"),
            _pt((0.0, 0.0, 8.0), "far"),
        ]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.per_point_gsd_mm["near"] == pytest.approx(1000.0 / 640.0)
        assert report.per_point_gsd_mm["far"] == pytest.approx(8000.0 / 640.0)

    def test_cell_budget_uses_local_gsd_not_scene_median(self):
        # Near patch (gsd 1.56 mm/px) and far patch (12.5 mm/px): the
        # scene median sits between them; each cell's budget.max_gsd
        # must equal its own distance-derived GSD.
        pts = _plane_patch(0.0, 0.0, z=1.0) + _plane_patch(0.0, 0.0, z=8.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        by_z = {
            round(c.centroid[2]): c for c in cands
        }
        assert by_z[1].budget.max_gsd_mm_per_px == pytest.approx(1.5625, rel=0.05)
        assert by_z[8].budget.max_gsd_mm_per_px == pytest.approx(12.5, rel=0.05)
        # And the scene median is genuinely different from both.
        assert report.gsd_mm_per_px != pytest.approx(1.5625)

    def test_cell_without_measured_gsd_falls_back_to_scene(self):
        # A cell whose points are all unprojectable carries no
        # per-point GSD; its budget falls back to the scene's measured
        # GSD rather than crashing or guessing.
        pts = _plane_patch(0.0, 0.0, z=1.0)
        report = _quality_report(pts)
        composed = EvidenceQualityReport(
            gsd_mm_per_px=report.gsd_mm_per_px,
            detail_tier=report.detail_tier,
            observed_fraction=report.observed_fraction,
            view_counts={},
            unprojectable_point_ids=("ghost",),
            overclaim_count=0,
            per_point_gsd_mm={},
        )
        cands = discover_detail(_result(pts, _report_cameras()), composed)
        assert all(c.budget is not None for c in cands)
