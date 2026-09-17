"""ROI generation tests (universal-perception directive sections
12/14): detail candidates become bounded work orders.

Contract under test:
  - ONLY detail-bearing cells seed ROIs (flat cells are never
    promoted -- the ROI analogue of the hallucination gate).
  - The ROI's aggregate budget is the weakest member budget: one
    single-view cell caps the whole ROI below L4/L3 claims.
  - Adjacent detail cells merge into ONE ROI (bounded growth), with
    unioned point provenance and tight spatial bounds.
  - Deterministic: byte-identical ROIs, stable order.
  - Status is "pending" -- generation creates work orders, it does
    not pretend refinement happened.
  - Empty discovery -> no ROIs, not an error or a fabricated ROI.
"""

from __future__ import annotations

import math

from perception.detail.discovery import discover_detail
from perception.detail.roi import PROCESSING_PENDING, generate_rois
from tests.test_evidence_quality import _camera, _result


def _plane_patch(cx, cy, z=2.0, n=40):
    pts = []
    for i in range(n):
        pts.append(_pt(
            (cx + (i % 8) * 0.05, cy + (i // 8) * 0.05, z),
            f"pl-{cx}-{i}",
        ))
    return pts


def _pt(xyz, tid, eid="c0"):
    from reconstruction.backend.interface import ReconstructedPoint

    return ReconstructedPoint(
        position=xyz, track_id=tid, source_evidence_ids=[eid]
    )


def _report(pts):
    from tests.test_detail_discovery import _quality_report, _report_cameras

    return _quality_report(pts)


def _shell(cx, cy):
    from tests.test_detail_discovery import _cyl_shell

    return _cyl_shell(cx, cy)


def _discover(pts):
    from tests.test_detail_discovery import _report_cameras

    report = _report(pts)
    return discover_detail(_result(pts, _report_cameras()), report)


class TestRoiContract:
    def test_flat_only_scene_generates_no_rois(self):
        cands = _discover(_plane_patch(0.0, 0.0))
        assert cands  # discovery ran
        assert all(not c.is_detail for c in cands)
        assert generate_rois(cands, voxel_size=1.0) == []

    def test_curved_cell_becomes_pending_roi(self):
        pts = _shell(0.0, 0.0)
        cands = _discover(pts)
        rois = generate_rois(cands, voxel_size=1.0)
        assert len(rois) == 1
        roi = rois[0]
        assert roi.status == PROCESSING_PENDING
        assert roi.parent_entity_id is None
        # Bounds are the cell's voxel bounds (tight, not fabricated).
        assert roi.bounds == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert roi.n_points == len(pts)
        assert roi.max_curvature > 0.1
        # Provenance names the producing stage.
        assert "discover_detail" in roi.provenance["discovery"]

    def test_adjacent_cells_merge_with_weakest_budget(self):
        # Two adjacent curved cells with MIXED evidence: cell A's
        # points multi-view, cell B's single-view. The ROI budget
        # must be bounded by the weaker cell (L2 cap), never by the
        # stronger one (L4) -- the group is only as good as its
        # weakest evidence.
        pts = _shell(0.0, 0.0) + _shell(1.0, 0.0)
        cands = _discover(pts)
        assert any(c.is_detail for c in cands)

        # Rebuild the report with per-track mixed view counts: multi
        # for the first shell's tracks, single for the second's.
        from perception.quality.assessment import EvidenceQualityReport

        report = _report(pts)
        multi_ids = {p.track_id for p in _shell(0.0, 0.0)}
        mixed_views = {
            tid: (3 if tid in multi_ids else 1)
            for tid in report.view_counts
        }
        mixed_report = EvidenceQualityReport(
            gsd_mm_per_px=1.0,  # fine (<=5) so L4 is gsd-justified
            detail_tier="fine",
            observed_fraction=1.0,
            view_counts=mixed_views,
            unprojectable_point_ids=(),
            overclaim_count=0,
        )
        from tests.test_detail_discovery import _report_cameras

        cands_mixed = discover_detail(
            _result(pts, _report_cameras()), mixed_report
        )
        levels = {c.budget.justified_level for c in cands_mixed}
        assert "L4" in levels and "L2" in levels  # mixed as constructed

        rois = generate_rois(cands_mixed, voxel_size=1.0)
        assert len(rois) == 1
        roi = rois[0]
        assert roi.budget.justified_level == "L2"
        assert roi.budget.coverage_capped is True

    def test_deterministic_and_stable_order(self):
        pts = _shell(0.0, 0.0) + _shell(3.0, 3.0)
        cands = _discover(pts)
        r1 = generate_rois(list(reversed(cands)), voxel_size=1.0)
        r2 = generate_rois(cands, voxel_size=1.0)
        assert [r.to_dict() for r in r1] == [r.to_dict() for r in r2]
        assert [r.roi_id for r in r1] == sorted(r.roi_id for r in r1)
