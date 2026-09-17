"""Structure-aware discovery/ROI tests (the real-data defect fix).

The real room capture (294,345-point fused.ply, the P6-01 artifact)
exposed a discovery defect the synthetic fixtures could not: the only
structural signal was the PCA curvature ratio, which a PLANE scores ~0
BY CONSTRUCTION. Every planar-but-ORIENTED surface (walls, floors,
ceilings -- a room's entire structure) was invisible to discovery, and
the real run produced exactly ONE ROI: the room's planar structure
dead-ended.

Fix under test (minimal, backend-level, wired through the existing
chain):
  - Discovery MEASURES planarity per cell (1 - lambda0/lambda2 from
    the SAME covariance as curvature -- no second pass): a planar
    cell scores ~1 whether it is oriented or not, while noise,
    blobs, and curved shells score lower. Recorded as `planarity` on
    every DetailCandidate.
  - ROIs may seed from structure cells (planarity >= threshold) as
    well as detail cells -- via an explicit opt-in flag with a
    documented default that preserves the existing detail-only
    contract (existing tests stay byte-identical).
  - Nothing about the detail path changes: curvature gate, budgets,
    min_points, determinism, and the tier mapping are untouched.
"""

from __future__ import annotations

import math

import pytest


def _pt(xyz, tid, eid="c0"):
    from reconstruction.backend.interface import ReconstructedPoint

    return ReconstructedPoint(
        position=xyz, track_id=tid, source_evidence_ids=[eid]
    )


class TestPlanarityMeasurement:
    def test_candidate_carries_measured_planarity(self):
        from perception.detail.discovery import discover_detail
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = [
            _pt((x * 0.1, y * 0.1, 2.0), f"p{x}-{y}")
            for x in range(6)
            for y in range(6)
        ]
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        assert cands
        # A plane scores ~1 planarity regardless of orientation, and
        # ~0 curvature -- the two signals are complementary.
        assert all(c.planarity > 0.95 for c in cands)
        assert all(c.curvature < 0.05 for c in cands)

    def test_planarity_and_curvature_complementary(self):
        from perception.detail.discovery import discover_detail
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        # Curved shell: planarity drops, curvature rises.
        pts = [
            _pt(
                (
                    0.5 + 0.3 * math.cos(a),
                    0.5 + 0.3 * math.sin(a),
                    0.4 + j * 0.1,
                ),
                f"sp-{j}-{k}",
            )
            for j in range(6)
            for k, a in enumerate(
                (k * 2.0 * math.pi / 10 + j * 0.1 for k in range(10))
            )
        ]
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        shell = [c for c in cands if c.curvature > 0.1]
        assert shell
        assert all(c.planarity < 0.95 for c in shell)


class TestStructureSeededRois:
    def _discover(self, pts, **kw):
        # Discovery records the measured is_structure flag on every
        # candidate (measurement is not policy); the ROI layer's
        # include_structure opt-in applies the policy.
        from perception.detail.discovery import discover_detail
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        report = _quality_report(pts)
        return discover_detail(_result(pts, _report_cameras()), report)

    def _wall_pts(self, offset=0.0):
        # A planar wall patch, tilted 30 degrees about z (an ORIENTED
        # plane -- normal is neither axis-aligned nor along up), placed
        # INSIDE the report cameras' frustum (z=1.0: half-width 0.5,
        # half-height 0.375) so the assessor can measure real view
        # support. A fixture outside the frustum is a lie that the
        # budget gate correctly punishes with tier "none".
        tilt = math.radians(30.0)
        pts = []
        for i in range(48):
            x, y = i % 8, i // 8
            px = offset + x * 0.04
            py = y * 0.04
            pts.append(_pt(
                (
                    px * math.cos(tilt) - py * math.sin(tilt) + 0.3,
                    px * math.sin(tilt) + py * math.cos(tilt),
                    1.0,
                ),
                f"w{offset}-{i}",
            ))
        return pts

    def test_structure_roi_seeded_from_oriented_plane(self):
        from perception.detail.roi import generate_rois

        pts = self._wall_pts()
        cands = self._discover(pts, include_structure=True)
        assert cands
        # The wall cell is planar and oriented: it must seed an ROI.
        rois = generate_rois(cands, voxel_size=1.0, include_structure=True)
        assert len(rois) >= 1
        roi = rois[0]
        assert roi.status == "pending"
        # Provenance records WHY this ROI exists: structure, not
        # detail curvature.
        assert roi.provenance.get("seed") == "structure"
        assert roi.provenance.get("planarity") >= 0.95

    def test_default_contract_unchanged(self):
        from perception.detail.roi import generate_rois

        pts = self._wall_pts()
        cands = self._discover(pts)
        # Default (include_structure=False): the oriented plane does
        # NOT seed an ROI -- existing behavior byte-identical.
        rois = generate_rois(cands, voxel_size=1.0)
        assert rois == []

    def test_noise_never_seeds_structure(self):
        from perception.detail.roi import generate_rois

        # Isotropic blob: neither planar nor curved-beyond-threshold
        # for the structure gate.
        n = 40
        pts = []
        for k in range(n):
            a = k * 2.399963
            z = 1.0 - 2.0 * k / (n - 1)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            pts.append(_pt(
                (0.5 + r * math.cos(a), 0.5 + r * math.sin(a), 0.5 + z),
                f"blob-{k}",
            ))
        cands = self._discover(pts, include_structure=True)
        rois = generate_rois(cands, voxel_size=1.0, include_structure=True)
        assert rois == []


class TestExecutorConsumesStructureRoi:
    def test_structure_roi_refines_to_plane(self):
        from perception.detail.refinement import refine_rois
        from perception.detail.roi import generate_rois

        tilt = math.radians(30.0)
        pts = []
        for i in range(48):
            x, y = i % 8, i // 8
            px = x * 0.04
            py = y * 0.04
            pts.append(_pt(
                (
                    px * math.cos(tilt) - py * math.sin(tilt) + 0.3,
                    px * math.sin(tilt) + py * math.cos(tilt),
                    1.0,
                ),
                f"w-{i}",
            ))
        from perception.detail.discovery import discover_detail
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        rois = generate_rois(cands, voxel_size=1.0, include_structure=True)
        assert rois
        registry = {p.track_id: p.position for p in pts}
        outcomes = refine_rois(
            rois, point_lookup=lambda pid: registry.get(pid)
        )
        assert outcomes[0].status == "refined"
        # The winning backend is the plane -- the true model for an
        # oriented wall patch.
        assert outcomes[0].refinement.backend == "plane"
        assert outcomes[0].refinement.rms_residual_m < 1e-6


class TestPipelineWiring:
    """The canonical one-call driver must expose the structure opt-in
    -- the real room capture discovered 96 structure cells that the
    driver silently dropped because it never passed the flag through."""

    def test_pipeline_passes_include_structure_through(self):
        from perception.detail.pipeline import run_detail_pipeline
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = self._wall_pts()
        result = _result(pts, _report_cameras())
        default = run_detail_pipeline(result, _report_cameras(), voxel_size=1.0)
        structured = run_detail_pipeline(
            result, _report_cameras(), voxel_size=1.0,
            include_structure=True,
        )
        assert default.summary["n_rois"] == 0
        assert structured.summary["n_rois"] >= 1

    @staticmethod
    def _wall_pts():
        tilt = math.radians(30.0)
        pts = []
        for i in range(48):
            x, y = i % 8, i // 8
            px = x * 0.04
            py = y * 0.04
            pts.append(_pt(
                (
                    px * math.cos(tilt) - py * math.sin(tilt) + 0.3,
                    px * math.sin(tilt) + py * math.cos(tilt),
                    1.0,
                ),
                f"pw-{i}",
            ))
        return pts
