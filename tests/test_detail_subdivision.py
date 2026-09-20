"""Adaptive subdivision (REAL_RECONSTRUCTION_PERCEPTION_CITY_READY
increment C; advances ledger P7-05/P7-06): octree-cell subdivision
driven by a MEASURED detail score — never by fixed per-benchmark
thresholds — with bounded work and measured stopping criteria.

Directive rule under test (mission §4):
  D = w1*curvature + w2*evidence_density (+ optional w3*importance)
  subdivide ONLY when D > threshold AND the evidence supports it
  (enough points to measure children), stopping at measured criteria:
  flat/below-threshold cells are leaves, depth and total cell count
  are hard bounds, and every stop carries its machine-readable reason.

Measured-input discipline:
  - curvature is the discovery module's PCA smallest-eigenvalue ratio
    (plane ~0, curved shell high) — one shared eigensolver, no new
    geometry math.
  - density is measured points per cell volume, normalized by the
    scene's reference spacing (also measured), not a magic constant.
  - scores live in [0, 1] by construction; nothing is fabricated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engine.math import Vec3  # noqa: E402


def _plane_points(n: int = 800) -> list:
    rng = np.random.default_rng(7)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, 1e-4, size=(n, 1))  # essentially flat
    return [Vec3(*map(float, p)) for p in np.hstack([xy, z])]


def _plane_plus_sphere_points(n_plane: int = 500, n_sphere: int = 800) -> list:
    rng = np.random.default_rng(11)
    xy = rng.uniform(-1.0, 0.0, size=(n_plane, 2))  # left half: flat
    plane = np.hstack([xy, np.zeros((n_plane, 1))])
    # Right half: curved shell (sphere section, r=0.5) — high curvature.
    dirs = rng.normal(size=(n_sphere, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sphere = 0.5 * dirs + np.array([0.5, 0.0, 0.0])
    pts = np.vstack([plane, sphere])
    return [Vec3(*map(float, p)) for p in pts]


@pytest.fixture()
def subdivide():
    from perception.detail.subdivision import subdivide_adaptive

    return subdivide_adaptive


class TestFlatRegions:
    def test_flat_plane_is_a_leaf_with_reason(self, subdivide):
        report = subdivide(_plane_points(), score_threshold=0.05)
        assert report.cells and report.cells[0].subdivided is False
        assert report.cells[0].stop_reason == "below_threshold"
        assert report.max_depth_used == 0
        assert report.cells[0].detail_score < 0.05  # measured plane ~0 curvature

    def test_scores_are_measured_in_unit_range(self, subdivide):
        report = subdivide(_plane_plus_sphere_points(), score_threshold=0.05)
        assert all(0.0 <= c.detail_score <= 1.0 for c in report.cells)


class TestStructuredRegions:
    def test_curved_region_subdivides_deeper_than_flat(self, subdivide):
        report = subdivide(_plane_plus_sphere_points(), score_threshold=0.05, max_depth=3)
        assert report.max_depth_used >= 1, "curved shell must justify subdivision"
        leaves = [c for c in report.cells if not c.subdivided]
        # The flat half's cells stop as below_threshold leaves at shallow
        # depth; some curved-region leaf must sit deeper.
        deep_leaves = [c for c in leaves if c.depth >= 2]
        assert deep_leaves, "curved evidence must produce deeper leaves"
        flat_leaves = [c for c in leaves if c.stop_reason == "below_threshold"]
        # The flat half stops immediately at shallow depth; deeper
        # below-threshold leaves can also exist where measured curvature
        # drops as cells shrink — that is measured behavior, not a defect.
        assert any(c.depth <= 1 for c in flat_leaves)

    def test_hierarchical_ids_are_stable_and_deterministic(self, subdivide):
        r1 = subdivide(_plane_plus_sphere_points(), score_threshold=0.05, max_depth=3)
        r2 = subdivide(_plane_plus_sphere_points(), score_threshold=0.05, max_depth=3)
        assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(
            r2.to_dict(), sort_keys=True
        )
        ids = [c.cell_id for c in r1.cells]
        assert len(ids) == len(set(ids)), "cell ids must be unique"
        for c in r1.cells:
            if c.depth > 0:
                parent = c.cell_id.rsplit("/", 1)[0]
                assert any(p.cell_id == parent for p in r1.cells), (
                    "every child's parent must appear in the report"
                )

    def test_min_points_gate_stops_measurement_free_cells(self, subdivide):
        # A threshold so low that everything qualifies, but cells too
        # small to measure children honestly must stop at min_points.
        report = subdivide(
            _plane_plus_sphere_points(),
            score_threshold=0.0,
            max_depth=6,
            min_points_per_cell=400,  # more than any octree leaf will hold
        )
        stops = [c for c in report.cells if c.stop_reason == "min_points"]
        assert stops, "small cells must refuse to subdivide on insufficient support"
        assert all(c.subdivided is False for c in stops)


class TestBoundedWork:
    def test_max_cells_budget_is_honored_and_recorded(self, subdivide):
        report = subdivide(
            _plane_plus_sphere_points(),
            score_threshold=0.05,
            max_depth=4,
            max_cells=7,
        )
        assert len(report.cells) <= 7
        assert report.budget_exhausted is True
        assert report.stopping_reason == "max_cells"
        assert report.work_units_spent <= 7

    def test_unbounded_case_terminates_on_leaves(self, subdivide):
        report = subdivide(
            _plane_points(), score_threshold=0.05, max_depth=4, max_cells=10_000
        )
        assert report.budget_exhausted is False
        assert report.stopping_reason == "all_leaves"

    def test_max_depth_is_respected(self, subdivide):
        report = subdivide(
            _plane_plus_sphere_points(), score_threshold=0.05, max_depth=1
        )
        assert all(c.depth <= 1 for c in report.cells)
        assert report.max_depth_used <= 1


class TestRealDataCalibration:
    def test_south_building_sparse_points_produce_measured_report(self):
        from perception.detail.subdivision import subdivide_adaptive
        """REAL evidence: the committed south-building sparse reference
        (datasets/south_building/sparse/points3D.txt) must run through
        the same subdivision and produce a fully measured report — no
        special-cased thresholds, no fabricated cells."""
        points_file = REPO / "datasets" / "south_building" / "sparse" / "points3D.txt"
        if not points_file.is_file():
            pytest.skip("committed south-building dataset missing")
        pts = []
        with open(points_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                pts.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
        assert len(pts) > 1000  # REAL sparse cloud

        report = subdivide_adaptive(
            pts, score_threshold=0.05, max_depth=3, max_cells=4096
        )
        assert report.cells
        assert report.work_units_spent <= 4096
        blob = json.dumps(report.to_dict())
        assert "score" in blob and "stop_reason" in blob
        # Every score is a measured value in [0, 1]; no NaN/inf leaks.
        for c in report.cells:
            assert np.isfinite(c.detail_score)
            assert 0.0 <= c.detail_score <= 1.0
