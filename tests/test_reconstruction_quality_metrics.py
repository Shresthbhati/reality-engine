"""One test per reconstruction quality metric, plus the registry contract (formula / inputs / limitations)."""

import numpy as np

from reconstruction import quality_metrics as qm
from world_ir.geometry_data import MeshData


def test_every_metric_documents_formula_inputs_limitations():
    assert len(qm.METRICS) == 10
    for spec in qm.METRICS.values():
        assert spec.formula and spec.inputs and spec.limitations, spec.name


def test_registration_rate():
    assert qm.registration_rate(24, 32)["value"] == 0.75
    assert qm.registration_rate(1, 0)["value"] is None
    assert qm.registration_rate(5, 4)["value"] is None


def test_reprojection_error_stats_and_refusal():
    r = qm.reprojection_error([0.5, 1.0, 1.5, float("nan")])
    assert r["median_px"] == 1.0 and r["n"] == 3
    assert qm.reprojection_error([])["value"] is None


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_pose_graph_cycle_error_detects_inconsistent_triangle():
    nodes = ["a", "b", "c"]
    good = [("a", "b", _rot(10)), ("b", "c", _rot(20)), ("a", "c", _rot(30))]
    assert qm.pose_graph_cycle_error(nodes, good)["max_deg"] < 1e-6
    bad = [("a", "b", _rot(10)), ("b", "c", _rot(20)), ("a", "c", _rot(45))]
    assert abs(qm.pose_graph_cycle_error(nodes, bad)["max_deg"] - 15.0) < 1e-3
    assert qm.pose_graph_cycle_error(nodes, good[:2])["value"] is None  # no closed triple -> refused


def test_scale_validity_never_allows_metres_without_validation():
    assert qm.scale_validity("metric_anchored", "validated")["metres_claim_allowed"] is True
    assert qm.scale_validity("metric_anchored", "unvalidated")["metres_claim_allowed"] is False
    assert qm.scale_validity("relative", None)["value"] == "relative"
    assert qm.scale_validity("sfm_gauge", "estimated_prior")["metres_claim_allowed"] is False
    assert qm.scale_validity(None, None)["value"] == "unknown"


def test_point_cloud_completeness_requires_same_frame_and_measures_recall():
    ref = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    est = np.array([[0, 0.01, 0], [1, 0, 0.01]], float)
    assert qm.point_cloud_completeness(est, ref, 0.05)["value"] is None            # frame not declared
    r = qm.point_cloud_completeness(est, ref, 0.05, same_frame=True)
    assert r["recall"] == 0.5 and r["precision"] == 1.0
    junk = np.vstack([est, np.random.default_rng(0).uniform(10, 20, (200, 3))])    # more points != more complete
    assert qm.point_cloud_completeness(junk, ref, 0.05, same_frame=True)["recall"] == 0.5


def test_depth_consistency_consistent_inconsistent_and_occluded():
    K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
    T = np.eye(4)
    flat = np.full((64, 64), 2.0)
    assert qm.depth_consistency(flat, flat, K, T)["value"] == 1.0
    assert qm.depth_consistency(flat, np.full((64, 64), 3.0), K, T)["value"] == 0.0   # B sees farther: contradiction
    r = qm.depth_consistency(flat, np.full((64, 64), 1.0), K, T)                        # B sees nearer: occlusion
    assert r["value"] is None and "occluded" in r["refused"]
    assert qm.depth_consistency(np.zeros((8, 8)), flat[:8, :8], K, T)["value"] is None


def _cube(flip_one=False, duplicate=False):
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)]
    if flip_one:
        f[0] = (0, 1, 2)
    if duplicate:
        f.append(f[3])
    return MeshData(v, f)


def test_mesh_integrity_clean_cube_then_each_defect():
    clean = qm.mesh_integrity(_cube())
    assert clean["valid"] and clean["n_boundary_edges"] == 0 and clean["n_winding_conflicts"] == 0
    assert qm.mesh_integrity(_cube(flip_one=True))["n_winding_conflicts"] > 0
    assert qm.mesh_integrity(_cube(duplicate=True))["n_duplicate_triangles"] == 1
    assert qm.mesh_integrity(MeshData([], []))["value"] is None


def test_surface_coverage_partial_reference():
    mesh = _cube()
    on_surface = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0], [0.0, 0.5, 0.5]])
    off = np.array([[5.0, 5.0, 5.0]])
    assert qm.surface_coverage(mesh, on_surface, 0.05)["value"] is None
    ok = qm.surface_coverage(mesh, np.vstack([on_surface, off]), 0.05, same_frame=True, n_samples=40000)
    assert abs(ok["value"] - 0.75) < 1e-9


def test_uncertainty_calibration_over_and_well_calibrated():
    rng = np.random.default_rng(1)
    sig = np.full(5000, 1.0)
    good = qm.uncertainty_calibration(rng.normal(0, 1.0, 5000), sig)
    assert 0.93 < good["coverage_95"] < 0.97 and good["verdict"] == "calibrated"
    assert qm.uncertainty_calibration(rng.normal(0, 3.0, 5000), sig)["verdict"] == "overconfident"
    assert qm.uncertainty_calibration([1.0], [0.0])["value"] is None


def test_cross_view_consistency_ignores_single_view_tracks():
    r = qm.cross_view_consistency([[0.5, 0.7], [0.4, 5.0], [0.3]], tol_px=2.0)
    assert r["n_tracks"] == 2 and r["value"] == 0.5
    assert qm.cross_view_consistency([[1.0]])["value"] is None
