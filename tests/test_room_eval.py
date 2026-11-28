"""The room benchmark's scorer must give ~zero error to a perfect reconstruction in an arbitrary gauge, and must
punish a wrong one -- otherwise the numbers in docs/BENCHMARKS.md mean nothing."""
import numpy as np

from benchmarks import room_eval as ev
from synthetic.room_photos import RoomSpec, look_at, orbit_path

ROOM = {"width": 5.0, "depth": 4.0, "height": 2.6}


def _gauge(seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return 0.37, q, rng.normal(size=3) * 3  # est = (gt - t) / s rotated back; we build est from gt below


def _to_est(p, s, R, t):  # inverse of gt = s R est + t
    return ((p - t) / s) @ R


def _face_pts(n=4000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    W, D, H = ROOM["width"], ROOM["depth"], ROOM["height"]
    pts = []
    for axis, off, (a, b) in ((2, 0, (W, D)), (2, H, (W, D)), (1, 0, (W, H)), (1, D, (W, H)), (0, 0, (D, H)), (0, W, (D, H))):
        u = rng.uniform(0, 1, (n, 2)) * np.array(a if axis == 2 else a, float)
        p = np.zeros((n, 3))
        others = [i for i in range(3) if i != axis]
        p[:, others[0]] = rng.uniform(0, [W, D, H][others[0]], n)
        p[:, others[1]] = rng.uniform(0, [W, D, H][others[1]], n)
        p[:, axis] = off
        pts.append(p)
    return np.concatenate(pts) + rng.normal(0, noise, (6 * n, 3))


def test_perfect_reconstruction_scores_near_zero_in_any_gauge():
    s, R, t = _gauge()
    path = orbit_path(RoomSpec(), 20)
    gt_c = np.array([c for _, c in path])
    gt_R = np.array([r for r, _ in path])
    est_c = _to_est(gt_c, s, R, t)
    est_R = np.array([R.T @ r for r in gt_R])  # R_gt = R @ R_est
    diag = float(np.sqrt(5.0**2 + 4.0**2 + 2.6**2))
    cam, sim = ev.score_cameras(est_c, est_R, gt_c, gt_R, diag)
    assert cam["position_rmse_frac_of_room_diagonal"] < 1e-9
    assert cam["rotation_median_deg"] < 1e-6
    pts = _to_est(_face_pts(), *sim_inverse(sim))
    sc = ev.score_points(pts, sim, ROOM)
    assert sc["frac_within_1pct_diag"] > 0.99
    assert abs(sc["extent_ratio_W_over_D"] - 1.25) < 0.02 and abs(sc["extent_ratio_H_over_D"] - 0.65) < 0.02
    planes = [_to_est(_face_pts(800, seed=i)[i * 800:(i + 1) * 800], *sim_inverse(sim)) for i in range(6)]
    pl = ev.score_planes(planes, sim, ROOM)
    assert pl["n_true_faces_recovered_of_6"] == 6 and pl["n_spurious"] == 0
    assert pl["recovered_face_pair_angle_error_deg"]["max"] < 1e-3
    assert abs(pl["plane_span_ratio_W_over_D"] - 1.25) < 1e-6 and abs(pl["plane_span_ratio_H_over_D"] - 0.65) < 1e-6


def sim_inverse(sim):
    """sim maps est->gt; return (s, R, t) of the gt = s R est + t form for _to_est (which inverts it)."""
    return sim


def test_wrong_shape_is_punished():
    s, R, t = _gauge()
    sim = (s, R, t)
    pts = _to_est(_face_pts(), s, R, t)
    squashed = pts.copy()
    squashed[:, 0] *= 0.7  # a room 30% too narrow
    sc = ev.score_points(squashed, sim, ROOM)
    assert sc["frac_within_1pct_diag"] < 0.9
    pl = ev.score_planes([_to_est(_face_pts(800)[i * 800:(i + 1) * 800], s, R, t) * np.array([0.7, 1, 1]) for i in range(6)], sim, ROOM)
    assert pl["n_true_faces_recovered_of_6"] < 6


def test_renderer_ground_truth_is_consistent():
    """The camera convention the renderer reports is OpenCV: z forward, y down, right-handed, looking inward."""
    R = look_at([1, 1, 1], [3, 1, 1])
    assert np.allclose(R[:, 2], [1, 0, 0]) and np.allclose(R[:, 1], [0, 0, -1])
    assert np.isclose(np.linalg.det(R), 1.0)
