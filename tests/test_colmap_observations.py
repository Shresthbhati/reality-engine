"""Per-image observations re-expressed in the job frame: exact under a true similarity, refused when the frames do not match.
Synthetic geometry (proves the mapping and the refusal rules, not any reconstruction)."""

from __future__ import annotations

import numpy as np

from reconstruction.colmap_observations import observations_in_job_frame
from reconstruction.similarity import umeyama

RNG = np.random.default_rng(1)
CENTRES = RNG.uniform(-1, 1, (6, 3))
POINTS = RNG.uniform(-2, 2, (40, 3))


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _model():
    return {f"ev{i}": {"centre": c, "xy": RNG.uniform(0, 100, (40, 2)), "xyz": POINTS} for i, c in enumerate(CENTRES)}


def test_points_follow_the_same_similarity_as_the_camera_centres():
    s, R, t = 0.2, _rot_z(33), np.array([1.0, -2.0, 0.5])
    job = {f"ev{i}": s * R @ c + t for i, c in enumerate(CENTRES)}
    obs, facts = observations_in_job_frame(_model(), job)
    assert facts["status"] == "ok" and abs(facts["scale_raw_to_job"] - s) < 1e-9 and facts["max_centre_residual"] < 1e-9
    assert np.allclose(obs["ev3"]["xyz"], s * POINTS @ R.T + t)  # points and cameras moved together
    assert obs["ev3"]["xy"].shape == (40, 2)  # pixels are untouched


def test_inconsistent_frames_are_refused_with_the_reason_not_guessed():
    job = {f"ev{i}": c for i, c in enumerate(CENTRES)}
    job["ev2"] = job["ev2"] + np.array([3.0, 0, 0])  # one camera in the wrong place: not a similarity
    obs, facts = observations_in_job_frame(_model(), job)
    assert obs == {} and facts["status"] == "refused" and "residual" in facts["reason"]


def test_too_few_common_cameras_is_refused():
    obs, facts = observations_in_job_frame(_model(), {"ev0": CENTRES[0], "ev1": CENTRES[1], "other": CENTRES[2]})
    assert obs == {} and "fewer than 4" in facts["reason"]


def test_umeyama_never_returns_a_reflection():
    s, R, t = umeyama(CENTRES, CENTRES * np.array([1, 1, -1]))
    assert np.linalg.det(R) > 0
