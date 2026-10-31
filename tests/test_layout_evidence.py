"""F4: single-image layout evidence -- parsed, projected through the real camera pose, and fused as weaker wall support.

The learned estimator is NOT bundled (see layout_evidence.estimator_status); these tests cover the honest half that IS
implemented: sidecar validation, metric projection onto the floor plane, and the fusion rule in solve_room_layout.
"""

import json

import numpy as np
import pytest

from perception.architecture.layout_evidence import (
    LAYOUT_MODEL_STATUS,
    LayoutEvidenceError,
    edges_from_hypotheses,
    estimator_status,
    layout_support,
    load_layout_sidecars,
    parse_layout_hypothesis,
    project_floor_edges,
)
from perception.architecture.room_layout import solve_room_layout
from synthetic.rgbd import look_at


class _Pose:
    def __init__(self, position, rotation, evidence_id="cam"):
        self.evidence_id, self.position, self.rotation = evidence_id, position, rotation


def _look_down_pose(y=2.0):
    p = look_at((0.0, y, 0.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    return _Pose(p.t, p.quat_wxyz())


def _horizontal_pose(y=2.0):
    p = look_at((0.0, y, 5.0), (0.0, y, 0.0), up=(0.0, 1.0, 0.0))
    return _Pose(p.t, p.quat_wxyz())


def test_estimator_is_reported_unavailable_with_a_reason_never_faked():
    status = estimator_status()
    assert status["status"] == "unavailable" and "HorizonNet" in status["reason"]
    assert status["unblocked_by"] and status == LAYOUT_MODEL_STATUS


def test_parse_accepts_the_documented_format_and_rejects_bad_geometry():
    hyp = parse_layout_hypothesis({"source": "horizonnet", "wall_floor": [[0.1, 0.8, 0.5, 0.7]], "confidence": 0.6},
                                  evidence_id="img_000")
    assert hyp.evidence_id == "img_000" and hyp.wall_floor == ((0.1, 0.8, 0.5, 0.7),) and hyp.confidence == 0.6
    with pytest.raises(LayoutEvidenceError, match="wall_floor"):
        parse_layout_hypothesis({"source": "x", "wall_floor": []}, evidence_id="e")
    with pytest.raises(LayoutEvidenceError, match="outside"):
        parse_layout_hypothesis({"wall_floor": [[0.0, 0.0, 1.7, 0.5]]}, evidence_id="e")
    with pytest.raises(LayoutEvidenceError, match="confidence"):
        parse_layout_hypothesis({"wall_floor": [[0.0, 0.0, 1.0, 0.5]], "confidence": 1.5}, evidence_id="e")


def test_sidecars_are_loaded_per_stem_and_bad_files_are_reported(tmp_path):
    (tmp_path / "layout").mkdir()
    (tmp_path / "layout" / "img_000.json").write_text(json.dumps({"wall_floor": [[0.0, 0.8, 1.0, 0.8]]}))
    (tmp_path / "layout" / "img_001.json").write_text("{not json")
    found, errors = load_layout_sidecars([tmp_path])
    assert set(found) == {"img_000"} and len(errors) == 1 and "img_001" in errors[0]


def test_projection_lands_on_the_floor_plane_in_front_of_the_camera():
    pose = _look_down_pose(2.0)
    intr = (400.0, 400.0, 320.0, 240.0)
    edges, errors = project_floor_edges(pose, [(0.2, 0.75, 0.8, 0.75)], intr, floor_y=0.0, image_size=(640, 480))
    assert not errors and len(edges) == 1
    # camera 2 m above the floor, looking straight down: v=0.75 is 0.3 * 2.0 m off-axis, u maps to the x axis
    (x0, z0), (x1, z1) = edges[0]
    assert z0 == pytest.approx(-0.6, abs=1e-6) and z1 == pytest.approx(-0.6, abs=1e-6)
    assert x0 == pytest.approx(0.96, abs=1e-6) and x1 == pytest.approx(-0.96, abs=1e-6)


def test_a_boundary_that_does_not_reach_the_floor_is_dropped_not_clamped():
    # a level camera at 2 m: a boundary above the horizon ray points up, so it never reaches the floor below
    pose = _horizontal_pose(2.0)
    intr = (400.0, 400.0, 320.0, 240.0)
    edges, errors = project_floor_edges(pose, [(0.0, 0.2, 1.0, 0.2)], intr, floor_y=0.0)
    assert edges == [] and errors and "does not reach the floor" in errors[0]
    below, errs = project_floor_edges(pose, [(0.0, 0.8, 1.0, 0.8)], intr, floor_y=0.0)
    assert len(below) == 1 and not errs


def test_layout_support_measures_the_covered_share_of_a_wall():
    edges = [((0.0, 0.0), (5.0, 0.0))]
    assert layout_support((0.0, 0.0), (5.0, 0.0), edges) == pytest.approx(1.0)
    assert layout_support((0.0, 2.0), (5.0, 2.0), edges) == pytest.approx(0.0)
    assert layout_support((0.0, 0.0), (5.0, 0.0), []) is None


def test_edges_from_hypotheses_skips_unposed_images():
    result = type("R", (), {"camera_poses": [_look_down_pose(2.0)]})()
    from perception.architecture.layout_evidence import LayoutHypothesis
    hyps = {"cam": LayoutHypothesis("cam", "x", ((0.2, 0.75, 0.8, 0.75),)),
            "ghost": LayoutHypothesis("ghost", "x", ((0.2, 0.75, 0.8, 0.75),))}
    edges, facts = edges_from_hypotheses(result, hyps, (400.0, 400.0, 320.0, 240.0), 0.0)
    assert facts["status"] == "ran" and facts["views_used"] == ["cam"] and facts["views_skipped"] == ["ghost"]
    assert edges and all(len(pt) == 2 for edge in edges for pt in edge)      # plan (x, z), floor projected


# ------------------------------------------------------------------ fusion into the S2 layout solver

_SQUARE = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]


def _wall(lay, ax, ay, bx, by):
    """The solved wall matching an expected edge, within regularisation round-off (corners are snapped)."""
    def same(w):
        ends = [w.a, w.b]
        return (any(np.hypot(p[0] - ax, p[1] - ay) < 1e-6 for p in ends)
                and any(np.hypot(p[0] - bx, p[1] - by) < 1e-6 for p in ends))
    return next(w for w in lay.walls if same(w))


def test_a_wall_no_point_saw_becomes_observed_when_a_layout_boundary_supports_it():
    north_edge = [((0.0, 4.0), (5.0, 4.0))]
    without = solve_room_layout(_SQUARE, [2.6] * 4, None)
    wall = _wall(without, 0.0, 4.0, 5.0, 4.0)
    assert wall.provenance == "INFERRED" and wall.support_basis == "none"

    with_layout = solve_room_layout(_SQUARE, [2.6] * 4, None, layout_edges_ab=north_edge)
    wall = _wall(with_layout, 0.0, 4.0, 5.0, 4.0)
    assert wall.provenance == "OBSERVED" and wall.support_basis == "layout_image"
    assert wall.layout_support_fraction == pytest.approx(1.0)
    assert 0.2 < wall.confidence <= 0.45, "a learned line must stay weaker than point support"
    other = _wall(with_layout, 0.0, 0.0, 5.0, 0.0)
    assert other.provenance == "INFERRED", "layout evidence on the north wall must not upgrade the south wall"


def test_points_still_outrank_layout_evidence_on_the_same_wall():
    pts = np.array([[x, 4.0] for x in np.linspace(0.05, 4.95, 60)])
    lay = solve_room_layout(_SQUARE, [2.6] * 4, pts, layout_edges_ab=[((0.0, 4.0), (5.0, 4.0))])
    wall = _wall(lay, 0.0, 4.0, 5.0, 4.0)
    assert wall.provenance == "OBSERVED" and wall.support_basis == "points" and wall.confidence >= 0.9
