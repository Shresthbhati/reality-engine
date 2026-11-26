"""A4.3, A4.4 & A4.5 tests: architectural perception 2.0, architectural relationship reasoning, and furniture 2.0."""
from __future__ import annotations

import numpy as np

from perception.intelligence.architectural_relations import reason_architectural_relationships
from perception.intelligence.architectural_v2 import (
    analyze_architectural_geometry_v2,
    detect_corner_candidates,
    estimate_vanishing_hypotheses,
    extract_image_line_segments,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
)
from perception.intelligence.furniture_v2 import analyze_furniture_v2, classify_object_mobility


def _make_obs(
    obs_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    category: str = "furniture",
    image_id: str = "room_01",
    confidence: float = 0.85,
    width: int = 800,
    height: int = 600,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec(
            frame_id="image_px",
            units="pixels",
            image_width=width,
            image_height=height,
        ),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(
                engine_id="test_engine",
                architecture_family="vlm",
                checkpoint_id="test_model_v1",
            ),
        ),
    )


def _synthetic_room_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a high-contrast synthetic room wireframe RGB image with clear wall, floor, and corner lines."""
    img = np.full((height, width, 3), 230, dtype=np.uint8)
    img[60:400, 120:124] = 20
    img[60:400, 520:524] = 20
    img[80:360, 320:324] = 20
    img[60:64, 120:524] = 20
    img[360:364, 120:524] = 20
    for t in range(120):
        y = 360 + t
        x_l = 120 - t
        x_r = 520 + t
        if 0 <= y < height and 0 <= x_l < width:
            img[y, x_l : min(width, x_l + 3)] = 20
        if 0 <= y < height and 0 <= x_r < width:
            img[y, x_r : min(width, x_r + 3)] = 20
    return img


def test_extract_line_segments_corners_and_vanishing_hypotheses() -> None:
    img = _synthetic_room_image()
    segments = extract_image_line_segments(img, min_length_px=30.0, max_segments=80)
    assert len(segments) >= 3

    corners = detect_corner_candidates(segments, proximity_px=25.0)
    assert len(corners) >= 1

    vanishing = estimate_vanishing_hypotheses(segments)
    assert isinstance(vanishing, list)
    assert len(vanishing) >= 1


def test_analyze_architectural_geometry_v2_outputs_planes_and_non_metric_flag() -> None:
    img = _synthetic_room_image(800, 600)
    obs = [
        _make_obs("w1", "wall", (0, 40, 380, 480), category="architecture"),
        _make_obs("w2", "wall", (380, 40, 800, 480), category="architecture"),
        _make_obs("f1", "floor", (0, 460, 800, 600), category="architecture"),
        _make_obs("c1", "ceiling", (0, 0, 800, 60), category="architecture"),
        _make_obs("d1", "door", (140, 120, 240, 460), category="architecture"),
        _make_obs("win1", "window", (480, 100, 640, 280), category="architecture"),
        _make_obs("cab1", "cabinet", (400, 240, 560, 460), category="furniture"),
    ]
    res = analyze_architectural_geometry_v2("room_01", img, obs)
    assert res.coordinate_frame == "image_space"
    assert len(res.plane_hypotheses) >= 3
    assert all(p.is_metric is False for p in res.plane_hypotheses)
    assert "cab1" in res.builtin_cabinet_ids


def test_architectural_relationships_detect_containment_junctions_and_contradictions() -> None:
    obs = [
        _make_obs("wall_l", "wall", (0, 50, 400, 480), category="architecture"),
        _make_obs("wall_r", "wall", (395, 50, 800, 480), category="architecture"),
        _make_obs("door_1", "door", (100, 140, 210, 470), category="architecture"),
        _make_obs("win_1", "window", (500, 100, 650, 260), category="architecture"),
        _make_obs("floor_1", "floor", (0, 465, 800, 600), category="architecture"),
        _make_obs("ceil_1", "ceiling", (0, 0, 800, 60), category="architecture"),
        _make_obs("col_1", "column", (380, 50, 420, 485), category="architecture"),
    ]
    graph = reason_architectural_relationships(obs)
    rel_types = {r.predicate for r in graph.relationships}
    assert "door_contained_in_wall" in rel_types
    assert "window_contained_in_wall" in rel_types
    assert "floor_meets_wall" in rel_types
    assert "ceiling_meets_wall" in rel_types
    assert len(graph.contradictions) == 0

    bad_obs = [
        _make_obs("floor_bad", "floor", (0, 10, 800, 120), category="architecture"),
        _make_obs("ceil_bad", "ceiling", (0, 450, 800, 590), category="architecture"),
    ]
    bad_graph = reason_architectural_relationships(bad_obs)
    assert len(bad_graph.contradictions) >= 1


def test_furniture_mobility_classification() -> None:
    assert classify_object_mobility("chair") == "movable_furniture"
    assert classify_object_mobility("cabinet", is_wall_contained=True) == "built_in_furniture"
    assert classify_object_mobility("refrigerator") == "appliance"
    assert classify_object_mobility("sink") == "fixture"


def test_analyze_furniture_v2_extracts_arrangements_support_and_occlusion() -> None:
    obs = [
        _make_obs("floor_1", "floor", (0, 420, 800, 600), category="architecture"),
        _make_obs("table_1", "table", (220, 300, 540, 480)),
        _make_obs("chair_1", "chair", (150, 310, 250, 470)),
        _make_obs("chair_2", "chair", (510, 310, 610, 470)),
        _make_obs("monitor_1", "monitor", (330, 210, 430, 315)),
        _make_obs("chair_front", "chair", (340, 330, 440, 485)),
        _make_obs("cab_builtin", "cabinet", (20, 150, 120, 440)),
    ]
    res = analyze_furniture_v2(obs, builtin_ids=("cab_builtin",))
    assert res.mobility_assignments["cab_builtin"] == "built_in_furniture"
    assert res.mobility_assignments["chair_1"] == "movable_furniture"
    assert len(res.arrangements) >= 1
    assert len(res.support_links) >= 1
    assert res.summary["classified_objects"] >= 5
