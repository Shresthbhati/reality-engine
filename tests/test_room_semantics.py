"""Interior semantics on an L-shaped room with a false ceiling, a window and furniture (deterministic point cloud).

Truth (z up, metres): L footprint area 24.0 (bounding box 30); false ceiling 2.5 over the y<3 arm, structural slab 2.8
over the other; window 2.0 x 1.2 (sill 0.9) in the y=0 wall; table 1.2x0.8x0.75 at (4.2, 1.2); sideboard 2.0x0.9x0.8 at
(1.5, 4.45). Real-image behaviour (COLMAP on rendered views) is covered by scripts/run_lroom_e2e.py.
"""

import math

import numpy as np
import pytest

from provenance import Uncertainty
from perception.architecture.scene import assemble_interior_scene
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult

S = 0.06
POLY = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
SLAB, FALSE = 2.8, 2.5
WINDOW = (1.5, 3.5, 0.9, 2.1)
TABLE = (3.6, 4.8, 0.8, 1.6, 0.75)
BOARD = (0.5, 2.5, 4.0, 4.9, 0.8)


def _grid(a0, a1, b0, b1, s=S):
    return [(a, b) for a in np.arange(a0, a1 + 1e-9, s) for b in np.arange(b0, b1 + 1e-9, s)]


def _in_l(x, y):
    return (0 <= x <= 6 and 0 <= y <= 3) or (0 <= x <= 3 and 3 <= y <= 5)


def build_points(*, furniture=True, false_ceiling=True, window=True):
    pts = []
    for x, y in _grid(0.05, 5.95, 0.05, 4.95):
        if _in_l(x, y):
            pts.append((x, y, 0.0))
            pts.append((x, y, FALSE if (false_ceiling and y < 3) else SLAB))
    for (x0, y0), (x1, y1) in zip(POLY, POLY[1:] + POLY[:1]):
        for t in np.linspace(0, 1, max(2, int(math.hypot(x1 - x0, y1 - y0) / S))):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            top = FALSE if (false_ceiling and y <= 3.0 and not (x <= 3 and y > 3)) else SLAB
            for z in np.arange(0, top + 1e-9, S):
                if window and y0 == y1 == 0 and WINDOW[0] <= x <= WINDOW[1] and WINDOW[2] <= z <= WINDOW[3]:
                    continue
                pts.append((x, y, z))
    if furniture:
        for x0, x1, y0, y1, h in (TABLE, BOARD):
            pts += [(x, y, h) for x, y in _grid(x0, x1, y0, y1, 0.06)]
            for z in np.arange(0.0, h, 0.06):
                pts += [(x, y0, z) for x in np.arange(x0, x1 + 1e-9, 0.06)] + [(x, y1, z) for x in np.arange(x0, x1 + 1e-9, 0.06)]
                pts += [(x0, y, z) for y in np.arange(y0, y1 + 1e-9, 0.06)] + [(x1, y, z) for y in np.arange(y0, y1 + 1e-9, 0.06)]
    return pts


def result_of(pts):
    rp = [ReconstructedPoint(position=tuple(map(float, p)), track_id=f"t{i}", source_evidence_ids=["ev-1"],
                             uncertainty=Uncertainty(confidence=0.9), color=(180, 170, 160)) for i, p in enumerate(pts)]
    poses = [ReconstructedCameraPose(evidence_id=f"ev-{k}", position=(1.5 + k * 0.7, 1.5 + 0.3 * k, 1.5),
                                     rotation=(1.0, 0, 0, 0), uncertainty=Uncertainty(confidence=0.95)) for k in range(6)]
    return ReconstructionResult(points=rp, camera_poses=poses, registration_status="success")


@pytest.fixture(scope="module")
def scene():
    return assemble_interior_scene(result_of(build_points()), up=(0, 0, 1))


def _room(sc):
    rooms = [e for e in sc.world.entities.values() if e.type.value == "room"]
    assert len(rooms) == 1, "one captured room must stay one room"
    return rooms[0]


def test_footprint_is_the_true_L_not_the_bounding_box(scene):
    cp = _room(scene).custom_properties
    assert cp["floor_area_m2"] == pytest.approx(24.0, rel=0.06)       # bounding box would say 30
    assert cp["bbox_floor_area_m2"] == pytest.approx(30.0, rel=0.02)
    assert 5 <= len(cp["footprint_polygon_xyz_m"]) <= 9
    assert cp["floor_height_m"] == pytest.approx(0.0, abs=0.03)


def test_false_ceiling_and_slab_are_layers_of_one_room(scene):
    cp = _room(scene).custom_properties
    layers = {l["role"]: l for l in cp["ceiling_layers"]}
    assert set(layers) == {"false_ceiling", "structural_ceiling"}
    assert layers["false_ceiling"]["height_above_floor_m"] == pytest.approx(2.5, abs=0.04)
    assert layers["structural_ceiling"]["height_above_floor_m"] == pytest.approx(2.8, abs=0.04)
    assert layers["false_ceiling"]["plan_coverage_fraction"] > layers["structural_ceiling"]["plan_coverage_fraction"]
    assert cp["ceiling_evidence"] == "measured"
    assert len(cp["footprint_edge_heights_m"]) == len(cp["footprint_polygon_xyz_m"])


def test_window_is_detected_as_an_opening_with_the_measured_size(scene):
    wins = [e for e in scene.world.entities.values() if e.type.value == "window"]
    assert len(wins) == 1
    c = wins[0].custom_properties
    assert c["width_m"] == pytest.approx(2.0, abs=0.25)
    assert c["height_m"] == pytest.approx(1.2, abs=0.25)
    assert c["sill_height_m"] == pytest.approx(0.9, abs=0.1)
    assert not [e for e in scene.world.entities.values() if e.type.value in ("door", "opening")]


def test_furniture_is_found_near_its_true_position_and_size(scene):
    objs = [e for e in scene.world.entities.values() if "furniture_candidate" in e.semantic_labels]
    assert len(objs) == 2
    centres = sorted((round(e.transform["position"]["x"], 1), round(e.transform["position"]["y"], 1)) for e in objs)
    truth = sorted([(4.2, 1.2), (1.5, 4.45)])
    for (cx, cy), (tx, ty) in zip(centres, sorted(truth, key=lambda t: t[0])):
        assert math.hypot(cx - tx, cy - ty) < 0.35
    assert sorted(round(e.custom_properties["height_m"], 1) for e in objs) == pytest.approx([0.7, 0.8], abs=0.15)
    assert all(e.custom_properties["room_id"] == _room(scene).id for e in objs)


def test_every_structural_and_object_entity_carries_a_measured_colour(scene):
    cp = _room(scene).custom_properties
    assert cp["floor_color_rgb"] == [180, 170, 160] and cp["wall_color_rgb"] == [180, 170, 160]
    objs = [e for e in scene.world.entities.values() if "furniture_candidate" in e.semantic_labels]
    assert all(e.custom_properties["mean_color_rgb"] == [180, 170, 160] for e in objs)


def test_without_ceiling_points_ceiling_evidence_is_missing_not_invented():
    sc = assemble_interior_scene(result_of([p for p in build_points(furniture=False, window=False) if p[2] < 2.0]),
                                 up=(0, 0, 1))
    cp = _room(sc).custom_properties
    assert cp.get("ceiling_layers") == [] and cp["ceiling_evidence"] == "missing"


def test_validation_is_clean(scene):
    assert scene.validation_issues == []
