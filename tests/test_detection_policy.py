import pytest

from perception.detection import policy
from perception.detection.interface import ImageDetection
from perception.detection.policy import classify, filter_detections, keep_ratio


def D(label, box=(0, 0, 5, 5)):
    return ImageDetection(label, 0.9, box)


@pytest.mark.parametrize("label", [
    "building", "Wall", "doors", "windows", "gate", "barricade", "Police Barricade", "jersey-barrier",
    "bollard", "fence", "railing", "pole", "lamp post", "street light", "traffic light", "stop sign",
    "trash can", "dustbin", "post box", "letter box", "bench", "fire hydrant", "steps", "stairs", "bridge",
    "metro station", "bus stop shelter", "kiosk", "tree", "couch", "sofa", "chairs", "dining table", "bed",
    "bookcase", "tv", "sink", "toilet", "bathtub", "refrigerator", "oven", "wardrobe", "curtains",
    "a red barricade",
])
def test_keep(label):
    assert classify(label) == "keep", label


@pytest.mark.parametrize("label", [
    "person", "people", "pedestrian", "crowd", "child", "children", "car", "taxi", "bus", "truck", "van",
    "motorcycle", "scooter", "motorbike", "bicycle", "rickshaw", "auto-rickshaw", "cart", "dog", "cow",
    "cat", "bird", "horse", "grass", "bushes", "undergrowth", "potted plant", "flower pots",
])
def test_drop(label):
    assert classify(label) == "drop", label


def test_unknown_reported_not_kept():
    kept, dropped, unknown = filter_detections([D("spaceship"), D("person"), D("bench")])
    assert [d.label for d in kept] == ["bench"]
    assert [d.label for d in dropped] == ["person"]
    assert [d.label for d in unknown] == ["spaceship"]
    stats = keep_ratio(kept, dropped, unknown)
    assert stats["unknown_labels"] == ["spaceship"] and stats["total"] == 3
    assert stats["keep_ratio"] == pytest.approx(1 / 3)
    assert keep_ratio([], [], [])["keep_ratio"] is None


def test_barricade_always_kept_even_beside_person():
    dets = [D("person", (0, 0, 10, 10)), D("police barricade", (2, 0, 12, 10)), D("jersey barrier"), D("bollard")]
    kept, dropped, _ = filter_detections(dets)
    assert {d.label for d in kept} == {"police barricade", "jersey barrier", "bollard"}
    assert [d.label for d in dropped] == ["person"]


def test_query_lists_consistent_with_policy():
    for q in policy.STRUCTURE_QUERIES:
        assert classify(q) == "keep", q
    for q in policy.TRANSIENT_QUERIES:
        assert classify(q) == "drop", q


def test_discrete_tree_kept_foliage_dropped():
    assert classify("tree") == "keep" and classify("palm trees") == "keep"
    assert classify("bush") == "drop" and classify("plant") == "drop"
