"""B04: whole -> part detection with scripted detections, full-image coordinates, containment and call budget."""
from types import SimpleNamespace as D

import numpy as np

from perception.detection.hierarchical import detect_hierarchy
from perception.ontology import Ontology, OntologyNode


def _onto():
    nodes = [
        OntologyNode("furniture/almirah", "almirah", prompts=("almirah",), components=("handle", "lock")),
        OntologyNode("furniture/almirah/handle", "handle", prompts=("door handle",)),
        OntologyNode("furniture/almirah/lock", "lock", prompts=("door lock",)),
    ]
    return Ontology(version="t", nodes={n.path: n for n in nodes})


class _Det:
    """Scripted: whole image sees an almirah at (100,100,500,700); inside its crop a handle and a lock (crop coords)."""
    def __init__(self, extra=None): self.calls = []; self.extra = extra or []
    def detect(self, img, queries):
        self.calls.append((img.shape[:2], tuple(queries)))
        out = []
        if "almirah" in queries:
            out.append(D(label="almirah", score=0.9, box=(100, 100, 500, 700)))
        if "door handle" in queries:
            out.append(D(label="door handle", score=0.8, box=(300, 250, 340, 300)))     # crop coords
        if "door lock" in queries:
            out.append(D(label="door lock", score=0.7, box=(310, 310, 330, 330)))
        return out + [d for d in self.extra if d.label in queries]


IMG = np.zeros((800, 600, 3), np.uint8)


def test_parts_link_to_their_whole_in_full_image_pixels():
    t = detect_hierarchy(IMG, "im1", _Det(), _onto(), ["furniture/almirah"])
    paths = list(t.class_path)
    assert paths[0] == "furniture/almirah" and t.parent_obs[0] == -1
    hi = paths.index("furniture/almirah/handle")
    assert t.parent_obs[hi] == 0
    assert t.bbox_px[hi].tolist() == [400.0, 350.0, 440.0, 400.0]       # crop + (100,100)


def test_part_outside_its_parent_is_dropped():
    far = D(label="door handle", score=0.99, box=(1000, 1000, 1040, 1040))
    t = detect_hierarchy(IMG, "im1", _Det([far]), _onto(), ["furniture/almirah"])
    assert all(not (b[0] > 900) for b in t.bbox_px)


def test_budget_of_one_call_stops_after_the_root_pass():
    d = _Det()
    t = detect_hierarchy(IMG, "im1", d, _onto(), ["furniture/almirah"], budget_calls=1)
    assert len(d.calls) == 1 and list(t.class_path) == ["furniture/almirah"]
