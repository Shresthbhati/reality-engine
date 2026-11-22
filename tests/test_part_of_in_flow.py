"""Sub-part detections reach the WorldIR as PART_OF through the real evidence flow. Detectors are SYNTHETIC test doubles."""
from __future__ import annotations

from engine.multiengine.contract import EngineAdapter, Task
from engine.multiengine.evidence_flow import StageSpec, run_evidence_flow
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry
from tests.test_multiengine_evidence_flow import SOURCES, _spec


class PartDetector(EngineAdapter):
    """SYNTHETIC: a door and a handle on it in image 1 (the handle names its host box), the same door WITHOUT parentage in image 2."""

    def __init__(self, sp, with_parent=True):
        self.spec, self.with_parent = sp, with_parent

    def validate_input(self, request):
        pass

    def load(self, device):
        pass

    def unload(self):
        pass

    def infer(self, request):
        def d(src, label, box, **extra):
            return ObservationDraft(task="detection", quantity="bbox", source_id=src, coordinate_system=f"image:{src}", image_region=box,
                                    predicted={"label": label, "score": 0.9, **extra}, confidence=0.9)

        parent = {"parent_pipeline_box_id": "b0"} if self.with_parent else {}
        return [d("img1.jpg", "door", (10, 10, 200, 400), pipeline_box_id="b0"),
                d("img1.jpg", "door handle", (170, 200, 190, 220), pipeline_box_id="b1", **parent)]


def _run(tmp_path, with_parent):
    reg = EngineRegistry()
    sp = _spec("det_p", "fam-p")
    reg.register(sp)
    return run_evidence_flow(tmp_path, "run-p", SOURCES[:1], registry=reg, adapters={"det_p": PartDetector(sp, with_parent)},
                             stages=[StageSpec(Task.DETECTION)])


def _part_of(res):
    return [(e.id, r.target_id) for e in res.bridge.entities.values() for r in e.relationships if r.kind.value == "part_of"]


def test_a_sub_part_detection_becomes_a_part_of_relationship_to_its_host(tmp_path):
    res = _run(tmp_path, with_parent=True)
    rels = _part_of(res)
    assert len(rels) == 1, rels
    child, parent = rels[0]
    assert parent in res.bridge.entities and child != parent  # the target is a real entity in the same world


def test_detections_without_box_ids_get_no_parent(tmp_path):
    assert _part_of(_run(tmp_path, with_parent=False)) == []  # nothing is inferred from labels or geometry
