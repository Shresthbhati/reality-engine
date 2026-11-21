"""The production detection engine: DetectionPipeline (tiling, sub-part crops, class filtering, masks) driven through the execution
manager by REAL-adapter-shaped inner engines.

The inner detector / segmenter here are SYNTHETIC scene-aware stand-ins (this file proves the wiring, the preservation of raw
candidates and masks, and the honesty rules -- not detector accuracy, which tests/benchmarks on real photographs measure). The
production guard refuses the pipeline module's own deterministic reference adapters; that is tested here too."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from apps.api import evidence_world_stage as ew
from engine.multiengine.adapters import pipeline_detection as pdet
from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    Device,
    EngineAdapter,
    EngineRequest,
    EngineRunStatus,
    EngineSpec,
    Task,
)
from engine.multiengine.mask_codec import count_holes, decode_mask, encode_mask
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry
from engine.multiengine.spatial_intelligence.detection_pipeline import RuleBasedDetectorAdapter, ThresholdSegmenterAdapter
from engine.multiengine.spatial_stage import _lift_mask
from tests.test_multiengine_evidence_flow import FakeDetector, _spec

W, H = 900, 700


def _blocked():
    return Availability(AvailabilityStatus.BLOCKED, "checkpoint missing (synthetic)", "download it")


class _SceneDetector(FakeDetector):
    """SYNTHETIC open-vocabulary detector: a door spanning the middle of whatever image it is handed; a door handle when asked for
    door parts on a crop; it echoes the label it was queried with."""

    def infer(self, request):
        out = []
        for s in request.sources:
            with Image.open(s) as im:
                w, h = im.size
            q = [str(x) for x in (request.payload.get("queries") or [])]
            if "door" in q:
                out.append(ObservationDraft(task="detection", quantity="bbox", source_id=s, coordinate_system="image:x",
                                            image_region=(0.3 * w, 0.1 * h, 0.7 * w, 0.9 * h), predicted={"label": "door", "score": 0.9}, confidence=0.9))
            part = next((x for x in q if x == "door_handle"), None)
            if part:
                out.append(ObservationDraft(task="detection", quantity="bbox", source_id=s, coordinate_system="image:x",
                                            image_region=(0.4 * w, 0.4 * h, 0.6 * w, 0.6 * h), predicted={"label": part, "score": 0.8}, confidence=0.8))
        return out


class _Predictor:
    def set_image(self, img):
        self.shape = img.shape[:2]

    def predict(self, box, multimask_output=False):
        h, w = self.shape
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m[y1:y2, x1:x2] = True
        cy, cx, hh, hw = (y1 + y2) // 2, (x1 + x2) // 2, max(3, (y2 - y1) // 8), max(3, (x2 - x1) // 8)
        m[cy - hh:cy + hh, cx - hw:cx + hw] = False  # a hole large enough to survive the pipeline's morphological cleanup
        return m[None], np.array([0.9]), None


class _FakeSam(EngineAdapter):
    def __init__(self, ok=True, fail_load=False):
        self.spec = EngineSpec(engine_id="fake_sam", name="fake sam", version="s1", tasks=frozenset({Task.SEGMENTATION}), input_schema="img",
                               output_schema="mask", probe=(lambda: Availability(AvailabilityStatus.AVAILABLE, "ok", evidence=("synthetic",))) if ok else _blocked)
        self._predictor = None
        self.fail_load = fail_load
        self.loaded = self.unloaded = 0

    def validate_input(self, request):
        pass

    def load(self, device):
        from engine.multiengine.contract import EngineError, EngineErrorKind

        if self.fail_load:
            raise EngineError(EngineErrorKind.MISSING_CHECKPOINT, "weights not found (synthetic)")
        self._predictor = _Predictor()
        self.loaded += 1

    def unload(self):
        self._predictor = None
        self.unloaded += 1

    def infer(self, request):
        return []


@pytest.fixture()
def photo(tmp_path):
    p = tmp_path / "p.png"
    Image.new("RGB", (W, H), (90, 100, 110)).save(p)
    return str(p)


def _engine(det_ok=True, seg="ok", tile=500):
    sp = _spec("scene_det", "fam-x")
    if not det_ok:
        sp = EngineSpec(**{**sp.__dict__, "probe": _blocked})
    det = _SceneDetector(sp)
    sam = None if seg is None else _FakeSam(ok=(seg != "blocked"), fail_load=(seg == "load_fails"))
    return pdet.PipelineDetectionEngineAdapter(det, sam, tile_size=tile, pin_checkpoints=False), det, sam


def _run(eng, photo):
    return eng.run(EngineRequest(sources=(photo,), payload={}, seed=42), Device("cpu"))


# ---------------------------------------------------------------------------------------------------------- codec
@pytest.mark.parametrize("seed", range(5))
def test_mask_codec_round_trips_exactly_including_holes(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((37, 53)) > 0.6
    assert np.array_equal(decode_mask(encode_mask(m)), m)
    ring = np.zeros((30, 30), bool)
    ring[5:25, 5:25] = True
    ring[12:18, 12:18] = False
    assert count_holes(ring) == 1 and np.array_equal(decode_mask(encode_mask(ring)), ring)


def test_mask_codec_handles_all_true_and_all_false_and_rejects_corruption():
    for m in (np.ones((4, 6), bool), np.zeros((4, 6), bool)):
        assert np.array_equal(decode_mask(encode_mask(m)), m)
    enc = encode_mask(np.ones((4, 6), bool))
    with pytest.raises(ValueError):
        decode_mask({**enc, "runs": [5]})
    with pytest.raises(ValueError):
        encode_mask(np.zeros((0, 3), bool))


# ---------------------------------------------------------------------------------------------------------- guards
def test_reference_stand_ins_are_refused_as_production_adapters():
    with pytest.raises(pdet.ProductionAdapterError):
        pdet.PipelineDetectionEngineAdapter(RuleBasedDetectorAdapter())
    with pytest.raises(pdet.ProductionAdapterError):
        pdet.AdapterSegmenter(ThresholdSegmenterAdapter())
    _, det, _ = _engine()
    with pytest.raises(pdet.ProductionAdapterError):
        pdet.PipelineDetectionEngineAdapter(det, ThresholdSegmenterAdapter())


def test_install_never_substitutes_a_missing_detector():
    reg, adapters = EngineRegistry(), {}
    with pytest.raises(KeyError):
        pdet.install(reg, adapters, detector_id="owlvit")
    assert pdet.ENGINE_ID not in adapters


def test_virtual_sources_and_test_double_predictions_are_refused():
    eng, _, _ = _engine()
    run = eng.run(EngineRequest(sources=("virtual:x",), payload={}, seed=1), Device("cpu"))
    assert run.status is EngineRunStatus.FAILED and "virtual" in (run.error or "")

    class Doubled(_SceneDetector):
        def infer(self, request):
            return [ObservationDraft(task="detection", quantity="bbox", source_id=s, coordinate_system="image:x", image_region=(1, 1, 9, 9),
                                     predicted={"label": "chair", "score": 0.8, "is_test_double": True}, confidence=0.8) for s in request.sources]

    with pytest.raises(Exception, match="test-double"):
        pdet.AdapterDetector(Doubled(_spec("d2", "f"))).detect(np.zeros((20, 20, 3), np.uint8), ["chair"])


# ---------------------------------------------------------------------------------------------------------- the engine
def test_engine_tiles_crops_subparts_segments_and_preserves_every_raw_candidate(photo):
    eng, _, sam = _engine(tile=500)
    run = _run(eng, photo)
    assert run.status is EngineRunStatus.SUCCEEDED, run.error
    f = eng.last_facts
    assert f["tiles_processed"] >= 4 and f["detector_forward_passes"] > f["tiles_processed"]  # tiled inference executed on a 900x700 photo
    assert f["subparts_detected"] >= 1  # the host crop of the door was searched for door parts
    assert {o.quantity for o in run.observations} == {"bbox", "detection_pipeline_summary"}
    boxes = [o for o in run.observations if o.quantity == "bbox"]
    parts = [o for o in boxes if o.predicted["parent_pipeline_box_id"]]
    assert parts and parts[0].predicted["host_class"] == "door" and parts[0].predicted["scale_tier"] == "micro"
    assert f["segmentation"]["status"] == "ran" and f["segmentation"]["coverage"] == 1.0 and f["segmentation"]["sam_failures"] == 0
    masked = boxes[0]
    m = decode_mask(masked.predicted["mask"])
    assert m.shape == (H, W) and m.any() and masked.predicted["mask"]["holes"] == 1  # the SAM-style hole survives the persisted encoding
    summary = next(o for o in run.observations if o.quantity == "detection_pipeline_summary")
    rows = summary.predicted["raw_detections"]
    assert len(rows) == f["raw_detections"] and {r["disposition"] for r in rows} <= {"kept", "suppressed_as_duplicate", "below_class_confidence"}
    assert sum(r["disposition"] == "kept" for r in rows) == len(boxes)  # raw candidates are accounted for, none silently dropped
    assert sam.loaded == 1 and run.reproducibility["tile_size"] == 500 and run.reproducibility["detector"] == "scene_det"


def test_small_images_are_not_tiled_and_the_facts_say_so(tmp_path):
    small = tmp_path / "s.png"
    Image.new("RGB", (400, 300), (1, 2, 3)).save(small)
    eng, _, _ = _engine(tile=800)
    assert _run(eng, str(small)).status is EngineRunStatus.SUCCEEDED
    assert eng.last_facts["tiles_processed"] == 0 and eng.last_facts["images"][0]["tiled"] is False


def test_an_unavailable_segmenter_is_explicit_and_boxes_carry_no_mask(photo):
    for mode in ("blocked", "load_fails", None):
        eng, _, _ = _engine(seg=mode)
        run = _run(eng, photo)
        assert run.status is EngineRunStatus.SUCCEEDED
        boxes = [o for o in run.observations if o.quantity == "bbox"]
        assert boxes and all(o.predicted["mask"] is None for o in boxes)
        seg = eng.last_facts["segmentation"]
        assert seg["status"] == "unavailable" and seg["masks"] == 0 and seg["reason"]


def test_an_unavailable_detector_makes_the_engine_unavailable_never_a_stand_in(photo):
    eng, _, _ = _engine(det_ok=False)
    run = _run(eng, photo)
    assert run.status is EngineRunStatus.UNAVAILABLE and not run.observations and "checkpoint missing" in (run.error or "")


# ---------------------------------------------------------------------------------------------------------- in the job stage
def _inputs():
    def f():
        reg = EngineRegistry()
        sp = _spec("scene_det", "fam-x")
        reg.register(sp)
        return reg, {"scene_det": _SceneDetector(sp), "fake_sam": _FakeSam()}, None

    return f


OPTS = {"detector_id": "scene_det", "segmenter_id": "fake_sam", "tile_size": 500, "pin_checkpoints": False}


def _root(tmp_path):
    return tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"


def _job(tmp_path, photo, inputs, **kw):
    return ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=[photo], base=tmp_path / "ws", flow_inputs=inputs, depth_engines=[], **kw)


def test_run_for_job_executes_the_pipeline_and_reports_what_it_did(tmp_path, photo):
    r = _job(tmp_path, photo, _inputs(), detection="pipeline", pipeline_options=OPTS)
    dp = r["detection_pipeline"]
    assert dp["mode"] == "pipeline" and dp["status"] == "succeeded" and dp["problems"] == []
    assert dp["tiles_processed"] >= 4 and dp["subparts_detected"] >= 1 and dp["segmentation"]["masks"] > 0 and dp["per_image"][0]["masks"] > 0
    assert r["status"] == "complete"
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    m = EvidenceMatrix(_root(tmp_path))
    assert {o.quantity for o in m.store.query()} == {"bbox", "detection_pipeline_summary"}
    assert any(run.engine_id == pdet.ENGINE_ID and run.status == "succeeded" for run in m.runs())  # the manager recorded the pipeline run
    assert m.verify() == []
    rd = ew.readiness(_root(tmp_path))["detection"]  # the frontend's view, read back from disk
    assert rd["status"] == "ran" and rd["engine"] == pdet.ENGINE_ID and rd["inner_detector"] == "scene_det" and rd["images"] == 1
    assert rd["tiles_processed"] == dp["tiles_processed"] and rd["masks"] == dp["segmentation"]["masks"] and rd["mask_coverage"] == 1.0
    assert rd["segmentation"] == "ran" and rd["segmentation_reason"] is None


def test_run_for_job_marks_a_run_without_masks_partial_not_complete(tmp_path, photo):
    def nosam():
        reg, ad, probe = _inputs()()
        ad["fake_sam"] = _FakeSam(ok=False)
        return reg, ad, probe

    r = _job(tmp_path, photo, nosam, detection="pipeline", pipeline_options=OPTS)
    assert r["status"] == "partial" and any("segmentation did not run" in p for p in r["detection_pipeline"]["problems"])
    rd = ew.readiness(_root(tmp_path))["detection"]
    assert rd["segmentation"] == "unavailable" and "checkpoint missing" in rd["segmentation_reason"] and rd["masks"] == 0


def test_run_for_job_falls_back_to_real_whole_image_detectors_and_says_so(tmp_path, photo):
    def only_det_a():
        reg = EngineRegistry()
        sp = _spec("det_a", "fam-a")
        reg.register(sp)
        return reg, {"det_a": _SceneDetector(sp)}, None

    r = _job(tmp_path, photo, only_det_a, detection="pipeline", pipeline_options={"detector_id": "owlvit"})
    dp = r["detection_pipeline"]
    assert dp["status"] == "not_run" and "not installed" in dp["reason"] and r["status"] == "partial"
    assert r["counts"]["runs"] >= 1  # the legacy detector still ran: evidence exists, the downgrade is on the record


def test_caller_supplied_engines_keep_the_legacy_path_by_default(tmp_path, photo):
    r = _job(tmp_path, photo, _inputs())
    assert r["detection_pipeline"]["mode"] == "legacy" and r["detection_pipeline"]["problems"] == []
    assert ew.readiness(_root(tmp_path))["detection"]["status"] == "not_run"  # no summary evidence: said so, not defaulted


# ---------------------------------------------------------------------------------------------------------- lifting side
def test_lift_mask_resamples_and_refuses_what_it_cannot_trust():
    m = np.zeros((40, 80), bool)
    m[10:30, 20:60] = True
    enc = encode_mask(m)
    assert np.array_equal(_lift_mask(enc, (80, 40), (80, 40)), m)
    half = _lift_mask(enc, (80, 40), (40, 20))
    assert half.shape == (20, 40) and half[10, 20] and not half[0, 0]
    assert _lift_mask(enc, (90, 40), (80, 40)) is None  # not the photograph's resolution: not used
    assert _lift_mask(None, (80, 40), (80, 40)) is None
    assert _lift_mask({**enc, "runs": [1]}, (80, 40), (80, 40)) is None  # corrupt: the box stands alone
