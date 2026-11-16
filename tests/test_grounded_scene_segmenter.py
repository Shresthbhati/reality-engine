"""3b/3c: open-vocabulary scene objects reach the multi-view perception stage as depth-cut masks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import engine.pipeline.open_vocab_furniture as ovf
import perception.detection.grounded_scene as gs
from engine.pipeline.vertical_slice import VerticalSliceOptions
from perception.depth.interface import DepthMap
from perception.segmentation.interface import SegmentationResult, SegmentedRegion
from tests.test_mapping_spine_perception import _empty_world, synthetic_registered_result  # noqa: F401 (fixture)


def test_clean_boxes_drops_weak_scene_sized_speck_and_duplicate_boxes():
    src = (1000, 1000)
    raw = [
        ("sofa", 0.9, (100, 500, 600, 800)),
        ("sofa", 0.6, (110, 510, 590, 790)),         # same class, IoU > 0.5 -> dropped
        ("bed", 0.8, (110, 510, 590, 790)),           # DIFFERENT class on the same pixels: one object, the sofa wins
        ("rug", 0.7, (700, 500, 950, 800)),           # elsewhere: kept
        ("wardrobe", 0.40, (50, 50, 300, 400)),       # below MIN_SCORE: the 'a wardrobe in a living room' kind of hit
        ("curtain", 0.7, (0, 0, 1000, 1000)),         # the whole frame: scene-level false positive
        ("potted plant", 0.7, (10, 10, 14, 14)),      # a speck
    ]
    kept = gs.clean_boxes(raw, src)
    assert [(k[0], k[1]) for k in kept] == [("sofa", 0.9), ("rug", 0.7)]


def test_prompt_variants_are_queried_and_reported_under_the_canonical_class(monkeypatch):
    """'coffee table' alone found the table in 0/6 views of a real room; 'wooden coffee table' found it in 6/6."""
    q = gs.query_prompts(("sofa", "coffee table", "rug"))
    assert q[:3] == ("sofa", "coffee table", "rug") and "wooden coffee table" in q and "area rug" in q
    assert "standing lamp" not in q                       # only variants of the requested classes are sent

    depth = np.full((40, 40), 5.0)
    depth[10:30, 10:30] = 2.0
    dm = DepthMap(evidence_id="ev-a", width=40, height=40, values=depth.tolist(), unit="meters")
    sent = []

    def fake_raw(path, prompts):
        sent.append(tuple(prompts))
        # the plain prompt finds nothing; the variant finds the table, and the rug is found by both phrasings
        return [("wooden coffee table", 0.8, (8, 8, 32, 32)), ("rug", 0.7, (0, 33, 8, 39)),
                ("area rug", 0.75, (0, 33, 8, 39))], (40, 40)

    monkeypatch.setattr(ovf, "_grounded_raw", fake_raw)
    monkeypatch.setattr(ovf, "item_path", lambda item: "ignored")
    seg = gs.GroundedSceneSegmenter({"ev-a": dm}, prompts=("coffee table", "rug"))
    out = seg.segment([SimpleNamespace(id="ev-a")])
    labels = [r.label for r in out[0].regions]
    assert "wooden coffee table" in sent[0] and labels.count("coffee table") == 1
    assert labels.count("rug") <= 1 and "wooden coffee table" not in labels and "area rug" not in labels


def _sam_scene(monkeypatch):
    depth = np.full((40, 40), 5.0)
    depth[10:30, 10:30] = 2.0                 # the object, 2 m away
    depth[0:10, 0:10] = 9.0                   # a far outlier patch inside the SAM mask: must be stripped
    dm = DepthMap(evidence_id="ev-a", width=40, height=40, values=depth.tolist(), unit="meters")
    monkeypatch.setattr(ovf, "_grounded_raw", lambda path, prompts: ([("sofa", 0.9, (4.0, 4.0, 34.0, 34.0))], (40, 40)))
    monkeypatch.setattr(ovf, "item_path", lambda item: "unused")
    import engine.core.image_io as iio
    from PIL import Image
    monkeypatch.setattr(iio, "open_oriented", lambda p: Image.new("RGB", (40, 40)))
    return dm


def test_sam_refined_mask_keeps_the_instance_outline_and_strips_depth_outliers():
    depth = np.full((40, 40), 2.0)
    depth[0:10, 0:10] = 9.0
    sam = np.zeros((80, 80), dtype=bool)      # full-resolution mask, twice the depth grid
    sam[0:60, 0:60] = True
    mask = gs.sam_refined_mask(depth, sam)
    assert mask.shape == (40, 40) and mask[10:30, 10:30].all() and not mask[0:10, 0:10].any() and not mask[35:, 35:].any()
    assert gs.sam_refined_mask(depth, np.zeros((80, 80), dtype=bool)) is None     # an empty SAM mask is no mask


def test_segmenter_uses_sam2_masks_and_counts_them(monkeypatch):
    dm = _sam_scene(monkeypatch)
    sam_mask = np.zeros((40, 40), dtype=bool)
    sam_mask[10:30, 10:30] = True

    class _Sam:
        def masks_for_boxes(self, image, boxes):
            return [sam_mask for _ in boxes]

    seg = gs.GroundedSceneSegmenter({"ev-a": dm}, mask_source="sam2", sam=_Sam())
    out = seg.segment([SimpleNamespace(id="ev-a")])
    (region,) = out[0].regions
    assert np.asarray(region.mask).sum() == 400 and seg.mask_counts == {"sam2": 1, "depth_band": 0} and seg.mask_fallback is None


def test_segmenter_falls_back_to_the_depth_band_when_sam2_is_missing_and_says_so(monkeypatch):
    from perception.segmentation.sam2_box_prompter import Sam2Unavailable

    dm = _sam_scene(monkeypatch)

    class _Missing:
        def masks_for_boxes(self, image, boxes):
            raise Sam2Unavailable("checkpoint not in the local cache")

    seg = gs.GroundedSceneSegmenter({"ev-a": dm}, mask_source="sam2", sam=_Missing())
    out = seg.segment([SimpleNamespace(id="ev-a")])
    assert len(out[0].regions) == 1 and seg.mask_counts == {"sam2": 0, "depth_band": 1}
    assert "checkpoint not in the local cache" in seg.mask_fallback


def test_unknown_mask_source_is_refused():
    import pytest

    with pytest.raises(ValueError):
        gs.GroundedSceneSegmenter({}, mask_source="magic")


def test_labels_map_to_world_types_and_unlisted_ones_are_not_guessed():
    t = gs.ENTITY_TYPE_BY_LABEL
    assert t["sofa"] == t["tv stand"] == t["rug"] == "furniture" and t["potted plant"] == "vegetation"
    assert "picture frame" not in t and "curtain" not in t


def test_depth_cut_keeps_the_foreground_object_and_drops_the_wall_behind_it():
    depth = np.full((40, 40), 5.0)           # wall at 5 m
    depth[10:30, 10:30] = 2.0                # object at 2 m
    # the detector's box is looser than the object: it also covers a margin of wall
    mask = gs.depth_refined_mask(depth, (8, 8, 32, 32), src_size=(40, 40))
    assert mask is not None and mask.shape == (40, 40)
    assert mask[10:30, 10:30].all()                      # the whole object
    assert not mask[8:10, 8:32].any() and not mask[30:32, 8:32].any()   # none of the wall margin
    assert not mask[0:8].any() and not mask[32:].any()   # nothing outside the box


def test_depth_cut_scales_boxes_to_the_depth_grid_and_refuses_empty_depth():
    depth = np.full((20, 30), 3.0)           # depth grid is smaller than the 600x400 image
    mask = gs.depth_refined_mask(depth, (300, 200, 600, 400), src_size=(600, 400))
    assert mask is not None and mask[10:, 15:].all() and not mask[:10].any() and not mask[:, :15].any()
    assert gs.depth_refined_mask(np.zeros((20, 30)), (0, 0, 300, 200), src_size=(600, 400)) is None


def test_segmenter_returns_labelled_masks_per_view_and_skips_views_without_depth(monkeypatch):
    depth = np.full((40, 40), 5.0)
    depth[10:30, 10:30] = 2.0
    dm = DepthMap(evidence_id="ev-a", width=40, height=40, values=depth.tolist(), unit="meters")
    seen = []

    def fake_raw(path, prompts):
        seen.append(tuple(prompts))
        return [("coffee table", 0.8, (8.0, 8.0, 32.0, 32.0))], (40, 40)

    monkeypatch.setattr(ovf, "_grounded_raw", fake_raw)
    items = [SimpleNamespace(id="ev-a", source_uri="file:///a.jpg"), SimpleNamespace(id="ev-b", source_uri="file:///b.jpg")]
    results = gs.GroundedSceneSegmenter({"ev-a": dm}).segment(items)

    assert [r.evidence_id for r in results] == ["ev-a"]                   # ev-b has no depth: skipped, not guessed
    (region,) = results[0].regions
    assert region.label == "coffee table" and region.confidence == 0.8 and region.evidence_id == "ev-a"
    assert np.asarray(region.mask).sum() == 400                           # the 20x20 foreground
    assert seen == [gs.query_prompts(gs.SCENE_PROMPTS)] and "rug" in gs.SCENE_PROMPTS and "potted plant" in gs.SCENE_PROMPTS


def _cand(cid, label, lo, hi, conf=0.8):
    from perception.instances.lifting import Vec3
    from perception.instances.object_resolution import MergedObjectCandidate

    mid = tuple((a + b) / 2 for a, b in zip(lo, hi))
    return MergedObjectCandidate(candidate_id=cid, label=label, position=Vec3(*mid), bounds_min=Vec3(*lo),
                                 bounds_max=Vec3(*hi), source_hypotheses=(), confidence=conf)


def _seen(cand, *views):
    import dataclasses
    return dataclasses.replace(cand, source_hypotheses=tuple(SimpleNamespace(evidence_id=v, region_id=f"r-{v}") for v in views))


def test_scene_filters_drop_outliers_slivers_and_single_view_objects():
    rng = np.random.default_rng(0)
    box = gs.scene_box(rng.uniform(-1.5, 1.5, size=(200, 3)))            # a ~3 unit room
    assert box is not None and gs.scene_box(rng.uniform(size=(10, 3))) is None   # too few points: no scene, no filter

    good = _seen(_cand("good", "sofa", (-0.5, -0.5, 0), (0.5, 0.5, 0.5)), "a", "b", "c")
    far = _seen(_cand("far", "bookshelf", (-11, 1, -2), (-4, 3, 1)), "a", "b")       # a depth blow-up outside the room
    huge = _seen(_cand("huge", "potted plant", (-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), "a", "b")   # as big as the room
    sliver = _seen(_cand("sliver", "potted plant", (0, 0, 0), (0.06, 0.08, 0.08)), "a")  # one photo of a 10-photo capture
    kept, dropped = gs.filter_implausible([good, far, huge, sliver], box, n_views=10)
    assert [c.candidate_id for c in kept] == ["good"]
    assert dropped == {"outside_scene": 1, "scene_sized": 1, "single_view": 1}


def test_single_view_rule_is_a_confidence_gate_in_small_captures():
    weak = _seen(_cand("weak", "sofa", (0, 0, 0), (1, 1, 1), conf=0.4), "a")
    firm = _seen(_cand("firm", "sofa", (3, 0, 0), (4, 1, 1), conf=0.8), "a")
    kept, dropped = gs.filter_implausible([weak, firm], None, n_views=3)            # 3 photos: one view may be all there is
    assert [c.candidate_id for c in kept] == ["firm"] and dropped == {"weak_single_view": 1}


def test_overlapping_same_label_candidates_fuse_by_the_validators_own_rule():
    from perception.instances.object_resolution import fuse_overlapping_candidates

    a = _cand("a", "sofa", (0, 0, 0), (2, 1, 1))
    b = _cand("b", "sofa", (0.2, 0, 0), (2.2, 1, 1))                 # IoU 0.82: one sofa seen twice
    c = _cand("c", "sofa", (5, 0, 0), (7, 1, 1))                     # elsewhere: a second sofa
    d = _cand("d", "rug", (0, 0, 0), (2, 1, 1))                      # same place, different class: untouched
    out = fuse_overlapping_candidates([a, b, c, d])
    assert [x.candidate_id for x in out] == ["a", "c", "d"]
    fused = out[0]
    assert abs((fused.bounds_max.x - fused.bounds_min.x) - 2.0) < 1e-9   # consensus extent (both are 2 m), not the 2.2 m union
    assert fuse_overlapping_candidates([a, c]) == [a, c]             # nothing overlaps: unchanged
    # same class, IoU only 0.38 but the centres are 0.9 apart on a 2 m object (< half its length): still one sofa
    e = _cand("e", "sofa", (0.9, 0, 0), (2.9, 1, 1))
    assert [x.candidate_id for x in fuse_overlapping_candidates([a, e])] == ["a"]
    f = _cand("f", "sofa", (1.3, 0, 0), (3.3, 1, 1))                 # centres 1.3 apart: two sofas side by side
    assert [x.candidate_id for x in fuse_overlapping_candidates([a, f])] == ["a", "f"]


def test_perception_stage_rolls_back_objects_that_invalidate_the_world(synthetic_registered_result, monkeypatch):
    """The stage must cost the objects, not the world (a real run lost its whole multi-view model to one duplicate)."""
    from engine.pipeline import vertical_slice as vs
    import perception.instances.object_resolution as orm
    from world_ir.validation import validate_world_ir

    result, _intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0), image_size=(64, 64), perception_model="grounded", object_resolver="union_find")

    class _Fake:
        def __init__(self, depth_by_id, **_options):
            pass

        def segment(self, evidence):
            mask = [[20 <= r < 44 and 20 <= c < 44 for c in range(64)] for r in range(64)]
            return [SegmentationResult(i.evidence_id, [SegmentedRegion(f"s-{i.evidence_id}", i.evidence_id, "sofa", mask, 0.8)],
                                       "grounding-dino-tiny") for i in evidence]

        def unload(self):
            pass

    monkeypatch.setattr(gs, "GroundedSceneSegmenter", _Fake)
    real_validate = validate_world_ir

    class _Bad:
        errors = ["entities x and y are the same class and overlap 59%"]

    monkeypatch.setattr("world_ir.validation.validate_world_ir", lambda w: _Bad())

    class _Ev:
        def __init__(self, eid):
            self.evidence_id = eid
            self.source_uri = f"file://{eid}.jpg"

    world = _empty_world()
    before = (set(world.entities), set(world.geometries), dict(world.metadata))
    facts = vs._perception_stage(result, world, [_Ev("IMG_A"), _Ev("IMG_B")], maps, options)

    assert facts["status"] == "rejected" and facts["entities_rejected"] >= 1 and "59%" in facts["note"]
    assert (set(world.entities), set(world.geometries), dict(world.metadata)) == before   # world exactly as it was
    assert real_validate(world).is_valid()


def test_perception_stage_uses_the_grounded_segmenter_and_labels_the_model(synthetic_registered_result, monkeypatch):
    from engine.pipeline import vertical_slice as vs

    result, _intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0), image_size=(64, 64), perception_model="grounded", object_resolver="union_find")
    built = {}

    class _Fake:
        def __init__(self, depth_by_id, **_options):
            built["depth_ids"] = sorted(depth_by_id)
            self.unloaded = False

        def segment(self, evidence):
            mask = [[20 <= r < 44 and 20 <= c < 44 for c in range(64)] for r in range(64)]
            return [SegmentationResult(
                evidence_id=i.evidence_id,
                regions=[SegmentedRegion(f"seg-{i.evidence_id}-000", i.evidence_id, "coffee table", mask, 0.8)],
                model_name="grounding-dino-tiny") for i in evidence]

        def unload(self):
            built["unloaded"] = True

    monkeypatch.setattr(gs, "GroundedSceneSegmenter", _Fake)

    class _Ev:
        def __init__(self, eid):
            self.evidence_id = eid
            self.source_uri = f"file://{eid}.jpg"

    world = _empty_world()
    facts = vs._perception_stage(result, world, [_Ev("IMG_A"), _Ev("IMG_B")], maps, options)
    assert facts["status"] == "ran", facts.get("note")
    assert built["depth_ids"] == ["IMG_A", "IMG_B"] and built["unloaded"] is True
    assert facts["entities_promoted"] == 1 and "Grounding DINO" in facts["note"]
    (ent,) = [e for e in world.entities.values() if e.id.startswith("entity-object")]
    assert ent.semantic_labels == ["coffee table"]
    from world_ir import EntityType
    assert ent.type == EntityType.FURNITURE


def test_perception_stage_drops_weak_single_view_objects_but_keeps_confirmed_ones(synthetic_registered_result, monkeypatch):
    from engine.pipeline import vertical_slice as vs

    result, _intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0), image_size=(64, 64), perception_model="grounded", object_resolver="union_find")
    mask = [[20 <= r < 44 and 20 <= c < 44 for c in range(64)] for r in range(64)]
    mask2 = [[4 <= r < 12 and 4 <= c < 12 for c in range(64)] for r in range(64)]   # a different spot: seen by IMG_A only

    class _Fake:
        def __init__(self, depth_by_id, **_options):
            pass

        def segment(self, evidence):
            out = []
            for i in evidence:
                regs = [SegmentedRegion(f"s-{i.evidence_id}-0", i.evidence_id, "sofa", mask, 0.8)]   # both views: confirmed
                if i.evidence_id == "IMG_A":
                    regs.append(SegmentedRegion("s-A-1", "IMG_A", "bed", mask2, 0.4))              # one view, weak: a hallucination
                out.append(SegmentationResult(i.evidence_id, regs, "grounding-dino-tiny"))
            return out

        def unload(self):
            pass

    monkeypatch.setattr(gs, "GroundedSceneSegmenter", _Fake)

    class _Ev:
        def __init__(self, eid):
            self.evidence_id = eid
            self.source_uri = f"file://{eid}.jpg"

    world = _empty_world()
    facts = vs._perception_stage(result, world, [_Ev("IMG_A"), _Ev("IMG_B")], maps, options)
    labels = sorted(e.semantic_labels[0] for e in world.entities.values() if e.id.startswith("entity-object"))
    assert labels == ["sofa"] and facts["resolution"]["dropped_weak_single_view"] == 1


def test_filter_implausible_can_say_what_it_dropped_and_why():
    from types import SimpleNamespace as NS

    import numpy as np

    from engine.math import Vec3
    from perception.detection.grounded_scene import filter_implausible

    box = (np.array([0.0, 0.0, 0.0]), np.array([10.0, 3.0, 10.0]))

    def cand(label, pos, views, conf=0.8):
        return NS(label=label, position=Vec3(*pos), bounds_min=Vec3(*(p - 0.5 for p in pos)), bounds_max=Vec3(*(p + 0.5 for p in pos)),
                  evidence_ids=tuple(f"v{i}" for i in range(views)), confidence=conf)

    detail: list = []
    kept, dropped = filter_implausible([cand("sofa", (5, 1, 5), 3), cand("tv", (80, 1, 5), 3)], box, n_views=3, detail=detail)
    assert [c.label for c in kept] == ["sofa"] and dropped == {"outside_scene": 1}
    assert detail == [{"label": "tv", "reason": "outside_scene", "views": 3, "confidence": 0.8, "position_m": [80.0, 1.0, 5.0]}]
    # without the argument nothing changes for existing callers
    assert filter_implausible([cand("tv", (80, 1, 5), 3)], box, n_views=3) == ([], {"outside_scene": 1})
