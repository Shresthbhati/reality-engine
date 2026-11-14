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


def test_clean_boxes_drops_scene_sized_and_speck_boxes_and_dedups_per_class():
    src = (1000, 1000)
    raw = [
        ("sofa", 0.9, (100, 500, 600, 800)),
        ("sofa", 0.6, (110, 510, 590, 790)),         # same class, IoU > 0.5 -> dropped
        ("rug", 0.8, (110, 510, 590, 790)),           # different class, same place -> kept
        ("curtain", 0.7, (0, 0, 1000, 1000)),         # the whole frame: scene-level false positive
        ("potted plant", 0.7, (10, 10, 14, 14)),      # a speck
    ]
    kept = gs.clean_boxes(raw, src)
    assert [(k[0], k[1]) for k in kept] == [("sofa", 0.9), ("rug", 0.8)]


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
    assert seen == [gs.SCENE_PROMPTS] and "rug" in gs.SCENE_PROMPTS and "potted plant" in gs.SCENE_PROMPTS


def _cand(cid, label, lo, hi, conf=0.8):
    from perception.instances.lifting import Vec3
    from perception.instances.object_resolution import MergedObjectCandidate

    mid = tuple((a + b) / 2 for a, b in zip(lo, hi))
    return MergedObjectCandidate(candidate_id=cid, label=label, position=Vec3(*mid), bounds_min=Vec3(*lo),
                                 bounds_max=Vec3(*hi), source_hypotheses=(), confidence=conf)


def test_overlapping_same_label_candidates_fuse_by_the_validators_own_rule():
    from perception.instances.object_resolution import fuse_overlapping_candidates

    a = _cand("a", "sofa", (0, 0, 0), (2, 1, 1))
    b = _cand("b", "sofa", (0.2, 0, 0), (2.2, 1, 1))                 # IoU 0.82: one sofa seen twice
    c = _cand("c", "sofa", (5, 0, 0), (7, 1, 1))                     # elsewhere: a second sofa
    d = _cand("d", "rug", (0, 0, 0), (2, 1, 1))                      # same place, different class: untouched
    out = fuse_overlapping_candidates([a, b, c, d])
    assert [x.candidate_id for x in out] == ["a", "c", "d"]
    fused = out[0]
    assert (fused.bounds_min.x, fused.bounds_max.x) == (0, 2.2)      # union extent
    assert fuse_overlapping_candidates([a, c]) == [a, c]             # nothing overlaps: unchanged


def test_perception_stage_rolls_back_objects_that_invalidate_the_world(synthetic_registered_result, monkeypatch):
    """The stage must cost the objects, not the world (a real run lost its whole multi-view model to one duplicate)."""
    from engine.pipeline import vertical_slice as vs
    import perception.instances.object_resolution as orm
    from world_ir.validation import validate_world_ir

    result, _intr, maps = synthetic_registered_result
    options = VerticalSliceOptions(
        intrinsics=(64.0, 64.0, 32.0, 32.0), image_size=(64, 64), perception_model="grounded", object_resolver="union_find")

    class _Fake:
        def __init__(self, depth_by_id):
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
        def __init__(self, depth_by_id):
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
