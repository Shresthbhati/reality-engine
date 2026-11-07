"""Task 8.7: the single-image path places detected furniture on the solved floor (honestly scaled)."""
import math

import numpy as np
from PIL import Image

from engine.pipeline.single_image import CONFIDENCE_CEILING, bootstrap_single_image
from evidence.image_check import inspect_image
from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap
from perception.instances.furniture_from_detections import FurnitureDetection

W, H, CAM_H, WALL_Z = 320, 240, 1.0, 5.0


def _room(tmp_path):
    p = tmp_path / "room.png"
    Image.fromarray(np.random.default_rng(1).integers(90, 120, (H, W, 3), dtype=np.uint8)).save(p)
    f = (W / 2) / math.tan(math.radians(30))
    vals = []
    for v in range(H):
        z = min(CAM_H * f / (v - H / 2), WALL_Z) if v > H / 2 + 1 else WALL_Z   # level camera: floor below the horizon
        vals.append([1.0 / z - 0.25] * W)
    item = EvidenceItem(id="ev-room", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    return item, inspect_image(p), DepthMap(evidence_id="x", width=W, height=H, values=vals)


def _sofa(label="couch"):
    return FurnitureDetection(label=label, score=0.9, box_xyxy=(100, 130, 220, 200), view_id="ev-room")


def test_detected_sofa_is_placed_on_the_floor_with_assumed_scale(tmp_path):
    item, facts, dm = _room(tmp_path)
    res = bootstrap_single_image(item, facts, depth_map=dm, furniture_detections=[_sofa()])
    ff = res.facts["furniture"]
    assert ff["attempted"] and ff["placed"] == 1, ff
    assert ff["scale_assumption"]["assumed_camera_height_m"] == 1.5
    ent = next(e for e in res.world.entities.values() if e.type.value == "furniture")
    assert ent.confidence <= CONFIDENCE_CEILING and ent.provenance.name == "INFERRED"
    assert "ASSUMED" in ent.uncertainty.note
    # resting on the floor: its base sits at the camera-height-below-origin level (relative units)
    assert abs(ent.transform["position"]["y"] + CAM_H) < 0.15 * CAM_H
    # unit honesty: dimensions are model units, never metres, in a relative world
    assert ent.custom_properties["dimensions"]["length_m"]["unit"] == "model_units"


def test_unknown_label_is_rejected_with_a_reason_not_dropped(tmp_path):
    item, facts, dm = _room(tmp_path)
    res = bootstrap_single_image(item, facts, depth_map=dm, furniture_detections=[_sofa("spaceship")])
    assert res.facts["furniture"]["placed"] == 0
    assert res.facts["furniture"]["rejected"][0]["reason"] == "unknown furniture label"


def test_no_detector_and_failing_detector_are_recorded_not_faked(tmp_path):
    item, facts, dm = _room(tmp_path)
    assert bootstrap_single_image(item, facts, depth_map=dm).facts["furniture"]["attempted"] is False

    def boom(*a):
        raise RuntimeError("weights missing")

    f2 = bootstrap_single_image(item, facts, depth_map=dm, furniture_detector=boom).facts["furniture"]
    assert f2["attempted"] is False and "weights missing" in f2["reason"]


def test_overlapping_detections_never_become_duplicate_entities(tmp_path):
    """Two detectors can call one object two things; the validator rejects same-type overlap, which used to fail the job."""
    item, facts, dm = _room(tmp_path)
    dets = [_sofa(), FurnitureDetection(label="armchair", score=0.6, box_xyxy=(102, 131, 222, 201), view_id="ev-room")]
    res = bootstrap_single_image(item, facts, depth_map=dm, furniture_detections=dets)
    ff = res.facts["furniture"]
    assert ff["placed"] == 1
    assert any("duplicate" in r.get("reason", "") for r in ff["rejected"]), ff["rejected"]
    from world_ir.validation import validate_world_ir

    dup = [i for i in validate_world_ir(res.world).issues if i.code == "invariant_6_duplicate_entity"]
    assert not dup, dup


def test_placed_furniture_carries_the_label_the_room_summary_counts(tmp_path):
    """Studio's insights bar counted 0 furniture because the entities lacked 'furniture_candidate'."""
    item, facts, dm = _room(tmp_path)
    res = bootstrap_single_image(item, facts, depth_map=dm, furniture_detections=[_sofa()])
    ent = next(e for e in res.world.entities.values() if e.type.value == "furniture")
    assert "furniture_candidate" in ent.semantic_labels
