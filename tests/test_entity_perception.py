"""Entity perception phase 1: taxonomy, small-object inspection, multi-view association, switchboard modules, completeness.

Support claim honestly limited to what these tests exercise: SYNTHETIC inputs. No learned detector is installed, so
real-photo detection of sockets/switches is NOT verified here.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.detection.backends import FakeDetector
from perception.detection.interface import ImageDetection
from perception.entities import taxonomy
from perception.entities.association import EntityObservation, associate
from perception.entities.completeness import completeness_report
from perception.entities.small_objects import inspect_small_objects, tile_boxes
from perception.entities.switchboard import ModuleObservation, PlateObservation, build_switchboard
from world_ir.schema_v1 import RelationshipKind
from world_ir.world_v1 import WorldIR

SOCKET, SWITCH = "electrical.switchboard.module.socket", "electrical.switchboard.module.switch"


# ------------------------------------------------------------------ taxonomy
def test_taxonomy_covers_all_eighteen_categories_with_stable_unique_ids():
    assert {c.category for c in taxonomy.all_classes()} == set(taxonomy.CATEGORIES) and len(taxonomy.CATEGORIES) == 18
    ids = [c.id for c in taxonomy.all_classes()]
    assert len(ids) == len(set(ids)) and len(ids) > 100
    assert all(c.parent is None or c.parent in ids for c in taxonomy.all_classes())


def test_small_objects_the_user_named_are_registered_as_small_and_map_from_detector_labels():
    for cid in (SOCKET, SWITCH, "electrical.switchboard", "electrical.junction_box", "opening.door.handle",
                "opening.door.hinge", "opening.door.lock", "opening.window.handle", "plumbing.pipe", "electrical.conduit",
                "electrical.switchboard.module.usb"):
        assert taxonomy.get(cid).small, cid
    assert taxonomy.class_for_label("Wall Socket").id == SOCKET
    assert taxonomy.class_for_label("door hinge").id == "opening.door.hinge"
    assert taxonomy.class_for_label("spaceship") is None
    assert taxonomy.ancestors(SOCKET) == ["electrical.switchboard.module", "electrical.switchboard"]


def test_a_class_is_not_called_supported_just_because_it_is_registered():
    assert taxonomy.support_level("plumbing.pipe") == "registered"
    assert "synthetic only" in taxonomy.support_level(SOCKET)


# ------------------------------------------------------------------ small-object inspection (coarse-to-fine)
def _coded_image(w=2400, h=1600):
    """Pixel (x, y) stores (x//10, y//10) in R, G so a fake detector can tell which tile it was handed."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (np.arange(w)[None, :] // 10).astype(np.uint8)
    img[..., 1] = (np.arange(h)[:, None] // 10).astype(np.uint8)
    return img


def _sees_socket_only_at_full_resolution(gx=1000, gy=700):
    """Like a real detector: a ~40 px socket vanishes in the downscaled photo but shows in a full-resolution tile."""
    def run(img, queries):
        h, w = img.shape[:2]
        if max(h, w) > 700 or "wall socket" not in queries:      # the 1024 px coarse image: the socket is not visible
            return []
        x0, y0 = int(img[0, 0, 0]) * 10, int(img[0, 0, 1]) * 10
        if x0 <= gx - 20 and gx + 20 <= x0 + w and y0 <= gy - 20 and gy + 20 <= y0 + h:
            return [ImageDetection("wall socket", 0.8, (gx - x0 - 20, gy - y0 - 20, gx - x0 + 20, gy - y0 + 20))]
        return []
    return run


def test_fine_pass_finds_a_small_socket_the_coarse_pass_misses_with_original_pixel_coordinates():
    res = inspect_small_objects(_coded_image(), FakeDetector(_sees_socket_only_at_full_resolution()), tile=640)
    socks = [d for d in res.detections if d.class_id == SOCKET]
    assert len(socks) == 1 and socks[0].stage == "fine"                    # overlapping tiles de-duplicated
    x0, y0, x1, y1 = socks[0].box
    assert abs((x0 + x1) / 2 - 1000) <= 12 and abs((y0 + y1) / 2 - 700) <= 12     # tile-origin coding is 10 px coarse


def test_fine_pass_is_limited_to_regions_and_reports_the_work_it_skipped():
    full = inspect_small_objects(_coded_image(), FakeDetector({}), tile=640)
    part = inspect_small_objects(_coded_image(), FakeDetector({}), tile=640, regions=[(0, 0, 700, 700)])
    assert full.tiles_run == full.tiles_total == len(tile_boxes(2400, 1600, 640)) and 0 < part.tiles_run < full.tiles_total


def test_without_a_detector_the_inspection_says_so_instead_of_returning_nothing_silently():
    r = inspect_small_objects(_coded_image(), None)
    assert r.status == "unavailable" and r.detections == [] and "no detector" in r.note


def test_unknown_labels_are_reported_not_invented_into_classes():
    r = inspect_small_objects(_coded_image(800, 600), FakeDetector({"spaceship": [((1, 1, 50, 50), 0.9)]}))
    assert r.detections == [] and "spaceship" in r.unmapped_labels


# ------------------------------------------------------------------ multi-view association
def _o(oid, ev, cls, x, y=1.0, z=0.0, sigma=0.004):
    return EntityObservation(oid, ev, cls, (x, y, z), sigma_m=sigma)


def test_a_row_of_three_sockets_seen_in_two_photos_is_three_entities_each_with_two_observations():
    xs = [0.0, 0.08, 0.16]                                                # 8 cm pitch, 2 mm noise between views
    obs = [_o(f"a{i}", "img1", SOCKET, x) for i, x in enumerate(xs)] + \
          [_o(f"b{i}", "img2", SOCKET, x + 0.002) for i, x in enumerate(xs)]
    clusters = associate(obs)
    assert len(clusters) == 3 and all(len(c.members) == 2 for c in clusters)
    assert all(c.association_confidence is not None and c.association_confidence > 0.5 for c in clusters)
    assert all(c.evidence_ids == ["img1", "img2"] for c in clusters)


def test_two_detections_in_one_image_are_never_merged_even_when_touching():
    assert len(associate([_o("a", "img1", SOCKET, 0.0), _o("b", "img1", SOCKET, 0.003)])) == 2


def test_different_classes_never_merge_and_distant_same_class_never_merges():
    assert len(associate([_o("a", "img1", SOCKET, 0.0), _o("b", "img2", SWITCH, 0.001)])) == 2
    assert len(associate([_o("a", "img1", SOCKET, 0.0), _o("b", "img2", SOCKET, 2.0)])) == 2


def test_ambiguity_keeps_entities_apart_rather_than_merging_wrongly():
    obs = [_o("a", "img1", SOCKET, 0.050), _o("b1", "img2", SOCKET, 0.047), _o("b2", "img2", SOCKET, 0.053)]
    clusters = associate(obs, gate_sigmas=3.0)
    assert all(len(c.members) == 1 for c in clusters) and any(c.ambiguous_with for c in clusters)


def test_appearance_can_veto_a_merge():
    a = EntityObservation("a", "i1", SOCKET, (0, 0, 0), 0.01, descriptor=(1.0, 0.0))
    b = EntityObservation("b", "i2", SOCKET, (0.002, 0, 0), 0.01, descriptor=(0.0, 1.0))
    assert len(associate([a, b])) == 2


# ------------------------------------------------------------------ switchboard modules
PLATE = PlateObservation(center=(2.0, 1.2, 0.0), normal=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), size_m=(0.146, 0.086),
                         depth_m=0.012, confidence=0.8, evidence_ids=("img1", "img2"), host_wall_id="wall-1")


def _two_gang(**kw):
    mods = [ModuleObservation(SOCKET, (2.0 + 0.035, 1.2, 0.0), (0.05, 0.05), 0.7, ("img1",), {"earth_pin": "unknown"}),
            ModuleObservation(SWITCH, (2.0 - 0.035, 1.2, 0.0), (0.04, 0.07), 0.9, ("img1", "img2"))]
    return build_switchboard(PLATE, mods, entity_id="sb-1", metric=True, **kw)


def test_switchboard_is_a_plate_with_individually_identifiable_modules_not_one_box():
    b = _two_gang(complete=True)
    plate, mods = b.entities[0], b.entities[1:]
    assert plate.custom_properties["entity_class"] == "electrical.switchboard" and len(mods) == 2
    assert plate.component_ids == [m.id for m in mods] and plate.custom_properties["module_count"] == 2
    assert plate.custom_properties["module_count_state"] == "complete"
    rows = plate.custom_properties["arrangement"]["rows"]
    assert len(rows) == 1 and len(rows[0]) == 2
    by_class = {m.custom_properties["entity_class"]: m.id for m in mods}
    assert set(by_class) == {SOCKET, SWITCH}
    # with the plate normal +z and up +y, 'right' = up x normal = +x: the switch (x-0.035) is left of the socket (x+0.035)
    assert rows[0] == [by_class[SWITCH], by_class[SOCKET]]
    assert all(any(r.kind == RelationshipKind.PART_OF and r.target_id == "sb-1" for r in m.relationships) for m in mods)
    assert any(r.kind == RelationshipKind.ATTACHED_TO and r.target_id == "wall-1" for r in plate.relationships)
    w = plate.custom_properties["dimensions"]["width"]
    assert w["value"] == pytest.approx(0.146) and w["state"] == "RECONSTRUCTED" and w["unit"] == "m"   # never MEASURED


def test_unknown_stays_unknown_and_an_incomplete_module_list_is_a_lower_bound():
    plate = PlateObservation(center=(0, 1, 0), normal=(0, 0, 1), confidence=0.6, evidence_ids=("img1",))
    b = build_switchboard(plate, [ModuleObservation(SOCKET, (0, 1, 0), None, 0.6, ("img1",))], entity_id="sb-2")
    p, m = b.entities
    w = p.custom_properties["dimensions"]["width"]
    assert w["value"] is None and w["state"] == "UNKNOWN" and w["unit"] == "model_units"
    assert p.custom_properties["module_count_state"] == "lower_bound" and p.custom_properties["reconstruction_status"] == "partial"
    assert set(p.custom_properties["unknown_attributes"]) >= {"size", "protrusion", "host_wall"}
    assert m.custom_properties["dimensions"]["width"]["value"] is None and m.custom_properties["reconstruction_status"] == "partial"


def test_implausible_or_foreign_modules_are_rejected_and_reported_and_outside_modules_are_flagged():
    mods = [ModuleObservation(SOCKET, (2.0, 1.2, 0.0), (0.9, 0.9), 0.9, ("img1",)),             # a 90 cm "socket"
            ModuleObservation("furniture.sofa", (2.0, 1.2, 0.0), None, 0.9, ("img1",)),
            ModuleObservation(SWITCH, (2.5, 1.2, 0.0), (0.04, 0.07), 0.9, ("img1",))]            # off the plate
    b = build_switchboard(PLATE, mods, entity_id="sb-3", metric=True)
    assert len(b.entities) == 2 and len(b.rejected) == 2 and any("outside" in w for w in b.warnings)


def test_a_two_row_board_keeps_its_arrangement():
    mods = [ModuleObservation(SOCKET, (2.0 - 0.03, 1.2 + 0.04, 0.0), (0.05, 0.05), 0.8, ("i",)),
            ModuleObservation(SOCKET, (2.0 + 0.03, 1.2 + 0.04, 0.0), (0.05, 0.05), 0.8, ("i",)),
            ModuleObservation(SWITCH, (2.0, 1.2 - 0.04, 0.0), (0.04, 0.07), 0.8, ("i",))]
    b = build_switchboard(PlateObservation((2.0, 1.2, 0.0), (0, 0, 1), size_m=(0.146, 0.146), confidence=0.8), mods,
                          entity_id="sb-4", metric=True, complete=True)
    assert b.plate.custom_properties["arrangement"]["columns_per_row"] == [2, 1]


def test_switchboard_entities_survive_a_worldir_roundtrip_with_nothing_lost():
    b = _two_gang(complete=True)
    w = WorldIR(id="w", name="w", entities={e.id: e for e in b.entities})
    w2 = WorldIR.from_dict(w.to_dict())
    p = w2.entities["sb-1"]
    assert p.custom_properties["module_count"] == 2 and p.component_ids == b.plate.component_ids
    assert w2.entities[p.component_ids[0]].relationships[0].kind == RelationshipKind.PART_OF
    assert "electrical.switchboard" in p.semantic_labels
    assert p.custom_properties["parametric"]["frame"]["normal"] == [0.0, 0.0, 1.0]


# ------------------------------------------------------------------ completeness
def test_completeness_never_claims_done_while_anything_is_missing_occluded_or_uninspected():
    ents = [{"class_id": "envelope.wall", "status": "reconstructed"}] * 4 + [
        {"class_id": "envelope.floor", "status": "reconstructed"}, {"class_id": "opening.door", "status": "partial"}]
    r = completeness_report("r1", "room", ents, inspected_fraction=0.8, occluded_fraction=0.1)
    assert r["complete_claim"] is False and r["completeness_estimate"] <= 0.95
    assert any("electrical.switchboard" in g for g in r["gaps"]) and any("never inspected" in g for g in r["gaps"])
    assert any("occluded" in g for g in r["gaps"]) and any("partially" in g for g in r["gaps"])


def test_everything_detected_and_reconstructed_is_still_not_complete_if_expected_categories_are_absent():
    r = completeness_report("r2", "room", [{"class_id": "envelope.wall", "status": "reconstructed"}] * 4, inspected_fraction=1.0)
    assert r["detected"] == r["reconstructed"] == 4 and r["complete_claim"] is False
    assert any(row["status"] == "missing_evidence" for row in r["expected"])


def test_a_fully_evidenced_region_is_complete_only_when_no_gap_remains():
    r = completeness_report("r3", "room", [{"class_id": "envelope.wall", "status": "reconstructed"}] * 4,
                            inspected_fraction=1.0, expected={"envelope.wall": (4, None)})
    assert r["complete_claim"] is True and r["completeness_estimate"] <= 0.95
