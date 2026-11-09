"""M3: material hypotheses are ranked, low-confidence, ESTIMATED -- and absent when the colour was never measured."""

import pytest

from perception.materials.material_hypotheses import CONFIDENCE_CAP, hypothesize_materials


def test_warm_brown_floor_ranks_wood_first_but_only_as_a_hypothesis():
    hyps = hypothesize_materials((150, 110, 70), "floor", spread_delta_e=3.0)
    assert [h["material"] for h in hyps][0] == "wood_floor"
    assert [h["score"] for h in hyps] == sorted((h["score"] for h in hyps), reverse=True)
    for h in hyps:
        assert h["state"] == "ESTIMATED" and h["asserted"] is False
        assert 0 < h["confidence"] <= CONFIDENCE_CAP
        assert h["evidence"]["color_rgb"] == [150, 110, 70] and "not verified" in h["note"]


def test_grey_floor_prefers_hard_floor_over_wood():
    assert hypothesize_materials((128, 128, 128), "floor", spread_delta_e=2.0)[0]["material"] == "concrete"


def test_flat_light_wall_is_painted_and_uniformity_raises_the_score():
    flat = hypothesize_materials((235, 230, 220), "wall", spread_delta_e=2.0)[0]
    unknown = hypothesize_materials((235, 230, 220), "wall")[0]
    assert flat["material"] == "painted_wall" and flat["score"] > unknown["score"]
    assert hypothesize_materials((235, 230, 220), "ceiling")[0]["material"] == "painted_ceiling"


@pytest.mark.parametrize("kwargs", [{"color_rgb": None}, {"color_rgb": (150, 110, 70), "color_state": "not_measured"}])
def test_no_measured_colour_means_no_hypothesis(kwargs):
    assert hypothesize_materials(kwargs.pop("color_rgb"), "floor", **kwargs) == []


def test_hypotheses_attach_only_to_image_measured_entities():
    from perception.materials.material_hypotheses import attach_material_hypotheses
    from world_ir import Entity, EntityType

    point_only = Entity(id="w1", type=EntityType.WALL, name="w1")
    point_only.custom_properties["mean_color_rgb"] = [200, 60, 60]       # from points, not a photo
    assert attach_material_hypotheses(point_only) is None and "material_hypotheses" not in point_only.custom_properties
    photo = Entity(id="w2", type=EntityType.WALL, name="w2")
    photo.custom_properties.update({"mean_color_rgb": [235, 230, 220], "color_source": "image_projected",
                                    "appearance": {"color_state": "measured_approximate", "spread_delta_e": 2.0}})
    got = attach_material_hypotheses(photo)
    assert got and photo.custom_properties["material_hypotheses"][0]["material"] == "painted_wall"
    assert "material" not in photo.custom_properties
