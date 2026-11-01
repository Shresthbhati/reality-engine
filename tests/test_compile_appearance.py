"""M1/M2 wiring: the compiler measures surface colour from posed photos (CompileOptions.appearance_views) and only gives a
photo colour to a surface a photo actually saw. SYNTHETIC fixture; asserts wiring, not real-room accuracy."""

from __future__ import annotations

import cv2
import numpy as np

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from evidence.session import EvidenceItem, EvidenceKind
from perception.materials.room_surfaces import views_from_photos
from synthetic.indoor import fidelity_room, observe
from tests.test_room_surfaces import GREEN, H, W, _quat_looking_at
from world_ir import EntityType
from world_ir.artifact_store import MemoryArtifactStore

UP = (0.0, 0.0, 1.0)
SURFACES = (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)


def _compile(tmp_path, with_photo: bool):
    # colourless points: any colour on a surface can only have come from the photo
    obs = observe(fidelity_room(), colors=None, voxel_m=0.08)
    views = ()
    if with_photo:
        img = tmp_path / "green.png"
        cv2.imwrite(str(img), np.full((H, W, 3), (GREEN[2], GREEN[1], GREEN[0]), np.uint8))
        item = EvidenceItem(id="ev-g", kind=EvidenceKind.PHOTO, source_uri=str(img),
                            metadata={"intrinsics": {"fx": 110.0, "fy": 110.0}})
        pose = ("ev-g", (2.5, 2.0, 1.3), _quat_looking_at((2.5, 2.0, 1.3), (2.5, 0.0, 1.3)))
        views, _, _ = views_from_photos([pose], {"ev-g": item})
    return compile_reconstruction_to_world(
        obs.result, CompileOptions(seed=42, up=UP, artifact_store=MemoryArtifactStore(), appearance_views=tuple(views)))


def test_no_photo_means_no_surface_gets_a_colour(tmp_path):
    world, diag = _compile(tmp_path, with_photo=False)
    assert "surface_appearance" not in world.metadata
    for e in world.entities.values():
        if e.type in SURFACES and "slab" in " ".join(e.semantic_labels):
            assert not e.custom_properties.get("mean_color_rgb"), e.id


def test_compiler_measures_colour_only_where_a_photo_saw_the_surface(tmp_path):
    world, diag = _compile(tmp_path, with_photo=True)
    assert not [w for w in diag.interior_warnings if "surface appearance" in w], diag.interior_warnings
    summary = world.metadata["surface_appearance"]
    assert summary["available"] is True and summary["measured"] >= 1
    slabs = [e for e in world.entities.values() if e.type in SURFACES and e.custom_properties.get("created_by") == "room_layout"]
    measured = [e for e in slabs if e.custom_properties.get("color_state") == "measured_approximate"]
    unseen = [e for e in slabs if e.custom_properties.get("color_state") == "not_measured"]
    assert measured and unseen, "one photo must see some surfaces and miss others"
    for e in measured:
        cp = e.custom_properties
        assert cp["color_source"] == "image_projected" and cp["mean_color_rgb"][1] > cp["mean_color_rgb"][0]
        assert cp["material_hypotheses"] and all(h["state"] == "ESTIMATED" and not h["asserted"] for h in cp["material_hypotheses"])
    for e in unseen:                      # nothing seen it: no colour, no hypothesis -- never a guess
        assert not e.custom_properties.get("mean_color_rgb") and "material_hypotheses" not in e.custom_properties
