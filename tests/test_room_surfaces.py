"""M2 integration: solid room surfaces -> image-projected appearance, and the honest "no photo" state.

SYNTHETIC: a real compiled fidelity room plus a photo rendered from inside it. Asserts wiring and the measured colour,
not real-room accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evidence.session import EvidenceItem, EvidenceKind
from perception.materials.room_surfaces import (
    ASSUMED_HFOV_DEG, INTRINSICS_SOURCE_ASSUMED, INTRINSICS_SOURCE_METADATA, measure_world_appearance,
    surfaces_from_world, views_from_photos,
)
from tests.test_room_fidelity_fixture import _compiled
from world_ir import EntityType
from world_ir.artifact_store import MemoryArtifactStore

W, H = 160, 120
GREEN = (30, 200, 30)


def _quat_looking_at(pos, target, up=(0, 0, 1)):
    """Camera (x right, y down, z forward) at pos looking at target -> (w, x, y, z)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    return (float(qw), float(qx), float(qy), float(qz))


def test_surfaces_come_from_the_solid_slabs_not_the_evidence_planes():
    _obs, world, _diag = _compiled("albedo", False)
    surfaces, by_id, warnings = surfaces_from_world(world)
    assert not warnings
    walls = [s for s in surfaces if s.kind == "wall"]
    assert len(walls) == 4 and len(surfaces) == 6        # 4 wall slabs + floor + ceiling
    for s in surfaces:
        assert s.surface_id in by_id
        assert s.interior_point is not None, s.surface_id
        assert np.isfinite(s.polygon).all() and len(s.polygon) >= 3
    ids = {s.surface_id for s in surfaces}
    assert all(e.custom_properties.get("wall_slab") for e in by_id.values()
               if e.type == EntityType.WALL and e.id in ids)


def test_no_photo_is_reported_and_never_invents_colour():
    _obs, world, _diag = _compiled("albedo", False)
    before = {e.id: tuple(e.custom_properties.get("mean_color_rgb") or ())
              for e in world.entities.values() if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)}
    summary = measure_world_appearance(world, [])
    assert summary["available"] is False and "no posed photo" in summary["reason"]
    after = {e.id: tuple(e.custom_properties.get("mean_color_rgb") or ())
             for e in world.entities.values() if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)}
    assert before == after, "an unavailable measurement must not change a surface colour"


def test_views_use_metadata_intrinsics_and_skip_unreadable_photos(tmp_path):
    import cv2

    img_path = tmp_path / "inside.png"
    cv2.imwrite(str(img_path), np.full((H, W, 3), (GREEN[2], GREEN[1], GREEN[0]), np.uint8))
    item = EvidenceItem(id="ev-1", kind=EvidenceKind.PHOTO, source_uri=str(img_path),
                        metadata={"intrinsics": {"fx": 110.0, "fy": 110.0, "cx": W / 2, "cy": H / 2}})
    pose = ("ev-1", (2.5, 2.0, 1.3), _quat_looking_at((2.5, 2.0, 1.3), (2.5, 0.0, 1.3)))
    views, sources, skipped = views_from_photos([pose], {"ev-1": item})
    assert len(views) == 1 and sources == [INTRINSICS_SOURCE_METADATA] and not skipped
    assert views[0].image_rgb.shape == (H, W, 3)
    # a missing file is skipped with a reason, never a fabricated camera
    views2, _, skipped2 = views_from_photos([("ev-missing", (0, 0, 0), (1, 0, 0, 0))],
                                            {"ev-missing": EvidenceItem("ev-missing", EvidenceKind.PHOTO, str(tmp_path / "nope.png"))})
    assert views2 == [] and skipped2 and "not locally readable" in skipped2[0]


def test_assumed_intrinsics_are_labelled_when_metadata_is_absent(tmp_path):
    import cv2

    img_path = tmp_path / "plain.png"
    cv2.imwrite(str(img_path), np.full((H, W, 3), 128, np.uint8))
    item = EvidenceItem(id="e", kind=EvidenceKind.PHOTO, source_uri=str(img_path))
    views, sources, _ = views_from_photos([("e", (0, 0, 0), (1, 0, 0, 0))], {"e": item})
    assert len(views) == 1 and sources == [INTRINSICS_SOURCE_ASSUMED]
    f = (W / 2.0) / math.tan(math.radians(ASSUMED_HFOV_DEG / 2.0))
    assert views[0].camera.intrinsics.fx == pytest.approx(f)


def test_measured_colour_reaches_the_wall_entity(tmp_path):
    import cv2

    _obs, world, _diag = _compiled("albedo", False)
    img_path = tmp_path / "green.png"
    cv2.imwrite(str(img_path), np.full((H, W, 3), (GREEN[2], GREEN[1], GREEN[0]), np.uint8))
    item = EvidenceItem(id="ev-g", kind=EvidenceKind.PHOTO, source_uri=str(img_path),
                        metadata={"intrinsics": {"fx": 110.0, "fy": 110.0}})
    pose = ("ev-g", (2.5, 2.0, 1.3), _quat_looking_at((2.5, 2.0, 1.3), (2.5, 0.0, 1.3)))
    views, sources, _ = views_from_photos([pose], {"ev-g": item})
    store = MemoryArtifactStore()
    summary = measure_world_appearance(world, views, artifact_store=store, intrinsics_sources=sources)
    assert summary["available"] is True and summary["measured"] >= 1 and summary["textured"] >= 1
    measured = [e for e in world.entities.values()
                if (e.custom_properties or {}).get("color_source") == "image_projected"]
    assert measured
    for e in measured:
        assert str(e.custom_properties["color_state"]).startswith("measured") or e.custom_properties["color_state"] == "weak"
        assert "appearance" in e.custom_properties
        rgb = e.custom_properties["mean_color_rgb"]
        assert abs(rgb[1] - GREEN[1]) < 12 and rgb[1] > rgb[0] and rgb[1] > rgb[2], rgb
        tex = e.custom_properties["appearance"]["texture"]
        assert tex and tex.get("hash") and tex.get("observed_fraction", 0) > 0
