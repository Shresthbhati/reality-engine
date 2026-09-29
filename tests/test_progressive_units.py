"""Fast (model-free) proofs for the evidence-progressive product path.

Covers: image validation/classification, evidence contribution, the
user-facing state machine, guidance derived from measurements, diff
categories, and the camera-up gravity prior. The real-photo, real-COLMAP,
real-MiDaS journeys live in tests/integration/test_progressive_product_journey.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.api.reconstruction_state import batch_headline, derive_state, failure_message
from engine.pipeline.guidance import build_guidance, coverage_degrees
from evidence.contribution import ContributionReport, analyze_contribution, quality_levels
from evidence.image_check import inspect_image
from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3, WorldIR, diff_worlds

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "south_building" / "images"


def _noise_photo(path: Path, seed: int = 0, size=(320, 240)) -> Path:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    Image.fromarray(base).resize(size, Image.BICUBIC).save(path, "JPEG")
    return path


# ---------------------------------------------------------------- image check


def test_garbage_bytes_are_not_images(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"jpeg-" + b"0" * 8)
    facts = inspect_image(p)
    assert not facts.ok and "not a decodable image" in facts.reason
    assert not facts.geometry_eligible


def test_tiny_image_rejected(tmp_path):
    p = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(p)
    assert "too small" in inspect_image(p).reason


def test_flat_graphic_is_context_not_geometry(tmp_path):
    """A floor plan / infographic (few flat colours) is classified, kept as
    context, and never becomes geometry evidence."""
    p = tmp_path / "plan.png"
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    for x in range(0, 400, 40):
        for y in range(100, 110):
            img.putpixel((x, y), (0, 0, 0))
    img.save(p)
    facts = inspect_image(p)
    assert facts.ok and facts.evidence_class == "flat_graphic"
    assert not facts.geometry_eligible


def test_natural_image_is_photograph_unverified_not_photograph(tmp_path):
    facts = inspect_image(_noise_photo(tmp_path / "n.jpg"))
    assert facts.evidence_class == "photograph_unverified"  # no EXIF => not claimed as verified
    assert facts.geometry_eligible


def test_declared_class_overrides_and_is_recorded(tmp_path):
    p = _noise_photo(tmp_path / "n.jpg")
    facts = inspect_image(p, "render")
    assert facts.evidence_class == "render" and facts.class_basis == "declared by uploader"
    assert not facts.geometry_eligible  # a render is a visual reference, not observed geometry
    assert not inspect_image(p, "hologram").ok


def test_exif_gps_is_read_only_when_present(tmp_path):
    p = tmp_path / "gps.jpg"
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "TestMake", "TestCam"
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (28.0, 36.0, 0.0), "E", (77.0, 12.0, 0.0)
    img.save(p, exif=exif)
    facts = inspect_image(p)
    assert facts.evidence_class == "photograph"
    assert facts.gps == pytest.approx((28.6, 77.2), abs=1e-6)
    assert inspect_image(_noise_photo(tmp_path / "nogps.jpg")).gps is None  # never invented


# --------------------------------------------------------------- contribution


@pytest.mark.skipif(not DATASET.exists(), reason="real photo dataset not present")
def test_contribution_labels_duplicate_new_view_and_unrelated():
    imgs = sorted(DATASET.glob("*.JPG"))
    rep = analyze_contribution([("a", imgs[15]), ("dup", imgs[15]), ("b", imgs[16]), ("far", imgs[0])])
    labels = {c.evidence_id: c.label for c in rep.per_image}
    assert labels["a"] == "first"
    assert labels["dup"] == "redundant"
    assert labels["b"] == "new_view"
    assert labels["far"] in ("disconnected", "new_view")  # measured, never invented
    assert rep.summary()["redundant"] == 1


@pytest.mark.skipif(not DATASET.exists(), reason="real photo dataset not present")
def test_prior_evidence_is_not_counted_as_new():
    imgs = sorted(DATASET.glob("*.JPG"))
    rep = analyze_contribution([("a", imgs[15]), ("b", imgs[16]), ("c", imgs[17])], prior_ids=["a", "b"])
    assert rep.summary()["new_images"] == 1


def test_quality_levels_unknown_when_not_measurable():
    q = {x["name"]: x for x in quality_levels(ContributionReport(available=False, note="cv2 missing"), 3, None)}
    assert q["Coverage"]["level"] == "unknown"
    assert q["Overlap"]["level"] == "unknown" and q["Spatial diversity"]["level"] == "unknown"


def test_single_image_quality_is_none_not_fabricated():
    q = {x["name"]: x for x in quality_levels(ContributionReport(available=True, note=""), 1, None)}
    assert q["Overlap"]["level"] == "none" and q["Spatial diversity"]["level"] == "none"


# ------------------------------------------------------------------ states


@pytest.mark.parametrize("status,stage,kind,expected", [
    (None, None, None, "CAPTURING"),
    ("queued", None, None, "ANALYZING"),
    ("running", "analyzing_evidence", None, "ANALYZING"),
    ("running", "building_rough_model", None, "BUILDING_ROUGH_MODEL"),
    ("running", "reconstructing", None, "RECONSTRUCTING"),
    ("running", "refining", None, "REFINING"),
    ("succeeded", None, None, "READY_TO_INSPECT"),
    ("partial", None, None, "PARTIALLY_COMPLETE"),
    ("failed", None, "insufficient_evidence", "NEEDS_MORE_EVIDENCE"),
    ("failed", None, None, "FAILED"),
    ("cancelled", None, None, "CANCELLED"),
    ("weird-new-status", None, None, "FAILED"),
])
def test_state_machine(status, stage, kind, expected):
    assert derive_state(status, stage, kind) == expected


def test_timeout_and_partial_never_read_as_complete():
    for status in ("failed", "cancelled", "partial", "queued", "running"):
        assert derive_state(status, None, None) != "READY_TO_INSPECT"


def test_failure_message_strips_traceback_and_class():
    tb = "Traceback (most recent call last):\n  File x\nRuntimeError: no usable photos"
    assert failure_message(tb) == "no usable photos"
    assert failure_message(None) is None


def test_batch_headline_is_measured_not_a_count_of_files():
    s = {"new_images": 4, "new_view": 3, "redundant": 1, "disconnected": 0}
    assert batch_headline(s) == "4 new images added. 3 add a new viewpoint. 1 mostly repeats an existing view."
    assert batch_headline({"new_images": 0}) is None


# ---------------------------------------------------------------- guidance


def _pose(eid, pos, yaw_deg):
    # camera-to-world quaternion (w,x,y,z), yaw about world +Y; looks along local +Z
    a = math.radians(yaw_deg) / 2
    return (eid, pos, (math.cos(a), 0.0, math.sin(a), 0.0))


def test_one_sided_capture_is_reported_as_one_sided_not_as_a_gap_between_photos():
    # cameras on the +Z side of a scene at the origin, all looking toward it
    poses = [_pose(f"c{i}", (x, 0.0, 10.0), 180.0) for i, x in enumerate((-3.0, -1.0, 1.0, 3.0))]
    cov = coverage_degrees(poses, (0.0, 0.0))
    assert cov["mode"] == "around_scene" and cov["degrees"] < 60
    g = build_guidance(level=2, entity_roles=["wall"], names={}, input_ids=[p[0] for p in poses],
                       registered_ids=[p[0] for p in poses], camera_poses=poses,
                       contribution_summary=None, bootstrap_facts=None, attempts=[],
                       scene_center=(0.0, 0.0))
    msg = " ".join(x["message"] for x in g)
    assert "one side" in msg and "sides and back" in msg
    assert all(x["basis"] for x in g)  # every item states what it rests on


def test_unregistered_images_are_named():
    poses = [_pose("a", (0, 0, 0), 0), _pose("b", (1, 0, 0), 10)]
    g = build_guidance(level=2, entity_roles=["wall", "floor"], names={"c": "IMG_9.jpg"},
                       input_ids=["a", "b", "c"], registered_ids=["a", "b"], camera_poses=poses,
                       contribution_summary=None, bootstrap_facts=None, attempts=[])
    assert any("IMG_9.jpg" in x["message"] and x["kind"] == "retake" for x in g)


def test_single_view_guidance_uses_measured_frame_truncation():
    facts = {"edge_truncation": {"left": 0.0, "right": 0.31, "top": 0.0, "bottom": 0.0}}
    g = build_guidance(level=0, entity_roles=["wall"], names={}, input_ids=["a"], registered_ids=["a"],
                       camera_poses=[], contribution_summary=None, bootstrap_facts=facts, attempts=[])
    text = " ".join(x["message"] for x in g)
    assert "right edge" in text and "left edge" not in text
    assert "unobserved" in text


def test_redundancy_advice_fires_only_when_measured():
    poses = [_pose("a", (0, 0, 0), 0), _pose("b", (1, 0, 0), 10)]
    kw = dict(level=2, entity_roles=["floor", "ceiling"], names={}, input_ids=["a", "b"],
              registered_ids=["a", "b"], camera_poses=poses, bootstrap_facts=None, attempts=[])
    assert not any(x["kind"] == "redundant" for x in build_guidance(
        contribution_summary={"new_images": 4, "redundant": 1}, **kw))
    assert any(x["kind"] == "redundant" for x in build_guidance(
        contribution_summary={"new_images": 4, "redundant": 3}, **kw))


# -------------------------------------------------------------------- diff


def test_confidence_only_change_is_not_a_geometry_change():
    def world(conf):
        w = WorldIR(id="w", name="w", main_branch_id="b")
        w.geometries["g"] = Geometry(id="g", type=GeometryType.PLANE, bounds_min=Vector3(0, 0, 0),
                                     bounds_max=Vector3(1, 1, 1), provenance=Provenance.ESTIMATED, confidence=0.3)
        w.entities["e"] = Entity(id="e", type=EntityType.WALL, geometry_ids=["g"],
                                 provenance=Provenance.INFERRED, confidence=conf)
        return w
    cats = diff_worlds(world(0.3), world(0.6)).categories()
    assert cats["confidence"] == 1
    assert cats["geometry"] == 0 and cats["identity"] == 0 and cats["provenance"] == 0


def test_geometry_change_and_identity_change_are_separated():
    a = WorldIR(id="w", name="w", main_branch_id="b")
    b = WorldIR(id="w", name="w", main_branch_id="b")
    for w, hi in ((a, 1.0), (b, 2.0)):
        w.geometries["g"] = Geometry(id="g", type=GeometryType.PLANE, bounds_min=Vector3(0, 0, 0),
                                     bounds_max=Vector3(hi, 1, 1), provenance=Provenance.ESTIMATED, confidence=0.3)
    b.entities["new"] = Entity(id="new", type=EntityType.WALL, provenance=Provenance.INFERRED, confidence=0.3)
    cats = diff_worlds(a, b).categories()
    assert cats["geometry"] == 1 and cats["identity"] == 1 and cats["confidence"] == 0


# ------------------------------------------------------------- frame prior


def _camera_to_world(forward, up=(0.0, 1.0, 0.0)):
    f = np.array(forward, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = -np.cross(r, f)
    return np.column_stack([r, d, f])


def _facade_result(pitch_deg: float, diverse: bool = True):
    from reconstruction.backend.interface import (
        ReconstructedCameraPose,
        ReconstructedPoint,
        ReconstructionResult,
    )
    from reconstruction.frame import _rotmat_to_qvec

    rng = np.random.default_rng(3)
    # a wall in the x=0 plane, rising from y=0 to 12
    pts = [ReconstructedPoint(position=(float(rng.normal(0, 0.01)), float(y), float(z)),
                              track_id=f"p{i:04d}", source_evidence_ids=["c0"])
           for i, (y, z) in enumerate(zip(rng.uniform(0, 12, 400), rng.uniform(-8, 8, 400)))]
    poses = []
    for i in range(6):
        yaw = (i - 2.5) * (6.0 if diverse else 0.0)
        p = math.radians(pitch_deg)
        fwd = (-math.cos(p) * math.cos(math.radians(yaw)), math.sin(p), math.cos(p) * math.sin(math.radians(yaw)))
        poses.append(ReconstructedCameraPose(
            evidence_id=f"c{i}", position=(12.0, 1.6, (i - 2.5) * 3.0),
            rotation=_rotmat_to_qvec(_camera_to_world(fwd)),
        ))
    return ReconstructionResult(points=pts, camera_poses=poses, registration_status="success")


def test_facade_with_pitched_cameras_gets_gravity_from_cameras_not_from_the_wall():
    from perception.geometry.planes import detect_planes
    from reconstruction.frame import canonicalize_frame

    rotated, rec = canonicalize_frame(_facade_result(pitch_deg=16.0), seed=42)
    assert rec.up_source == "camera_up_vectors+vertical_plane"
    plane = detect_planes(rotated, seed=42).planes[0]
    assert abs(plane.normal[1]) < 0.05, "the facade must end up VERTICAL (normal horizontal), not a floor"


def test_identical_camera_rotations_keep_the_dominant_plane_prior():
    from reconstruction.frame import canonicalize_frame

    _, rec = canonicalize_frame(_facade_result(pitch_deg=16.0, diverse=False), seed=42)
    assert rec.up_source == "dominant_plane"  # no orientation diversity => camera-up carries no evidence
