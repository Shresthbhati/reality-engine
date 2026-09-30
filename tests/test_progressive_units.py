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


# ---- degenerate feature geometry must classify, never crash --------------------------------------
def test_verified_pair_degenerate_inputs_are_classified_not_raised(monkeypatch):
    import cv2
    import numpy as np
    from evidence import contribution as c

    class KP:
        def __init__(self, pt): self.pt = pt

    class M:
        def __init__(self, i, d, r):
            self.queryIdx = self.trainIdx = i
            self.distance = d

    des = np.zeros((10, 4), np.float32)
    # ten matches all landing on ONE image point: fundamental matrix is undefined
    same = [KP((5.0, 5.0)) for _ in range(10)]
    fa = (same, des, 100.0)
    monkeypatch.setattr(cv2.BFMatcher, "knnMatch",
                        lambda self, a, b, k=2: [[M(i, 0.1, 0), M(i, 1.0, 0)] for i in range(10)])
    n, shift, status = c._verified_pair(fa, fa)
    assert (n, shift, status) == (0, None, "DEGENERATE_GEOMETRY")

    # OpenCV itself raising is also a classification
    spread = [KP((float(i), float(i * i))) for i in range(10)]
    monkeypatch.setattr(cv2, "findFundamentalMat",
                        lambda *a, **k: (_ for _ in ()).throw(cv2.error("boom")))
    n, shift, status = c._verified_pair((spread, des, 100.0), (spread, des, 100.0))
    assert status == "DEGENERATE_GEOMETRY" and n == 0

    # too few ratio-test survivors is INSUFFICIENT, distinct from DEGENERATE
    monkeypatch.setattr(cv2.BFMatcher, "knnMatch", lambda self, a, b, k=2: [])
    assert c._verified_pair(fa, fa)[2] == "INSUFFICIENT_MATCHES"


# ---- multi-view failure must not collapse N photographs into 1 ------------------------------------
def test_multiview_failure_keeps_every_usable_photo(tmp_path):
    from engine.pipeline.progressive import EvidenceInput, run_progressive
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.depth.interface import DepthMap

    h, w = 60, 80
    # a ground plane receding to a back wall: real plane structure for the RANSAC stage
    vals = [[min(1.0, 0.15 + 0.8 * (r / h)) for _ in range(w)] for r in range(h)]
    dm = DepthMap(evidence_id="x", width=w, height=h, values=vals)

    inputs = []
    for k in range(3):
        p = _noise_photo(tmp_path / f"p{k}.jpg", seed=k)
        item = EvidenceItem(id=f"ev-{k}", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
        inputs.append(EvidenceInput(item=item, facts=inspect_image(p), name=p.name, path=p))

    def failing_slice(items, options):
        raise RuntimeError("no initial image pair")

    res = run_progressive(inputs, vs_options=None, vertical_slice_fn=failing_slice, bootstrap_depth_map=dm)

    assert res.level == 0 and res.model_state == "ROUGH"
    # every photograph appears in the world, each with its own visible surface and camera
    assert {f"boot{k}-visible-surface" for k in (1, 2, 3)} <= set(res.world.entities)
    assert {f"boot{k}-camera" for k in (1, 2, 3)} <= set(res.world.entities)
    # the fusion is declared display-only and no image is claimed as registered
    fusion = res.world.metadata["fusion"]
    assert fusion["layout"] == "display_only" and set(fusion["display_offsets"]) == {"ev-0", "ev-1", "ev-2"}
    assert res.registered_ids == [] and res.input_ids == ["ev-0", "ev-1", "ev-2"]
    assert any("independent single-view" in d for d in res.degraded)
    assert res.attempts[0]["outcome"] == "failed" and res.attempts[-1]["outcome"] == "succeeded"


# ---- the legacy two-image SfM gate must stay behind the progressive engine ----------------------
def test_only_known_modules_reach_the_two_image_orchestrator():
    """apps/api must reach multi-view only through run_progressive; a direct import
    of the orchestrator / vertical_slice from an API route would reintroduce the
    2-image gate for one-photo uploads."""
    import re

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for py in (root / "apps" / "api").glob("*.py"):
        text = py.read_text(encoding="utf-8")
        if py.name == "jobs.py":
            # the worker may build vertical_slice OPTIONS for run_progressive, nothing else
            assert "orchestrator" not in text.lower().replace("orchestrated", ""), py
            continue
        if re.search(r"^\s*(from|import)\s+reconstruction\.orchestrator", text, re.M):
            offenders.append(py.name)
    # routes_sessions.py only probes importability for /api/status (no reconstruction call)
    assert offenders in ([], ["routes_sessions.py"]), offenders


# ---- single image: openings are candidates found in pixels, and absence is a valid answer -----------
def _wall_scene(tmp_path, with_openings: bool):
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.depth.interface import DepthMap

    w, h = 320, 240
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 175, np.uint8) + rng.integers(0, 3, (h, w, 3), dtype=np.uint8)
    if with_openings:
        img[60:115, 100:165] = 55       # window-like: dark, not touching the bottom
        img[125:h, 215:265] = 45        # door-like: dark, taller than wide, touching the bottom
    p = tmp_path / ("wall_open.png" if with_openings else "wall_plain.png")
    Image.fromarray(img).save(p)
    # a vertical wall tilted about the vertical axis: z = c / (1 - m (u - cx) / f)
    f = (w / 2) / math.tan(math.radians(30))
    vals = [[(1 - 0.5 * (u - w / 2) / f) / 4.0 - 0.25 for u in range(w)] for _ in range(h)]  # d = 1/z - 0.25
    dm = DepthMap(evidence_id="x", width=w, height=h, values=vals)
    item = EvidenceItem(id="ev-wall", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    return item, inspect_image(p), dm


def test_single_view_wall_yields_opening_candidates_with_honest_confidence(tmp_path):
    from engine.pipeline.single_image import OPENING_CONFIDENCE_CEILING, bootstrap_single_image

    item, facts, dm = _wall_scene(tmp_path, with_openings=True)
    res = bootstrap_single_image(item, facts, depth_map=dm)
    ops = {e.id: e for e in res.world.entities.values()
           if (e.custom_properties.get("bootstrap") or {}).get("role") == "opening_candidate"}
    assert res.facts["openings"]["attempted"] and len(ops) >= 2, res.facts["openings"]
    kinds = {e.type.value for e in ops.values()}
    assert kinds == {"door", "window"}
    for e in ops.values():
        assert e.provenance == Provenance.INFERRED and e.confidence <= OPENING_CONFIDENCE_CEILING
        assert "heuristic, not recognition" in e.uncertainty.note
    # the unseen stays unknown: openings never replace the UNKNOWN entity
    assert res.world.entities["boot-unobserved"].confidence == 0.0


def test_plain_wall_has_no_opening_candidates(tmp_path):
    from engine.pipeline.single_image import bootstrap_single_image

    item, facts, dm = _wall_scene(tmp_path, with_openings=False)
    res = bootstrap_single_image(item, facts, depth_map=dm)
    assert res.facts["openings"]["attempted"] and res.facts["openings"]["candidates"] == 0
    assert not [e for e in res.world.entities.values() if e.type.value in ("door", "window")]


def test_corridor_axis_needs_camera_between_two_walls_and_is_absent_for_one_wall(tmp_path):
    from engine.pipeline.single_image import bootstrap_single_image
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.depth.interface import DepthMap

    w, h = 320, 240
    f = (w / 2) / math.tan(math.radians(30))
    rows = []
    for v in range(h):
        row = []
        for u in range(w):
            a, b = (u - w / 2) / f, -(v - h / 2) / f
            cands = [12.0]
            if abs(a) > 1e-6:
                cands.append(1.0 / abs(a))          # side walls at x = +-1
            if abs(b) > 1e-6:
                cands.append(1.0 / abs(b))          # floor / ceiling at y = -+1
            row.append(1.0 / min(cands) - 0.25)     # d = 1/z - 0.25
        rows.append(row)
    dm = DepthMap(evidence_id="x", width=w, height=h, values=rows)
    p = tmp_path / "corridor.png"
    Image.fromarray(np.random.default_rng(5).integers(90, 100, (h, w, 3), dtype=np.uint8)).save(p)
    item = EvidenceItem(id="ev-cor", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    res = bootstrap_single_image(item, inspect_image(p), depth_map=dm)
    assert {"wall", "floor"} <= set(res.facts["plane_roles"]), res.facts["plane_roles"]
    cor = res.facts["corridor"]
    assert cor is not None and cor["provenance"] == "INFERRED" and cor["confidence"] <= 0.2
    assert abs(cor["axis"][2]) > 0.95           # runs along the viewing direction

    # one tilted wall (no second wall around the camera) must NOT be read as a corridor
    item2, facts2, dm2 = _wall_scene(tmp_path, with_openings=False)
    assert bootstrap_single_image(item2, facts2, depth_map=dm2).facts["corridor"] is None


@pytest.mark.skipif(not DATASET.exists(), reason="real photo dataset not present")
def test_duplicate_is_redundant_even_if_opencv_cannot_fit_a_fundamental_matrix(monkeypatch):
    """A zero-baseline pair makes F ill-posed; some OpenCV builds return no mask.
    The duplicate must still be measured as the same viewpoint, not read as a new view."""
    import cv2

    monkeypatch.setattr(cv2, "findFundamentalMat", lambda *a, **k: (None, None))
    imgs = sorted(DATASET.glob("*.JPG"))
    rep = analyze_contribution([("a", imgs[15]), ("dup", imgs[15]), ("b", imgs[16])])
    labels = {c.evidence_id: c.label for c in rep.per_image}
    assert labels["dup"] == "redundant", labels
    assert rep.summary()["redundant"] == 1


# ---- single-image openings: detect only what the pixels support; ambiguous stays unresolved -------------------
def _wall_with(tmp_path, paint):
    """The same receding wall as _wall_scene, with caller-chosen dark rectangles painted on it."""
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.depth.interface import DepthMap

    w, h = 320, 240
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 175, np.uint8) + rng.integers(0, 3, (h, w, 3), dtype=np.uint8)
    paint(img)
    p = tmp_path / "wall_custom.png"
    Image.fromarray(img).save(p)
    f = (w / 2) / math.tan(math.radians(30))
    vals = [[(1 - 0.5 * (u - w / 2) / f) / 4.0 - 0.25 for u in range(w)] for _ in range(h)]
    dm = DepthMap(evidence_id="x", width=w, height=h, values=vals)
    item = EvidenceItem(id="ev-wall", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    return item, inspect_image(p), dm


def _openings(res):
    return {e.id: e for e in res.world.entities.values()
            if (e.custom_properties.get("bootstrap") or {}).get("role") == "opening_candidate"}


def test_visible_doorway_and_enclosed_window_carry_evidence_dimensions_and_resolved_kinds(tmp_path):
    from engine.pipeline.single_image import bootstrap_single_image

    item, facts, dm = _wall_scene(tmp_path, with_openings=True)
    ops = _openings(bootstrap_single_image(item, facts, depth_map=dm))
    assert {e.type.value for e in ops.values()} == {"door", "window"}
    for e in ops.values():
        b = e.custom_properties["bootstrap"]
        assert b["evidence_id"] == "ev-wall" and b["resolved_kind"] is True and b["kind_basis"]
        dim = b["dimensions"]
        assert 0.0 < dim["width_frac_of_wall"] < 1.0 and 0.0 < dim["height_frac_of_wall"] <= 1.0
        assert "metric size is unknown" in dim["units"]          # relative scale: never claimed as metres


def test_ambiguous_opening_is_a_generic_opening_not_a_guessed_door_or_window(tmp_path):
    from engine.pipeline.single_image import bootstrap_single_image

    def paint(img):                                       # dark, touches the bottom, but wider than tall: not door-shaped
        img[190:240, 100:230] = 50

    item, facts, dm = _wall_with(tmp_path, paint)
    ops = _openings(bootstrap_single_image(item, facts, depth_map=dm))
    assert ops, "the rectangle is real evidence of a void and must not vanish"
    for e in ops.values():
        assert e.type.value == "opening" and e.custom_properties["bootstrap"]["resolved_kind"] is False
        assert "cannot be determined from one view" in e.custom_properties["bootstrap"]["kind_basis"]


def test_rectangle_cut_off_by_the_wall_region_edge_is_refused_as_incomplete_wall(tmp_path):
    from engine.pipeline.single_image import bootstrap_single_image

    def paint(img):                                       # runs into the wall's left edge: missing data, not an opening edge
        img[60:130, 0:55] = 55

    item, facts, dm = _wall_with(tmp_path, paint)
    res = bootstrap_single_image(item, facts, depth_map=dm)
    assert res.facts["openings"]["attempted"] and not _openings(res)
