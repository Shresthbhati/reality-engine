"""Evidence-world stage: depth -> alignment -> lifting -> POSITIONED entities -> persisted, reloaded, honestly described.

SYNTHETIC scene (a pinhole camera, a known plane with a nearer object, an inverse-depth map): this proves the wiring and the honesty
rules -- gauge scale is never called metres, an unalignable depth map yields rejections not geometry, an empty or partial world is
never "complete" -- NOT the accuracy of any real depth model or detector (see the benchmark contract for that)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from apps.api import evidence_world_stage as ew
from engine.math import Quat, Vec3
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.registry import EngineRegistry
from tests.test_multiengine_evidence_flow import FakeDetector, _spec
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

W, H, F = 800, 600, 800.0
OBJ = (300.0, 200.0, 500.0, 400.0)  # detection box; the object is nearer (z=2) than the background plane (z=3)


def _z(u, v):
    return 2.0 if OBJ[0] <= u <= OBJ[2] and OBJ[1] <= v <= OBJ[3] else 3.0


DX = {"a.png": 0.0, "b.png": 0.3}  # two genuinely different viewpoints (identical poses give parallel rays: nothing to triangulate)


def _camera(dx: float = 0.0):
    return PinholeCamera(CameraIntrinsics(fx=F, fy=F, cx=W / 2, cy=H / 2, width=W, height=H),
                         CameraExtrinsics(Vec3(dx, 0.0, 0.0), Quat.identity()))


def _box(dx: float):
    """The object (world x in [-0.25, 0.25], y in [-0.25, 0.25] at z=2) as seen from a camera at x=dx."""
    return (400 + (-0.25 - dx) * 400, 200.0, 400 + (0.25 - dx) * 400, 400.0)


def _points():
    pts = []
    for u in np.linspace(20, W - 20, 12):
        for v in np.linspace(20, H - 20, 9):
            z = _z(u, v)
            pts.append(((u - W / 2) / F * z, (v - H / 2) / F * z, z))
    return pts


def _inverse_depth(dx: float = 0.0):
    """What a monocular network returns: an affine function of 1/z (here 5/z + 1), with no unit."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_hit, y_hit = (u - 400) * 2.0 / F + dx, (v - 300) * 2.0 / F  # where each pixel's ray meets the z=2 plane
    z = np.where((np.abs(x_hit) <= 0.25) & (np.abs(y_hit) <= 0.25), 2.0, 3.0)
    return (5.0 / z + 1.0).astype(np.float32)


class _OneBoxDetector(FakeDetector):
    def infer(self, request):
        return [ObservationDraft(task="detection", quantity="bbox", source_id=s, coordinate_system=f"image:{s}",
                                 image_region=_box(DX[os.path.basename(s)]), predicted={"label": "chair", "score": 0.9}, confidence=0.9)
                for s in request.sources]


class _Depth:
    def __init__(self, arr):
        self.spec = type("S", (), {"engine_id": "fake_depth", "name": "fake", "version": "v9"})()
        self.arr, self.unloaded = arr, 0

    def validate_input(self, request):
        pass

    def load(self, device):
        pass

    def unload(self):
        self.unloaded += 1

    def infer(self, request):
        return [ObservationDraft(task="depth", quantity="depth_map", source_id=s, coordinate_system=f"cam:{s}",
                                 predicted={"array": self.arr if not callable(self.arr) else self.arr(s), "scale_state": "RELATIVE_UNSCALED"})
                for s in request.sources]


@pytest.fixture()
def photos(tmp_path):
    out = []
    for i, name in enumerate(("a.png", "b.png")):
        p = tmp_path / "src" / name
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (W, H), (120 + i, 120, 120)).save(p)  # distinct bytes: identical files would (correctly) share a cache entry
        out.append(str(p))
    return out


def _inputs(depth):
    def f():
        reg = EngineRegistry()
        sp = _spec("det_a", "fam-a")
        reg.register(sp)
        return reg, {"det_a": _OneBoxDetector(sp), "fake_depth": depth}, None

    return f


def _per_view(dx_map=DX):
    return lambda src: _inverse_depth(dx_map[os.path.basename(src)])


def _run(tmp_path, photos, depth, **kw):
    cams = {p: _camera(DX[os.path.basename(p)]) for p in photos}
    return ew.run_for_job(team_id="t1", world_id="w1", job_id=kw.pop("job", "job-1"), sources=photos, base=tmp_path / "ws",
                          flow_inputs=_inputs(depth), depth_engines=["fake_depth"], cameras=cams, **kw)


def _root(tmp_path, job="job-1"):
    return tmp_path / "ws" / "evidence-world" / "t1" / "w1" / job


def test_aligned_depth_gives_positioned_gauge_scale_entities_persisted_and_reloadable(tmp_path, photos):
    depth = _Depth(_per_view())
    r = _run(tmp_path, photos, depth, points3d=_points())
    assert r["status"] == "complete" and depth.unloaded == 1
    lift = r["lifting"]
    assert lift["status"] == "ran" and lift["lifted"] == 2 and lift["rejected"] == 0
    assert "sfm_gauge" in lift["scale"] and any("NOT metres" in x for x in lift["limitations"])
    assert r["world_ir"]["status"] == "persisted" and r["world_ir"]["reload_verified"] is True

    rd = ew.readiness(_root(tmp_path))  # a FRESH read of what is on disk
    w = rd["world"]
    # the same chair seen in BOTH photographs occupies the same 3D space, so the two detections are ONE entity (multi-view merge)
    assert w["status"] == "persisted" and w["object_entities"] == 1 and w["entities_with_geometry"] == w["entities"]  # surfaces are entities too
    from worldstore.store import WorldStore

    store = WorldStore(_root(tmp_path) / ew.WORLDSTORE_REL)
    ent = next(e for k, e in store.load_version(ew.head_version(_root(tmp_path))).entities.items() if k.startswith("ent-"))
    assert len({o.data_uri for o in ent.observations if o.metadata.get("quantity") == "bbox"}) == 2  # evidence from both photographs
    assert set(w["scale_states"]) == {"sfm_gauge"} and w["state"] == "complete"
    assert rd["export_ready"] is True and rd["metric_export_ready"] is False  # geometry exists, but it is NOT metres
    assert "not physically anchored metres" in rd["export_scale_warning"]


def test_the_lifted_position_and_extent_match_the_known_scene(tmp_path, photos):
    from worldstore.store import WorldStore

    _run(tmp_path, photos, _Depth(_per_view()), points3d=_points())
    store = WorldStore(_root(tmp_path) / ew.WORLDSTORE_REL)
    vid = sorted((_root(tmp_path) / ew.WORLDSTORE_REL / "versions").glob("*.json"))[-1].stem
    world = store.load_version(vid)
    objects = {k: e for k, e in world.entities.items() if k.startswith("ent-")}
    assert len(objects) == 1
    for e in objects.values():
        p = e.transform["position"]
        # the object centre is the optical axis at z=2 (synthetic ground truth of THIS scene, in the scene's own units)
        assert abs(p["x"]) < 0.1 and abs(p["y"]) < 0.1 and abs(p["z"] - 2.0) < 0.15
        assert e.transform["scale_state"] == "sfm_gauge" and e.transform["units"] == "sfm_units" and e.transform["physically_metric"] is False
        g = world.geometries[e.geometry_ids[0]]
        width = g.bounds_max.x - g.bounds_min.x  # 200 px at z=2 with f=800 -> 0.5 units; 5th-95th percentile extent -> ~0.45
        assert 0.38 < width < 0.55
        assert g.quality_metrics["scale"]["units"] == "sfm_units"


def test_an_unalignable_depth_map_rejects_every_region_and_invents_no_geometry(tmp_path, photos):
    noise = np.random.default_rng(0).uniform(1.0, 6.0, (H, W)).astype(np.float32)  # depth unrelated to the sparse points
    r = _run(tmp_path, photos, _Depth(noise), points3d=_points())
    assert r["lifting"]["lifted"] == 0 and r["lifting"]["rejected"] == 2
    assert any("not_aligned" == d["status"] for i in r["lifting"]["images"] for d in i["depth_engines"])
    rd = ew.readiness(_root(tmp_path))
    assert rd["world"]["entities_with_position"] == 0 and rd["world"]["state"] == "partial"  # entities exist, none positioned
    assert rd["export_ready"] is False and "no entity has a measured position" in rd["export_not_ready_reason"]
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    assert len(EvidenceMatrix(_root(tmp_path)).rejected()) >= 2  # the refusals stay on the record with reasons


def test_without_sparse_points_the_lift_is_skipped_with_the_reason(tmp_path, photos):
    r = _run(tmp_path, photos, _Depth(_per_view()), points3d=[])
    assert r["lifting"]["status"] == "skipped" and "no sparse SfM points" in r["lifting"]["reason"]
    assert ew.readiness(_root(tmp_path))["world"]["entities_with_position"] == 0


def _chair(i, lo, hi, n_obs):
    from provenance import Provenance
    from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Observation, Vector3

    g = Geometry(id=f"g{i}", type=GeometryType.BOX, lod_level=0, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi),
                 provenance=Provenance.ESTIMATED, confidence=None)
    obs = [Observation(id=f"o{i}{k}", sensor_type="camera", timestamp=0.0, frame_id="world", data_uri="x", data_hash="", metadata={},
                       confidence=None) for k in range(n_obs)]
    e = Entity(id=f"ent-{i}", type=EntityType.FURNITURE, name="Chair", geometry_ids=[g.id], semantic_labels=["furniture/seating/chair"],
               observations=obs, provenance=Provenance.ESTIMATED)
    return e, g


def test_unresolved_same_class_overlap_withholds_only_the_weaker_entity_and_records_why():
    from world_ir.validation import validate_world_ir
    from world_ir.world_v1 import WorldIR

    w = WorldIR(id="t", name="t")
    for i, (lo, hi, n) in enumerate([((0, 0, 0), (1, 1, 1), 3), ((0.05, 0.05, 0.05), (1.05, 1.05, 1.05), 1), ((5, 5, 5), (6, 6, 6), 1)]):
        e, g = _chair(i, lo, hi, n)
        w.entities[e.id], w.geometries[g.id] = e, g
    assert any("invariant_6" in str(x) for x in validate_world_ir(w).errors)  # the world would be refused as it stands
    withheld = ew._withhold_unresolved_duplicates(w)
    assert [x["entity_id"] for x in withheld] == ["ent-1"] and withheld[0]["kept_entity_id"] == "ent-0"  # fewer observations loses
    assert set(w.entities) == {"ent-0", "ent-2"} and "g1" not in w.geometries  # the far, unrelated chair is untouched
    assert "identity unresolved" in withheld[0]["reason"] and w.metadata["withheld_entities"] == withheld
    assert validate_world_ir(w).errors == []


def test_cameras_are_keyed_by_str_even_when_the_jobs_paths_are_path_objects(tmp_path):
    # REGRESSION (found on real data): Path keys never matched the str sources, so every camera was silently dropped
    from pathlib import Path

    p = tmp_path / "x.png"
    cams = ew.cameras_from_poses([("e1", (0, 0, 0), (1, 0, 0, 0))], {"e1": Path(p)},
                                 {"fx": F, "fy": F, "cx": W / 2, "cy": H / 2}, (W, H))
    assert list(cams) == [str(p)]


def test_without_cameras_nothing_is_lifted_and_the_reason_is_recorded(tmp_path, photos):
    r = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=photos, base=tmp_path / "ws",
                       flow_inputs=_inputs(_Depth(_per_view())), depth_engines=["fake_depth"], points3d=_points())
    assert "no registered cameras" in r["lifting"]["reason"] and r["lifting"]["status"] == "skipped"
