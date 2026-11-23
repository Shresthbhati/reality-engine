"""Adapters + SceneManifest + splits, against tiny GENERATED fixtures in each dataset's documented layout (tests/benchmark_dataset_fixtures.py).
Real datasets are never present, downloaded or required here."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from benchmarks.datasets import adapters as A
from benchmarks.datasets import manifest as M
from benchmarks.datasets import splits as S
from benchmarks.datasets.licences import LICENCES
from benchmarks.datasets.selection import PROPOSED_SELECTION, selection_report
from tests import benchmark_dataset_fixtures as F


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("adapters must never open a network connection")
    monkeypatch.setattr(socket.socket, "connect", boom)


# ------------------------------------------------------------------- access gating
@pytest.mark.parametrize("name", sorted(A.ADAPTERS))
def test_every_adapter_refuses_without_terms_attestation_and_lists_user_actions(tmp_path, name):
    with pytest.raises(A.AccessNotAcknowledged) as e:
        A.ADAPTERS[name](tmp_path)
    assert LICENCES[name].user_actions and str(LICENCES[name].user_actions[0])[:20] in str(e.value)
    with pytest.raises(A.DatasetLayoutError):
        A.ADAPTERS[name](tmp_path / "does-not-exist", terms_acknowledged=True)


def test_licence_records_complete_and_unverified_is_explicit():
    assert set(LICENCES) == set(A.ADAPTERS) == {"scannet", "scannetpp", "arkitscenes", "structured3d", "3rscan", "multiscan"}
    for lic in LICENCES.values():
        assert lic.user_actions and lic.terms_accepted_by_user is None and lic.commercial_use is None   # nobody accepted anything on the user's behalf
    assert LICENCES["arkitscenes"].verified is False and LICENCES["scannet"].verified is True


# -------------------------------------------------------------------------- per dataset
def _check(sm: M.SceneManifest, root: Path):
    assert M.validate_scene(sm, root) == []
    again = M.scene_from_dict(json.loads(json.dumps(sm.to_dict())))
    assert again == sm
    assert sm.licence.terms_accepted_by_user is True      # the user attestation, recorded


def test_scannet(tmp_path):
    sid = F.make_scannet(tmp_path)
    ad = A.ScanNetAdapter(tmp_path, terms_acknowledged=True)
    assert ad.scene_ids() == ["scene0000_00", "scene0000_01"]
    sm = ad.load_scene(sid)
    _check(sm, tmp_path)
    assert [i.id.split("/")[1] for i in sm.images] == ["0", "1", "2", "4"]                       # frame 3 had a non-finite pose
    assert any("1 frame(s) skipped" in n for n in sm.notes)
    im = sm.images[0]
    assert im.intrinsics == {"fx": 577.8, "fy": 578.0, "cx": 320.0, "cy": 240.0} and im.depth_unit_to_metre == 0.001
    assert im.pose_provenance == "estimated" and im.sha256 == M.sha256_file(tmp_path / im.path)
    assert sm.group_id == "scene0000" and sm.data_kinds == ["rgbd"] and sm.frame.camera_axes == "unknown"
    assert {a.kind for a in sm.annotations} == {"mesh", "instance"} and all(a.provenance_class == "estimated" for a in sm.annotations)
    assert not any(a.provenance_class == "measured" for a in sm.annotations)                      # ScanNet has no independent physical GT
    assert len(ad.load_scene(sid, max_images=2).images) == 2
    assert A.ScanNetAdapter(tmp_path, terms_acknowledged=True).load_scene("scene0000_01").images == []   # .sens not exported -> no images, with a note
    assert "needs export" in " ".join(A.ScanNetAdapter(tmp_path, terms_acknowledged=True).load_scene("scene0000_01").notes)


def test_scannetpp(tmp_path):
    sid = F.make_scannetpp(tmp_path)
    ad = A.ScanNetPPAdapter(tmp_path, terms_acknowledged=True)
    assert ad.scene_ids() == [sid]
    sm = ad.load_scene(sid)
    _check(sm, tmp_path)
    assert len(sm.images) == 3 and any("is_bad" in n for n in sm.notes)                        # one is_bad frame skipped
    assert sm.images[0].intrinsics["fx"] == 1500.0 and sm.frame.camera_axes == "opengl" and sm.images[0].width == 1920
    byclass = {a.kind: a.provenance_class for a in sm.annotations}
    assert byclass == {"point_cloud": "measured", "mesh": "fitted", "semantic": "estimated", "instance": "estimated"}
    assert sm.data_kinds == ["real_image", "physical_gt"] and sm.extra["official_split"] == "nvs_sem_val"
    # OpenGL -> OpenCV pose conversion flips the camera y/z axes and keeps the centre
    g = M.pose_c2w_opencv(sm.images[0], sm.frame)
    assert np.allclose(g[:3, :3], np.diag([1, -1, -1])) and np.allclose(g[:3, 3], np.asarray(sm.images[0].pose_c2w)[:3, 3])


def test_scannetpp_scene_without_scan_data_is_not_physical_gt(tmp_path):
    sid = F.make_scannetpp(tmp_path, with_scans=False)
    sm = A.ScanNetPPAdapter(tmp_path, terms_acknowledged=True).load_scene(sid)
    assert sm.annotations == [] and sm.data_kinds == ["real_image"] and any("no scan data" in n for n in sm.notes)
    assert M.validate_scene(sm, tmp_path) == []


def test_arkitscenes(tmp_path):
    vid = F.make_arkit(tmp_path)
    ad = A.ARKitScenesAdapter(tmp_path, terms_acknowledged=True)
    assert ad.scene_ids() == [vid]
    sm = ad.load_scene(vid)
    _check(sm, tmp_path)
    assert len(sm.images) == 4 and sm.images[0].intrinsics == {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0} and sm.images[0].width == 64
    # traj gives world-to-camera t = (-0.3 i, 0, -1.5) with identity rotation -> camera centre = -t
    assert np.allclose(np.asarray(sm.images[2].pose_c2w)[:3, 3], [0.6, 0.0, 1.5], atol=1e-9)
    assert np.allclose(np.asarray(A.ARKitScenesAdapter(tmp_path, terms_acknowledged=True).load_scene(vid, traj_is_world_to_camera=False)
                                  .images[2].pose_c2w)[:3, 3], [-0.6, 0.0, -1.5], atol=1e-9)
    boxes = sm.annotations_of("box3d")
    assert len(boxes) == 1 and boxes[0].label == "sofa" and boxes[0].provenance_class == "estimated"
    assert sm.annotations_of("mesh")[0].provenance_class == "estimated" and sm.extra["official_split"] == "Training"
    assert sm.data_kinds == ["rgbd"] and "physical_gt" not in sm.data_kinds


def test_structured3d(tmp_path):
    sid = F.make_structured3d(tmp_path)
    sm = A.Structured3DAdapter(tmp_path, terms_acknowledged=True).load_scene(sid)
    _check(sm, tmp_path)
    assert sm.data_kinds == ["synthetic"] and all(a.provenance_class == "synthetic" for a in sm.annotations) and sm.layout_verified
    assert sm.frame.unit_to_metre == 0.001 and sm.extra["official_split"] == "train"
    im = sm.images[1]
    assert im.intrinsics["fx"] == pytest.approx(40 / np.tan(0.785398163), rel=1e-6)                    # W = 80 -> fx = 40 / tan(45 deg) = 40
    assert im.intrinsics["fy"] == pytest.approx(20 / 0.5, rel=1e-6)                                    # H = 40, tan(yfov) = 0.5 -> 40
    c2w = np.asarray(M.pose_c2w_opencv(im, sm.frame))
    assert np.allclose(c2w[:3, 3], [1.0, 2.0, 1.5])                                                    # mm -> m
    assert np.allclose(c2w[:3, 2], [1, 0, 0]) and np.allclose(c2w[:3, 1], [0, 0, -1])                  # z = view direction, y = -up (OpenCV)
    assert np.isclose(np.linalg.det(c2w[:3, :3]), 1.0)
    assert A.Structured3DAdapter(tmp_path, terms_acknowledged=True).scene_ids() == [sid]


def test_3rscan_pairs_share_a_group(tmp_path):
    ref, resc = F.make_3rscan(tmp_path)
    ad = A.RScan3Adapter(tmp_path, terms_acknowledged=True)
    assert sorted(ad.scene_ids()) == sorted([ref, resc])
    a, b = ad.load_scene(ref), ad.load_scene(resc)
    for sm in (a, b):
        _check(sm, tmp_path)
    assert a.group_id == b.group_id == ref and a.extra["role"] == "reference" and b.extra["role"] == "rescan"
    assert b.extra["transform_to_reference"] == list(range(16)) and a.extra["official_split"] == "train"
    assert a.images[0].intrinsics == {"fx": 756.0, "fy": 756.0, "cx": 480.0, "cy": 270.0} and a.images[0].width == 960
    assert {x.kind for x in a.annotations} == {"mesh", "instance"}


def test_3rscan_without_metadata_and_zip_only(tmp_path):
    ref, _ = F.make_3rscan(tmp_path)
    (tmp_path / "3RScan.json").unlink()
    sm = A.RScan3Adapter(tmp_path, terms_acknowledged=True).load_scene(ref)
    assert sm.group_id == ref and any("LEAK" in n for n in sm.notes)
    (tmp_path / "zonly").mkdir()
    (tmp_path / "zonly" / "sequence.zip").write_bytes(b"PK")
    z = A.RScan3Adapter(tmp_path, terms_acknowledged=True).load_scene("zonly")
    assert z.images == [] and any("unzip" in n for n in z.notes)                                     # never extracts archives


def test_multiscan(tmp_path):
    sid = F.make_multiscan(tmp_path)
    ad = A.MultiScanAdapter(tmp_path, terms_acknowledged=True)
    assert ad.scene_ids() == ["scene_00000_00", "scene_00000_01"]
    sm = ad.load_scene(sid)
    _check(sm, tmp_path)
    assert len(sm.images) == 4 and sm.group_id == "scene_00000" and sm.frame.camera_axes == "opengl"
    assert sm.images[0].intrinsics == {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0}
    assert np.allclose(np.asarray(sm.images[2].pose_c2w)[:3, 3], [0.6, 0.4, 1.5]) and np.allclose(np.asarray(sm.images[2].pose_c2w)[3], [0, 0, 0, 1])
    assert len(sm.annotations_of("part")) == 1 and sm.annotations_of("part")[0].data["articulations"][0]["type"] == "rotation"
    assert sm.annotations_of("box3d")[0].label == "door.1"
    mp4only = F.make_multiscan(tmp_path / "b", with_frames=False)
    nf = A.MultiScanAdapter(tmp_path / "b", terms_acknowledged=True).load_scene(mp4only)
    assert nf.images == [] and any("decode" in n for n in nf.notes)                                  # no video decoding, ever


# ------------------------------------------------------------------ manifest rules
def _sm(tmp_path):
    F.make_scannetpp(tmp_path)
    return A.ScanNetPPAdapter(tmp_path, terms_acknowledged=True).load_scene("0a1b2c3d4e")


def test_validate_detects_tampered_missing_and_inconsistent(tmp_path):
    sm = _sm(tmp_path)
    assert M.validate_scene(sm, tmp_path) == []
    (tmp_path / sm.images[0].path).write_bytes(b"changed")
    assert any("does not match its recorded sha256" in p for p in M.validate_scene(sm, tmp_path))
    (tmp_path / sm.images[1].path).unlink()
    assert any("does not exist" in p for p in M.validate_scene(sm, tmp_path))
    assert not any("sha256" in p for p in M.validate_scene(sm, tmp_path, verify_hashes=False))
    sm2 = _sm(tmp_path / "x")
    sm2.annotations[0].provenance_class = "bogus"
    assert any("provenance_class" in p for p in M.validate_scene(sm2, tmp_path / "x"))


def test_kinds_and_provenance_must_agree(tmp_path):
    sm = _sm(tmp_path)
    sm.data_kinds = ["real_image"]
    assert any("lacks 'physical_gt'" in p for p in M.validate_scene(sm, tmp_path))
    sm = _sm(tmp_path / "y")
    sm.annotations = [a for a in sm.annotations if a.provenance_class != "measured"]
    assert any("requires at least one 'measured'" in p for p in M.validate_scene(sm, tmp_path / "y"))
    F.make_structured3d(tmp_path / "z")
    s3 = A.Structured3DAdapter(tmp_path / "z", terms_acknowledged=True)
    sm3 = s3.load_scene("scene_00007")
    sm3.annotations[0].provenance_class = "measured"
    assert any("synthetic scene" in p for p in M.validate_scene(sm3, tmp_path / "z"))
    sm3.images[0].pose_provenance = None
    assert any("without pose_provenance" in p for p in M.validate_scene(sm3, tmp_path / "z"))


def test_json_schema_file_matches_module_and_rejects_bad_documents(tmp_path):
    schema_file = Path(__file__).resolve().parent.parent / "benchmarks" / "datasets" / "scene_manifest.schema.json"
    assert json.loads(schema_file.read_text(encoding="utf-8")) == M.SCENE_MANIFEST_SCHEMA
    sm = _sm(tmp_path)
    d = sm.to_dict()
    assert M.schema_problems(d) == []
    d["split"] = "everything"
    d["images"][0]["sha256"] = "nothex"
    del d["frame"]["camera_axes"]
    probs = M.schema_problems(d)
    assert len(probs) >= 3
    with pytest.raises(M.ManifestFormatError):
        M.scene_from_dict(d)
    p = tmp_path / "m.json"
    M.save_scene_manifest(sm, p)
    assert M.load_scene_manifest(p) == sm
    p.write_text("{not json")
    with pytest.raises(M.ManifestFormatError):
        M.load_scene_manifest(p)


def test_to_benchmark_manifest_keeps_only_measured_dimensions(tmp_path):
    sm = _sm(tmp_path)
    sm.split = "test"
    sm.annotations += [M.Annotation(id="d1", kind="dimension", provenance_class="measured", data={"name": "w", "value_m": 3.0}, uncertainty_m=0.005),
                       M.Annotation(id="d2", kind="dimension", provenance_class="fitted", data={"name": "h", "value_m": 2.5}, uncertainty_m=0.005)]
    bm = M.to_benchmark_manifest([sm], "t")
    gt = bm["scenes"][0]["ground_truth"]
    assert [i["name"] for i in gt["room_dimensions"]["items"]] == ["w"] and gt["camera_poses"]["provenance"] == "estimated"
    assert bm["scenes"][0]["images"][0] == {"path": sm.images[0].path, "sha256": sm.images[0].sha256}


# -------------------------------------------------------------------------- splits
def _groups(n, per=2):
    out = []
    for g in range(n):
        for k in range(per):
            out.append(M.SceneManifest(dataset="d", scene_id=f"s{g:02d}_{k}", group_id=f"g{g:02d}", data_kinds=["rgbd"]))
    return out


def test_splits_reproducible_order_independent_group_atomic():
    scenes = _groups(20)
    a = S.assign_splits(scenes, seed=7)
    b = S.assign_splits(list(reversed(scenes)), seed=7)
    assert {s.scene_id: s.split for s in a} == {s.scene_id: s.split for s in b}
    assert {s.scene_id: s.split for s in a} != {s.scene_id: s.split for s in S.assign_splits(scenes, seed=8)}
    by_group = {}
    for s in a:
        by_group.setdefault(s.group_id, set()).add(s.split)
    assert all(len(v) == 1 for v in by_group.values())                                               # rescans never straddle splits
    counts = {k: sum(1 for g in by_group.values() if k in g) for k in ("development", "validation", "test")}
    assert counts == {"development": 10, "validation": 4, "test": 6}
    assert all(s.split == "unassigned" for s in scenes)                                              # inputs untouched
    tiny = S.assign_splits(_groups(3), seed=1)
    assert {s.split for s in tiny} == {"development", "validation", "test"}                          # every split gets a group when possible
    with pytest.raises(ValueError):
        S.assign_splits(scenes, seed=1, fractions={"development": 0.5, "test": 0.4})


def test_leakage_checks_group_image_and_unassigned(tmp_path):
    scenes = _groups(2, per=1)
    assert any("no split" in p for p in S.leakage_problems(scenes, tmp_path))
    a, b = F.make_3rscan(tmp_path)
    ad = A.RScan3Adapter(tmp_path, terms_acknowledged=True)
    sa, sb = ad.load_scene(a), ad.load_scene(b)
    sa.split, sb.split = "development", "test"                                                       # a room and its rescan on both sides
    assert any("group leakage" in p for p in S.leakage_problems([sa, sb], tmp_path))
    sb.split = "development"
    assert S.leakage_problems([sa, sb], tmp_path) == []
    # identical image bytes in two different scenes -> the benchmark_manifest image-leakage check fires
    sb2 = ad.load_scene(b)
    sb2.split, sb2.group_id, sb2.scene_id = "test", "other", "other-scene"
    sb2.images = [type(i)(**{**i.__dict__, "id": f"o/{i.id}"}) for i in sa.images]
    sa.split = "development"
    assert any("leakage" in p for p in S.leakage_problems([sa, sb2], tmp_path))


def test_selection_is_proposed_not_available():
    r = selection_report()
    assert r["status"] == "PROPOSED" and r["available"] is False and r["total_scene_units"] == 115
    assert {k: v["count"] for k, v in PROPOSED_SELECTION.items()} == {"scannet": 30, "scannetpp": 15, "arkitscenes": 20, "multiscan": 10,
                                                                      "structured3d": 30, "3rscan": 10}
    assert all(v["status"] == "PROPOSED" for v in PROPOSED_SELECTION.values())
