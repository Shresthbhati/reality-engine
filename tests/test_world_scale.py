"""Applying a scale calibration to a world: consistent transform of every geometry kind, uncertainty propagation, no double scaling, the
gauge parent preserved, idempotent, fresh-process reload. Synthetic world with a known truth: proves the operation, not a reconstruction."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trimesh")
import trimesh  # noqa: E402

from engine.multiengine.alignment import Sim3  # noqa: E402
from engine.multiengine.scale_anchors import PointAnchor, ScaleCalibration, calibrate_points  # noqa: E402
from engine.multiengine.world_scale import AlreadyCalibrated, CalibrationRefused, apply_calibration, commit_calibrated_version  # noqa: E402
from provenance import Provenance  # noqa: E402
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402
from worldstore.store import WorldStore  # noqa: E402

S = 2.0  # metres per gauge unit (known truth for the test)


class _Arts:
    def __init__(self):
        self.d = {}

    def put(self, b):
        h = hashlib.sha256(b).hexdigest()
        self.d["artifact://" + h] = b
        return "artifact://" + h, h

    def get(self, uri):
        return self.d[uri]


def _world(arts):
    uri, h = arts.put(trimesh.creation.box(extents=(1, 2, 3)).export(file_type="ply"))
    hull = [[0, 0, 5], [4, 0, 5], [4, 2, 5], [0, 2, 5]]
    geoms = {
        "geom-wall": Geometry(id="geom-wall", type=GeometryType.PLANE, bounds_min=Vector3(0, 0, 5), bounds_max=Vector3(4, 2, 5),
                              quality_metrics={"plane": {"normal": [0, 0, 1], "offset": 5.0}, "observed_hull": hull, "observed_area": 8.0,
                                               "rms_residual": 0.01, "scale_state": "sfm_gauge", "units": "sfm_units"}),
        "geom-box": Geometry(id="geom-box", type=GeometryType.BOX, bounds_min=Vector3(0.5, 0.0, 4.4), bounds_max=Vector3(1.5, 1.0, 5.0),
                             quality_metrics={"scale_state": "sfm_gauge", "uncertainty": {"position": 0.02, "dimensions": [0.03, 0.03, 0.03]}}),
        "geom-mesh": Geometry(id="geom-mesh", type=GeometryType.MESH, data_uri=uri, data_hash=h, vertex_count=8, triangle_count=12,
                              bounds_min=Vector3(-0.5, -1, -1.5), bounds_max=Vector3(0.5, 1, 1.5), quality_metrics={"voxel_length_gauge": 0.05}),
    }
    rel = Relationship(kind=RelationshipKind.MOUNTED_ON, target_id="surf-wall-1", provenance=Provenance.ESTIMATED,
                       metadata={"centre_to_plane_gauge": 0.1, "scale_state": "sfm_gauge"})
    ents = {
        "surf-wall-1": Entity(id="surf-wall-1", type=EntityType.WALL, geometry_ids=["geom-wall"],
                              transform={"position": {"x": 2.0, "y": 1.0, "z": 5.0}, "frame_id": "world", "scale_state": "sfm_gauge"}),
        "ent-box": Entity(id="ent-box", type=EntityType.UNKNOWN, geometry_ids=["geom-box"], relationships=[rel],
                          transform={"position": {"x": 1.0, "y": 0.5, "z": 4.7}, "frame_id": "world", "scale_state": "sfm_gauge"}),
        "surf-fused-surface": Entity(id="surf-fused-surface", type=EntityType.STRUCTURE, geometry_ids=["geom-mesh"]),
    }
    return WorldIR(id="w1", entities=ents, geometries=geoms)


def _cal(kind="scale"):
    if kind == "scale":
        return ScaleCalibration(status="validated", reason="t", kind="distance", scale_m_per_gauge=S, scale_sigma=0.02 * S,
                                residuals=[{"anchor_id": "a", "location_gauge": [0.0, 0.0, 0.0]}])
    th = 0.5
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    return ScaleCalibration(status="validated", reason="t", kind="points", scale_m_per_gauge=S, scale_sigma=0.04,
                            transform=Sim3.from_arrays(S, R, np.array([1.0, -2.0, 0.5])),
                            residuals=[{"anchor_id": "a", "residual_m": 0.0, "location_gauge": [0.0, 0.0, 0.0]}])


def test_scale_only_transforms_every_geometry_kind_consistently_and_flags_metric():
    arts = _Arts()
    w, rec = apply_calibration(_world(arts), _cal(), parent_version="v0", artifacts=arts, cameras={"cam": [1, 1, 1]})
    p = w.entities["ent-box"].transform
    assert p["scale_state"] == "metric_anchored" and p["units"] == "m" and p["physically_metric"] is True
    assert (p["position"]["x"], p["position"]["z"]) == pytest.approx((2.0, 9.4))
    assert p["pre_calibration"]["scale_state"] == "sfm_gauge"  # the gauge value stays recorded
    assert np.allclose(w.geometries["geom-box"].quality_metrics["dimensions_m"], [2.0, 2.0, 1.2])
    pl = w.geometries["geom-wall"].quality_metrics
    assert pl["plane"]["offset"] == pytest.approx(10.0) and pl["observed_area"] == pytest.approx(32.0) and pl["rms_residual"] == pytest.approx(0.02)
    m = trimesh.load(io.BytesIO(arts.get(w.geometries["geom-mesh"].data_uri)), file_type="ply")
    assert np.allclose(m.extents, [2, 4, 6]) and w.geometries["geom-mesh"].data_hash != _world(arts).geometries["geom-mesh"].data_hash
    assert w.entities["ent-box"].relationships[0].metadata["centre_to_plane_m"] == pytest.approx(0.2)
    assert rec["camera_centres_physical"]["cam"] == pytest.approx([2, 2, 2]) and rec["applied"]["meshes"] == 1


def test_scale_uncertainty_is_propagated_into_position_and_dimension_uncertainty():
    arts = _Arts()
    w, _ = apply_calibration(_world(arts), _cal(), parent_version="v0", artifacts=arts)
    u = w.geometries["geom-box"].quality_metrics["uncertainty"]
    # dimension: sqrt((2*0.03)^2 + (extent*2*0.02)^2) per axis, extents (1.0, 1.0, 0.6)
    assert u["dimensions"][0] == pytest.approx(np.hypot(0.06, 1.0 * 0.04)) and u["dimensions"][2] == pytest.approx(np.hypot(0.06, 0.6 * 0.04))
    assert u["position"] > 2 * 0.02  # the scale term adds to the scaled position sigma
    assert u["propagated_from_gauge"]["position"] == 0.02  # the gauge uncertainty is kept


def test_rotation_and_translation_move_planes_hulls_and_positions_together():
    arts = _Arts()
    cal = _cal("points")
    w, rec = apply_calibration(_world(arts), cal, parent_version="v0", artifacts=arts)
    pos = w.entities["surf-wall-1"].transform
    assert np.allclose([pos["position"][k] for k in "xyz"], cal.transform.apply(np.array([2.0, 1.0, 5.0]))[0])
    pl = w.geometries["geom-wall"].quality_metrics
    n = np.asarray(pl["plane"]["normal"])
    assert np.allclose(np.asarray(pl["observed_hull"]) @ n, pl["plane"]["offset"], atol=1e-4)  # transformed hull lies on the transformed plane
    assert pos["frame_id"] == rec["frame_id"] and w.geometries["geom-box"].quality_metrics["bounds_after_rotation"]
    assert np.allclose(w.geometries["geom-box"].quality_metrics["dimensions_m"], [2.0, 2.0, 1.2])


def test_refusals_not_applicable_double_scaling_and_untransformable_geometry():
    arts = _Arts()
    with pytest.raises(CalibrationRefused, match="failed"):
        apply_calibration(_world(arts), ScaleCalibration(status="failed", reason="anchors disagree", kind="distance"), parent_version="v0", artifacts=arts)
    once, _ = apply_calibration(_world(arts), _cal(), parent_version="v0", artifacts=arts)
    with pytest.raises(AlreadyCalibrated):
        apply_calibration(once, _cal(), parent_version="v1", artifacts=arts)
    with pytest.raises(CalibrationRefused, match="artifact store"):
        apply_calibration(_world(arts), _cal(), parent_version="v0", artifacts=None)  # would half-scale: the mesh would stay in gauge
    w = _world(arts)
    w.geometries["geom-box"].type = GeometryType.UNKNOWN
    with pytest.raises(CalibrationRefused, match="no consistent transform"):
        apply_calibration(w, _cal(), parent_version="v0", artifacts=arts)


def test_commit_is_a_new_immutable_idempotent_version_and_reloads_in_a_fresh_process(tmp_path):
    arts = _Arts()
    store = WorldStore(tmp_path / "ws")
    v0 = store.save_version(_world(arts), parent=None, version_id="v-gauge").version_id
    before = json.dumps(store.load_version(v0).to_dict(), sort_keys=True)
    vid, rec = commit_calibrated_version(store, v0, _cal(), artifacts=arts)
    assert vid != v0 and store.parents(vid) == [v0]
    assert json.dumps(store.load_version(v0).to_dict(), sort_keys=True) == before  # the original unscaled world is untouched
    assert commit_calibrated_version(store, v0, _cal(), artifacts=arts)[0] == vid  # same request -> same version, not a second scaling
    assert len(store.list_versions()) == 2
    with pytest.raises(AlreadyCalibrated):  # a different calibration on the calibrated version would double-scale
        commit_calibrated_version(store, vid, _cal("points"), artifacts=arts)
    code = ("import json,sys;sys.path.insert(0,sys.argv[1]);from worldstore.store import WorldStore;s=WorldStore(sys.argv[2]);w=s.load_version(sys.argv[3]);"
            "print(json.dumps({'problems':s.verify_version(sys.argv[3]),'state':w.metadata['scale_state'],"
            "'cal':w.metadata['scale_calibration']['id'],'z':w.entities['ent-box'].transform['position']['z']}))")
    out = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), str(tmp_path / "ws"), vid],
                         capture_output=True, text=True, timeout=120)
    got = json.loads(out.stdout)
    assert got["problems"] == [] and got["state"] == "metric_anchored" and got["cal"] == rec["id"] and got["z"] == pytest.approx(9.4)


def test_a_calibration_estimated_from_points_applies_end_to_end():
    rng = np.random.default_rng(5)
    pts = []
    for i in range(8):
        p = rng.uniform(-3, 3, 3)
        pts.append(PointAnchor(f"p{i}", tuple(p), tuple(S * p + np.array([1.0, 0, 0]) + rng.normal(0, 0.003, 3)), 0.01, "survey",
                               "2026-11-22T10:00:00+00:00", "synthetic", held_out=i >= 6))
    cal = calibrate_points(pts)
    arts = _Arts()
    w, rec = apply_calibration(_world(arts), cal, parent_version="v0", artifacts=arts)
    assert cal.status == "validated" and w.metadata["scale_state"] == "metric_anchored" and rec["scale_m_per_gauge"] == pytest.approx(S, rel=5e-3)


def test_cached_determinations_nested_scale_labels_and_relationship_prose_follow_the_calibration():
    """Audit regression: after calibration NO cached spatial quantity may still read as gauge next to a metric transform."""
    arts = _Arts()
    w0 = _world(arts)
    w0.entities["ent-box"].custom_properties["evidence"] = {"determinations": {
        "position": {"status": "resolved", "frame": "world", "value": [1.0, 0.5, 4.7],
                     "uncertainty": {"method": "source_covariance", "covariance": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]}},
        "dimensions": {"status": "resolved", "frame": "world", "value": [1.0, 1.0, 0.6], "uncertainty": None}}}
    w0.geometries["geom-box"].quality_metrics["scale"] = {"scale_state": "sfm_gauge", "units": "sfm_units"}
    w0.entities["ent-box"].relationships.append(Relationship(kind=RelationshipKind.OVERLAPS, target_id="surf-wall-1",
                                                             metadata={"basis": "boxes overlap by 0.05 (all axes)"}))
    for cal in (_cal(), _cal("points")):
        w, rec = apply_calibration(copy_world(w0), cal, parent_version="v0", artifacts=arts)
        det = w.entities["ent-box"].custom_properties["evidence"]["determinations"]
        pos = w.entities["ent-box"].transform["position"]
        assert np.allclose(det["position"]["value"], [pos["x"], pos["y"], pos["z"]])  # the cache and the transform agree
        assert det["position"]["scale_state"] == "metric_anchored" and det["position"]["pre_calibration"]["value"] == [1.0, 0.5, 4.7]
        assert np.allclose(det["dimensions"]["value"], [2.0, 2.0, 1.2])
        cov = np.asarray(det["position"]["uncertainty"]["covariance"])
        assert det["position"]["uncertainty"]["scale_term"] == "propagated" and np.all(np.linalg.eigvalsh(cov) >= -1e-12)
        assert w.geometries["geom-box"].quality_metrics["scale"]["scale_state"] == "metric_anchored"
        assert w.geometries["geom-box"].quality_metrics["pre_calibration"]["scale"]["scale_state"] == "sfm_gauge"
        ov = next(r for r in w.entities["ent-box"].relationships if r.kind is RelationshipKind.OVERLAPS)
        assert "sfm_gauge_units" in ov.metadata["basis_numbers_in"]


def copy_world(w):
    import copy

    return copy.deepcopy(w)


def test_an_entity_orientation_with_a_rotating_calibration_is_refused_rather_than_left_unrotated():
    arts = _Arts()
    w = _world(arts)
    w.entities["ent-box"].transform["orientation"] = {"w": 1, "x": 0, "y": 0, "z": 0}
    with pytest.raises(CalibrationRefused, match="orientation"):
        apply_calibration(w, _cal("points"), parent_version="v0", artifacts=arts)
    apply_calibration(w, _cal(), parent_version="v0", artifacts=arts)  # a pure scale needs no rotation of the orientation: allowed
