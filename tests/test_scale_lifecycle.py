"""ONE scale lifecycle: calibration is a new immutable version, every quantity follows it (property test over random worlds and scales), a fresh
process reloads and exports it, and nothing that is estimated / unvalidated / unscaled is ever labelled physically measured."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trimesh")
import trimesh  # noqa: E402

from engine.multiengine.scale_anchors import ScaleCalibration, calibrate_baseline  # noqa: E402
from engine.multiengine.world_scale import (  # noqa: E402
    CAL_KEY, AlreadyCalibrated, apply_calibration, commit_calibrated_version, scale_block, scale_kind, world_calibration,
)
from provenance import Provenance  # noqa: E402
from tests.test_world_scale import _Arts  # noqa: E402
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402
from worldstore.store import WorldStore  # noqa: E402


# ------------------------------------------------------------------------------------------ a random world carrying every quantity kind
def _random_world(arts: _Arts, rng: np.random.Generator) -> WorldIR:
    verts_scale = rng.uniform(0.5, 3.0, 3)
    mesh = trimesh.creation.box(extents=tuple(verts_scale))
    uri, h = arts.put(mesh.export(file_type="ply"))
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    pts = rng.uniform(-2, 2, (4, 3))
    pts -= np.outer(pts @ n, n)  # in-plane points
    off = float(rng.uniform(1, 4))
    hull = (pts + n * off).tolist()
    hp = np.asarray(hull)
    boxes, ents = {}, {}
    for i in range(3):
        c = rng.uniform(-3, 3, 3)
        d = rng.uniform(0.2, 1.5, 3)
        boxes[f"gb{i}"] = Geometry(
            id=f"gb{i}", type=GeometryType.BOX, bounds_min=Vector3(*(c - d / 2)), bounds_max=Vector3(*(c + d / 2)),
            quality_metrics={"scale_state": "sfm_gauge", "uncertainty": {"position": [0.02, 0.02, 0.03], "dimensions": [0.03, 0.04, 0.05]}})
        ents[f"e{i}"] = Entity(
            id=f"e{i}", type=EntityType.UNKNOWN, geometry_ids=[f"gb{i}"],
            transform={"position": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}, "frame_id": "world", "scale_state": "sfm_gauge"},
            relationships=[Relationship(kind=RelationshipKind.MOUNTED_ON, target_id="surf", provenance=Provenance.ESTIMATED,
                                        metadata={"centre_to_plane_gauge": float(rng.uniform(0.05, 0.4)),
                                                  "tolerance_gauge": 0.01, "scale_state": "sfm_gauge"})])
        ents[f"e{i}"].custom_properties["evidence"] = {"determinations": {
            "position": {"status": "resolved", "frame": "world", "value": c.tolist(),
                         "candidates": [{"value": (c + 0.01).tolist(), "engine": "x"}],
                         "uncertainty": {"method": "cov", "covariance": (np.eye(3) * 0.0004).tolist()}},
            "dimensions": {"status": "resolved", "frame": "world", "value": d.tolist(), "candidates": [{"value": d.tolist()}], "uncertainty": [0.03, 0.04, 0.05]}}}
    boxes["gp"] = Geometry(
        id="gp", type=GeometryType.PLANE, bounds_min=Vector3(*hp.min(0)), bounds_max=Vector3(*hp.max(0)),
        quality_metrics={"plane": {"normal": n.tolist(), "offset": off}, "observed_hull": hull, "observed_area": 7.5, "rms_residual": 0.01,
                         "inlier_count": 1234, "scale_state": "sfm_gauge", "units": "sfm_units", "fraction": 0.4})
    boxes["gm"] = Geometry(
        id="gm", type=GeometryType.MESH, data_uri=uri, data_hash=h, vertex_count=8, triangle_count=12,
        bounds_min=Vector3(*(-verts_scale / 2)), bounds_max=Vector3(*(verts_scale / 2)),
        quality_metrics={"voxel_length_gauge": 0.05, "sdf_trunc_gauge": 0.2, "surface_area_gauge2": 3.5, "views_fused": 4})
    ents["surf"] = Entity(id="surf", type=EntityType.WALL, geometry_ids=["gp"],
                          transform={"position": {"x": float(hp[:, 0].mean()), "y": float(hp[:, 1].mean()), "z": float(hp[:, 2].mean())},
                                     "frame_id": "world", "scale_state": "sfm_gauge"})
    ents["mesh"] = Entity(id="mesh", type=EntityType.STRUCTURE, geometry_ids=["gm"])
    return WorldIR(id="wp", entities=ents, geometries=boxes)


def _flatten(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flatten(v, path + (str(k),))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _flatten(v, path + (str(i),))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


_SKIP = ("pre_calibration", "propagated_from_gauge", "uncertainty", "scale_calibration", "scale", "observations", "metadata")
# genuinely unit-free quantities (and the labelled gauge originals, which stay next to their metre twins)
_UNITLESS = re.compile(r"(normal|inlier_count|fraction|vertex_count|triangle_count|views_fused|lod_level|confidence|version|_gauge2?$|^\d+$)")


@pytest.mark.parametrize("s", [0.05, 0.37, 2.5, 23.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_property_every_length_scales_by_s_every_area_by_s2_and_unitless_quantities_do_not_change(s, seed):
    arts = _Arts()
    w0 = _random_world(arts, np.random.default_rng(seed))
    cal = ScaleCalibration(status="validated", reason="t", kind="distance", scale_m_per_gauge=s, scale_sigma=0.01 * s,
                           residuals=[{"anchor_id": "a", "location_gauge": [0.0, 0.0, 0.0]}])
    w1, _ = apply_calibration(copy.deepcopy(w0), cal, parent_version="v0", artifacts=arts)
    b, a = {"g": {k: g.to_dict() for k, g in w0.geometries.items()}, "e": {k: e.to_dict() for k, e in w0.entities.items()}}, \
        {"g": {k: g.to_dict() for k, g in w1.geometries.items()}, "e": {k: e.to_dict() for k, e in w1.entities.items()}}
    fb = {p: v for p, v in _flatten(b) if not any(x in _SKIP for x in p)}
    fa = {p: v for p, v in _flatten(a) if not any(x in _SKIP for x in p)}
    scaled = unchanged = 0
    for p, old in fb.items():
        new = fa[p]
        if old == 0.0:
            assert new == pytest.approx(0.0, abs=1e-9), p
        elif new == pytest.approx(old, rel=1e-9):
            assert _UNITLESS.search(p[-1]) or any(_UNITLESS.search(x) for x in p), f"{p} did not follow the calibration and is not a declared unitless quantity"
            unchanged += 1
        else:  # a length scales by s; the one area (observed_area) by s^2
            k = 2 if p[-1] == "observed_area" else 1
            assert new / old == pytest.approx(s ** k, rel=1e-5), (p, old, new)
            scaled += 1
    assert scaled > 40 and unchanged > 5
    # plane equation stays a plane: the scaled hull lies on the scaled plane; mesh vertices are the scaled vertices
    pl = w1.geometries["gp"].quality_metrics
    nrm, off = np.asarray(pl["plane"]["normal"]), pl["plane"]["offset"]
    assert np.allclose(np.asarray(pl["observed_hull"]) @ nrm, off, atol=1e-4 * max(1.0, s))
    m0 = trimesh.load(__import__("io").BytesIO(arts.get(w0.geometries["gm"].data_uri)), file_type="ply", process=False)
    m1 = trimesh.load(__import__("io").BytesIO(arts.get(w1.geometries["gm"].data_uri)), file_type="ply", process=False)
    assert np.allclose(m1.vertices, s * m0.vertices, rtol=1e-5, atol=1e-6)
    # cached determinations, relationship distances and their uncertainties follow too (value x s; sigma never smaller than s x gauge sigma)
    for i in range(3):
        e0, e1 = w0.entities[f"e{i}"], w1.entities[f"e{i}"]
        d0, d1 = e0.custom_properties["evidence"]["determinations"], e1.custom_properties["evidence"]["determinations"]
        assert np.allclose(d1["position"]["value"], s * np.asarray(d0["position"]["value"]))
        assert np.allclose(d1["dimensions"]["value"], s * np.asarray(d0["dimensions"]["value"]))
        assert all(u >= s * g - 1e-12 for u, g in zip(d1["dimensions"]["uncertainty"], d0["dimensions"]["uncertainty"], strict=True))
        assert d1["dimensions"]["uncertainty_state"] == "NOT_MEASURED"        # a scaled sampling spread is still an uncalibrated lower bound
        assert w1.geometries[f"gb{i}"].quality_metrics["uncertainty"]["dimension_uncertainty_state"] == "NOT_MEASURED"
        r0, r1 = e0.relationships[0].metadata, e1.relationships[0].metadata
        assert r1["centre_to_plane_m"] == pytest.approx(s * r0["centre_to_plane_gauge"]) and r1["tolerance_m"] == pytest.approx(s * 0.01)
    q1 = w1.geometries["gm"].quality_metrics
    assert q1["voxel_length_m"] == pytest.approx(s * 0.05) and q1["sdf_trunc_m"] == pytest.approx(s * 0.2) and q1["surface_area_m2"] == pytest.approx(s * s * 3.5)


# --------------------------------------------------------------------------------------------- a new immutable version, fresh-process reload
def test_calibration_is_a_new_version_the_parent_hash_is_unchanged_and_a_fresh_process_reloads_and_exports_it(tmp_path):
    arts = _Arts()
    store = WorldStore(tmp_path / "ws")
    v0 = store.save_version(_random_world(arts, np.random.default_rng(3)), parent=None, version_id="v-gauge").version_id
    rec0 = json.loads((tmp_path / "ws" / "versions" / f"{v0}.json").read_text())
    bytes0 = (tmp_path / "ws" / "versions" / f"{v0}.json").read_bytes()
    cal = calibrate_baseline(0.8, 2.0, method="known_length", anchor_ids=("A", "B"))
    vid, rec = commit_calibrated_version(store, v0, cal, artifacts=arts)
    assert vid != v0 and rec["parent_artifact_hash"] == rec0["artifact_hash"]
    assert (tmp_path / "ws" / "versions" / f"{v0}.json").read_bytes() == bytes0 and store.verify_version(v0) == []   # the parent record AND its world bytes
    assert store.load_version(v0).metadata.get(CAL_KEY) is None and store.parents(vid) == [v0]
    code = (
        "import json,sys;sys.path.insert(0,sys.argv[1]);from worldstore.store import WorldStore;from sdk import reality;"
        "s=WorldStore(sys.argv[2]);w=s.load_version(sys.argv[3]);g=reality.export(w,'gltf')[0];"
        "print(json.dumps({'v':s.verify_version(sys.argv[3]),'block':__import__('engine.multiengine.world_scale',fromlist=['x']).scale_block(w),"
        "'extras':g['asset']['extras'],'z':w.entities['e0'].transform['position']['z']}))")
    out = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), str(tmp_path / "ws"), vid],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    z0 = store.load_version(v0).entities["e0"].transform["position"]["z"]
    assert got["v"] == [] and got["z"] == pytest.approx(2.5 * z0)
    assert got["block"]["units"] == "m" and got["block"]["scale_kind"] == "anchored_unvalidated" and got["block"]["physically_measured"] is False
    assert got["extras"]["scale"]["scale_kind"] == "anchored_unvalidated" and got["extras"]["scale"]["units"] == "m"   # the GLB says so in the file


# ----------------------------------------------------------------------------------- one lifecycle: the legacy reader sees the new version
def test_the_calibrated_version_also_carries_the_legacy_scale_block_so_every_legacy_reader_sees_it_as_metric():
    arts = _Arts()
    cal = calibrate_baseline(0.8, 2.0, method="known_length", anchor_ids=("A", "B"))
    w, rec = apply_calibration(_random_world(arts, np.random.default_rng(4)), cal, parent_version="v0", artifacts=arts)
    sc = w.metadata["scale"]
    assert sc["state"] == "metric" and sc["meters_per_unit"] == pytest.approx(2.5) and sc["dimension_state"] == "MEASURED"
    assert sc["calibration"] == world_calibration(w) == cal.to_dict() and sc["calibration_id"] == rec["id"]


def test_a_world_the_legacy_route_already_scaled_is_never_scaled_again():
    arts = _Arts()
    legacy = _random_world(arts, np.random.default_rng(5))
    cal = calibrate_baseline(0.8, 2.0, method="known_length", anchor_ids=("A", "B"))
    legacy.metadata["scale"] = {"state": "metric", "meters_per_unit": 2.5, "calibration": cal.to_dict()}
    with pytest.raises(AlreadyCalibrated):
        apply_calibration(legacy, cal, parent_version="v0", artifacts=arts)
    bare = _random_world(arts, np.random.default_rng(5))
    bare.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}       # a bare metric label is still "already metres"
    with pytest.raises(AlreadyCalibrated):
        apply_calibration(bare, cal, parent_version="v0", artifacts=arts)


# --------------------------------------------------------------------- sfm_gauge / relative_depth / estimated / anchored: never mixed
def _w(**md) -> WorldIR:
    w = WorldIR(id="x")
    w.metadata.update(md)
    return w


def test_the_five_scale_kinds_are_distinct_and_only_a_validated_calibration_is_physically_measured():
    valid = ScaleCalibration(status="validated", reason="t", kind="distance", scale_m_per_gauge=2.0, scale_sigma=0.02, basis="laser", dimension_state="MEASURED").to_dict()
    unval = calibrate_baseline(0.8, 2.0, method="known_length").to_dict()
    prior = calibrate_baseline(0.8, 2.0, method="prior_accepted_by_user", prior=True).to_dict()
    kinds = {
        "sfm_gauge": _w(),
        "relative_depth": _w(scale_state="relative_depth"),
        "estimated": _w(scale={"state": "metric", "meters_per_unit": 2.5, "calibration": prior}),
        "anchored_unvalidated": _w(scale={"state": "metric", "meters_per_unit": 2.5, "calibration": unval}),
        "physically_anchored": _w(scale={"state": "metric", "meters_per_unit": 2.0, "calibration": valid}),
    }
    assert {k: scale_kind(w) for k, w in kinds.items()} == {k: k for k in kinds}
    blocks = {k: scale_block(w) for k, w in kinds.items()}
    assert [k for k, b in blocks.items() if b["physically_measured"]] == ["physically_anchored"]
    assert [k for k, b in blocks.items() if b["units"] == "m"] == ["anchored_unvalidated", "physically_anchored"]   # an ESTIMATE is never metres
    assert blocks["anchored_unvalidated"]["dimension_state"] == "UNVALIDATED" and blocks["anchored_unvalidated"]["reference_dimension_state"] == "MEASURED"
    assert blocks["estimated"]["dimension_state"] == "ESTIMATED" and blocks["physically_anchored"]["dimension_state"] == "MEASURED"
    assert scale_kind(_w(scale={"state": "metric", "meters_per_unit": 1.0})) == "sfm_gauge"      # a bare label is not evidence


def test_the_export_element_report_never_says_measured_for_an_unvalidated_estimated_or_unscaled_world():
    from apps.api.routes_export import _element_report

    def world(**md):
        w = WorldIR(id="r", metadata=md)
        w.entities["e"] = Entity(id="e", type=EntityType.UNKNOWN, provenance=Provenance.RECONSTRUCTED)
        return w

    cal = lambda **k: calibrate_baseline(0.8, 2.0, method="known_length", **k).to_dict()  # noqa: E731
    valid = ScaleCalibration(status="validated", reason="t", kind="distance", scale_m_per_gauge=2.0, basis="laser", dimension_state="MEASURED").to_dict()
    rows = {name: _element_report(w, ["e"])[0] for name, w in {
        "unscaled": world(), "unvalidated": world(scale={"state": "metric", "calibration": cal()}),
        "estimated": world(scale={"state": "metric", "calibration": cal(prior=True)}), "validated": world(scale={"state": "metric", "calibration": valid})}.items()}
    assert [n for n, r in rows.items() if r["state"] == "measured"] == ["validated"]
    assert rows["unvalidated"]["state"] == "anchored_unvalidated" and rows["estimated"]["state"] == "estimated" and rows["unscaled"]["state"] == "unscaled"
    assert all(r["provenance_state"] == "measured" for r in rows.values())   # what was observed is not lost, only the physical claim is capped


def _plan_world(**scale):
    from tests.test_room_fidelity_fixture import _compiled
    from world_ir.measure import stamp_world_dimensions

    _o, w, _d = copy.deepcopy(_compiled("albedo", False))
    w.metadata["scale"] = scale
    stamp_world_dimensions(w)
    return w


@pytest.mark.parametrize("name,scale,units,title,forbidden", [
    ("validated", {"state": "metric", "calibration": {"status": "validated", "basis": "laser", "scale_m_per_gauge": 2.0, "dimension_state": "MEASURED"}},
     "units: m ", "Measured floor plan", "UNVALIDATED"),
    ("unvalidated", {"state": "metric", "calibration": calibrate_baseline(0.8, 2.0, method="known_length").to_dict()},
     "units: m ", "NOT independently validated", "Measured floor plan"),
    ("estimated", {"state": "metric", "calibration": calibrate_baseline(0.8, 2.0, method="prior_accepted_by_user", prior=True).to_dict()},
     "units: model units", "ESTIMATED scale", "Measured floor plan"),
])
def test_floorplan_svg_dxf_and_pdf_label_the_scale_kind_and_cap_measured_dimension_states(name, scale, units, title, forbidden):
    from exporters.floorplan import (build_room_report, export_to_floorplan_dxf_with_report, export_to_floorplan_svg_with_report, report_to_pdf)

    w = _plan_world(**scale)
    svg, _ = export_to_floorplan_svg_with_report(w)
    dxf, _ = export_to_floorplan_dxf_with_report(w)
    rep = build_room_report(w)
    assert units in svg and title in svg and forbidden not in svg
    assert title in dxf and units.replace("units: ", "") in dxf
    assert rep["scale_kind"] == {"validated": "physically_anchored", "unvalidated": "anchored_unvalidated", "estimated": "estimated"}[name]
    assert rep['scale_kind'].encode() in report_to_pdf(rep)
    if name != "validated":   # no measured-class dimension state survives in a world whose scale is not a validated measurement
        assert not re.search(r"· (MEASURED|VERIFIED|OBSERVED)\b", svg) and not any(s in ("MEASURED", "VERIFIED", "OBSERVED") for r in rep["rooms"] for s in r["dimension_states"])


def test_the_evidence_world_calibration_reads_as_metric_in_the_floorplan_export_too():
    """The evidence-world calibration used to leave the floor plan in 'model units' because only the legacy block was read."""
    from exporters.floorplan import _scale_units

    arts = _Arts()
    cal = ScaleCalibration(status="validated", reason="t", kind="distance", scale_m_per_gauge=2.0, scale_sigma=0.04, basis="laser", dimension_state="MEASURED")
    w, _ = apply_calibration(_random_world(arts, np.random.default_rng(6)), cal, parent_version="v0", artifacts=arts)
    only_new_key = copy.deepcopy(w)
    only_new_key.metadata.pop("scale")        # a reader that only knows the evidence-world record gets the same answer
    for world in (w, only_new_key):
        unit, _ = _scale_units(world)
        assert unit == "m" and unit.kind == "physically_anchored"


def test_ifc_declares_when_its_metres_are_not_physically_measured(tmp_path):
    pytest.importorskip("ifcopenshell")
    from sdk import reality

    w = _plan_world(state="metric", calibration=calibrate_baseline(0.8, 2.0, method="known_length").to_dict())
    try:
        content, _ = reality.export(w, "ifc")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"IFC export of the floor-plan fixture is not available here: {exc}")
    text = content if isinstance(content, str) else content.decode("latin-1")
    assert "scale_kind=anchored_unvalidated" in text and "physically_measured=False" in text
