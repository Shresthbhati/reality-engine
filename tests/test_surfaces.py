"""Observed surfaces: planes classified against +Y up, observed hulls only (nothing extrapolated), scale never metric.
A SYNTHETIC room (floor, ceiling, two walls, a table top, a slanted board) -- proves the rules, not any reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")

from engine.multiengine import surfaces  # noqa: E402
from engine.multiengine.evidence_matrix import EvidenceMatrix  # noqa: E402
from world_ir.schema_v1 import EntityType, GeometryType  # noqa: E402

RNG = np.random.default_rng(0)


def _grid(n, a, b, make):
    u, v = RNG.uniform(a[0], a[1], n), RNG.uniform(b[0], b[1], n)
    return make(u, v) + RNG.normal(0, 0.002, (n, 3))


def _room():
    floor = _grid(6000, (-2, 2), (1, 5), lambda x, z: np.stack([x, np.full_like(x, -1.2), z], 1))      # y = -1.2, 4 x 4 = 16 u^2
    ceiling = _grid(6000, (-2, 2), (1, 5), lambda x, z: np.stack([x, np.full_like(x, 1.3), z], 1))     # y = +1.3
    back = _grid(5000, (-2, 2), (-1.2, 1.3), lambda x, y: np.stack([x, y, np.full_like(x, 5.0)], 1))   # z = 5, 4 x 2.5 = 10 u^2
    left = _grid(5000, (1, 5), (-1.2, 1.3), lambda z, y: np.stack([np.full_like(z, -2.0), y, z], 1))   # x = -2
    table = _grid(1500, (-0.6, 0.6), (2.5, 3.3), lambda x, z: np.stack([x, np.full_like(x, -0.4), z], 1))  # a small horizontal top
    return np.vstack([floor, ceiling, back, left, table])


CENTRES = np.array([[0.0, 0.0, 0.0], [0.2, 0.05, 0.1]])  # sensors at mid height


def test_planes_are_classified_by_the_up_axis_and_hulls_are_observed_extent_only():
    planes = surfaces.fit_planes(_room(), CENTRES, scene_depth=3.0)
    by = {}
    for p in planes:
        by.setdefault(p.kind, []).append(p)
    assert len(by["floor"]) == 1 and len(by["ceiling"]) == 1 and len(by["wall"]) >= 2
    floor = by["floor"][0]
    assert abs(floor.offset * np.sign(floor.normal[1]) + 1.2) < 0.02 and 14.5 < floor.area <= 16.2  # observed area ~ the true 16, never above it
    back = next(p for p in by["wall"] if abs(p.normal[2]) > 0.9)
    assert 9.0 < back.area <= 10.2
    assert abs(floor.normal @ surfaces.UP) > 0.99
    # a plane that is neither horizontal nor vertical is reported, not forced into a class
    slanted = _grid(4000, (-1, 1), (1, 3), lambda x, z: np.stack([x, 0.6 * z, z], 1))
    kinds = {p.kind for p in surfaces.fit_planes(np.vstack([_room(), slanted]), CENTRES, scene_depth=3.0)}
    assert "skipped" in kinds


def test_surface_entities_carry_observed_hull_unknown_note_and_gauge_scale(monkeypatch):
    cloud = _room()
    monkeypatch.setattr(surfaces, "fused_cloud", lambda aligned, stride=6: (cloud, np.zeros(len(cloud), int), CENTRES))
    aligned = {"img": (np.full((8, 8), 3.0), None)}
    ents, geoms, notes, rep = surfaces.build_surface_entities(aligned, EvidenceMatrix(), "run-1", with_mesh=False)
    assert rep["status"] == "ran" and {e.type for e in ents} >= {EntityType.FLOOR, EntityType.CEILING, EntityType.WALL}
    for e in ents:
        g = next(x for x in geoms if x.id == e.geometry_ids[0])
        q = g.quality_metrics
        assert g.type is GeometryType.PLANE and q["scale_state"] == "sfm_gauge" and "not inferred" in q["unknown"] and len(q["observed_hull"]) >= 3
        assert e.transform["physically_metric"] is False and e.custom_properties["surface"]["observed"] is True
        assert e.external_references and e.external_references[0].kind == "model_output"  # the fused cloud they came from
    assert any("nothing beyond the observed extent" in n for n in notes)


def test_too_few_points_is_skipped_with_the_reason_not_a_made_up_room(monkeypatch):
    monkeypatch.setattr(surfaces, "fused_cloud", lambda aligned, stride=6: (np.zeros((10, 3)), np.zeros(10, int), CENTRES))
    ents, geoms, _, rep = surfaces.build_surface_entities({"img": (np.ones((4, 4)), None)}, EvidenceMatrix(), "r")
    assert ents == [] and geoms == [] and rep["status"] == "skipped" and "depth points" in rep["reason"]
    assert surfaces.build_surface_entities({}, EvidenceMatrix(), "r")[3]["status"] == "skipped"


def test_a_fused_surface_replaces_the_point_cloud_mesh_reuses_its_stored_artifact_and_keeps_its_measured_provenance(monkeypatch):
    cloud = _room()
    monkeypatch.setattr(surfaces, "fused_cloud", lambda aligned, stride=6: (cloud, np.zeros(len(cloud), int), CENTRES))

    def boom(*a, **k):
        raise AssertionError("the ball-pivoting mesh must not be built when a fused surface exists (two meshes of one geometry)")

    monkeypatch.setattr(surfaces, "observed_mesh", boom)
    m = EvidenceMatrix()
    uri, sha = m.artifacts.put(b"ply\nfused surface bytes")
    fusion = {"method": "open3d.UniformTSDFVolume", "voxel_length_gauge": 0.04, "views_with_valid_depth": 3,
              "mesh": {"vertices": 120, "triangles": 200, "valid": True, "problems": [], "components": 2, "boundary_edges": 40,
                       "degenerate_triangles": 0, "bounds_gauge": [[-2.0, -1.2, 1.0], [2.0, 1.3, 5.0]]},
              "mesh_support": {"vertices_supported_by_ge2_views": 0.7, "views_that_did_not_contribute": []},
              "mesh_artifact": {"uri": uri, "sha256": sha, "bytes": 22, "format": "ply"},
              "per_view": [{"source": "a.png", "integrated_pixels": 10, "rejected_as_inconsistent": 2, "contributed_to_surface": True}]}
    ents, geoms, _, rep = surfaces.build_surface_entities({"img": (np.full((8, 8), 3.0), None)}, m, "run-1", fusion=fusion)
    g = next(x for x in geoms if x.type is GeometryType.MESH)
    assert [x.id for x in geoms if x.type is GeometryType.MESH] == ["geom-fused-surface"]  # exactly one mesh
    assert g.data_uri == uri and g.data_hash == sha and m.artifacts.get(g.data_uri).startswith(b"ply")  # the existing artifact, not a new one
    assert g.quality_metrics["scale_state"] == "sfm_gauge" and g.quality_metrics["mesh_support"]["vertices_supported_by_ge2_views"] == 0.7
    e = next(x for x in ents if x.id == "surf-fused-surface")
    assert e.transform["physically_metric"] is False and e.custom_properties["surface"]["per_view"][0]["rejected_as_inconsistent"] == 2
    assert rep["mesh"]["status"] == "fused"


def test_the_mesh_preserves_holes_and_is_stored_as_an_artifact(monkeypatch):
    cloud = _room()
    cloud = cloud[~((np.abs(cloud[:, 0]) < 0.7) & (cloud[:, 1] < -1.0))]  # a hole in the floor
    monkeypatch.setattr(surfaces, "fused_cloud", lambda aligned, stride=6: (cloud, np.zeros(len(cloud), int), CENTRES))
    m = EvidenceMatrix()
    ents, geoms, _, rep = surfaces.build_surface_entities({"img": (np.full((8, 8), 3.0), None)}, m, "run-1")
    assert rep["mesh"]["status"] == "built" and rep["mesh"]["triangles"] > 100
    g = next(x for x in geoms if x.type is GeometryType.MESH)
    assert m.artifacts.get(g.data_uri)[:3] == b"PLY" or m.artifacts.get(g.data_uri)[:3] == b"ply"
    assert "holes" in g.quality_metrics["unknown"]
