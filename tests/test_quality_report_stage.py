"""P110: `quality_report` stage -- geometry validity of the reconstructed mesh artifact."""
from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("trimesh")

from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.s12_quality_report import QualityReport, mesh_validity
from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geometry_data import MeshData

CUBE_V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def _no_trimesh(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "trimesh" or name.startswith("trimesh."):
            raise ImportError("blocked for test")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_closed_cube_passes():
    r = mesh_validity(CUBE_V, CUBE_F)
    assert r["verdict"] == "PASS"
    assert (r["watertight"], r["non_manifold_edges"], r["degenerate_faces"], r["components"]) == (1.0, 0, 0, 1)


def test_cube_missing_one_face_warns():
    r = mesh_validity(CUBE_V, CUBE_F[1:])
    assert r["verdict"] == "WARN" and r["watertight"] == 0.0 and r["components"] == 1


def test_two_disjoint_cubes_fail():
    r = mesh_validity(np.vstack([CUBE_V, CUBE_V + 5.0]), np.vstack([CUBE_F, CUBE_F + 8]))
    assert r["verdict"] == "FAIL" and r["components"] == 2


def test_degenerate_face_fails():
    f = np.vstack([CUBE_F, [[0, 0, 1]]])
    r = mesh_validity(CUBE_V, f)
    assert r["verdict"] == "FAIL" and r["degenerate_faces"] >= 1


def test_trimesh_import_failure_is_not_evaluable(monkeypatch):
    _no_trimesh(monkeypatch)
    r = mesh_validity(CUBE_V, CUBE_F)
    assert r == {"verdict": "NOT_EVALUABLE", "reason": "trimesh not installed"}
    assert "PASS" not in repr(r)


def _ctx(world=None, store=None) -> StageContext:
    state: dict = {}
    if world is not None:
        state["ladder"] = SimpleNamespace(prog=SimpleNamespace(world=world), fine={})
    if store is not None:
        state["artifact_store"] = store
    return StageContext(job_id="j", workdir=Path("."), artifacts={}, params={}, state=state)


def _world_with_mesh(vertices, faces):
    store = MemoryArtifactStore()
    uri, digest = store.put(MeshData(vertices.tolist(), faces.tolist()).to_bytes())
    geom = SimpleNamespace(type=SimpleNamespace(value="mesh"), data_uri=uri, data_hash=digest)
    return SimpleNamespace(geometries={"geom-mesh-room": geom}), store


def test_stage_spec_is_cpu_required():
    assert (QualityReport.spec.name, QualityReport.spec.resource_class, QualityReport.spec.required) == (
        "quality_report", "cpu", True)


def test_stage_records_mesh_metrics_with_prefix():
    world, store = _world_with_mesh(CUBE_V, CUBE_F)
    res = QualityReport().run(_ctx(world, store))
    assert res.status == "succeeded"
    assert res.metrics == {"mesh_watertight": 1.0, "mesh_non_manifold_edges": 0, "mesh_degenerate_faces": 0,
                           "mesh_components": 1, "mesh_verdict": "PASS"}


def test_stage_open_shell_warns_but_does_not_degrade():
    world, store = _world_with_mesh(CUBE_V, CUBE_F[1:])
    res = QualityReport().run(_ctx(world, store))
    assert res.metrics["mesh_verdict"] == "WARN" and res.status == "succeeded"


def test_stage_fail_verdict_is_degraded_not_failed():
    world, store = _world_with_mesh(np.vstack([CUBE_V, CUBE_V + 5.0]), np.vstack([CUBE_F, CUBE_F + 8]))
    res = QualityReport().run(_ctx(world, store))
    assert res.metrics["mesh_verdict"] == "FAIL" and res.status == "degraded" and res.warnings


@pytest.mark.parametrize("make_ctx", [
    lambda: _ctx(),                                                           # no ladder at all
    lambda: _ctx(SimpleNamespace(geometries={}), MemoryArtifactStore()),      # ladder world has no mesh
    lambda: _ctx(_world_with_mesh(CUBE_V, CUBE_F)[0]),                        # mesh recorded but no store to read it
    lambda: _ctx(_world_with_mesh(CUBE_V, CUBE_F)[0], MemoryArtifactStore()),  # artifact missing from the store
])
def test_stage_without_a_readable_mesh_is_not_evaluable(make_ctx):
    res = QualityReport().run(make_ctx())
    assert res.metrics == {"mesh_verdict": "NOT_EVALUABLE"}
    assert res.status == "skipped" and res.warnings
    assert "PASS" not in repr(res.metrics)


def test_stage_with_trimesh_missing_is_not_evaluable(monkeypatch):
    world, store = _world_with_mesh(CUBE_V, CUBE_F)
    _no_trimesh(monkeypatch)
    res = QualityReport().run(_ctx(world, store))
    assert res.metrics["mesh_verdict"] == "NOT_EVALUABLE" and "PASS" not in repr(res.metrics)
