"""End-to-end: synthetic two-room scene -> compiled WorldIR -> WorldStore -> viewer.

The REAL MODEL OPTIONAL leg of the Agent-5 integration matrix stays manual
(COLMAP/MiDaS runs are hardware + network dependent -- see
docs/REALITY_ENGINE_CURRENT_STATUS.md). This file is the always-runnable
E2E leg: deterministic synthetic reconstruction input through the REAL
compiler, REAL persistence, REAL exporters, and the REAL viewer builder.
No mocks of the engine; the only synthetic part is the input geometry,
which is the documented _two_room_scene fixture with closed-form
expectations (ring 2.5x2.25, area 5.625 m^2, height 2 m).
"""
from __future__ import annotations

import json

import pytest

from apps.cli.main import _artifacts_dir_for, main
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from reconstruction.backend.interface import ReconstructedCameraPose
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore


def _compiled_world_with_store(store_root):
    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(_CAMS)
    )
    store = FileArtifactStore(store_root)
    world, diag = compile_reconstruction_to_world(
        result, CompileOptions(seed=42, artifact_store=store)
    )
    return world, diag


@pytest.mark.e2e
def test_e2e_compile_persist_inspect_export_viewer(tmp_path, capsys):
    world_path = tmp_path / "world.json"
    store_root = _artifacts_dir_for(str(world_path))
    world, diag = _compiled_world_with_store(store_root)
    world_path.write_text(
        json.dumps(world.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    assert diag.rooms_detected >= 1
    assert len(world.entities) > 0

    # Persist in WorldStore under a fresh instance (process-restart property).
    from worldstore.store import WorldStore

    store_dir = tmp_path / "store"
    v1 = WorldStore(store_dir).save_version(world, parent=None, version_id="v-e2e-1")
    reloaded = WorldStore(store_dir).load_version(v1.version_id)
    assert set(reloaded.entities) == set(world.entities)
    assert WorldStore(store_dir).verify_version(v1.version_id) == []

    # Inspect + query through the CLI surface a real user touches.
    assert main(["inspect", str(world_path)]) == 0
    assert json.loads(capsys.readouterr().out)["entities"] > 0
    assert main(["query", "nearest", str(world_path), "1.0", "1.0", "1.0", "--k", "2"]) == 0
    assert capsys.readouterr().out.strip() != ""

    # Every exporter emits non-empty real output.
    for fmt, name in (("gltf", "s.gltf"), ("usda", "s.usda"), ("blender", "s.py")):
        out = tmp_path / name
        assert main(["export", str(world_path), "--format", fmt, "-o", str(out)]) == 0
        assert out.read_text(encoding="utf-8").strip() != ""
    capsys.readouterr()

    # The offline viewer builds with the real world embedded.
    viewer_out = tmp_path / "viewer.html"
    assert main(["viewer", "--worldir", str(world_path), "-o", str(viewer_out)]) == 0
    html = viewer_out.read_text(encoding="utf-8")
    assert "import(window.__appUrl)" in html
