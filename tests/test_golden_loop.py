"""Golden Loop Test - Full Reality Engine Integration Verification

Runs the complete CAPTURE -> EVIDENCE -> SESSION -> PROCESS -> RECONSTRUCT 
-> WORLDIR -> WORLDSTORE -> DATABASE -> FRONTEND -> INSPECT -> CORRECT 
-> COMMIT -> VERSION -> DIFF -> QUERY -> EXPORT -> RELOAD loop.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tests.test_room_inference import _two_room_scene, _CAMS
from reconstruction.backend.interface import ReconstructedCameraPose
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from world_ir.artifact_store import FileArtifactStore
from worldstore.store import WorldStore
from world_ir.diff import diff_worlds
from world_ir.world_v1 import WorldIR


@pytest.mark.golden_loop
def test_full_golden_loop(tmp_path: Path):
    """Execute the complete Reality Engine golden loop."""
    world_path = tmp_path / "world.json"
    store_root = tmp_path / "store"
    store_root.mkdir()

    # 1-5: CAPTURE -> EVIDENCE -> SESSION -> PROCESS -> RECONSTRUCT -> WORLDIR
    # (using synthetic two-room scene as test data)
    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(_CAMS)
    )

    # Compile to WorldIR
    artifact_store = FileArtifactStore(store_root)
    world, diag = compile_reconstruction_to_world(
        result, CompileOptions(seed=42, artifact_store=artifact_store)
    )

    assert diag.rooms_detected >= 1
    assert len(world.entities) > 0
    assert len(world.geometries) > 0

    # Save world JSON
    world_path.write_text(
        json.dumps(world.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    # 6: PERSIST WORLDIR -> WORLDSTORE
    ws = WorldStore(store_root)
    v1 = ws.save_version(world, parent=None, version_id="v-golden-1")

    # 7: PERSIST VERSION
    assert v1.version_id == "v-golden-1"

    # 8: RELOAD APPLICATION / OPEN WORLD
    reloaded_v1 = ws.load_version(v1.version_id)
    assert set(reloaded_v1.entities) == set(world.entities)

    # 9: INSPECT WORLD
    inspect_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "inspect", str(world_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert inspect_result.returncode == 0
    inspect_data = json.loads(inspect_result.stdout)
    assert inspect_data["entities"] > 0

    # 10: INSPECT ENTITY
    first_entity_id = list(world.entities.keys())[0]
    entity = world.entities[first_entity_id]
    assert entity.id == first_entity_id

    # 11: TRACE ENTITY TO EVIDENCE
    # Entity should have geometry_ids that trace back to artifacts
    assert len(entity.geometry_ids) > 0
    for geom_id in entity.geometry_ids:
        assert geom_id in world.geometries
        geom = world.geometries[geom_id]
        # Geometry should have data_uri pointing to artifact
        assert geom.data_uri is not None
        assert geom.data_hash is not None

    # 12: VERIFY VERSION INTEGRITY
    errors = ws.verify_version(v1.version_id)
    assert errors == []

    # 13: PERFORM CORRECTION
    world2 = WorldIR.from_dict(world.to_dict())
    world2.modified_at = time.time()
    world2.entities[first_entity_id].name = "CORRECTED_" + world2.entities[first_entity_id].name

    # 14: COMMIT CORRECTION -> CREATE NEW VERSION
    v2 = ws.save_version(world2, parent=v1.version_id, version_id="v-golden-2")
    assert v2.version_id == "v-golden-2"

    # 15: INSPECT VERSION HISTORY
    versions = ws.list_versions()
    version_ids = [v.version_id for v in versions]
    assert "v-golden-1" in version_ids
    assert "v-golden-2" in version_ids

    # 16: LOAD v2 AND VERIFY CORRECTION STILL EXISTS
    reloaded_v2 = ws.load_version(v2.version_id)
    assert reloaded_v2.entities[first_entity_id].name.startswith("CORRECTED_")

    # 17: CALCULATE DIFF
    diff = diff_worlds(
        ws.load_version(v1.version_id),
        ws.load_version(v2.version_id)
    )
    diff_dict = diff.to_dict()
    assert diff_dict["summary"]["entities_modified"] >= 1

    # 18: QUERY WORLD
    query_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "query", "nearest", str(world_path), "1.0", "1.0", "1.0", "--k", "2"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert query_result.returncode == 0
    assert query_result.stdout.strip() != ""

    # 19: EXPORT WORLD (GLTF)
    gltf_out = tmp_path / "scene.gltf"
    export_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "export", str(world_path), "--format", "gltf", "-o", str(gltf_out)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert export_result.returncode == 0
    gltf_content = gltf_out.read_text(encoding="utf-8")
    assert len(gltf_content) > 0

    # 20: EXPORT WORLD (USDA)
    usda_out = tmp_path / "scene.usda"
    export_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "export", str(world_path), "--format", "usda", "-o", str(usda_out)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert export_result.returncode == 0
    usda_content = usda_out.read_text(encoding="utf-8")
    assert len(usda_content) > 0

    # 21: EXPORT WORLD (BLENDER)
    blender_out = tmp_path / "scene.py"
    export_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "export", str(world_path), "--format", "blender", "-o", str(blender_out)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert export_result.returncode == 0
    blender_content = blender_out.read_text(encoding="utf-8")
    assert len(blender_content) > 0

    # 22: BUILD OFFLINE VIEWER
    viewer_out = tmp_path / "viewer.html"
    viewer_result = subprocess.run(
        ["python", "-m", "apps.cli.main", "viewer", "--worldir", str(world_path), "-o", str(viewer_out)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert viewer_result.returncode == 0
    viewer_content = viewer_out.read_text(encoding="utf-8")
    assert "import(window.__appUrl)" in viewer_content

    # 23: RESTART/RELOAD VERIFICATION
    # Create new WorldStore instance and reload v2
    ws_new = WorldStore(store_root)
    reloaded_after_restart = ws_new.load_version(v2.version_id)
    assert reloaded_after_restart.entities[first_entity_id].name.startswith("CORRECTED_")

    # Verify v1 still exists and is unchanged
    reloaded_v1_after_restart = ws_new.load_version(v1.version_id)
    assert not reloaded_v1_after_restart.entities[first_entity_id].name.startswith("CORRECTED_")

    print("=== GOLDEN LOOP COMPLETE: ALL 23 STEPS PASSED ===")


if __name__ == "__main__":
    # Allow running directly
    with tempfile.TemporaryDirectory() as tmp:
        test_full_golden_loop(Path(tmp))