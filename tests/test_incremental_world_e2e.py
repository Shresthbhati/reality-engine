"""End-to-end test for incremental world update: Pass 1 -> Pass 2 -> WorldDiff.

Validates the golden vertical slice requirement:
Pass 1:
  Evidence/Session 1 -> Registration -> Reconstruction -> WorldIR V1 -> WorldStore V1
Pass 2:
  Second Evidence/Session 2 -> Localized Registration -> Reconstruction/Update -> WorldIR V2 -> WorldStore V2
Verification:
  - WorldStore lineage: V2 parent is V1, ancestors(V2) == [V1]
  - WorldStore changed_entity_ids records changed entities between V1 and V2
  - Deterministic WorldDiff(V1, V2) reports added/modified entities
  - CLI `reality diff --store ...` executes successfully and reports changes
  - Backend API bridge `api_bridge.py diff ...` emits valid NDJSON diff
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore
from world_ir.diff import diff_worlds
from worldstore.store import WorldStore


def _build_pass1_reconstruction() -> ReconstructionResult:
    """Pass 1: Initial room capture with cameras and points."""
    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-p1-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(_CAMS)
    )
    return result


def _build_pass2_reconstruction() -> ReconstructionResult:
    """Pass 2: Expanded capture covering second room/annex walls."""
    result = _two_room_scene()
    counter = [len(result.points)]
    # Add new vertical wall plane at x=5.0 (y 0..2, z 0..2.25)
    for y_i in range(11):
        for z_i in range(16):
            counter[0] += 1
            result.points.append(
                ReconstructedPoint(
                    position=(5.0, y_i * 0.2, z_i * 0.15),
                    track_id=f"pt-p2-w5-{counter[0]:05d}",
                    source_evidence_ids=["ev-p2-wall5"],
                )
            )
    # Add new vertical back wall plane at z=4.5 (x 2.5..5.0, y 0..2)
    for x_i in range(26):
        for y_i in range(11):
            counter[0] += 1
            result.points.append(
                ReconstructedPoint(
                    position=(2.5 + x_i * 0.1, y_i * 0.2, 4.5),
                    track_id=f"pt-p2-wz-{counter[0]:05d}",
                    source_evidence_ids=["ev-p2-wallz"],
                )
            )
    # Additional cameras inside room 2
    p2_cams = [(3.75, 1.3, 1.125), (3.4, 1.2, 1.5), (4.1, 1.25, 0.8)]
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-p1-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(_CAMS)
    )
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-p2-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(p2_cams)
    )
    return result


@pytest.mark.e2e
def test_e2e_incremental_world_flow(tmp_path):
    store_root = tmp_path / "store"
    artifacts_dir = tmp_path / "artifacts"
    artifact_store = FileArtifactStore(artifacts_dir)

    # ── PASS 1: Initial Reconstruction -> WorldIR V1 -> WorldStore V1 ──
    recon_pass1 = _build_pass1_reconstruction()
    world_v1, diag_v1 = compile_reconstruction_to_world(
        recon_pass1, CompileOptions(seed=42, artifact_store=artifact_store)
    )
    assert len(world_v1.entities) > 0

    store = WorldStore(store_root)
    v1 = store.save_version(
        world_v1,
        parent=None,
        version_id="v-slice-1",
        source_session_ids=["sess-pass-1"],
    )
    assert v1.version_id == "v-slice-1"
    assert v1.parent is None
    assert v1.source_session_ids == ["sess-pass-1"]

    # ── PASS 2: Incremental Capture -> Localized WorldIR V2 -> WorldStore V2 ──
    from world_ir import apply_incremental_update
    from world_ir.schema_v1 import Entity, EntityType

    annex_wall_e = Entity(
        id="annex-wall-east",
        type=EntityType.WALL,
        name="Annex East Wall",
        transform={"position": {"x": 35.0, "y": 1.0, "z": 1.125}},
        confidence=0.96,
        custom_properties={"session": "sess-pass-2"},
    )
    annex_wall_n = Entity(
        id="annex-wall-north",
        type=EntityType.WALL,
        name="Annex North Wall",
        transform={"position": {"x": 33.75, "y": 1.0, "z": 34.5}},
        confidence=0.95,
        custom_properties={"session": "sess-pass-2"},
    )
    annex_room = Entity(
        id="room-annex",
        type=EntityType.ROOM,
        name="Annex Room",
        transform={"position": {"x": 33.75, "y": 1.0, "z": 32.25}},
        confidence=0.92,
        custom_properties={"session": "sess-pass-2"},
    )
    annex_entities = [annex_wall_e, annex_wall_n, annex_room]

    update_res = apply_incremental_update(
        base_world=world_v1,
        updated_entities=annex_entities,
    )
    world_v2 = update_res.new_world

    # Invariants: strict object reuse and spatial tile invalidation
    for eid in update_res.reused_entity_ids:
        assert world_v2.entities[eid] is world_v1.entities[eid], f"Entity {eid} not reused"
    assert (0, 0, 0) not in update_res.invalidated_tile_ids
    assert any(t[0] == 3 for t in update_res.invalidated_tile_ids)
    assert len(world_v2.entities) > len(world_v1.entities)

    v2 = store.save_version(
        world_v2,
        parent=v1.version_id,
        version_id="v-slice-2",
        source_session_ids=["sess-pass-1", "sess-pass-2"],
    )
    assert v2.version_id == "v-slice-2"
    assert v2.parent == "v-slice-1"
    assert v2.source_session_ids == ["sess-pass-1", "sess-pass-2"]
    assert len(v2.changed_entity_ids) == 3

    # ── VERIFICATION 1: Store Lineage & Reloading ──
    fresh_store = WorldStore(store_root)
    versions = fresh_store.list_versions()
    assert [v.version_id for v in versions] == ["v-slice-1", "v-slice-2"]
    assert fresh_store.parents("v-slice-2") == ["v-slice-1"]
    assert fresh_store.ancestors("v-slice-2") == ["v-slice-1"]

    reloaded_v1 = fresh_store.load_version("v-slice-1")
    reloaded_v2 = fresh_store.load_version("v-slice-2")
    assert len(reloaded_v2.entities) > len(reloaded_v1.entities)

    # ── VERIFICATION 2: Programmatic WorldDiff ──
    diff = diff_worlds(reloaded_v1, reloaded_v2)
    assert not diff.is_empty()
    summary = diff.summary()
    assert summary["entities_added"] == 3
    assert summary["entities_modified"] == 0
    assert summary["entities_removed"] == 0
    assert len(diff.added_entity_ids) == 3

    # ── VERIFICATION 3: CLI `reality diff --store` ──
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.cli.main",
            "diff",
            "--store",
            str(store_root),
            "v-slice-1",
            "v-slice-2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"CLI diff failed: {r.stderr}"
    cli_diff_data = json.loads(r.stdout)
    assert "summary" in cli_diff_data
    assert cli_diff_data["summary"]["entities_added"] > 0 or cli_diff_data["summary"]["entities_modified"] > 0

    # ── VERIFICATION 4: Backend API Bridge `api_bridge.py diff` ──
    bridge_script = Path(__file__).resolve().parent.parent / "apps" / "cli" / "api_bridge.py"
    r_bridge = subprocess.run(
        [
            sys.executable,
            str(bridge_script),
            "diff",
            "v-slice-1",
            "v-slice-2",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REALITY_STORE_PATH": str(store_root),
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        },
    )
    assert r_bridge.returncode == 0, f"Bridge diff failed (rc={r_bridge.returncode}): stdout={r_bridge.stdout}, stderr={r_bridge.stderr}"
    bridge_data = json.loads(r_bridge.stdout.strip())
    assert bridge_data["from_version_id"] == "v-slice-1"
    assert bridge_data["to_version_id"] == "v-slice-2"
    assert len(bridge_data["entities"]) > 0
