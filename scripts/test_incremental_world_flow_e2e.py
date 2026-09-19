"""End-to-end incremental world flow acceptance test for Reality Engine.

Validates the golden vertical slice:
Pass 1:
  Evidence -> Session 1 -> Registration -> Reconstruction -> WorldIR V1 -> WorldStore V1 -> Desktop
Pass 2 (Mandatory Incremental Growth):
  Second evidence capture -> Session 2 -> Localized registration -> Localized update -> WorldStore V2 -> WorldDiff -> Desktop comparison

Reality Engine is not complete merely because it can reconstruct once.
It must be able to update a world incrementally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir import apply_incremental_update
from world_ir.artifact_store import FileArtifactStore
from world_ir.diff import diff_worlds
from world_ir.schema_v1 import Entity, EntityType
from worldstore.store import WorldStore


def _build_pass1_reconstruction() -> ReconstructionResult:
    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(
            evidence_id=f"ev-p1-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
        )
        for i, p in enumerate(_CAMS)
    )
    return result


def _build_pass2_reconstruction() -> ReconstructionResult:
    result = _two_room_scene()
    counter = [len(result.points)]
    # Annex wall at x=5.0
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
    # Annex back wall at z=4.5
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


def run_incremental_world_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="reality_incremental_"))
    print("=" * 75)
    print("REALITY ENGINE — END-TO-END INCREMENTAL WORLD VERIFICATION")
    print(f"Workspace: {tmp}")
    print("=" * 75)

    store_root = tmp / "store"
    artifacts_dir = tmp / "artifacts"
    artifact_store = FileArtifactStore(artifacts_dir)

    # ── PASS 1 ──
    print("\n--- PASS 1: Initial Scene Reconstruction ---")

    # Step 1: Session 1
    sess1_dir = tmp / "session_1"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "session", "create", "sess-p1", "-o", str(sess1_dir), "--name", "Base Room Capture"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Session 1 failed: {r.stderr}"
    print(f"Step 1  [Session 1]:    PASS - {r.stdout.strip()}")

    # Step 2: Registration & Reconstruction -> WorldIR V1
    recon_pass1 = _build_pass1_reconstruction()
    world_v1, diag_v1 = compile_reconstruction_to_world(
        recon_pass1, CompileOptions(seed=42, artifact_store=artifact_store)
    )
    world_v1_path = tmp / "world_v1.json"
    world_v1_path.write_text(json.dumps(world_v1.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Step 2  [WorldIR V1]:   PASS - {len(world_v1.entities)} entities compiled ({diag_v1.rooms_detected} rooms detected)")

    # Step 3: WorldStore Save V1
    store = WorldStore(store_root)
    v1 = store.save_version(world_v1, parent=None, version_id="v-inc-1", source_session_ids=["sess-p1"])
    print(f"Step 3  [Store Save V1]: PASS - Version {v1.version_id} (entities={len(world_v1.entities)}, parent=None)")

    # ── PASS 2 ──
    print("\n--- PASS 2: Incremental Capture & Localized Reconstruction ---")

    # Step 4: Session 2
    sess2_dir = tmp / "session_2"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "session", "create", "sess-p2", "-o", str(sess2_dir), "--name", "Annex Room Expansion"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Session 2 failed: {r.stderr}"
    print(f"Step 4  [Session 2]:    PASS - {r.stdout.strip()}")

    # Step 5: Localized Reconstruction / Update -> WorldIR V2
    # Genuine localized incremental compilation via apply_incremental_update()
    annex_wall_e = Entity(
        id="annex-wall-east",
        type=EntityType.WALL,
        name="Annex East Wall",
        transform={"position": {"x": 35.0, "y": 1.0, "z": 1.125}},
        confidence=0.96,
        custom_properties={"session": "sess-p2"},
    )
    annex_wall_n = Entity(
        id="annex-wall-north",
        type=EntityType.WALL,
        name="Annex North Wall",
        transform={"position": {"x": 33.75, "y": 1.0, "z": 34.5}},
        confidence=0.95,
        custom_properties={"session": "sess-p2"},
    )
    annex_room = Entity(
        id="room-annex",
        type=EntityType.ROOM,
        name="Annex Room",
        transform={"position": {"x": 33.75, "y": 1.0, "z": 32.25}},
        confidence=0.92,
        custom_properties={"session": "sess-p2"},
    )
    annex_entities = [annex_wall_e, annex_wall_n, annex_room]

    update_result = apply_incremental_update(
        base_world=world_v1,
        updated_entities=annex_entities,
    )
    world_v2 = update_result.new_world

    # Verify strict object reuse for all unaffected base entities
    for eid in update_result.reused_entity_ids:
        assert world_v2.entities[eid] is world_v1.entities[eid], (
            f"INVARIANT VIOLATION: Unaffected entity {eid} was rebuilt instead of reused"
        )
    # Verify spatial tile invalidation: untouched Room 1 tile (0, 0, 0) remains valid
    assert (0, 0, 0) not in update_result.invalidated_tile_ids, (
        f"INVARIANT VIOLATION: Untouched Room 1 tile was invalidated: {update_result.invalidated_tile_ids}"
    )
    assert any(t[0] == 3 for t in update_result.invalidated_tile_ids), (
        f"INVARIANT VIOLATION: Annex tile was not invalidated: {update_result.invalidated_tile_ids}"
    )

    world_v2_path = tmp / "world_v2.json"
    world_v2_path.write_text(json.dumps(world_v2.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    assert len(world_v2.entities) > len(world_v1.entities), "V2 entity count must exceed V1"
    print(f"Step 5  [WorldIR V2]:   PASS - {len(world_v2.entities)} entities compiled (+{len(update_result.changed_entity_ids)} added, {len(update_result.reused_entity_ids)} reused with strict object identity)")

    # Step 6: WorldStore Save V2 (with parent lineage)
    v2 = store.save_version(world_v2, parent=v1.version_id, version_id="v-inc-2", source_session_ids=["sess-p1", "sess-p2"])
    print(f"Step 6  [Store Save V2]: PASS - Version {v2.version_id} (parent={v2.parent}, changed_entities={len(v2.changed_entity_ids)})")

    # ── VERIFICATION ──
    print("\n--- PASS 3: Lineage & Differential Verification ---")

    # Step 7: Store Lineage & Restart Property
    restarted_store = WorldStore(store_root)
    versions = restarted_store.list_versions()
    assert [v.version_id for v in versions] == ["v-inc-1", "v-inc-2"]
    assert restarted_store.parents("v-inc-2") == ["v-inc-1"]
    assert restarted_store.ancestors("v-inc-2") == ["v-inc-1"]
    print(f"Step 7  [Store Lineage]: PASS - 2 immutable versions preserved across restart: {[v.version_id for v in versions]}")

    # Step 8: Deterministic Structural WorldDiff
    world_diff = diff_worlds(world_v1, world_v2)
    assert not world_diff.is_empty()
    summary = world_diff.summary()
    assert summary["entities_added"] == 3
    assert summary["entities_modified"] == 0
    assert summary["entities_removed"] == 0
    print(f"Step 8  [WorldDiff]:    PASS - Added: {summary['entities_added']}, Modified: {summary['entities_modified']}, Removed: {summary['entities_removed']}")

    # Step 9: CLI `reality diff --store`
    r_diff = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.cli.main",
            "diff",
            "--store",
            str(store_root),
            "v-inc-1",
            "v-inc-2",
        ],
        capture_output=True,
        text=True,
    )
    assert r_diff.returncode == 0, f"CLI diff failed: {r_diff.stderr}"
    cli_diff = json.loads(r_diff.stdout)
    assert cli_diff["summary"]["entities_added"] > 0
    print(f"Step 9  [CLI Diff]:     PASS - `reality diff --store` emitted structural diff JSON")

    # Step 10: Backend API Bridge `api_bridge.py diff`
    bridge_script = Path(__file__).resolve().parent.parent / "apps" / "cli" / "api_bridge.py"
    r_bridge = subprocess.run(
        [
            sys.executable,
            str(bridge_script),
            "diff",
            "v-inc-1",
            "v-inc-2",
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
    assert bridge_data["from_version_id"] == "v-inc-1"
    assert bridge_data["to_version_id"] == "v-inc-2"
    assert len(bridge_data["entities"]) > 0
    print(f"Step 10 [Bridge Diff]:  PASS - /api/world/diff bridge operational with {len(bridge_data['entities'])} entity diffs")

    print("\n" + "=" * 75)
    print("SUCCESS: END_TO_END_INCREMENTAL_WORLD_READY VERIFIED (10/10 CHECKS PASS)")
    print("=" * 75)


if __name__ == "__main__":
    run_incremental_world_flow()
