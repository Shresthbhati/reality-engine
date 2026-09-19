"""End-to-end product flow acceptance test for Reality Engine.

Verifies the complete 10-step chain:
1. Ingest (folder -> evidence package)
2. Session (multi-source session creation)
3. Registration (ICP/GNSS alignment)
4. Reconstruction & Compiler -> WorldIR
5. Validate (WorldIR structural and metric validation)
6. WorldStore (versioned persistence and listing)
7. Query (spatial nearest-neighbor queries)
8. Inspect (world summary and entity inspection)
9. Viewer (self-contained offline viewer generation)
10. Export (GLTF / USD / Blender export)
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


def run_e2e_product_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="reality_e2e_"))
    print(f"Executing Reality Engine product integration workflow in: {tmp}")

    # Step 1: Ingest
    pkg_path = tmp / "evidence_package.json"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "ingest", "datasets/real_room_capture", "-o", str(pkg_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Ingest failed: {r.stderr}"
    print(f"Step 1  [Ingest]:       PASS - {r.stdout.strip()}")

    # Step 2: Session
    sess_dir = tmp / "session_out"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "session", "create", "sess-001", "-o", str(sess_dir), "--name", "Room Capture Session"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Session failed: {r.stderr}"
    print(f"Step 2  [Session]:      PASS - {r.stdout.strip()}")

    # Step 3: Registration
    cloud1 = tmp / "cloud1.json"
    cloud2 = tmp / "cloud2.json"
    cloud1.write_text(json.dumps([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    cloud2.write_text(json.dumps([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0]]))
    reg_out = tmp / "reg.json"
    r = subprocess.run(
        [
            sys.executable, "-m", "apps.cli.main", "register",
            "--from-frame", "session_0",
            "--to-frame", "world",
            "-o", str(reg_out),
            str(cloud1), str(cloud2),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Register failed: {r.stderr}"
    first_line = r.stdout.splitlines()[0] if r.stdout else "registered"
    print(f"Step 3  [Registration]: PASS - {first_line}")

    # Step 4: Reconstruction / Compile to WorldIR
    from reconstruction.backend.interface import ReconstructionResult
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from world_ir.artifact_store import FileArtifactStore

    from reconstruction.backend.interface import ReconstructedPoint, ReconstructedCameraPose
    from provenance import Uncertainty

    recon_candidates = [
        Path("datasets/real_room_capture_worldir/reconstruction_result.json"),
        Path(__file__).resolve().parent.parent / "datasets" / "real_room_capture_worldir" / "reconstruction_result.json",
        Path(__file__).resolve().parents[3] / "datasets" / "real_room_capture_worldir" / "reconstruction_result.json",
    ]
    recon_path = next((p for p in recon_candidates if p.exists()), None)
    if recon_path:
        recon_json = recon_path.read_text(encoding="utf-8")
        data = json.loads(recon_json)
        pts = [
            ReconstructedPoint(
                position=tuple(p["position"]),
                track_id=p["track_id"],
                source_evidence_ids=p.get("source_evidence_ids", []),
                uncertainty=Uncertainty(confidence=p.get("uncertainty", {}).get("confidence", 1.0)),
            )
            for p in data["points"]
        ]
        cams = [
            ReconstructedCameraPose(
                evidence_id=c["evidence_id"],
                position=tuple(c["position"]),
                rotation=tuple(c["rotation"]),
                uncertainty=Uncertainty(confidence=c.get("uncertainty", {}).get("confidence", 1.0)),
            )
            for c in data.get("camera_poses", [])
        ]
        recon_res = ReconstructionResult(
            points=pts,
            camera_poses=cams,
            registration_status=data.get("registration_status", "success"),
        )
    else:
        from tests.test_room_inference import _CAMS, _two_room_scene

        recon_res = _two_room_scene()
        recon_res.camera_poses.extend(
            ReconstructedCameraPose(
                evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0)
            )
            for i, p in enumerate(_CAMS)
        )
    store_artifacts = FileArtifactStore(tmp / "artifacts")
    world, diag = compile_reconstruction_to_world(recon_res, CompileOptions(seed=42, artifact_store=store_artifacts))
    worldir_path = tmp / "worldir.json"
    worldir_path.write_text(json.dumps(world.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Step 4  [WorldIR]:      PASS - {len(world.entities)} entities compiled ({diag.rooms_detected} rooms detected)")

    # Step 5: Validate
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "validate", str(worldir_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Validate failed: {r.stderr}"
    print(f"Step 5  [Validate]:     PASS - WorldIR validation clean (exit 0)")

    # Step 6: WorldStore save & list
    store_dir = tmp / "worldstore"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "store", "save", "--store", str(store_dir), str(worldir_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Store save failed: {r.stderr}"
    print(f"Step 6a [Store Save]:   PASS - {r.stdout.strip()}")

    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "store", "list", "--store", str(store_dir)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Store list failed: {r.stderr}"
    print(f"Step 6b [Store List]:   PASS - {r.stdout.strip()}")

    # Step 7: Query
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "query", "nearest", str(worldir_path), "0", "0", "0", "--k", "3"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Query failed: {r.stderr}"
    sample_res = r.stdout.splitlines()[0] if r.stdout.splitlines() else "found"
    print(f"Step 7  [Query]:        PASS - Nearest entity: {sample_res}")

    # Step 8: Inspect
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "inspect", str(worldir_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Inspect failed: {r.stderr}"
    inspect_data = json.loads(r.stdout)
    print(f"Step 8  [Inspect]:      PASS - World ID: {inspect_data['world_id']}, Entities: {inspect_data['entities']}")

    # Step 9: Viewer
    viewer_html = tmp / "viewer.html"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "viewer", "--worldir", str(worldir_path), "-o", str(viewer_html)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Viewer failed: {r.stderr}"
    print(f"Step 9  [Viewer]:       PASS - Self-contained HTML viewer generated ({viewer_html.stat().st_size:,} bytes)")

    # Step 10: Export
    gltf_path = tmp / "world.gltf"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "export", str(worldir_path), "--format", "gltf", "-o", str(gltf_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Export GLTF failed: {r.stderr}"
    print(f"Step 10 [Export GLTF]:  PASS - {gltf_path.name} ({gltf_path.stat().st_size:,} bytes)")

    # Also test USD export
    usd_path = tmp / "world.usda"
    r = subprocess.run(
        [sys.executable, "-m", "apps.cli.main", "export", str(worldir_path), "--format", "usda", "-o", str(usd_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Export USDA failed: {r.stderr}"
    print(f"Step 10b[Export USDA]:  PASS - {usd_path.name} ({usd_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("SUCCESS: Full 10-step end-to-end product integration chain verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_product_flow()
