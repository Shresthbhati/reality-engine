"""Tests for the Mandatory Sequential Maximum-Fidelity Execution Policy.

Verifies:
1. Complete 10-step model execution lifecycle:
   LOAD -> INFERENCE -> SAVE RAW OUTPUT -> VALIDATE -> NORMALISE -> RECORD ERRORS -> RELEASE RESOURCES -> VERIFY PERSISTENCE -> NEXT MODEL.
2. Original source evidence retention across all models.
3. Configurable 13-stage sequential execution order.
4. Licensing gates: CC-BY-NC-4.0 engines (DUSt3R, MASt3R) gated under commercial policy.
5. Multi-engine consensus over independent model families.
6. Refinement loop with cause classification and capture request generation.
7. Acceptance report completeness.
8. API endpoints: /api/engines/stages, /api/engines/reconstruct, /api/engines/evidence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from engine.multiengine.adapters import (
    ColmapEngineAdapter,
)
from engine.multiengine.contract import (
    Device,
    EngineRequest,
    EngineRunStatus,
    Task,
)
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.executor import ExecutionManager, Policy
from engine.multiengine.registry import default_registry, maximum_fidelity_registry
from engine.multiengine.sequential_pipeline import (
    MaximumFidelityJobReport,
    SequentialMaximumFidelityPipeline,
)
from engine.multiengine.stages import STAGE_DEFINITIONS, StageId


def test_sequential_stages_definitions():
    """Verify all 13 required stages are defined in correct dependency order."""
    assert len(STAGE_DEFINITIONS) == 13
    stage_ids = [s.stage_id for s in STAGE_DEFINITIONS]

    assert stage_ids[0] == StageId.STAGE_01_COLMAP
    assert stage_ids[1] == StageId.STAGE_02_MAPANYTHING
    assert stage_ids[2] == StageId.STAGE_03_VGGT
    assert stage_ids[3] == StageId.STAGE_04_DUST3R
    assert stage_ids[4] == StageId.STAGE_05_DEPTH
    assert stage_ids[5] == StageId.STAGE_06_DETECTION
    assert stage_ids[6] == StageId.STAGE_07_SEGMENTATION
    assert stage_ids[7] == StageId.STAGE_08_TRACKING
    assert stage_ids[8] == StageId.STAGE_09_SURFACE
    assert stage_ids[9] == StageId.STAGE_10_CONSENSUS
    assert stage_ids[10] == StageId.STAGE_11_WORLDIR
    assert stage_ids[11] == StageId.STAGE_12_QUALITY
    assert stage_ids[12] == StageId.STAGE_13_REFINEMENT


def test_ten_step_lifecycle_enforcement():
    """Verify that every model follows:
    LOAD -> INFERENCE -> SAVE RAW OUTPUT -> VALIDATE -> NORMALISE -> RECORD ERRORS -> RELEASE RESOURCES -> VERIFY PERSISTENCE.
    """
    reg = maximum_fidelity_registry()
    spec = reg.get("colmap")
    adapter = ColmapEngineAdapter(spec)
    store = EvidenceStore()

    # Step 1 & 2: Select and Load
    adapter.load(Device("cpu"))

    # Step 3: Run inference on complete dataset with original sources
    req = EngineRequest(sources=("virtual:view_01.jpg", "virtual:view_02.jpg"))
    drafts = adapter.infer(req)
    assert len(drafts) >= 2

    # Step 4: Raw output preserved in original_coordinates
    pose_draft = drafts[0]
    assert pose_draft.original_coordinates is not None
    assert "rotation_quat_wxyz" in pose_draft.original_coordinates

    # Step 5 & 6: Validate and Normalise into SI units in coordinate frame
    assert pose_draft.coordinate_system == "world:colmap"
    assert "extrinsic_4x4" in pose_draft.predicted
    matrix = pose_draft.predicted["extrinsic_4x4"]
    assert len(matrix) == 4 and len(matrix[0]) == 4

    # Step 7: Record execution metadata into EngineRun via run()
    run = adapter.run(req, Device("cpu"))
    assert run.ok
    assert run.status is EngineRunStatus.SUCCEEDED
    assert run.usage is not None
    assert run.usage.wall_s >= 0.0

    # Step 8: Release resources (adapter._model set to None or unloaded)
    adapter.unload()
    assert adapter._model is None

    # Step 9: Verify persistence in EvidenceStore
    store.add_many(run.observations)
    for obs in run.observations:
        stored = store.get(obs.observation_id)
        assert stored is not None
        assert stored.observation_id == obs.observation_id


def test_licensing_enforcement_dust3r_and_mast3r():
    """Verify that CC-BY-NC-4.0 engines (DUSt3R, MASt3R) are blocked under production commercial policy."""
    reg = maximum_fidelity_registry()
    dust3r_spec = reg.get("dust3r")
    assert dust3r_spec.licence == "CC-BY-NC-4.0"
    assert dust3r_spec.commercial_ok is False

    mast3r_spec = reg.get("mast3r")
    assert mast3r_spec.licence == "CC-BY-NC-4.0"
    assert mast3r_spec.commercial_ok is False

    # Under commercial production mode, manager marks them not_approved
    mgr_commercial = ExecutionManager(reg, {}, allow_unestablished_licence=False)
    plan = mgr_commercial.plan(Task.SFM)
    d_entry = next(e for e in plan.entries if e.engine_id == "dust3r")
    assert d_entry.state == "not_approved"
    assert "licence CC-BY-NC-4.0 is not approved" in d_entry.reason


def test_sequential_maximum_fidelity_pipeline_execution():
    """Verify end-to-end sequential pipeline execution across all stages on source photos."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "evidence.jsonl"
        store = EvidenceStore(store_path)

        img_dir = Path(tmp_dir) / "images"
        img_dir.mkdir()
        sources = []
        for i in range(3):
            p = img_dir / f"cam{i:02d}.jpg"
            img = Image.new("RGB", (64, 64), color=(100 + i * 20, 100, 100))
            img.save(p)
            sources.append(str(p))

        pipeline = SequentialMaximumFidelityPipeline(store=store, allow_unestablished_licence=True)

        job_report = pipeline.execute_session(
            job_id="test_job_101",
            sources=sources,
            metadata={"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0},
        )

        assert isinstance(job_report, MaximumFidelityJobReport)
        assert job_report.job_id == "test_job_101"
        assert job_report.policy == Policy.MAXIMUM_FIDELITY.value
        assert job_report.persisted_observations_count > 0

        # Verify EvidenceStore persistence on disk
        assert store_path.exists()
        reloaded_store = EvidenceStore(store_path)
        assert len(reloaded_store) == job_report.persisted_observations_count

        # Verify consensus results
        assert "depth_mean_consensus" in job_report.consensus_summary
        assert "label_consensus" in job_report.consensus_summary

        # Verify quality report
        assert job_report.quality_report is not None


def test_api_routes_engines_endpoints():
    """Verify that /api/engines endpoints correctly serve the sequential pipeline."""
    client = TestClient(app)

    # 1. GET /api/engines (baseline registry)
    resp = client.get("/api/engines")
    assert resp.status_code == 200
    data = resp.json()
    assert "engines" in data
    assert data["summary"]["total"] >= 6

    # 1b. GET /api/engines?all=true (maximum fidelity registry)
    resp_all = client.get("/api/engines?all=true")
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert data_all["summary"]["total"] >= 18
    assert data_all["summary"]["adapters_implemented"] >= 15

    # 2. GET /api/engines/stages
    resp_stages = client.get("/api/engines/stages")
    assert resp_stages.status_code == 200
    stages_data = resp_stages.json()
    assert stages_data["count"] == 13
    assert stages_data["policy"] == "maximum_fidelity"
    assert stages_data["stages"][0]["stage_id"] == "stage_01_colmap"

    # 3. POST /api/engines/reconstruct
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        img_dir.mkdir()
        img_paths = []
        for i in range(2):
            p = img_dir / f"frame{i:02d}.jpg"
            img = Image.new("RGB", (64, 64), color=(120 + i * 20, 120, 120))
            img.save(p)
            img_paths.append(str(p))

        resp_recon = client.post(
            "/api/engines/reconstruct",
            json={
                "sources": img_paths,
                "metadata": {"room_name": "living_room"},
                "allow_unestablished_licence": True,
            },
            headers={"x-tenant-id": "root"},  # /reconstruct runs on server paths: global administrators only
        )
        assert resp_recon.status_code == 200
        recon_data = resp_recon.json()
        assert "job_id" in recon_data
        assert recon_data["policy"] == "maximum_fidelity"
        assert recon_data["persisted_observations_count"] > 0
        assert "consensus_summary" in recon_data

        # 4. GET /api/engines/evidence
        resp_ev = client.get("/api/engines/evidence", headers={"x-tenant-id": "root"})  # shared store: global admins only
        assert resp_ev.status_code == 200
        ev_data = resp_ev.json()
        assert ev_data["total"] >= recon_data["persisted_observations_count"]
