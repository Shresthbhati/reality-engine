"""Integration and contract tests for spatial_inputs.v1 producer and API job hook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.api import evidence_world_stage
from perception.scene_intelligence import build_visual_scene_graph
from perception.spatial_inputs import (
    SCHEMA_VERSION,
    SPATIAL_INPUTS_FILE,
    CameraInput,
    ScaleStateInput,
    SpatialInputsV1,
    SpatialInputValidationError,
    SpatialObservationInput,
    get_inputs_provider,
    list_inputs_providers,
    produce_spatial_inputs_from_scene_graph,
    register_inputs_provider,
)
from perception.world_bridge import assess_image_quality, build_visual_observations_from_detections
from world_ir.observations import DepthSource


def test_provider_registry_and_custom_registration():
    providers = list_inputs_providers()
    assert "perception_scene_intelligence" in providers
    assert "evidence_matrix_snapshot" in providers
    fn = get_inputs_provider("perception_scene_intelligence")
    assert callable(fn)

    with pytest.raises(KeyError):
        get_inputs_provider("nonexistent_provider")

    register_inputs_provider("custom_test_provider", fn)
    assert "custom_test_provider" in list_inputs_providers()


def test_produce_spatial_inputs_from_scene_graph_deterministic_and_honest(tmp_path):
    rgb = np.full((240, 320, 3), 128, dtype=np.uint8)
    qual = assess_image_quality(rgb)
    depth = np.full((240, 320), 2.5, dtype=np.float32)
    obs = build_visual_observations_from_detections(
        source_evidence_id="ev_0",
        source_image_id="ev_0",
        model_id="grounding_dino",
        model_version="tiny_hf_v1",
        depth_map=depth,
        depth_source=DepthSource.RELATIVE,
        quality=qual,
        detections=[
            {"label": "wall", "bbox_px": [10.0, 10.0, 300.0, 200.0], "score": 0.72},
            {"label": "door", "bbox_px": [50.0, 40.0, 140.0, 210.0], "score": 0.65},
        ],
    )
    sg = build_visual_scene_graph(obs, processed_views=["ev_0"])
    si1 = produce_spatial_inputs_from_scene_graph("run_001", sg)
    si2 = produce_spatial_inputs_from_scene_graph("run_001", sg)

    assert si1.schema_version == SCHEMA_VERSION
    assert si1.cameras == ()  # never fabricated
    assert si1.scale_state.basis == "relative_uncalibrated"
    assert si1.scale_state.is_metric is False
    assert len(si1.observations) == 2

    d1 = si1.to_dict()
    d2 = si2.to_dict()
    assert d1["content_sha256"] == d2["content_sha256"]

    out_path = tmp_path / SPATIAL_INPUTS_FILE
    si1.write_json(out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "spatial_inputs.v1"
    assert loaded["content_sha256"] == d1["content_sha256"]


def test_spatial_inputs_validation_rejects_duplicate_ids_and_uncalibrated_world_pos():
    dup_obs = SpatialObservationInput(
        observation_id="obs_dup",
        evidence_id="ev_0",
        image_id="ev_0",
        frame_index=None,
        timestamp_s=None,
        label="chair",
        category="object",
        bbox_px=(10.0, 10.0, 50.0, 50.0),
        has_mask=False,
        mask_ref=None,
        depth_source="relative",
        depth_evidence=None,
        world_pos=None,
        bounds_min=None,
        bounds_max=None,
        sigma_m=None,
        model_id="test",
        model_version="1.0",
        confidence=0.8,
        uncertainty={},
    )
    bad_dup = SpatialInputsV1(
        run_id="r1",
        scale_state=ScaleStateInput("relative_uncalibrated", "relative", False),
        cameras=(),
        observations=(dup_obs, dup_obs),
        structural_observations=(),
        cross_view_associations=(),
        relationships=(),
        room_hypotheses=(),
    )
    with pytest.raises(SpatialInputValidationError, match="duplicate observation_id"):
        bad_dup.validate()

    bad_pos = SpatialInputsV1(
        run_id="r1",
        scale_state=ScaleStateInput("relative_uncalibrated", "relative", False),
        cameras=(),
        observations=(
            SpatialObservationInput(
                observation_id="obs_1",
                evidence_id="ev_0",
                image_id="ev_0",
                frame_index=None,
                timestamp_s=None,
                label="chair",
                category="object",
                bbox_px=None,
                has_mask=False,
                mask_ref=None,
                depth_source="relative",
                depth_evidence=None,
                world_pos=(1.0, 2.0, 3.0),  # forbidden without cameras under relative depth
                bounds_min=None,
                bounds_max=None,
                sigma_m=None,
                model_id="test",
                model_version="1.0",
                confidence=0.8,
                uncertainty={},
            ),
        ),
        structural_observations=(),
        cross_view_associations=(),
        relationships=(),
        room_hypotheses=(),
    )
    with pytest.raises(SpatialInputValidationError, match="without registered cameras"):
        bad_pos.validate()


def test_real_api_evidence_world_stage_emits_spatial_inputs_v1(tmp_path):
    from PIL import Image
    from engine.multiengine.executor import TorchResourceProbe
    from tests.test_evidence_world_job import _setup

    img_path = tmp_path / "room.jpg"
    Image.new("RGB", (320, 240), (140, 130, 120)).save(img_path)

    reg, adapters = _setup()

    facts = evidence_world_stage.run_for_job(
        team_id="team_t",
        world_id="world_t",
        job_id="job_t",
        sources=[str(img_path)],
        base=tmp_path / "ws",
        flow_inputs=lambda: (reg, adapters, TorchResourceProbe()),
        detection="legacy",
    )
    assert "spatial_inputs_v1" in facts
    si_meta = facts["spatial_inputs_v1"]
    assert si_meta["schema_version"] == "spatial_inputs.v1"
    assert si_meta["observation_count"] >= 1
    si_file = evidence_world_stage.evidence_root("team_t", "world_t", "job_t", base=tmp_path / "ws") / SPATIAL_INPUTS_FILE
    assert si_file.is_file()
    payload = json.loads(si_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "spatial_inputs.v1"
    assert payload["content_sha256"] == si_meta["content_sha256"]


def test_fastapi_spatial_inputs_route_auth_404_and_200(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    import importlib
    from engine.multiengine.executor import TorchResourceProbe
    from tests.test_evidence_world_job import _setup

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))

    import apps.api.db as db_mod
    import apps.api.main as main_mod

    importlib.reload(db_mod)
    importlib.reload(main_mod)

    A = {"x-tenant-id": "tenant-a"}
    B = {"x-tenant-id": "tenant-b"}
    with TestClient(main_mod.app) as client:
        w = client.post("/api/worlds", json={"name": "si-world"}, headers=A)
        assert w.status_code in (200, 201), w.text
        wid = w.json()["id"]

        img_path = tmp_path / "room.jpg"
        Image.new("RGB", (320, 240), (140, 130, 120)).save(img_path)
        reg, adapters = _setup()
        evidence_world_stage.run_for_job(
            team_id="tenant-a",
            world_id=wid,
            job_id="run-si-1",
            sources=[str(img_path)],
            base=tmp_path / "ws",
            flow_inputs=lambda: (reg, adapters, TorchResourceProbe()),
            detection="legacy",
        )

        # 200 for owning tenant
        r_ok = client.get(f"/api/worlds/{wid}/evidence-world/run-si-1/spatial-inputs", headers=A)
        assert r_ok.status_code == 200, r_ok.text
        body = r_ok.json()
        assert body["schema_version"] == "spatial_inputs.v1"
        assert len(body["observations"]) >= 1

        # 404 for missing file
        si_path = evidence_world_stage.evidence_root("tenant-a", wid, "run-si-1", base=tmp_path / "ws") / SPATIAL_INPUTS_FILE
        si_path.unlink()
        r_missing = client.get(f"/api/worlds/{wid}/evidence-world/run-si-1/spatial-inputs", headers=A)
        assert r_missing.status_code == 404

        # 404 for cross-tenant access
        r_other = client.get(f"/api/worlds/{wid}/evidence-world/run-si-1/spatial-inputs", headers=B)
        assert r_other.status_code == 404
