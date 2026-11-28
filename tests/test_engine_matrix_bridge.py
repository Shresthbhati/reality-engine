"""Tests for the multi-engine -> canonical Evidence Matrix bridge.

The bridge exists because multi-engine observations previously stopped at the
job's JSON facts: nothing in the persisted World or the Evidence Matrix showed
that these engines had run, and a fresh process could not recover their lineage.

Pinned here:

* real engine observations are written into the run's Evidence Matrix;
* a NEW reader of the same store recovers them (reload, not in-memory state);
* no calibrated confidence is claimed for any imported observation;
* every engine execution -- failed, timed out, unavailable, skipped -- is kept as
  a run record with its status, so a failure is evidence too;
* an engine that produced nothing invents nothing;
* the lineage block survives a World save/reload through the WorldStore;
* re-importing the same report does not duplicate observations.
"""

from __future__ import annotations

import pytest

from engine.multiengine.evidence_matrix import EvidenceMatrix
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_matrix_bridge import (
    import_engine_outputs,
    import_multi_engine_report,
    world_lineage_block,
)
from perception.engine_orchestrator import (
    EngineExecution,
    MultiEngineReport,
    merge_engine_outputs,
)


def _obs(label="wall", arch="grounding_dino", image="img1", bbox=(1.0, 2.0, 30.0, 40.0),
         confidence=0.72) -> ObservationRecord:
    return ObservationRecord(
        evidence_id=f"{arch}:{image}:{label}",
        source_image_id=image,
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(engine_id="detection", architecture_id=arch,
                               checkpoint_id="ckpt", version="1.0", license="Apache-2.0"),
        label=label,
        category="object",
        bbox_px=bbox,
        confidence=confidence,
        calibration_quality="unverified",
        metadata={"model": "stub"},
    )


def _report(outputs, run_id="run-1") -> MultiEngineReport:
    report = merge_engine_outputs(outputs)
    report.run_id = run_id
    return report


def test_engine_observations_are_written_to_the_matrix_and_recovered_by_a_new_reader(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="grounding_dino", run_id="run-1",
                                          observations=(_obs(),), status="succeeded")])
    summary = import_multi_engine_report(report, matrix_root=tmp_path)

    assert summary["status"] == "imported"
    assert summary["n_observations_imported"] == 1
    assert summary["reload"]["missing"] == []

    oids = summary["observation_ids_by_engine"]["grounding_dino"]
    fresh = EvidenceMatrix(tmp_path)  # a different object over the same store: a reload
    stored = fresh.store.get(oids[0])
    assert stored is not None
    assert stored.source_id == "img1"
    assert stored.coordinate_system == "image"
    assert stored.predicted["scale_state"] == "image"
    assert stored.predicted["units"] == "pixels"
    assert stored.predicted["bbox_px"] == [1.0, 2.0, 30.0, 40.0]
    assert stored.confidence == pytest.approx(0.72)
    # raw, uncalibrated: the bridge never claims a calibration
    assert stored.calibration is None
    assert fresh.context(oids[0]).calibrated_confidence is None


def test_every_engine_execution_is_recorded_even_when_it_did_not_run(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="grounding_dino", run_id="run-1",
                                          observations=(_obs(),), status="succeeded")])
    report.engine_executions = [
        EngineExecution(engine_id="grounding_dino", architecture_id="grounding_dino", status="succeeded",
                        version="1.0", license="Apache-2.0", device="cpu", attempts=1, wall_time_s=1.5,
                        n_observations=1),
        EngineExecution(engine_id="sam2", architecture_id="sam2", status="skipped",
                        error="dependency_unmet: grounding_dino did not succeed", reason="dependency_unmet",
                        version="1.0", device="cpu"),
        EngineExecution(engine_id="depth_anything", architecture_id="depth_anything_v2", status="failed",
                        error="RuntimeError: model load failed", reason="exception", version="1.0"),
        EngineExecution(engine_id="lightglue_aliked", architecture_id="lightglue_aliked",
                        status="unavailable", error="kornia is not installed", reason="unavailable"),
        EngineExecution(engine_id="slow_engine", architecture_id="sift", status="timed_out",
                        error="exceeded 5s timeout", reason="timeout"),
    ]
    summary = import_multi_engine_report(report, matrix_root=tmp_path)

    assert summary["n_run_records"] == 5
    assert summary["engine_statuses"] == {
        "depth_anything": "failed",
        "grounding_dino": "succeeded",
        "lightglue_aliked": "unavailable",
        "sam2": "skipped",
        "slow_engine": "timed_out",
    }
    runs = EvidenceMatrix(tmp_path).runs()
    by_engine = {r.engine_id: r for r in runs}
    assert by_engine["sam2"].status == "skipped"
    assert "dependency_unmet" in (by_engine["sam2"].error_kind or "")
    assert by_engine["depth_anything"].status == "failed"
    assert "model load failed" in by_engine["depth_anything"].error


def test_an_engine_that_produced_nothing_adds_no_observation(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="depth_anything", run_id="run-1",
                                          observations=(), status="partial")])
    summary = import_multi_engine_report(report, matrix_root=tmp_path)
    assert summary["n_observations_imported"] == 0
    assert len(EvidenceMatrix(tmp_path).store) == 0


def test_reimporting_the_same_report_does_not_duplicate_observations(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="sift", run_id="run-1", observations=(_obs(arch="sift"),),
                                          status="succeeded")])
    first = import_multi_engine_report(report, matrix_root=tmp_path)
    second = import_multi_engine_report(report, matrix_root=tmp_path)
    assert first["n_observations_imported"] == 1
    assert second["n_observations_imported"] == 0
    assert len(EvidenceMatrix(tmp_path).store) == 1


def test_a_report_without_a_run_id_is_refused(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="sift", run_id="", observations=(_obs(arch="sift"),),
                                          status="succeeded")])
    report.run_id = ""
    summary = import_multi_engine_report(report, matrix_root=tmp_path)
    assert summary["status"] == "refused"
    assert "run id" in summary["reason"]


def test_convenience_wrapper_imports_raw_outputs(tmp_path):
    summary = import_engine_outputs(
        [UnifiedEngineOutput(engine_id="sift", run_id="run-9",
                             observations=(_obs(arch="sift"),), status="succeeded")],
        matrix_root=tmp_path, run_id="run-9",
    )
    assert summary["status"] == "imported"
    assert summary["n_observations_imported"] == 1


def test_lineage_block_is_small_and_carries_no_metric_claim(tmp_path):
    report = _report([UnifiedEngineOutput(engine_id="grounding_dino", run_id="run-1",
                                          observations=(_obs(),), status="succeeded")])
    block = world_lineage_block(import_multi_engine_report(report, matrix_root=tmp_path))
    assert block["n_observations_imported"] == 1
    assert "grounding_dino" in block["observation_content_ids_by_engine"]
    assert block["calibrated_confidence"].startswith("none")


def test_lineage_file_round_trips_and_absent_or_corrupt_is_none(tmp_path):
    from apps.api.evidence_world_stage import (
        read_multi_engine_evidence,
        write_multi_engine_evidence,
    )

    assert read_multi_engine_evidence(tmp_path) is None
    assert write_multi_engine_evidence(tmp_path, {"status": "imported", "n_observations_imported": 2}) is True
    assert read_multi_engine_evidence(tmp_path) == {"status": "imported", "n_observations_imported": 2}

    (tmp_path / "multi_engine_evidence.json").write_text("{not json", encoding="utf-8")
    assert read_multi_engine_evidence(tmp_path) is None  # corrupt is absent, never half-read


def test_world_lineage_survives_a_worldstore_save_and_reload(tmp_path):
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    report = _report([UnifiedEngineOutput(engine_id="grounding_dino", run_id="run-1",
                                          observations=(_obs(),), status="succeeded")])
    block = world_lineage_block(import_multi_engine_report(report, matrix_root=tmp_path / "matrix"))

    world = WorldIR(id="evw-run-1", name="evidence world run-1")
    world.metadata["multi_engine_evidence"] = block

    store = WorldStore(tmp_path / "store")
    store.save_version(world, parent=None, version_id="v-lineage")
    back = store.load_version("v-lineage")

    assert back.metadata["multi_engine_evidence"]["observation_content_ids_by_engine"] == \
        block["observation_content_ids_by_engine"]
    assert back.metadata["multi_engine_evidence"]["engine_statuses"] == block["engine_statuses"]
