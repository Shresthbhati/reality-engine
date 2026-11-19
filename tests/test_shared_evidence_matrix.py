"""Tests for the Append-Only Shared Evidence Matrix and Artifact Offloading."""

from pathlib import Path

import pytest

from engine.multiengine.evidence_matrix import (
    EvidenceMatrixEntry,
    EvidenceMatrixError,
    EvidenceMatrixStore,
)


def test_evidence_matrix_basic_add_and_query(tmp_path: Path) -> None:
    store = EvidenceMatrixStore(storage_root=tmp_path)

    # Offload artifact
    rel_path = store.save_artifact(
        evidence_id="ev_001",
        filename="point_cloud.ply",
        data=b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
    )
    assert rel_path == "evidence_artifacts/ev_001/point_cloud.ply"

    # Add entry
    entry = EvidenceMatrixEntry(
        evidence_id="ev_001",
        model_id="colmap_mvs",
        architecture_id="colmap",
        checkpoint_id="3.11",
        run_id="run_test",
        source_ids=["img01.jpg", "img02.jpg"],
        input_parameters={"patch_size": 11},
        prediction_type="point_cloud",
        raw_output_reference=rel_path,
        coordinate_frame="canonical_world",
        confidence=0.95,
    )
    assert store.add_entry(entry) is True
    assert len(store) == 1

    fetched = store.get_entry("ev_001")
    assert fetched.evidence_id == "ev_001"
    assert fetched.model_id == "colmap_mvs"
    assert fetched.content_hash != ""

    # Verify querying
    results = store.query(model_id="colmap_mvs")
    assert len(results) == 1
    assert results[0].evidence_id == "ev_001"

    empty_results = store.query(model_id="non_existent")
    assert len(empty_results) == 0


def test_evidence_matrix_immutability_and_hash_verification(tmp_path: Path) -> None:
    store = EvidenceMatrixStore(storage_root=tmp_path)

    data = b"raw depth tensor buffer 12345"
    rel_path = store.save_artifact("ev_002", "depth.bin", data)

    entry = EvidenceMatrixEntry(
        evidence_id="ev_002",
        model_id="depth_anything_v2",
        architecture_id="depth_anything",
        checkpoint_id="metric_indoor",
        run_id="run_test",
        source_ids=["img01.jpg"],
        input_parameters={"mode": "metric"},
        prediction_type="depth_map",
        raw_output_reference=rel_path,
        coordinate_frame="camera_local",
        confidence=0.88,
    )
    store.add_entry(entry)

    # Immutability: duplicate add must raise EvidenceMatrixError
    with pytest.raises(EvidenceMatrixError, match="already exists"):
        store.add_entry(entry)

    # verify_immutable should pass cleanly
    valid, errors = store.verify_immutable()
    assert valid is True
    assert errors == []

    # Tamper with file artifact directly on disk
    artifact_file = tmp_path / rel_path
    artifact_file.write_bytes(b"corrupted contents")

    valid_tampered, errors_tampered = store.verify_immutable()
    assert valid_tampered is False
    assert len(errors_tampered) == 1
    assert "corrupted" in errors_tampered[0]


def test_evidence_matrix_reloads_persisted_index(tmp_path: Path) -> None:
    store1 = EvidenceMatrixStore(storage_root=tmp_path)
    rel_path = store1.save_artifact("ev_reload", "info.txt", "Some text")
    entry = EvidenceMatrixEntry(
        evidence_id="ev_reload",
        model_id="test_model",
        architecture_id="test_arch",
        checkpoint_id="v1",
        run_id="run_r",
        source_ids=["src1"],
        input_parameters={},
        prediction_type="text",
        raw_output_reference=rel_path,
        coordinate_frame="world",
        confidence=1.0,
    )
    store1.add_entry(entry)

    # Re-instantiate store from same directory
    store2 = EvidenceMatrixStore(storage_root=tmp_path)
    assert len(store2) == 1
    reloaded_entry = store2.get_entry("ev_reload")
    assert reloaded_entry.evidence_id == "ev_reload"
    assert reloaded_entry.content_hash == entry.content_hash

    # Verify retrieved artifact content
    content = store2.get_artifact(rel_path)
    assert content == b"Some text"
