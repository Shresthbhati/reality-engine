"""A4.1 & A4.2 tests: indoor dataset registry, ingestion adapters, integrity checks, and ground-truth workflow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from perception.datasets import (
    INDOOR_DATASET_CATALOG,
    AnnotationProvenance,
    DatasetIntegrityReport,
    DatasetSpec,
    GroundTruthAnnotation,
    NormalizedDatasetSample,
    ReviewStatus,
    build_human_review_manifest,
    compute_inter_annotator_agreement,
    create_scene_disjoint_splits,
    export_ground_truth_json,
    import_ground_truth_json,
    normalize_dataset_sample,
    score_ground_truth_dataset_quality,
    validate_dataset_integrity,
)
from perception.datasets.ade20k_and_nyu_adapters import (
    ADE20K_CLASS_MAP,
    NYU_CLASS_MAP,
    load_ade20k_samples,
    load_nyu_depth_v2_samples,
    load_synthetic_indoor_samples,
)
from perception.intelligence.open_vocab import Domain


def _sample_raw(sample_id: str, scene_id: str, label: str = "chair", bbox: tuple[int, int, int, int] = (20, 30, 120, 180)) -> dict:
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "image_path": f"images/{scene_id}/{sample_id}.jpg",
        "image_width": 640,
        "image_height": 480,
        "boxes": [
            {"label": label, "bbox_xyxy": list(bbox)},
        ],
    }


def test_indoor_dataset_catalog_contains_all_required_datasets() -> None:
    expected = {
        "scannet",
        "scannetpp",
        "sunrgbd",
        "nyu_depth_v2",
        "ade20k",
        "structured3d",
        "hypersim",
        "replica",
    }
    assert expected.issubset(set(INDOOR_DATASET_CATALOG.keys()))
    for key in expected:
        spec = INDOOR_DATASET_CATALOG[key]
        assert isinstance(spec, DatasetSpec)
        assert spec.dataset_id == key
        assert spec.license
        assert len(spec.annotations_available) >= 1
        assert spec.url

    assert INDOOR_DATASET_CATALOG["structured3d"].is_synthetic is True
    assert INDOOR_DATASET_CATALOG["hypersim"].is_synthetic is True
    assert INDOOR_DATASET_CATALOG["scannet"].is_synthetic is False
    assert INDOOR_DATASET_CATALOG["ade20k"].is_synthetic is False


def test_normalize_dataset_sample_maps_labels_and_flags() -> None:
    raw = {
        "sample_id": "s_001",
        "scene_id": "scene_a",
        "image_path": "images/scene_a/s_001.jpg",
        "image_width": 640,
        "image_height": 480,
        "boxes": [
            {"label": "armchair", "bbox_xyxy": [10, 20, 300, 300]},
            {"label": "door", "bbox_xyxy": [100, 40, 200, 420]},
        ],
    }
    sample = normalize_dataset_sample(raw, dataset_id="ade20k")
    assert isinstance(sample, NormalizedDatasetSample)
    assert sample.dataset_id == "ade20k"
    assert sample.scene_id == "scene_a"
    assert sample.image_width == 640
    assert sample.image_height == 480
    assert sample.is_synthetic is False
    assert len(sample.boxes) == 2
    assert sample.boxes[0]["canonical_label"] == "chair"


def test_scene_disjoint_splits_prevent_scene_leakage() -> None:
    samples = []
    for scene_idx in range(10):
        scene_id = f"scene_{scene_idx:02d}"
        for frame_idx in range(4):
            samples.append(
                normalize_dataset_sample(
                    _sample_raw(f"{scene_id}_f{frame_idx}", scene_id),
                    dataset_id="scannet",
                )
            )
    splits = create_scene_disjoint_splits(samples, train_ratio=0.60, val_ratio=0.20, seed="test_seed")
    assert set(splits.keys()) == {"train", "val", "test"}
    total = sum(len(v) for v in splits.values())
    assert total == len(samples)

    train_scenes = {s.scene_id for s in splits["train"]}
    val_scenes = {s.scene_id for s in splits["val"]}
    test_scenes = {s.scene_id for s in splits["test"]}
    assert train_scenes.isdisjoint(val_scenes)
    assert train_scenes.isdisjoint(test_scenes)
    assert val_scenes.isdisjoint(test_scenes)


def test_scene_disjoint_splits_are_deterministic() -> None:
    samples = [
        normalize_dataset_sample(_sample_raw(f"s_{i}", f"scene_{i // 2}"), dataset_id="nyu_depth_v2")
        for i in range(16)
    ]
    s1 = create_scene_disjoint_splits(samples, seed="repro_seed")
    s2 = create_scene_disjoint_splits(samples, seed="repro_seed")
    for split_name in ("train", "val", "test"):
        assert [x.sample_id for x in s1[split_name]] == [x.sample_id for x in s2[split_name]]


def test_validate_dataset_integrity_detects_scene_leakage_and_corrupt_samples() -> None:
    s1 = normalize_dataset_sample(_sample_raw("f1", "shared_scene"), dataset_id="sunrgbd")
    s2 = normalize_dataset_sample(_sample_raw("f2", "shared_scene"), dataset_id="sunrgbd")
    report = validate_dataset_integrity(
        "sunrgbd",
        [s1, s2],
        splits={"train": [s1], "val": [s2], "test": []},
        check_files_exist=False,
    )
    assert isinstance(report, DatasetIntegrityReport)
    assert report.leakage_detected is True
    assert report.is_healthy is False

    bad_sample = NormalizedDatasetSample(
        sample_id="bad_1",
        dataset_id="sunrgbd",
        scene_id="scene_bad",
        image_path="missing.jpg",
        image_width=0,
        image_height=480,
        boxes=(),
        is_synthetic=False,
    )
    report_bad = validate_dataset_integrity(
        "sunrgbd",
        [bad_sample],
        splits={"train": [bad_sample], "val": [], "test": []},
        check_files_exist=False,
    )
    assert report_bad.is_healthy is False
    assert len(report_bad.corrupt_samples) >= 1


def test_ade20k_nyu_and_synthetic_adapters_load_manifests(tmp_path: Path) -> None:
    ade_manifest = tmp_path / "ade20k.json"
    ade_manifest.write_text(
        json.dumps([
            {
                "sample_id": "ade_1",
                "scene_id": "bedroom_01",
                "image_path": "ade/ade_1.jpg",
                "width": 800,
                "height": 600,
                "objects": [
                    {"name": "bed", "bbox_xyxy": [100, 200, 500, 520]},
                    {"name": "wall", "bbox_xyxy": [0, 0, 800, 450]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    ade_samples = load_ade20k_samples(ade_manifest)
    assert len(ade_samples) == 1
    assert ade_samples[0].is_synthetic is False
    assert "bed" in ADE20K_CLASS_MAP

    nyu_manifest = tmp_path / "nyu.json"
    nyu_manifest.write_text(
        json.dumps([
            {
                "sample_id": "nyu_1",
                "scene_id": "kitchen_02",
                "image_path": "nyu/nyu_1.png",
                "width": 640,
                "height": 480,
                "objects": [
                    {"label": "counter", "bbox_xyxy": [50, 180, 420, 440]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    nyu_samples = load_nyu_depth_v2_samples(nyu_manifest)
    assert len(nyu_samples) == 1
    assert nyu_samples[0].is_synthetic is False
    assert "chair" in NYU_CLASS_MAP

    syn_manifest = tmp_path / "structured3d.json"
    syn_manifest.write_text(
        json.dumps([
            {
                "sample_id": "s3d_1",
                "scene_id": "room_99",
                "image_path": "s3d/s3d_1.png",
                "width": 1024,
                "height": 768,
                "boxes": [
                    {"label": "sofa", "bbox_xyxy": [200, 300, 700, 620]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    syn_samples = load_synthetic_indoor_samples(syn_manifest, dataset_id="structured3d")
    assert len(syn_samples) == 1
    assert syn_samples[0].is_synthetic is True


def test_human_reviewed_provenance_requires_reviewer_and_timestamp() -> None:
    with pytest.raises(ValueError):
        AnnotationProvenance(
            annotator_id="ann_1",
            review_status=ReviewStatus.HUMAN_REVIEWED,
            reviewer_id=None,
            reviewed_at_utc=None,
        )

    valid = AnnotationProvenance(
        annotator_id="ann_1",
        review_status=ReviewStatus.HUMAN_REVIEWED,
        reviewer_id="rev_1",
        reviewed_at_utc="2026-11-26T12:00:00Z",
    )
    assert valid.review_status == ReviewStatus.HUMAN_REVIEWED


def test_inter_annotator_agreement_and_quality_scoring() -> None:
    prov_a = AnnotationProvenance(annotator_id="ann_a", review_status=ReviewStatus.IN_REVIEW)
    prov_b = AnnotationProvenance(annotator_id="ann_b", review_status=ReviewStatus.IN_REVIEW)
    prov_rev = AnnotationProvenance(
        annotator_id="ann_a",
        review_status=ReviewStatus.HUMAN_REVIEWED,
        reviewer_id="rev_lead",
        reviewed_at_utc="2026-11-26T12:10:00Z",
    )

    anns_a = [
        GroundTruthAnnotation(
            annotation_id="g1",
            image_id="img_1",
            raw_label="wooden chair",
            canonical_label="chair",
            bbox_px=(100.0, 120.0, 220.0, 300.0),
            image_width=640,
            image_height=480,
            provenance=prov_rev,
        ),
        GroundTruthAnnotation(
            annotation_id="g2",
            image_id="img_1",
            raw_label="door",
            canonical_label="door",
            bbox_px=(300.0, 40.0, 420.0, 420.0),
            image_width=640,
            image_height=480,
            provenance=prov_a,
        ),
    ]
    anns_b = [
        GroundTruthAnnotation(
            annotation_id="h1",
            image_id="img_1",
            raw_label="chair",
            canonical_label="chair",
            bbox_px=(104.0, 122.0, 218.0, 298.0),
            image_width=640,
            image_height=480,
            provenance=prov_b,
        ),
        GroundTruthAnnotation(
            annotation_id="h2",
            image_id="img_1",
            raw_label="window",
            canonical_label="window",
            bbox_px=(305.0, 45.0, 415.0, 415.0),
            image_width=640,
            image_height=480,
            provenance=prov_b,
        ),
    ]
    agreement = compute_inter_annotator_agreement(anns_a, anns_b, iou_threshold=0.50)
    assert agreement["matched_pairs"] == 2
    assert agreement["label_agreement_on_matched"] == pytest.approx(0.5, abs=1e-6)
    assert agreement["mean_matched_iou"] > 0.85

    quality = score_ground_truth_dataset_quality(anns_a)
    assert quality["total_annotations"] == 2
    assert 0.0 <= quality["quality_score"] <= 1.0
    assert quality["human_reviewed_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_human_review_manifest_and_json_roundtrip(tmp_path: Path) -> None:
    image_paths = [f"datasets/indoor/img_{i:02d}.jpg" for i in range(15)]
    manifest = build_human_review_manifest(image_paths, required_count=15)
    assert manifest["required_image_count"] == 15
    assert len(manifest["items"]) == 15

    prov = AnnotationProvenance(
        annotator_id="ann_1",
        review_status=ReviewStatus.HUMAN_REVIEWED,
        reviewer_id="rev_1",
        reviewed_at_utc="2026-11-26T12:30:00Z",
    )
    ann = GroundTruthAnnotation(
        annotation_id="ann_001",
        image_id="img_00",
        raw_label="oak dining table",
        canonical_label="table",
        bbox_px=(80.0, 160.0, 460.0, 410.0),
        image_width=640,
        image_height=480,
        provenance=prov,
        cross_view_identity_id="obj_table_1",
        is_occluded=True,
        material_label="wood",
    )
    out_file = tmp_path / "gt_export.json"
    export_ground_truth_json([ann], out_file)
    reloaded = import_ground_truth_json(out_file)
    assert len(reloaded) == 1
    assert reloaded[0].annotation_id == "ann_001"
    assert reloaded[0].canonical_label == "table"
    assert reloaded[0].cross_view_identity_id == "obj_table_1"
    assert reloaded[0].provenance.review_status == ReviewStatus.HUMAN_REVIEWED
