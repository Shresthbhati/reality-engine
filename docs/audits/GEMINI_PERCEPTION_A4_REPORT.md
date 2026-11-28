# GEMINI PERCEPTION — WORKSTREAM A4 AUDIT & EXECUTION REPORT

**Branch:** `feat/gemini-december-perception`  
**Starting Baseline (A3 Completion):** `8bf3f547`  
**Date:** 2026-11-26  
**Scope:** Workstream A4 — Evidence-Grade Perception, Architectural Extraction, Dataset Infrastructure, Multi-View/Depth Integration, Calibration, Claude Handoff Contract, and Production Diagnostics 2.0.

---

## 1. Executive Summary

Workstream A4 delivers a complete, tested, provenance-preserving perception architecture built on top of the A3 foundation (`8bf3f547`), without modifying any Claude-owned reconstruction, pose-graph, or WorldStore files.

Key outcomes:
- **14 new production modules** across `perception/datasets/`, `perception/intelligence/`, and `perception/contracts/` (`3,050` net new production lines), plus `benchmarks/perception/a4_accuracy_suite.py` (`221` lines).
- **5 new automated test suites** (`1,510+` net new test lines) covering dataset licensing/splitting/integrity, human-reviewed ground-truth workflows, architectural geometry 2.0, architectural relationship reasoning, elevation profiles, furniture 2.0, multi-view identity/continuity, depth-aware association, open-vocab & CLIP/SigLIP verification, fusion 2.0, active perception 2.0, calibration/ECE/MCE/Brier, scene readiness audits, `perception_handoff.v1`, and diagnostics 2.0.
- **100% pass rate** across all A4, A3, and December/contracts/calibration/pipeline/evidence-world regression suites (`182` tests executed, `182` passed, `0` skipped).
- **Real-photograph production execution** verified on `datasets/south_building/images` (`P1180141.JPG`, `P1180145.JPG`) with Grounding DINO (`IDEA-Research/grounding-dino-tiny`) on CUDA (`NVIDIA GeForce RTX 4050 Laptop GPU`).

---

## 2. Implemented Subsystems (A4.1 – A4.13)

| Workstream | Module(s) | Capabilities Delivered |
| :--- | :--- | :--- |
| **A4.1 Indoor Dataset Infrastructure** | `perception/datasets/indoor_registry.py`, `perception/datasets/ade20k_and_nyu_adapters.py`, `perception/datasets/scannet_and_sunrgbd_adapters.py` | Catalog of 8 datasets (`scannet`, `scannetpp`, `sunrgbd`, `nyu_depth_v2`, `ade20k`, `structured3d`, `hypersim`, `replica`) with licensing/terms enforcement (`RestrictedDatasetTermsError`), camera intrinsics & 4x4 pose validation, deterministic scene-disjoint train/val/test splits, integrity & leakage detection, and strict `is_synthetic` vs `is_physically_surveyed` provenance. |
| **A4.2 Human-Reviewed Ground Truth** | `perception/datasets/ground_truth.py` | `ReviewStatus` (`unreviewed`, `in_review`, `human_reviewed`, `disputed`), `AnnotationProvenance` invariant forbidding `HUMAN_REVIEWED` without `reviewer_id` + `reviewed_at_utc`, inter-annotator agreement (box IoU + label agreement), dataset quality scoring, 15-image human-review manifest builder, and JSON import/export. |
| **A4.3 Architectural Perception 2.0** | `perception/intelligence/architectural_v2.py`, `perception/intelligence/architectural_elevation.py` | OpenCV Canny + Hough line-segment extraction (supporting both OpenCV 4.x `(N,1,4)` and OpenCV 5.0 `(N,4)` outputs), corner candidates, vanishing hypotheses, image-space plane hypotheses (`is_metric=False`), wall-thickness jamb span hypotheses in pixels, built-in cabinet detection, left-to-right wall elevation profiles, opening-to-wall ratios, and room-envelope completeness scoring. |
| **A4.4 Architectural Relationship Reasoning** | `perception/intelligence/architectural_relations.py` | Infers `window_contained_in_wall`, `door_contained_in_wall`, `wall_connected_to_wall`, `floor_meets_wall`, `ceiling_meets_wall`, `column_intersects_floor`, and `opening_interrupts_wall_boundary`; builds parent-child component graphs and flags floor-above-ceiling contradictions. |
| **A4.5 Furniture Perception 2.0** | `perception/intelligence/furniture_v2.py` | Explicit mobility taxonomy (`movable_furniture`, `built_in_furniture`, `appliance`, `fixture`), support surface linking, furniture arrangement grouping (`dining_or_meeting_set`, `workstation`, `seating_lounge`), and foreground occlusion pair detection. |
| **A4.6 Multi-View Perception** | `perception/intelligence/multiview.py` | Integrates external `CameraPoseConstraint` baseline compatibility with appearance/box/class matching, disambiguates identical chairs across views, builds visibility tracks, and tracks architectural continuity. |
| **A4.7 Depth-Aware Perception** | `perception/intelligence/depth_aware.py` | `DepthMapContract` enforcing `relative` vs `sfm_gauge` vs `metric` unit honesty, valid fraction, median depth, IQR, boundary discontinuity, and front-to-back relative object ranking. |
| **A4.8 Open-Vocabulary & CLIP/SigLIP** | `perception/intelligence/open_vocab.py`, `perception/intelligence/clip_semantic.py` | Open-vocabulary canonicalization with unknown-class preservation, survey prompt generation, confusion-pair tracking, and offline-cached `ClipSemanticVerifier` (`local_files_only=True`, `is_calibrated=False`). |
| **A4.9 Perception Fusion 2.0** | `perception/intelligence/fusion_v2.py` | Family-grouped correlation-aware fusion (`engine_family`), image-quality weighting, detector/mask/depth disagreement tracking, and propagated uncertainty. |
| **A4.10 Active Perception 2.0 & Audit** | `perception/intelligence/active_v2.py`, `perception/intelligence/scene_audit_v2.py` | Compute-budgeted refinement prioritization (`poor` images, poorly observed walls, uncertain openings/furniture) with explicit stopping criteria, plus `ScenePerceptionAuditReport` for survey readiness. |
| **A4.11 Accuracy, Benchmarks & Calibration** | `benchmarks/perception/a4_accuracy_suite.py`, `perception/intelligence/calibration_v2.py` | Mask IoU, boundary F1, IDF1, relationship F1, bootstrap 95% AP CIs, regression comparison, ECE/MCE/Brier reliability bins, and temperature scaling (strictly refusing `is_calibrated=True` on synthetic or unlabelled data). |
| **A4.12 Claude Handoff Contract** | `perception/contracts/claude_handoff_v1.py` | Versioned `perception_handoff.v1` schema (`HandoffObservation`, `HandoffBundle`) with strict validation and JSON roundtrip without coupling internal modules. |
| **A4.13 Production Diagnostics 2.0 & Pipeline** | `perception/intelligence/diagnostics_v2.py`, `perception/intelligence/pipeline_v2.py` | Surfaces `model_timeout`, `gpu_memory_pressure`, `missing_depth_association`, `dataset_corruption_or_leakage`, and `ground_truth_fp_fn_summary`, composed end-to-end via `run_a4_perception_pipeline()`. |

---

## 3. Automated Verification Results

All suites passed after the final code edit:

| Suite | Scope | Result |
| :--- | :--- | :--- |
| `tests/test_a4_*.py` (5 files) | All A4.1–A4.13 modules & end-to-end pipeline | **44 passed, 0 failed** |
| `tests/test_perception_intelligence_*_a3.py` (7 files) | A3 perception intelligence regression | **87 passed, 0 failed** |
| `tests/test_december_perception_suite.py`, `test_normalized_detection_contracts.py`, `test_multiengine_calibration.py`, `test_detection_pipeline_engine.py`, `test_evidence_world_job.py` | Core detection, contracts, calibration, and API job integration | **51 passed, 0 failed** |
| **Total Verified** | **17 test files** | **182 passed, 0 failed** |

---

## 4. Real-Photograph Production Execution (`datasets/south_building/images`)

- **Images:** `P1180141.JPG`, `P1180145.JPG` (`1024x768`, unlabelled real outdoor architectural photographs)
- **Active Detector:** `detection:grounding_dino:IDEA-Research/grounding-dino-tiny` (`threshold=0.30`, `DEFAULT_SURVEY_QUERIES`)
- **Runtime & Peak VRAM:** `20.94s`, `1836.06 MB` on CUDA (`NVIDIA GeForce RTX 4050 Laptop GPU`)
- **Outputs:**
  - `19` raw detections (`handoff_accepted=19`, `handoff_rejected=0`)
  - Image quality profiles: both `good` (`laplacian_var`: `7233.88` and `10640.62`)
  - Architectural geometry 2.0: `80` line segments/image, `2` vanishing hypotheses/image, `2` plane hypotheses/image (`is_metric=False`), wall-thickness jamb spans `4.57–5.71 px`, `3` architectural relationships, `0` contradictions
  - Multi-view: `13` total identities, `4` confirmed matches, `6` multi-view tracked, `1` continuous architectural element, `2` split proposals
  - Diagnostics 2.0: `status="degraded"` with `3` honest warnings (`repeated_duplicate_detections`, `all_masks_missing`, `inconsistent_object_identity`)
  - Benchmark honesty: `mode="unlabelled_real"`, `accuracy_claim_allowed=False`, `detection_map=None` (`"Unlabelled dataset: precision/recall/AP/ECE are None; counts are not accuracy."`)

---

## 5. Dataset Provenance & Honesty Guardrails

- **Real vs. Synthetic Separation:** `Structured3D`, `Hypersim`, and `Replica` are explicitly distinguished from physical-world survey benchmarks; `EvaluationMode.SYNTHETIC` and `EvaluationMode.UNLABELLED_REAL` force `accuracy_claim_allowed=False`.
- **Terms-of-Use Enforcement:** `ScanNet` and `ScanNet++` require `terms_accepted=True` and raise `RestrictedDatasetTermsError` otherwise.
- **No Fabricated Calibration:** `calibration_v2.py` and `clip_semantic.py` refuse to set `is_calibrated=True` without human-reviewed real ground truth.

---

## 6. Remaining Blockers & Next Steps

1. **Human-Reviewed 15-Image Indoor Ground Truth:** The 15-image review manifest workflow (`build_human_review_manifest`) is implemented and tested, but physical-world mAP/ECE calibration awaits human-reviewed annotations on 15+ real indoor survey photographs.
2. **SAM2 Mask Generation in Multi-Image CLI Probe:** The 2-photo Grounding DINO probe ran detector-only (`all_masks_missing` flagged honestly by `diagnostics_v2.py`); full SAM2 mask association requires invoking the segmentation stage in the multi-engine orchestrator.
3. **Offline CLIP/SigLIP Checkpoint:** `ClipSemanticVerifier` runs when a local HF cache checkpoint (`google/siglip-base-patch16-224` or `openai/clip-vit-base-patch32`) is present and cleanly reports `status="unavailable"` otherwise.
