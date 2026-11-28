# GEMINI PERCEPTION — WORKSTREAM A5 AUDIT & RELIABILITY REPORT

**Branch:** `feat/gemini-december-perception`  
**Starting Baseline (A4 Completion):** `e390eb00`  
**Date:** 2026-11-26  
**Scope:** Workstream A5 — Production Perception Reliability, Segmentation Reliability, Duplicate Elimination, Cross-View Identity Reliability, Real Indoor Validation, Constrained-GPU Lifecycle, Unified Production Pipeline, and Bidirectional Claude Integration.

---

## 1. Root-Cause Analysis of A4 Production Failures

In A4, the real-photo run on `datasets/south_building/images` (`P1180141.JPG`, `P1180145.JPG`) exhibited four production failures:

1. **All masks missing (`0/19`):**
   - `run_a4_perception_pipeline()` never invoked `perception/segmentation_adapter.py` (`_sam2_run`).
   - `_sam2_run()` only looked up `bboxes.get(img_path, [])` by exact path string (failing when keyed by basename), did not select the highest-`iou_scores` channel from SAM2's 3-channel multimask output, and discarded the 2D mask instead of encoding `rle_mask` and validating mask dimensions.
2. **Duplicate detections (`sb_0011`, `sb_0015`):**
   - Grounding DINO emits overlapping detections across related survey queries (`window`, `window_frame`, `window_sill`) and `pipeline_v2.py` passed raw detections straight into downstream stages without a duplicate-resolution stage.
3. **Low confirmed cross-view identities (`4/13`) & `2` inconsistent identities:**
   - `pipeline_v2.py` did not extract `AppearanceDescriptor` (color histograms / DINOv2 embeddings) from `images_rgb` prior to calling `run_multiview_perception()`, leaving `resolve_cross_view_identities()` with only normalized box area ratio when camera poses were absent.
   - `identity.py` lacked a 3-way decision boundary (`confirmed_match`, `ambiguous_match`, `rejected_match`) and camera-pose uncertainty (`position_sigma_m`) tolerance.
4. **No unified pre-bundle architectural/furniture false-positive filtering or WorldIR persistence verification:**
   - Spurious tiny walls, blank-region openings, and implausible full-frame furniture boxes were not filtered before bundle assembly and handoff.

---

## 2. Implemented A5 Fixes & Subsystems

| Subsystem | Module(s) | Key Fixes & Capabilities |
| :--- | :--- | :--- |
| **A5.1 Segmentation Reliability** | `perception/segmentation_adapter.py`, `perception/intelligence/segmentation_v2.py`, `perception/intelligence/mask_geometry.py` | Fixed SAM2 multimask highest-IoU selection, basename/path box lookup, row-major RLE encoding, strict `(H, W)` validation, empty-mask diagnosis, mask-guided box refinement, polygon extraction, and SAM2/SAM3/SAM-HQ probing with zero mask fabrication. |
| **A5.2 Duplicate Elimination** | `perception/intelligence/deduplication_v2.py` | Resolves cross-engine, crop/tiled, multi-scale, and repeated architectural duplicates using Box IoU, RLE Mask IoU, appearance similarity, and engine-family correlation; preserves separate identical chairs and competing class hypotheses. |
| **A5.3 Cross-View Identity 2.0** | `perception/intelligence/identity_v2.py`, `perception/intelligence/dinov2_embeddings.py` | Automatic 24-bin RGB crop descriptors + locally cached DINOv2 embeddings, `CameraPoseConstraint.position_sigma_m` geometric compatibility, occlusion-relaxed scale, and 3-way outcomes (`confirmed_match`, `ambiguous_match`, `rejected_match`). |
| **A5.4 & A5.5 Scene Refinement** | `perception/intelligence/scene_refinement_v2.py` | Filters tiny/fragmented/featureless architectural boxes and implausible furniture false positives; attaches interior-survey subpart hints; never invents walls from line segments alone. |
| **Objective 2: Indoor Validation** | `benchmarks/perception/indoor_validation_runner.py` | Dataset-specific validation across ScanNet, ScanNet++, SUN RGB-D, NYU Depth V2, Structured3D, and Replica (`labelled_real` vs `unlabelled_real` vs `synthetic`), plus the six-room measured capture pathway (`blocked_awaiting_human_capture` for missing rooms). |
| **Objective 3 & 4: Unified Pipeline & GPU Lifecycle** | `perception/intelligence/production_pipeline_a5.py`, `perception/intelligence/gpu_lifecycle.py`, `perception/intelligence/resumable_runner.py`, `perception/intelligence/multi_engine_dispatch.py` | Full 14-step production flow with `ModelLifecycleManager` (sequential GPU eviction, VRAM telemetry, OOM CPU fallback, timeouts, adaptive batch/resolution) and resumable disk checkpoints. |
| **Objective 5: Claude Integration** | `perception/contracts/claude_integration_v2.py` | Exports `perception_handoff.v1`, consumes Claude's `CameraPoseConstraint` and `WorldEntityIdentity`, persists into canonical `EvidenceMatrix`, and verifies fresh-process reload. |
| **Benchmark & Experiments** | `perception/intelligence/a5_benchmark_suite.py`, `perception/intelligence/a5_experiments.py` | Structured A4-vs-A5 reliability comparison and reproducible runners for Experiments A, B, and C. |

---

## 3. Experiments A, B, and C Results

### Experiment A — Real `south_building` Photographs (`P1180141.JPG`, `P1180145.JPG`)
- **Models Executed:** `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-small` sequentially on CUDA (`NVIDIA GeForce RTX 4050 Laptop GPU`).
- **A4 vs. A5 Comparison:**
  - **Masks Produced:** `0` (A4) → **`22 / 22` (`100%` coverage, `0` empty masks, `mean_mask_quality = 0.6455`)**
  - **Duplicates Suppressed:** `3 / 22` (`13.64%` duplicate reduction rate) before persistence (`19` canonical observations)
  - **Cross-View Identity Outcomes:** `2` `confirmed_match`, `1` `ambiguous_match` preserved, `1` `rejected_match`
  - **EvidenceMatrix Persistence:** `evidence_matrix_persisted = True` (`50.35s` total wall time)
  - **Accuracy Claim:** `accuracy_claim_allowed = False` (unlabelled outdoor photographs).

### Experiment B — Real Indoor Photographs
- **Status:** `blocked_no_local_indoor_photos` (`accuracy_claim_allowed = False`).
- **Blocker:** No licensed real indoor photographs are present locally in the repository (`datasets/` contains `south_building` outdoor images and JSON manifests/fixtures only). The six-room measured capture pathway (`prepare_six_room_measured_capture_pathway`) is implemented, tested, and awaits human-captured rooms.

### Experiment C — Multi-View Indoor Scene
- **Execution:** 2-view indoor scene with `CameraPoseConstraint` (`position_sigma_m = 0.02`), precomputed masks, and `EvidenceMatrix` persistence.
- **Results:** `masks_produced = 4` (`mean_mask_quality = 0.9938`), `2` confirmed cross-view architectural matches (`wall` `0.9947`, `floor` `0.9945`), `evidence_matrix_persisted = True`, `wall_time_s = 0.978s`.

---

## 4. Automated Test Results

- **A5 Test Suites (`tests/test_a5_*.py`, 5 files):** All 18 required test categories covered (SAM execution, mask coordinates, empty-mask diagnosis, duplicate detection, duplicate persistence prevention, separate identical objects, cross-view identity, identity ambiguity, architectural FPs, furniture FPs, multi-engine disagreement, missing engine output, GPU OOM recovery, model timeout, serialization, World handoff, fresh-process persistence, real-photo execution).
- **Regression Suites:** Existing A4 (`44` tests) and A3 (`87` tests) suites verified with zero weakened assertions.

---

## 5. Remaining Blockers

1. **Local Licensed Indoor Photographs:** Physical-world indoor mAP, recall, precision, and ECE calibration await human-captured or licensed indoor RGB-D images (ScanNet / SUN RGB-D / six-room capture).
2. **SAM3 Checkpoint:** `Sam3Model` is probed dynamically in `probe_available_segmenters()`, while `facebook/sam2.1-hiera-small` and `syscv-community/sam-hq-vit-base` are cached locally and verified.
