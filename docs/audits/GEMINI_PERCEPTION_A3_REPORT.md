# Gemini Perception A3 Engineering Expansion Report

**Branch:** `feat/gemini-december-perception`  
**Starting commit:** `f9b7d8be`  
**Previous baseline:** `a4d9ef67`  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (`torch 2.14.0+cu130`, Python 3.14.7)

---

## 1. Subsystem Architecture Implemented (A3.1 – A3.13)

| Work Package | Module | Responsibility |
|---|---|---|
| A3.1 Data Model | `perception/intelligence/data_model.py` | Versioned (`perception.intelligence.v1`), strictly validated, JSON-serializable observation schemas with explicit `CoordinateSystemSpec`, `ScoreSemantics`, `ModelExecutionProvenance`, `DetectionProvenance`, `ObservationRevision`, and `PerceptionBundle`. |
| A3.2 Architectural Intelligence | `perception/intelligence/architectural.py` | Distinguishes primary surfaces (`wall`, `floor`, `ceiling`), supports (`column`, `beam`), `opening_candidate` vs `verified_opening`, framing, circulation (`staircase`, `railing`), host-wall hypotheses (`observed`/`hypothesized`/`unknown`), boundary continuity, and competing structural conflicts. |
| A3.3 Furniture Intelligence | `perception/intelligence/furniture.py` | Hierarchical furniture & fixture classification, subpart ownership (`table_leg`, `chair_seat`, `drawer`, `drawer_handle`, etc.), per-image duplicate suppression, and movable vs fixed-fixture distinction. |
| A3.4 Identity & Association | `perception/intelligence/identity.py` | Cross-view identity resolution combining embeddings, histograms, semantic compatibility, and `CameraPoseConstraint` geometry while preserving ambiguity (4 identical chairs stay 4 identities) plus merge/split proposals. |
| A3.5 Multi-Engine Orchestration | `perception/intelligence/orchestration.py` | `PerceptionEngineScheduler` with task/quality/class/resolution-aware routing, sequential GPU execution, `EngineHealthState` circuit-breaker, failure isolation, and correlation-aware family fusion. |
| A3.6 Adaptive Image Processing | `perception/intelligence/image_adaptive.py` | Resolution, Laplacian blur, MAD noise, exposure, contrast, and texture analysis; adaptive tiling + targeted high-res crop proposals; crop-to-image coordinate transformation and duplicate reconciliation. |
| A3.7 Segmentation Intelligence | `perception/intelligence/segmentation_intel.py` | Mask quality scoring (solidity, box IoU, RGB edge alignment), `visible_object_mask` / `occluded_object_hypothesis` / `architectural_surface_mask` / `structural_boundary` classification, overlapping instance pixel separation, and depth association. |
| A3.8 Material & Appearance | `perception/intelligence/materials_intel.py` | Color/texture descriptors and ranked material hypotheses (`wood`, `metal`, `glass`, `concrete`, `brick`, `stone`, `ceramic`, `fabric`, `plastic`, `paint`) strictly flagged `is_physically_verified=False`, plus cross-view consensus. |
| A3.9 Scene Relationships | `perception/intelligence/relationships.py` | Spatial predicate inference (`door_within_wall`, `window_within_wall`, `chair_beside_table`, `cabinet_against_wall`, `bed_against_wall`, `light_attached_to_ceiling`, `resting_on_floor`) and contradiction detection. |
| A3.10 Active Refinement | `perception/intelligence/refinement.py` | Information-gain / cost prioritized refinement planner for missing architecture, missing masks, competing labels, unverified openings, and ambiguous identities with budget-exhaustion stopping. |
| A3.11 Benchmarks & Runner | `benchmarks/perception/evaluator.py`, `benchmarks/perception/run_a3_benchmark.py` | Precision, Recall, F1, AP, IoU, duplicate rate, mask quality, identity accuracy, ECE/Brier calibration, runtime, VRAM, and failure rate; strictly separates `synthetic`, `unlabelled_real`, `labelled_real`, and `physically_measured`. |
| A3.12 World Integration | `perception/intelligence/world_integration.py`, `engine/multiengine/adapters/pipeline_detection.py` | Bridges `ObservationDraft` <-> `NormalizedPerceptionObservation` <-> `PerceptionBundle` <-> `EvidenceMatrix` / `WorldIR` and executes automatically inside `PipelineDetectionEngineAdapter.infer()`. |
| A3.13 Diagnostics | `perception/intelligence/diagnostics.py` | Detects empty detections on non-blank images, implausible volumes, duplicates, missing masks, architectural conflicts, identity split proposals, model failures, and serialization errors. |
| Visual Embeddings | `perception/intelligence/embeddings.py` | Sequential `facebook/dinov2-small` crop embedder (local HF cache) with deterministic 32-dim color+gradient descriptor fallback. |

---

## 2. Verification Results

### Automated Tests
- `tests/test_perception_intelligence_a3.py`, `tests/test_perception_intelligence_extended_a3.py`, `tests/test_perception_intelligence_e2e_a3.py`, `tests/test_perception_intelligence_scenarios_a3.py`, `tests/test_perception_intelligence_contracts_a3.py`, `tests/test_perception_intelligence_benchmarks_a3.py`, `tests/test_december_perception_suite.py`, `tests/test_normalized_detection_contracts.py`, `tests/test_multiengine_calibration.py`: **All passing** (verified via `pytest`).

### Real-Photo Execution (`python -m benchmarks.perception.run_a3_benchmark --max-images 2`)
- **Dataset:** `datasets/south_building/images` (`P1180141.JPG`, `P1180145.JPG`)
- **Mode:** `unlabelled_real` (`accuracy_claim_allowed = false`)
- **Active engines:** `grounding_dino` + `mask_rcnn` + `sam` + `perception.intelligence` bundle
- **Duplicate rate:** `0.0263`
- **Wall time:** `50.869s`
- **Peak VRAM:** `1619.1 MB`
- **Failure rate:** `0.0`
- **Calibration state:** `uncalibrated` (`calibrated_confidence = None`, raw scores preserved).

---

## 3. Remaining Blockers
1. **No physically surveyed indoor room dataset with labelled 2D/3D boxes:** `datasets/real_room_capture` is explicitly a synthetic fixture (`is_physically_surveyed: false`), and `datasets/south_building` is an unlabelled outdoor facade. Real indoor precision/recall/AP and Platt calibration fitting await mounting a licensed indoor dataset (e.g. SUN RGB-D or ScanNetv2).
