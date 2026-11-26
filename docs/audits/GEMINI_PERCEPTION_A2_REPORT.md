# Gemini Perception A2 Execution Report

**Branch:** `feat/gemini-december-perception`  
**Baseline:** `a4d9ef67` (PR #395)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (`torch 2.14.0+cu130`, Python 3.14.7)  
**Scope:** Real-photo production perception run (`apps.api.evidence_world_stage.run_for_job`) separated from synthetic unit-test fixtures.

---

## 1. Real-Photograph Production Run (`datasets/south_building/images`)

- **Dataset:** `datasets/south_building/images`
- **Image identities:** `P1180141.JPG`, `P1180145.JPG` (`1024 x 768` px RGB)
- **Job status:** `complete` (`wall_s = 99.39s`, `peak_vram_mb = 433.7 MB`)
- **Primary detector:** `grounding_dino` (`IDEA-Research/grounding-dino-tiny`)
- **Secondary routed detector:** `mask_rcnn` (`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)
- **Primary segmenter:** `sam` (`sam_vit_b_01ec64.pth`)
- **Multi-engine orchestrator engines executed (`5/5 succeeded`):** `grounding_dino`, `sam2` (`facebook/sam2.1-hiera-small`), `depth_anything` (`depth-anything/Depth-Anything-V2-Small-hf`), `lightglue_aliked`, `sift` (`total_wall_s = 7.175s`)

### Per-Image & Aggregate Counts

| Metric | `P1180141.JPG` | `P1180145.JPG` | Total |
|---|---:|---:|---:|
| Image dimensions | `1024 x 768` | `1024 x 768` | — |
| Tiles processed (`tile_size=800`) | 2 | 2 | 4 |
| Raw detections | 247 | 323 | 570 |
| Passed class confidence | 181 | 207 | 388 |
| Suppressed as duplicate | — | — | 357 |
| Below class confidence | — | — | 62 |
| Refined boxes (`kept`) | 60 | 91 | 151 |
| Subparts detected | 21 | 57 | 78 |
| Instance masks (`sam`) | 60 | 91 | 151 (`100%` coverage, `0` failures) |
| Multi-engine fused clusters | 45 (8 competing) | 59 (14 competing, 1 multi-family) | 104 |
| Runtime (pipeline per-image) | 20.95s | 30.41s | 51.36s |

### Cross-View Association, Persistence & Calibration State

- **Cross-view association (pipeline):** `pairs_evaluated = 1`, `matched = 19`, `ambiguous = 12`, `unmatched = 29`
- **Cross-view identity (multi-engine stage):** `matched = 4`, `ambiguous = 2`, `unmatched = 2`
- **Evidence Matrix persisted `bbox` observations:** `190` (`151` from `detection_pipeline` + `39` from `multi_engine_stage` / secondary)
- **Canonical entities:** `134` (`conflicts = 0`, `withheld_entities = 0`)
- **WorldIR persistence:** `status = "persisted"`, `version_id = "v-d449657e1b6e7275"`, `reload_verified = True`, `validation = "valid"`
- **Calibration state:** `all_calibrated_confidence_none = True` (`uncalibrated` — no fitted calibration dataset exists yet; raw detector scores are preserved separately).
- **Failures:** `0`

---

## 2. Regression & Synthetic Fixture Verification (Separate from Real-Photo Measurements)

- `tests/test_detection_pipeline_engine.py`: **19 passed** (`46.42s`) — includes blank uniform image guard (`np.std(arr, axis=(0, 1)).max() < 2.0` prevents Grounding DINO / Depth Anything from hallucinating detections or depth maps on solid-grey test frames).
- `tests/test_december_perception_suite.py`: **7 passed** — verifies BERT subword rejection, `MACRO` scale prompting, A2/A3 taxonomy classes, task-specific routing, family-grouped fusion without double-counting, ambiguity-preserving cross-view association, and local model probes.
- `tests/test_normalized_detection_contracts.py` & `tests/test_multiengine_calibration.py`: **24 passed** combined with `test_december_perception_suite.py` (`6.68s`), and **43 passed** (`67.61s`) across all four suites together.
- `tests/test_evidence_world_job.py`: **30 passed** (`52.14s`) together with the perception, contract, and calibration suites.

---

## 3. Known Gaps

1. **No labelled indoor survey ground truth yet:** `datasets/south_building` is an outdoor building facade (no indoor furniture), and `datasets/real_room_capture` must be checked via its manifest before claiming physical ground truth. Detection precision/recall on furnished indoor rooms remains unmeasured until a labelled indoor benchmark dataset is evaluated.
2. **Calibration unverified:** `calibrated_confidence` stays `None` until fitted against a labelled indoor development split.
