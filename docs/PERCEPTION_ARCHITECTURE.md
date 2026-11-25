# Reality Engine Perception Architecture & Inventory

## 1. End-to-End Pipeline

```
IMAGE / VIDEO INGESTION (frame indices, timestamps, duplicate-frame reduction)
  -> IMAGE QUALITY ANALYSIS (resolution, Laplacian blur, directional motion blur, exposure, clipped pixels, noise)
  -> ARCHITECTURAL & OBJECT DETECTION (Mask R-CNN, OWL-ViT, Grounding DINO, Florence-2)
  -> NORMALIZATION BOUNDARY (perception/detection/normalized.py: NormalizedDetectionBatch)
  -> ARCHITECTURAL SEGMENTATION (SAM 2.1 Hiera-Small box-prompted masks + validity/boundary/occlusion checks)
  -> DEPTH ESTIMATION & MASK-LINKED EVIDENCE (Depth Anything V2 Small / Metric Indoor Small -> median, IQR, valid fraction, discontinuity)
  -> CROSS-VIEW ASSOCIATION (DINOv2 embeddings + epipolar camera geometry + repeated-structure disambiguation)
  -> SCENE GRAPH & ROOM UNDERSTANDING (17 relationship types, epistemic states, multi-cue room hypotheses)
  -> MATERIAL HYPOTHESES (mask-cropped color + roughness, unverified ESTIMATED state)
  -> OBSERVATION LEDGER & WORLDIR BRIDGE (perception/world_bridge.py -> WorldIR / WorldStore)
```

## 2. Normalized Detection Contract (`perception/detection/normalized.py`)

All model-specific outputs (`Detection.box`, `ImageDetection.box_xyxy`, Florence-2 `<OD>` dicts, Grounding DINO tensors) pass through `normalize_backend_output(...)` / `normalize_detection(...)`:
- Explicit coordinate convention (`PIXEL_XYXY` or `NORMALIZED_XYXY`) with explicit `(image_width, image_height)`
- Rejects negative, out-of-bounds, NaN/Inf, and degenerate (`x1 <= x0` or `y1 <= y0`) boxes
- Preserves `confidence=None` when a model emits no score (e.g. Florence-2)
- Preserves unknown categories without forcing them into fixed COCO labels
- Separates `DetectionStatus.EMPTY` (genuine 0 detections) from `DetectionStatus.FAILED` (model error)

## 3. Model Inventory, Licenses & Runtime Requirements

| Capability | Model / Adapter | Checkpoint / Source | License | RTX 4050 VRAM (Sequential) | Probe Key (`perception/availability.py`) |
|---|---|---|---|---|---|
| Relative Depth | `DepthAnythingV2Backend` | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | ~246 MB alloc | `depth_anything` |
| Metric Indoor Depth | `DepthAnythingV2MetricBackend` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Apache-2.0 | ~250 MB alloc | `depth_anything_metric` |
| Closed-Vocab Det + Seg | `MaskRCNNDetector` | `maskrcnn_resnet50_fpn_coco` | BSD-3-Clause | ~630 MB alloc | `mask_rcnn` |
| Open-Vocab Det (1) | `OwlVitDetector` | `google/owlvit-base-patch32` | Apache-2.0 | ~630 MB alloc | `owlvit` |
| Open-Vocab Arch Det (2) | `GroundingDinoDetector` | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | ~1.2 GB alloc | `grounding_dino` |
| Second-Opinion Det | `Florence2Detector` | `florence-community/Florence-2-base` | MIT | ~1.5 GB alloc | `florence2` |
| Box-Prompted Seg | `Sam2BoxPrompter` | `facebook/sam2.1-hiera-small` | Apache-2.0 | ~3.1 GB peak alloc | `sam2_hf` |
| Cross-View Embeddings | `DINOv2LocalEmbedder` | `facebook/dinov2-small` | Apache-2.0 | ~200 MB alloc | `dinov2` |

Runtime requirements: Python 3.14, PyTorch 2.14+cu130, `transformers` 5.23+, `torchvision`, `opencv-python`, `numpy`, `Pillow`, NVIDIA GPU >= 6 GB VRAM (RTX 4050 Laptop GPU verified).

## 4. Feature Flags & Production Enablement

- `RE_EVIDENCE_WORLD` (default OFF): Opt-in gate in `apps/api/evidence_world_stage.py` controlled by `ENABLEMENT_CRITERIA`. Remains opt-in until `real_multi_photo_job_run`, `detector_accuracy_benchmark`, `confidence_calibration`, and `quality_gate_thresholds` are satisfied on labelled datasets.
- `RE_EVIDENCE_WORLD_DETECTION` (`pipeline` | `legacy`)
- `RE_EVIDENCE_WORLD_DEPTH_ENGINES` (comma-separated depth adapter names)
- `RE_EVIDENCE_WORLD_LIFT` (`spatial` | `legacy`)

## 5. Benchmark Instructions

Run the contract, adversarial, scene intelligence, bridge, spatial reasoning, and real COLMAP photographic RTX 4050 benchmarks:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_normalized_detection_contracts.py tests/test_perception_mega_sprint.py tests/test_open_vocab_real_photo_benchmark.py tests/test_scene_intelligence_e2e.py tests/test_evidence_world_bridge_golden.py tests/test_spatial_reasoning_benchmark.py -v -s
```

## 6. Capability Classification Matrix

| Capability | Status | Notes |
|---|---|---|
| Normalized detection contract (`normalized.py`) | 1. Implemented and tested | 12/12 contract tests pass |
| Grounding DINO + OWL-ViT + Mask R-CNN + Florence-2 sequential pipeline | 1. Implemented and tested | Verified at 640px & 1024px on RTX 4050 |
| SAM 2.1 architectural segmentation + mask quality (`arch_segmenter.py`) | 1. Implemented and tested | Verified on real COLMAP photos (181/182 valid masks @ 640px) |
| Mask-linked depth evidence (`evidence_enrichment.py`) | 1. Implemented and tested | Preserves relative vs metric scale status |
| Image quality & observability (`world_bridge.py`, `scene_intelligence.py`) | 1. Implemented and tested | Separates `missing_evidence` from `suggestions` |
| DINOv2 + geometry cross-view association | 1. Implemented and tested | Disambiguates repeated structures; preserves contradictions |
| Video frame deduplication & temporal tracking | 1. Implemented and tested | Preserves frame indices, timestamps, tracks |
| WorldIR / WorldStore scene intelligence round-trip | 1. Implemented and tested | Verified with `WorldStore.save_version` / `load_version` |
| `spatial_inputs.v1` producer & `register_inputs_provider` (`perception/spatial_inputs.py`) | 1. Implemented and tested | Emitted by `apps/api/evidence_world_stage.py` (`spatial_inputs_v1.json`) |
| Production API job stage (`RE_EVIDENCE_WORLD=1`) | 2. Implemented; opt-in by default | Intentionally gated by `ENABLEMENT_CRITERIA` in `apps/api/evidence_world_stage.py` |
| Labelled interior precision/recall benchmark (ScanNet/Matterport3D) | 5. Blocked | No ScanNet/Matterport3D/Replica ground-truth datasets present on disk |

## 7. Real Photographic Multi-Session & FP16 Benchmarks (RTX 4050 Laptop GPU, 6140.5 MB VRAM)

Unscored real COLMAP sessions (`3` photos/session, `max_edge=640`, sequential `DepthAnythingV2 -> MaskRCNN -> OWL-ViT -> GroundingDINO -> SAM2.1 -> DINOv2`):
- `wld_1a3fc79fe10f`: `27.593s`, `122` observations (`91` object, `17` architecture, `14` opening), `121` valid SAM2 masks, `295` cross-view associations, `2168.9 MB` peak PyTorch alloc
- `wld_841c36c182db`: `22.261s`, `128` observations (`99` object, `14` architecture, `15` opening), `127` valid SAM2 masks, `317` cross-view associations, `2214.1 MB` peak PyTorch alloc
- `wld_b21dadebf399`: `26.590s`, `115` observations (`87` object, `14` architecture, `14` opening), `114` valid SAM2 masks, `271` cross-view associations, `2080.8 MB` peak PyTorch alloc

SAM 2.1 FP16 autocast vs FP32 trade-off (`6` boxes @ `640px`):
- FP32: `0.966s`, `538.4 MB` peak alloc
- FP16 autocast: `0.931s`, `535.8 MB` peak alloc, `mean_iou_vs_fp32 = 0.8946` (FP32 retained for boundary accuracy)

## 8. Known Limitations
- Ground-truth labelled indoor datasets (ScanNet, Matterport3D, Replica) are not present in this checkout; real-photo evaluations on COLMAP sessions are therefore qualitative/unscored for precision/recall.
- Minimal synthetic line-art rooms produce `0` detections from pretrained visual detectors (Mask R-CNN, OWL-ViT).
- `RE_EVIDENCE_WORLD` remains opt-in (`False` by default) pending dataset-calibrated confidence and quality thresholds.
