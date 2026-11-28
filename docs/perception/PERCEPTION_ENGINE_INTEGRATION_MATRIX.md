# Perception Engine Integration Matrix (Phase 2)

**Hardware Target:** Local execution on constrained NVIDIA GPU (RTX 4050, **6 GB VRAM** / `5.2 GB` safe budget in `engine/hardware/device_manager.py`).  
**Policy:** Engines execute **sequentially** (never loading all large models simultaneously), unload weights + `torch.cuda.empty_cache()` after each engine, preserve `ModelIdentity` provenance, and never silently substitute an unrelated detector on failure.

---

## 1. Engine Capability & Integration Matrix

| Engine ID | Model / Checkpoint | Detects / Computes | ROI vs Full Image | Output Type | Persists Observation? | Feeds Entity Construction? | Fused With Other Engines? | Model Identity Preserved? | Failures Visible? | Offline Reproducible? |
|---|---|---|---|---|---|---|---|---|---|---|
| `grounding_dino` | `IDEA-Research/grounding-dino-tiny` | Open-vocab 2D objects (large + 29 small classes via `production_queries`) | Full image + overlapping tiles (`tile_windows`) | 2D boxes + grounded ontology paths + scores | **Yes** (`ObservationTable` + Matrix) | **Yes** (`multi_engine_stage` + `world_bridge`) | **Yes** (with `maskrcnn`, `sam2`) | **Gap:** hardcodes `hardware_class="gpu"` on CPU | **Gap:** partial failures graded `"succeeded"` | **Gap:** retries without `local_files_only=True` |
| `maskrcnn` | `maskrcnn_resnet50_fpn` (COCO-80) | Closed-vocab 2D objects | Full image | 2D boxes + RLE masks in `metadata` | **Yes** | **Partial** (boxes yes; masks in `metadata`, not `.mask`) | **Yes** (boxes) | **Yes** (`hardware_class=device`) | **Gap:** partial failures graded `"succeeded"` | **Yes** (local torchvision cache) |
| `sam2` | `facebook/sam2.1-hiera-small` | Prompted instance masks | ROI (prompted by `grounding_dino` boxes) | Binary masks -> `MaskStore` `sha256://` | **Yes** (`ObservationRecord.mask` + `mask_ref`) | **Yes** (mask provenance; 3D lifting pending) | **Yes** (depends on `grounding_dino`) | **Yes** (PR #414/#416) | **Yes** (PR #416) | **Yes** (PR #416 strict offline) |
| `depth_anything` | `Depth-Anything-V2-Small-hf` | Relative monocular depth | Full image | Depth stats + `depth_map_ref` | **Yes** | **Partial** (relative only; not metric mesh) | **Partial** | **Gap:** hardcodes `hardware_class="gpu"` | **Gap:** partial failures graded `"succeeded"` | **Gap:** omits `local_files_only=True` |
| `midas` | `midas_dpt_hybrid` | Relative inverse depth | Full image | Depth stats | **Yes** | **Partial** | **Partial** | **Yes** (`hardware_class=device`) | **Gap:** partial failures graded `"succeeded"` | **Yes** (cached torch.hub) |
| `sift` | OpenCV SIFT + FLANN | 2D keypoints & two-view E/F geometry | Full image | Two-view inliers, Sampson error, `R, t` | **Yes** | **Yes** (pair verification) | **Yes** (with `lightglue_aliked`) | **Yes** (`cpu`) | **Gap:** partial pair failures graded `"succeeded"` | **Yes** (100% local) |
| `lightglue_aliked` | Kornia ALIKED + LightGlue | Learned 2D keypoints & two-view E/F geometry | Full image | Two-view inliers, Sampson error, `R, t` | **Yes** | **Yes** (pair verification) | **Yes** (with `sift`) | **Gap:** hardcodes `"gpu"` on CPU | **Gap:** partial pair failures graded `"succeeded"` | **Yes** (requires `torch`+`kornia`) |

---

## 2. Constrained Hardware Execution Policy (6 GB VRAM)

1. **Sequential Wave Execution (`MultiEngineOrchestrator`):** Peak single-engine VRAM is `4,000 MB` (`maskrcnn`) and `3,072 MB` (`sam2`), fitting within the `5,200 MB` safe budget when executed one-at-a-time with `del model; torch.cuda.empty_cache()` between engines.
2. **Per-Run Peak VRAM Reset:** Adapters calling `torch.cuda.max_memory_allocated(0)` must call `torch.cuda.reset_peak_memory_stats(0)` at run start so earlier engines do not inflate later engines' reported peak VRAM.
3. **Tiled Inference Resolution Policy:** Full-image resizing (~800px) erases `<40px` fixtures on 4000px photos; overlapping `tile_windows(scale=2|3, overlap=0.2)` preserves native pixel resolution within the same 2048 MB VRAM envelope.
4. **Strict Offline & Licence Gate:** Production paths must enforce `local_files_only=True` without silent network fallback and pass `perception.model_registry.assert_production_licences`.
