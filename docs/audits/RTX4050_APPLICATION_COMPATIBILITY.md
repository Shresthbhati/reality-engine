# Reality Engine: RTX 4050 Laptop (6 GiB) Application Compatibility

Target: HP Victus, NVIDIA GeForce RTX 4050 Laptop GPU (6 GiB VRAM). Branch `claude/integration-2026-11-05`, HEAD `4aed118f` (working tree had uncommitted edits to `perception/`, `reconstruction/matching/`, `engine/technologies/`; none of them were touched by this audit). Date 2026-11-06.

This file replaces an earlier draft that mixed "estimated" and "measured" values (for example "dense MVS ~6 GB VRAM", "Mask R-CNN ~2 GB", "MiDaS ~2 GB", "SAM works via torch.hub"). Those figures were not reproducible: dense MVS measured 0.4 GiB below, MiDaS cannot load at all on this machine, and SAM's repo backend is blocked. Nothing in the tables below is estimated. A model that cannot run here is listed as `UNAVAILABLE: <reason>`, with no number.

## 0. How to read the numbers

- **Shared machine.** Other agents ran small tests on this machine while these measurements were taken. `nvidia-smi` showed 0 MiB used on the GPU before every GPU run, and CPU-only runs show a constant +73 MiB on the GPU that belongs to someone else (display or another session), not to the benchmarked process. I did not measure CPU load from the other agents, so CPU timings may be slightly inflated.
- **Repeats.** Every row is 3 independent process launches. Cells are `min / median` of the 3. Single-run items are marked `n=1`.
- **Timing basis.** Weights were already in OS file cache and in `~/.cache/torch/hub` (no downloads were made). `load s` is model construction + `.to(device)` after Python/torch import and CUDA context creation, so it excludes the ~1-2 s of import and context start-up. `first` is the first image (cuDNN/kernel warm-up); `steady` is the median of images 2..N. Images: the first 8 of `datasets/south_building/images` (1024x768 JPEG). "512" means the same images resized to 512x384 with INTER_AREA.
- **Memory columns.**
  - `torch max alloc / reserved`: `torch.cuda.max_memory_allocated / max_memory_reserved` after `reset_peak_memory_stats`. Reserved is what the caching allocator holds from the driver.
  - `nvidia-smi peak delta`: peak `memory.used` sampled continuously (about every 0.1-0.2 s, so a very short spike could be missed) minus the value just before launch. It includes the CUDA context (about 90-115 MiB) and the allocator cache. This is the number that corresponds to "how much of the 6141 MiB card is gone".
  - `peak RAM`: Windows `PeakWorkingSetSize` of the process (psutil is not installed; read through `psapi` via ctypes).
- **Windows WDDM caveat (important).** The GPU runs under WDDM. The driver lets one process reserve more than the physical 6 GiB by spilling to shared system memory instead of raising OOM. Several rows below show `torch reserved` larger than the card (12,650 MB on a 6,141 MiB GPU). For those rows `nvidia-smi` pegs near the card size and host RAM balloons. Treat them as "not practical", not as "fits".
- **fp16** is `torch.autocast(float16)` applied by the harness; none of the repo code paths below use it today (they run fp32 even though `DeviceInfo.fp16` is True). It is a what-if, not current behaviour.
- `file:line` references were verified against the working tree on 2026-11-06; `apps/api/jobs.py` and `engine/pipeline/vertical_slice.py` were being edited by other sessions at the time, so those two may drift.
- Scripts and raw results are outside the repo (system temp, `rtxbench/`). Nothing in the repo was edited, committed, installed or downloaded.

## 1. Hardware and software (observed)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, WDDM, 6141 MiB total, 5920 MiB free when idle (0 MiB used) |
| Driver | 617.42 (CUDA UMD 13.4) |
| `torch.cuda.mem_get_info()` | total 6140 MiB, free **5080 MiB** when idle (the OS keeps about 1 GiB) |
| CPU | Intel Core i7-13620H (13th gen) |
| RAM | 15.65 GiB |
| Python / torch / torchvision | 3.14.7 / 2.14.1+cu130 / 0.29.1+cu130 (CUDA 13.0) |
| kornia / OpenCV / numpy | 0.9.0 / 5.0.0 (pip wheel, `cv2.cuda.getCudaEnabledDeviceCount() == 0`) / 2.5.3 |
| COLMAP | 4.2.0 (commit be5e291, built with CUDA), `C:\Users\shres\tools\colmap-extracted\bin\colmap.EXE` |
| Missing packages (probed) | psutil, timm, transformers, segment_anything, lightglue (not needed, kornia is used), pycolmap, open3d, onnxruntime, ultralytics |
| Cached weights (`~/.cache/torch/hub/checkpoints`) | maskrcnn_resnet50_fpn_coco, disk_lightglue, aliked_lightglue, aliked-n16, depth-save (DISK), sam_vit_b, dpt_hybrid_384, midas_v21_small_256, resnet18, efficientnet_b0, tf_efficientnet_lite3. HF cache has Depth-Anything-V2-Small and OWL-ViT-B/32. |

## 2. Per-model measurements (1024x768 input, plus one reduced size)

### 2.1 Mask R-CNN ResNet50-FPN (torchvision, COCO_V1), 8 images (CPU: 3 images)

Repo path: `perception/detection/maskrcnn_backend.py`. Note `MaskRCNNDetector.__init__(device="cpu")` (line 77) and `engine/pipeline/vertical_slice.py:964` constructs it without a device, so the pipeline runs it on CPU today. The 1024 rows use torchvision's default transform (internally resized to 1066x800); the 512 rows set `min_size=384, max_size=512` so the network really sees 512x384.

| config | load s | first img s | steady s/img | torch max alloc MB | torch max reserved MB | nvidia-smi peak delta MiB | peak RAM MB |
|---|---|---|---|---|---|---|---|
| cuda 1024 fp32 | 0.81 / 0.84 | 0.66 / 0.75 | 0.100 / 0.103 | 606 / 606 | 830 / 830 | 927 / 927 | 1575 / 1594 |
| cuda 1024 fp16 | 0.91 / 1.44 | 0.95 / 1.61 | 0.086 / 0.092 | 622 / 622 | 818 / 818 | 921 / 921 | 1705 / 1709 |
| cuda 512 fp32 | 0.81 / 0.82 | 0.67 / 0.82 | 0.064 / 0.086 | 300 / 300 | 460 / 460 | 557 / 557 | 1488 / 1505 |
| cuda 512 fp16 | 1.24 / 1.46 | 1.04 / 1.45 | 0.089 / 0.107 | 336 / 336 | 458 / 458 | 561 / 561 | 1610 / 1617 |
| cpu 1024 fp32 | 0.78 / 0.88 | 4.61 / 5.71 | 4.84 / 5.03 | n/a | n/a | n/a | 1417 / 1418 |
| cpu 512 fp32 | 0.77 / 0.80 | 1.68 / 1.68 | 1.61 / 1.62 | n/a | n/a | n/a | 1105 / 1110 |

GPU vs CPU at 1024: 0.103 s vs 5.03 s per image (about 49x). fp16 autocast gives no useful speed or memory gain for this model.

### 2.2 SAM ViT-B (checkpoint `sam_vit_b_01ec64.pth`), 8 images (CPU: 2)

The repo backend (`perception/segmentation/sam_backend.py:53`) imports `segment_anything`, which is **not installed**, so in-app SAM is `UNAVAILABLE` (confirmed by `perception.availability`: "the segment_anything package is not installed"). To still get a number I put the cached torch-hub source tree (`~/.cache/torch/hub/facebookresearch_segment-anything_main`) on `sys.path` and ran `SamPredictor.set_image` + one point prompt. That is an experiment, not what the app does. SAM resizes the long side to 1024 internally, so a 512 input saves nothing. Only ViT-B weights are cached; ViT-L / ViT-H: not measured (weights not on disk).

| config | load s | first img s | steady s/img (encoder) | torch max alloc MB | torch max reserved MB | nvidia-smi peak delta MiB | peak RAM MB |
|---|---|---|---|---|---|---|---|
| cuda 1024 fp32 | 0.81 / 0.85 | 0.63 / 0.64 | 0.341 / 0.342 | 2770 / 2770 | 3096 / 3096 | 3191 / 3191 | 1543 / 1555 |
| cuda 1024 fp16 | 0.78 / 0.79 | 0.52 / 0.53 | 0.196 / 0.197 | 1759 / 1759 | 2590 / 2590 | 2691 / 2691 | 1733 / 1775 |
| cuda 512 fp32 | 0.79 / 0.80 | 0.62 / 0.63 | 0.347 / 0.347 | 2770 / 2770 | 3096 / 3096 | 3191 / 3191 | 1524 / 1550 |
| cpu 1024 fp32 | 0.71 / 0.71 | 6.14 / 6.15 | 6.17 / 6.23 | n/a | n/a | n/a | 3625 / 3625 |

### 2.3 DISK + LightGlue via kornia (repo `reconstruction/matching/lightglue.py`, `features="disk"`)

This is the matcher the API's scene gate selects (`make_pair_verifier(paths, "auto")` picks `lightglue` when kornia imports). 8 images, 7 consecutive pairs (CPU: 4 images, 3 pairs), RANSAC included in pair time. DISK is run with no keypoint cap (`kf.DISK.from_pretrained("depth")`, line 66), so about 10.6k keypoints per image at 1024.

| config | load s | feat 1st s | feat steady s/img | match steady s/pair | kpts/img | torch max alloc MB | torch max reserved MB | nvidia-smi peak delta MiB | peak RAM MB |
|---|---|---|---|---|---|---|---|---|---|
| cuda 1024 fp32 | 0.64 / 0.72 | 0.45 / 0.53 | 0.124 / 0.125 | 0.857 / 0.863 | 10600 | 3601 / 3601 | **12650 / 12650** | **5907 / 5907** | 5630 / 8341 |
| cuda 1024 fp16 | 0.74 / 0.87 | 0.57 / 0.69 | 0.097 / 0.098 | 3.02 / 3.05 | 10594 | 3595 / 3595 | 12444 / 12444 | 5911 / 5911 | 7790 / 8400 |
| cuda 512 fp32 | 0.66 / 0.71 | 0.47 / 0.49 | 0.043 / 0.044 | 0.165 / 0.175 | 2492 | 520 / 520 | 660 / 660 | 757 / 757 | 1471 / 1471 |
| cuda 512 fp16 | 0.60 / 0.62 | 0.33 / 0.33 | 0.033 / 0.039 | 0.174 / 0.175 | 2488 | 354 / 354 | 466 / 466 | 563 / 563 | 1413 / 1415 |
| cpu 1024 fp32 | 0.62 / 1.13 | 1.69 / 1.70 | 1.71 / 1.72 | 45.7 / 47.6 | 10730 | n/a | n/a | n/a | 8061 / 8095 |
| cpu 512 fp32 | 0.69 / 0.76 | 0.44 / 0.51 | 0.409 / 0.422 | 2.86 / 2.96 | 2546 | n/a | n/a | n/a | 1386 / 1445 |

At 1024 the allocator reserved 12.6 GB on a 6.1 GiB card: the whole device is consumed (`nvidia-smi` +5.9 GiB), the rest spills to host RAM (5.6-8.3 GB peak), and fp16 makes pair matching 3.5x slower (spill-bound). Practical conclusion: DISK+LightGlue at 1024 with an uncapped keypoint count is **not practical** on this GPU; at 512 it is comfortable (757 MiB).

Keypoint cap experiment (direct kornia calls, DISK `n=` argument; 6 images, 5 pairs, 3 runs each, results identical across runs):

| DISK setting | kpts/img | match s/pair (median) | torch max alloc MB | torch max reserved MB | RANSAC inliers per pair (5 pairs) |
|---|---|---|---|---|---|
| 1024, no cap (current) | 10916 | 1.12-1.36 | 3601 | 12642 | 1090, 852, 639, 251, 1338 |
| 1024, n=4096 | 4096 | 0.073-0.077 | 1905 | 2496 | 504, 371, 222, 91, 492 |
| 1024, n=2048 | 2048 | 0.028-0.030 | 1897 | 2508 | 190, 168, 87, 39, 199 |
| 512, n=2048 | 2048 | 0.038 | 516 | 666 | 312, 258, 166, 79, 272 |

The scene-identity gate needs at least 12 verified inliers per pair (`reconstruction/scene_identity.py:114`), so these caps keep a wide margin on this dataset. I did not evaluate downstream grouping or reconstruction quality with the cap (not measured).

### 2.4 ALIKED + LightGlue via kornia (`features="aliked"`)

Same harness as 2.3.

| config | load s | feat 1st s | feat steady s/img | match steady s/pair | kpts/img | torch max alloc MB | torch max reserved MB | nvidia-smi peak delta MiB | peak RAM MB |
|---|---|---|---|---|---|---|---|---|---|
| cuda 1024 fp32 | 0.68 / 0.70 | 0.59 / 0.61 | 0.065 / 0.065 | 0.095 / 0.097 | 4869 | 1276 / 1276 | 1590 / 1590 | 1687 / 1687 | 1597 / 1597 |
| cuda 1024 fp16 | 0.65 / 0.68 | 0.67 / 0.68 | 0.056 / 0.057 | 0.094 / 0.096 | 4861 | 1214 / 1214 | 1546 / 1546 | 1649 / 1649 | 1866 / 1866 |
| cuda 512 fp32 | 0.66 / 0.67 | 0.56 / 0.56 | 0.028 / 0.029 | 0.060 / 0.062 | 1122 | 361 / 361 | 444 / 444 | 541 / 541 | 1627 / 1627 |
| cuda 512 fp16 | 0.66 / 0.66 | 0.62 / 0.62 | 0.029 / 0.029 | 0.057 / 0.059 | 1119 | 346 / 346 | 430 / 430 | 533 / 533 | 1823 / 1825 |
| cpu 1024 fp32 | 0.65 / 0.68 | 1.01 / 1.03 | 1.03 / 1.04 | 7.20 / 7.22 | 4722 | n/a | n/a | n/a | 2339 / 2344 |
| cpu 512 fp32 | 0.70 / 0.71 | 0.31 / 0.32 | 0.251 / 0.257 | 0.491 / 0.497 | 1143 | n/a | n/a | n/a | 1139 / 1141 |

ALIKED+LightGlue at 1024 is about 9x faster per pair than uncapped DISK and uses 1.7 GiB instead of the whole card, with no code change other than selecting it. fp16 has no benefit.

### 2.5 Classical CPU matchers (repo `sift`, `orb`; 7 consecutive pairs; 3 runs)

| matcher | max_side | steady s/pair (min of the 3 runs' medians / median) | median inliers |
|---|---|---|---|
| SIFT (OpenCV, CPU) | 1024 | 0.166 / 0.166 | 16 |
| SIFT | 512 | 0.060 / 0.061 | 13 |
| ORB (OpenCV, CPU) | 1024 | 0.033 / 0.034 | 12 |
| ORB | 512 | 0.024 / 0.024 | 12 |

(For comparison, LightGlue-DISK on the same consecutive pairs gives 251 to 2887 inliers; SIFT/ORB give 12-16.) These need no GPU.

### 2.6 COLMAP 4.2.0 CUDA: SIFT extraction + exhaustive matching, 8 images (28 pairs)

Subprocess wall time including process start (about 0.3 s). GPU memory is `nvidia-smi`-sampled (peak delta); RAM is the process `PeakWorkingSetSize`.

| stage | max_image_size | device | wall s | nvidia-smi peak delta MiB | peak RAM MB |
|---|---|---|---|---|---|
| feature_extractor | 1024 | GPU | 0.55 / 0.61 | 703 / 703 | 199 / 200 |
| feature_extractor | 1024 | CPU | 2.23 / 2.32 | 0 | 1810 / 1810 |
| exhaustive_matcher | 1024 | GPU | 0.52 / 0.54 | 805 / 805 | 187 / 187 |
| exhaustive_matcher | 1024 | CPU | 5.99 / 6.19 | 0 | 187 / 187 |
| feature_extractor | 512 | GPU | 0.44 / 0.45 | 241 / 241 | 187 / 189 |
| feature_extractor | 512 | CPU | 0.77 / 0.79 | 0 | 495 / 496 |
| exhaustive_matcher | 512 | GPU | 0.27 / 0.28 | 21 / 55 | 178 / 178 |
| exhaustive_matcher | 512 | CPU | 1.28 / 1.30 | 0 | 97 / 98 |

GPU vs CPU at 1024: extraction about 3.8x, matching about 11x faster. Full 32-image run (n=1): extraction 1.16 s / 703 MiB, matching 4.05 s / 883 MiB, `mapper` 14.7 s (CPU, 104 MB RAM) registering **22 of 32** images (4,440 points, mean reprojection error 0.247 px), which agrees with the acceptance report's "22 of 32 registered".

### 2.7 COLMAP dense MVS on that 22-image model (GPU, `geom_consistency=true`, as `colmap_backend.py` calls it)

`patch_match_stereo` outputs were deleted between repeats (COLMAP otherwise skips existing depth maps and returns in 0.2 s, which I first mis-measured and then redid). Flags `--PatchMatchStereo.max_image_size` set to the undistorted size.

| stage | size | wall s | nvidia-smi peak delta MiB | peak RAM MB | result |
|---|---|---|---|---|---|
| image_undistorter (n=1) | 1024 | 0.43 | n/a | 134 | 22 images |
| patch_match_stereo | 1024 | 302.12 / 302.24 | 391 / 391 | 759 / 759 | |
| stereo_fusion | 1024 | 4.38 / 4.46 | 0 | 473 / 473 | 234,330 points |
| image_undistorter (n=1) | 512 | 0.53 | n/a | 141 | 22 images |
| patch_match_stereo | 512 | 136.36 / 136.44 | 163 / 163 | 351 / 352 | |
| stereo_fusion | 512 | 1.25 / 1.25 | 0 | 171 / 171 | 65,462 points |

Dense MVS is **not** VRAM-bound on this card: at most about 0.4 GiB of GPU memory, and 0.76 GB of RAM. It is time-bound (5 minutes at 1024). Cross-check against the repo's own recorded run `datasets/south_building/runs/run_20260921T164048_gpu.json`: patch_match_stereo 343.7 s, stereo_fusion 4.46 s, 246,816 points (VRAM was not recorded there). The acceptance report did not run dense MVS (gated off at 69% registration < 80%, `engine/pipeline/dense_gate.py:27`), so the dense stage was not exercised through the API in that run. Meshing / texturing: not measured.

### 2.8 Not measured or unavailable

| capability | status |
|---|---|
| MiDaS small / DPT_Hybrid (`perception/depth/midAS_backend.py`) | `UNAVAILABLE: ModuleNotFoundError: No module named 'timm'` (raised in `_load_model`; weights are cached but cannot be built) |
| Depth-Anything-V2-Small | `UNAVAILABLE: transformers not installed` (HF weights are cached) |
| OWL-ViT | `UNAVAILABLE: transformers not installed` (HF weights are cached) |
| SAM via repo backend | `UNAVAILABLE: segment_anything package not installed` (see 2.2 for the hub-source experiment) |
| SAM ViT-L / ViT-H | not measured: weights not cached |
| OneFormer, Grounding DINO | not probed (registry entries; not in `perception.availability` reporters) |
| XFeat | `UNAVAILABLE: XFeat repo not in torch.hub cache` (needs network) |
| DeDoDe | `UNAVAILABLE: weights not cached` (dedode_detector_L.pth, dedode_descriptor_B.pth) |
| LoFTR, MASt3R | not implemented in the repo (MASt3R also non-commercial) |
| AKAZE, BRISK | `UNAVAILABLE: this OpenCV build (5.0.0) has no cv2.AKAZE_create / BRISK_create` |
| OpenCV CUDA | `UNAVAILABLE: 0 CUDA devices` (pip wheel without CUDA) |
| Depth maps / metric depth, COLMAP meshing, texturing, the full pipeline end-to-end | not measured in this audit |

## 3. Scheduler and OOM behaviour

### 3a. Does the scheduler return VRAM to baseline between models?

Code: `ModelResidentScheduler` (`perception/stage_runner.py:60-140`) evicts the previous model by setting its `_current_model_ref` to None and calling `release_vram()` (gc + `torch.cuda.empty_cache()`, `engine/core/device.py:67`; re-exported by `perception/device.py`). I ran `PerceptionStageRunner.run_pipeline` with a segmentation stage (Mask R-CNN on 3 images) then an embeddings stage (DISK+LightGlue, 5 pairs). 3 runs, identical numbers each time:

| moment | nvidia-smi MiB | torch allocated MB | torch reserved MB |
|---|---|---|---|
| baseline (CUDA context only) | 87 | 0 | 2 |
| inside stage 1 (Mask R-CNN), before return | 783 | 181.7 | 686 |
| after eviction of stage 1 | 117 | 8.1 | 20 |
| inside stage 2 (DISK+LightGlue), before return | 5891 | 91 | 12,662 |
| after eviction of stage 2 | 117 | 8.1 | 20 |

Yes: when a stage function returns without keeping a reference, VRAM returns to baseline (+30 MiB of CUDA kernel/context growth). Three caveats, all observed:

1. **The scheduler can only free what the stage dropped.** `_current_model_ref` is never assigned anywhere (`stage_runner.py:70,81` are the only mentions), so it owns nothing. A stage that stores its model in a long-lived object leaves it resident: with a deliberately leaky stage `nvidia-smi` stayed at 289 MiB (181 MB allocated) after "eviction" and only returned to 117 MiB after the reference was cleared and `release_vram()` called.
2. **Its peak accounting is wrong.** `PerceptionStageReport.overall_peak_vram_mb` was **8.1 MB** in all three runs, while torch's real peaks were 3,586 MB allocated / 12,662 MB reserved. Cause: `peak = max(start_vram, end_vram)` (`stage_runner.py:136`) and `peak_task_vram = max(vram_before, vram_after)` (line 205) sample `memory_allocated()` before and after the stage, after the model was already freed.
3. **The estimates it plans with are off.** `est_vram` comes from `ModelSpec.measured_vram_mb` (`stage_runner.py:111`; values in `perception/model_registry.py`). Declared vs this audit's `nvidia-smi` peak delta: Mask R-CNN 900 vs 927 MiB (good); `sam_vit_b` 950 vs 3191 (3.4x low); `lightglue_aliked` 450 vs 1687 (3.7x low); DISK has no entry (the runner's default embeddings id `xfeat` is 350, vs 5907 for uncapped DISK).
4. **It is not wired into production.** `git grep` finds no caller of `PerceptionStageRunner`, `ModelResidentScheduler`, `model_scope` or `release_vram()` outside `perception/stage_runner.py`, `perception/device.py`, `engine/core/device.py` and tests. `engine/hardware/device_manager.py` (a second, separate budget tracker) is also not imported outside its own package, and its `release_vram` only does `gc.collect()` with no `empty_cache()`.

### 3b. What happens when another heavy GPU consumer holds VRAM?

Test: a separate process allocates and holds 4.5 GiB or 5.5 GiB (`nvidia-smi` read 4693 and 5717 MiB), then another process runs the repo's `select_device()`, the scheduler's `model_scope`, and DISK+LightGlue at 1024.

- `torch.cuda.mem_get_info()` reported **5080 MiB free in both cases and with no holder at all**. The "free VRAM" probe in `select_device` (`engine/core/device.py:54`) is blind to other processes on this WDDM setup. Consequences: `select_device()` and every `model_scope(...)` chose `cuda` even with 5.5 GiB held, and the 3 checked model ids (Mask R-CNN, MiDaS, SAM ViT-B) were all given the GPU. The "fall back to CPU when too little VRAM is free" logic never triggers here.
- No OOM happened in either direction. WDDM silently paged the idle holder out, and the matcher (reserved 9.7 GB on a 6 GiB card) completed. Warm 4-pair wall time: baseline 8.53 / 8.79 s (runs 8.53, 8.79, 9.29), with 5.5 GiB held 8.98 / 9.11 s (runs 19.75, 8.98, 9.11). So an idle holder cost nothing except a one-time 2.2x penalty on the first run (paging). I did **not** test a holder that is actively computing (compute contention), nor two simultaneous heavy jobs; not measured.
- The acceptance report's allocator line (`[W... CUDACachingAllocator.cpp:3934] memory allocation failed with OOM ... free: 0`, `docs/audits/acceptance_run/api_server_1.log:7`) is a warning-level allocator retry, not an exception; I could not reproduce it in isolation (0 occurrences in 4 runs with the GPU otherwise idle), so it presumably arose from simultaneous GPU use (API plus COLMAP/other process) at that moment. Not confirmed.

### 3c. What holds the ~5.8 GiB after an API job, and does the API release it?

The acceptance report observed an idle API at 5,790 MiB (`REALITY_ENGINE_FINAL_ACCEPTANCE_REPORT.md:66`, "likely the scene-identity matcher, not confirmed"). I reproduced it by calling exactly what the API calls: `_scene_gate` (`apps/api/jobs.py:490`; invoked on a thread of the API process at `jobs.py:701`) does `make_pair_verifier(paths, "auto")` (`jobs.py:504`) then `group_images(...)` (`jobs.py:505`), which selects `LightGlueMatcher(features="disk")` on CUDA with no keypoint cap. 8 images, 3 runs, identical:

| moment | nvidia-smi MiB | torch allocated MB | torch reserved MB |
|---|---|---|---|
| baseline | 87 | 0 | 2 |
| inside gate after grouping, matcher alive | 5903 | 93.5 | 12,376 |
| after the function returned (matcher unreferenced) | **5903** | **8.1** | **12,376** |
| after `gc.collect()` | 5903 | 8.1 | 12,376 |
| after `release_vram()` (gc + `empty_cache`) | **109** | 8.1 | 20 |

Evidence for the culprit: after the job the matcher's tensors are gone (allocated falls from 93.5 MB to 8.1 MB, 8 cached feature sets were only 35.6 MB), but the **PyTorch CUDA caching allocator inside the API process keeps 12.4 GB reserved**, so the card stays full (5,903 MiB, matching the acceptance 5,790 MiB) until `empty_cache()`. The allocator has so much cached because uncapped DISK yields about 9,000-10,900 keypoints per image and LightGlue's transformer builds large per-pair intermediates (peak 3.6 GB allocated, 12.6 GB reserved, see 2.3).

Does the API release it? **No.** `git grep` for `empty_cache` / `release_vram` finds nothing under `apps/`, and `LightGlueMatcher` has no close/unload method (`reconstruction/matching/lightglue.py`; `_feat` at line 46 caches GPU tensors per image for the matcher's lifetime, which ends with `_scene_gate`'s local scope). The memory is held for the API process lifetime, as the acceptance run saw ("0 MiB after kill"). A 32-image gate run (496 exhaustive pairs) was not repeated; 8-image behaviour already saturates the card.

## 4. Classification per capability (RTX 4050, 6 GiB, WDDM)

Labels: **comfortable** (<= 1.7 GiB `nvidia-smi` delta, fast), **sequential** (fits alone, must not share the GPU with another model), **reduced** (fits only at the stated reduced setting), **CPU-only** (no GPU path or GPU path not usable here), **not practical** (works only via host-memory spill or too slow), **UNAVAILABLE** (cannot run here; reason given).

| Capability | Class | Basis |
|---|---|---|
| COLMAP SIFT extraction + matching (GPU) | comfortable | 0.7-0.9 GiB, 8 images in about 1.1 s total |
| COLMAP mapper (sparse SfM) | CPU-only (inherently) | 14.7 s, 104 MB RAM, 22/32 registered (n=1) |
| COLMAP dense `patch_match_stereo` | comfortable on memory, slow in time | 391 MiB at 1024 (302 s), 163 MiB at 512 (136 s) |
| COLMAP `stereo_fusion` | CPU-only (inherently) | 4.4 s, 473 MB RAM |
| COLMAP meshing / texturing | not measured | |
| Mask R-CNN (GPU) | comfortable | 927 MiB, 0.10 s/img |
| Mask R-CNN as the pipeline runs it (device=cpu default) | CPU-only | 5.0 s/img (0.10 s on GPU) |
| SAM ViT-B (hub-source experiment) | sequential | 3.2 GiB fp32 / 2.7 GiB fp16, 0.34 / 0.20 s per image |
| SAM ViT-B via repo backend | UNAVAILABLE | segment_anything not installed |
| SAM ViT-L / ViT-H | not measured | weights not cached |
| DISK + LightGlue, 1024, uncapped (current API default) | **not practical** | reserves 12.6 GB on a 6.1 GiB card, +5.9 GiB, 5.6-8.3 GB host RAM, 0.86 s/pair |
| DISK + LightGlue, 512 | comfortable | 757 MiB, 0.17 s/pair |
| DISK + LightGlue, 1024 with `n=4096` cap | sequential | 2.5 GB reserved, 0.074 s/pair (cap not wired in repo; direct kornia test) |
| ALIKED + LightGlue, 1024 | comfortable | 1.7 GiB, 0.097 s/pair |
| ALIKED + LightGlue, 512 | comfortable | 541 MiB, 0.062 s/pair |
| SIFT / ORB matchers | CPU-only | 0.17 / 0.03 s per pair at 1024 |
| MiDaS small / DPT_Hybrid | UNAVAILABLE | no `timm` |
| Depth-Anything-V2, OWL-ViT | UNAVAILABLE | no `transformers` |
| XFeat | UNAVAILABLE | repo not cached (network) |
| DeDoDe | UNAVAILABLE | weights not cached |
| AKAZE, BRISK | UNAVAILABLE | OpenCV 5.0.0 build lacks them |
| OpenCV CUDA kernels | UNAVAILABLE | 0 CUDA devices |
| LoFTR, MASt3R | UNAVAILABLE | not implemented in repo |
| Any two of {SAM, DISK@1024, Mask R-CNN+...} resident together | not practical | SAM 3.2 GiB + uncapped DISK (full card) exceed 6 GiB; Mask R-CNN (0.9) + SAM (3.2) = 4.1 GiB is the only heavy pair that sums under the card, and I did not run them together (not measured) |

## 5. Recommendations (ranked; each tied to a measured number)

1. **Cap DISK keypoints (or default the scene gate to ALIKED, or to 512 px).** Uncapped DISK at 1024 reserved 12,650 MB (full card plus host spill) and 0.86 s/pair; `n=4096` gives 2,496 MB and 0.074 s/pair (14x faster, 5x less memory) and still 91-504 inliers per pair against the gate's minimum of 12 (`scene_identity.py:114`). ALIKED needs no code change beyond the selection: 1,687 MiB, 0.097 s/pair. Quality impact beyond inlier counts: not evaluated. Touch points: `reconstruction/matching/lightglue.py:66-71` (pass `n=` to the detector call) and `scene_identity.py:100-101`.
2. **Release VRAM when the scene gate finishes.** Call `release_vram()` in a `finally` at the end of `_scene_gate` (`apps/api/jobs.py:490`) and give `LightGlueMatcher` a `close()` that clears `_feat`/models. Measured effect: idle API 5,903 MiB -> 109 MiB. This single call removes the acceptance report's "idle API holds ~5.8 GiB" and the contention it creates with COLMAP dense (391 MiB) and Mask R-CNN (927 MiB).
3. **Run GPU torch work out of the API process.** The API thread pool shares one CUDA context and one caching allocator for its lifetime (the state in 3c only clears on `empty_cache` or process exit). A short-lived subprocess per torch GPU stage frees everything, including the 87-117 MiB context, when it exits, and keeps COLMAP (already a subprocess; at most 0.9 GiB sampled) isolated from torch. Not prototyped; the measured justification is the 5,903 MiB persistent hold.
4. **Wire `PerceptionStageRunner` into the pipeline and fix its accounting.** It demonstrably returns VRAM to baseline (783 -> 117 MiB, 5,891 -> 117 MiB) but nothing calls it. Make `overall_peak_vram_mb` use `torch.cuda.max_memory_allocated/reserved` (reset per stage) instead of `max(before, after)`; today it reports 8.1 MB against a true 3,586 / 12,662 MB (`stage_runner.py:136,205`). Make `_current_model_ref` real or document that the stage function must drop its model.
5. **Do not trust `mem_get_info` for the "enough free VRAM" decision on this machine.** It returned 5,080 MiB with 0, 4.5 and 5.5 GiB held by another process, so CPU fallback in `select_device` / `model_scope` never fires. Use an NVML/`nvidia-smi` reading or the scheduler's own accounting. Also consider `torch.cuda.set_per_process_memory_fraction(0.85)` so a runaway allocation raises a catchable OOM instead of spilling to host RAM (12,650 MB reserved on a 6,141 MiB card; the fraction setting itself was not tested here).
6. **Run Mask R-CNN on the GPU inside a scheduler scope.** `vertical_slice.py:964` uses the CPU default: 5.03 s/img vs 0.103 s/img on GPU (49x), for 927 MiB. For a 22-32 image set that is about 110-160 s vs 2-3 s (arithmetic from the medians).
7. **Resolution caps by workload.** Dense MVS: 1024 -> 512 halves time (302 -> 136 s) but cuts the fused cloud 3.6x (234k -> 65k points); VRAM is not the constraint (391 vs 163 MiB), so keep 1024 unless time-boxed. Mask R-CNN 512: 927 -> 557 MiB, 0.10 -> 0.06-0.09 s. SAM: do not bother lowering input (same 3,191 MiB, same time at 512); use fp16 autocast (3,191 -> 2,691 MiB, 0.34 -> 0.20 s) if SAM is enabled. fp16 autocast otherwise gives no benefit (Mask R-CNN, ALIKED) or hurts (DISK@1024, 3.5x slower).
8. **Correct the registry VRAM numbers the scheduler plans with** (`perception/model_registry.py`): `sam_vit_b` 950 -> about 3,200 MiB, `lightglue_aliked` 450 -> about 1,700 MiB, add a DISK entry (capped vs uncapped). The scheduler's CPU-fallback and budget check (`RTX4050_MAX_USABLE_VRAM_MB = 5200`, `stage_runner.py:29,111-126`) is only as good as these numbers.
9. **Environment gaps to close or declare.** `timm` (MiDaS), `transformers` (Depth-Anything-V2, OWL-ViT; their HF weights are already cached) and `segment_anything` (SAM; ViT-B weights cached) are the only things between this machine and three perception stages, so depth/segmentation/open-vocab detection currently never reach the world, as the acceptance report noted. Installing them was out of scope for this audit.

## 6. Summary (10 lines)

1. Hardware: RTX 4050 Laptop 6141 MiB under WDDM (5080 MiB usable per `mem_get_info`), i7-13620H, 15.65 GiB RAM, torch 2.14.1+cu130, COLMAP 4.2.0 CUDA; shared with other agents' tests.
2. Comfortable on GPU: COLMAP SIFT (<=0.9 GiB), Mask R-CNN (0.93 GiB, 0.10 s/img), ALIKED+LightGlue (1.7 GiB at 1024), DISK+LightGlue at 512 (0.76 GiB).
3. COLMAP dense MVS is time-bound, not VRAM-bound: 302 s and 391 MiB at 1024 (234k points), 136 s and 163 MiB at 512 (65k points); the old "~6 GB" claim was wrong.
4. Not practical as shipped: DISK+LightGlue at 1024 uncapped reserves 12.6 GB on a 6.1 GiB card (spills to 5.6-8.3 GB host RAM); `n=4096` cuts that to 2.5 GB and 14x in time.
5. The ~5.8 GiB idle hold is reproduced and explained: the API process's CUDA caching allocator keeps 12.4 GB reserved after `_scene_gate` returns (tensors already freed, allocated 8 MB); `release_vram()` takes it 5,903 -> 109 MiB; the API never calls it.
6. The one-model scheduler does return VRAM to baseline (783 -> 117 MiB) but only if the stage drops its model, reports a peak of 8.1 MB vs a true 3,586/12,662 MB, uses registry estimates 3-4x too low, and nothing in production calls it.
7. `torch.cuda.mem_get_info` stays at 5080 MiB even with 5.5 GiB held by another process, so CPU fallback never triggers; an idle co-tenant caused no OOM (one 2.2x first-run slowdown); an active competing compute job was not tested.
8. Mask R-CNN runs on CPU in the pipeline today (5.0 s/img vs 0.10 s on GPU).
9. UNAVAILABLE here: MiDaS (no timm), Depth-Anything-V2 and OWL-ViT (no transformers), repo SAM (no segment_anything), XFeat, DeDoDe, AKAZE/BRISK, OpenCV CUDA, LoFTR/MASt3R.
10. fp16 autocast helps only SAM (3.2 -> 2.7 GiB, 0.34 -> 0.20 s); other models gain nothing and uncapped DISK gets 3.5x slower.

## 7. What I could not measure, and why

- Dense MVS through the API/pipeline, COLMAP meshing and texturing, and an end-to-end multi-stage run: only COLMAP's dense CLI stages were run directly on a fresh 22-image sparse model; the full pipeline was out of scope for the time and the GPU-sharing constraints.
- Depth models (MiDaS small, DPT_Hybrid, Depth-Anything-V2), OWL-ViT, XFeat, DeDoDe, OneFormer, Grounding DINO: missing dependencies or weights, no downloads allowed.
- SAM ViT-L/H: weights not cached. SAM ViT-B was measured through the torch-hub source tree, not through the repo backend, which is blocked.
- CPU contention from other agents' tests: not quantified (shared machine); CPU timings may carry noise.
- An actively computing second GPU consumer, and two heavy jobs running simultaneously: only an idle memory holder was tested.
- A 32-image (496-pair) scene-gate run: 8 images already saturate the card, so the longer run was skipped.
- Downstream quality under the DISK keypoint cap or lower resolutions: only inlier counts were compared.
- `set_per_process_memory_fraction` as an OOM guard: proposed, not tested.
- Mapper, undistorter and the 32-image extraction/matching are `n=1`; every other row is 3 launches.
- `nvidia-smi` is sampled at about 0.1-0.2 s, so a spike shorter than that could be missed in the COLMAP rows.
