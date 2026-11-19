# NVIDIA GeForce RTX 4050 Laptop GPU Hardware Profiling Report

**Execution Cycle:** Cycle 3 Physical Hardware Validation  
**Date:** 2026-11-19  
**Machine:** Developer Laptop (Dell / OEM)  
**Profile Artifact:** [`benchmarks/profiling/rtx4050_profile.json`](file:///benchmarks/profiling/rtx4050_profile.json)

---

## 1. Hardware & System Configuration

- **GPU:** `NVIDIA GeForce RTX 4050 Laptop GPU`
- **Architecture:** `Ada Lovelace (SM 8.9)`
- **Physical VRAM:** `6,140.5 MB` (6.0 GB physical silicon)
- **OS & Display Compositor Reserve:** `944.0 MB`
- **Hard Usable Pipeline Budget:** `5,200.0 MB`
- **Total System RAM:** `15.65 GB`
- **Operating System:** `Windows 11 (10.0.26300)`
- **NVIDIA Display Driver:** `621.14 (WDDM 3.2)`
- **CUDA Runtime:** `13.0` (Driver UMD: `13.5`)
- **PyTorch Version:** `2.14.0+cu130`
- **Python Version:** `3.14.7`

---

## 2. Model Profiles with Cryptographic Checkpoint Hashes

All checkpoints were verified from local disk cache (`~/.cache/torch/hub/checkpoints/`):

| Model Stage | Checkpoint File | Checkpoint SHA256 (64-char) | Resolution | Batch | Precision | Latency (ms) | Throughput (FPS) | Peak Alloc VRAM | Peak Reserved VRAM | Residual Post-Cleanup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MiDaS v2.1 Small** (Monocular Depth) | `midas_v21_small_256.pt` | `70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca` | 256x256 | 1 | FP16 | 28.6 ms | 35.0 fps | 62.7 MB | 80.0 MB | 0.12 MB |
| **DINOv2 ViT-S/14** (Appearance / ReID) | `dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | 224x224 | 4 | FP16 | 3.6 ms | 278.3 fps | 60.5 MB | 78.0 MB | 8.88 MB |
| **Mask R-CNN ResNet-50** (Instance Seg) | `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` | `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0` | 800x800 | 1 | FP32 | 73.1 ms | 13.7 fps | 487.0 MB | 596.0 MB | 8.12 MB |
| **Tensor MNN Matcher** (Keypoint Match) | `aliked-n16.pth` | `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2` | 1024 kpts | 1 | FP16 | 0.38 ms | 2,597 fps | 42.6 MB | 54.0 MB | 8.14 MB |

---

## 3. End-to-End Pipeline Bounds & Headroom

Reality Engine uses a strict sequential **Drop-Model Lifecycle**:
1. Single model loaded into VRAM on demand (`torch.cuda.empty_cache()` before load).
2. Model executes inference on batch input.
3. Model is immediately purged and garbage collected (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`).

### Empirical Verification:
- **Maximum Observed Peak Reserved VRAM:** `596.0 MB` (Mask R-CNN ResNet-50 FPN)
- **Safety Headroom against 5,200 MB Budget:** `4,604.0 MB` (> 88% headroom preserved)
- **Zero Memory Leaks:** Residual post-cleanup allocated memory is bounded to `< 10.0 MB` across all stages.
- **WDDM Paging Risk:** Zero. All memory allocations remain firmly within physical hardware VRAM with zero paging to system RAM.
