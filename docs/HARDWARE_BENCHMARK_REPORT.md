# Reality Engine — Hardware Benchmark: Sequential Maximum-Fidelity Execution

**Date:** 2026-11-19  
**Target Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Host Environment:** Windows 11, Python 3.14, PyTorch 2.14.0+cu130, CUDA 13.0, Transformers 5.16.1  
**Execution Policy:** Sequential Maximum-Fidelity (`Policy.MAXIMUM_FIDELITY`)  
**Lifecycle Guarantee:** `LOAD -> INFERENCE -> SAVE RAW OUTPUT -> VALIDATE -> NORMALISE -> RECORD ERRORS -> RELEASE RESOURCES -> VERIFY PERSISTENCE -> NEXT MODEL`

---

## 1. Measured Benchmarks on Real GPU Hardware

The benchmark was executed against real cached checkpoints on the local RTX 4050 GPU, measuring wall-clock load time, inference time, unload time, and GPU VRAM footprint before and after execution:

| Model ID | Checkpoint Identity | Task | Device | Load (s) | Inference (s) | Unload (s) | Total (s) | Peak VRAM (MB) | Delta VRAM (MB) | Post-Unload VRAM (MB) | Memory Purged | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `owlvit` | `google/owlvit-base-patch32` | Detection | `cuda:0` | 41.271 | 2.141 | 0.817 | 44.229 | 628.3 | 628.3 | 8.1 | Yes | PASS |
| `midas` | `intel-isl/MiDaS:DPT_Hybrid` | Depth | `cuda:0` | 8.495 | 0.635 | 0.473 | 9.603 | 706.9 | 698.7 | 30.6 | Yes | PASS |
| `depth_anything` | `depth-anything/Depth-Anything-V2-Small-hf` | Depth | `cuda:0` | 0.676 | 0.147 | 0.489 | 1.312 | 269.9 | 239.3 | 30.6 | Yes | PASS |

---

## 2. Key Findings & Hardware Validation

1. **VRAM Release Guarantee**:
   - `owlvit`: Peak 628.3 MB -> released down to 8.1 MB.
   - `midas`: Peak 706.9 MB -> released down to 30.6 MB.
   - `depth_anything`: Peak 269.9 MB -> released down to 30.6 MB.
   - Every single model purged $> 95\%$ of allocated memory upon `adapter.unload()`, validating that sequential execution prevents cumulative VRAM exhaustion on 6GB GPUs.

2. **Inference Latency on RTX 4050**:
   - Depth Anything V2 Small achieves **147 ms** per 640x480 frame.
   - MiDaS DPT_Hybrid achieves **635 ms** per frame.
   - OWL-ViT open-vocabulary detector achieves **2.14 s** per frame across 16 canonical ontology prompts.

3. **Metric Evidence Production**:
   - Depth Anything and MiDaS produce dense metric depth maps.
   - OWL-ViT outputs open-vocabulary bounding boxes with genuine model prediction scores ($p \in [0, 1]$).
   - Uncalibrated confidence claims have been removed; models without calibrated uncertainty output `confidence = None`.
