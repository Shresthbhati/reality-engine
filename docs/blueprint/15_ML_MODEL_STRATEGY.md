# 15 — ML model strategy

## Rule zero

**ML proposes, geometry verifies (P3).** A model returns Observations. It never
writes entities, surfaces or measurements.

## The model registry (one place to add or swap models)

Every model is a backend behind a task interface plus an entry in the registry
(`perception/model_registry.py`; copy the licence-gate pattern from
`reconstruction/matching/` on the integration branch).

Registry entry:
```
task: detection | segmentation | depth | normals | features | matching | embedding | material
name, version, source_url, code_license, weights_license, commercial_ok: bool
vram_mb (measured), cpu_fallback: bool, input_size, output_contract
benchmark: {dataset, metric, value, runtime_ms, measured_at}
status: candidate | benchmarked | default | rejected (+reason)
```

A model becomes `default` only when (a) `commercial_ok` is true, (b) it beats
the current default on our benchmark ([18](18_VALIDATION_AND_BENCHMARKS.md))
with the numbers recorded, and (c) it fits the GPU budget.

## Candidates per task (benchmark, don't adopt blindly)

| Task | Current in repo | Candidates to benchmark | Licence notes (verify before use) |
|------|-----------------|-------------------------|-----------------------------------|
| Local features + matching | SIFT (OpenCV) | ALIKED + LightGlue, DISK + LightGlue, LoFTR / Efficient LoFTR, RoMa | SuperPoint/SuperGlue weights are non-commercial → benchmark only |
| SfM / MVS | COLMAP | GLOMAP (global SfM, same ecosystem); keep COLMAP MVS | COLMAP is BSD; check GLOMAP |
| Instance segmentation | Mask R-CNN (torchvision), SAM | SAM 2, Mask2Former / OneFormer | SAM/SAM 2 Apache-2.0 |
| Open-vocabulary detection | — | Grounding DINO, OWLv2, YOLO-World | Ultralytics YOLO is AGPL → never the default |
| Depth | MiDaS (relative) | Depth Anything V2 (metric variants), Metric3D v2, UniDepth, Depth Pro | DA-V2 licence differs per model size — check each |
| Normals | derived from depth | Metric3D v2 normals, Omnidata | verify |
| Appearance embeddings (re-ID helper) | colour histogram (`instances/appearance.py`) | DINOv2 (small/base) | Apache-2.0 |
| Point-cloud completion | — | PoinTr, SnowflakeNet | Phase 10 only; output is PRIOR_GENERATED |

Record every result in `docs/TECHNOLOGY_REGISTRY.md` and `.agent/LICENSES.yaml`.

## GPU budget (RTX 4050, 6 GB — re-check on the real machine)

- Verify CUDA first: `python -c "import torch; print(torch.cuda.is_available())"`.
  The last audit found CPU-only torch. Install the CUDA wheel **inside the
  project venv**, never globally.
- Device selection is centralised (`perception/device.py`, integration branch):
  `REALITY_DEVICE` override, 6 GB budget.
- **One heavy model resident at a time**: the scheduler loads a model → runs it
  on all images → unloads it, task by task (S7 order: segmentation → depth → embeddings).
- FP16 where supported; base/small variants first; CPU fallback for anything
  that won't fit (slow is fine, wrong is not).
- Record peak VRAM and per-image runtime in every benchmark row.

## Multi-scale processing (P11)

1. Whole image at reduced size: quality, scene type, big objects, architecture.
2. High-res crops only where needed: small objects (sockets, switches, handles)
   inside wall/furniture regions already found.
3. Never claim a fine detail when the crop has too few pixels: detector output
   carries `pixel_size`, and observations under the class's `min_pixels`
   ([17](17_ONTOLOGY.md)) are dropped or marked low-confidence (P10).

## Training (not in v1)

v1 integrates pretrained models only. When fine-tuning starts: split by
**property/room, never by photo**; record dataset licence and consent;
measure per region and per category, never one overall number.
