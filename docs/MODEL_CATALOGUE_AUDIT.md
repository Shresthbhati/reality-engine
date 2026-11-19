# 125-Slot Model Catalogue Audit Report

**Date:** 2026-11-19  
**Module:** `engine/multiengine/catalogue.py`  
**Status:** Audit & Dynamic Probe Complete

---

## 1. Executive Summary

Reality Engine catalogues 125 prospective integration slots spanning 10 computer vision and geometry domains:
1. Geometry and 3D Reconstruction: 15 slots
2. Depth and Metric Geometry: 15 slots
3. Detection and Recognition: 20 slots
4. Segmentation: 15 slots
5. Feature Matching and Tracking: 12 slots
6. Point-Cloud Understanding and 3D Detection: 12 slots
7. Room and Architectural Understanding: 12 slots
8. Material, Surface Normal and Lighting: 8 slots
9. Mesh and Topology: 8 slots
10. Image Enhancement and Restoration: 8 slots

The 125-slot catalogue is an aspirational roadmap and architecture registry, **not** a claim of 125 fully operational models running in the repository.

---

## 2. Dynamic Checkpoint Probing

### 2.1 Elimination of Hardcoded Checkpoint Counts
- **Defect:** `ModelCatalogue.audit()` previously returned a static `"available_checkpoints": 6` dictionary field regardless of actual host environment state.
- **Remediation:** Implemented `probe_local_checkpoints()` with LRU caching (`@functools.lru_cache(maxsize=1)`).
- **Probing Methodology:**
  - Scans `Path.home() / ".cache" / "torch" / "hub" / "checkpoints"` for Torch weights (e.g. `sam_vit_b_01ec64.pth`).
  - Scans `Path.home() / ".cache" / "huggingface" / "hub"` for HuggingFace models (e.g. `models--depth-anything--Depth-Anything-V2-Small-hf`, `models--google--owlvit-base-patch32`).
  - Summarizes actual cache counts dynamically and attaches `checkpoint_probe_details` to the audit report.

### 2.2 Current Audit Summary (Dynamic Probe)
- **Total Slots:** 125
- **Planned Slots:** 94
- **Implemented Model Classes / Adapters:** 26
- **Verified Adapters:** 21
- **Production Wired Engines:** 4 (`colmap`, `depth_anything`, `midas`, `mask_rcnn`)
- **Commercial Restrictions:** 15 non-commercial licences flagged (CC-BY-NC-4.0 for DUSt3R/MASt3R, etc.)
- **Dynamic Checkpoint Count:** Probed from host filesystem cache.

---

## 3. Verification Suite

Verified by `tests/test_model_catalogue_125.py`:
- `test_catalogue_slot_count_is_125`: Verifies exactly 125 catalogued slots across 10 domains.
- `test_catalogue_domain_distribution`: Verifies expected slot count per domain.
- `test_catalogue_audit_summary`: Validates dynamic probe output structure.
- `test_dynamic_checkpoint_probe`: Tests cache directory resolution and probe mechanics.
- `test_commercial_licence_flags`: Ensures non-commercial licenses are explicitly guarded.
