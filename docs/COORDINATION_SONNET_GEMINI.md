# Reality Engine — Architecture & Execution Coordination: Gemini & Sonnet

**Date:** 2026-11-19  
**Authors:** Gemini (Spatial Perception, Geometry, Reconstruction & Fusion Lead) & Sonnet (Platform Architecture, Consensus & Contracts)

---

## 1. Executive Summary

This document establishes the binding architectural contract and integration interface between the Spatial Perception / Universal Fusion layer (owned by Gemini) and the Platform Architecture / Consensus / Evidence Store layer (owned by Sonnet).

Both agents adhere to the **Mandatory Sequential Maximum-Fidelity Execution Policy**:
- Original source evidence is preserved and passed to specialist models.
- No fabricated geometry, synthetic confidence, or invented fallback classes are permitted.
- Honest execution statuses (`COMPLETE`, `PARTIAL`, `BLOCKED`, `FAILED`) are surfaced across APIs and benchmarks.

---

## 2. WorldIR Confidence Semantics & Contract

### 2.1 The Issue
Previously, WorldIR `Entity.confidence` defaulted to `0.5`, which violated the core rule against invented/synthetic confidence values.

### 2.2 Resolution & Shared Decision: Option (B)
- In WorldIR 2.0, `Entity.confidence` is typed as `float | None = None`.
- If an entity's confidence has not been computed through calibrated, measured consensus or correlation discounting, it remains `None`.
- When set, a companion `confidence_basis: str | None` must be populated, referencing the specific calibration methodology (e.g., `"correlation_discounted_pooling:N=3"`, `"bayesian_reprojection_calibrated:0.42px"`).
- Synthetic or arbitrary fallback confidences (such as 0.5 or 0.92) are strictly prohibited.

---

## 3. Canonical Entity & Fusion Contracts

### 3.1 Geometry Invariants
1. **Unobserved 3D Geometry**: If multi-view triangulation or depth unprojection does not observe a finite 3D bounding box, `bounding_box_3d` remains `None`. Universal Fusion never fabricates bounding boxes from empty space or arbitrary defaults.
2. **Reprojection Error**: Measured strictly via pinhole camera projection using verified camera intrinsics and extrinsics. Uncalibrated cameras yield `reprojection_error_px = None`. Hardcoded constants (such as 0.85 px) are eliminated.
3. **Semantic Classification**: Semantic resolution maps ontology synonyms and parts without fallback guessing. Unidentified entities are preserved as `"unidentified"` and flagged for active capture or additional specialist passes; they are never defaulted to `"chair"`.

### 3.2 Correlation Discounting & Uncertainty Pooling
Observations from models within the same model family (e.g., ViT-backbone detectors, DPT depth estimators) share inductive biases. As implemented in `UniversalFusionPipeline` (Stages 8 & 10):
$$\omega_i = \frac{1}{\sqrt{N_{\text{family}(i)}}}$$
Confidence pooling applies this discounting directly and penalizes detected semantic and geometric conflicts.

---

## 4. Evidence Matrix & Artifact Compatibility

### 4.1 Storage & Verification
- `EvidenceMatrixStore` and `EvidenceStore` share immutable, append-only guarantees.
- Large binary payloads (point clouds, depth maps, segment masks) are offloaded to filesystem artifacts under `evidence_artifacts/` with content-addressed SHA-256 hashes.
- Ingestion verifies SHA-256 integrity before reading. Any hash mismatch raises `EvidenceTamperedError` immediately.

### 4.2 Cross-View Association Ledger
- `association.py` produces an explicit, queryable decision ledger for every observation pair.
- Ontology part-whole and label-conflict rules prevent impossible merges (e.g., merging a chair leg into an entire desk).
- Unpaired detections remain standalone entities with unique identifiers.

---

## 5. Active Capture Requests & Refinement

- Capture requests (`engine/multiengine/capture_requests.py`) are triggered exclusively by:
  1. Entities with unobserved 3D geometry (`bounding_box_3d is None`).
  2. Severe geometric conflict (bounding box overlap without support, or center distance $> 0.5$ m for same entity).
  3. Severe semantic conflict (contradictory class classifications across views).
  4. Camera pose uncertainty exceeding tolerance.
- Requests include camera pose suggestions, required field-of-view, and targeting metadata to resolve ambiguities incrementally.

---

## 6. Operational Status & Catalogue Alignment

The 125-slot Target Model Catalogue (`engine/multiengine/catalogue.py`) is audited honestly:
- **Total Slots:** 125
- **Planned Slots:** 94
- **Installed Architectures:** 26
- **Available Local Checkpoints:** 6 (`colmap`, `depth_anything`, `mask_rcnn`, `midas`, `owlvit`, `sam`)
- **Verified Adapters:** 21
- **Production Wired Engines:** 4 (`colmap`, `depth_anything`, `midas`, `mask_rcnn`)
- **Blocked Engines:** 5 (environment, CUDA, or runtime dependency restrictions)
- **Licence-Restricted Engines:** 15 (non-commercial licenses like CC-BY-NC-4.0)
- **Real-Data Validated Engines:** 0 (awaiting physical measured-room captures)

---

## 7. Next Implementation Steps for Integration

1. Wire `UniversalFusionPipeline` and `canonical.py` directly into the production job runner (`apps/api/jobs.py`).
2. Connect `routes_engines.py` endpoints to live job state.
3. Ingest real-world photo datasets into `benchmarks/measured_rooms/` when physical photography is completed, replacing the `BLOCKED` status with measured physical metrics.
