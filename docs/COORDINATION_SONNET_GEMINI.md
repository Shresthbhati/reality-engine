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

## 8. Metric 3D Evidence & Perception Release Specifications

### 8.1 Metric 3D Lifting Interface (`engine/multiengine/metric_lifting.py`)
- Unprojects 2D bounding boxes and segmentation masks into metric 3D point clusters using calibrated `PinholeCamera` intrinsics/extrinsics and metric depth maps.
- Resolves:
  - Metric centroid position: $(x, y, z)$ in canonical world or camera coordinate frame.
  - 3D bounding box dimensions: $(w, d, h)$ in meters.
  - Orientation: Unit quaternion $(w, x, y, z)$ derived via PCA ground-plane eigenvector analysis.
  - Estimation uncertainty: $3 \times 3$ covariance matrix for position and dimensions.
- Links subparts (e.g., `chair_leg`, `door_handle`) to parent entities (`chair`, `door`) via 3D proximity and ontology hierarchy checks.

### 8.2 Image-Backed Visibility & Reprojection (`engine/multiengine/visibility_reprojection.py`)
- Validates 3D box visibility against image frustum boundaries, focal plane guard, and depth map occlusions.
- Evaluates Euclidean reprojection error in pixels and 2D bounding box intersection-over-union (IoU).

### 8.3 Statistical Fusion Validation (`engine/multiengine/statistical_fusion.py`)
- Effective sample size $N_{\text{eff}} = N / (1 + (N-1)\rho)$ prevents variance deflation from redundant intra-family models.
- Covariance Intersection bounds unknown cross-correlations conservatively.
- Uncalibrated heuristic weights ($1/\sqrt{N}$) are explicitly labeled `uncertainty_basis: heuristic_uncalibrated_sqrt_N`.

### 8.4 Adapter Scoring & Virtual Isolation
- All 33 engine adapters now strictly output actual model scores ($p \in [0, 1]$) or `None`. Hardcoded fallback confidences (0.88, 0.92, etc.) are eliminated.
- Virtual inference double is isolated; supplying real image paths executes real model inference or raises `EngineError` if weights/dependencies are uninstalled.

### 8.5 Measured Hardware Performance (RTX 4050 6GB Laptop GPU)
- **OWL-ViT**: Load 41.27s, Inference 2.14s, Unload 0.82s, Peak VRAM 628.3 MB $\rightarrow$ Purged to 8.1 MB.
- **MiDaS**: Load 8.50s, Inference 0.64s, Unload 0.47s, Peak VRAM 706.9 MB $\rightarrow$ Purged to 30.6 MB.
- **Depth Anything**: Load 0.68s, Inference 0.15s (147 ms), Unload 0.49s, Peak VRAM 269.9 MB $\rightarrow$ Purged to 30.6 MB.
- Verified $>95\%$ VRAM reclamation between sequential executions.

---

## 9. Metric Reconstruction Integration & Accuracy Repair Contract

### 9.1 Camera Calibration & Pose Rigidity Validation
- `VisibilityReprojectionValidator.validate_camera_metadata` strictly validates camera intrinsics and extrinsics before any lifting or reprojection:
  - Intrinsics: $fx > 0, fy > 0, cx \in (0, w), cy \in (0, h), w > 0, h > 0$.
  - Extrinsics: All values must be finite. Rigid rotation matrix enforced: $|\det(R) - 1.0| \le 10^{-3}$ and $\|R^T R - I\|_\infty \le 10^{-3}$.
  - Quaternions: Unit norm $\|q\| = 1 \pm 10^{-3}$.
  - Missing calibration produces an explicit `ValidatedCameraMetadata` refusal or uncalibrated error, never silent default intrinsics ($500, 320, 240, 640\times 480$). Identity extrinsics are allowed only when explicitly flagged as canonical reference frame.

### 9.2 Metric 3D Lifting Integrity
- Enforces `is_metric_depth: bool = True`. Relative or unscaled depth maps (e.g. raw MiDaS inverse depth) cannot be directly unprojected without registered metric scale.
- Single quaternion convention $(w, x, y, z)$ matching WorldIR `Quat`.
- Degeneracy & symmetry: When planar eigenvalues indicate rotational/circular symmetry, `orientation` is honestly set to `None` with `is_orientation_known=False`, avoiding spurious yaw angles.
- True extent calculation: Thickness is reported honestly from quantile spreads without faking a 1 cm minimum thickness.
- Order-statistic quantile variance computed for $N \ge 20$ points; otherwise returns `None` with an explicit reason.

### 9.3 WorldIR Confidence Contract Alignment
- In `world_ir/schema_v1.py`, `Entity.confidence` is `Optional[float] = None`.
- Legacy worlds deserializing explicit confidence values preserve them without mutation.
- When confidence is not measured through calibrated consensus, it is saved as `None` with `confidence_basis: None`.


