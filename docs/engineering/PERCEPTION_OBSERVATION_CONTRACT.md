# Perception and Multi-View Observation Contract

## 1. Overview and Architecture

Reality Engine's perception subsystem transforms 2D images and video sequences into rich, calibrated, multi-view 3D observations. It guarantees scale honesty, geometric integrity, and explicit uncertainty quantification across all perception stages.

```
+-----------------------------------------------------------------------------------------+
|                                1. VIDEO INGESTION & SAMPLING                            |
| Selectable strategies: UNIFORM | KEYFRAME_SHARPNESS | SHOT_BOUNDARY | ADAPTIVE         |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                2. UNIVERSAL 2D PERCEPTION                               |
| - Domain & Scale-Aware Prompts (Macro / Standard / Micro)                              |
| - Whole-Image Open-Vocabulary Detection                                                 |
| - Overlapping Tiled Inference (for small & distant fittings)                            |
| - Host-Crop Subpart Re-Detection (sockets, switches, door handles, drawers)             |
| - Class-Specific Confidence Threshold Gating                                            |
| - Subpart-Preserving Duplicate Suppression (WBF)                                        |
| - Vectorized Contour Boundary & Instance Mask Refinement (scipy.ndimage)                |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                             3. TEMPORAL IDENTITY TRACKING                               |
| - 2D Velocity Prediction (vx, vy) with EMA Smoothing under Camera Motion                |
| - Appearance Embedding Cosine Similarity Distance                                       |
| - Frame-to-Frame Track State Coasting & Track Termination                               |
| - Explicit Non-Physical Track Provenance (is_physical_identity=False)                   |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                        4. GEOMETRIC PROJECTION & SPATIAL LIFTING                        |
| - Optical Center & Pinhole Camera Ray Construction                                      |
| - Depth Alignment & Monocular Unprojection                                              |
| - Covariance & Uncertainty Propagation                                                  |
| - Unit-Aware Scale Contract Enforcement                                                 |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         5. MULTI-VIEW RECONSTRUCTION INTELLIGENCE                       |
| - Epipolar Ray Triangulation & Robust Closest Point of Approach (CPA)                   |
| - Huber-Loss Triangulation Residual Minimization                                        |
| - Cluster-Wide Mutual Exclusion (zero same-frame / same-image collisions)               |
| - Transitive Scale, Frame, and Anchor Consistency Verification                          |
| - Competing Identity Hypothesis Ledger                                                  |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         6. UNIFIED PRODUCTION OBSERVATION CONTRACT                      |
| -> Canonical WorldIR v1 Entities & Relationships                                        |
| -> Compliant metric3d.v1 Observations                                                   |
| -> Execution Telemetry & Model Provenance Fact Sheets                                   |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Scale Integrity and Unit Contract

Reality Engine strictly distinguishes between five scale regimes. No unit coercion or default metric assumption is permitted:

| Scale State | Coordinate Frame | Units | Physical Anchor Required? | Permitted Operations |
|---|---|---|---|---|
| `METRIC_ANCHORED` | `world:canonical` | `m` | **Yes** (`anchor://` or `artifact://`) | Physical dimension claims, millimeter CAD comparison |
| `SYNTHETIC_SCALE_ALIGNED` | `world:synthetic` | `synthetic_m` | **No** (forbidden) | Synthetic scene rendering, unit-consistent simulation |
| `SFM_GAUGE` | `sfm_world` | `sfm_units` | **No** (forbidden) | Epipolar geometry, relative structure, bundle adjustment |
| `RELATIVE_DEPTH` | `camera:local` | `relative` | **No** (forbidden) | Local depth ranking, front-to-back ordering |
| `UNKNOWN` | `frame:unspecified` | `unknown` | **No** (forbidden) | Qualitative tagging only, no spatial comparison |

### Critical Scale Honesty Invariants:
1. `is_physically_metric` MUST be `True` if and only if `scale_state == "metric_anchored"` AND `physical_anchor_ref` is verified.
2. In `PhysicalMergedEntity` and `Metric3DObservation`, synthetic or gauge coordinates are serialized to `position_synthetic` or `position_gauge`. `position_m` and `dimensions_m` are populated **strictly** when physically anchored.
3. Observations from incompatible scale states (e.g. `sfm_gauge` with `metric_anchored`) are refused during clustering and recorded in the competing hypothesis ledger.

---

## 3. Data Contracts and Interfaces

### 3.1 DetectionInput & DetectionBox
Standardized 2D observation format:
```python
@dataclass
class DetectionBox:
    box_id: str                              # Unique detection instance ID
    class_name: str                          # Canonical taxonomy class name
    box_2d: tuple[float, float, float, float]# (x1, y1, x2, y2) in image pixels
    confidence: float                        # Model confidence [0.0, 1.0]
    source_engine: str                       # e.g. "owlvit", "grounding_dino", "sam"
    source_image_id: str                     # Source image URI or filename
    mask_2d: np.ndarray | None = None        # Binary segmentation mask (H, W)
    parent_box_id: str | None = None         # ID of host object for subparts
    child_box_ids: tuple[str, ...] = ()      # IDs of subparts hosted inside this box
    scale: DetectionScale = STANDARD         # MACRO, STANDARD, or MICRO
    mask_polygon: Sequence[tuple[float, float]] | None = None # Simplified polygon contour
    mask_area_px: int = 0                    # Pixel area of instance mask
    is_truncated: bool = False               # Touches image boundary
    occlusion_level: str = "none"            # "none", "partial", "heavy"
    frame_index: int | None = None           # Sequential frame index (video)
    timestamp_sec: float | None = None       # Video timestamp
    shot_id: int | None = None               # Video shot ID
    attributes: dict[str, Any] = ...         # Material, finish, color, state
```

### 3.2 CrossViewRayCandidate
Calibrated 3D observation candidate:
```python
@dataclass(frozen=True)
class CrossViewRayCandidate:
    observation_id: str                      # Unique observation ID
    source_image_id: str                     # Source photo ID
    class_name: str                          # Taxonomy class
    origin: np.ndarray                       # Camera center C in world frame (3,)
    direction: np.ndarray                    # Normalized unit ray direction (3,)
    box_2d: tuple[float, float, float, float]# 2D bounding box
    camera: PinholeCamera                    # Intrinsics and extrinsics
    lifted_entity: SpatialEntity3D | None    # Unprojected 3D entity if available
    appearance_features: tuple[float, ...]   # Normalized DINOv2 appearance embedding
    scale_state: str                         # METRIC_ANCHORED, SFM_GAUGE, etc.
    scale_units: str                         # "m", "sfm_units", etc.
    is_physically_metric: bool               # Physical metric validity flag
    coordinate_frame: str                    # Coordinate frame identifier
    physical_anchor_ref: str | None          # Anchor URI if physically anchored
    track_id: str | None                     # Upstream temporal track ID
```

### 3.3 PhysicalMergedEntity
Resolved physical entity produced by multi-view association:
```python
@dataclass(frozen=True)
class PhysicalMergedEntity:
    canonical_entity_id: str                 # Unique physical entity UUID
    primary_class: str                       # Consensus taxonomy class
    member_observation_ids: tuple[str, ...]  # Observations merged into this entity
    sources: tuple[str, ...]                 # Distinct camera views observing entity
    position: tuple[float, float, float]     # Triangulated 3D centroid
    dimensions: tuple[float, float, float]   # 3D bounding extents
    dimension_status: str                    # "measured", "prior_bounded", "unknown"
    confidence: float                        # Multi-view consensus score
    reprojection_residual_px: float          # Mean reprojection error across views
    triangulation_residual: float            # CPA ray distance residual
    scale_state: str                         # Preserved scale contract
    scale_units: str                         # Associated unit string
    is_physically_metric: bool               # Metric truth flag
    coordinate_frame: str                    # Reference coordinate frame
    competing_hypotheses: tuple[dict, ...]   # Recorded alternative merges / splits
```

---

## 4. Multi-View Reconstruction Consumer API

Reconstruction consumers access perception observations through `SpatialSceneResult`:

```python
from engine.multiengine.spatial_intelligence import (
    reconstruct_spatial_scene,
    DetectionInput,
)

# 1. Provide 2D detections from detection engine
detections = [
    DetectionInput(
        detection_id="det_001",
        source_image_id="IMG_0000.jpg",
        class_name="door",
        box_2d=[1564.0, 907.0, 2168.0, 1814.0],
        confidence=0.91,
    ),
    # ... additional detections across views
]

# 2. Provide camera calibrations
cameras = {
    "IMG_0000.jpg": camera_0,
    "IMG_0001.jpg": camera_1,
}

# 3. Execute spatial scene intelligence
scene_result: SpatialSceneResult = reconstruct_spatial_scene(
    detections=detections,
    cameras=cameras,
    depth_maps=optional_depth_maps,
    scale_state="sfm_gauge",
    scale_units="sfm_units",
    coordinate_frame="sfm_world",
)

# 4. Ingest into WorldIR v1
world_ir = scene_result.to_world_ir()

# 5. Ingest into metric3d.v1 observations
metric3d_obs = scene_result.to_metric3d_observations(
    raw_artifact_ref="artifact://job_123/perception_run.json"
)
```

---

## 5. Verification & Telemetry Contract

The perception benchmark verifies and reports six quantitative pillars:
1. **Detection Accuracy:** Macro & micro Precision, Recall, and F1 at IoU 0.50 and 0.75.
2. **Part Hierarchy Accuracy:** Parent-child link precision, recall, and F1.
3. **Segmentation Mask Quality:** Mean Mask IoU, Median Mask IoU, and Boundary IoU.
4. **Temporal Tracking:** IDF1, MOTA, ID switches, and track fragmentations.
5. **Multi-View Association:** Pairwise association precision, recall, and reprojection residuals (mean, median, P90).
6. **Hardware Telemetry:** Total execution latency, peak host process RSS memory (MB), and peak CUDA VRAM memory (MB).
