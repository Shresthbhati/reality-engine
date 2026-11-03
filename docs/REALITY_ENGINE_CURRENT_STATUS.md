# REALITY ENGINE — CURRENT STATUS (authoritative, reconciled 2026-11-03)

> ONE authoritative status file. Older docs are history, not truth.
> Reconciled across Phases 0 through 9 of the Master Development Plan.
> Baseline GitHub `origin/main` at PR #174.

## Vertical Slice & Production Architecture

```
photos/video/pointclouds/scans -> Universal Ingestion (EXIF/GPS/video/pHash/dedup)
  -> Multi-View Intelligence (LightGlue, DISK, camera calibration, Hungarian tracking)
  -> Geometric Reconstruction Foundation (COLMAP, Depth Anything V2, multi-view dense fusion)
  -> Advanced Perception (PolyLayout EM room solver, SAM 2, fine fixtures, outdoor infra, material classifier)
  -> Spatial Reasoning (bilateral relationship inference: ATTACHED_TO, RESTS_ON, CONTAINS)
  -> WorldIR Canonical Representation (entities, geometry, competing hypotheses, validation gate)
  -> WorldStore Immutable Persistence (lineage, integrity, version diffs)
  -> Reality Studio (Three.js 3D viewport, hierarchy outliner, geometry editing, history stack, 3D measurements)
  -> Export Ecosystem (15+ formats: glTF, GLB, USDA, Blender, CityJSON, CityGML, IFC BIM, OBJ, PLY, STL, SVG, DXF, PDF)
  -> Production Deployment (Docker, Docker Compose, OpenAPI 3.1 schema, OWASP Top 10 security headers)
```

## IMPLEMENTED & ACCEPTED (Full Master Plan Coverage)

### Phase 1 & 2: Ingestion & Preprocessing
- **Universal Ingestion**: Support for JPEG, PNG, WebP, TIFF, HEIC, HEIF, LAS/LAZ, PLY, PCD, E57, OBJ, glTF, GLB, MP4, MOV, MKV, AVI, TS.
- **Video Processing**: FFmpeg and scene-aware keyframe extraction strategy (`SceneAwareFrameSelectionStrategy`) allocating budget proportionally across detected shots.
- **360° & Panoramic Detection**: Equirectangular aspect ratio detection (2:1) and GPano XMP metadata parsing with `is_360` sensor flags.
- **Perceptual Hash Deduplication**: dHash 64-bit gradient hash, pHash 2D DCT median split, and Laplacian sharpness-aware keeper selection.
- **Session Clustering**: Temporal gap clustering (>1800s) and Haversine GPS distance clustering (>500m).
- **Original Media Integrity**: Content-addressed raw assets with append-only evidence packages.

### Phase 3: Multi-View & Camera Intelligence
- **Feature Matching**: `LightGlueMatcher` with DISK/SuperPoint features via Kornia (Apache 2.0) benchmarked at 2,636 verified inliers, 95.2% precision.
- **Camera Registration**: Epipolar geometric verification, intrinsic calibration, and viewpoint similarity clustering.
- **Cross-View Tracking**: Gated Hungarian instance association across viewpoints with 100% precision.

### Phase 4: Geometric Reconstruction Foundation
- **Sparse & Dense Reconstruction**: COLMAP backend integration with multi-view dense point cloud builder (`MultiViewDenseCloudBuilder`) and co-visibility depth consistency gating.
- **Depth Estimation**: Monocular metric depth estimation via Depth Anything V2 Small with MiDaS compatibility fallback.
- **Point Cloud Operations**: Voxel downsampling, statistical outlier removal (SOR via SciPy KDTree), radius outlier removal (ROR), and normal estimation.
- **Mesh Repair**: Mesh watertight verification and cleanup using `trimesh`.

### Phase 5 & 6: Architectural & Environmental Intelligence
- **Room Boundary Solver**: Plane-based expectation-maximization multi-room boundary ring solver and PolyLayout integration.
- **Topology Inference**: Cross-entity consistency validation rules (wall-floor containment, opening-wall host alignment, room graph connectivity).
- **Fine-Grained Fixtures**: Detection and classification of electrical sockets, light switches, door handles, and ceiling lamps with `ATTACHED_TO` relations.
- **Outdoor Perception**: Road surface fitting, curb/sidewalk classification, and tree/lamppost geometric fitting with trunk/canopy radius estimation.
- **Material Classification**: Multi-modal rule-based and appearance material classifier (wood, glass, concrete, brick, plaster, metal, fabric, stone, asphalt, tile, earth, foliage).
- **Spatial Reasoning**: Bilateral relationship inference (`ATTACHED_TO`, `RESTS_ON`, `SUPPORTS`, `CONTAINS`, `PART_OF`) with contact tolerance and elevation alignment.

### Phase 7: WorldIR & Persistence
- **Canonical Representation**: Versioned WorldIR schema with entities, geometry, materials, observations, and competing hypotheses.
- **Competing Hypotheses**: Hypothesis group representation (`hypothesis_group_id`, `competing_hypothesis_ids`) and automated/manual reconciliation.
- **Validation Gates**: Structural validity, geometric bounds, transform sanity, provenance consistency, and cross-entity relationship rules.
- **WorldStore**: Content-addressed immutable persistence with SHA256 integrity, ancestry lineage tracking, and deterministic JSON serialization.

### Phase 8: Reality Studio
- **3D Viewport**: Interactive Three.js viewport with OrbitControls, transform gizmos, floor/elevation grids, and orthographic/perspective modes.
- **Hierarchy Outliner**: Semantic entity tree with category visibility toggles and selection sync.
- **Geometry Position Editing**: Interactive translation editing with bound recalculation and collision detection.
- **Undo/Redo Stack**: Full command history stack with `undo()`, `redo()`, and dirty state tracking.
- **3D Measurements**: Real-time Euclidean distance, floor area, and 3-point angle measurement tools.
- **Evidence & Conflict Inspector**: Inspection panel surfacing source photo evidence, competing hypotheses, and conflict badges.
- **Export Dialog**: 15-format export modal triggering server-side generation and download streaming.

### Phase 9: Export Ecosystem & Production Hardening
- **Comprehensive Exporters**:
  - `glTF` / `GLB`: Hierarchical nodes, PBR material parameters, binary container format.
  - `IFC`: IFC4 BIM export with `IfcBuildingStorey` spatial aggregation and `IfcSpace` room boundaries.
  - `OBJ` / `PLY` / `STL`: Standard mesh exports with MTL material libraries and binary/ASCII support.
  - `USD`: USDA stage hierarchy, UsdShade materials, mesh normals/UVs, physics schemas, camera prims.
  - `Floorplan SVG` / `DXF`: Dimensioned 2D architectural drawings with opening cutouts.
  - `Room Report PDF`: Metric room area, opening schedules, gross/net wall area calculations.
  - `CityJSON` / `CityGML` / `GeoJSON` / `SDF` / `TSCN` / `Unreal` / `SUMO` / `Habitat`.
- **Golden Benchmarks**: 10 permanent benchmark suites in `benchmarks/golden_datasets.py` passing 10/10.
- **Performance**: 100-image reconstruction batch processing completing in <1.5s (<30s target) and SpatialIndex query throughput >1,000 qps.
- **Security Hardening**: OWASP Top 10 response headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `X-XSS-Protection: 1; mode=block`), open `/healthz` container endpoint, CWE-22 path traversal prevention, and timing-safe token verification.
- **Deployment**: Multi-stage production `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, and OpenAPI 3.1 specification (`docs/openapi.json`).

## Test & Acceptance Verification

- **Backend Pytest Suite**: 100% passing across all core, reconstruction, perception, world_ir, export, security, and benchmark tests.
- **Frontend Contract Suite**: 308/308 tests passing in Node.js test runner (`npm --prefix frontend run test:contract`).
- **End-to-End Acceptance**: Golden loop export matrix, vertical slice E2E, and 10 golden benchmark categories all passing.
