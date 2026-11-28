# Perception Capability Baseline (Phase 0)

**Base Commit (`origin/main`):** `34a073f25f57c47f23a5efc0e28c30a13bcbe1ba` (`fix(perception): import ModelIdentity so SAM2 masks are emitted (#414)`)  
**Open PRs at Baseline:**
- `#416` (`fix/segmentation-lazy-torch`) — SAM2 adapter torch-optional, offline-strict, per-run accurate
- `#415` (`security/tenant-isolation-evidence-ownership`) — Tenant isolation & evidence ownership

**Hardware Baseline (Local Verification):** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), PyTorch `2.14.1+cu130` (`cuda: True`).

---

## 1. Component-by-Component Capability Inventory

| # | Capability | Source Files | Model / Algorithm | Invocation Path | Input -> Output | Coordinate Frame | Provenance | Hardware / Offline | Production Used? | Known Limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Open-Vocab Detection** | `perception/detection_adapter.py`, `perception/detection/ontology_grounding.py` | `IDEA-Research/grounding-dino-tiny` | `MultiEngineOrchestrator` -> `run_multi_engine_for_job` | RGB + bounded ontology queries -> 2D bboxes, grounded paths, scores | `IMAGE_PIXEL` | `ModelIdentity` (hardcodes `"gpu"` even on CPU) | 2048 MB VRAM; retries online if `local_files_only` fails | **Yes** | Class-agnostic NMS in `ontology_grounding.nms`; partial failures graded `"succeeded"` |
| 2 | **Closed-Vocab Detection + Masks** | `perception/maskrcnn_adapter.py` | `maskrcnn_resnet50_fpn` (COCO 80) | `MultiEngineOrchestrator` -> `run_multi_engine_for_job` | RGB -> 2D bboxes + RLE in `metadata` | `IMAGE_PIXEL` | `ModelIdentity(hardware_class=device)` | 4000 MB VRAM; offline torchvision weights | **Yes** | Writes `mask_rle` to `metadata` instead of `ObservationRecord.mask`; `break` on first low score |
| 3 | **Instance Segmentation** | `perception/segmentation_adapter.py`, `perception/mask_persistence.py` | `facebook/sam2.1-hiera-small` | `MultiEngineOrchestrator` (depends on `grounding_dino`) | RGB + bboxes -> `ObservationRecord.mask` + `MaskStore` `sha256://` | `IMAGE_PIXEL` | `ModelIdentity(hardware_class=device)` | 3072 MB VRAM; PR #416 fixes lazy torch & strict offline | **Yes** | Requires `grounding_dino` boxes; masks not yet lifted to 3D per-instance meshes |
| 4 | **Semantic Segmentation** | `perception/scene_type.py` | SegFormer / heuristic | `apps/api/jobs.py` (`seg_evidence_quality`) | RGB -> scene type & blank-frame check | `IMAGE_PIXEL` | Stage report | CPU/GPU | **Yes** | Scene-level classification only; not per-entity semantic segmentation |
| 5 | **Monocular Depth (DAv2)** | `perception/depth_adapter.py` | `depth-anything/Depth-Anything-V2-Small-hf` | `MultiEngineOrchestrator` | RGB -> relative inverse depth stats & ref | `CAMERA_METRIC` (mislabelled; actually relative) | `ModelIdentity(hardware_class="gpu")` | 2048 MB VRAM; omits `local_files_only=True` | **Yes** | Relative depth only; no metric calibration |
| 6 | **Monocular Depth (MiDaS)** | `perception/midas_adapter.py` | `midas_dpt_hybrid` | `MultiEngineOrchestrator` | RGB -> relative inverse depth stats | `CAMERA_METRIC` / relative | `ModelIdentity(hardware_class=device)` | 2000 MB VRAM | **Yes** | Relative depth; not fused into dense 3D mesh |
| 7 | **Feature Extraction & Matching (SIFT)** | `perception/engine_adapters.py` | OpenCV SIFT + FLANN + RANSAC E/F | `MultiEngineOrchestrator` | Image pairs + intrinsics -> two-view inliers, Sampson error, relative pose | `IMAGE_PIXEL` / `CAMERA_LOCAL` | `ModelIdentity(cpu)` | CPU only; always offline | **Yes** | Partial pair failures graded `"succeeded"`; basename key collisions |
| 8 | **Feature Matching (LightGlue)** | `perception/engine_adapters.py` | Kornia ALIKED + LightGlue | `MultiEngineOrchestrator` | Image pairs + intrinsics -> two-view inliers & relative pose | `IMAGE_PIXEL` / `CAMERA_LOCAL` | `ModelIdentity(hardware_class="gpu")` | 1024 MB VRAM; requires `torch`+`kornia` | **Yes** | Partial pair failures graded `"succeeded"` |
| 9 | **Camera Localization & SfM** | `reconstruction/colmap_session.py`, `registration/photo_registration.py` | COLMAP CLI / `pycolmap` + PnP | `engine/pipeline/vertical_slice.py` | Multi-view RGB -> sparse point cloud + camera poses | `SFM_GAUGE` -> `+Y up` | Reconstruction report | CPU/GPU | **Yes** | Relative scale unless anchored; fails on textureless/small-baseline sets |
| 10 | **Multi-View Correspondence & Identity** | `engine/multiengine/spatial_intelligence/association.py`, `world_ir/entity_reid.py`, `perception/multiview/room_associator.py` | Hungarian / greedy geometric + label matching | `multi_engine_stage.py` (partial) | Multi-view detections -> `AssociationDecision` | `WORLD` / `CAMERA` | Ledger `AssociationDecision` | CPU | **Partial** | `rekey_entities` incomplete; `association.py` not fully driven by real 3D back-projection |
| 11 | **Object Tracking** | `perception/instances/track_backend.py` | Optical flow / IoU tracking | Video ingestion | Video frames -> track IDs | `IMAGE_PIXEL` | Track metadata | Requires `imageio` | **Video only** | Not used for multi-angle still photos |
| 12 | **Geometric Reconstruction** | `reconstruction/meshing/`, `perception/architecture/room_semantics.py`, `openings.py` | Plane fitting + occupancy raster + Poisson/marching cubes | `vertical_slice.py` | Point cloud -> planes, rooms, openings, `furniture_candidate` boxes | `WORLD` (`+Y up` / Z-up mix) | `Provenance.RECONSTRUCTED` | CPU | **Yes** | `openings.py` has depth-noise phantom positives & drops edge/full-height doors |
| 13 | **Architectural Understanding** | `perception/architecture/` (`scene.py`, `room_graph.py`, `stairs.py`, `roofs.py`, `corridors.py`) | Geometric heuristics | `vertical_slice.py` | Point cloud -> rooms, walls, floors, ceilings | `WORLD` | Stage report | CPU | **Yes (Rooms/Walls)** | Stairs/roofs/beams largely unvalidated on real captures |
| 14 | **Small-Object Detection** | `perception/detection/ontology_grounding.py`, `perception/entities/small_objects.py`, `perception/evaluation/small_object_benchmark.py` | Tiled Grounding DINO + NMS | `detection_adapter.py` | High-res tiles -> small-class detections | `IMAGE_PIXEL` | `Grounding.is_small` | 2048 MB VRAM | **Partial** | Class-agnostic NMS merges neighboring distinct classes; benchmark ignores class labels |
| 15 | **Outdoor Scene Understanding** | `config/ontology.yaml` (`road`, `street`, `vegetation`, `exterior`), `geo/osm_context.py` | Open-vocab queries + OSM context | `detection_adapter.py` (`domain="outdoor"`) | Outdoor RGB -> 2D boxes | `IMAGE_PIXEL` | `Grounding.group` | 2048 MB VRAM | **Unvalidated** | No outdoor real-photo benchmark score yet; indoor geometry compiler assumes rooms |
| 16 | **Spatial Relationships** | `world_ir/relations.py`, `engine/multiengine/spatial_intelligence/relationships.py`, `perception/architecture/space_graph.py` | AABB proximity & topology | `world_compiler.py` | Entities -> `SUPPORTED_BY`, `INSIDE`, `CONNECTED_TO` | `WORLD` | `Relationship` | CPU | **Partial** | `world_ir/relations.py` hardcodes Z-up on `Frame.Y_UP` `WorldIR` |
| 17 | **Uncertainty Estimation** | `uncertainty/propagation.py`, `world_ir/uncertainty_model.py`, `engine/multiengine/evidence_fusion.py` | Variance propagation & IVW fusion | `evidence_fusion.py` | `FusableObservation` -> `FusedPrediction` | `WORLD` | `Provenance` + `Uncertainty` | CPU | **Partial** | `evidence_fusion.py` has value/obs misalignment & hardcoded `0.01` precision |

---

## 2. Real-Photo Datasets in Repository

| Dataset | Path | Contents | Ground Truth Available? |
|---|---|---|---|
| **Real Room Capture** | `datasets/real_room_capture/` | 21 iPhone 14 Pro photos + COLMAP `cameras.txt`, `images.txt`, `points3D.txt` | `ground_truth_identities.json` |
| **Room Capture** | `datasets/room_capture/` | 28 interior photos + `manifest.json` | Camera/manifest only (no 3D object GT) |
| **South Building** | `datasets/south_building/` | 32 outdoor building/facade/vegetation photos + COLMAP `sparse/` | Sparse SfM only |
| **Scene Disjoint Evaluation** | `datasets/scene_disjoint_evaluation/` | 10 indoor+outdoor photos (`IMG_0000..0004` + `P1180141..157`) | `ground_truth_identities.json` |
| **Measured Rooms** | `benchmarks/measured_rooms/` | `room_01_living_studio` (21 JPEGs) + `ground_truth.json` for rooms 1–6 | Laser-survey `ground_truth.json` |
