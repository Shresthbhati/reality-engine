# Reality Engine — Updated Implementation Backlog
## Comprehensive Work Queue, Prioritization (RICE), and Acceptance Targets

> **Document ID**: `RE-BKLG-009`  
> **Status**: AUTHORITATIVE WORK QUEUE  
> **Baseline**: `origin/main` @ `6db0f86`  
> **Reconciliation Date**: 2026-11-03  

---

## 1. Prioritization Framework (RICE Scoring)

Every backlog item is evaluated using standardized RICE scoring:

$$\text{RICE Score} = \frac{\text{Reach} \times \text{Impact} \times \text{Confidence}}{\text{Effort}}$$

- **Reach (R)**: Percentage of user workflows affected (1 to 10).
- **Impact (I)**: Impact on output quality and utility (3 = Massive, 2 = High, 1 = Moderate, 0.5 = Low).
- **Confidence (C)**: Engineering certainty based on existing foundations (1.0 = High, 0.8 = Medium, 0.5 = Low).
- **Effort (E)**: Person-weeks or sprint points required (1 to 10).

---

## 2. Master Implementation Backlog

| ID | Title & Description | Target Level | Priority | RICE | Affected Files | Dependencies | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| **E01-1** | **Universal Autonomous Orchestrator**<br/>Single-call pipeline automatically evaluating evidence, choosing reconstruction strategy, adapting to hardware, and managing error recovery without user configuration. | **L3** | **P0** | **96** | `engine/orchestrator/universal_pipeline.py`<br/>`sdk/reality.py`<br/>`tests/test_universal_pipeline.py` | None | Raw folder of photos $\to$ validated WorldIR executed autonomously in $<30\text{s}$ for 100 images. (Implemented, Unit & Integration Verified) |
| **E02-1** | **Metric Scale Fusion Engine**<br/>RANSAC-based 3-way fusion engine combining camera EXIF sensor intrinsics, known physical entity priors (switchboards, doors), and GNSS RTK baselines. | **L3** | **P0** | **90** | `reconstruction/scale/metric_anchor.py`<br/>`tests/test_metric_scale_fusion.py` | E01-1 | Metric scale error $\le 1.5\%$ on real room test benchmarks without ground markers. (Implemented & Unit/Integration Verified) |
| **E03-1** | **Multi-View Dense Cloud Fusion**<br/>Multi-view ray unprojection with co-visibility depth consistency voting and statistical outlier removal using Depth Anything V2. | **L4** | **P0** | **88** | `reconstruction/mvs/dense_cloud.py`<br/>`reconstruction/mvs/point_cloud_ops.py` | E02-1 | Point cloud density $> 500\text{ pts/m}^3$; Chamfer distance $\le 15\text{mm}$ to TLS ground truth. (Implemented & Unit/Integration Verified) |
| **E03-2** | **Watertight Mesh & Sharp Edge Repair**<br/>Screened Poisson surface meshing with Neumann boundary constraints and feature-preserving decimation for sharp corners. | **L4** | **P1** | **78** | `reconstruction/meshing/repair.py` | E03-1 | Watertight manifold mesh; $> 85\%$ planar face reduction with zero non-manifold edges. |
| **E04-1** | **Non-Manhattan Room Boundary Solver**<br/>Generalization of EM wall boundary ring solver to support arbitrary wall angles, bay windows, acute alcoves, and corridors. | **L3** | **P0** | **92** | `evidence/promote_rooms.py`<br/>`tests/test_non_manhattan_rooms.py` | E01-1 | 100% closed room polygons on non-Manhattan benchmark datasets with area error $\le 2.0\%$. (Implemented & Unit/Integration Verified) |
| **E04-2** | **Host-Wall Opening Carving**<br/>Volumetric negative boolean extrusion carving doors and windows into host wall slabs with trim and frame schedules. | **L4** | **P1** | **84** | `perception/architecture/openings.py`<br/>`exporters/ifc_bridge.py` | E04-1 | Zero floating doors; clean structural opening cutouts in exported IFC and glTF models. |
| **E04-3** | **Multi-Storey Staircase & Vertical Topology**<br/>Detection of stair treads, risers, stringers, and landings linking vertical storey elevations into continuous frame graph. | **L3** | **P1** | **75** | `engine/compiler/topology_graph.py`<br/>`world_ir/coordinates.py` | E04-1 | Continuous coordinate transformation connecting ground floor to upper levels across stairs. |
| **E05-1** | **High-Res Wall Patch Inspection for Fixtures**<br/>Orthorectified high-resolution surface patch extractor for fine architectural fixtures (sockets, switches, valves, handles). | **L3** | **P0** | **89** | `perception/fixtures/patch_extractor.py`<br/>`perception/fixtures/detector.py` | E04-1 | $\ge 90\%$ precision and $\ge 85\%$ recall on electrical sockets and light switches. (Implemented & Unit/Integration Verified) |
| **E05-2** | **3D Oriented Bounding Box & 6-DoF Solver**<br/>Minimum-volume oriented bounding box (OMBB) solver constrained by supporting surface contact normal and taxonomy priors. | **L3** | **P1** | **80** | `perception/instances/bounding.py`<br/>`perception/entities/taxonomy.py` | E05-1 | 3D bounding box $\text{IoU} \ge 0.70$ for furniture; $\ge 0.60$ for small fixtures. (Implemented & Unit/Benchmark Verified) |
| **E05-3** | **Multi-View Hungarian Instance Association**<br/>Bipartite Hungarian tracking associating visual appearance embeddings across viewpoints to maintain persistent entity IDs. | **L4** | **P0** | **86** | `perception/instances/association.py` | E05-1 | $\ge 98\%$ object identity consistency without duplicate entity generation across viewpoints. (Implemented & Unit/Integration Verified) |
| **E06-1** | **Multi-Modal PBR Material & Texture Classifier**<br/>Surface appearance classification predicting albedo, roughness, and metallic parameters across 12 standard architectural materials. | **L3** | **P1** | **76** | `perception/materials/classifier.py`<br/>`engine/render/textures/` | E03-2 | Macro F1-score $\ge 90\%$; automatic assignment of PBR parameters on exported glTF/USD assets. |
| **E06-2** | **Seam-Blended UV Texture Projection**<br/>Perspective-corrected photo unprojection with multi-band Poisson seam blending generating clean UV texture atlases. | **L3** | **P2** | **68** | `engine/render/textures/projection.py` | E06-1 | Zero visible seam boundary discontinuities on planar wall and floor textures. |
| **E07-1** | **Temporal Multi-Epoch Alignment & Diffing**<br/>Multi-temporal point cloud and entity registration detecting added, removed, moved, renovated, or damaged entities. | **L3** | **P1** | **77** | `world_ir/diff.py`<br/>`world_ir/temporal.py` | E01-1 | Detection of furniture displacements $\ge 5\text{cm}$ and wall alterations between capture epochs. |
| **E08-1** | **Intelligent Next-Best-View Capture Planner**<br/>Ray-casting surface coverage analysis identifying unobserved regions and computing recommended camera poses. | **L3** | **P1** | **74** | `planning/capture_planner.py`<br/>`apps/api/routes_assistant.py` | E03-1 | Recommendations reduce reconstruction uncertainty by $\ge 40\%$ upon new evidence ingestion. |
| **E09-1** | **Reality Studio 3D Interactive Transform Gizmos**<br/>Interactive translation and rotation gizmos with real-time room boundary snapping and collision prevention in Three.js. | **L3** | **P0** | **85** | `frontend/src/components/studio/`<br/>`frontend/src/lib/studio/` | E04-1 | 60 FPS viewport interaction; 100% reversible undo/redo stack on all user edits. |
| **E09-2** | **3D Surface-Projected & Polygon Area Tool**<br/>Interactive measurement tools supporting point-to-point distance, wall-projected distance, and polygon floor area calculation. | **L4** | **P1** | **82** | `frontend/src/lib/studio/measure-model.ts` | E09-1 | Verified accuracy matching server-side `world_ir/measure.py` within $\pm 0.1\%$. (Implemented & Contract Verified) |
| **E10-1** | **IFC4 BIM Semantic Property Set Enrichment**<br/>Enrichment of exported IFC4 spaces and building elements with MEP property sets (Pset_WallCommon, Pset_SpaceCommon). | **L4** | **P1** | **81** | `exporters/ifc_bridge.py` | E04-1 | Validated compliance in buildingSMART IFC4 model checker and Autodesk Revit import. (Implemented & Unit/Integration/Determinism Verified) |
| **E10-2** | **2D Scaled Floor Plan Multi-Sheet PDF**<br/>Multi-page PDF architectural drawing package with title blocks, opening schedules, and net wall area calculations. | **L4** | **P1** | **79** | `exporters/floorplan.py` | E04-1 | Dimensioned PDF sheets conforming to architectural drawing conventions. (Implemented & Unit/Contract Verified) |
| **E11-1** | **Grounded Reality Copilot Spatial Engine**<br/>Natural-language spatial intelligence translating queries to verified SceneGraph queries with zero hallucinations. | **L3** | **P1** | **83** | `assistant/copilot_engine.py`<br/>`assistant/query_translator.py` | E05-3 | 100% precision on spatial entity queries; zero fabricated measurements or geometry. (Implemented & Unit/Integration/Contract Verified) |
| **E12-1** | **Multi-Tenant Project Isolation & Rate Limiting**<br/>Enterprise RBAC permissions, project-scoped data segregation, token bucket rate limiting, and worker sandbox isolation. | **L4** | **P0** | **87** | `apps/api/auth.py`<br/>`apps/api/main.py`<br/>`apps/api/db.py` | None | Strict project isolation; zero privilege escalation; verified in `test_security_audit.py`. |
