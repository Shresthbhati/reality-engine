> **SUPERSEDED (2026-11-10).** Current status lives in [docs/STATUS.md](docs/STATUS.md); current plan in docs/blueprint/34_MASTER_PLAN.md. Kept for history.

# Reality Engine — Implementation Dependency Graph
## Execution Sequencing, Critical Path Analysis, and Parallelization Lanes

> **Document ID**: `RE-DEP-006`  
> **Status**: APPROVED EXECUTION SPECIFICATION  
> **Baseline**: `origin/main` @ `6db0f86`  

---

## 1. Executive Sequencing Philosophy

Reality Engine follows a strict **Foundation-First Execution Discipline**:
- High-level architectural reasoning and semantic fixture localizers cannot operate reliably if camera poses and metric scale are uncalibrated.
- Mesh generation cannot be evaluated if point cloud filtering and depth consistency are unstable.
- Every layer produces typed, schema-validated contracts consumed by downstream modules.

---

## 2. Global Dependency DAG (Directed Acyclic Graph)

```mermaid
flowchart TD
    subgraph L0 ["Layer 0: Core Contracts & Benchmark Foundations"]
        L0_1["0.1 WorldIR v1 Schema Extension<br/>world_ir/schema_v1.py"]
        L0_2["0.2 Entity Taxonomy Expansion<br/>perception/entities/taxonomy.py"]
        L0_3["0.3 Golden Dataset Ground Truth Fixtures<br/>benchmarks/golden_datasets.py"]
    end

    subgraph L1 ["Layer 1: Ingestion, Multi-View Intelligence & Metric Anchor"]
        L1_1["1.1 Universal Ingestion Preprocessing<br/>evidence/importers.py, video.py, dedup.py"]
        L1_2["1.2 Feature Matching & Camera Registration<br/>reconstruction/matching/lightglue_matcher.py"]
        L1_3["1.3 Metric Scale Estimation Engine<br/>reconstruction/scale/metric_anchor.py"]
    end

    subgraph L2 ["Layer 2: Dense Reconstruction & Surface Meshing"]
        L2_1["2.1 Depth Anything V2 Dense Unprojection<br/>reconstruction/depth/depth_anything.py"]
        L2_2["2.2 Multi-View Dense Cloud Builder<br/>reconstruction/mvs/dense_cloud.py"]
        L2_3["2.3 Point Cloud Filtering & Outlier Removal<br/>reconstruction/mvs/point_cloud_ops.py"]
        L2_4["2.4 Surface Meshing & Watertight Repair<br/>reconstruction/meshing/repair.py"]
    end

    subgraph L3 ["Layer 3: Architectural Layout & Topology"]
        L3_1["3.1 RANSAC Plane Detection & Normal Classification<br/>perception/geometry/planes.py, orientation.py"]
        L3_2["3.2 Non-Manhattan Room Boundary Ring Solver<br/>evidence/promote_rooms.py, scene.py"]
        L3_3["3.3 Host-Wall Opening Cutout Engine<br/>perception/architecture/openings.py"]
        L3_4["3.4 Building Topology Graph & Multi-Storey Linkage<br/>engine/compiler/topology_graph.py"]
    end

    subgraph L4 ["Layer 4: Entity Perception & Fine Fixture Localizer"]
        L4_1["4.1 High-Resolution Wall-Surface Patch Extraction<br/>perception/fixtures/patch_extractor.py"]
        L4_2["4.2 Fine Fixture Detector (Sockets, Switches, Valves)<br/>perception/fixtures/detector.py"]
        L4_3["4.3 Multi-View Hungarian Instance Association<br/>perception/instances/association.py"]
        L4_4["4.4 Outdoor Infrastructure & Landscape Fitting<br/>perception/environment/outdoor.py"]
    end

    subgraph L5 ["Layer 5: Spatial Reasoning, Materials & WorldIR Gate"]
        L5_1["5.1 Multi-Modal Material & Texture Classifier<br/>perception/materials/classifier.py"]
        L5_2["5.2 Bilateral Relationship Inference (ATTACHED_TO, etc.)<br/>perception/spatial_reasoning.py"]
        L5_3["5.3 Automated Cross-Entity Consistency Gate<br/>world_ir/validation.py"]
        L5_4["5.4 Competing Hypothesis Reconciliation<br/>world_ir/conflict.py"]
        L5_5["5.5 Immutable WorldStore Commit Engine<br/>world_ir/store.py"]
    end

    subgraph L6 ["Layer 6: Reality Studio Viewport & Interaction"]
        L6_1["6.1 Three.js 3D Viewport & Gizmo Interaction<br/>frontend/src/components/studio/ThreeViewport.tsx"]
        L6_2["6.2 Real-Time 3D Measurement Tools<br/>frontend/src/lib/studio/measure-model.ts"]
        L6_3["6.3 Outliner Hierarchy & Layer Toggles<br/>frontend/src/components/studio/OutlinerPanel.tsx"]
        L6_4["6.4 Undo/Redo Command History Stack<br/>frontend/src/lib/studio/history-stack.ts"]
        L6_5["6.5 Evidence & Conflict Inspector UI<br/>frontend/src/components/studio/InspectorPanel.tsx"]
    end

    subgraph L7 ["Layer 7: Universal Exporters & Reality Copilot"]
        L7_1["7.1 IFC4 BIM & Space Aggregation Exporter<br/>exporters/ifc_bridge.py"]
        L7_2["7.2 Universal Scene Description (USD) Exporter<br/>exporters/usd/exporter.py"]
        L7_3["7.3 2D Scaled Floor Plans (SVG, DXF, PDF)<br/>exporters/floorplan.py"]
        L7_4["7.4 Standard Mesh Exporters (OBJ, PLY, STL)<br/>exporters/mesh/"]
        L7_5["7.5 Reality Copilot Grounded Query Engine<br/>assistant/copilot_engine.py"]
    end

    subgraph L8 ["Layer 8: Production Hardening & Autonomous Operation"]
        L8_1["8.1 Universal Pipeline Autonomous Orchestrator<br/>engine/orchestrator/universal_pipeline.py"]
        L8_2["8.2 Temporal Multi-Epoch Diffing Engine<br/>world_ir/diff.py, temporal.py"]
        L8_3["8.3 Intelligent Next-Best-View Capture Planner<br/>planning/capture_planner.py"]
        L8_4["8.4 OWASP Top 10 Security & Multi-Tenant Isolation<br/>apps/api/auth.py, main.py"]
    end

    L0_1 & L0_2 --> L1_1 & L1_2
    L0_3 --> L1_3
    L1_1 --> L1_2 --> L1_3
    L1_3 --> L2_1 --> L2_2 --> L2_3 --> L2_4
    L2_2 --> L3_1 --> L3_2 --> L3_3 --> L3_4
    L3_1 --> L4_1 --> L4_2 --> L4_3
    L2_2 --> L4_4
    L2_4 & L4_2 --> L5_1
    L3_4 & L4_3 & L4_4 --> L5_2 --> L5_3 --> L5_4 --> L5_5
    L5_5 --> L6_1 & L6_2 & L6_3 & L6_4 & L6_5
    L5_5 --> L7_1 & L7_2 & L7_3 & L7_4 & L7_5
    L5_5 --> L8_1 & L8_2 & L8_3 & L8_4
```

---

## 3. Critical Path Analysis

The critical path determines the minimum sequence of dependent tasks that govern project delivery:

$$\text{Critical Path} = \mathbf{L0.1} \to \mathbf{L1.2} \to \mathbf{L1.3} \to \mathbf{L2.2} \to \mathbf{L3.1} \to \mathbf{L3.2} \to \mathbf{L4.2} \to \mathbf{L5.2} \to \mathbf{L5.5} \to \mathbf{L7.1}$$

1. **L0.1 Schema Definition**: Unifies `EntityType` and `RelationshipKind` contracts.
2. **L1.2 Multi-View Registration**: Produces verified camera poses from unorganized photos.
3. **L1.3 Metric Scale Anchor**: Constrains relative poses to true physical meters.
4. **L2.2 Dense Cloud Builder**: Unprojects multi-view depth maps into dense point clouds.
5. **L3.1 Plane Detection**: Fits planar wall, floor, and ceiling surfaces.
6. **L3.2 Room Boundary Solver**: Solves closed topological cycles of room boundary walls.
7. **L4.2 Fixture Detector**: Identifies electrical, plumbing, and mechanical fixtures.
8. **L5.2 Spatial Reasoning**: Binds fixtures to walls via `ATTACHED_TO` relationships.
9. **L5.5 WorldStore Commit**: Atomically persists the complete validated WorldIR.
10. **L7.1 IFC BIM Export**: Exports verified architectural digital twins to industry CAD/BIM tools.

---

## 4. Parallelization Workstream Lanes

Once Layer 0 and Layer 1 are stabilized, development can proceed concurrently across 4 independent lanes without blocking dependencies:

### Lane A: Architectural & Reconstruction Track (Geometric Core)
* **Scope**: L2.1 $\to$ L2.2 $\to$ L2.4 $\to$ L3.1 $\to$ L3.2 $\to$ L3.3 $\to$ L3.4
* **Team**: Reconstruction Engineer + Architectural Intelligence Engineer
* **Files**: `reconstruction/`, `perception/geometry/`, `perception/architecture/`

### Lane B: Entity Perception & Material Track (Semantic Core)
* **Scope**: L4.1 $\to$ L4.2 $\to$ L4.3 $\to$ L4.4 $\to$ L5.1 $\to$ L5.2
* **Team**: Perception Engineer + Material & Texture Engineer
* **Files**: `perception/fixtures/`, `perception/instances/`, `perception/materials/`

### Lane C: Reality Studio & Interaction Track (Frontend & Presentation)
* **Scope**: L6.1 $\to$ L6.2 $\to$ L6.3 $\to$ L6.4 $\to$ L6.5 $\to$ L7.5
* **Team**: Frontend Engineer + Reality Copilot Engineer
* **Files**: `frontend/src/components/studio/`, `frontend/src/lib/studio/`, `assistant/`

### Lane D: Exporters, Platform & Security Track (Infrastructure & Delivery)
* **Scope**: L7.1 $\to$ L7.2 $\to$ L7.3 $\to$ L7.4 $\to$ L8.1 $\to$ L8.2 $\to$ L8.4
* **Team**: Exporters Lead + Platform Reliability Engineer
* **Files**: `exporters/`, `apps/api/`, `engine/orchestrator/`, `benchmarks/`
