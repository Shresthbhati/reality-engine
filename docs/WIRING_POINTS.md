# Reality Engine — Frontend Architecture & Backend Wiring Guide

**Document Authority:** Reality Engine V11 UI/UX Architecture  
**Target Application:** `frontend/` (Next.js 16 + React 19 + Turbopack + Tailwind v4 + Zustand + R3F)  
**Date:** 2026-09-15  
**Status:** Frontend Complete & Production Verified — Ready for Backend Integration  

---

## 1. Executive Summary & Architecture

Reality Engine is a professional spatial computing workstation and reality-compilation platform comprising 8 dedicated operational surfaces:

1. **Reality Studio (3D Viewport Workstation):** Deep inspection & editing of reconstructed worlds (R3F 3D viewport, World Outliner, Entity Inspector, and contextual Evidence Trace).
2. **Reality Build (Compilation Pipeline):** Multi-stage photogrammetry, SLAM, and semantic fusion DAG orchestrator (COLMAP, OpenMVS, SAM2, WorldIR compiler).
3. **Desktop Data Loader (Ingestion):** Bulk multi-sensor ingestion (drone photogrammetry, smartphone scans, LiDAR, GNSS logs, IMU).
4. **City Builder (World Authoring):** Multi-session compositional world authoring combining reconstructed fragments, GIS layers, and procedural generative rules.
5. **Evidence Explorer (Provenance Verification):** Cryptographic & algorithmic lineage graph answering *"Why does Reality Engine believe this entity exists?"*
6. **Simulation Control Room:** Physical simulation and hydrodynamic / agent propagation environment.
7. **Engine Diagnostics:** CUDA/GPU hardware metrics, real-time compute load, memory fragmentation, and pipeline event streams.
8. **Settings & Standards:** Coordinate Reference System (EPSG), SI dimensional constraints, precision solvers, and downstream export compilers (OpenUSD, IFC 4x3, glTF, Unreal, ROS 2).

---

## 2. Shared Domain Model & Universal Work Units

All applications in the ecosystem operate over the canonical **Work Unit Hierarchy**:

```text
PROJECT (e.g. Kolkata_Industrial_Site)
  │
  ├── CAPTURE SESSIONS (Phone_001, Drone_001, LiDAR_001)
  │     ├── Sources & Calibrations
  │     ├── Camera Trajectories & GNSS Logs
  │     └── Self-contained World Fragments
  │
  ├── WORLDS (Composed WorldIR Representation)
  │     ├── Entities (World, Site, Building, Floor, Room, Wall, Terrain, Road)
  │     ├── Geometries (LOD0-2 Meshes, 3D Gaussian Splats, Point Clouds)
  │     └── Semantic Annotations & Physical Materials
  │
  ├── BUILDS (Execution Pipelines)
  │     └── Capture → Time Sync → Calibration → Localization → Registration → Sparse → Dense → Fusion → Perception → WorldIR
  │
  ├── EVIDENCE & PROVENANCE CHAINS
  │     └── Image → Feature → Match → Camera Pose → Depth → Point → Observation → Entity
  │
  └── SIMULATION SCENARIOS
        └── Boundary Conditions → Fluid/Rigid Dynamics → Agent Dynamics → State Deltas
```

---

## 3. Zustand Global Store Wiring (`src/store/re-store.ts`)

All components consume state from `useREStore`. When connecting to Python FastAPI / gRPC backends, replace the initial mock state with reactive network fetchers:

### Store Actions & Backend Endpoint Mapping

| Store Action | API Method | Backend Route | Data Schema |
| :--- | :--- | :--- | :--- |
| `fetchProject(id)` | `GET` | `/api/v1/projects/{id}` | `Project` |
| `fetchWorld(id)` | `GET` | `/api/v1/worlds/{id}` | `World` |
| `fetchEntities(worldId)` | `GET` | `/api/v1/worlds/{worldId}/entities` | `Entity[]` |
| `patchEntity(id, delta)` | `PATCH` | `/api/v1/entities/{id}` | `Partial<Entity>` |
| `fetchSessions(projId)` | `GET` | `/api/v1/projects/{projId}/sessions` | `Session[]` |
| `createSession(payload)` | `POST` | `/api/v1/sessions` | `CreateSessionRequest` |
| `uploadSessionChunk(id)` | `POST` | `/api/v1/sessions/{id}/upload` | Multipart Chunk Stream |
| `startBuild(sessionIds)` | `POST` | `/api/v1/builds` | `{ sessionIds: string[] }` |
| `retryStage(buildId, stg)` | `POST` | `/api/v1/builds/{buildId}/stages/{stg}/retry` | None |
| `composeWorld(worldId)` | `POST` | `/api/v1/worlds/{worldId}/compose` | `{ sessionIds: string[] }` |
| `generateProcedural(rule)`| `POST` | `/api/v1/worlds/{worldId}/procedural/generate` | `ProceduralRule` |
| `fetchProvenance(entId)` | `GET` | `/api/v1/entities/{entId}/provenance-dag` | `ProvenanceDAG` |
| `fetchObservations(entId)`| `GET` | `/api/v1/entities/{entId}/observations` | `Observation[]` |
| `startSimulation(scenId)` | `POST` | `/api/v1/simulations/{scenId}/run` | `SimulationParams` |

---

## 4. WebSocket Live Streaming Channels

Connect the frontend event bus to the following WebSocket endpoints:

| Channel URL | Payload Type | Target UI Component | Frequency |
| :--- | :--- | :--- | :--- |
| `ws://host/ws/v1/compute/metrics` | `ComputeMetrics` | `ComputeMetrics.tsx` | 1000 ms (1 Hz) |
| `ws://host/ws/v1/builds/{id}/progress`| `PipelineStageInfo` | `PipelineStageList.tsx` | Event-driven |
| `ws://host/ws/v1/logs` | `LogEntry` | `LogStream.tsx` | Real-time stream |
| `ws://host/ws/v1/simulations/{id}` | `SimulationTick` | `SimWorldView.tsx`, `SimTimeline.tsx` | 60 Hz |
| `ws://host/ws/v1/entities/mutations` | `EntityDelta` | `WorldOutliner.tsx`, `Viewport3D.tsx` | Event-driven |

---

## 5. Viewport 3D Geometry Streaming (`src/components/workspaces/studio/Viewport3D.tsx`)

The 3D Viewport is implemented with **React Three Fiber** and **Three.js**.
To stream actual geometry from the Reality Engine backend:

1. **Point Clouds:**
   - Stream binary `.ply` or quantized octree point tiles via `THREE.BufferGeometryLoader` or `Potree` / `copc.js` into `SamplePointCloud()`.
   - Wire coordinates to `WorldIR` bounding box centers.
2. **Meshes & Splats:**
   - Stream `.glb` / Draco-compressed LOD2 meshes into the scene graph.
   - For Gaussian Splatting, mount `@mkkellogg/gaussian-splats-3d` onto the canvas.
3. **Camera Trajectories:**
   - Wire `useREStore().sessions[...].sources` camera extrinsics into wireframe frustums using `THREE.CameraHelper` instances along the GNSS flight path.

---

## 6. How to Run & Verify the Frontend

```powershell
# Navigate to frontend directory
cd frontend

# Development server with instant hot-reload
npm run dev

# Production verification build (verified with Next 16 Turbopack)
npm run build
```

Open `http://localhost:3000` to interact with the full Reality Engine professional workstation.
