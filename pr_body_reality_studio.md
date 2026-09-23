# PR: Desktop Reality Studio Experience — Persistent Spatial Operating Environment

## Summary

Implements the official desktop Reality Studio experience centered on the **World** as the primary product object, strictly honoring the canonical product lifecycle:

$$\text{World} \longrightarrow \text{Evidence} \longrightarrow \text{Sessions} \longrightarrow \text{Processing} \longrightarrow \text{Reconstruction} \longrightarrow \text{WorldIR} \longrightarrow \text{Inspection} \longrightarrow \text{Query} \longrightarrow \text{Correction} \longrightarrow \text{Version} \longrightarrow \text{Diff} \longrightarrow \text{Export}$$

Builds the spatial operating environment around the existing core components (`SpatialWorkstation`, `World3DViewport`, `WorldNavPanel`, `AdaptiveInspector`, `BottomContextBar`, `CommandPalette`, `WorldMap`), consuming live backend APIs and authenticated datasets with zero mock models or synthetic versions.

---

## Changes

### 1. Persistent Spatial Operating Environment
- **`frontend/src/components/shell/AppShell.tsx`**:
  - Immersive spatial studio route handling: routes `/` and `/worlds/[id]` occupy the full viewport without dashboard sidebars or letterboxing.
  - Maintains global command palette (`⌘K`) listening across the entire application while reserving generic dashboard navigation for secondary admin screens (`/sessions`, `/evidence`, `/reports`, `/settings`).
- **`frontend/src/components/workspace/SpatialWorkstation.tsx`**:
  - **Honest World State**: When opening uncompiled or pre-reconstruction worlds, the workstation remains persistent and operational, displaying an authentic "Room Reconstruction Pending" state with attached sessions and evidence, rather than crashing into a fatal error screen.
  - **Studio Technical Header**: Reality Studio branding, live status pulse, active world identity, coordinate frame badge (`metric_enu`), active version pill (`v1.0.0`), 3D vs Geospatial Map switcher, layout toggles (`[`, `]`, `\`), and ⌘K trigger.
  - **Integrated Query State**: Manages real-time spatial query filters synchronized between `WorldNavPanel`, `World3DViewport`, and `AdaptiveInspector`.
  - **Keyboard Hotkeys & Event Dispatching**: Global listeners for `[`, `]`, `\`, `F`, `M`, `1`, `2`, `3`, `O`, `U`, `Esc`, `⌘K`, `trigger-export`, `open-room-construction`, `open-version-diff`, `apply-spatial-query`.

### 2. World-First Navigation
- **`frontend/src/components/navigation/WorldNavPanel.tsx`**:
  - World-first switcher across live API worlds.
  - **Spatial Hierarchy**: Categorizes entities into **Structural Planes** (floors, walls, ceilings) and **Spatial Objects** (furniture, equipment) with confidence meters and instant framing (`[F]`).
  - **Evidence Context**: Lists calibrated camera viewpoints with real inlier percentages and residual errors (in mm).
  - **Sessions & Spatial Anchors**: Displays attached capture sessions and authentic coordinate frame anchors (Metric ENU origin, bounding center) without synthetic mock places.
  - **Version Lineage**: Lists immutable WorldStore versions with timestamps and change summaries, plus direct access to the **Version Diff** comparison modal.
  - **Pipeline Overview**: Live visual status of the 5 canonical room construction stages.
  - **Direct Exports**: Trigger downloads of canonical WorldIR (JSON), point cloud (PLY), camera poses (JSON), and pipeline report (JSON).

### 3. Real-Time Spatial Query & Viewport Isolation
- **`frontend/src/lib/viewport/three-scene.ts`**:
  - Implemented `applyEntityFilter(matchingIds: Set<string> | null)` in `WorldSceneController`: isolates matching entities at full opacity while dimming non-matching objects to ghost outlines ($0.04$ opacity).
  - Preserves bounds calculation, wireframe selection outlines, and viewport resize stability.
- **`frontend/src/components/viewport/World3DViewport.tsx`**:
  - Added `queryMatchingIds` prop and reactive scene sync.
  - Added HUD query matching badge (`Query: N matching`) next to point cloud and camera telemetry.
  - Refined empty/pending state overlay for uncompiled worlds.

### 4. Deep Inspection & Evidence Photo Viewer
- **`frontend/src/components/inspector/AdaptiveInspector.tsx`**:
  - Multi-tab inspection: Overview, Geometry & Bounds (dimensions and volume in $m^3$), Evidence Context, Observations, Provenance, and Uncertainty.
  - **Live Authentic Photo Viewer**: Integrated genuine captured camera photos (`IMG_0000.jpg`, etc.) directly for observing cameras via `/api/worlds/[id]/evidence/[evidenceId]/image`.
  - **In-Place Entity Correction**: Operators can reclassify types, adjust confidence, update semantic tags, write a commit rationale, and persist to WorldStore via `/api/worlds/[id]/commit`.
  - **World Overview**: Comprehensive diagnostic facts when no entity is selected (Reconstruction backend, camera registration ratio, metric scale baseline, monocular depth fusion metrics).

### 5. Room Construction Workflow
- **`frontend/src/components/workspace/RoomConstructionModal.tsx`**:
  - Modal walking through the 7 canonical pipeline stages:
    1. Evidence Ingestion & Validation
    2. SfM Multi-View Pose Estimation
    3. Metric Scale Calibration
    4. Monocular Depth Alignment & Fusion
    5. Structural Plane Promotion
    6. WorldIR Canonical Compilation
    7. WorldStore Version Snapshot
  - Configurable SfM backend (`colmap`), depth model (`midas` / `dpt`), and GPU acceleration toggle.

### 6. Command Discoverability
- **`frontend/src/components/shell/CommandPalette.tsx`**:
  - Accessible via `⌘K` / `Ctrl+K`: Layer toggles (`1`, `2`, `3`, `O`, `U`), view presets (Isometric, Top, Front), spatial queries (floors, walls, objects, high confidence, low confidence review), version comparison, and direct artifact exports (WorldIR, PLY, Cameras, Report).

### 7. API Routes
- **`frontend/src/app/api/worlds/[id]/evidence/[evidenceId]/image/route.ts`**:
  - Serves genuine captured camera images (`datasets/room_capture/images/{evidenceId}.jpg`) for authentic evidence inspection.
- **`frontend/src/app/api/worlds/[id]/report/route.ts`**:
  - Supports local fallback to `datasets/room_capture/pipeline_out/report.json` when the backend service is offline.

---

## Design Principles Followed

- **Canonical Product Model Integrity**: Every stage of the 12-step model is represented without shortcuts or architectural divergence.
- **Zero Mock Data Guarantee**: No fake world objects, no mock versions, no simulated geometry; unavailable states are represented honestly.
- **Persistent Spatial Operating Environment**: The spatial workstation is the primary operating canvas, not an embedded dashboard widget.
- **Professional Information Density**: Dark slate palette (`#08090b` / `#0e1013`), cyan accents (`#00e5ff`), monospace telemetry, and calm technical HUDs.

---

## Testing & Verification

```bash
# 1. Type check
cd frontend && npx tsc --noEmit
# Exit code: 0 (0 errors)

# 2. Production build
npm run build
# Compiled successfully with Turbopack (exit code: 0)
```
