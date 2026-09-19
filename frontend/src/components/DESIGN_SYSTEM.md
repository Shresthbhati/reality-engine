# Reality Engine Design System (REDS) — UI/UX Foundation

> **Status**: Production Foundation  
> **Visual Source of Truth**: `ui ux pics/`  
> **Owner**: Agent 1 (Frontend / UI / UX Foundation & Design System)  
> **Collaborator**: Agent 2 (Specialized World / Capture Workflows & Interaction Screens)

---

## 1. Visual Philosophy & Identity

Reality Engine is a professional, high-performance spatial-computing workstation for reconstructing, persisting, and querying digital twins of physical reality.

It is designed to feel like an authentic engineering instrument—comparable to Unreal Engine, Blender, Cesium, and VS Code—never a generic SaaS dashboard or a toy game launcher.

### Key Visual Tenets:
1. **Dark-First Obsidian Surfaces**: Deep backgrounds (`#08090b`, `#0a0b0e`, `#101217`) minimize eye fatigue during extended 3D inspection and maintain extreme dynamic contrast for point clouds and wireframes.
2. **Electric Cyan & Azure Accent**: High-visibility cyan (`#00e5ff`) with glowing indicators (`rgba(0, 229, 255, 0.25)`) signals active spatial tools, selection outlines, and verified data.
3. **Engineering Truth & Tabular Metrics**: Monospace fonts (`JetBrains Mono`, `tabular-nums`) are used consistently for coordinates, elevation, geometric uncertainty, and camera frame indices. Low-confidence data or sparse reconstructions are presented honestly without fake statistics.
4. **Scale-Aware UI**: Controls and inspectors adapt dynamically to the active spatial scale level.

---

## 2. Design Tokens (`reds-tokens.css` & `globals.css`)

### Base Surfaces
```css
--re-bg-base:        #08090b;  /* Deepest viewport & canvas background */
--re-bg-elevated:    #101217;  /* Sidebar panels, top/bottom bars */
--re-bg-surface:     #151821;  /* Cards, input fields, accordion items */
--re-bg-overlay:     #1a1d28;  /* Dropdowns, tooltips, dialogs */
--re-bg-hover:       rgba(255, 255, 255, 0.05);
```

### Borders
```css
--re-border-subtle:  #171922;
--re-border-default: #1f222b;
--re-border-strong:  #2d323f;
--re-border-focus:   #00e5ff;
--re-border-active:  rgba(0, 229, 255, 0.4);
```

### Accent & Global Modes
```css
--re-accent:         #00e5ff;  /* Electric cyan */
--re-accent-blue:    #3d8ef7;  /* Engineering azure */

/* Global Mode Colors */
--re-mode-explore:   #38bdf8;
--re-mode-inspect:   #00e5ff;
--re-mode-edit:      #f59e0b;
--re-mode-capture:   #a855f7;
--re-mode-build:     #10b981;
--re-mode-review:    #ec4899;
```

### Continuous Spatial Scales (LOD Hierarchy)
```css
--re-scale-room:        #ec4899;  /* 0 – 10 m (Sub-cm / mm) */
--re-scale-building:    #a855f7;  /* 10 – 50 m (cm) */
--re-scale-street:      #8b5cf6;  /* 50 – 200 m */
--re-scale-plot:        #6366f1;  /* 100 – 300 m */
--re-scale-block:       #3b82f6;  /* 200 – 1,000 m */
--re-scale-multiblock:  #0ea5e9;  /* 1 – 3 km */
--re-scale-locality:    #06b6d4;  /* 3 – 8 km */
--re-scale-ward:        #14b8a6;  /* 5 – 15 km */
--re-scale-district:    #10b981;  /* 15 – 40 km */
--re-scale-city:        #22c55e;  /* 40+ km (Metropolitan) */
```

---

## 3. Master Workspace Layout

The application organizes space into 5 coordinated functional areas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TITLEBAR: Logo | Global Modes | Spatial Scale Ribbon | Search | World | SB  │
├─────────────────┬─────────────────────────────────────────┬─────────────────┤
│                 │                                         │                 │
│                 │                                         │                 │
│  LEFT PANEL:    │        CENTER: 3D WORLD VIEWPORT        │  RIGHT PANEL:   │
│  World Nav &    │        Persistent spatial view,         │  Context        │
│  Entity Tree    │        HUD overlays, toolbar, gizmo     │  Inspector      │
│  (⌘B to toggle) │                                         │  (⌘I to toggle) │
│                 │                                         │                 │
│                 │                                         │                 │
├─────────────────┴─────────────────────────────────────────┴─────────────────┤
│ BOTTOM: Contextual Timeline / Processing / Capture Guidance / Evidence (⌘J) │
├─────────────────────────────────────────────────────────────────────────────┤
│ STATUS BAR: Version | Mode | Scale | Status | Coords · CRS · GPU · 60 FPS   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. TitleBar (`components/shell/TitleBar.tsx`)
- **Left**: Branding emblem + version tag + tagline ("Real places. Persistent digital worlds.")
- **Center**:
  - Global Modes: `EXPLORE`, `INSPECT`, `EDIT`, `CAPTURE`, `BUILD`, `REVIEW`
  - Global Spatial Scale Ribbon: `ROOM` → `BUILDING` → `STREET` → `PLOT` → `BLOCK` → `MULTI-BLOCK` → `LOCALITY` → `WARD` → `DISTRICT` → `CITY`
- **Right**:
  - Command palette shortcut (`Ctrl+K`)
  - Active world selector (`Middletown | V7.3 Active`)
  - Engine connectivity state (`ONLINE ●`)
  - Notification stack trigger
  - Settings and user profile

### 2. World Navigation (`components/shell/WorldNavPanel.tsx`)
- Collapsible Left Panel (`⌘B`)
- Categorical accordions matching visual references:
  - **WORLDS**: Active / Archived world twins with coverage area
  - **SESSIONS**: Capture sessions with sensor tags and frame counts
  - **EVIDENCE**: Multi-modal sensor datasets (Ground Photos, Aerial, Lidar, Satellite, Logs)
  - **PLACES**: Curated spatial landmarks with auto-scale targeting
  - **BOOKMARKS**: Critical defect/observation bookmarks (e.g. Bridge Crack V7)
  - **VERSIONS**: Temporal baseline vs current state
- Search filter for live entity/session querying
- Tab toggle to switch between `WORLD NAV` and `ENTITIES` (the fine-grained object outliner)

### 3. Persistent 3D Viewport (`components/workspaces/studio/Viewport3D.tsx`)
- Hero canvas running Three.js with OrbitControls
- Viewport toolbar (perspective/orthographic, shading modes: RGB, wireframe, normals, point cloud, splats, confidence, coverage)
- Spatial Scale & Mode HUD badge floating over center canvas
- Responsive collapse restore pills (`⌘B` and `⌘I`)

### 4. Context Inspector (`components/shell/ContextInspectorPanel.tsx`)
- Collapsible Right Panel (`⌘I`)
- Structured inspection accordions:
  - **GEOMETRY**: Mesh count, triangles, vertices, LOD resolution
  - **SEMANTICS**: Architectural class, material properties, physical height
  - **EVIDENCE**: Sensor provenance, source keyframes, pixel ground resolution
  - **PROVENANCE**: Reconstruction algorithms, daemon build version
  - **UNCERTAINTY**: Geometric error gauge (cm/mm) + Semantic confidence bar (%)
  - **TEMPORAL HISTORY**: Historical modifications, bulb replacement, installation dates
  - **RELATIONSHIPS**: Spatial connectivity graph (e.g. Connected to Power Grid, Located on Sidewalk)
- Tab toggle to switch between `CONTEXT INSPECTOR` and `ATTRIBUTES`

### 5. Contextual Bottom Drawer (`components/shell/ContextualBottomDrawer.tsx`)
- Collapsible Bottom Drawer (`⌘J`)
- 4-column workstation view:
  - **TIMELINE**: Multi-stage progression (`Evidence → Session → Reconstruction → Quality Check → World Update → Query/Review`) with play/pause and stage pips
  - **PROCESSING**: Active asynchronous tasks with real progress bars (Lidar Point Cloud 87%, Semantic Segmentation 100%, Mesh Refinement pending)
  - **CAPTURE GUIDANCE**: Automated gap and occlusion capture recommendations
  - **EVIDENCE SOURCES**: Multi-modal thumbnail preview cards (RGB, Depth, PLY, Normals)

### 6. Engineering Status Bar (`components/shell/StatusBar.tsx`)
- Persistent footer (24px)
- Reality Engine identity & version (`v2.3.1`)
- Active Global Mode & Spatial Scale
- Real-time coordinates in EPSG:32645 Projected UTM + WGS84 Geographic Lat/Long + Elevation
- GPU VRAM telemetry + 60 FPS counter
- Dock drawer state toggle

---

## 4. Reusable UI Primitives Catalog (`components/ui/`)

| Primitive | Path | Purpose |
|---|---|---|
| `Modal` | `@/components/ui/modal` | Accessible dialog with backdrop blur, size variants (`sm` to `full`), Escape key trapping |
| `ToastContainer` / `ToastItem` | `@/components/ui/toast` | Notification stack supporting `info`, `success`, `warning`, `error`, `pipeline` |
| `EmptyState` | `@/components/ui/empty-state` | Spatial empty views with icons, descriptive guidance, and action buttons |
| `Skeleton` / `GeometricLoader` / `Spinner` | `@/components/ui/loading-state` | Shimmer placeholders, pulsed geometric loaders, and view busy overlays |
| `ErrorState` | `@/components/ui/error-state` | Honest diagnostic error boxes with error codes, stack trace details, and retry triggers |
| `TimelinePrimitive` | `@/components/ui/timeline-primitive` | Multi-stage pipeline progression bar with status pips, connector wires, and time ticks |
| `ConfidenceGauge` | `@/components/ui/confidence-gauge` | Dual-indicator error & confidence gauges matching reference designs |
| `Button` | `@/components/ui/button` | Styled buttons with variants (`default`, `secondary`, `outline`, `ghost`, `danger`, `accent`) |
| `Badge` | `@/components/ui/badge` | Compact status and category badges |
| `Input` / `Select` / `Checkbox` | `@/components/ui/*` | High-density form and filter inputs |

---

## 5. Parallel Development Contract (Agent 1 & Agent 2)

- **Agent 1 Scope**:
  - Layout framework, application shell, title bar, left navigation panel, context inspector panel, bottom contextual drawer, status bar, command palette, design tokens, and reusable UI primitives.
- **Agent 2 Scope**:
  - Specialized world/capture workflows (e.g., custom camera alignment views, LiDAR-to-camera calibration tools, specialized segmentation painting, mobile capture instruments).
- **Integration Rule**:
  - Agent 2 should consume primitives from `@/components/ui` and tokens from `reds-tokens.css`.
  - Avoid modifying `components/shell/` directly unless adding an explicit top-level route/tool hook.
