# Reality Engine — Onboarding Guide

*Auto-generated from the project knowledge graph (commit `2e2750b6`, analyzed 2026-11-05).*

---

## Project Overview

**Reality Engine — Professional Spatial Computing Workstation**

Reality Engine turns captured evidence (overlapping photos of a place) into a versioned 3D world. Photos go through COLMAP-based reconstruction into a canonical **WorldIR** with per-entity provenance and uncertainty; adding more photos improves the *same* world as a new version you can diff, query and export (glTF / USD / Blender / IFC). A FastAPI engine and a Next.js **Studio** sit on top of the same pipeline the `reality` CLI drives.

| | |
|---|---|
| **Languages** | batch, config, css, dockerfile, html, javascript, json, mako, markdown, powershell, python, toml, txt, typescript, unknown, yaml |
| **Frameworks** | Docker, GitHub Actions, Next.js, React, Tailwind CSS, Zustand |
| **Entry doors** | `reality` CLI (`apps/cli/main.py`) and FastAPI engine (`apps/api/main:app`) |
| **Canonical artifact** | WorldIR (versioned 3D world schema in `world_ir/`) |
| **Graph freshness** | Graph commit == `git rev-parse HEAD` ✓ (no uncommitted source changes) |

---

## Architecture Layers

The codebase is organized into 10 architectural layers. Each layer below lists its purpose and the key files new developers should read first.

| Layer | Purpose | Key files |
|---|---|---|
| **API Engine** | FastAPI backend: REST routes, job queue, models, storage, auth, and world lifecycle services. | `apps/api/main.py`, `apps/api/models.py`, `apps/api/jobs.py`, `apps/api/worldstore_service.py`, `apps/api/routes_worlds.py`, `apps/api/reconstruction_state.py` |
| **CLI** | Reality command-line interface: ingest, reconstruct, compile, run, register, query, store, and viewer commands. | `apps/cli/main.py`, `apps/cli/mobile_bridge.py`, `apps/cli/api_bridge.py` |
| **Studio Frontend** | Next.js Studio: React pages, components, API route handlers, client library, and mobile shell. | `frontend/src/lib/api/client.ts`, `frontend/src/app/layout.tsx`, `frontend/src/components/shell/AppShell.tsx`, `frontend/src/lib/api/reconstructions.ts` |
| **World Model & Persistence** | Canonical WorldIR schema, serialization, validation, diff, temporal state, and the worldstore persistence layer. | `world_ir/schema_v1.py`, `world_ir/world_v1.py`, `world_ir/diff.py`, `world_ir/temporal.py`, `worldstore/store.py`, `worldstore/tiles.py` |
| **Reconstruction & Registration Pipeline** | COLMAP-based SfM/MVS reconstruction, dense pipeline, meshing, texturing, scale/georeferencing, and cross-session registration. | `reconstruction/orchestrator.py`, `reconstruction/backend/colmap_backend.py`, `reconstruction/pipeline/vertical_slice.py`, `reconstruction/scale/base.py`, `reconstruction/frame.py`, `registration/registration.py` |
| **Perception** | Depth estimation, detection, segmentation, instance reasoning, architecture understanding, materials, and spatial reasoning backends. | `perception/spatial_reasoning.py`, `perception/detection/maskrcnn_backend.py`, `perception/architecture/scene.py`, `perception/architecture/room_graph.py`, `perception/detail/pipeline.py` |
| **Generation** | Procedural grammars (apartment, building, district, room, urban), layout solver, and synthetic scene/data generation. | `procedural/layout_solver.py`, `procedural/apartment_grammar.py`, `procedural/building_grammar.py`, `synthetic/scene.py`, `synthetic/vio.py` |
| **Evidence & Data** | Evidence sessions, datasets, importers (photos/video/point clouds), dedup, promotion (planes/rooms/reconstruction), and multi-source capture. | `evidence/session.py`, `evidence/packages.py`, `evidence/importers.py`, `evidence/multi_source.py`, `evidence/promote_rooms.py`, `evidence/dedup.py` |
| **Engine Core & Trajectory** | Core engine modules (math, logging, config, jobs, scene graph, render, world), compiler pipeline, and trajectory backends/optimization. | `engine/pipeline/vertical_slice.py`, `engine/compiler/world_compiler.py`, `engine/core/logging.py`, `engine/core/config.py`, `engine/scene_graph/graph.py`, `trajectories/trajectory.py` |
| **Tooling & Infrastructure** | Python SDK, scripts, benchmarks, desktop/Electron, offline viewer, docs, config, Dockerfiles, CI workflows, root-level files. | `sdk/reality.py`, `Dockerfile`, `Dockerfile.frontend`, `pyproject.toml`, `scripts/run_vertical_slice.py`, `apps/viewer/build_viewer.py` |

The remaining 2 layers (CLI, Studio Frontend) are covered above. Full node lists live in `.ua/intermediate/layers.json`.

---

## Key Concepts

- **WorldIR** — The canonical versioned 3D world schema (`world_ir/schema_v1.py`, `world_ir/world_v1.py`). Every pipeline converges on this. Worlds can be branched, diffed, and versioned over time.
- **Evidence-first provenance** — Every observation carries provenance and uncertainty (`provenance/__init__.py`). Nothing fabricates geometry; backends that have nothing real to offer fail loudly rather than invent data.
- **`vertical_slice()`** — The single end-to-end pipeline: evidence items → COLMAP SfM → metric scale anchoring → frame canonicalization → [opt.] depth → compile → validated WorldIR → [opt.] perception/mesh/dense MVS/fusion/detail (`engine/pipeline/vertical_slice.py`).
- **Honest scale anchoring** — A world is METRIC only when a measured baseline registers; otherwise it stays RELATIVE (unit-less), honestly recorded in `world.metadata["scale"]` (`reconstruction/scale/base.py`).
- **Semantic perception** — Optional Mask R-CNN detection lifts masks through metric depth into 3D object hypotheses, merged across views and promoted into WorldIR entities (`perception/instances/association.py`, `evidence/promote_objects.py`).
- **Job queue & orphan recovery** — Background jobs run the vertical slice; orphaned jobs are recovered on API startup (`apps/api/jobs.py`, `reconstruction/proc.py`).
- **Deltas & correction** — `world_delta.py` computes deltas between world snapshots and reconciles camera/geometry conflicts; an autonomous correction loop exists (`engine/quality/autonomous_correction_loop.py`).

---

## Guided Tour (recommended learning path)

Follow these 12 steps, in order, to go from "what is this?" to "I understand how it works."

| # | Step | Entry point |
|---|---|---|
| 1 | Project overview | `document:README.md` |
| 2 | 60-second start: photos in, 3D world out | `document:QUICKSTART.md` |
| 3 | Entry point: the `reality` CLI | `file:apps/cli/main.py` |
| 4 | Entry point: FastAPI engine | `file:apps/api/main.py` |
| 5 | Dependency chain: ingest evidence → COLMAP reconstruction | `file:apps/cli/main.py` → `file:evidence/session.py` → `file:reconstruction/backend/colmap_backend.py` → `file:reconstruction/orchestrator.py` |
| 6 | Dependency chain: reconstruct → compile to WorldIR | `file:engine/pipeline/vertical_slice.py` → `file:engine/compiler/world_compiler.py` |
| 7 | WorldIR: canonical versioned 3D world schema | `file:world_ir/schema_v1.py` → `file:world_ir/world_v1.py` → `file:world_ir/validation.py` |
| 8 | World persistence: versioning, diff and temporal state | `file:worldstore/store.py` → `file:world_ir/diff.py` → `file:world_ir/temporal.py` |
| 9 | Provenance, logging and job queue | `file:provenance/__init__.py` → `file:engine/core/logging.py` → `file:apps/api/jobs.py` |
| 10 | Frontend Studio: Next.js API client and shell | `file:frontend/src/lib/api/client.ts` → `file:frontend/src/app/layout.tsx` |
| 11 | Ops: Docker, CI and configuration | `service:Dockerfile` → `pipeline:.github/workflows/ci.yml` → `config:pyproject.toml` |
| 12 | Conclusion: how the pieces fit together | `document:ARCHITECTURE.md` |

---

## File Map (by layer)

Below is a representative file map — key files, what they do, and their complexity. The full node inventory (828 file-level nodes) is in `.ua/intermediate/scan-result.json`.

### API Engine (`apps/api/`)
| File | What it does | Complexity |
|---|---|---|
| `apps/api/main.py` | FastAPI app: middleware, security headers, startup/shutdown, job worker | moderate |
| `apps/api/models.py` | SQLAlchemy ORM models (worlds, versions, sessions, evidence, jobs) | complex |
| `apps/api/jobs.py` | Job routes: listing, detail, SSE events, cancellation | moderate |
| `apps/api/worldstore_service.py` | WorldStore adoption/reconciliation service | complex |
| `apps/api/world_lifecycle.py` | World lifecycle state derivation | moderate |
| `apps/api/reconstruction_state.py` | User-facing reconstruction state & failure messages | moderate |
| `apps/api/routes_worlds.py` | World CRUD + diff/commit/restore/archive/duplicate/versions | complex |
| `apps/api/routes_reconstructions.py` | Reconstruction routes | complex |
| `apps/api/routes_proposals.py` | World/pipeline propose/accept/reject | moderate |
| `apps/api/routes_export.py` | World export formats & downloads | moderate |
| `apps/api/routes_evidence_quality.py` | Evidence quality keep/discard decisions | moderate |
| `apps/api/routes_intent.py` | World intent, hardware detection, time estimates | moderate |
| `apps/api/routes_jobs.py`, `routes_sites.py`, `routes_query.py`, `routes_measurements.py`, ... | Additional API surface | moderate |

### CLI (`apps/cli/`)
| File | What it does | Complexity |
|---|---|---|
| `apps/cli/main.py` | `reality` CLI: ingest/session/reconstruct/compile/run/register/query/store/viewer | complex |
| `apps/cli/mobile_bridge.py` | Mobile bundle ingest + capture-task derivation/merge | complex |
| `apps/cli/api_bridge.py` | CLI-to-API bridge | moderate |

### Studio Frontend (`frontend/`)
| File | What it does | Complexity |
|---|---|---|
| `frontend/src/app/layout.tsx` | Root layout / shell | moderate |
| `frontend/src/lib/api/client.ts` | Typed HTTP client (base URL/auth, ApiError, retries) | moderate |
| `frontend/src/lib/api/reconstructions.ts` | Reconstruction API layer + `useWorldStatus` polling hook | moderate |
| `frontend/src/components/navigation/WorldNavPanel.tsx` | World navigation panel | complex |
| `frontend/src/components/inspector/AdaptiveInspector.tsx` | Adaptive inspector | complex |
| `frontend/src/components/viewport/World3DViewport.tsx` | 3D viewport (Three.js) | complex |
| `frontend/src/components/home/HomeDashboard.tsx` | Home dashboard | complex |
| `frontend/src/components/shell/CommandPalette.tsx` | Command palette | moderate |
| `frontend/src/components/workspace/SpatialWorkstation.tsx` | Spatial workstation | complex |
| `frontend/src/components/reconstruct/GuidedFlow.tsx` | Guided reconstruct flow | moderate |

### World Model & Persistence (`world_ir/`, `worldstore/`)
| File | What it does | Complexity |
|---|---|---|
| `world_ir/schema_v1.py` | Core v1 schema (enums, Vector3, Geometry, Material, Observation, confidence) | complex |
| `world_ir/world_v1.py` | WorldIR root model (branches, scenarios, simulation, temporal state) | complex |
| `world_ir/diff.py` | Structured world diffs | complex |
| `world_ir/temporal.py` | Temporal epochs & change classification | moderate |
| `world_ir/validation.py` | Validation suite (geometry, transforms, provenance, confidence) | complex |
| `worldstore/store.py` | Versioned on-disk world store (sequence locking, atomic writes) | complex |
| `worldstore/tiles.py` | Tiled/partitioned version storage | moderate |
| `worldstore/proposals.py` | Change proposal pipeline | moderate |

### Reconstruction & Registration Pipeline (`reconstruction/`, `registration/`)
| File | What it does | Complexity |
|---|---|---|
| `reconstruction/orchestrator.py` | Backend probe → accept → execute → fallback → provenance stamp | complex |
| `reconstruction/backend/colmap_backend.py` | COLMAP sparse reconstruction backend | complex |
| `reconstruction/pipeline/vertical_slice.py` | End-to-end vertical slice | complex |
| `reconstruction/scale/base.py` | Metric scale anchoring | complex |
| `reconstruction/scale/metric_anchor.py` | Metric scale fusion | moderate |
| `reconstruction/frame.py` | Frame canonicalization (up axis +Y) | moderate |
| `reconstruction/fusion/fusion.py` | Multi-source fusion | moderate |
| `reconstruction/meshing/mesh.py` | Surface mesh (Poisson) | moderate |
| `registration/registration.py` | Registration engine | complex |
| `registration/cross_session.py` | Cross-session registration | complex |
| `reconstruction/scene_identity.py` | Scene identity / pairwise verification | moderate |

### Perception (`perception/`)
| File | What it does | Complexity |
|---|---|---|
| `perception/spatial_reasoning.py` | Spatial relationships (on/adjacent/attached) | moderate |
| `perception/detection/maskrcnn_backend.py` | Mask R-CNN detection backend | moderate |
| `perception/architecture/scene.py` | Interior scene assembler | moderate |
| `perception/architecture/room_graph.py` | Room graph + multi-storey building graph | moderate |
| `perception/architecture/promotion.py` | Promote components → WorldIR entities | moderate |
| `perception/detail/pipeline.py` | Detail discovery + ROI refinement | moderate |
| `perception/depth/midAS_backend.py` | MiDaS depth backend | moderate |
| `perception/instances/association.py` | Cross-view hypothesis association | moderate |
| `perception/quality/assessment.py` | Evidence quality assessment | moderate |
| `perception/reasoning/physical_plausibility.py` | Collision / floating-object validation | moderate |

### Evidence & Data (`evidence/`)
| File | What it does | Complexity |
|---|---|---|
| `evidence/session.py` | Multi-source capture session | moderate |
| `evidence/packages.py` | Content-addressed evidence packages | complex |
| `evidence/importers.py` | Photo/video/LAS/PLY/PCD importers + EXIF | complex |
| `evidence/multi_source.py` | Multi-source session (sources/components/composite detection) | complex |
| `evidence/promote_rooms.py` | Room detection from wall lines → entities | moderate |
| `evidence/dedup.py` | Deduplication | moderate |
| `evidence/video.py` | Video keyframe ingestion | moderate |
| `evidence/cad_importer.py` | DXF floor plan import | moderate |

### Engine Core & Trajectory (`engine/`, `trajectories/`)
| File | What it does | Complexity |
|---|---|---|
| `engine/pipeline/vertical_slice.py` | End-to-end vertical slice | complex |
| `engine/compiler/world_compiler.py` | Compile reconstruction → validated WorldIR | complex |
| `engine/compiler/building_envelope.py` | Building envelope/storey levels | moderate |
| `engine/compiler/topology_graph.py` | Vertical topology graph | moderate |
| `engine/core/logging.py` | Logging (top fan-in node) | moderate |
| `engine/core/config.py` | Configuration | moderate |
| `engine/scene_graph/graph.py` | Scene graph | moderate |
| `engine/math/math3.py` | 3D math | moderate |
| `trajectories/trajectory.py` | Trajectory model | moderate |
| `trajectories/backend/orb_slam3_backend.py` | ORB-SLAM3 backend | moderate |

### Tooling & Infrastructure (`sdk/`, `scripts/`, `exporters/`, root)
| File | What it does | Complexity |
|---|---|---|
| `sdk/reality.py` | Public Python SDK | moderate |
| `Dockerfile` | Multi-stage slim runtime image | moderate |
| `Dockerfile.frontend` | Next.js Studio production image | moderate |
| `pyproject.toml` | Python project config | moderate |
| `scripts/run_vertical_slice.py` | Vertical-slice runner script | moderate |
| `apps/viewer/build_viewer.py` | Offline viewer builder | moderate |
| `benchmarks/reconstruction_benchmark.py` | Reconstruction benchmark | moderate |
| `exporters/ifc_bridge.py` | IFC export | moderate |
| `exporters/gltf/exporter.py`, `exporters/usd/exporter.py` | glTF / USD exporters | moderate |

---

## Complexity Hotspots

Approach these files with extra care — they carry the most structural weight or are most likely to contain subtle behavior:

**Highest structural complexity (`complex`):**
- `apps/cli/main.py` — the top code entry point; every subcommand is a direct call into SDK/orchestrator
- `apps/api/models.py` — SQLAlchemy ORM core (worlds, versions, sessions, evidence, jobs)
- `apps/api/routes_worlds.py` — 2129 lines; world CRUD + lifecycle
- `apps/cli/mobile_bridge.py` — mobile bundle ingest + task merge logic
- `engine/pipeline/vertical_slice.py` — the end-to-end pipeline; all optional stages (depth/perception/mesh/dense/fusion/detail)
- `engine/compiler/world_compiler.py` — plane/room promotion + validation gate
- `world_ir/schema_v1.py` / `world_ir/world_v1.py` — canonical schema root
- `world_ir/diff.py` — structured world diffs
- `worldstore/store.py` — versioned on-disk store
- `reconstruction/orchestrator.py` — backend selection/failure fallback logic
- `reconstruction/backend/colmap_backend.py` — COLMAP subprocess orchestration
- `registration/registration.py` / `registration/cross_session.py` — registration math
- `frontend/src/components/navigation/WorldNavPanel.tsx` — 3063 lines; shell navigation
- `frontend/src/components/inspector/AdaptiveInspector.tsx` — 2073 lines; adaptive inspector
- `frontend/src/components/viewport/World3DViewport.tsx` — 1309 lines; 3D viewport
- `frontend/src/components/home/HomeDashboard.tsx` — 629 lines; home dashboard

**Large but moderate** (still significant surface area):
- `evidence/clocks.py` (1103 lines), `evidence/importers.py`, `evidence/multi_source.py`
- `apps/api/worldstore_service.py` (579 lines), `apps/api/routes_reconstructions.py` (465 lines)
- `engine/technologies/adapters.py` (860 lines), `engine/technologies/catalog.py` (771 lines)
- `reconstruction/meshing/repair.py` (694 lines)

---

## Getting Started

1. **Read the tour in order** (steps 1 → 12 above). Start with `README.md` and `QUICKSTART.md`.
2. **Run the CLI:** `python apps/cli/main.py --help` (or `reality --help` if installed).
3. **Start the API:** `uvicorn apps.api.main:app --port 8100` then visit `http://localhost:8100/api/health`.
4. **Start the Studio:** `cd frontend && npm run dev` (port 3000). Set `REALITY_API_TOKEN` for authenticated access.
5. **Explore the knowledge graph:** The dashboard is available at the tokenized URL from `/understand-dashboard`, viewing `.ua/knowledge-graph.json`.

---

## Recommended Next Steps

- Save this guide to `docs/UA_ONBOARDING.md` and commit it to the repo for the team.
- Re-run `/understand` whenever you want the graph refreshed (e.g. after a significant feature lands).
- Re-run `/understand-onboard` to regenerate this guide from an updated graph.
