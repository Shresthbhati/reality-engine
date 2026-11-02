# 10 — System architecture

## The picture

```
 ┌──────────────────────────── FRONTEND (Next.js, three.js) ────────────────────────────┐
 │ Home · Worlds · Capture · 3D Workstation · Evidence · Reports · Settings              │
 └───────────────────────────────▲──────────────────────────────────────────────────────┘
                                 │ HTTP/JSON + job events   — never imports engine code
 ┌───────────────────────────────┴──────── API (FastAPI, apps/api) ──────────────────────┐
 │ routes: worlds · sessions · uploads · jobs · evidence · measurements · export · query  │
 │ app DB (SQLAlchemy: SQLite dev / Postgres prod)   object storage (content-addressed)   │
 └───────────────▲───────────────────────────────────────────────▲──────────────────────┘
                 │ enqueue / poll                                │ read versions
 ┌───────────────┴──────── WORKER (apps/api/jobs.py loop) ───┐   │
 │ claims jobs, heartbeats, retries, cancel; calls pipeline   │   │
 └───────────────▲────────────────────────────────────────────┘  │
                 │ Python calls                                   │
 ┌───────────────┴───────────── ENGINE LIBRARIES (pure Python, no web) ──────────────────┐
 │ engine/pipeline  ─ orchestrates stages, progressive results, world_delta, guidance     │
 │ evidence/        ─ ingest, sessions, image checks, media metadata                      │
 │ reconstruction/  ─ matching, SfM (COLMAP), dense/MVS, fusion, meshing, scale           │
 │ perception/      ─ per-image models → Observations; architecture fitting               │
 │ registration/    ─ cross-session / new-photo registration                              │
 │ world_ir/        ─ CONTRACT: schema, ledger, association, validation, incremental      │
 │ worldstore/      ─ immutable versioned persistence of WorldIR                          │
 │ exporters/       ─ GLB/glTF, OBJ, PLY, USD, IFC bridge, JSON                           │
 │ uncertainty/ provenance/ events/ engine/core  ─ cross-cutting                          │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

## Layer rules (dependencies point one way: top depends on bottom)

| Layer | May import | Must NOT import |
|-------|-----------|-----------------|
| `engine/core`, `uncertainty`, `provenance`, `events` | stdlib, numpy | anything above |
| `world_ir` | core layer | perception, reconstruction, api |
| `perception`, `reconstruction`, `evidence`, `registration` | core, `world_ir` **types** (Observation, enums) | `worldstore`, `apps`, each other's internals; **world_ir writers** (P3) |
| `worldstore` | core, `world_ir` | perception, reconstruction, apps |
| `engine/pipeline` | everything above it in this table | `apps`, frontend |
| `exporters` | `world_ir`, `worldstore` | perception, reconstruction internals |
| `apps/api` | `engine/pipeline`, `worldstore`, `exporters`, `world_ir` | model backends directly |
| `frontend` | API over HTTP only | Python anything |

Enforce this with an import-boundary test (`tests/test_import_boundaries.py`)
that fails on any forbidden import. That test is how P3 stays true over time.

## Runtime topology

| Process | Started by | Responsibility |
|---------|-----------|----------------|
| API | `start.bat` → uvicorn `apps.api.main:app` (port 8100) | HTTP, validation, app-state DB writes, enqueue jobs. Never runs heavy compute inside a request. |
| Worker | same launcher, separate process | Claim job → run pipeline → write version to WorldStore → mirror row in app DB → emit events. |
| Frontend | `start.bat` → Next.js (port 3100) | UI. Talks to the API only. |
| COLMAP | spawned by the worker as a subprocess | SfM/MVS. Owned by the job; killed on cancel. |

v1 runs everything on one machine. Production later = the same processes in
containers, Postgres instead of SQLite, S3-compatible storage instead of local
disk. Nothing in code should assume "same machine" beyond config.

## Data flow (one reconstruction)

1. Upload → `evidence` stores the original (immutable, content-hashed) → `Evidence` row.
2. User starts reconstruction → `Job(kind=reconstruct)` row.
3. Worker runs the pipeline ([12_PIPELINE.md](12_PIPELINE.md)) stage by stage;
   after each stage it records progress (stage name, real counts) on the job.
4. Perception writes **Observations** to the ledger; association decides entities.
5. The validated WorldIR is committed to **WorldStore** as a new version (parent = previous).
6. The API mirrors the version in `world_versions`; the frontend sees the job
   finish and reloads.

## Where state lives (one source of truth per kind)

| State | Owner | Notes |
|-------|-------|-------|
| World geometry, entities, evidence links | WorldStore (WorldIR versions) | Immutable versions. |
| World metadata (name, status, current version) | App DB `worlds` | Mutable app state. |
| Original photos/videos, meshes, point clouds | Object storage, by sha256 | Referenced by URI. |
| Observations + association decisions | Ledger inside the WorldIR version (v1) | Can move to its own table later; contract unchanged. |
| Jobs, progress, errors | App DB `jobs` | Heartbeat + retry already implemented. |
| UI preferences | browser localStorage | Never authoritative. |

## Cross-cutting rules

- **IDs**: stable, prefixed (`ent-`, `obs-`, `img-`, `ses-`, `wld-`, `ver-`).
  Entity IDs survive re-reconstruction through re-identification.
- **Coordinates**: right-handed, metres when calibrated; frames are defined in
  `docs/architecture/COORDINATE_FRAMES.md` (keep it authoritative).
- **Errors**: typed exceptions from `engine/core/errors.py`; the API maps them
  to clear messages; nothing is swallowed (P14).
- **Logging**: structured logs (`engine/core/logging.py`) with job ID + stage.
- **Config**: `config/` TOML + env overrides; no magic constants inside stages.
