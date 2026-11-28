# Production-Path Reachability Report — Reality Engine V1

Automated static analysis of stage execution, module reachability, output/consumer pairing,
and feature-flag-gated release functionality. Generated from source inspection (not runtime).

## 1. Default API job stage sequence (`apps/api/jobs.py`)

Segments (from `seg_*` handlers): evidence_resolve, evidence_quality, reconstruct, compile_validate, changeset_commit

Reconstruct segment drives `run_progressive` (engine/pipeline/progressive.py) whose fine stages run
through `engine/pipeline/stages/ladder.run_fine_stages`: pose -> scale -> depth -> layout -> perception
-> mesh -> dense -> detail -> finalize, reported via `LadderReport` inside the reconstruct rung.

Coarse stage runner (`engine/pipeline/stages/runner.run_stages`) is used via `registry.build_stages`
with profile `interior_survey_standard` (default).

## 2. Module reachability (production callers)

| Module | Symbol searched | Production callers | Status |
|---|---|---|---|
| `engine/orchestrator/universal_pipeline.py` | `universal_pipeline` | engine\orchestrator\__init__.py | REACHABLE |
| `perception/engine_orchestrator.py` | `engine_orchestrator.MultiEngineOrchestrator` | perception\depth_adapter.py, perception\detection_adapter.py, perception\engine_adapters.py, perception\engine_matrix_bridge.py | REACHABLE |
| `apps/api/evidence_world_stage.py` | `evidence_world_stage.run_for_job` | apps\api\routes_evidence_world.py | OPT-IN ONLY |
| `engine/pipeline/stages/s14_spatial_global.py` | `s14_spatial_global / SpatialGlobal` | engine\pipeline\stages\registry.py | OPT-IN ONLY |
| `engine/multiengine/spatial_intelligence/association.py` | `spatial_intelligence.association` | engine\multiengine\spatial_intelligence\__init__.py, engine\multiengine\spatial_intelligence\active_capture.py, engine\multiengine\spatial_intelligence\integration.py | REACHABLE |
| `engine/multiengine/association.py` | `engine.multiengine.association` | apps\api\routes_evidence_world.py, engine\multiengine\canonical.py, engine\multiengine\conflicts.py, engine\multiengine\evidence_flow.py | REACHABLE |

## 3. Stage output pairing (`engine/pipeline/stages/registry.py`)

Stages declaring `requires_outputs`: 0

`run_stages` enforces: a succeeded/degraded stage missing a declared `requires_outputs` is failed.
`s11_compile_validate.requires_outputs=('world_ir',)` — world IR is content-addressed and persisted.

## 4. Feature flags disabling intended release functionality

| Flag | Effect |
|---|---|
| `RE_EVIDENCE_WORLD` | Disables evidence_world_stage (Gemini detection/segmentation/spatial lift/association bridge). Not on default path. |
| `RE_SPATIAL_GLOBAL` | Omits s14_spatial_global from default profile; spatial intelligence off canonical path. |
| `REALITY_DENSE_AUTO` | Evidence-gated dense escalation ON by default but dense_mvs_enabled=False; Level 3 dense off default path. |
| `REALITY_OPEN_VOCAB` | Open-vocab furniture detection (Grounding DINO) — not on live API path (RELEASE.md). |
| `REALITY_TEST_BACKEND` | Refused unless REALITY_ENV=test (enforced in jobs.py:789). Test seam only. |

## 5. Reachability gap summary

- **IG-1**: Three competing orchestrators (`vertical_slice`, `progressive`, `universal_pipeline`); `universal_pipeline` is UNWIRED.
- **IG-2**: Stages (`vertical_slice.run_*`) invoked both inside and outside `run_stages`.
- **IG-3**: Cache keys hash all accumulated `ctx.artifacts` instead of declared `spec.inputs` — FIXED in `runner.py`.
- **IG-4**: Timed-out stage threads abandoned; may mutate shared state.
- **IG-5**: Evidence-world integration is opt-in (`RE_EVIDENCE_WORLD=1`).
- **IG-6**: Spatial intelligence (`s14_spatial_global`) not in default profile.
- **IG-9**: Four association + four taxonomy implementations with inconsistent contracts.
- **IG-10**: API / CLI / benchmark execute different pipeline subsets.

## 6. Method
Static grep/AST over `engine/`, `reconstruction/`, `perception/`, `world_ir/`, `worldstore/`, `apps/`. No runtime execution.