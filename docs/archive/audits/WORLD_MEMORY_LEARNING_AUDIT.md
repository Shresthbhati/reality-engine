# World Memory / Learned Representations Audit

**Date:** 2026-09-13. Direct repository inspection, not inference from prior docs.

Scoped, honest snapshot against the world-memory campaign brief. Most phases below are MISSING — this campaign is 91 phases; this document establishes the baseline and records the one real increment built in this session.

| Area | Status | Evidence |
|---|---|---|
| Canonical world state (WorldIR) | IMPLEMENTED | Unchanged from prior audits — entities/geometry/provenance/uncertainty, real validation gate. |
| World versioning / diff | PARTIAL | `world_ir/diff.py` (`diff_worlds`) exists (prior session) — a structural diff between two snapshots. No `WorldVersion` graph, no branch/merge, no semantic-change classification (Phase 60) yet. |
| Entity memory (persistent identity across sessions) | **MISSING before this session, PARTIAL now** | Nothing tracked entity identity across two independent WorldIR snapshots at all. This session added `world_ir/entity_reid.py` — see below. Still missing: an actual persistence layer that stores match history over >2 snapshots, consolidation, decay/staleness. |
| Cross-session entity re-identification (Phase 4) | **PARTIAL (new this session)** | `world_ir/entity_reid.py`: deterministic geometric+type matcher (MATCH/POSSIBLE_MATCH/NO_MATCH/UNRESOLVED) between two WorldIR snapshots. No appearance/embedding signal (see below — deliberately not built, no real embedding model exists in this repo). |
| Semantic embeddings / vector storage (Phases 5-14, 20, 29, 48-50) | **MISSING, deliberately not attempted** | No embedding model of any kind is installed or adapted in this repository. Building a stub/fake embedding layer here would violate the campaign's own explicit rule ("no fake embeddings"). This entire area remains open and requires a real vision-embedding model integration (a separate, larger effort with real license/hardware evaluation, matching `docs/TECHNOLOGY_REGISTRY.md`'s existing rigor for depth/segmentation candidates) before any of Phases 5-14/29/48-50 can be honestly started. |
| Memory conflicts / consolidation (Phases 16-19) | PARTIAL (pre-existing) | `reconstruction/fusion/fusion.py` (prior session) already does inverse-variance fusion with explicit `CONFLICT` provenance and preserved contributing observations for scalar quantities — the right pattern, but scoped to measurements fused within one compile, not repeated-observation consolidation across sessions/time. No staleness/freshness model exists. |
| Retrieval (Phases 8-15, 31, 39, 51, 70) | **MISSING** | `engine/scene_graph/spatial_index.py` (prior session) gives geometric nearest/radius/region queries and `SceneGraph` gives relationship queries — real, but neither is "retrieval" in the campaign's sense (no semantic/hybrid/provenance-filtered query API exists). |
| Knowledge gaps / active learning (Phases 52-56) | MISSING | No gap-detection engine exists. |
| Digital twin memory integration (Phase 85) | MISSING | No twin-health/staleness concept exists anywhere. |

## This session's change

**`world_ir/entity_reid.py`** (Phases 3/4) — cross-session entity re-identification using real, already-existing evidence (`EntityType` + geometric position via the same resolution `SpatialIndex` uses), never a fake or placeholder embedding matcher. Classifies each `before`-world entity against its best same-type geometric candidate in an `after` world as MATCH / POSSIBLE_MATCH / NO_MATCH / UNRESOLVED, with the distance and reasoning attached — "similarity is not identity" (campaign rule 9) is respected by keeping type+distance separate, explicit fields rather than a collapsed score.

Also renamed `engine/scene_graph/spatial_index.py`'s private `_entity_position()` to public `entity_position()` so `entity_reid.py` reuses it instead of duplicating position-resolution logic (avoids exactly the "duplicated abstractions" the campaign audit phase calls out).

**Verification:** `tests/test_entity_reid.py` (11 tests: match/possible-match/no-match/unresolved classification, type-exclusivity, deterministic tie-breaking, determinism, input validation, no-mutation). Full suite: 978 passed, 1 skipped (967 baseline + 11), zero regressions.

## Largest remaining gap

Everything embedding/vector-retrieval-shaped (Phases 5-15, 20-30, 39, 48-51) requires a real learned representation model this repository does not have installed. That is the campaign's central ask and its central blocker: it cannot be honestly built without first doing real model selection/licensing work (the same rigor `docs/TECHNOLOGY_REGISTRY.md` already applies to depth/segmentation candidates), which is out of scope for a single increment.
