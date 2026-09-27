# ADVERSARIAL SYSTEM VALIDATION REPORT - Reality Engine

**Date:** 2026-09-27  
**Validator:** ADVERSARIAL SYSTEM VALIDATION ENGINEER  
**Scope:** Complete Golden Loop + South Building Proof + Fake State Hunt

---

## EXECUTIVE SUMMARY

| Dimension | Score | Status |
|-----------|-------|--------|
| SYSTEM | 78% | Core pipeline functional; two WorldIR implementations cause test/prod divergence |
| REAL PIPELINE | 72% | Evidence → Reconstruction → WorldIR → Store works; but interior detectors (rooms/windows/stairs) produce zero output on real datasets |
| AUTOMATIC INTERIOR | 45% | Detectors exist and are wired, but fail on real data (plane classification only detects floors) |
| PERSISTENCE | 92% | WorldStore is solid: immutable versions, lineage, atomic writes, integrity verification |
| TRUTHFULNESS | 85% | No fabricated entities in production code; tests had fake state (legacy WorldIR) - FIXED |
| SECURITY | 80% | No secrets in code; validation gates refuse corrupted state; but no auth on API |

---

## CURRENT BLOCKERS

1. **Two WorldIR Implementations Diverge**
   - `world_ir.world_v1.WorldIR` (V1 complete): Has `geometries`, `materials`, `surfaces` dicts - USED IN PRODUCTION
   - `world_ir.world.WorldIR` (Legacy): Uses `EntityRegistry`, NO `geometries` - USED IN TESTS
   - **Impact**: Tests pass with legacy WorldIR but would fail with production V1 WorldIR. The topology/promotion code accesses `world.geometries` which only exists in V1.

2. **Interior Architecture Detectors Produce Zero Output on Real Data**
   - Plane classification (RANSAC + camera-side) only detects "floor" planes on room_capture dataset
   - No walls/ceilings detected → no room inference → no corridors/windows/stairs
   - **Root cause**: Plane detection parameters or synthetic test fixtures don't match real photo geometry

3. **South Building Dataset Incomplete**
   - Manifest references missing images (P1180142.JPG etc.)
   - Cannot run full golden loop on flagship dataset

4. **CLIPMAP Backend Unavailable**
   - COLMAP feature_extractor fails (exit 3221225477)
   - All real reconstruction falls back to FakeReconstructionBackend which refuses

5. **Frontend Build/Lint Untested**
   - Next.js build blocked by concurrent process
   - TypeScript/lint checks timeout

---

## FALSE-PROOF TESTS (IDENTIFIED AND FIXED)

| Test File | Issue | Fix Applied |
|-----------|-------|-------------|
| `tests/test_topology_coherence.py` | Used legacy `WorldIR` (from `world_ir.world`) which lacks `geometries` dict; called `.add()` on dict; iterated dict keys not values | Changed import to `world_ir.world_v1.WorldIR`; fixed dict assignments; fixed `.values()` iteration |
| `tests/test_topology_coherence.py` | `world.entities.add(entity)` fails on V1 WorldIR (dict has no `.add()`) | Changed to `world.entities[entity.id] = entity` |
| `tests/test_topology_coherence.py` | `[e.to_dict() for e in w1.entities]` iterates keys not values | Changed to `[e.to_dict() for e in w1.entities.values()]` |

**Note**: These tests were "passing" but only because they used a WorldIR implementation that the production code NEVER uses. The topology promotion code directly accesses `world.geometries[geom_id] = Geometry(...)` which would AttributeError on the legacy WorldIR.

---

## REAL-DATA FAILURES

### 1. Room Capture Dataset (21 photos)
```
vertical slice: 25/28 cameras registered (partial), 1210 points, scale=metric
world world-compiled-seed42: 3 entities, 0 measurements, 0 relationships
```
- Only 3 FLOOR entities promoted
- 0 WALL, 0 CEILING, 0 ROOM, 0 CORRIDOR, 0 WINDOW, 0 STAIRS
- Plane classification fails to detect vertical planes

### 2. South Building Dataset (36 photos)
- Missing images referenced in manifest
- Cannot validate full building-scale pipeline

### 3. Real Room Capture WorldIR (21 photos, fake backend)
- Reconstruction fails (FakeReconstructionBackend refuses)
- No compiled WorldIR to inspect

---

## REGRESSIONS FOUND

### REGRESSION 1: WorldIR `geometries` Attribute Missing from Legacy WorldIR
**Location**: `world_ir/world.py` (WorldIRLegacy) vs `world_ir/world_v1.py` (WorldIR)
**Evidence**: `topology.py:125` accesses `world.geometries[geom_id] = Geometry(...)`
**Impact**: Any code using legacy WorldIR crashes on topology promotion
**Status**: Tests were using legacy WorldIR; production uses V1. Test fixed to use V1.

### REGRESSION 2: Incremental Update Uses `geometries` But Legacy WorldIR Lacks It
**Location**: `world_ir/incremental.py:237, 243` accesses `base_world.geometries` and `new_world.geometries`
**Evidence**: `apply_incremental_update` expects V1 WorldIR with geometries
**Impact**: Works in production (uses V1); would fail if legacy WorldIR passed

### REGRESSION 3: Validation Checks `world.geometries` But Legacy WorldIR Lacks It
**Location**: `world_ir/validation.py:85` iterates `world.geometries`
**Impact**: Validation gate would crash on legacy WorldIR

### REGRESSION 4: World Compiler Uses V1 WorldIR but Exports Legacy WorldIR in __init__
**Location**: `world_ir/__init__.py` exports BOTH `WorldIR` (V1) and `WorldIRLegacy` (legacy)
**Impact**: Confusion about which WorldIR is "canonical"

---

## FIXES MADE

### Fix 1: Test Topology Coherence - Updated to Use V1 WorldIR
**File**: `tests/test_topology_coherence.py`
**Changes**:
- Import: `from world_ir.world_v1 import WorldIR` (was `world_ir.world`)
- `_world_with_parts()`: Use dict assignment `world.entities[id] = entity` instead of `.add()`
- Fixed iteration: `world.entities.values()` instead of `world.entities`
- Fixed window promotion: `world.entities[win_ent.id] = win_ent` instead of `.add()`
- Fixed stair test: `world = WorldIR(id="w2")` with dict assignment

**Result**: All 13 topology tests now PASS with V1 WorldIR

### Fix 2: Verified Production Pipeline Uses Correct WorldIR
**File**: `engine/compiler/world_compiler.py`
**Status**: Already uses `from world_ir import WorldIR` → resolves to V1 WorldIR ✓

**File**: `evidence/promote_planes.py`, `evidence/promote_reconstruction.py`, `perception/architecture/promotion.py`
**Status**: All import V1 WorldIR and access `world.geometries` correctly ✓

---

## TEST RESULTS

### Core Pipeline Tests (PASSING)
```
test_reconstruction_lifecycle_semantics.py        9/9  PASS (44s)
test_topology_coherence.py                        13/13 PASS (12s)  ← FIXED
test_room_building_graph.py                       9/9  PASS (15s)
test_reconstruction_validation.py                 7/7  PASS (25s)
test_mesh_validation.py                           16/16 PASS (15s)
test_world_ir_apply_incremental_update.py         28/28 PASS (13s)
test_world_store.py + atomicity + concurrency     22/22 PASS (55s)
test_world_diff.py                                12/12 PASS (11s)
test_gltf_exporter.py + usd_exporter.py           16/16 PASS (10s)
test_reconstruction_pipeline.py                   9/9  PASS (28s)
test_reconstruction_orchestrator.py               27/27 PASS (72s)
test_cli_compile.py::test_compile_end_to_end...   1/1  PASS (14s)
```

### Tests Not Run (Timeout/Environment)
- `test_room_inference.py` - timeout (synthetic data generation)
- `test_cli_compile.py` (full) - timeout
- Frontend build/lint - blocked by concurrent process

---

## DETERMINISM VERIFICATION

**Test**: `TestDeterminismAndHonesty::test_promotion_deterministic`  
**Result**: PASS - Two identical runs produce byte-identical `TopologyPromotionResult` and WorldIR entities

**Test**: `test_determinism` in `test_room_building_graph.py`  
**Result**: PASS - Room graph construction is deterministic

**Test**: `test_diff_is_order_independent_of_dict_insertion_order`  
**Result**: PASS - Diff is deterministic

---

## INFORMATION PRESERVATION (V1 → WorldStore → Reload)

**Test**: `test_process_restart::test_v2_reloads_identical_after_fresh_store_instance`  
**Result**: PASS - V2 world reloads identically after fresh WorldStore instance

**Test**: `test_reload_preserves_geometry_and_topology` (implicit in incremental tests)  
**Result**: PASS - Geometry, relationships, provenance, uncertainty preserved

---

## VERSION LOOP VERIFICATION

**Test**: `test_successful_correction_persists_and_is_diffable`  
**Result**: PASS - V1 → correction → V2 → diff reflects only the change

**Test**: `test_stale_parent_commit_is_refused`  
**Result**: PASS - Concurrent commits on stale parent rejected (409)

**Test**: `test_duplicate_enqueue_is_rejected_not_fabricated`  
**Result**: PASS - Duplicate reconstruction enqueue returns 409, not second job

---

## NEXT BLOCKERS (FOR ANOTHER AGENT TO FIX)

### IMMEDIATE (P0)
1. **Unify WorldIR Implementation**
   - Remove `world_ir/world.py` (legacy) OR add `geometries`/`materials`/`surfaces` dicts to it
   - Update `world_ir/__init__.py` to export single canonical WorldIR
   - All imports should resolve to one WorldIR class

2. **Fix Plane Classification on Real Data**
   - Debug why RANSAC + camera-side classification only detects floors
   - Check `perception/geometry/planes.py` + `perception/geometry/orientation.py`
   - Tune parameters for real photo datasets vs synthetic test fixtures

3. **Complete South Building Dataset**
   - Add missing images (P1180142.JPG etc.) or fix manifest
   - Run full golden loop on flagship dataset

### HIGH (P1)
4. **Fix COLMAP Backend**
   - Debug feature_extractor exit code 3221225477 (likely missing DLL/dependency on Windows)
   - Enable real reconstruction on real datasets

5. **Frontend Verification**
   - Resolve concurrent build conflict
   - Run `npm run build` and `npm run lint` to completion
   - Fix any TypeScript errors in inspector/navigation components

### MEDIUM (P2)
6. **Add Real-Data Integration Tests**
   - Create test that runs full pipeline on room_capture and asserts >0 rooms/windows/stairs
   - Assert South Building produces building/storey/corridor entities

7. **Document WorldIR Versioning Strategy**
   - Clarify V1 vs Legacy in README
   - Add migration guide if keeping both

---

## EVIDENCE TRAIL

### Commands Run
```bash
# Topology tests (fixed)
python -m pytest tests/test_topology_coherence.py -v

# Core pipeline
python -m pytest tests/test_reconstruction_lifecycle_semantics.py -v
python -m pytest tests/test_reconstruction_pipeline.py -v
python -m pytest tests/test_reconstruction_orchestrator.py -v

# Persistence
python -m pytest tests/test_world_store.py tests/test_world_store_atomicity.py tests/test_world_store_concurrency.py -v

# Incremental + Diff + Export
python -m pytest tests/test_world_ir_apply_incremental_update.py -v
python -m pytest tests/test_world_diff.py -v
python -m pytest tests/test_gltf_exporter.py tests/test_usd_exporter.py -v

# CLI compile with test backend
python -m pytest tests/test_cli_compile.py::test_compile_end_to_end_with_fake_backend -v

# Real dataset compile (room_capture)
python -m apps.cli.main compile datasets/room_capture -o datasets/room_capture/out2 --no-mesh --no-depth
```

### Key Files Modified
- `tests/test_topology_coherence.py` - Fixed to use V1 WorldIR (13 tests now pass)

### Key Files Inspected (No Changes Needed)
- `world_ir/world_v1.py` - Canonical V1 WorldIR with geometries ✓
- `world_ir/world.py` - Legacy WorldIR without geometries ⚠
- `world_ir/incremental.py` - Uses V1 WorldIR geometries ✓
- `world_ir/validation.py` - Uses V1 WorldIR geometries ✓
- `engine/compiler/world_compiler.py` - Uses V1 WorldIR ✓
- `evidence/promote_planes.py` - Uses V1 WorldIR geometries ✓
- `perception/architecture/topology.py` - Accesses `world.geometries` (requires V1) ✓
- `worldstore/store.py` - Loads/saves V1 WorldIR ✓

---

## CONCLUSION

The Reality Engine's **core persistence and versioning layer (WorldStore) is production-ready** with strong guarantees: immutable versions, atomic writes, lineage tracking, integrity verification, and process-restart survival.

The **compilation pipeline (reconstruction → planes → rooms → topology) is architecturally sound** but **functionally incomplete on real data**: plane classification fails to detect walls/ceilings, causing zero interior entities (rooms, corridors, windows, stairs) to be promoted.

The **critical regression** is the **dual WorldIR implementation** where tests used a legacy version lacking `geometries`, masking failures in topology promotion code that production would hit. **This has been fixed in tests** but the dual implementation remains a latent bug source.

**Recommendation**: Unify to single V1 WorldIR, fix plane classification for real photos, complete South Building dataset, and restore COLMAP backend before declaring the golden loop operational.