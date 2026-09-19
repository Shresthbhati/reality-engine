# INTEGRATION DEFECT REPORT: NON-LOCALIZED INCREMENTAL UPDATE PIPELINE

- **Defect ID**: `INCREMENTAL_E2E_DEFECT_001`
- **Severity**: HIGH (Architecture & Semantic Invariant Gap)
- **Reporting Agent**: Antigravity (Continuous Integration & System Operator)
- **Impacted Subsystems**: `engine/compiler.py`, `world_ir/incremental.py`, `scripts/test_incremental_world_flow_e2e.py`
- **Primary Owner**: Claude (`agent/claude-city-world-core` / WorldOS Owner)
- **Status**: RESOLVED (Verified by `WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY` & `INCREMENTAL_E2E_VALIDATED`)
- **Resolution Commit**: ad9121b (Merge checkpoint WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY)
- **Validation Suite**: `tests/test_true_localized_incremental_acceptance.py`, `scripts/test_incremental_world_flow_e2e.py`

---

## 1. Executive Summary

Following an adversarial integration audit of `scripts/test_incremental_world_flow_e2e.py`, we have verified Claude's audit finding:

> **The existing incremental E2E flow proves that two separately reconstructed WorldIRs can be versioned in WorldStore with parent-child lineage and diffed, but does NOT prove true localized incremental compilation.**

Specifically, Version 2 (V2) in the current test harness is **Category B** (a full global re-compilation of combined points that happens to be saved as V2 and diffed), rather than **Category A** (a genuinely localized incremental update).

Furthermore, `world_ir.incremental.affected_closure()` currently has **zero production callers** across `engine/compiler.py`, `apps/cli/main.py`, and `sdk/reality.py`.

---

## 2. Audit Findings & What Was Proven vs. Not Proven

### A. What Was Proven (Working as Specified)
1. **WorldStore Immutability & Lineage**:
   - Versions cannot be overwritten; saves append to an immutable sequence (`WorldStoreError: version already exists`).
   - Lineage pointers (`parent`, `parents()`, `ancestors()`) survive process restarts.
2. **Deterministic Snapshot Diffing**:
   - `world_ir.diff.diff_worlds(world_v1, world_v2)` correctly detects differences across entities and geometries.
3. **CLI & API Bridge Services**:
   - `reality diff --store <dir> <v1> <v2>` and `/api/world/diff?base=<v1>&head=<v2>` reliably compute and return NDJSON diff payloads.
4. **Desktop Studio Differential Visualization**:
   - `WorldDiff.tsx` correctly consumes `/api/world/diff` and renders color-coded cards (`ADDED`, `REMOVED`, `MODIFIED`) with honest `DIFF: UNAVAILABLE` error diagnostics adhering to the Real-Data Firewall.

### B. What Was NOT Proven (The Semantic Gap)
1. **No Genuine Localization**:
   - `_build_pass2_reconstruction()` in `scripts/test_incremental_world_flow_e2e.py` pools all points from Room 1 and Room 2 and invokes `compile_reconstruction_to_world()` over the entire point cloud.
   - Plane detection (`detect_planes`) and room inference (`detect_rooms`) run globally from scratch.
2. **Untouched Entities Are NOT Reused**:
   - Rather than keeping Room 1 entities untouched, the global recompile re-partitions points and shifts inlier thresholds.
   - In the diff output:
     ```
     Step 8 [WorldDiff]: Added: 2, Modified: 7, Removed: 1
     ```
     Out of 8 original entities in Room 1, **7 were modified and 1 was removed**. Zero entities were preserved untouched.
3. **Untouched Geometries Are NOT Reused**:
   - Geometries are re-generated and re-hashed by the compiler rather than reused by content digest.
4. **Zero Production Usage of `affected_closure()`**:
   - `world_ir.incremental.affected_closure()` is tested only in unit tests (`tests/test_world_ir_incremental.py`); it is never called during compilation or store updates.
5. **No Spatial Invalidation**:
   - No spatial tile or chunk invalidation occurs in `SpatialIndex` or `SpatialTiling`.

---

## 3. Exact Failing Path & Reproduction

### Reproduction Command
```bash
python scripts/test_incremental_world_flow_e2e.py
```

### Trace & Instrumentation Analysis
In `scripts/test_incremental_world_flow_e2e.py`:
```python
# Pass 2 constructs full combined point cloud:
recon_pass2 = _build_pass2_reconstruction()

# Compiler re-runs full global planar segmentation and room clustering:
world_v2, diag_v2 = compile_reconstruction_to_world(
    recon_pass2, CompileOptions(seed=42, artifact_store=artifact_store)
)
```

Inspecting `world_diff = diff_worlds(world_v1, world_v2)` shows:
- Expected for localized update:
  - `entities_added`: 2 (new annex walls/structures)
  - `entities_modified`: 0 (room 1 is untouched)
  - `entities_removed`: 0
- Actual measured reality:
  - `entities_added`: 2
  - `entities_modified`: 7 (room 1 walls and planes were perturbed by global plane detector)
  - `entities_removed`: 1 (doorway segment merged differently)

---

## 4. Required Fix

### Assigned Owner: Claude (`agent/claude-city-world-core`)
Implement **`WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY`**:

1. **Incremental Update Function**:
   Provide a canonical function in WorldOS / compiler:
   ```python
   def update_world_localized(
       base_world: WorldIR,
       new_evidence: ReconstructionResult | Sequence[Entity],
       *,
       artifact_store: ArtifactStore,
   ) -> LocalizedUpdateResult:
   ```
2. **Affected Closure Integration**:
   - Identify directly touched entities/regions from `new_evidence`.
   - Compute `closure = affected_closure(base_world, touched_ids, relationship_kinds=DEFAULT_PROPAGATING_KINDS)`.
3. **Subgraph Grafting & Object Reuse**:
   - For all $e \in \text{base\_world.entities} \setminus \text{closure}$: preserve entity and geometry objects verbatim.
   - For all $e \in \text{closure}$: re-evaluate, update, or add entities.
4. **Spatial Invalidation Tracking**:
   - Report invalidated spatial tiles/chunks alongside the updated `WorldIR`.

---

## 5. Verification Protocol Post-Fix

Once `WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY` is published in `.handoffs/`:
1. Consume the checkpoint on `agent/antigravity-integration`.
2. Update `scripts/test_incremental_world_flow_e2e.py` and `tests/test_incremental_world_e2e.py` to use `update_world_localized`.
3. Assert the 10 Critical Invariants:
   - [ ] 1. Untouched entities are reused (`v2.entities[id] == v1.entities[id]`)
   - [ ] 2. Untouched geometries are reused (identical artifact digests)
   - [ ] 3. Unrelated tiles are not invalidated
   - [ ] 4. Changed entities have correct lineage
   - [ ] 5. Provenance is preserved
   - [ ] 6. Uncertainty is preserved / updated
   - [ ] 7. V1 remains immutable on disk
   - [ ] 8. V2 is deterministic
   - [ ] 9. Process restart preserves lineage
   - [ ] 10. Failure during update leaves V1 intact
4. Publish `.handoffs/INCREMENTAL_E2E_VALIDATED.md`.
