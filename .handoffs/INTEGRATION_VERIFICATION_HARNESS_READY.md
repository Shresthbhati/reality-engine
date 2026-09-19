# HANDOFF: INTEGRATION VERIFICATION HARNESS & ADVERSARIAL ACCEPTANCE SUITE

**Agent:** antigravity-integration  
**Branch:** agent/antigravity-integration  
**Checkpoint:** INTEGRATION_VERIFICATION_HARNESS_READY  
**Status:** CHECKPOINT_READY  
**Consumes:**
- `agent/cline-mobile-experience`
- `agent/kilocode-desktop-studio`
- `agent/claude-city-world-core`
- `agent/freebuff-reconstruction-perception`
- `agent/opencode-infrastructure-audit`

---

## 1. Executive Summary

As the independent integration owner and adversarial system verifier for Reality Engine, we have implemented and validated the permanent **Integration & Acceptance Verification Harness** in `tests/integration/` and the formal **Quality State Model & Firewall** in `world_ir/quality_state.py`.

The harness exercises and verifies the complete vertical pipeline across all subsystem boundaries:
$$\text{Mobile} \rightarrow \text{Evidence} \rightarrow \text{Session} \rightarrow \text{Registration} \rightarrow \text{Reconstruction} \rightarrow \text{WorldIR} \rightarrow \text{WorldStore} \rightarrow \text{Query} \rightarrow \text{Desktop} \rightarrow \text{Export}$$

Across **52 adversarial integration tests**, the harness ensures that no subsystem can silently violate contracts, drop provenance, fabricate certainty, mix coordinate frames, leak test doubles/mock data into production, or falsely claim localized incremental compilation.

---

## 2. Test Architecture & Coverage Matrix

All 52 tests run under `pytest tests/integration/` and pass with 100% compliance (**52/52 PASSED** in 38.72s):

| Test Module | Phase / Focus | Test Count | Key Invariants Verified |
| :--- | :--- | :---: | :--- |
| `tests/integration/test_pipeline_matrix.py` | Phase 1 & 2: Pipeline Matrix (Stages A–J) | 14 | Verifies each of the 10 stages (A through J) for valid input/contract adherence, output structure, provenance preservation, uncertainty propagation, coordinate frame discipline, and explicit failure diagnostics. |
| `tests/integration/test_contract_compatibility.py` | Phase 3: Subsystem Contracts | 5 | Validates Cline `EvidencePackage` $\to$ `Session` $\to$ FreeBuff `ReconstructionResult` $\to$ Claude `WorldIR` $\to$ `WorldStore` $\to$ KiloCode Desktop Viewer bridge payload contract. |
| `tests/integration/test_provenance_firewall.py` | Phase 4: Provenance Firewall | 4 | Enforces strict Entity $\to$ Geometry $\to$ EvidenceItem/Asset lineage; rejects `"mock"`, `"demo"`, or `"unknown"` provenance as canonical truth; asserts explicit `UNAVAILABLE` state diagnostics. |
| `tests/integration/test_uncertainty_firewall.py` | Phase 5: Uncertainty Firewall | 4 | Enforces $[0.0, 1.0]$ confidence bounds; non-increasing composed uncertainty (never fabricates certainty); covariance matrix propagation; end-to-end uncertainty preservation from sensor to WorldStore. |
| `tests/integration/test_coordinate_frame_firewall.py` | Phase 6: Coordinate Frame Firewall | 4 | Validates multi-hop coordinate transformation chains; asserts loud failures on frame mismatches; validates rigid inversion identity ($T \cdot T^{-1} = I$); verifies spatial tiling stability across negative coordinates and boundaries. |
| `tests/integration/test_failure_injection.py` | Phase 7: Adversarial Failure Injection | 12 | Tests all 12 canonical failure modes (missing artifact, corrupt artifact, missing reconstruction, invalid WorldIR, invalid provenance, unavailable backend, registration failure, degraded reconstruction, invalid tile parameter, WorldStore immutability violation, malformed evidence package, unsupported export format). Asserting: **EXPLICIT FAILURE + DIAGNOSTICS + NO CORRUPTED WORLD + NO FAKE SUCCESS**. |
| `tests/integration/test_real_synthetic_separation.py` | Phase 8: Real / Synthetic Separation | 8 | Validates 4-state Quality Model (`REAL`, `SYNTHETIC`, `TEST DOUBLE`, `UNAVAILABLE`); real data firewall; synthetic metadata requirements (algorithm, seed/prompt, confidence bound); test double persistence prevention; demo data containment; mixed scene isolation without cross-contamination. |
| `tests/integration/test_golden_world_flow.py` | Phase 9: Golden Vertical Slice Flow | 1 | Complete two-pass end-to-end integration test executing real capture $\to$ Session 1 $\to$ Reconstruction $\to$ World V1 $\to$ WorldStore $\to$ SpatialTiles $\to$ Desktop; followed by mandatory second localized rescan $\to$ Session 2 $\to$ Registration $\to$ `affected_closure()` $\to$ `apply_incremental_update()` $\to$ World V2 $\to$ WorldStore V2 $\to$ `diff_worlds()` $\to$ Desktop comparison. Strictly verifies object identity (`is`) on untouched entities. |

---

## 3. The Quality State Model (`world_ir/quality_state.py`)

Every entity, geometry, point cloud, and pose within the Reality Engine declares one of four explicit quality states:

```python
class QualityState(str, Enum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    TEST_DOUBLE = "TEST DOUBLE"
    UNAVAILABLE = "UNAVAILABLE"
```

### Invariants Enforced:
1. **`REAL`**:
   - Traced directly to physical sensor evidence via verified `EvidencePackage`, monotonic timestamp, and valid camera calibration.
   - Rejects unverified synthetic artifacts or test doubles via `RealSyntheticFirewall.assert_real_evidence()`.
2. **`SYNTHETIC`**:
   - Explicitly generated by procedural generation, simulation, or AI inference.
   - Must be accompanied by `SyntheticMetadata` (`algorithm`, `seed_or_prompt`, `confidence_bound`). Missing or empty parameters raise immediate validation errors.
   - Can never claim `Provenance.OBSERVED` or silent canonical authority.
3. **`TEST DOUBLE`**:
   - Synthesized fixtures or mock files (e.g. `[DEMO] points.ply`, test boxes).
   - Classified automatically by `RealSyntheticFirewall.classify_source_uri()`.
   - Blocked by `RealSyntheticFirewall.assert_safe_for_production()` from ever being persisted to a production `WorldStore` or served as real.
4. **`UNAVAILABLE`**:
   - Explicit representation of absent or uncaptured data.
   - Never represented as empty arrays, null pointers, or zero-filled matrices.
   - Requires an explicit `unavailable_reason`.
5. **Mixed Scene Isolation**:
   - In scenes containing both real and synthetic elements (e.g. a synthetic chair placed in a scanned real room), `RealSyntheticFirewall.verify_scene_isolation()` ensures that no operation promotes the synthetic chair to real or demotes the real room to synthetic.

---

## 4. Golden Vertical Slice & Incremental Compilation Invariants

The harness verifies that:
1. **Pass 1 (Base World)**:
   - Successfully compiles sensor evidence into `WorldIR` V1 and saves version `v-golden-1` in `WorldStore`.
   - Partitions entities across `SpatialTiles` grid and verifies $O(1)$ spatial queries.
   - Serves complete world payloads to Desktop Studio through `api_bridge.cmd_load_world()`.
2. **Pass 2 (Mandatory Incremental Rescan)**:
   - Aligns Session 2 point cloud to Session 1 via `align_session()`.
   - Computes `affected_closure()` targeting only modified region (`room-annex`).
   - Verifies that untouched region (`room-1`) is **not** in the affected closure.
   - Applies localized update via `apply_incremental_update(base_world=v1, ...)`.
   - **Strict Identity Invariant**: `world_v2.entities["room-1"] is world_v1.entities["room-1"]` (true memory reference identity, proving zero re-compilation of untouched entities).
   - Records V2 in `WorldStore` with `parent="v-golden-1"` and lineage `store.parents("v-golden-2") == ["v-golden-1"]`.
   - Calculates `diff_worlds(world_v1, world_v2)` and confirms diff contains only `room-annex`, with zero phantom diffs on `room-1`.
   - Validates live diff consumption via `api_bridge.cmd_diff("v-golden-1", "v-golden-2")`.

---

## 5. How Downstream Agents Consume This Harness

Any agent can run the integration suite to verify their branch against the whole system:

```bash
# Run full integration harness (52 tests)
pytest -v -p no:asyncio tests/integration/

# Run specific firewall checks
pytest -v -p no:asyncio tests/integration/test_real_synthetic_separation.py
pytest -v -p no:asyncio tests/integration/test_provenance_firewall.py
pytest -v -p no:asyncio tests/integration/test_failure_injection.py

# Run golden two-pass incremental vertical slice
pytest -v -p no:asyncio tests/integration/test_golden_world_flow.py
```
