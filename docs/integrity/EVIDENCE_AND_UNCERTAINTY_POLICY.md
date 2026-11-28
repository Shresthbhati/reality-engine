# Evidence & Uncertainty Policy — Reality Engine V1

> Coherent policy for evidence quality and uncertainty governance across the reconstructed world.
> Every reconstructed claim is distinguishable as directly observed / geometrically measured / multi-view inferred / model predicted / derived from other geometry / unknown or unresolved.
>
> Part of the KILO integrity mandate (§4: "Evidence and uncertainty governance"). Owner: World Reconstruction Integrity.
> Basis: `world_ir/observations.py`, `world_ir/schema_v1.py` (Measurement / UncertainScalar), `provenance/provenance.py` (Uncertainty), `reconstruction/spatial/uncertainty.py` (chain_sigma, fuse_inverse_variance, consistent, weakest), `world_ir/uncertainty_model.py`, `world_ir/quality_state.py`.

## 0. Governing rule

> Every reconstructed claim carries an explicit epistemic status. Confidence is not confused with geometric accuracy. Missing evidence never becomes positive evidence.

## 1. The six epistemic statuses

**Schema location:** `reconstruction/spatial/uncertainty.py` `model.Status`.

| Status | Meaning | Evidence basis |
|---|---|---|
| `DIRECTLY_OBSERVED` | directly promoted from sensor evidence | raw capture |
| `GEOMETRICALLY_INFERRED` | estimated/inferred from partial geometric evidence | multi-view triangulation, plane fit |
| `SEMANTICALLY_INFERRED` | inferred from semantics/classification | detector label, taxonomy |
| `WEAKLY_SUPPORTED` | some evidence but below confidence threshold | single-view, low overlap |
| `CONFLICTING` | conflicting observations | contradictory depth/label/position |
| `UNKNOWN` | no evidence | absent input |

## 2. Evidence categories (distinguishable per claim)

Every reconstructed claim must be tagged with exactly one of these:

| Category | Definition | Example |
|---|---|---|
| **Directly observed** | Promoted from sensor evidence; raw capture | a pixel observation lifted to 3D |
| **Geometrically measured** | Computed from measured geometry with propagation | plane fit from points, triangulated position |
| **Multi-view inferred** | Inferred from agreement across views | entity identity confirmed in ≥2 views |
| **Model predicted** | Forecast/extrapolation, not yet observed | predicted future position |
| **Derived from other geometry** | Computed from existing geometry statements | room area from corner nodes |
| **Unknown or unresolved** | No evidence; status explicitly UNKNOWN | missing observation, failed depth |

**Contract rule:** `UNKNOWN` propagates as UNKNOWN — a quantity computed from an unknown input is unknown (`None`), never zero and never high confidence. (`reconstruction/spatial/uncertainty.py`: "UNKNOWN propagates as UNKNOWN")

## 3. Confidence vs. geometric accuracy

| Concept | What it means | What it does NOT mean |
|---|---|---|
| Confidence | producer-stated probability in [0,1] | geometric accuracy |
| Precision / sigma | measurement spread (m or rad) | a probability |
| Calibrated | `True` only with measured physical ground truth (RE-GEO-000013) | default |

**Contract rules (mandate §4):**
- **Confidence is not confused with geometric accuracy.** A high-confidence classification can still have large positional uncertainty; a low-confidence position can still be correctly classified.
- **Confidence is per-dimension.** `ConfidenceVector` (11 dims: classification, identity, position, orientation, dimensions, geometry, completeness, material, colour, topology, cross_view). Each dimension is `None` (not measured) or a float in [0,1]. A wall can be known to ~1 cm in position while its class is a single-view guess; or certainly present while its length is only a lower bound. A single scalar hides exactly that.
- **Spread (sigma) and confidence are different facts and are never converted into one another.** A dimension with no sigma is `None`, not 0.
- **A confidence is never upgraded.** The compiler never upgrades a value's status.
- **`None` means "not measured" — never fabricated, never defaulted to 0.5.**

## 4. Sigma (spread) model

**Schema location:** `world_ir/observations.py` `sigma_for()`; `reconstruction/spatial/uncertainty.py` `chain_sigma()`, `fuse_inverse_variance()`, `consistent()`.

### 4a. Per-depth-source sigma

`sigma_for(depth_source, range_m)` = max(SIGMA_FLOOR_M, fraction × range_m):

| Depth source | Fraction of range | Floor |
|---|---|---|
| `METRIC` | 0.02 | 0.05 m |
| `MVS` | 0.03 | 0.05 m |
| `RELATIVE` | 0.25 | 0.05 m |

**Contract rule:** `world_pos=None` means "not lifted" — never a made-up position. `sigma_m` is always ≥ SIGMA_FLOOR_M (0.05 m).

### 4b. Uncertainty propagation

| Operation | Rule | Function |
|---|---|---|
| Independent errors chain | add in quadrature; any unknown (None) → result unknown | `chain_sigma(*sigmas)` |
| Fuse independent (value, sigma) estimates | inverse-variance weighting; estimates without sigma excluded; none weightable → (None, None) | `fuse_inverse_variance()` |
| Consistency check | agree within k sigma; None sigma → undecidable (not "yes") | `consistent(a, b, k, floor)` |
| Weakest status | propagate the weakest status across a set | `weakest(statuses)` |

**Contract rules:**
- Downstream geometry inherits upstream sigma: an opening's across-wall position carries its host's offset sigma; room area carries the first-order propagated sigma of its corner nodes; a corner shared by two walls has the posterior sigma of the joint constraint solve.
- `UncertainScalar` (value + sigma + basis) supersedes legacy `precision`/`confidence` for formal uncertainty propagation (P10-02). `Measurement.precision` is the legacy field.

## 5. Evidence governance

### 5a. Evidence vs. observation

| Concept | Definition |
|---|---|
| Evidence | raw capture an observation traces back to (image, LiDAR scan, etc.) |
| Observation | one 2D detection lifted into the world frame from evidence |

Evidence fields live on the acquisition record; observations link to evidence via `observation.evidence_id` / `data_uri` + `data_hash`.

### 5b. Evidence quality rules

| Rule | Contract |
|---|---|
| Byte-identical evidence deduplicated | first kept; duplicates dropped with reason |
| Garbage "photos" refused | before COLMAP; never mixed into observed geometry |
| Missing evidence | never becomes positive evidence |
| Conflicting observations | retained and reconciled explicitly; never silently resolved to a single value |
| Unsupported geometry | not presented as verified |
| Evidence traceable | to originating image, camera, engine, and processing stage |

### 5c. Evidence state vocabulary

`evidence_state` must use the documented vocabulary (Blueprint invariant 1). OBSERVED / PARTIALLY_OBSERVED entities cite at least one `observation_id` (invariant 2). INFERRED is not also OBSERVED (invariant 3).

## 6. Confidence defaults that are forbidden

| Forbidden default | Why | Correct behavior |
|---|---|---|
| `confidence = 0.5` | manufactured "half-confident" value | `None` = not measured |
| `global_confidence = 0.5` | manufactured global confidence | `None` = not measured |
| `Measurement.precision = 0.01` | manufactured precision | use `UncertainScalar` or leave unset |
| `Transform` uncertainty `confidence = 1.0` | manufactured certainty | `Uncertainty()` = 0.0 unless measured |
| `visibility` defaulted | fabricated assessment | `None` = not assessed |
| `embedding` vector defaulted | fabricated appearance | `embedding_ref` = reference only, never the vector |

## 7. Fusion & conflict handling

| Situation | Contract |
|---|---|
| Multi-source fusion | ≥ 2 sources with ≥ 10 points each; conflicts recorded, never winner-picked |
| Fusion refuses RELATIVE | "fusing unit-less ranges would silently claim meters" |
| Conflicting observations | both hypotheses kept with provenance until a later version agrees |
| Dense MVS conflicts | recorded via `dense_gate.dense_readiness`; never silently resolved |
| Mesh fusion conflicts | recorded, never winner-picked |

## 8. Persistence of uncertainty

| Item | Contract |
|---|---|
| `Uncertainty` serialized | `{"confidence": ..., "note": ..., "calibrated": ...}` |
| `UncertainScalar` | value + optional sigma + mandatory derivation basis |
| Round-trip | uncertainty survives `to_dict()`/`from_dict()` unchanged |
| `None` sigma | serialized as absent; deserialized as None (unknown), never 0 |

**Validation:** `world_ir/uncertainty_model.py` `_clean_conf()` handles None confidence (unmeasured) gracefully. `world_ir/quality_state.py` validates `confidence_bound` in [0.0, 1.0].

## 9. Summary of what must never happen

- ✗ Confidence confused with geometric accuracy
- ✗ Missing evidence becoming positive evidence
- ✗ Conflicting observations silently resolved
- ✗ Unsupported geometry presented as verified
- ✗ Uncertainty lost in fusion or persistence
- ✗ Evidence untraceable to its originating image/camera/engine/stage
- ✗ Unknown values converted into plausible defaults
- ✗ `None` coerced to `0.0` / `0.5` / `1.0`
