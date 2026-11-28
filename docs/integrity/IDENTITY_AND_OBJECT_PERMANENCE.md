# Identity & Object Permanence — Reality Engine V1

> Authoritative rules for multi-view object identity: how the system decides that two observations represent the same physical object, cross-view identity, re-identification, occlusion handling, duplicate suppression, object splitting/merging, persistent entity identifiers, identity confidence, contradiction handling, and version-to-version identity continuity.
>
> Part of the KILO integrity mandate (§5: "Multi-view identity and object permanence"). Owner: World Reconstruction Integrity.
> Basis: `world_delta.py`, `world_ir/entity_reid.py` (carry_identity), `perception/instances/association.py` (resolve_hypotheses, EntityTrack), `world_ir/diff.py` (diff_worlds), `reconstruction/frame.py` (canonicalize_like_head), `worldstore/lifecycle.py`.

## 0. Governing rule

> One physical object = one entity however many photos show it. A detection is an observation, never an entity (BUILD_SPEC rule 2). Do not merge entities solely because their class labels match.

## 1. Entity identity contract

**Schema location:** `world_ir/schema_v1.py` `Entity.id`, `Entity.type`, `Entity.geometry_ids`, `Entity.relationships`; `world_ir/entity_reid.py` `carry_identity`.

| Field / concept | Contract |
|---|---|
| Entity id | `ent-<uuid>`; re-derived each rebuild — **NOT a stable match key** |
| Identity continuity | by TYPE + POSITION in the previous frame (`world_delta.entity_records` / `entity_continuity`) |
| One physical object | = one entity however many photos show it |
| Detection vs entity | a detection is an observation, never an entity |
| Do not merge on label | class-label equality alone is insufficient grounds for merge |
| Identity confidence | per-observation `AssociationDecision.probability` (likelihood-based weight); `EntityTrack.confidence` |

## 2. Cross-view identity (association)

**Schema location:** `perception/instances/association.py` `resolve_hypotheses()` (production association) — NOTE: competing implementations exist.

### 2.1 Production association: gated Hungarian (`resolve_hypotheses`)

The production path uses gated Hungarian assignment with per-image one-to-one constraint:

| Gate | Contract |
|---|---|
| Label compatibility | child→parent label map (armchair→chair, sofa→couch, etc.); exact or parent match |
| 3D Mahalanobis distance | uncertainty-aware: relative-depth observations matched loosely, metric ones tightly; chi² 3DOF 99% = 11.34 |
| Normal difference | `MAX_NORMAL_DIFF_DEG = 30.0`, `NORMAL_SIGMA_DEG = 15.0` |
| Visual appearance | cosine similarity gate `MIN_APPEARANCE_SIM_GATE = 0.20`; running appearance embeddings across views |
| Two-view epipolar | geometric verification; `geom_verified` flag on every `AssociationDecision` |
| One-to-one per image | two detections in the same image can never be the same entity; an entity takes at most one detection per image |
| No transitive chaining | A~B and B~C does NOT merge A with C (fixes union-find failure mode) |

### 2.2 Association decision states

**Schema location:** `world_ir/observations.py` `AssociationState`.

| State | Meaning |
|---|---|
| `SAME_ENTITY_CONFIRMED` | observation joined an existing entity |
| `SAME_SURFACE_CONFIRMED` | same surface, different entity context |
| `DIFFERENT_ENTITY` | observation is a different physical object |
| `PHYSICAL_INTERSECTION` | objects physically intersect |
| `IMAGE_OCCLUSION` | occlusion within one image |
| `PARTIAL_OVERLAP` | partial overlap both ways |
| `AMBIGUOUS` | best & second-best within log(2) margin without appearance disambiguation — kept as separate provisional entity, NOT merged |
| `INSUFFICIENT_EVIDENCE` | not enough evidence to decide |

**Contract rule:** When best and second-best candidate are within a likelihood margin without appearance disambiguation, the observation is NOT merged — it becomes a provisional entity marked `AMBIGUOUS`, with both candidates recorded. Ambiguity is retained, never forced.

## 3. Competing association implementations (to be converged)

| Module | Key symbol | Status |
|---|---|---|
| `perception/instances/association.py` | `resolve_hypotheses` | PRODUCTION (gated Hungarian) |
| `perception/entities/association.py` | `associate`, `EntityObservation` | legacy union-find; deprecated |
| `engine/multiengine/association.py` | `associate`, `Candidate`, `PairScore`, `AssociatedEntity` | competing ray/entity model |
| `engine/multiengine/spatial_intelligence/association.py` | `CrossViewRayCandidate`, `TemporalIdentityTracker`, `PhysicalMergedEntity` | experimental/temporal |

**Contract rule:** The production path is `perception.instances.association.resolve_hypotheses`. Competing implementations are not merged into the production path without a demonstrated defect + migration plan.

## 4. Re-identification & duplicate suppression

| Concept | Contract |
|---|---|
| Re-identification | by TYPE + POSITION in previous frame (world_delta); ids re-derived each rebuild |
| Duplicate suppression | non-destructive: preserves part-inside-parent relationships |
| Same-image collision | two detections in one image cannot be one entity |
| Entity per image | an entity takes at most one detection per image |
| Transitive chaining | forbidden (A~B, B~C does not merge A with C) |

## 5. Object splitting & merging

**Schema location:** `world_ir/spatial_continuity.py` `reconcile()`; `world_delta.py` `compute_delta()`, `physical_changes()`.

| Relation | Definition | World change? |
|---|---|---|
| `preserved` | one-to-one, both cover ≥ 60%, negligible shift/tilt | No |
| `refined` | one-to-one, both cover ≥ 60%, moved/tilted/resized | No (representation) |
| `extended` | previous surface covered by candidate that reaches further | No (representation) |
| `reduced` | candidate covers only part of previous surface | Yes (partial loss) |
| `split` | one previous surface → several fragments jointly covering it | No (representation) |
| `merge` | several previous surfaces → one candidate covering each | No (representation) |
| `regrouped` | several↔several, union of each side covers the other | No (representation) |
| `ambiguous` | overlapping but not classifiable | reported, not forced |
| `removed` | nothing in candidate overlaps it | Yes |
| `new` | nothing in previous version overlaps it | Yes |

**Contract rule:** Splits, merges, extensions, reductions, and regroupings are changes of **representation**, not of the physical world — they are reported as such, not as world changes. Only `removed`/`reduced` (not reproduced) and genuinely new entities count as world changes.

**Ambiguity:** overlapping but unclassifiable relations (many-to-many, partial both ways) are reported as `ambiguous`, never forced into a category.

## 6. Persistent entity identifiers

| Concept | Contract |
|---|---|
| Id stability | ids are re-derived each rebuild; NOT stable across versions |
| Match key | TYPE + POSITION in the previous frame (world_delta) |
| Identity carry-over | `world_ir/entity_reid.carry_identity` — carries identity across versions |
| Version-to-version | `world_delta.decide()` verdict {ACCEPT, ACCEPT_WITH_UNCERTAINTY, REJECT} + reasons + uncertainties, each citing a measured fact |
| Conflicts | never silently resolved; both hypotheses kept with provenance |

## 7. Identity confidence

| Concept | Contract |
|---|---|
| Observation confidence | `AssociationDecision.probability` — likelihood-based weight of chosen candidate; 0 when none passed gates |
| Entity confidence | `EntityTrack.confidence` — fused across observations |
| AMBIGUOUS | best & second-best within log(2) ≈ 0.693 margin without appearance disambiguation |
| Confidence never fabricated | `None` = not measured; association code treats None as "no measurement" |

## 8. Contradiction handling

| Concept | Contract |
|---|---|
| Contradictory observations | retained; `CONFLICT` provenance; never silently resolved |
| Both hypotheses | kept with provenance until a later version agrees |
| Unresolved conflicts | carried forward (`world_delta.reconcile_conflicts`) |
| Camera mismatch | `Transform.then()` frame mismatch raises `ValueError` |
| Scale conflict | incompatible scale states refused during clustering; recorded in competing hypothesis ledger |

## 9. Version-to-version identity continuity

**Schema location:** `world_delta.py` `decide()`, `reconcile_conflicts()`, `physical_changes()`, `describe_changes()`; `worldstore/lifecycle.py` adoption protocol.

| Step | Contract |
|---|---|
| Snapshot | `world_delta.snapshot(world)` — captures current state |
| Compute delta | `compute_delta(parent, candidate)` — per-entity/entity-record changes |
| Decide | verdict {ACCEPT, ACCEPT_WITH_UNCERTAINTY, REJECT} + reasons; each uncertainty cites a measured fact |
| Reconcile conflicts | `reconcile_conflicts()` — both hypotheses kept; never silently resolved |
| Physical changes | `physical_changes()` — what actually changed in the world (not representation) |
| Adoption | intent file → WorldStore version → read-back verify → DB HEAD + mirror row in ONE transaction → intent cleared |
| Failed adoption | undoes its own pending HEAD move; HEAD stays authoritative |
| Rejected candidate | HEAD stays authoritative; new evidence kept and retried |

**Contract rule:** Ids are re-derived each rebuild, so they are useless for matching across versions. Identity continuity is established by TYPE + POSITION in the previous frame, not by id equality.

## 10. What this contract forbids

- Do not merge entities solely because class labels match.
- Do not treat an id as a stable cross-version match key.
- Do not silently resolve conflicting observations.
- Do not treat representation changes (split/merge/regroup/extend/reduce) as physical world changes.
- Do not force ambiguous overlaps into a category.
- Do not manufacture identity confidence.
