# Multi-View Entity Identity (Phase 4)

**Core Implementation:**
- `engine/multiengine/multi_view_association.py` (`MultiViewEntityObservation`, `ReconciledPhysicalEntity`, `associate_multiview_observations`)
- `engine/multiengine/association.py` (`Candidate`, `Decision`, `MERGED` / `REFUSED`, `associate`)
- `world_ir/entity_reid.py` (`identity_plan`, `rekey_entities`, `carry_identity`)
- `world_ir/observations.py` (`AssociationDecision`, `AssociationState`, `attach_ledger`)

## Identity Invariants

1. **Same-Image Hard Refusal:** Two detections from the same `source_id` (same photograph) **never** merge into one physical entity.
2. **Semantic Compatibility:** Incompatible ontology classes (e.g. `window` vs `tree`, `switch` vs `socket`) **never** merge.
3. **Geometric & Reprojection Consistency:** Association combines 3D position/extent compatibility, unit-aware uncertainty, camera visibility (`z_cam > 0.1`), and reprojection residuals rather than merging solely on class or appearance similarity.
4. **Explicit Uncertainty:** When calibration or 3D lift is missing or ambiguous, observations remain separate with an explicit `REFUSED` `Decision` reason in the ledger.
5. **Version-Safe Re-Keying:** `rekey_entities()` preserves `ent.id`, `ent.parent_id`, `relationships`, `transforms`, `temporal_events`, `body_states`, `boundary_element_ids`, `host_wall_id`, and `association_decisions`.
