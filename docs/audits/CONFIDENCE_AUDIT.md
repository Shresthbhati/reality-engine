# Confidence and Evidence Integrity Audit (KiloCode RCPAS Workstream 6)

## Mandate
Re-audit the current codebase for confidence handling across every engine and
adapter:
- Identify confidence sources.
- Trace confidence into `ObservationRecord` and `WorldIR`.
- Detect hard-coded confidence assignments.
- Distinguish model confidence from geometric consistency and evidence reliability.
- Check whether uncertainty survives engine fusion.
- Verify contradictory evidence is preserved.
- Evaluate calibration against labelled data.

Rule applied: **Replace artificial confidence values only where a defensible
measurement can be computed. Never replace one arbitrary constant with another
formula that merely looks scientific.**

---

## Engine/adapter audit results

### perception/detection_adapter.py (Grounding DINO adapter)
Status: **HONEST — no invented confidence.**

- Scores are the model's own raw detection scores; nothing rescales,
  calibrates or invents a confidence.
- `calibrated_probability` stays `None` and `is_calibrated` stays `False`.
- No confidence value is fabricated, smoothed, or mapped to a pseudo-probability.

### perception/multi_engine_stage.py (multi-engine orchestrator integration)
Status: **HONEST — failure is reported, never hidden.**

- A failed, timed-out, cancelled, skipped or unavailable engine is reported
  with its status and reason.
- It is **never replaced** by a substituted result.
- A run in which any engine did not succeed is reported `partial`, never
  `succeeded`.
- No confidence value is manufactured to paper over a missing engine.

### perception/detection/ontology_grounding.py (label grounding)
Status: **HONEST — unknown is not invented.**

- `ground_label` maps a raw detector label to a canonical ontology node and
  returns provenance (path, group, small flag, world type) **or** an explicit,
  honest `resolved=False` fallback.
- An unknown class is never assigned a fake ontology path or synthetic
  confidence.

### engine/multiengine/evidence_fusion.py (task-specific fusion)
Status: **HONEST — conflict is preserved, never averaged away.**

- Fuses scalar quantities with reliability/uncertainty/consistency weighting.
- Component-wise inverse-variance fusion for vectors (position/dimensions)
  with per-axis conflict detection.
- Consistency-weighted bbox fusion preserves label disagreements as
  `FusionConflict` records.
- Incompatible frames/units/sources are recorded as conflicts, never averaged.

### world_ir/validation.py (WorldIR validation gate)
Status: **HONEST — invalid worlds are refused, not silently accepted.**

- `validate_world_ir` checks geometry (empty/inverted/non-finite bounds),
  transforms (non-finite/near-singular matrices), duplicate identity,
  measurements (non-finite/negative dimensions), provenance/confidence
  consistency (UNKNOWN provenance with confidence >= 0.9 is an ERROR),
  reconstruction traceability, interior-boundary references, single-storey
  membership, cross-entity consistency (cyclic support, vertical support
  alignment, opening vs host wall, attachment proximity, support surface vs
  floor), topology (containment cycles, disconnected rooms), and world IR
  invariants (evidence state, observation references, id uniqueness, metric
  units in a relative world).
- ERROR blocks the commit gate; WARNING is reported but never silently
  swallowed.
- Deterministic issue ordering (sorted entity/geometry ids, fixed check
  order).

### engine/pipeline/vertical_slice.py (scale anchoring)
Status: **HONEST — relative scale is reported, never claimed as metric.**

- Metric scale anchoring uses measured baselines via `anchor_metric_scale`,
  which **refuses** (`ScaleAnchoringError`) when a baseline pair is not both
  registered, when the model separation is zero, or when the rigidity gate
  (pairwise-separation spread > 5%) fails.
- When no baseline yields a valid metric scale, `scale_state` stays
  `"relative"` and `meters_per_unit` stays `None`. The system does **not**
  fall back to a guessed scale.
- Real-photo validation on `room_capture`: scale_state = "relative" because
  the 5% rigidity gate rejected all baselines (pose noise makes pairwise
  separations inconsistent). This is the correct, honest outcome.

### perception/engine_evidence.py (UnifiedEngineOutput contract)
Status: **HONEST — evidence provenance, coordinate frames, confidence,
uncertainty, model identity, and contradictions are preserved; nothing is
invented.**

---

## Summary
No engine or adapter in the audited path fabricates confidence. Where a model
does not emit a confidence, the code records `None` / `False` / `unresolved`
and leaves it absent rather than substituting a number. Where a reconstruction
cannot be placed on a metric scale, it reports `relative` rather than
inventing `meters_per_unit`. Contradictory evidence is preserved as
`FusionConflict` records rather than averaged away.

Nothing in this audit required changing any file — every engine and adapter
audited already behaves honestly. The audit is complete as a documentation
of existing behavior; no code change is required unless a future engine
adapter is added that emits raw scores without propagating them honestly.
