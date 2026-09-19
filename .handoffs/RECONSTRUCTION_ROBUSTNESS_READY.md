# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (see git log -- "feat(reconstruction): RECONSTRUCTION_ROBUSTNESS...")
Checkpoint: RECONSTRUCTION_ROBUSTNESS_READY
Status: CHECKPOINT_READY

## What exists

1. **`reconstruction/robustness.py`** — item-level outcome classifier over
   the five-state vocabulary (ACCEPTED / REJECTED / DEGRADED / UNRESOLVED /
   FAILED), each with a non-empty reason naming the measured value. Run-level
   `classify_reconstruction_run` maps `registration_status`
   (success/partial/failed) onto the same vocabulary; unknown statuses
   refuse to classify. Thresholds are injectable per capture domain.

2. **`reconstruction/robustness_admission.py`** — quality-aware compute
   admission composing `classify_evidence_items` with the orchestrator
   (orchestrator untouched): REJECTED/FAILED items are excluded from
   reconstruction spend WITH recorded reasons; UNRESOLVED proceed flagged
   (absent metrics never veto good captures); batch-level
   `AdmissionError` names admitted/excluded counts when admission leaves
   too little evidence. `reconstruct_with_admission` records
   submitted/admitted/excluded into run diagnostics.

3. **Scale hardening** — orchestrator input-gate duplicate scan rewritten
   O(N²) → single pass (20k items with a duplicate: >10 s class → <1 s,
   refusal contract unchanged). `benchmarks/robustness_scale.py`: real
   timing+tracemalloc harness over 20/50/100/500/1000/5000 images —
   linear scaling, deterministic 80/10/5/5 outcome mix, peak 9.8 MB
   python allocations at 5k images. Backend labeled TEST DOUBLE in the
   harness docstring and output.

4. **Main repairs (cross-agent, all `3dc02ab` isolation fallout)** —
   28 stale `engine.physics.math3` imports rewritten to `engine.math.math3`;
   `Camera.look_at/right/is_visible` + `Viewport` real frustum
   culling/`set_camera` implemented to their committed test contracts;
   platform-boundary guards rewritten for the post-isolation architecture
   (allowlist widened to `engine.math`; simulation-bridge guard now
   enforces the bridge's ABSENCE from core); SDK external-consumer flow
   updated to the post-isolation facade.

5. **Real-evidence verification** — the 294,345-point GPU COLMAP
   `room_capture_mvs` fused.ply (gitignored artifact, restored locally
   from the architecture worktree) passes 4/4
   `tests/test_dense_real_capture_integration.py` on this branch:
   parse → extent → canonical geometry with provenance → content-addressed
   artifact roundtrip.

## Public contracts

```python
from reconstruction.robustness import (
    classify_evidence_items,      # -> AdmissionReport
    classify_reconstruction_run,  # -> RunAdmission
    ACCEPTED, REJECTED, DEGRADED, UNRESOLVED, FAILED,
)
report = classify_evidence_items(items)          # thresholds injectable
report.counts / report.accepted_ids / report.rejected_ids / ...

from reconstruction.robustness_admission import (
    admit_for_reconstruction,       # (items) -> (admitted, AdmissionDecision)
    reconstruct_with_admission,     # (backend, items) -> (ReconstructionRun, decision)
)
```

Metric schema (importer-recorded): `metadata["quality"]` =
`laplacian_variance` / `luma_mean` / `clipped_fraction` / `measured`.

## How to consume

- Frontend/agents: `AdmissionReport.to_dict()` is the display-ready
  per-item outcome list with reasons — bind status panels to it.
- Pipeline drivers: call `reconstruct_with_admission(backend, items)`
  instead of `orchestrator.run(items)` when captures may contain
  low-quality frames; the run diagnostics carry
  `evidence_validation["admitted"/"excluded"/"submitted"]`.
- Do not re-classify: one item, one outcome, recorded once.

## Tests run

- `tests/test_reconstruction_robustness.py` — 17 tests (red-first)
- `tests/test_robustness_admission.py` — 7 tests (red-first)
- `tests/test_dense_real_capture_integration.py` — 4 real-data tests
- Full unbounded suite at branch head: **1881 passed / 8 skipped / 1 failed**
  (the 1 = SDK test pre-dating the isolation split; repaired, re-run green;
  affected-cluster re-gate after repair: 48 passed incl. real-data tests).

## Runtime verification

- Scale harness executed for real (numbers above are measured, not estimated).
- Real COLMAP capture integration executed for real on this machine.

## Dependencies

- numpy/Pillow already used by the importer for metrics (present).
- No new dependencies.

## Known limitations

- Classification trusts importer-recorded metrics; items imported
  without pixel metrics are UNRESOLVED (flagged, not vetoed).
- The scale harness's reconstruction stage is the FakeReconstructionBackend
  (TEST DOUBLE): it measures pipeline bookkeeping scale, never SfM quality.
- Real COLMAP spend-avoidance E2E (admission demonstrably saving GPU work
  on a real capture) is the named next gate.

## Known bugs

- None known in this checkpoint's scope. (Pre-existing: EXR depth adapter
  remains environmentally blocked — no EXR codec on this machine.)

## Next agent

Any agent wiring captures into reconstruction; Antigravity for status
surfaces.

## Exact action for next agent

1. Consume `reconstruct_with_admission` in capture-driven pipelines.
2. For the spend-avoidance gate: run COLMAP with/without admission on a
   real mixed-quality capture and record measured savings in TASKS.yaml.
3. Desktop/mobile UI: `AdmissionReport.to_dict()` → per-frame status panel.
