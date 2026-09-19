# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: feat(reconstruction): RECONSTRUCTION_CONTRACT_V1 stage-result surface
Checkpoint: RECONSTRUCTION_CONTRACT_V1_READY
Status: CHECKPOINT_READY

## What exists

`perception/contract.py` — the versioned V1 result surface for the
evidence→perception→reconstruction pipeline, composed ENTIRELY from the
repo's canonical stage types (no competing payload shapes):

- `CONTRACT_VERSION = "1.0"` and `StageStatus` — the exact status
  vocabulary `complete | partial | unavailable | failed`.
- Stage envelopes: `DepthResult` (canonical `DepthMap`), `PoseResult`
  (canonical `ReconstructedCameraPose`), `ObservationResult`
  (detections/segmentations; empty lists under COMPLETE are a real
  "found nothing" outcome), `FusionResult` (canonical `FusedEstimate`
  fields), `QualityResult` (canonical `EvidenceQualityReport`).
- `ReconstructionContract` + `contract_reconstruction_from_run()` —
  the top-level surface composed VERBATIM from the orchestrator's
  `ReconstructionRun`: status = `diagnostics.final_status`
  (`success|partial|failed`), detail = the run's error text, result =
  the backend's canonical `ReconstructionResult` (never rewritten,
  never upgraded; failed runs carry no result payload).
- Honesty invariants enforced in `__post_init__`: complete/partial
  REQUIRE their payload; unavailable/failed FORBID it (no zombie data,
  no silent degradation).

## Files changed

- `perception/contract.py` (new)
- `tests/test_reconstruction_contract.py` (new, 20 tests, written red-first)

## Public contracts

```python
from perception.contract import (
    CONTRACT_VERSION, StageStatus, ContractError,
    DepthResult, PoseResult, ObservationResult,
    FusionResult, QualityResult,
    ReconstructionContract, contract_reconstruction_from_run,
)
```

Every envelope: frozen dataclass, `.to_dict()` for UI/JSON consumption,
`detail` always carries the reason for unavailable/failed states.

## How to consume

- Studio/integration (Antigravity): render stage panels from
  `*.to_dict()` — `status` drives UNAVAILABLE/DEGRADED/FAILED banners,
  `detail` is the user-facing reason string. Never infer status from
  payload presence; the contract guarantees it.
- Orchestrator consumers: `contract_reconstruction_from_run(run)` turns
  any orchestrated run into the V1 surface in one call.

## Tests run

`python -m pytest tests/test_reconstruction_contract.py -q`

## Test results

20 passed (status vocabulary, payload/absence invariants for all five
envelopes, verbatim run composition, failed-run honesty).

## Runtime verification

Contract module imported and exercised via the test suite; no runtime
process needed for a type surface.

## Dependencies

Canonical types only: `perception/depth/interface.DepthMap`,
`reconstruction/backend/interface.ReconstructionResult`/`ReconstructedCameraPose`,
`perception/quality/assessment.EvidenceQualityReport`,
`reconstruction/fusion/fusion.FusedEstimate` (by field access),
`reconstruction/orchestrator.ReconstructionRun`, `provenance.Uncertainty`.

## Known limitations

- `FusionResult.fused` is typed `Any` in the envelope (field access only)
  to keep the module import-cycle-free from `reconstruction/fusion`;
  the payload remains the canonical `FusedEstimate`.
- Envelopes serialize payloads to summary dicts, not full arrays
  (depth `values` are summarized as row counts) — full arrays are for
  the artifact store, not the UI contract.

## Known bugs

None known.

## Next agent

Antigravity (studio/product integration) — consume this contract for
reconstruction-stage UI. Integration agent — the WorldStore bridge can
compose `ReconstructionContract` from orchestrated runs today.

## Exact action for next agent

1. `git fetch origin && git checkout agent/freebuff-reconstruction-perception`
2. Render stage status from `to_dict()` output; treat `detail` as the
   reason string for unavailable/failed banners.
3. For new stage results, construct envelopes (never raw canonical types)
   so the honesty invariants hold at the boundary.
