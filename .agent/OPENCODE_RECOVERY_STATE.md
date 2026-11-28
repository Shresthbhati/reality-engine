# OpenCode Recovery State — Mask contract contribution
Branch: recovery/opencode-mask-contract @ fb59ac22 (from origin/main 5b33b7b9)

Note: main's .agent/EXECUTION_STATE.md is preserved untouched. This file records
only the OpenCode recovery contribution.

## Contribution
Verified mask-contract fix reconciled onto main:
- perception/engine_evidence.py: ObservationRecord.mask + mask_ref, _mask_descriptor()
- perception/mask_persistence.py (new): persist_masks, save/load MaskStore
- perception/multi_engine_stage.py: post-orchestration mask persistence
- perception/segmentation_adapter.py: ObservationRecord.mask = selected mask
- tests/test_mask_persistence.py, tests/test_segmentation_adapter_mask.py

## Verification (on recovery branch, main codebase)
- 10/10 mask-contract tests pass
- Full regression suite: 123 passed, 1 skipped, 1 FAILED (pre-existing on baseline
  c10f9a2c too — test_detection_pipeline_engine expects only bbox+summary evidence
  quantities but cached HF weights enable real engines adding wall/ceiling/depth_map)

## Push
origin/recovery/opencode-mask-contract @ fb59ac22 verified via git ls-remote.
