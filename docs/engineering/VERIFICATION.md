# Verification Requirements

A status transition is legitimate only with the evidence below.
Claims without execution are never acceptable.

## Evidence gates per transition

| Transition | Required evidence |
|---|---|
| MISSING → FOUNDATION | Interfaces/schema exist and import; intent documented |
| FOUNDATION → PARTIAL | Real behavior exercised through at least one integration path (not just unit-isolated) |
| PARTIAL → IMPLEMENTED | Working end-to-end for the defined scope + tests + diagnostics + docs |
| IMPLEMENTED → VERIFIED | Verification at the tier the feature demands: deterministic tests, and real-data/real-model validation where the claim involves real perception/reconstruction |
| anything → BROKEN | A currently-failing verification, named exactly (command + failure) |
| anything → BLOCKED | DECISION REQUIRED / OPTIONS / TRADEOFFS / RECOMMENDATION / DEPENDENCIES recorded in the backlog |

## Rules

1. **Run the real path.** "It compiles", "the interface exists", "the
   import succeeds" are not verification.
2. **Tier discipline.** Unit-verified ≠ integration-verified ≠
   real-data-verified. Never collapse them into one claim.
3. **Record actual commands and results.** `.agent/execution/STATE.md`
   holds the last verification (command + observed result).
4. **Environment ≠ implementation.** Misclassifying a broken torch-hub
   cache as an implementation bug (or vice versa) corrupts the backlog.
5. **No invented numbers.** Benchmarks, metrics, camera counts, entity
   counts must come from observed runs; the flagship E2E numbers in the
   backlog come from real executed runs on `datasets/room_capture`.
6. **Uncertainty is preserved.** Verification of approximate methods
   (metric-by-alignment depth, AABB objects) asserts the *labeled*
   approximation, not perfection.

## Session discipline

End every execution session by updating:
- `.agent/execution/STATE.md` (completed, in-progress, blockers, last verification)
- `.agent/execution/TASKS.yaml` (task statuses)
- [`../implementation/PENDING_IMPLEMENTATION.md`](../implementation/PENDING_IMPLEMENTATION.md)
  (status transitions with this session's evidence)
