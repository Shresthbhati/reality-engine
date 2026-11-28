# OpenCode 200K Reliability / Security / Validation Report

**Executed by:** OpenCode primary agent. **Date:** 2026-11-26.
**Evidence labels:** VERIFIED (independently reproduced), AGENT-REPORTED (not reproduced),
UNVERIFIED (not tested/measurable).

## 1. Honest headline

The 200,000 net-new LOC target was **not met**. Two genuine, bounded, fully-tested
contributions were delivered (≈547 net lines total). No code was inflated, copied,
or fabricated to reach the number. Subagents were unavailable ("free usage limit
reached"), so the multi-workstream parallel execution the directive assumes could
not happen. Functionality and verification took precedence over the numeric target,
as the repo's AGENTS.md requires.

## 2. Independent audit (Task 1) — key findings

- `origin/main` = `5b33b7b9` (PR #396, A4/A5 perception). 127 remote branches,
  90+ worktrees, 14 stashes exist locally.
- The Gemini A5 work (`e390eb00`→`127d4fbe`) is **already in main** — `git diff
  127d4fbe origin/main -- perception/ tests/` is empty (VERIFIED). No lost Gemini
  perception work found.
- `stash@{6}` holds 695 lines of backend API work; `stash@{12}` holds 224 lines of
  Studio inspector work — both UNCOMMITTED, left in place for their owning agents
  (not overwritten). Classification: UNCOMMITTED.
- My earlier A5-line work sat on `c10f9a2c` (merge-base `1ab49047`, NOT an ancestor
  of main). It was rebased onto main in `recovery/opencode-mask-contract`.

## 3. Delivered work

### 3.1 `recovery/opencode-mask-contract` @ `768f1de2` (pushed, VERIFIED)
Reconciled the mask-contract fix onto main:
- `ObservationRecord.mask` + `mask_ref`, `_mask_descriptor()` compact JSON.
- `perception/mask_persistence.py` — RLE persistence + atomic save/load.
- `multi_engine_stage.py` — post-orchestration mask persistence.
- `segmentation_adapter.py` — `mask=mask_2d` (kept main's superior `_select_best_2d_mask`).
- 10/10 new tests pass on main's codebase. During cherry-pick resolution I kept
  main's robust mask logic and DROPPED my obsolete reshape fix (main already
  handles stacked masks) — no clobbering of other agents' work.
- Full regression (9 modules): 123 passed, 1 skipped, 1 FAILED — the failure is
  **pre-existing** (fails identically on baseline `c10f9a2c`): cached HF weights
  let real engines add `wall`/`ceiling`/`depth_map` quantities the test asserts
  against (VERIFIED, not introduced by this work).

### 3.2 `security/upload-path-traversal` @ `4cf76458` (pushed, VERIFIED)
Closed a real defense-in-depth gap at the upload staging boundary:
- `validate_staging_id()` rejects path separators, `..`, absolute paths, NUL
  bytes, URL-encoded traversal, and over-long ids.
- `staging_path()` centralises the validated path for all staging operations.
- `resumable_finalize` now calls `assert_visible(db, Upload, upload_id, auth)` —
  previously the endpoint had **no authorization check** (it could finalize any
  upload id if reachable). Now consistent with the `/chunk` endpoint.
- 33 tests pass: new path-security tests (16 traversal cases), resumable upload,
  tenant isolation.

## 4. Tasks not executed (honest status)

| Task | Status | Reason |
|---|---|---|
| 2 WorldStore data integrity | UNVERIFIED (not newly implemented) | Mature existing implementation; no new gap found in the audited slice |
| 3 API & job orchestration | PARTIAL (3.2 auth fix only) | scope bounded by available capacity |
| 4 GPU/compute management | NOT RUN | existing VRAM-aware orchestrator reviewed, no new work |
| 5 Security | VERIFIED (3.2) | one genuine gap fixed |
| 6 Observability | NOT RUN | existing observability_service.py reviewed |
| 7 Performance | NOT RUN | no benchmark executed this session |
| 8 Adversarial testing | PARTIAL (path traversal/adversarial ids) | bounded suite only |
| 9 Golden workflow | NOT RUN | requires real models end-to-end; not executed |
| 10 CI | NOT RUN | failing shard diagnosed (pre-existing, section 3.1) but not fixed |
| 11 Delivery | VERIFIED | this report + ledger written and pushed |

## 5. Push verification (VERIFIED)

```
git ls-remote --heads origin recovery/opencode-mask-contract
  -> 768f1de2bad408e68ba718df1ab41cad98f18d2c
git ls-remote --heads origin security/upload-path-traversal
  -> 4cf76458896c8ac772e20a655399754921ddacbc
```
No force-push to main; no merges into main; no PRs opened (no authorization).
Both branches are from `origin/main@5b33b7b9` and independently reviewable.

## 6. Agent utilization (honest)

- Subagents: 8 launched in the prior session, ALL returned "Free usage limit
  reached"; zero work produced; NOT counted.
- Skills: `find-skills` (guidance only). No implementation skill used.
- Primary OpenCode agent performed all delivered work.

## 7. Blockers

- Human-dependent: six measured rooms, 15-image annotation manifest, main branch
  protection (RELEASE.md).
- Capacity: subagent usage limit prevents the 11 parallel workstreams.
- The one failing regression test is pre-existing and environment-dependent
  (cached model weights); fixing it is out of scope for these two branches.

## 8. Conclusion

Two small, real, verified contributions were delivered and pushed to independently
reviewable branches. The repository's actual state is more mature than the directive
assumes; the highest-value action was honest reconciliation and one genuine security
fix, not manufacturing LOC. All results above are reproducible from the cited commands.
