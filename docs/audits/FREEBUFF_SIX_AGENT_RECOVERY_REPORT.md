# Six-Agent Progress Recovery Report (OpenCode execution)

**Report type:** Recovery audit + one verified contribution reconciled to GitHub.
**Executed by:** OpenCode (primary agent). Note: this recovery report was requested
under the "FREEBUFF" recovery directive; it is executed by OpenCode. No other
agent/subagent performed work in this session — see "Agent utilization" below.

## 1. Quantitative accounting (VERIFIABLE)

| Item | Value | Label |
|---|---|---|
| Starting `origin/main` commit | `5b33b7b9` ("Workstream A4/A5 evidence-grade perception & production reliability (#396)") | VERIFIED |
| Earlier local baseline from A5 directive | `c10f9a2c` (diverged from main at merge-base `1ab49047`; NOT an ancestor of main) | VERIFIED |
| Recovery branch | `recovery/opencode-mask-contract` @ `768f1de2` (from `origin/main`) | VERIFIED |
| Remote verification | `git ls-remote --heads origin recovery/opencode-mask-contract` → `768f1de2bad408e68ba718df1ab41cad98f18d2c` | VERIFIED |
| Net diff vs main | 425 insertions, 1 deletion across 8 files | VERIFIED |
| Production LOC (net new) | 133 (engine_evidence.py +31, mask_persistence.py +73, multi_engine_stage.py +24, segmentation_adapter.py +5) | VERIFIED |
| Test LOC (net new) | 271 (test_mask_persistence.py +107, test_segmentation_adapter_mask.py +164) | VERIFIED |
| State/docs LOC | ~21 (.agent/OPENCODE_RECOVERY_STATE.md; main's EXECUTION_STATE.md preserved) | VERIFIED |
| PRs opened | 0 (not authorized to open/merge PRs into main) | VERIFIED |

## 2. Agent progress audit

Classification legend: VERIFIED RECOVERED / ALREADY IN MAIN / UNCOMMITTED /
RECOVERED FROM GIT OBJECTS / MISSING / SUPERSEDED / CONFLICTING.

### Gemini — Perception
- Assignment: A3/A4/A5 perception (detection, segmentation, identity, fusion, benchmarks).
- Branch `feat/gemini-december-perception` @ `127d4fbe` contains A3/A4/A5 commits
  (`e390eb00` A4, `39235af6` A5, `127d4fbe` A5 tests) — the exact starting commit from
  the original A5 directive.
- `git diff --stat 127d4fbe origin/main -- perception/ tests/` is EMPTY → the entire
  A3/A4/A5 perception tree is byte-identical to main. **ALREADY IN MAIN** via PR #396
  (`5b33b7b9`, which lists the A5 report, benchmark suite, adapters, contracts).
- No missing/unrecoverable Gemini work found.

### Claude — Reconstruction
- Many `claude/*` branches/worktrees exist locally; `origin/main` includes merged Claude
  work (#394/#395 and earlier PRs: perception-to-reconstruction pipeline, multi-engine,
  camera registration, WorldIR v2, spatial compiler, persistence).
- Representative worktrees: `claude/sonnet-evidence-world-rebase` (d1759d4c),
  `b9/integration`, `claude/phase8-export`, etc. — local only, not on main.
- Classification: **ALREADY IN MAIN** for merged items; remaining `claude/*` branches are
  local-only follow-up work, not lost main work. Not merged (no authorization).

### FreeBuff — End-to-end reconstruction
- `agent/freebuff-reconstruction` @ `bd4267ef` (worktree
  `.claude/worktrees/agent4-recon-perception`) and `agent/freebuff-reconstruction-perception`,
  `agent/freebuff-auto-recon-sprint` branches exist locally.
- These are local worktrees; main contains the integrated perception-to-reconstruction
  pipeline (#394). **ALREADY IN MAIN (integrated form)** + local-only follow-ups.

### Antigravity — Reality Studio
- `agent/antigravity-integration` @ `c08bc5fa` worktree exists; `origin` has
  `feat/gaussian-splatting-and-hybrid-viewport`, `feat/multi-room-navigation...`, etc.
- Studio work merged via earlier PRs (#389 adaptive inspector, #391 sprint10). **ALREADY IN MAIN**.

### OpenCode — Platform reliability
- My own contribution this session: mask-contract fix, now on recovery branch pushed to
  GitHub. **VERIFIED RECOVERED** → `recovery/opencode-mask-contract` @ `768f1de2`.

### KiloCode — Independent QA
- `cat-computer` branch/worktree @ `d2b45e0e`, `agent/cline-*` branches, `cline/*` worktrees.
- No KiloCode-specific commits identified in main beyond merged QA. **ALREADY IN MAIN** /
  local-only branches.

## 3. Uncommitted work found (not recovered into a branch — preserved in place)

- `stash@{6}` "other-agent-wip-backend": 695 insertions across `apps/api/jobs.py`,
  `routes_worlds.py`, `routes_misc.py`, `storage.py`, `apps/cli/main.py`, `pyproject.toml`,
  tests — real backend work. CLASSIFICATION: **UNCOMMITTED**. Not applied (belongs to
  another agent's ownership; applying could overwrite newer main versions).
- `stash@{12}` "local changes before merge" (studio-viewer): 224 insertions to
  `EntityInspector.tsx`. CLASSIFICATION: **UNCOMMITTED**.
- Other stashes (0,1,2,3,4,5,7,8,9,10,11,13) contain WIP/tool state and are preserved.

## 4. Production workflow verification (as found in main)

The full chain is present in code: ingestion → perception → detection/segmentation →
cross-view identity → camera registration → depth/geometry → reconstruction →
evidence fusion → WorldIR → WorldStore → Studio → correction → versioned World → export.
Evidence of integration: #394 (perception→reconstruction, multi-engine, camera
registration), #395 (real-capture multi-engine measurements), #396 (A4/A5 perception).
This is a code-level finding (VERIFIED from git history and file inventory), not a
fresh end-to-end runtime claim — see "Tests".

## 5. Tests executed (this session)

| Command | Result | Label |
|---|---|---|
| `pytest tests/test_mask_persistence.py tests/test_segmentation_adapter_mask.py` (recovery branch) | 10 passed | VERIFIED |
| `pytest tests/test_mask_persistence.py tests/test_segmentation_adapter_mask.py tests/test_mask_store.py tests/test_observation_table.py` | 16 passed | VERIFIED |
| Full regression (9 observation/engine modules) | 123 passed, 1 skipped, 1 FAILED | VERIFIED |
| Failing test: `test_detection_pipeline_engine.py::test_run_for_job_executes_the_pipeline_and_reports_what_it_did` | fails identically on baseline `c10f9a2c` (pre-existing): cached HF model weights let real engines add `wall`/`ceiling`/`depth_map` evidence quantities the test asserts against | VERIFIED (pre-existing, not introduced by recovery) |

Real models: Grounding DINO, SAM 2, Depth Anything weights were downloaded/loaded by
the test harness (CUDA RTX 4050 available); no accuracy metrics claimed.

## 6. Agent utilization (honest)

- Subagents: 8 were launched (workstreams A–F, K, L) but ALL returned
  "Free usage limit reached" errors; NONE produced any work. They are NOT counted.
- Primary OpenCode agent: performed the audit, fix, tests, reconciliation, push.
- Skills invoked: `find-skills` (external skills CLI guidance — not used for implementation).
- No other agents contributed this session.

## 7. Remaining tasks / blockers

- PRs: not opened (no authorization to merge into main). Recovery branch is available at
  `origin/recovery/opencode-mask-contract`.
- Human-dependent blockers remain per RELEASE.md: six measured rooms (photos + laser
  measurements), 15-image annotation manifest, branch protection on main.
- Uncommitted stashes `{6}` and `{12}` need their owning agents to reconcile.
- The 100,000 LOC programme target is NOT met by this session; functionality and
  verification took precedence. No code was inflated to reach a numerical target.
- Free usage limit prevented subagent-based parallel workstreams; all workstreams except
  the mask contract remain unimplemented this session.

## 8. Conclusion

One legitimate, verified contribution (mask contract) was recovered, reconciled onto
`origin/main` without overwriting other agents' work, tested (10/10 new tests), pushed to
GitHub, and verified remotely. The rest of the six-agent work is either already in main
or remains in local-only branches/stashes owned by other agents. No fabricated results,
accuracy claims, agent utilization, or pushes are reported.
