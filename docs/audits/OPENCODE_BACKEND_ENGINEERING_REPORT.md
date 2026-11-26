# OpenCode Backend Engineering Report — High-Intensity Directive

**Executed by:** OpenCode primary agent. **Date:** 2026-11-26.
**Base commit (recorded before any change):** `origin/main` = `5b33b7b9`.

## 1. Agency actually activated

| Agent | Session | Work | Result |
|---|---|---|---|
| Independent reviewer (general subagent) | `ses_de72ee8d9ffez47FWizVIKSPML` | Static review of PR #397/#398 diffs | **COMPLETED** — found MAJOR production-wiring gaps in #397; APPROVE (with corrections) for #398 |
| 8 earlier workstream subagents (prior session) | — | — | all failed "free usage limit"; zero work |

Skills: `find-skills` (guidance only). No implementation skill used.

## 2. Independent audit & recovery (Tasks 1)

- `origin/main` initially `5b33b7b9`; now `ae037211` after the repo owner merged PRs #397/#398 (I did NOT merge anything).
- **stash@{6} recovery analysis**: based on old main `34be8ea7`. Its jobs.py reconstruction path is superseded by main's mature 2340-line jobs.py (heartbeats, leases, stale reap, reclamation, GPU release — all verified present). Its WorldIR/cameras/points/diff endpoints are already in main (`world_worldir`, `world_cameras`, `world_points`, `get_world_diff`); its CLI `incremental-update` is buggy (passes `reconstruction=None` while claiming "run internally", and jobs.py had undefined `mapping`). **Decision: do NOT apply** — it would regress main. Recorded as SUPERSEDED/incomplete; nothing blindly popped.
- Other agent branches/stashes preserved untouched.

## 3. Delivered branches & PRs

| PR | Branch | Head | State | Purpose |
|---|---|---|---|---|
| #397 | recovery/opencode-mask-contract | `768f1de2` | MERGED (by repo owner) | Mask contract: ObservationRecord.mask/mask_ref + RLE persistence |
| #398 | security/upload-path-traversal | `fa6d8985` | MERGED (by repo owner) | Upload staging path validation + finalize defense-in-depth |
| #399 | fix/mask-production-wiring | `0565502c` | OPEN | Fix review findings: durable sha256:// mask store, bridge carries mask_ref, dedup constants |
| #401 | fix/review-followups | `a44bf11c` | OPEN | Correct brittle CI assertion + misleading auth comment |

No force-push to main; no unauthorized merges by me. All push verifications via `git ls-remote`.

## 4. Exact LOC accounting (net-new vs base)

| Branch | Production | Tests | Docs | Total |
|---|---:|---:|---:|---:|
| #397 (merged) | 133 | 271 | 21 | 425 |
| #398 (merged) | 100 | 136 | 151 | 387 |
| #399 (open) | 90 | 15 | 0 | 105 |
| #401 (open) | 0 | 9 | 0 | 12 |
| **Total** | **323** | **431** | **172** | **926** |

Far below the 200,000 ambition; no inflation. Each line has a purpose and a test.

## 5. Tests actually executed

| Command | Result |
|---|---|
| mask + adapter + matrix bridge (PR #399 branch) | 22 passed |
| detection pipeline + upload security + resumable upload (PR #401 branch) | 43 passed |
| previously failing `test_run_for_job_executes_the_pipeline_and_reports_what_it_did` | 1 passed (was failing) |
| full regression earlier (9 modules) | 123 passed, 1 skipped, 1 pre-existing failure — now fixed in #401 |
| security tests vs unmodified main storage.py | collection error (proves new behavior) |
| finalize owner-auth test vs unmodified main routes_misc.py | still passes (does NOT prove exploit; reported honestly) |

## 6. Key findings (honest)

- **PR #397 review found my claim false**: masks did not reach persistent consumers. Fixed in #399 — mask store now content-addressed (`sha256://`), matrix bridge carries `mask_ref`/descriptor/URI into world lineage.
- **PR #398 review found the "auth fix" redundant**: the uploads router already had `upload_scope`. I corrected the claim and comment; code kept as harmless defense in depth.
- **stash@{6}** substantially superseded by main; its novel CLI command was buggy — not recovered.
- The one full-suite failure was environment-dependent test brittleness, now fixed.

## 7. Production blockers remaining

- Human-dependent (RELEASE.md): six measured rooms, 15-image annotation manifest, branch protection on main.
- No multi-tenant SaaS readiness claim: auth is loopback/local-first by default.
- Real-SAM2 end-to-end not exercised (tests use fakes); no accuracy metrics claimed.

## 8. Conclusion

Recovered one valid contribution, fixed its review-identified production gaps, hardened the upload boundary, and repaired a brittle CI test. Three PRs are open/merged with exact, reproducible test evidence. Functional verification took precedence over the LOC target.
