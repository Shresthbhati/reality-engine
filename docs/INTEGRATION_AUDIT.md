# Integration Audit — Canonical API Boundary & Delivery Gate

**Date:** 2026-09-22 · **Branch:** `feat/canonical-api-client` (based on `main` @ `2642bac`)
**Scope:** mechanical verification only — every claim below was checked with a command
(typecheck, lint, build, git, `gh`, route greps), not narrative.

## 1. Delivery lifecycle status

| Step | Status | Evidence |
| --- | --- | --- |
| PR #84 (canonical API migration: 8 desktop pages + `src/lib/api/*`) | **MERGED** | `gh pr view 84` → `MERGED` |
| CI on #84 | GREEN | Core Tests (3.10/3.11/3.12) + "Lint & Type Check" all `SUCCESS` |
| Follow-up fixes in this session | committed on `feat/canonical-api-client` | see §4 |
| `npx tsc --noEmit` | **PASS (exit 0)** | run after fixes |
| `npm run lint` (eslint) | **0 errors**, 3 warnings | config added this session — see §5 |
| `npm run build` (next build, Turbopack) | PASS/see §4 | after UTF-8 + RSC fixes |
| Frontend test suite | **DOES NOT EXIST** | only `dev/build/start/lint` scripts; no vitest/jest/playwright config |

## 2. Canonical data boundary (`@/lib/api`) — verified state

**On `@/lib/api` (canonical):** all rewired desktop pages — sessions list & detail,
evidence list & detail & view, analysis list & detail & new, results list & detail;
plus `lib/search.ts` (CommandPalette search).

**Still on `@/lib/data` (stale static boundary) — 14 files:**

- Desktop (6): `reports/page.tsx`, `reports/new/page.tsx`, `reports/[id]/page.tsx`,
  `reports/[id]/edit/page.tsx`, `sessions/[id]/map/page.tsx`, `worlds/new/page.tsx`
- Mobile (8): `m/page.tsx`, `m/camera/page.tsx`, `m/sessions/page.tsx`,
  `m/sessions/[id]/page.tsx`, `m/evidence/page.tsx`, `m/evidence/[id]/page.tsx`,
  `m/worlds/page.tsx`, `m/worlds/[id]/page.tsx`

`lib/data.ts` was emptied to real empty arrays on `main` (`db02a34`), so these
consumers render permanently-empty lists, and detail routes that call `notFound()`
on a miss (`sessions/[id]/map`, `reports/[id]`) **always 404**. This is the single
largest open item for product coherence: **desktop/mobile parity on the canonical
boundary is not achieved**. Per gate scope (review/audit, no frontend rewrites)
these are listed here, not silently rewritten.

## 3. Backend ↔ adapter contract

Every endpoint the adapters call exists in `apps/api`:

| Adapter call | Backend route |
| --- | --- |
| `listSessions()` | `GET /api/sessions` (`routes_sessions.py:180`) |
| `getSession(id)` | `GET /api/sessions/{id}` (`routes_sessions.py:203`) |
| `listWorlds()` / `getWorld()` | `GET /api/worlds[/{id}]` (`routes_worlds.py:58/84`) |
| `listEvidence({session_id})` | `GET /api/evidence?session_id=` (`routes_misc.py:125`) |
| `getEvidence(id)` | `GET /api/evidence/{id}` (`routes_misc.py:136`) |
| activity feed | `GET /api/activity` (`routes_jobs.py:90`) |
| Analysis / Results / Reports | **no backend by design** → `unsupportedResource()` / `getAnalysisDetail` / `getResults` / `getReports` return explicit `{available:false, reason, items:[]}`; pages render the reason, never mock rows. `unsupportedForAnalysis()` added this session (was imported but missing). |

Note: another agent has **uncommitted WIP** on `apps/api/routes_misc.py`,
`routes_worlds.py`, `tests/test_application_api.py` in this worktree — untouched
by this gate (not ours to commit).

**WorldIR:** canonical form is the Python `world_ir` package; the translation
layer is the FastAPI application mirror (`apps/api` World/WorldVersion DTOs →
`WorldDto` → `toWorldRow()`). The frontend has no direct WorldIR coupling and
none is needed; UI naming ("World"/"World version") maps to the application
mirror, not to WorldIR JSON. No redesign required.

## 4. Defects found & fixed this session

1. `sessions/[id]/page.tsx` — imported `unsupportedForAnalysis` that **did not
   exist**; `never[]`-typed `analysis`/`results` crashed `.map()` typing;
   `evidence` variable undefined; missing `EmptyState`/`FileText` imports. Fixed.
2. `results/[id]/page.tsx` — `await` inside non-async `ProvenanceChain`
   (compile error); `analysis?.name` on a const narrowed to `undefined`
   (`never` type error); missing `EmptyState`; unused `notFound`/`getEvidence`.
   Evidence count hoisted into the async page; `null` = "API didn't answer" →
   segment omitted, never a fabricated count. Fixed.
3. `CommandPalette` — `searchAll()` is async but was assigned through a sync
   `useMemo` (`Promise<SearchResult[]>` where `SearchResult[]` expected).
   Rebuilt as query-tagged async state with stale-response cancellation and
   react.dev adjust-state-during-render resets. Fixed.
4. `maplibre-gl@6.9.1` declares `dist/maplibre-gl.d.ts` but ships **no `.d.ts`
   at all** → `TS7016` across map components. Added `@types/maplibre-gl`
   (`^1.13.2`, package.json + lockfile). Fixed.
5. `results/page.tsx` contained an invalid UTF-8 byte at offset 1055 (`0x85`,
   a cp1252 `…` written by an earlier Python fixer script) — Turbopack refused
   to parse it, so **`next build` failed outright**. Repaired; full `src/` scan
   shows no other invalid bytes. Also missing the `"use client"` directive
   (invisible to tsc/eslint, fatal to the build) — added; project-wide scan
   shows no other hook-using file lacks it.
6. Deleted this branch's leftover scaffolding: `frontend/_rw_*.py`, `_dbg*.py`,
   `_fix_analysis.py`, `rewrite_analysis_page.py` (untracked one-shot scripts).

## 5. Lint did not exist — now it does

`npm run lint` was bare `eslint` with **no config anywhere** (ESLint ≥9 requires
`eslint.config.*`) — it had never passed. Added `frontend/eslint.config.mjs`
(standard `eslint-config-next/core-web-vitals` + `/typescript` flat config).

Result: **0 errors**. Three warnings remain, all in files outside this gate's
scope (left for their owners; warnings do not fail the run):

- `worlds/[id]/WorldWorkspaceClient.tsx:86` — `react-hooks/exhaustive-deps`
- `m/camera/page.tsx:117,164` — `@next/next/no-img-element`

Rule-driven fixes landed in: analysis/evidence/sessions list pages (redundant
sync `setState` at effect start removed; retry button sets loading in the event
handler), `WorldMap` (latest-callback refs updated in an effect, not during
render), `settings` and `m/camera` (documented one-line `eslint-disable`s for
SSR-hydration/client-only reads that cannot use a lazy initializer without HTML
mismatch), `sessions` load effect (documented disable: all `setState` sits
post-`await`; the rule cannot see through the `useCallback` indirection).

## 6. CI gaps (documented, not silently expanded)

`.github/workflows/ci.yml` runs **Python only** — the job named "Lint & Type
Check" runs `ruff --exit-zero` + `mypy || true`; **nothing typechecks, lints, or
builds `frontend/`**. That is why PR #84 merged green while the branch did not
pass local `tsc`. Recommended follow-up (not done here to avoid landing infra
changes outside gate scope): add a `frontend` job —
`npm ci && npx tsc --noEmit && npm run lint && npm run build`.

## 7. E2E chain readiness (Evidence → Session → Processing Job → World →
Version → Entity → Evidence/Provenance)

- **Works:** sessions/evidence/worlds CRUD list+detail pages are on the
  canonical client; world-versions endpoint exists
  (`GET /api/worlds/{id}/versions`); provenance chain renders only segments the
  real data resolves (unknown counts omitted, never guessed).
- **Blocked segments:** Analysis/Result/Report (no backend — explicitly surfaced
  as unsupported), the 14 stale `@/lib/data` consumers (§2), and **zero
  automated frontend tests** (no runner exists — the "tests" step of the
  delivery lifecycle cannot be executed until one is added).
- Repo junk noted (not ours to remove): `MagicMock/`, `baseline_pytest*.txt`,
  `push_out.txt`/`push_err.txt` — the latter records a prior push rejected for a
  211 MB `_datasets_dl/.../database.db` (needs Git LFS or history cleanup before
  that data can ship).

## 8. Other agents

Branches exist for antigravity, opencode, kilocode, claude, freebuff and cline
worktrees; **no open PRs** at audit time (freebuff #86/#87 and claude #85 are
merged into `main` and already incorporated here via the fast-forward merge).
Nothing else is delivered to integrate yet.

