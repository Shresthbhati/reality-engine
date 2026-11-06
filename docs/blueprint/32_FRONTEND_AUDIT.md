# 32 — Frontend audit and library matrix (build-plan U.1, U.2)

Written 2026-11-06 at `08047317` (origin/main). Docs only: nothing was installed, built or changed.
Read-only inputs: the tree at that commit, `frontend/package-lock.json`, `docs/openapi.json`, `.github/workflows/*`,
`npm view <pkg>` (registry metadata only, no install) and one docs lookup (shadcn changelog).
Companion to [14_FRONTEND_UX.md](14_FRONTEND_UX.md) (intent) and the older [STUDIO_APPLICATION_AUDIT.md](../audits/STUDIO_APPLICATION_AUDIT.md) (2026-11-05, which this file corrects in §9).

**Evidence tags.** **[O]** observed this session (file read, grep, script or registry output). **[I]** inferred from observed facts; not run.
**[N]** not verified. Nothing was run in a browser, `node_modules` is absent in this worktree, and no build, lint, typecheck or test was executed;
everything about runtime behaviour is therefore [I] or [N].

## 1. Summary

1. One Next.js app (`frontend/`), App Router, React 19, Tailwind 4, three.js; ≈46k lines in 261 files under `src/` [O].
2. 39 page routes (29 desktop, 10 under `/m`); 11 of them return 404 unless `NEXT_PUBLIC_SHOW_UNFINISHED=1` [O]. 28 Next route handlers proxy to FastAPI [O].
3. The browser reaches the backend by **two transports**: direct to `:8100` (`lib/api/client.ts`) and same-origin Next proxies. Only 10 of the 28 proxies have a browser caller (an 11th is exercised only by a test harness) [O]; 4 proxy a backend path that does not exist and have no caller [O].
4. One real broken link: `atlasUrl()` builds `/api/worlds/{id}/artifacts/{sha}` against the Next origin, where no such route exists (the backend route does) [O route absence; the 404 itself is [I]].
5. Design tokens are a single CSS-variable set in `globals.css` (good), but the components bypass Tailwind's theme: 1,648 `bg-[var(--…)]`-style classes, 1,016 inline `style={{}}`, 780 raw palette classes in 28 files [O].
6. Already installed and **unused**: `zustand` (0 imports) [O]. `clsx` and `tailwind-merge` are used through one `cn()` helper (8 call-site files) [O].
7. Two UIs for one world: "Place" (Simple) and "Workstation" (Pro), switched by a localStorage flag [O]. This is a third noun beside Session and World.
8. Hand-rolled where a library normally owns it: data fetching (`useApi`, 131 lines), dialogs (11 `role="dialog"`), command palette (426 lines), upload queue (≈530 lines), SVG charts, fixed-width panes [O].
9. Tests: 9 Playwright specs (≈25 tests), 54 node contract files (≈389 tests) that **no CI workflow runs**, 3 pytest guards, no accessibility test [O].
10. Library decision (§8): adopt **shadcn/ui on Base UI**, **Allotment** and **TanStack Query**; use the installed **Zustand**, **Lucide** and **three**; approve-but-defer 6 more; reject Drei, Dockview, Uppy, Sonner, React Aria.

## 2. Stack facts [O]

| Item | Value | Source |
|---|---|---|
| Framework / router | Next.js `16.3.5` (pinned), App Router only, no `pages/` | `frontend/package.json`, `src/app` |
| React | `19.2.8` (pinned) | package.json |
| Language | TypeScript `^5.9.3` (locked 5.9.3), `strict` | package.json, `tsconfig.json` |
| Styling | Tailwind `^4.1.16` (locked 4.3.3) via `@tailwindcss/postcss`; tokens as CSS variables in `src/app/globals.css`; **no `@theme` block**, no `tailwind.config` | `postcss.config.mjs`, globals.css |
| Themes | Dark default; light via `data-theme` or OS; two light blocks that the file says "must stay identical" | globals.css |
| Fonts | System stack; `--font-mono` names JetBrains Mono but nothing loads it ([I] it falls back) | globals.css:51; no `next/font` import |
| 3D | `three ^0.186.0` used **imperatively** (`lib/viewport/three-scene.ts`, 2,835 lines); `OrbitControls` and `TransformControls` from `three/addons`; no `@react-three/*` | grep: 3 `three` imports |
| Maps | `maplibre-gl ^6.10.0` (2 files); style fetched from `demotiles.maplibre.org` (needs internet; [I] conflicts with the offline desktop build) | `components/map/*` |
| Client state | `useState`/`useReducer`/`createContext` (one file uses context); localStorage prefs (`lib/prefs.ts`); `zustand` installed, **0 imports** | grep |
| Data fetching | Hand-rolled `useApi` (`lib/api/hooks.ts`, 131 lines): no cache, no dedupe, no invalidation; ad-hoc `setInterval` polling | file read |
| Upload | Multipart `POST /api/uploads` (`lib/api/evidence.ts`), XHR queue for mobile (`lib/mobile/upload*.ts`), drop zone (`PhotoDropZone.tsx`, 266 lines) | file read |
| Charts / tables | Hand-written SVG; 3 `<table>`; no chart or table library; no list virtualization | grep |
| Forms | 5 `<form>`; hand validation; server 422 shown verbatim (e2e asserts it) | grep, `world.spec.ts` |
| Toasts | None; inline `role="status"`/`aria-live` only | grep |
| Shells | `AppShell` (desktop) and `MobileShell` (`/m/*`) | `app/(desktop)/layout.tsx`, `app/m/layout.tsx` |
| Hosting | `next dev` on `:3100` via `launch_ui.bat`; `Dockerfile.frontend` (node 20); Electron loads the Next standalone server (`REALITY_DESKTOP=1`) | `launch_ui.bat`, `desktop/README.md`, `next.config.ts` |
| Other front-ends | `apps/studio` (Vite) is a 4-file skeleton ("Not yet implemented"); it depends on `@reality-engine/worldir`, which does not exist in the repo. `desktop/` is an Electron shell, not a UI. | `apps/studio/README.md`, no `packages/` dir |

## 3. Installed packages (from `frontend/package.json` + lockfile) [O]

Licences are the `license` field in `package-lock.json`. "Files" = source files under `src/` importing it.

| Package | Spec → locked | Licence | Role | Files |
|---|---|---|---|---|
| next | 16.3.5 | MIT | framework | – |
| react / react-dom | 19.2.8 | MIT | runtime | – |
| three | ^0.186.0 → 0.186.0 | MIT | 3D (+ `@types/three`) | 2 `from "three"` + 3 addon imports |
| maplibre-gl | ^6.10.0 → 6.10.0 | BSD-3-Clause | maps | 2 |
| lucide-react | ^0.475.0 → 0.475.0 | ISC | icons | 97 |
| zustand | ^5.0.15 → 5.0.15 | MIT | declared "keep" in 14, **unused** | **0** |
| clsx, tailwind-merge | ^2.1.1, ^3.7.0 | MIT | `cn()` | 1 (`lib/utils.ts`), `cn(` used in 8 |
| tailwindcss, @tailwindcss/postcss | ^4.1.16 → 4.3.3 | MIT | styling (dev) | – |
| @playwright/test | 1.63.0 | Apache-2.0 | e2e | `e2e/*` |
| eslint, eslint-config-next, plugins | 9.39.5 / 16.3.5 | MIT | lint (dev) | – |
| typescript | 5.9.3 | Apache-2.0 | (dev) | – |
| @types/maplibre-gl | ^1.13.2 → 1.13.2 | MIT | types for maplibre **v1** while `maplibre-gl` is v6 ([I] redundant; verify before removing) | – |

Licence profile of all 485 locked packages [O]: MIT 372, Apache-2.0 38, ISC 26, MPL-2.0 13, LGPL-3.0-or-later 10 (+4 Apache-AND-LGPL), BSD 14, other 8.
No GPL/AGPL. Transitive notes that matter for the **desktop installer** [I: whether they are packed depends on the `electron-builder` config, unchecked]:
`@img/sharp-libvips-*` and `@img/sharp-*` (LGPL-3.0-or-later, optional platform binaries of Next's image optimizer; the app uses no `next/image` [O]);
`lightningcss*` and `axe-core` 4.13.0 (MPL-2.0); `caniuse-lite` (CC-BY-4.0). None is new to this track.

## 4. Routes → backend APIs

### 4.1 Two transports [O]

- **Direct**: `lib/api/client.ts` → `${apiBaseUrl()}/api/...`, where the base is `window.__REALITY_API_BASE__` (Electron) or `NEXT_PUBLIC_API_BASE_URL` or `http://localhost:8100`. Used by every `lib/api/*` function. Server components use the same client, so they also target `:8100` unless the env says otherwise (the prior audit's fix #6 [N whether fixed]).
- **Next proxy**: same-origin `/api/...` handled by `src/app/api/**/route.ts` (28 files), forwarding to `REALITY_BACKEND_URL` (default `:8100`) with the server-side token (`lib/api/backend.ts`).

Backend route set: 118 paths in `docs/openapi.json` [O]; I extracted 124 (method, path) pairs from the router decorators for the comparison.

### 4.2 Page routes (static trace of imports → `lib/api` functions; approximate: [I])

Pages with no row call no API.

| Route | Backend calls | Notes |
|---|---|---|
| `/` Home | GET `/api/worlds`, `/api/status`, `/api/jobs`, `/api/activity?limit=6` | `HomeDashboard` via `useWorlds`, `useApi` |
| `/worlds` | GET `/api/worlds?status=`; POST `archive`, `restore`, `duplicate`; PATCH, DELETE `/api/worlds/{id}` | server fetch; Active/Archived/Trash tabs |
| `/worlds/new` | POST `/api/worlds`, PUT `…/intent`, POST `/api/intent/estimate`, `…/attach/{session}`, POST `/api/reconstructions` | `WorldIntentWizard` |
| `/worlds/[id]` | Simple: GET `…/quality`, worldir, evidence artifact. Pro: worldir, versions, diff, report, capture-coverage, provenance, query/*, export + formats, commit, procedural (rooms, buildings, districts, cities, apartments, corridors, generated, templates), jobs/cancel, reconstructions, `sessions/{id}/reconstruct`. **Via Next proxy**: `worldir`, `points`, `cameras`, `report`, `space-graph`, `jobs/{id}`, `technologies`, evidence image | `WorldPageSwitch` → `PlaceViewer` or `SpatialWorkstation` |
| `/worlds/[id]/pipeline` | GET `…/report`; proxy `jobs/{id}` | linked only from `PipelineGraph` (Pro) |
| `/worlds/[id]/proposals` | `…/proposals` (list/get/create/accept/reject) | linked only from `WorldNavPanel` (Pro) |
| `/capture` | GET `…/capture-requests`, PATCH `…/capture-requests/{id}`, GET sessions, POST `/api/reconstructions` | |
| `/reconstruct` | POST `/api/reconstructions`, POST `/api/sessions/{id}/reconstruct`, GET `…/status` | Home "Upload Data" goes here; nav "Capture" goes to `/capture`. Plan row 11.4 says they were merged; both pages exist [O] |
| `/sessions`, `/sessions/new`, `/sessions/[id]`, `/sessions/[id]/map` | GET/POST `/api/sessions`, GET `…/{id}`, `…/location`, `…/trajectory`, `/api/evidence`, `/api/jobs/{id}`, POST `worlds/{id}/attach/{session}` | |
| `/evidence`, `/evidence/[id]`, `/evidence/[id]/view` | GET `/api/evidence`, `/api/evidence/quality`, `/api/evidence/{id}`, `…/quality`; PUT `…/decision` | |
| `/sites`, `/sites/[id]` | `/api/sites` CRUD, `…/anchor`, `…/compose`, `…/placements`; POST `/api/geo/context` | |
| `/activity` | GET `/api/activity` | |
| `/settings` | **none** (local prefs only) | Processing/Data are localStorage, no backend setting [O: no `apiGet/apiPut` in the file] |
| `/analysis`, `/analysis/new`, `/analysis/[id]` | none real (`lib/api/analysis.ts` returns "unsupported") | 404 unless `SHOW_UNFINISHED` |
| `/results`, `/results/[id]` | none real | 404 unless `SHOW_UNFINISHED` |
| `/reports`, `/reports/new`, `/reports/[id]`, `/reports/[id]/edit` | none real; `lib/data.ts` empty arrays | 404 unless `SHOW_UNFINISHED`. The backend serves report/quality/measurements/export; unwired |
| `/team` | none | 404 unless `SHOW_UNFINISHED` |
| `/m`, `/m/sessions`, `/m/sessions/[id]`, `/m/evidence`, `/m/evidence/[id]`, `/m/worlds`, `/m/worlds/[id]`, `/m/more` | GET sessions, evidence, `/api/evidence/quality`, worlds, `…/versions`; DELETE evidence; POST `…/reconstruct` | server components |
| `/m/camera` | POST `/api/uploads` (+ `/api/uploads/check`), polling | image-only |
| `/m/team` | none | 404 unless `SHOW_UNFINISHED` |

### 4.3 The 28 Next route handlers [O]

| Status | Handlers |
|---|---|
| **Used by a caller** (11: 10 in the browser, `sessions` GET only by the contract harness) | `worlds/[id]/worldir`, `…/points`, `…/cameras`, `…/report` (export + `fetchWorldPoints` + viewer), `…/space-graph` (`WorldNavPanel`), `jobs/[jobId]` (`PipelineGraph`, `RoomConstructionModal`), `jobs/[jobId]/cancel`, `technologies`, `technologies/benchmark` (`TechnologiesPanel`), `worlds/[id]/evidence/[evidenceId]/image` (`AdaptiveInspector`, `InsightsPanel`), `sessions` GET (contract harness) |
| **Forward to an existing backend route; no browser caller found** (13) | `worlds` GET/POST, `worlds/[id]` GET/PATCH/DELETE, `archive`, `restore`, `duplicate`, `commit`, `diff`, `versions`, `export`, `export/formats`, `export/[hash]/download`, `entities/[entityId]/provenance`, `sessions/[sessionId]/reconstruct`. The matching `lib/api` functions call `:8100` directly [I: no relative-URL caller found by grep] |
| **Forward to a backend path that does not exist; no caller** (4) | `jobs/compile` → `/api/jobs/compile`; `jobs/ingest` → `/api/jobs/ingest`; `sessions/ingest-mobile`; `sessions/mobile-tasks`. None of the four paths is in `docs/openapi.json` (118 paths). Only the CLI has `ingest-mobile`/`mobile-tasks` verbs (`apps/cli/main.py:1161,1169`) |

The pytest guard `test_frontend_bridge_proxies_a_real_api_contract` does not catch these: it checks a hand-written table (`jobs/ingest/route.ts` → `/api/uploads`, test line 70) and never reads the URL inside the route file (`jobs/ingest/route.ts:17` targets `/api/jobs/ingest`) [O].

### 4.4 Backend routes with no UI

The prior audit lists them (rollback, world package import/export, resumable upload, job SSE, assistant, registration, reclassify/split/merge wrappers, `GET /api/worlds/{id}/query/spaces`, `/api/system/verification`). I did not re-derive that list; treat it as [N] for today's tree.

## 5. Duplicates and competing systems

| # | Finding | Evidence | Verdict |
|---|---|---|---|
| D1 | **Two world UIs**: Simple "Place" (`components/simple/*`, `lib/places.ts`, `lib/copy.ts`; `PlaceViewer` 367 lines) and Pro "Workstation" (`workspace/*`, `navigation/*`, `inspector/*`; `SpatialWorkstation` 966 lines plus files of 1,300–3,000 lines), chosen by localStorage `re-pro-mode` | `WorldPageSwitch.tsx`, `lib/prefs.ts:19` [O] | Product decision for U.6: one viewer shell with progressive disclosure, or two. "Place" is a third noun; the product model is Session → World |
| D2 | `ui/WorldCard` (used by `/worlds`) vs `simple/PlaceCard` (used only by `PlaceLibrary`, which nothing imports) | grep [O] | Delete `PlaceLibrary` + `PlaceCard`, or merge into one card |
| D3 | Unreferenced files: `components/shell/MobileSheet.tsx`, `components/simple/PlaceLibrary.tsx`, `lib/use-media-query.ts`. Test-only: `lib/api/commit-outcome.ts`, `lib/studio/measure-model.ts`. `MODE_TABS` is exported and tested but only feeds `activeModeId`; no component imports either | name-based import scan + grep [O]; the scan can miss dynamic imports [I] | Candidates for a cleanup task |
| D4 | **Two shells and two route trees**: `AppShell` (Sidebar, TopBar, StatusFooter) and `MobileShell` (bottom tabs), with a parallel `/m/*` set of 10 pages re-implementing sessions/evidence/worlds lists | layouts, `components/mobile/*` [O] | Fine while `/m` is "capture + view companion" (14); do not restyle both in U.3–U.6 |
| D5 | **Four status/pill implementations**: `ui/StatusBadge` (sessions/analysis), `simple/bandStyle` (quality bands), `EvidenceStatusPill` (in `ui/EvidenceCard.tsx`), `ReconstructionStatusBar` (708 lines) | imports [O] | One `Badge` primitive in U.3 |
| D6 | **Two measurement models**: `lib/viewport/measure-model.ts` (225 lines, used by the app) and `lib/studio/measure-model.ts` (358, imported only by its test; it re-imports the viewport one) | grep [O] | Layered, not duplicated, but the Studio layer is not used by any component |
| D7 | Type homes: `lib/types.ts`, `lib/api/types.ts`, `types/worldir.ts`; `lib/activity.ts` is a pure re-export of `lib/api/activity.ts`; `*-model.ts` beside `*.ts` in `lib/api` | file list [O] | Consolidate only if touched |
| D8 | Styling bypass: raw Tailwind palette classes (780 uses in 28 files, e.g. `text-neutral-400`), 17 hard-coded hex values in 7 files, 1,016 inline styles, 1,648 `[var(--…)]` arbitrary classes | grep [O] | U.3 must map tokens into `@theme` first, then migrate by directory |
| D9 | Competing design **systems**: none. One token set, one icon set (Lucide), one `cn()`. The risk is the opposite: no component library, so every dialog/menu/tab is bespoke | §2 [O] | – |
| D10 | Monoliths (prior audit, sizes re-checked [O]): `WorldNavPanel` 3,063, `three-scene` 2,835, `AdaptiveInspector` 2,075, `World3DViewport` 1,309, `SpatialWorkstation` 966 lines | `wc -l` | Split before restyling (U.6) |

## 6. Disconnected, 404 and unreachable workflows

| # | Finding | Evidence |
|---|---|---|
| W1 | `atlasUrl()` emits `/api/worlds/{id}/artifacts/{sha256}` relative to the Next origin. There is no `app/api/worlds/[id]/artifacts` handler and `next.config.ts` has no rewrites. The backend route exists (`GET /api/worlds/{world_id}/artifacts/{sha256}`). Textured shell parts would 404 when a WorldIR carries an `artifact://` atlas. Not run. | `lib/viewport/room-shell-model.ts:323–327`, `three-scene.ts:1088` [O]; effect [I] |
| W2 | 4 dead proxies (§4.3) and 13 unneeded ones: maintenance surface with no user | §4.3 [O] |
| W3 | 11 pages 404 by default (`analysis*`, `results*`, `reports*`, `team`, `m/team`). Navigation is now filtered (`visibleNav`, `isGatedHref`, TopBar Export guard, Home quick-start filter), so the dead links reported on 2026-11-05 are gone | `Sidebar.tsx:22`, `TopBar.tsx:122`, `HomeDashboard.tsx:483` [O] |
| W4 | Reports has no screen although the backend serves `report`, `quality`, `measurements`, `export`; blueprint 14 assigns them to Reports | prior audit §3 + routes above |
| W5 | Reachable only from Pro-mode internals: `/worlds/[id]/pipeline`, `/worlds/[id]/proposals`; Simple mode cannot reach either | links found only in `PipelineGraph.tsx:206`, `WorldNavPanel.tsx` [O] |
| W6 | `/capture` and `/reconstruct` both exist and cross-link, while build plan 11.4 says they are one merged screen | tree + plan [O] |
| W7 | Settings "Processing" and "Data" are localStorage, no backend | §4.2 [O] |
| W8 | Resumable upload (`/api/uploads/resumable/*`), job SSE (`/api/jobs/{id}/events`), rollback, package import/export have no UI | prior audit §3b; routes [O] |
| W9 | The MapLibre style is fetched from the internet; the offline desktop shows no basemap | `WorldMap.tsx:9` [O]; effect [I] |

## 7. Test coverage

| Layer | What exists | Gap |
|---|---|---|
| Playwright (`frontend/e2e`, 2 projects: desktop 1440×900, mobile 390×844) | 9 spec files, ≈25 tests: route smoke for 11 desktop routes + 5 `/m` routes (no uncaught console errors), gated routes return 404, Home numbers equal the API, honest empty/error states, `/evidence` counters, world canvas renders, Build/Apartment/Building/District flows against the live API with 422 shown verbatim, completion, wizard, layout, shell | Not smoke-tested: `/capture`, `/sites`, `/sites/[id]`, `/sessions/[id]`, `/sessions/[id]/map`, `/evidence/[id]`, `/evidence/[id]/view`, `/worlds/[id]/proposals`, most `/m/*`. **No accessibility check** (no axe in `e2e`). The config serves `:3000` while the launchers use `:3100` |
| Node contract tests (`frontend/tests/contract/*.test.mts`, `npm run test:contract`) | 54 files, ≈389 `test()/it()`: pure view-model logic (inspector, measure, proposals, capture, panels, nav) plus `sessions-proxy.contract.mjs` | **Not run by any CI workflow** [O: grep of `.github/workflows`]. Only `sessions-proxy.contract.mjs` runs, via a pytest wrapper |
| pytest guards | `test_frontend_backend_contract.py` (table-based, see §4.3), `test_frontend_no_fake_state.py` (source-level honesty guards), `test_frontend_sessions_proxy_contract.py` | The table does not read the routes' real targets |
| CI | `ci.yml` frontend job: `npm ci`, ESLint, `tsc --noEmit`, `next build` (Node 22). `e2e.yml`: Playwright on push to `main`/`agent/**`/`feat/**`/`fix/**` and PRs to `main` | No visual regression, no screenshot assertions, no contract run |
| Ad-hoc scripts | `frontend/tests/*.mjs` (adversarial audit, CDP telemetry, GPU lifecycle, spatial stress, browser runtime benchmark) | Manual; not in CI |

Pass/fail of any of the above on this commit is [N]. Test counts come from a `test(`/`it(` regex and are approximate.

## 8. Compatibility and licence matrix (U.2)

Registry data were read with `npm view` on 2026-11-06 [O]. "React" = peer range vs the repo's React 19.2.8. "Size" = registry `unpackedSize`, not bundle size.
Rule applied: prefer what is installed; every add names the code it replaces or the requirement it meets; one choice per responsibility; nothing is installed by this document.

| Responsibility | Candidate | Version | Licence | React 19.2.8 | Size | Fit in this repo | Decision |
|---|---|---|---|---|---|---|---|
| Accessible primitives + styled recipes (dialog, menu, tabs, tooltip, popover, select, drawer, toast) | **shadcn/ui (Base UI flavour)** | CLI `shadcn` 4.23.0 (copies source into the repo; not a runtime dep) | MIT | n/a | 0.2 MB CLI | shadcn changelog: "Base UI as the Default" and a Toast built on Base UI (July 2026) [O, search excerpts; pages not read in full]. Reuses the installed `cn()` (clsx + tailwind-merge). Recipes use `--background`/`--primary` names, so tokens must be mapped via Tailwind `@theme` (U.3); the default shadcn palette is not used | **ADOPT (U.3)** |
|  | Base UI | `@base-ui/react` 1.9.0 | MIT | `^17 \|\| ^18 \|\| ^19` ✓ | 10.4 MB | The runtime package behind the choice above; replaces 11 hand-rolled dialogs, bespoke menus/tabs, and covers the 14 `outline-none` sites to audit | **ADOPT (with shadcn)** |
|  | React Aria Components | 1.22.1 | Apache-2.0 | ✓ | 6.9 MB | Strongest a11y, but a second primitive family beside shadcn | REJECT (fallback if a complex widget, e.g. a date range, is ever needed) |
|  | radix-ui (shadcn "radix" flavour) | 1.8.1 | MIT | ✓ | 0.1 MB | No longer the shadcn default | REJECT |
| Icons | **Lucide** | installed 0.475.0 (registry 1.70.0) | ISC | ✓ | – | 97 files already use it. A 0.x → 1.x jump is a separate, optional chore | **KEEP installed version** |
| 3D scene | **three** (+ `three/addons`) | installed 0.186.0 | MIT | n/a | – | `WorldSceneController` is an imperative class; OrbitControls/TransformControls already imported | **KEEP** |
|  | Drei (+ R3F) | `@react-three/drei` 10.7.9 + `@react-three/fiber` 9.8.2 | MIT | drei `^19` ✓; fiber `>=19 <19.4` ✓ (caps React below 19.4) | 1.8 + 2.4 MB | Needs the 2,835-line controller rewritten as R3F components for no user-visible gain; ties React upgrades to fiber's range | **REJECT** |
| Panel layout | **Allotment** | 1.20.6 | MIT | `^17 \|\| ^18 \|\| ^19` ✓ | 0.2 MB | Today: fixed `w-[248px]`/`w-[320px]` asides and a bottom tablist on small screens (`SpatialWorkstation.tsx:684,758,862`). 14 asks that panels collapse and the viewport dominates: resizable + collapsible splits suffice | **ADOPT (U.6)** |
|  | Dockview | `dockview-react` 8.8.3 | MIT | ✓ | 1.9 MB | Tabbed/floating/dockable windows plus layout serialization is new product scope, not in 14 | REJECT (revisit if user-arranged layouts become a requirement) |
| Client UI state | **Zustand** | installed 5.0.15 (registry 5.1.0) | MIT | `>=18` ✓ | – | Declared "keep" in 14, 0 imports. First consumer: selection / layer / viewport mode shared by tree, viewport and inspector (today passed through components with 27–31 `useState` each, per the prior audit) | **USE (U.6)** |
| Server state | **TanStack Query** | `@tanstack/react-query` 5.106.1 | MIT | `^18 \|\| ^19` ✓ | 1.8 MB | Replaces `useApi` (131 lines) and scattered `setInterval` polling; gives dedupe, cache, invalidation after commit/export, and conditional polling driven by the backend `stage`. Keep the `apiGet…` functions as the fetchers, so the `ApiError` contract and the "never fabricate data" rule are unchanged. Server components keep fetching directly | **ADOPT (U.4/U.5)** |
| Data tables | **TanStack Table** | `@tanstack/react-table` 9.3.1 (new major; check the API at first use) | MIT | `>=18` ✓ | 0.1 MB | Only 3 `<table>` today | **APPROVED, DEFER** until a list needs sort/filter/column control |
| Long lists | **TanStack Virtual** | 3.14.26 | MIT | ✓ | 0.1 MB | No virtualization today; first likely consumers are the hierarchy tree (`WorldNavPanel`) and evidence grids on large worlds [I] | **APPROVED, DEFER** until measured jank |
| Drag and drop | **dnd-kit** | `@dnd-kit/core` 6.3.1 (registry last modified 2024-12-05); `@dnd-kit/react` 0.5.3 (pre-1.0) | MIT | ✓ / ✓ | 1.1 / 0.2 MB | No consumer: only native file drop and pointer-capture drag in `SiteGrid` | **DEFER** (choose between the two at first consumer) |
| Upload | **Keep in-house** | – | – | – | – | The backend resumable protocol is custom form fields (`/resumable/start`, `/chunk` with `offset`, `/{id}/finalize`; `routes_misc.py:246–311` [O]), not tus or S3-multipart, so Uppy would need a custom uploader plugin and saves nothing. The gap is UI, not plumbing | **KEEP** |
|  | Uppy | `@uppy/core` 6.3.1, `@uppy/react` 6.1.1, `@uppy/dashboard` 6.0.3 | MIT | `^18 \|\| ^19` ✓ | 1.1 + 0.1 + 0.7 MB | as above | REJECT |
| Charts | **ECharts** | 6.1.0 (+ `echarts-for-react` 3.0.6) | Apache-2.0 / MIT | wrapper ✓ | 60 MB unpacked | No chart library today (hand SVG in `QualityParts`, `MapPlotView`, `SiteGrid`, `CaptureCoverage`). Needed only for a real Reports screen. If added: import `echarts/core` modules only, lazy-load on the Reports route, keep out of the viewer chunk | **APPROVED, DEFER** |
| Animation | **CSS first; Motion if needed** | `motion` 14.10.0 | MIT | `^18 \|\| ^19` ✓ | 1.0 MB | 166 CSS transition/keyframe hits; `prefers-reduced-motion` is handled in CSS in only 2 places. Base UI exposes CSS enter/exit states [I], so no JS library is required for U.4–U.7 | **DEFER** (if added: `LazyMotion`, `MotionConfig reducedMotion="user"`) |
| Forms | **React Hook Form + Zod** | `react-hook-form` 7.91.0, `zod` 4.6.7, `@hookform/resolvers` 7.0.3 | MIT ×3 | RHF ✓ | 1.6 / 6.1 / 1.2 MB | 5 forms today. The server stays the source of truth (422 shown verbatim, tested); schemas must cover shape and required fields only, not re-implement backend rules. Resolver ↔ Zod 4 compatibility not checked [N] | **APPROVED, DEFER** to the U.5 create-session form |
| Toasts | **shadcn Toast (on Base UI)** | via the primitives row | MIT | ✓ | – | No toast system exists. Use only to confirm an action the backend already completed; errors and progress stay inline (the current `role="status"` pattern) | **ADOPT (with primitives)** |
|  | Sonner | 2.0.8 | MIT | ✓ | 0.2 MB | A second toast system beside the one above | REJECT |
| Command palette | **Keep existing** `CommandPalette.tsx` (426 lines, real `lib/search.ts`) | – | – | – | – | Works against real data; swap only if the U.8 keyboard walkthrough finds combobox/listbox gaps | **KEEP** |
|  | cmdk | 1.1.1 | MIT | `^18 \|\| ^19` ✓ | – | What shadcn's Command wraps | NOT NOW |
| Styling / router / maps / e2e | Tailwind 4 + CSS-variable tokens, Next App Router, maplibre-gl, Playwright | installed | MIT / BSD-3 / Apache-2.0 | ✓ | – | Nothing to choose | **KEEP** |

**Not on the shortlist but required by U.8 ("axe clean")**: `@axe-core/playwright` as a dev dependency. `axe-core` 4.13.0 (MPL-2.0) is already in the lockfile transitively [O]; the wrapper's version and licence were not looked up [N]. It needs its own row before install.

### 8.1 Resulting foundation, one per responsibility

| Responsibility | Foundation | Status |
|---|---|---|
| Framework, routing, styling | Next 16 App Router, Tailwind 4, CSS-variable tokens | installed |
| Primitives, recipes, toast | shadcn/ui on Base UI | **to add (U.3)** |
| Icons | Lucide | installed |
| 3D | three (imperative) | installed |
| Panel layout | Allotment | **to add (U.6)** |
| Client state | Zustand | installed, unused |
| Server state | TanStack Query | **to add (U.4/U.5)** |
| Tables / virtual lists | TanStack Table / Virtual | approved, deferred |
| Drag and drop | dnd-kit | deferred |
| Upload | in-house | keep |
| Charts | ECharts | approved, deferred |
| Motion | CSS; Motion deferred | – |
| Forms | React Hook Form + Zod | approved, deferred |
| Command palette | in-house | keep |
| Maps | maplibre-gl | installed |

Net new runtime packages if all three "adopt" rows land: `@base-ui/react`, `allotment`, `@tanstack/react-query` (all MIT, all peer-compatible with React 19.2.8 [O]). Rejected: Drei, Dockview, Uppy, Sonner, React Aria, the Radix flavour.

### 8.2 Licence verdict

Every shortlisted package is MIT except React Aria (Apache-2.0, rejected), ECharts (Apache-2.0, deferred) and Lucide (ISC, installed) [O]. None is GPL/AGPL/LGPL. Apache-2.0 needs the licence/NOTICE carried into the desktop installer if ECharts is ever added. The LGPL entries in §3 are Next's optional sharp binaries and predate this track.

## 9. Corrections to the 2026-11-05 audit

| Prior statement | This audit |
|---|---|
| "Every frontend `/api` call maps to a real backend route; no dangling call" | True for `lib/api` calls. It missed `atlasUrl()` → `/api/worlds/{id}/artifacts/{sha}` on the Next origin (W1) |
| 4 proxies (`jobs/compile`, `jobs/ingest`, `sessions/ingest-mobile`, `sessions/mobile-tasks`): "backend exists, no frontend caller" | Caller: none (agrees). Backend path: none of the four is in the 118-path OpenAPI (§4.3) |
| Nav/Home/TopBar link to gated Reports | Fixed in the current tree (W3). `MODE_TABS` is not rendered by any component |
| `lib/data.ts` empty arrays consumed by dead reports pages | Still true: imported by `reports/new`, `reports/[id]`, `reports/[id]/edit` only [O] |

## 10. Flags and follow-ups

- **U.1 is only partly done by this file**: the plan's "14 updated with keep/replace per responsibility" is not done. §8.1 is the content to port into `14_FRONTEND_UX.md` (its "Stack (keep)" line still lists zustand as in use).
- Product decisions this audit cannot make: D1 (one world UI or two; "Place" vs "World" vocabulary), W4 (Reports scope), whether `/m` stays a separate route tree.
- Cheap fixes found, not done here (docs-only): add a Next proxy or point `atlasUrl` at the API origin (W1); delete the 4 dead proxies and make the pytest guard read each route's real `BACKEND_URL` target; run `test:contract` in CI; add an axe check to Playwright; remove `PlaceLibrary`/`PlaceCard`/`MobileSheet`/`use-media-query`.
- Everything marked [I] or [N] needs a browser run, a build, or `npm install` to confirm. None was done.

## Appendix — how the numbers were produced

- Routes/files: `git ls-files frontend/src`; pages = `app/**/page.tsx` outside `app/api` (39); handlers = `app/api/**/route.ts` (28).
- Backend routes: regex over `apps/api/*.py` for `APIRouter(prefix=…)` + `@router.<verb>("…")`; cross-checked against `docs/openapi.json` (118 paths).
- Page → API: import closure from each `page.tsx`, mapping imported `lib/api` function names to the path literals inside their bodies; helper-built paths and hooks (`useWorlds`) were completed by hand.
- Usage counts: `grep -rE` over `src/**/*.{ts,tsx}` (imports, `style={{`, `className=`, `var(--`, `[var(--…)]`, palette classes, hex literals, `<table>`, `<form>`, `role="dialog"`).
- Orphans: files in `components/` and `lib/` whose basename is imported by no other source file (basename match; can miss dynamic imports).
- Packages: `frontend/package-lock.json` (`license`, `version`, `engines`); candidates via `npm view <pkg> version license peerDependencies --json` on 2026-11-06.
