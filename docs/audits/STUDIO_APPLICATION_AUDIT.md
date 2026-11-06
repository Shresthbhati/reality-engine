# Reality Studio application audit (frontend vs backend, static)

Date: 2026-11-05. Branch `claude/integration-2026-11-05`, HEAD `4aed118f`.
Method: static reading of `frontend/src` plus a throw-away contract script (kept outside the repo) that extracts every `/api/...` string literal from non-test frontend code and diffs it against `docs/openapi.json`.
**Nothing here was browser-verified, no dev server was started, no test suite was run.** "Verified" means read in code or observed in a command output; everything else says "not verified".

This replaces the previous version of this file, which contained literal `\n` sequences, cited a non-existent `(desktop)/home/page.tsx` (Home is `frontend/src/app/(desktop)/page.tsx`), and listed steps (login, "image upload IMPLEMENTED") with no frontend evidence.

## 1. Contract diff

Observed: `from apps.api.main import app; app.openapi()` yields 118 paths / 138 method-operations, identical to `docs/openapi.json` (empty symmetric difference). Note: `app.routes` on this FastAPI shows `_IncludedRouter` wrappers, so `openapi()` is the only reliable route listing.

Two transports exist: (a) `lib/api/client.ts:20` `apiBaseUrl()` talks to the backend directly (default `http://localhost:8100`); (b) 28 Next route handlers under `frontend/src/app/api` proxy to `REALITY_BACKEND_URL` (`lib/api/backend.ts:8`), reached with same-origin relative URLs.

### 1a. Frontend calls with no backend route

| Call | Result |
|---|---|
| All ~91 distinct (method, path) pairs found by literal match, plus the helper-composed paths (`base(worldId)` in completion/proposals/scale/measurements, `objectPath()` in procedural.ts:138, `?${query}` in capture-requests.ts:41) | **Every one maps to an existing OpenAPI route and method.** Script reported 4 "NO_ROUTE", all false positives (a `{jobId}` doc comment, a `${query}` suffix, a relative image URL handled by the Next proxy, a template built from `KIND_PATH`), each checked by hand. |
| `/api/worlds/{w}/evidence/{e}/image` (AdaptiveInspector.tsx:1918) | Not a backend route by design: Next proxy `app/api/worlds/[id]/evidence/[evidenceId]/image/route.ts:26` forwards to backend `/api/evidence/{id}/artifact`. Works only when the page is served by the Next server (see 1d). |

Conclusion: no dangling frontend call found. Method mismatches: none found at the path level.

### 1b. Backend routes the UI never calls (no frontend reference found by grep)

| Backend route | Note |
|---|---|
| `POST /api/worlds/{id}/rollback` | No UI to roll back to a prior version (journey step 14). |
| `POST /api/worlds/import/package`, `GET /api/worlds/{id}/export/package` | World package import/export (Task 11.2) is API-only. |
| `POST /api/uploads/resumable/{start,chunk,{id}/finalize}` | Resumable upload API-only; UI uses one-shot multipart. |
| `GET /api/jobs/{id}/events` | UI polls instead (reconstructions.ts:285, NewPlaceFlow.tsx:286). |
| `POST /api/worlds/{id}/assistant/{ask,call,capture-plan,confirm}`, `GET /api/assistant/tools` | HomeDashboard.tsx:599-621 shows a disabled "assistant not available" control, correctly labelled. |
| `POST /api/registration/{pose-graph,session-alignments}` | No UI. |
| `GET /api/worlds/{id}/spatial-intelligence` | Panel exists but derives its data from `world.metadata` (World3DViewport.tsx:246-275), never from this endpoint. |
| `GET /api/worlds/{id}/entities/{id}/confidence`, `GET /api/worlds/{id}/query/spaces` | No UI. |
| `GET /api/system/verification`, `GET /api/technologies/{id}`, `/health`, `/healthz` | UI uses `/api/health`, `/api/status`, list + benchmark only. |
| `POST /api/worlds/{id}/corrections/{reclassify,split,merge}` | Client wrappers exist (`lib/api/corrections.ts:23,32,41`, re-exported by `lib/api/index.ts:30`) but **no component imports them** (grep: zero callers). API-only. |
| `/proposals`, `/proposals/{id}`, `/approve`, `/reject` (top-level, 4 ops) | UI uses the world-scoped `/api/worlds/{id}/proposals` set instead. |
| Next proxies `app/api/jobs/compile`, `jobs/ingest`, `sessions/ingest-mobile`, `sessions/mobile-tasks` | Proxy exists, backend exists, **no frontend caller found**. |

### 1c. Shape mismatches provable from code

| Finding | Evidence | Confidence |
|---|---|---|
| **Transform correction payload likely violates the backend contract.** Inspector sends `transform: {...(entity.transform \|\| {}), position}`; backend requires exactly the keys `{position, rotation}` with `rotation` a `{w,x,y,z}` unit-quaternion dict. The frontend `Transform` type is `{position?, rotation?: [n,n,n,n], scale?}` (`types/*.ts:7-11`), and the Python `Transform.to_dict` side is matrix-based. Any extra key (`scale`, frames) or array rotation yields HTTP 422. | AdaptiveInspector.tsx:401; routes_worlds.py:1067-1083 (`set(value) != {"position","rotation"}` at :1069) | Contract mismatch proven by code; whether real worlds hit it **not verified** (no runtime). |
| Correction history entry is pushed **before** the commit succeeds. A rejected commit leaves an undo entry for a change the server never saw. | AdaptiveInspector.tsx:407 (push) vs :444 (await commit) | Proven by code. |

### 1d. Transport risks

| Risk | Evidence |
|---|---|
| Server-rendered pages call `apiBaseUrl()` on the Node side, where `window.__REALITY_API_BASE__` does not exist (client.ts:21-24). The Electron shell sets only `REALITY_BACKEND_URL` for the UI process (desktop/src/main.js:221), not `NEXT_PUBLIC_API_BASE_URL`. So server components (`worlds/page.tsx`, `sessions/[id]`, `evidence/[id]`, all of `/m/*`) fall back to `http://localhost:8100` in the desktop app with a dynamic API port. Not verified at runtime. | client.ts:20-26, main.js:210-224, list of `export default async function` pages |
| Mixed transports: `worldir.ts:29` and SpatialWorkstation export use same-origin Next proxies, `worlds.ts` uses direct backend. Two code paths for the same resource, and a static host with no Next server breaks the proxy half. | worldir.ts:29, SpatialWorkstation.tsx (export handler) |
| Bearer token is `NEXT_PUBLIC_API_TOKEN` (browser-visible) by design for local use. | client.ts:30-33 |

## 2. Fabricated or demo data

| Item | Where | Reaches screen as real? |
|---|---|---|
| `SESSIONS/WORLDS/EVIDENCE/ANALYSIS/RESULTS/REPORTS/PLACES/WORLD_VERSIONS` | `lib/data.ts:9-16` | **All empty arrays**, no seed rows. Only consumed by dead reports pages (below). Safe. |
| `unsupportedResource()` for Analysis/Results/Reports | `lib/api/analysis.ts:12-62`, `unsupported.ts:12-17` | Returns `available:false` + reason text; `reports/page.tsx` renders the reason per tab. Labelled as unavailable. Safe. |
| `reports/new`, `analysis/new` forms | `reports/new/page.tsx` (`onSubmit={e=>e.preventDefault()}`), `analysis/new/page.tsx:34` | Submit does nothing. Analysis "run" button becomes enabled when a session has evidence (analysis/new:103) yet has no handler: a live-looking dead control. Hidden unless `NEXT_PUBLIC_SHOW_UNFINISHED=1` (see 3). |
| `Math.random` / `setTimeout` fake progress | whole `src` (non-test) | None found. `setTimeout/setInterval` hits are real polling (reconstructions.ts:285, NewPlaceFlow.tsx:272-288, ReconstructPanel.tsx:37, RoomConstructionModal.tsx:191, AppShell.tsx:31) or UI timers. Only `Math.random` is a history-entry id (history-stack.ts:32). |
| Next proxies formerly serving `datasets/room_capture/pipeline_out/*` | worlds, cameras, points, report, worldir, provenance, image route headers | Removed; routes now forward to backend and return 502/503 when it is down (comments cite the old behaviour). Safe. |
| `capture_plan`, `temporal_diff` overlays | World3DViewport.tsx:230-232 read `world.metadata.capture_plan` / `.temporal_diff` | No non-test Python writes these keys (grep) and no caller passes the props, so these overlays can never show real data today. Not fabricated, but unreachable: **UI-only**. |
| Spatial-intelligence panel | World3DViewport.tsx:246-275 | Built only from fields present in `worldir` metadata; returns `null` when absent. Honest. |
| Seed fields in builders | `ApartmentBuilder`, `BuildModal`... `seed` | User-entered generation seed, not data. |

No evidence of presented-as-real demo data. Residual risk is dead-but-enabled controls, not fake numbers.

## 3. Per-page table

Default mode is **Simple** (`prefs.ts:19,80-82`: Pro only when localStorage `re-pro-mode === "true"`). `/worlds/[id]` renders `PlaceViewer` in Simple and `SpatialWorkstation` in Pro (`simple/WorldPageSwitch.tsx`). Corrections, inspector, nav tree are Pro-only.

`/analysis`, `/reports`, `/results`, `/team` (and `/m/team`) return 404 unless `NEXT_PUBLIC_SHOW_UNFINISHED=1` (`lib/unfinished.ts:5`; each `layout.tsx:5`). The flag is not set in any repo file.

| Route | Data source | Loading / empty / error / retry | Claim vs backend |
|---|---|---|---|
| `/` Home | `HomeDashboard`: worlds, jobs, activity, storage via real API | Yes / yes / yes / yes (23 loading, 23 retry hits) | Real. **"Create Report" quick action links `/reports/new` (HomeDashboard.tsx:34) which 404s by default.** |
| `/worlds` | server fetch `listWorlds(status)` | Error banner, empty states, tabs Active/Archived/Trash | Real; SSR transport risk 1d. |
| `/worlds/new` | `WorldIntentWizard` -> `POST /api/worlds`, `PUT /intent`, `POST /intent/estimate`, attach sessions | Yes | Real. File picker says "Nothing is uploaded here" (honest). |
| `/worlds/[id]` | Simple: PlaceViewer (`getWorldQuality`, worldir, evidence artifacts). Pro: SpatialWorkstation (worldir, versions, cameras, points, report, provenance, commit, diff, export, capture coverage) | Yes (7 loading/6 error refs) | Real. Overlays capture_plan/temporal_diff dead (sec. 2). Simple mode: no corrections. |
| `/worlds/[id]/pipeline` | `PipelineGraph` -> `/worlds/{id}/report` | No page-level states; relies on component | Real stages. Not verified. |
| `/worlds/[id]/proposals` | `ProposalsPanel` -> world proposals + completion proposals | Panel-level | Real API. |
| `/reconstruct` | `useWorldStatus`, `PhotoDropZone` -> `POST /api/reconstructions` | Yes | Real. Drop zone accepts `image/*` only (PhotoDropZone.tsx:230,244). |
| `/capture` | `listCaptureRequests`, `capture-coverage` | Loading/error, no retry button | Real. |
| `/sessions`, `/sessions/new`, `/sessions/[id]`, `/sessions/[id]/map` | sessions/jobs/trajectory/location API | Yes mostly | Real. `sessions/[id]` hardcodes empty analysis/results with unsupported text, labelled. |
| `/evidence`, `/evidence/[id]`, `/evidence/[id]/view` | evidence + evidence-quality API | `[id]`: no loading UI (server comp, `notFound`) | Real. |
| `/sites`, `/sites/[id]` | sites API (CRUD, placements, anchor, compose) | Yes (19 error refs) | Real. |
| `/activity` | `getActivity` | Yes | Real. |
| `/settings` | local prefs only (theme/density/pro/processing) | n/a | Appearance real; "Processing/Data" are local prefs, **no backend setting** (not verified beyond grep: no apiGet/apiPut in file). |
| `/analysis`, `/results` | redirect to `/reports` (client `router.replace`) | none | No backend resource. Gated 404. |
| `/reports`, `/reports/new`, `/reports/[id]`, `/reports/[id]/edit` | `lib/api/analysis.ts` unsupported + empty `lib/data.ts` | Tabs show reason text | **No backend per this build, confirmed** (analysis.ts header). But backend does expose `/worlds/{id}/report`, `/quality`, `/measurements`, `/export`, which the blueprint (`14_FRONTEND_UX.md`) assigns to Reports. Page is unwired to them. `[id]` pages always `notFound()` (empty arrays). Export/Share buttons permanently `disabled`. |
| `/team` | none (static EmptyState) | n/a | Gated 404. |
| `/m`, `/m/worlds*`, `/m/sessions*`, `/m/evidence*`, `/m/more` | server `list*/get*` | Mixed (1 loading ref each) | Real. |
| `/m/camera` | `CaptureScreen`: `/api/uploads`, queue reducer, polling | Yes | Real; image-only (`accept="image/*"`, CaptureScreen.tsx:354). |

**Dead navigation (verified in code):** `RAIL_ITEMS` "Reports" (shell-logic.ts:40), `MODE_TABS` "Analyze" and "Export" (:28-29), and the TopBar Export button (`EXPORT_HREF="/reports"`, shell-logic.ts:48, TopBar.tsx:122) all point at `/reports`, which 404s by default. No `SHOW_UNFINISHED` filter in the nav. "Compare" tab has `href: null` (shell-logic.ts:27) and is labelled "Not available yet".

### 3b. User journey (mission lists 15 named steps; undo added as 16th)

Legend: E2E = UI + backend both present and wired; UI = UI without backend support; API = backend only; Missing.

| # | Step | Status | Evidence / gap |
|---|---|---|---|
| 1 | Create project | **E2E** | WorldIntentWizard / NewPlaceFlow -> `POST /api/worlds`, `PUT /intent` (worlds.ts:46, intent.ts:25). |
| 2 | Import | **E2E (photos)**; video partial | Photos: `POST /api/reconstructions` (reconstructions.ts:235). Backend extracts video frames (routes_reconstructions.py:53-71) but main drop zone is image-only; wizard/CaptureCoverage accept `video/*`. Resumable upload API-only. |
| 3 | Validation | **Partial** | `POST /api/uploads/check` called only from NewPlaceFlow (NewPlaceFlow.tsx:9,56); Studio `PhotoDropZone` has no pre-flight. |
| 4 | Reconstruct | **E2E** | create + `retryReconstruction` (reconstructions.ts:244), cancel wired (NewPlaceFlow, SpatialWorkstation, mobile). |
| 5 | Progress | **E2E (polling)** | `GET /worlds/{id}/status`, `/jobs/{id}`; stage labels in `lib/places.ts`. SSE events endpoint unused. |
| 6 | Open world | **E2E** | `/worlds/[id]` -> worldir/versions. |
| 7 | Navigate 3D | **E2E** (not browser-verified) | World3DViewport + three-scene; points/cameras via proxy. |
| 8 | Inspect entity | **E2E, Pro mode only** | AdaptiveInspector + provenance proxy. Simple mode has no entity inspector. |
| 9 | Evidence | **E2E** | provenance + `/api/evidence/{id}/artifact`; thumbnails via Next proxy. |
| 10 | Uncertain regions | **Partial** | Uncertainty layer from per-entity `confidence` (three-scene.ts:868-870, toggle World3DViewport.tsx:1075); capture-request list on `/capture`; in-viewport capture plan is UI-only (sec. 2). |
| 11 | Correct | **Partial** | Pro mode; generic field/transform/dimension edits via `POST /commit` (worlds.ts:86). Reclassify/split/merge API-only. Transform payload shape risk (1c). |
| 12 | Persist | **E2E** | Commit mints a WorldStore version; UI refetches versions + IR (SpatialWorkstation.tsx:276-292). |
| 13 | Compare versions | **E2E** | VersionDiffModal -> `GET /worlds/{id}/diff` (VersionDiffModal.tsx:5,52). |
| 14 | Reopen | **Partial** | Worlds list + version selector (`worldir?version=`) work. Rollback and package import are API-only. |
| 15 | Export | **E2E for formats; package API-only** | ExportPanel -> `/export/formats`, `POST /export`, download_url (ExportPanel.tsx:99,124). Export button in the TopBar goes to a 404. |
| 16 | Undo/redo | **UI only, not persisted** | Client `HistoryStack` (history-stack.ts, depth 50). Undo only resets the X/Y/Z input fields (AdaptiveInspector.tsx:228-238) and a preview event; it never calls the backend, so the committed version is unchanged. Only the `transform` mode pushes history (:407). |

### 3c. Where corrections live

Corrections are **persisted server-side** via `POST /api/worlds/{id}/commit` (SpatialWorkstation.tsx:283), which validates fields (`_COMMIT_MUTABLE_FIELDS`, routes_worlds.py:1024) and versions the world. Client-held state is limited to: the edit form, the preview overlay, and the undo stack. Undo is not a persisted operation (no revert-commit), and `parent_version_id` is taken from `versions[0]` (SpatialWorkstation.tsx:282), i.e. the newest version, not the version the user is viewing, which may silently base an edit on the wrong parent when a historical version is selected. Not verified at runtime.

## 4. Component risk (>800 lines)

| File | Lines | Suggested split seams |
|---|---|---|
| `components/navigation/WorldNavPanel.tsx` | 3063 | `HierarchyExplorer` (:1654-2489, 835 lines) to its own file; `BuildingTopologyInspector` (:2489) separate; main component (:117-1558, 1440 lines) into tab bodies; bookmarks storage helpers (:92-115) into a hook. |
| `lib/viewport/three-scene.ts` | 2835 | One `WorldSceneController` class with ~91 methods. Seams already commented: outdoor layer (:1138) and camera/screenshot (:1535); also picking, layer visibility, uncertainty shading (:868). Extract as mixins/modules taking the controller. |
| `components/inspector/AdaptiveInspector.tsx` | 2073 | `EntityDetails` is a single 1678-line function (:163-1841) with 27 `useState`. Extract correction form + `handleSaveCorrection` (:366-460), provenance/evidence section, linked-entities builder, overview facts. |
| `components/viewport/World3DViewport.tsx` | 1309 | Overlay panels (capture plan, temporal diff, spatial intelligence, technologies) and the toolbar (:961-1092) as components; keyboard handler (:645) to a hook. 29 `useState`. |
| `components/workspace/SpatialWorkstation.tsx` | 966 | Data hooks (versions/IR/status/export/commit) vs layout; 31 `useState`. Export handler and commit handler to hooks. |
| `lib/viewport/outdoor-model.ts` | 804 | Pure functions already (35 top-level); split by feature (roads/rail/bridges/trees) if it grows. Lowest risk. |

Also 600-710 lines: ReconstructionStatusBar (708), room-shell-model (664), HomeDashboard (629).

## 5. Accessibility (code-visible only; no screen reader, no axe run)

| Check | Result |
|---|---|
| `aria-label` 229, `role=` 148, `onKeyDown` 12 across tsx | Decent coverage on controls. |
| `<img>` alt | 13 of 13 `<img>` tags have `alt` (multi-line scan). |
| Click-only non-semantic elements | 2: backdrop `div`s (NotificationsPanel.tsx:38, DeletePlaceDialog.tsx:67), acceptable for scrims but need Escape handling (not verified). |
| Dialogs | 15 `role="dialog"`/`aria-modal` hits; focus trapping **not verified**. |
| 3D viewport | Canvas created imperatively by three-scene; no `aria-label`/`role`/`tabIndex` found on canvas mount; keyboard shortcuts are window-level (World3DViewport.tsx:645) and there is no keyboard path to select/inspect entities except via the hierarchy panel. Axis gizmo has `role="img"` (:940). |
| `outline-none` without a `focus` class on the same line | 14 occurrences, each a possible invisible focus ring (not individually checked). |
| `prefers-reduced-motion` | 2 references only; camera/transition motion otherwise unguarded. |
| Automated a11y tests | e2e has 9 specs (`world.spec.ts` 10 tests, others 1-3); `grep axe|a11y` finds nothing in e2e. |

## 6. Ranked fix list

| # | Fix | File(s) | Effort |
|---|---|---|---|
| 1 | Remove or hide dead nav: Reports rail item, Analyze/Export mode tabs, TopBar Export, Home "Create Report" when `SHOW_UNFINISHED` is off (or point Export at the real ExportPanel / world report). | `lib/shell/shell-logic.ts:28-48`, `components/home/HomeDashboard.tsx:34`, `components/shell/TopBar.tsx:122` | S |
| 2 | Fix the transform correction contract: send exactly `{position, rotation{w,x,y,z}}` (or add a position-only change to the backend) and add a contract test against `_validate_transform`; surface 422 text. | `components/inspector/AdaptiveInspector.tsx:395-405`, `apps/api/routes_worlds.py:1067` | S-M |
| 3 | Make undo honest: either push history only after commit succeeds and have undo call a revert commit, or label it "preview only". Use the viewed version as `parent_version_id`. | `AdaptiveInspector.tsx:407-444`, `SpatialWorkstation.tsx:276-292,369`, `lib/viewport/history-stack.ts` | M |
| 4 | Wire the missing journey steps that already have backend support: version rollback, world package import/export. | new UI in SpatialWorkstation / Worlds; `POST /rollback`, `/import/package`, `/export/package` | M |
| 5 | Build the Reports page from real data (quality, measurements, report, exports) per blueprint 14, retire `lib/data.ts` and `unsupportedResource` stubs. | `app/(desktop)/reports/*`, `lib/api/analysis.ts`, `lib/data.ts` | L |
| 6 | Desktop SSR base URL: have server components read `REALITY_BACKEND_URL` (or set `NEXT_PUBLIC_API_BASE_URL` at launch) instead of defaulting to 8100. | `lib/api/client.ts:20-26`, `desktop/src/main.js:221` | S |
| 7 | Either call `reclassify/split/merge` from the inspector or delete the dead wrappers; same for dead overlays (`capture_plan`, `temporal_diff`) until a backend writes them. | `lib/api/corrections.ts`, `World3DViewport.tsx:230-232` | S-M |
| 8 | Pre-flight check (`/uploads/check`) and video acceptance on the Studio drop zone, matching the backend's video frame picker. | `components/reconstruct/PhotoDropZone.tsx:230,244` | S-M |
| 9 | Split the three monoliths: `EntityDetails`, `HierarchyExplorer`, `WorldSceneController` (seams in section 4). | files in section 4 | L |
| 10 | Accessibility pass: viewport `role`/`aria-label` + keyboard entity picking, check the 14 `outline-none` sites, dialog focus trap, reduced-motion guard, add an axe check to e2e. | `World3DViewport.tsx`, `three-scene.ts`, e2e | M |

## 7. Summary (10 lines)

1. Live FastAPI schema equals `docs/openapi.json`: 118 paths, 138 operations, no drift.
2. Every frontend `/api` call maps to a real backend route and method; no dangling calls found.
3. About 28 backend routes (paths) have no UI: rollback, package import/export, resumable upload, job SSE, assistant, registration, reclassify/split/merge.
4. No fabricated data reaches the screen: `lib/data.ts` is empty, analysis/results/reports return an explicit "not served" reason, no fake progress.
5. Analysis/Results/Reports/Team 404 by default, but rail, mode tabs, TopBar Export and Home "Create Report" still link to them.
6. Reports has no backend binding even though the backend serves report, quality, measurements and export.
7. Corrections persist through `POST /commit`; undo/redo is client-only and never reverts the server version.
8. The transform correction payload very likely fails backend validation (422); not runtime-verified.
9. Capture-plan and temporal-diff overlays read metadata nothing writes; they cannot display real data today.
10. Five files exceed 900 lines (WorldNavPanel 3063, three-scene 2835, AdaptiveInspector 2073); a11y basics are decent but the 3D viewport has no keyboard or screen-reader path.

## 8. Could not verify

- Any runtime behaviour: no browser, no dev server, no Playwright, no test suite run (per task rules).
- Whether real worlds' `entity.transform` serialises in a shape the commit validator accepts (static mismatch only).
- Desktop-app SSR base-URL failure (inferred from main.js and client.ts, not run).
- Whether `/api/worlds/{id}/cameras` returns cameras for COLMAP worlds (the previous audit's claim; I did not test it).
- Dialog focus trapping, Escape handling on scrims, and the 14 `outline-none` sites.
- Whether the mission's "16-step" journey has an official list; I used the 15 named steps plus undo.
- Contract script is heuristic: dynamic URL building I did not recognise (string concatenation, e.g. `lib/api/quality.ts`, `"/api/worlds/" + id + "/quality"`) was checked by hand only for the cases I found.
