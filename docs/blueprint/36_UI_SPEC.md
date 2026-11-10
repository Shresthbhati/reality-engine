# 36 — Clean UI Spec

Companion to `34_MASTER_PLAN.md` (decision D10). Implemented by cards U501–U520 in `35_TASK_CARDS.md`.
Scope: the existing Next.js app in `frontend/`. No second app, no renderer change, no new state library (Zustand is already installed).

**Today [V]:** 40 routes, 26 API proxies, 32 Studio/workspace panel components, a 976-line `SpatialWorkstation.tsx`, nine specialist tools each with its own full-screen modal, 25+ command-palette entries.
**Target:** 12 routes, one Studio layout with four fixed zones and one drawer, one selection, one status vocabulary, one way to show how much the model knows.

---

## 1. Six rules

1. **One journey.** Capture → Processing → Review → World → Deliver. Every screen is one step of it or a list of them.
2. **The viewport is the product.** In Studio nothing covers the model except one drawer, and only when asked for.
3. **One of each.** One selection, one inspector, one representation switch, one place for specialist tools, one status parser.
4. **Say what is known.** Every entity and dimension carries an evidence badge. Unknown and not-seen are drawn, never hidden.
5. **Errors are errors.** A failed request shows an error block with a retry. It never shows an empty list, a zero, or a default.
6. **Advanced is one click away, not on screen.** Diagnostics, builders and specialist analysis live behind `Analyze`, `More`, or Settings.

---

## 2. Routes: 40 → 12

**Primary (12).** Desktop: `/` · `/worlds` · `/worlds/[id]` · `/capture` · `/sessions/[id]` · `/deliveries` · `/settings` · `/sites/[id]`. Mobile: `/m` · `/m/camera` · `/m/sessions/[id]` · `/m/worlds/[id]`.

**Navigation:** Home · Worlds · Capture · Deliveries — and Settings pinned at the bottom. Nothing else.

| Current route | Action | Destination / note |
|---|---|---|
| `/` | keep | Home |
| `/worlds` | keep | gains a `Sites` tab |
| `/worlds/[id]` | keep | Studio |
| `/capture` | keep | the only way to start a session |
| `/sessions/[id]` | keep | session detail: photos, quality, job |
| `/settings` | keep | tabs: Workspace · Team · Billing · Storage · Diagnostics |
| `/sites/[id]` | keep | reached from Worlds › Sites |
| `/reconstruct` | redirect | `/capture` |
| `/sessions/new` | redirect | `/capture` |
| `/sessions` | redirect | `/` (Home lists recent sessions) |
| `/sessions/[id]/map` | redirect | `/sessions/:id?view=map` |
| `/worlds/[id]/adoptions` | redirect | `/worlds/:id?mode=review` |
| `/worlds/[id]/proposals` | redirect | `/worlds/:id?mode=review` |
| `/worlds/[id]/pipeline` | redirect | `/worlds/:id?panel=timeline` |
| `/sites` | redirect | `/worlds?tab=sites` |
| `/team` | redirect | `/settings?tab=team` |
| `/activity` | redirect | `/` (activity lives in the top-bar notifications panel) |
| `/results`, `/results/[id]` | redirect | `/deliveries` |
| `/reports`, `/reports/new` | redirect | `/deliveries` |
| `/reports/[id]`, `/reports/[id]/edit` | redirect | `/deliveries?report=:id` |
| `/analysis`, `/analysis/new`, `/analysis/[id]` | redirect | `/deliveries` (analysis runs from Studio › Analyze) |
| `/evidence` | redirect | `/` |
| `/evidence/[id]`, `/evidence/[id]/view` | deep link | kept, not in nav; opened from a session photo |
| `/worlds/new` | deep link | kept, not in nav; opened from the `New world` menu on `/worlds` |
| `/m` | keep | mobile home: worlds + sessions in one list |
| `/m/camera` | keep | capture |
| `/m/sessions/[id]` | keep | upload queue and status |
| `/m/worlds/[id]` | keep | read-only viewer |
| `/m/sessions`, `/m/worlds`, `/m/evidence`, `/m/team`, `/m/more` | redirect | `/m` |
| `/m/evidence/[id]` | deep link | kept, not in nav |

Redirects are added first (`U502`); page files are deleted only after `U501` shows zero inbound links and the owner approves (`U520`). Query parameters `mode`, `panel`, `tab`, `view`, `report` are read once on load and written into the Studio store or the page's tab state.

---

## 3. Studio: four zones and one drawer

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ TOP   Home › Project › World name        v12 ▾   ● Processing   Share  Export  ⋯ │
├──────────────┬───────────────────────────────────────────────┬──────────────────┤
│ LEFT  Tree   │ VIEWPORT                                      │ RIGHT Inspector  │
│ 🔍 search    │                                               │ ┌──────────────┐ │
│ Storey ▾     │   ┌ floating toolbar (one row) ────────────┐  │ │Props│Evid│Meas│Iss│
│  Living room │   │ Orbit Walk Top Section │ Measure       │  │ └──────────────┘ │
│   Walls (4)  │   │ Points·Mesh·Splat·Plan │ Analyze       │  │  Wall W-03       │
│   Openings   │   └────────────────────────────────────────┘  │  Length 4.21 m   │
│   Fixtures   │                                               │  ▨ Reconstructed │
│   Furniture  │                                               │  ±  uncalibrated │
│ Layers ▸     │                          evidence legend ▸    │  3 photos ▸      │
├──────────────┴───────────────────────────────────────────────┴──────────────────┤
│ BOTTOM Timeline (collapsed by default)  ◦session ◦job ◦v11 ◦session ◦job ◦v12   │
└───────────────────────────────────────────────────────────────────────────────┘
                                         DRAWER (right, 420 px, over Inspector): Analyze tool
```

| Zone | Contains | Never contains |
|---|---|---|
| **Top bar** | breadcrumb, version switcher, live job chip, presence avatars, `Share`, `Export`, overflow `⋯` | tool launchers |
| **Left — Tree** | search field, storey selector, rooms → entities, layers | forms, charts |
| **Viewport** | the model; one floating toolbar; evidence legend (collapsible, bottom-right); transform gizmo only in Edit mode | panels, modals |
| **Right — Inspector** | four tabs for the selected thing: Properties · Evidence · Measurements · Issues | anything not about the selection |
| **Bottom — Timeline** | sessions, jobs (with real stage list), versions, changes; closed by default | progress percentages |
| **Drawer — Analyze** | exactly one specialist tool at a time, lazy-loaded, badged `Preliminary` | a second drawer |

**Sizes:** left 280 px, right 340 px, drawer 420 px, top 48 px, timeline 40 px closed / 220 px open. Below 1100 px the left zone becomes an overlay; below 768 px the app serves `/m/worlds/[id]`.

**Modes** (segmented control in the top bar, bound to the store):

| Mode | Shows | Hidden |
|---|---|---|
| **Inspect** (default) | everything read-only; Measure tool | gizmo, edit actions |
| **Edit** | gizmo, edit actions in Inspector, undo/redo in top bar | — |
| **Review** | appears only when a change proposal is pending; timeline opens; Inspector shows the proposed change for the selection with Accept / Reject | Edit actions |

**Hard limits:** one drawer; one modal at a time and only for destructive confirmation or Share/Export; no nested panels; no panel may open on its own.

---

## 4. Where the 32 existing components go

| Component (`frontend/src/components/…`) | New home |
|---|---|
| `workspace/SceneTree` | Left |
| `workspace/QueryPanel` | Left — the search field (filters become chips under it) |
| `studio/StoreySelector` | Left — header |
| `inspector/AdaptiveInspector` | Right — split into four tabs |
| `workspace/InsightsPanel`, `studio/SpatialIntelligencePanel` | Right › Issues (and Properties › Relationships) |
| `studio/TransformGizmoToolbar` | Viewport toolbar, Edit mode only |
| `studio/SplatControlsOverlay` | Popover on the `Splat` option of the representation switch |
| `studio/SpatialMarkupModal` | Viewport `Annotate` tool + Right › Issues |
| `studio/CollabPresenceBar` | Top bar — avatars only |
| `workspace/ExportPanel` | Top bar `Export` → sheet |
| `workspace/ActionsMenu` | Top bar `⋯` |
| `studio/TemporalDiffSlider` | Bottom timeline (compare two versions) |
| `studio/CapturePlanPanel` | Capture page and session detail ("What to capture next") |
| `studio/SectionDrawingModal` | Analyze › Drawings |
| `studio/TakeoffScheduleModal` | Analyze › Quantities |
| `studio/ClashDetectionPanel`, `LifeSafetyAccessibilityModal`, `ThermalEnvelopeModal`, `SurfaceQualityPanel` | Analyze › Checks |
| `studio/MepTopologyPanel` | Analyze › Services |
| `workspace/ConstructionProgressModal` | Analyze › Change |
| `studio/TechnologiesPanel` | Settings › Diagnostics |
| `workspace/GeneratedPanel`, `CompleteInteriorPanel` | Edit mode › a `Generated` section in the tree; every item badged Generated |
| `workspace/BuildModal`, `RoomConstructionModal`, `ApartmentBuilder`, `BuildingBuilder`, `DistrictBuilder`, `RoomListEditor`, `OpeningsEditor` | Out of Studio. `/worlds` › `New world` › `Build from dimensions` |
| `workspace/SpatialWorkstation` | Composition only (`U505`) |

Nothing is deleted by this table; components move and lose their own chrome.

---

## 5. Evidence language

One mapping, used by the tree, the inspector, the viewport overlay and exports. Pattern and label carry the meaning; colour only reinforces it.

| State | Label | Pattern | Tone token | Meaning shown on hover |
|---|---|---|---|---|
| `OBSERVED` | Observed | solid | `--ev-strong` | Seen or measured directly in your photos |
| `RECONSTRUCTED` | Reconstructed | solid | `--ev-muted` | Computed from several photos |
| `ESTIMATED` | Estimated | dashed | `--ev-strong` | Estimated by a model from the photos |
| `INFERRED` | Inferred | dashed | `--ev-muted` | Assumed from context — not seen |
| `GENERATED` | Generated | dotted | `--ev-accent` | Created by you or a tool — not captured |
| `UNKNOWN` | Unknown | hatched | `--ev-neutral` | Not enough evidence |
| `CONFLICT` | Conflict | solid + ⚠ | `--ev-warn` | Sources disagree — needs review |

Coverage (from D9) uses the same component with labels: Seen · Seen, absent · Hidden behind something · Out of view · Too blurry or far · Not captured.

Tokens map onto the existing design tokens (`frontend/tests/contract/design-tokens.test.mts`); add the five `--ev-*` tokens there if no equivalent exists, with light and dark values and a contrast ratio of at least 4.5:1 against the surface.

**Dimensions** always render as `value unit · badge · uncertainty`, for example `4.21 m · Reconstructed · uncalibrated`. Until card E206 lands, uncertainty reads `uncalibrated`, not a number.

---

## 6. Screens

**Home `/`** — three stacked blocks, nothing else: *Needs attention* (failed or partial jobs, pending reviews; hidden when empty), *Continue* (last three worlds as cards), *Recent sessions* (list with status). One primary button: `New capture`.

**Capture `/capture`** — a three-step strip that never becomes a wizard with hidden state: `1 Choose world (or new)` → `2 Add photos` → `3 Process`. Step 2 shows photo cards with a single quality reason each (`U514`) and the "What to capture next" list (`U515`). Step 3 shows the job chip and stage list, and a link `Open when ready`. Leaving the page never stops the job.

**Session `/sessions/[id]`** — photos grid, quality summary, job status with stages, and one call to action that depends on state: `Processing…` / `Review changes` / `Open world` / `Retry` (with the failure reason above it).

**Studio `/worlds/[id]`** — section 3.

**Review (Studio, mode `review`)** — the timeline opens with the proposal; the tree shows change chips next to entities; groups in order: New · Changed · Removed — needs review · Not seen this visit (kept) · Conflicts. `Accept all safe` applies only New and Changed marked supported. Removed and Conflicts require a per-item decision. "Not seen" has no controls.

**Deliveries `/deliveries`** — one table: name, type (Plan PDF, DXF, IFC, GLB, Schedule, Report, Share link), world and **version**, created, status, action. Every row names the version it came from. Share links show scope, expiry and a `Revoke` button.

**Settings `/settings`** — tabs: Workspace · Team · Billing · Storage · Diagnostics.

**Mobile `/m/*`** — home is one list; camera is full-screen with a shutter, a photo counter, the upload-queue indicator and at most one hint line; session detail shows queue and status; world is a read-only viewer with Orbit and Walk.

---

## 7. States and words

| Situation | Component | Wording pattern |
|---|---|---|
| Loading | `StateBlock kind="loading"` | `Loading worlds…` |
| Nothing yet | `kind="empty"` | `No worlds yet.` + `New capture` |
| Request failed | `kind="error"` (`role="alert"`) | `Couldn't load worlds.` + reason + `Retry` |
| Partial result | `kind="partial"` | `Partly reconstructed — ceiling and one wall are missing.` + `See what to capture` |
| Job running | job chip | stage name from the backend, e.g. `Finding camera positions` — no percentage |
| Job status unknown | job chip, neutral style | `Status unknown — refresh or check the job log` |
| Specialist output | `Preliminary` badge | `Automated analysis from the current model. Not a professional certification.` |

Copy rules: sentence case; verbs on buttons (`Export plan`, not `Plan export`); one name per thing (World, Session, Version, Change — never Place, Scan, Snapshot, Proposal in the UI); no internal terms on default screens (COLMAP, SfM, WorldIR, pipeline, artifact appear only under Diagnostics).

---

## 8. What stays out

No dashboards of charts on Home. No per-tool toolbars. No second inspector. No progress percentages. No toasts for errors that need action (use the error block). No theme or density switches. No onboarding carousel — the Capture page is the onboarding.

---

## 9. Done means

- [ ] `U501` inventory shows 12 primary routes; every other old route answers with a redirect or is a listed deep link.
- [ ] In Studio, with nothing selected and no tool open, exactly four zones are visible and the viewport takes at least 60 % of the window at 1440×900.
- [ ] Selecting in the tree, the viewport or search highlights the same entity in all three and fills the inspector.
- [ ] Every specialist tool opens in the drawer with the `Preliminary` badge; two cannot be open together.
- [ ] A forced API failure on Home, Worlds and Session shows the error block, never an empty state (Playwright).
- [ ] axe reports no serious or critical violations on Home, Capture and Studio.
- [ ] A first-time user gets from Home to a downloaded plan PDF without opening Analyze or Settings.
