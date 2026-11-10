# 36 — UI spec

> **Status:** target specification. Reduces the legacy surface from ~40 routes to
> 12, and defines the Studio as four fixed zones plus one drawer. Session and
> World are the only primary concepts; the UI shows backend state only — no
> fake UI state, no measured % unless measured, no buttons that do nothing.
>
> This spec is the contract for the Phase U track in `30_BUILD_PLAN.md` and is
> read by `35_TASK_CARDS.md` (U.1–U.8).

---

## 1. Route reduction: 40 → 12

The legacy frontend grew organically to roughly 40 routes (capture variants,
multiple world viewers, settings fragments, report variants, session variants).
The spec collapses them to 12 named routes, each backed by a real API endpoint.

| # | Route | Purpose | Backed by |
|---|-------|---------|-----------|
| 1 | `/` | Home dashboard: recent worlds, jobs, failures, storage | `GET /api/worlds`, `GET /api/jobs`, `GET /api/storage` |
| 2 | `/capture` | New session / guided capture | `POST /api/sessions`, CaptureRequests |
| 3 | `/sessions` | Session list + status | `GET /api/sessions` |
| 4 | `/sessions/[id]` | Session detail + processing states | `GET /api/sessions/[id]` |
| 5 | `/worlds` | World list + Trash/Archived views | `GET /api/worlds?status=` |
| 6 | `/worlds/[id]` | World workspace (viewport-first) | `GET /api/worlds/[id]` |
| 7 | `/worlds/[id]/measure` | Measurement tool | `GET /api/worlds/[id]/measurements` |
| 8 | `/worlds/[id]/export` | World package export/import | `GET /api/worlds/[id]/export` |
| 9 | `/worlds/[id]/diff` | Version diff view | `GET /api/worlds/[id]/diff` |
| 10 | `/worlds/[id]/settings` | World-level settings honesty | `GET /api/worlds/[id]/settings` |
| 11 | `/settings` | App settings (Appearance real; Processing/Data real or task-opened; Account/Workspace/Integrations = "Not available in this version") | `GET /api/settings` |
| 12 | `/reports` | Evidence/provenance/timeline reports | `GET /api/reports` |

All other legacy routes redirect to one of these 12 or return 410. No route is
kept "because it might be useful later."

---

## 2. Studio shell: four fixed zones + one drawer

The Studio (`/worlds/[id]`) is a single fixed shell. Every world route renders
the same shell; content changes by selection, not by swapping shells.

```
┌─────────────────────────────────────────────────────────────┐
│  TOP BAR                                                       │
│  [logo]  world name  [jobs]  [user]  [help]                  │
├──────────┬──────────────────────────────┬───────────────────┤
│          │                              │                   │
│ LEFT     │ CENTRE                       │ RIGHT               │
│          │                              │                   │
│ hierarchy│   VIEWPORT (viewport-first)  │ contextual        │
│ (tree)   │                              │ inspector         │
│          │  [Real / Wireframe /         │ (varies by          │
│  rooms   │   Evidence / Confidence]     │ selection type)     │
│  walls   │                              │                   │
│  doors   │                              │                   │
│  ...     │                              │                   │
│          │                              │                   │
├──────────┴──────────────────────────────┴───────────────────┤
│  STATUS FOOTER                                                │
│  [stage label]  [failure reason]  [progress]  [storage]      │
└─────────────────────────────────────────────────────────────┘
                      ▲
                      │ one drawer (slides up from footer)
              ┌───────┴───────┐
              │  capture      │
              │  request      │
              │  drawer       │
              │ (target region, │
              │  suggested      │
              │  poses,         │
              │  instruction    │
              │  text, expected │
              │  gain)          │
              └───────────────┘
```

### Zone A — Top bar (global)
- Logo, current world name, jobs indicator, user, help.
- The jobs indicator shows real `stage` + `stage_counts` from the API; no measured % unless measured.

### Zone B — Left hierarchy tree (fixed)
- Rooms → walls → doors → windows → surfaces → openings → entities.
- Tree nodes carry evidence state (OBSERVED / RECONSTRUCTED / ESTIMATED / INFERRED / GENERATED / UNKNOWN / CONFLICT).
- Selecting a tree node sets the selection (see Zone D) and the inspector.

### Zone C — Centre viewport (viewport-first)
- The dominant area. Renders the current world: point cloud, mesh, wireframe, or real view.
- Three toggles: **Real** (image-derived textures), **Wireframe** (geometry), **Evidence** (evidence-state colours) + **Confidence** overlay.
- Viewport controls: orbit / aerial / third-person / first-person as one continuous camera (no manual mode switches).
- Selection highlight: the selected object is highlighted in the viewport and the tree simultaneously.

### Zone D — Right contextual inspector (fixed)
- Varies by selection type:
  - **Room:** dimensions, floor/ceiling heights, wall count, openings.
  - **Wall:** length, height, thickness (labelled default unless measured), openings.
  - **Door/Window:** width, height, host wall, sill/head.
  - **Surface:** material + confidence, coverage fraction, evidence state.
  - **Entity:** source photos with masks, association decisions, provenance chain.
- No inspector content when nothing is selected (empty state, not a generic form).

### Zone E — Status footer (fixed)
- Stage label (real backend stage, e.g. "Reconstructing the 3D structure").
- Failure reason when a job failed (never hidden).
- Progress only when measured by the backend.
- Storage usage.

### Drawer — Capture request (slides up from footer)
- Shown when a CaptureRequest exists (target region, suggested poses, instruction text, expected gain).
- The user adds a photo → sees what changed (diff view).
- Drawer dismisses when the request is satisfied or explicitly closed.

---

## 3. Selection contract

One selection at a time (single mode default; multi optional). Selection state flows:
`viewport ↔ tree ↔ search ↔ natural-language query ↔ inspector`.

- Selecting an object in **any** interface highlights the same object in **all** interfaces (tree node highlighted, viewport object highlighted, inspector shows that object).
- Selection persists across the three viewport modes (Real/Wireframe/Evidence).
- Clearing selection in one interface clears it everywhere.
- Selection of a non-existent object = "not found," never silently ignored.

---

## 4. Honesty rules (UI contract)

| Rule | Requirement |
|------|-------------|
| Backend state only | The UI reflects `GET /api/*` state; no client-invented progress, stage, or entity. |
| No measured % unless measured | Progress is shown only when the backend reports a measured value; otherwise "working…" / stage label. |
| No fake states | No fake loading, no buttons that do nothing, no placeholder data presented as complete. |
| Empty states | Nothing selected → empty inspector; no capture request → drawer closed; no worlds → empty world list. |
| Error visibility | Failed jobs show the failure reason; refused photos show the reason; export errors show to the user (not console-only). |
| Evidence colour legend | The Evidence view includes a legend mapping colours to provenance states. |
| WCAG 2.2 AA | Keyboard navigation, visible focus states, colour contrast, touch targets — verified by `axe` + design-tokens tests. |
| Responsive | 1280 / 1920 / 375 breakpoints; mobile capture path is a real camera flow, not a fallback placeholder. |

---

## 5. Acceptance (Phase U gate)

- U.1 frontend audit committed: routes→APIs, installed libs, duplicate components, existing tokens; keep/replace per responsibility.
- U.2 library compatibility + licence matrix committed; nothing installed without a row.
- U.3 shared design tokens (light + dark); contrast checks pass (`tests/contract/design-tokens.test.mts`, WCAG AA).
- U.4–U.8 Playwright E2E + screenshots at 1280 / 1920 / 375 for home, session flow, world workspace, adoption review, mobile.
