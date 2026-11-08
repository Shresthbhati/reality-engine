# 14 — Frontend and user experience

## Technology stack & foundation (one per responsibility)

Audited and aligned with [32_FRONTEND_AUDIT.md](32_FRONTEND_AUDIT.md) §8.1. Everything lives in `frontend/`.

| Responsibility | Foundation | Decision / Status |
|---|---|---|
| Framework, routing, styling | Next 16 App Router, React 19, Tailwind 4, CSS-variable tokens | Installed, Keep |
| Primitives, recipes, toast | shadcn/ui on Base UI (`@base-ui/react`) | Adopt (U.3 / U.4) |
| Icons | Lucide (`lucide-react`) | Installed, Keep |
| 3D Viewport | Three.js (imperative) | Installed, Keep |
| Panel layout | Allotment | Adopt (U.6) |
| Client UI state | Zustand | Installed, prune/replace for server state |
| Server state | TanStack Query (`@tanstack/react-query`) | Adopt (U.4 / U.5) |
| Tables / virtual lists | TanStack Table / Virtual | Approved, deferred |
| Drag and drop | dnd-kit | Deferred |
| Upload | In-house resumable chunked upload (`POST /api/uploads/resumable/*`) | Keep |
| Charts | ECharts | Approved, deferred (Apache-2.0 notice required) |
| Motion | CSS transitions / keyframes | Keep; Motion deferred |
| Forms | React Hook Form + Zod | Approved, deferred |
| Command palette | In-house `CommandPalette.tsx` against real `lib/search.ts` | Keep |
| Maps | maplibre-gl | Installed, Keep |
| End-to-end / a11y testing | Playwright + `@axe-core/playwright` | Installed / dev dependency |

Rejected: Drei (unnecessary abstraction/version lock), Dockview (over-complex for paneling), Uppy (in-house handles sha256 chunks), Sonner, React Aria (Apache-2.0), Radix flavour (deprecated in favor of Base UI).

## Navigation (desktop)

| Item | Purpose | Existing route(s) |
|------|---------|-------------------|
| Home | Recent worlds, running/failed jobs, "Create world" | `(desktop)/page` |
| Worlds | All worlds; filters Active / Archived / Trash | `(desktop)/worlds` |
| Capture | Upload photos, manage sessions, open capture requests | `(desktop)/sessions`, `(desktop)/reconstruct` → merge into one Capture screen |
| 3D Workstation | Inspect/edit one world | `(desktop)/worlds/[id]` |
| Evidence | Photos, masks, observations per world | `(desktop)/evidence` |
| Reports | Quality, measurements, exports | `(desktop)/reports`, `results`, `analysis` → consolidate |
| Settings | Appearance · Processing · Data | `(desktop)/settings` |

`team` and `activity` stay but are secondary (activity can live on Home).
Mobile (`/m/*`) stays a capture + view companion; not a v1 priority.

## Core flows

1. **Create world** → name + what it is (room / apartment / house / outdoor) →
   drop photos → "Start" → lands on the world with live progress.
2. **Progress** → the stage labels from [12](12_PIPELINE.md#user-facing-stage-labels-the-ui-uses-these--p13),
   each with real counts ("9 of 12 photos placed"). Partial/failed stages show
   the reason and the next action.
3. **Review** → the 3D workstation opens on the Real view; a quality chip shows
   coverage, conflicts and open capture requests.
4. **Improve** → capture request card: thumbnail/marker, instruction, "Add photo".
5. **Manage** → world card menu: Open, Rename, Duplicate, Export, Archive, Delete.
   Delete opens a confirmation listing what will be removed.

## 3D workstation layout

```
┌──────────┬──────────────────────────────────────────┬─────────────┐
│ Structure│                                          │ Inspector   │
│ tree:    │              VIEWPORT                    │ (selected   │
│ Space    │  [Real | Wireframe | Evidence]  top-left │  entity):   │
│  Surfaces│  status chip top-right                   │ class, dims │
│  Openings│                                          │ confidence  │
│  Entities│                                          │ evidence    │
│          │       contextual toolbar (measure, ...)  │ source pics │
└──────────┴──────────────────────────────────────────┴─────────────┘
```

- Panels collapse; the viewport dominates.
- Selecting an entity in the tree or the viewport highlights it in both.
- The inspector's "Evidence" section shows the source photos with the mask
  overlaid (P7). "Why is this one object?" shows the association decisions.
- The measurement tool shows value, unit, uncertainty and scale state; relative
  worlds show "units" and a "Set scale" action (user enters one known length).
- Evidence view colours: observed / partial / occluded / inferred / generated /
  conflicted / unknown, with a legend. It is an overlay toggle, never baked into materials.
- Corrections (Phase 8+): reclassify, split entity, merge entities — each one
  writes a user-basis decision and a new version.

## Visual direction

Neutral charcoal and dark-grey surfaces, one restrained accent, high-contrast
text. No bright-blue chrome. The existing design tokens in `frontend/src` stay
the source of truth; extend tokens rather than adding ad-hoc colours.

## States every screen must handle

Loading (skeleton) · Empty (with the one next action) · Error (message + retry)
· Partial (what worked, what didn't) · API unreachable ("Can't reach the engine —
is it running?"). No screen may show placeholder or demo data (P14).

## Accessibility & responsiveness

Keyboard-reachable controls, visible focus, labelled icon buttons, AA contrast,
layouts usable at 1280px and 1920px; tablet/mobile get a simplified viewer.

## UX acceptance per screen

A screen is done when it uses real API data, handles the five states above,
passes its Playwright test, and was screenshot-checked in the browser at 1280
and 1920 widths.
