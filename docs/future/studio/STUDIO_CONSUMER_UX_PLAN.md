# Reality Studio: consumer UI/UX and backend completion plan

Status: IN PROGRESS (2026-10-13). Phases 0, 1 and 2 are implemented and verified (see section 11); Phases 3-5 are not started.
This file supersedes the capability list in `REALITY_STUDIO.md` for UI work.
Source of truth for the product: `Reality_Engine_Complete_Master_Document.docx`
(Oct 2026). Section references below (for example §19.1) point to that document.
Registry rules from `CLAUDE.md` §51 still apply: advance a status only with
evidence, and update `docs/implementation/PENDING_IMPLEMENTATION.md` in the
same change.

This plan is written so another agent can execute it task by task. Each task
has an ID, the files it touches, and acceptance criteria. Do the phases in
order (§10). Backend tasks (B-xx) are scheduled inside the UI phase that needs
them.

---

## 1. What was observed (2026-10-13, live app)

The app was run on `localhost:3011` (frontend) and `localhost:8100` (API). A
world was built from 10 real South Building photos
(`datasets/south_building/images`).

| # | Observation | Evidence |
|---|---|---|
| O1 | The world workspace reads as an engineer console, not a product. Labels include `WORLDIR`, `ENU Frame`, `WorldStore version v-209dc49f2854`, `HEAD`, `Registration Status partial`, `UNENCLOSED WALLS struct-plane-000` and `ColmapReconstructionBackend`. | `/worlds/{id}` screenshot |
| O2 | The top bar has about 9 controls of equal weight (3D Workspace, Geospatial Map, Commands, Add evidence, Construction, Query, Export, plus layout toggles). Nothing tells a first-time user what to do next. | same |
| O3 | The 3D view showed a sparse point column inside an orange debug bounding box. The user's place, and the photos that built it, are not the hero. | same |
| O4 | The most useful message is buried in a thin, truncated status strip: "7 image(s) could not be placed in the model (P1180141.JPG, ...). Retake t...". | same |
| O5 | `/` shows only a dropzone with no app chrome. `isSpatialStudio` in `components/shell/AppShell.tsx` hides TopBar and Sidebar on `/`. A returning user cannot see their places from Home. | `AppShell.tsx` |
| O6 | The world card on `/worlds` has no thumbnail, quality or status. It is a grey globe icon. | `/worlds` screenshot |
| O7 | **Retracted.** `/sessions` and `/evidence` sat on "Loading..." while a job ran, which looked like an API stall. A probe on a real 10-photo job (141 samples, one every 0.5 s) measured a worst read of 0.059 s and none over 1 s, so the API was responsive; the delay was the Next dev server compiling those pages for the first time. | probe run 2026-10-13 |
| O8 | The job finished with `status=partial` but `stage=committing_version`. A terminal job reports a non-terminal stage, and `progress` was `None` throughout. | `GET /api/jobs/{id}` |
| O9 | Analysis, Results, Reports and Team are UI-only. `lib/api/unsupported.ts` says there is no backend, and Team is an empty placeholder. They still exist as routes as if they were features. | `lib/api/unsupported.ts`, `app/(desktop)/team/page.tsx` |
| O10 | Backend features with **zero** UI callers: `POST /api/worlds/{id}/rollback`, `POST /api/worlds/{id}/scale`, `GET /api/worlds/{id}/query/spaces`, `GET /api/system/verification`, `/api/sessions/mobile-tasks`. These are among the app's best qualities and are invisible. | grep of `frontend/src` |
| O11 | Endpoints that do not exist: rename world, delete world, world thumbnail, upload pre-flight check, entity search by type/provenance/confidence, human review of inferences, saved measurements/annotations, job logs, job progress events (no SSE/WebSocket anywhere in `apps/api`), auth. | route list of `apps/api/*.py` |
| O12 | Registry drift: `PENDING_IMPLEMENTATION.md` says WorldStore (P4.20) and CI (P7.29) are MISSING, while `IMPLEMENTATION_STATUS.md` says both are IMPLEMENTED. Master doc §27 lists documentation drift as a known risk. | both files |

## 2. The product's best qualities (what the UI must lead with)

These are real, tested backend capabilities. The redesign exists to put them in
front of the user.

1. **Photos in, place out.** Drop photos and get a 3D place. One photo gives a
   rough model, and more photos make it better (`POST /api/reconstructions`).
2. **The same place keeps improving.** Adding photos creates a new immutable
   version of the same world. Failed or worse rebuilds never replace the
   current one (CE-05, CE-08).
3. **It tells you what it is unsure about.** Per-photo placement, failure
   guidance, conflicts, and explicit "relative scale" (master doc §14, §19.2:
   "never make uncertainty invisible").
4. **Every claim traces back to photos.** `GET /api/worlds/{id}/entities/{eid}/provenance`.
5. **History you can undo.** Versions, a diff of what changed in ten
   categories (PT-08), and rollback (CE-10).
6. **Real measurements once you set scale.** `POST /api/worlds/{id}/scale`.
7. **Honest exports to 6 formats.** glTF, USD, IFC, CityGML, CityJSON and
   Blender, with an export report of what was exported, skipped and why
   (PT-01, §18.4).

## 3. Design principles for the consumer UI

1. **One obvious next action per screen.** Use a single primary button and
   demote everything else.
2. **Plain language first, technical detail on demand.** Every engine term maps
   to a consumer phrase (§4). Raw values (IDs, frames, hashes, backend names)
   live only in **Pro mode**.
3. **Uncertainty is visual, not only textual.** Show weak areas, unplaced
   photos and conflicts in the viewport and on photo thumbnails.
4. **Progress is the product.** A build takes minutes. Show stages in plain
   words, show photos being placed live, and never show a frozen spinner.
5. **Nothing fake.** Hide a feature with no backend; don't stub it. This keeps
   the existing rule in `unsupported.ts`.
6. **Keep Pro mode.** The current panels (`SpatialWorkstation`,
   `WorldNavPanel`, `AdaptiveInspector`, `BottomContextBar`,
   `ReconstructionStatusBar`) are valuable to engineers. **Do not delete
   them.** Put them behind a Pro mode toggle.

## 4. Language map (implement as one module)

Create `frontend/src/lib/copy.ts` and use it for all Simple-mode strings.

| Engine term | Consumer label | Notes |
|---|---|---|
| World | Place | UI label only. Keep API and route names (`/worlds`, `world_id`). |
| Evidence / photograph | Photo | Video becomes "Video", plan or render becomes "Plan / drawing". |
| Registered camera | Photo placed | "7 of 10 photos placed" |
| Unregistered image | Couldn't place this photo | Always pair with the reason and a tip from the API `guidance`. |
| Version / HEAD | Version 3 / Current | Never show `v-209dc49f2854` outside Pro mode. |
| Rollback | Restore this version | |
| Scale `relative` | Not real size yet | CTA: "Set real size" |
| Scale metric | Real size | Shown in the user's unit (Settings) |
| OBSERVED | Seen in photos | |
| RECONSTRUCTED | Built from several photos | |
| ESTIMATED | Measured estimate | |
| INFERRED | Best guess | |
| GENERATED | Added, not from photos | |
| UNKNOWN | Not enough photos | |
| CONFLICT | Photos disagree | |
| Confidence score | Quality: Good / Fair / Weak / Unknown | Master doc §14.2: a score is not a probability. Show the band. The raw number is Pro only. |
| Job `partial` | Ready, but some photos weren't used | |
| Job `failed` | Couldn't build this version. Your previous version is safe. | CE-05 guarantees this. Say it. |
| Coordinate frame, ENU, WorldIR, WorldStore, COLMAP, session | (hidden) | Pro mode only |

## 5. Information architecture

```
Home  /                    Library of places + "New place" (dropzone if empty)
New   /reconstruct         Drop -> check photos -> upload -> build progress
Place /worlds/[id]         Simple viewer (default) | Pro mode (current workstation)
  ├─ Photos drawer         Grid: placed / not placed + why + tip
  ├─ Quality card          One verdict + top 3 actions
  ├─ History drawer        Versions, what changed, Restore
  ├─ Measure tool          Needs real size -> Set real size flow
  └─ Export sheet          Pick by intent, show export report, download
Settings /settings         Units, theme, Pro mode default, "What's verified"
Mobile /m                  Capture-first: camera + guidance + upload queue
```

**Remove from navigation and the command palette** (keep the files, gate them
behind `NEXT_PUBLIC_SHOW_UNFINISHED=1`): Analysis, Results, Reports, Team. Move
Sessions and Evidence into Pro mode, because consumers think in places and
photos, not sessions.

## 6. Screen specs

### 6.1 Home `/` (UX-02)
- Always render app chrome (fix O5). Remove `/` from `isSpatialStudio`.
  `SpatialWorkstation` must no longer be the root page.
- **Zero places:** the current `PhotoDropZone` is the hero. Add a three-step
  strip below it: "1 Walk around and take photos · 2 Drop them here · 3
  Explore, then add more to improve it".
- **Has places:** a card grid with a "New place" primary button. Each card
  shows:
  - a thumbnail (B-03)
  - the name
  - "N photos · M placed"
  - a quality badge
  - a "Not real size yet" chip when the scale is relative
  - relative updated time
  - a live "Building… 40%" state when a job runs (B-01)
  - an overflow menu with Rename, Delete (typed-name confirm) and Export (B-04)

  All of it comes from one call (B-11).
- Sort by last updated. Show search only when there are more than 8 places.
- **States:** loading uses skeleton cards. The error state shows retry plus
  "Is the engine running?" with a copyable start command.

### 6.2 New place `/reconstruct` (UX-03; needs B-01, B-02)
1. **Drop** photos, a folder or a video (existing `PhotoDropZone`).
2. **Check (new).** Call `POST /api/uploads/check` (B-02) before upload. Show
   a per-photo row with these flags:
   - blurry
   - too small
   - duplicate
   - has location
   - not a photo (plan, render or screenshot)

   Show an overall verdict: "Good set", "Add a few more angles" (fewer than 8
   photos) or "3 photos look blurry". The user can remove flagged photos or
   continue anyway. Never block.
3. **Name** the place (prefilled from the folder name or date), then click
   **Build**.
4. **Build progress** is a full page with:
   - plain stages: Reading photos → Matching photos → Placing photos in 3D →
     Building surfaces → Checking quality → Saving version
   - a percentage and elapsed time
   - a grid of photo thumbnails that tick to "placed" as the job reports
     registered cameras (B-01). A mini-viewer with appearing camera frustums
     is an optional enhancement.

   The user may leave: "We'll keep building. You'll find it on Home." The
   notifications API already exists.
5. On success, open the place viewer with a one-time "Here's your place"
   coach mark.
- **Failure:** "Couldn't build this place", the reason in plain words, the top
  fix from `guidance`, then "Try again" and "Show details" (Pro: logs, B-05).
- **Partial:** "Ready. 7 photos weren't used", plus "See why".

### 6.3 Place viewer `/worlds/[id]`, Simple mode (UX-04)
Desktop layout (at least 1024 px wide):
- **Top bar (56 px):** back to Home · place name (inline rename) · version pill
  ("Version 3 · Current", opens History) · spacer · **Add photos** (primary)
  · Export · ⋯ menu (Pro mode toggle, Settings).
- **Viewport:** full bleed. The camera auto-frames the model. Hide the orange
  bounding box and axis/frame widgets unless Pro. Point size auto-scales. If a
  mesh exists, show it by default with points as an option.
- **Floating bottom toolbar:** Orbit · Measure · Photos · Quality overlay ·
  Reset view. Each has a keyboard shortcut, documented in
  `REALITY_STUDIO_KEYBOARD_REFERENCE.md`.
- **Status chip (top-left of the viewport):** the single most important fact,
  in this priority order:
  1. job running
  2. photos not placed
  3. not real size
  4. weak areas
  5. "Looks good"

  Clicking it opens the Quality card.
- **Right drawer (on selection):**
  - what it is (plain type)
  - a quality band
  - "Seen in N photos" with a thumbnail strip (provenance endpoint); clicking
    a thumbnail flies to that photo's viewpoint (cameras endpoint) and
    overlays the photo at 50% opacity
  - size (only if real size is set)
  - actions: Measure from here, "This is wrong" (B-06)
- **Pro mode** renders the current `SpatialWorkstation` unchanged. Persist the
  choice in `localStorage` key `re-pro-mode` (wrapped in try/catch).

Under 1024 px, drawers become bottom sheets and the toolbar becomes a labelled
bottom bar.

### 6.4 Photos drawer (UX-05)
- Left drawer with a grid of all photos for this place. Each photo has a
  status corner: ✓ placed, ⚠ not placed, • new in this version.
- Filter: All / Placed / Not placed.
- Clicking a placed photo flies to its camera. Clicking an unplaced photo
  shows the reason plus a tip, for example: "Taken from too far away. Retake
  closer, overlapping about 60% with a neighbouring photo."
- Delete uses the existing `DELETE /api/evidence/{id}`. The confirm says the
  next version rebuilds without it.
- **Add photos** (here and in the top bar) runs the same check + upload flow
  with `world_id` set, producing a new version of this place, not a new place.

### 6.5 Quality card and Quality overlay (UX-06; needs B-07)
- **Card:** a verdict (Good, Fair or Weak) and three rows, each with an action:
  - Photos placed X/Y → "Show unplaced"
  - Real size: No → "Set real size"
  - Weak areas: N → "Show on model"

  It also lists known limitations from `GET /api/system/verification` that
  apply to this place (for example "indoor rooms not yet verified"). Data
  comes from the existing `GET /report`, `/coverage` and `/status` plus B-07.
- **Overlay:** colour points or mesh by band (good, fair, weak, unknown).
  Conflicts use a distinct pattern or outline, not colour alone.

### 6.6 History drawer (UX-07)
- A vertical timeline: "Version 3 · 2 photos added · 14 min ago", with the ten
  change categories (PT-08, `lib/api/reconstruction-decision.ts`) as plain
  chips such as "2 surfaces refined" or "1 new area". Show "How this version
  was built" from the existing `how-built` data.
- **Compare two versions** uses the existing `VersionDiffModal`, restyled with
  §4 copy.
- **Restore this version** calls `POST /rollback` (currently unused). The
  confirm must say: "Nothing is deleted. Your photos stay. The next build
  starts from this version."

### 6.7 Measure and Set real size (UX-08; needs B-08)
- If scale is relative, Measure first opens **Set real size**: click two
  points with a known distance, then enter the distance and unit. This calls
  `POST /scale` (currently unused). Report the result plainly, including the
  stated uncertainty of the reference.
- Measure is click-to-click distance in the user's unit. Measurements can be
  named and saved (B-08) and are listed in the right drawer. They are stored as
  user annotations, never as observed geometry (§15.4).

### 6.8 Export sheet (UX-09; needs B-09) and printable report (UX-10)
- Pick by intent, with the format underneath:
  - 3D apps / Blender → glTF, Blender script
  - Film & USD tools → USD
  - Building / BIM → IFC
  - City & GIS → CityGML, CityJSON
- Before download, run a dry run (B-09) and show the report: "41 objects
  exported, 3 skipped (no surface data)", plus units and coordinate notes
  (§18.4). Then Download.
- **Print place report (UX-10):** a print stylesheet over the existing
  `GET /report` data (photos, quality, versions, limitations). It replaces the
  unbacked Reports section. No new backend is needed.

### 6.9 Settings (UX-11)
- Units (metric / imperial), theme (system / light / dark), reduce motion
  (already exists), default to Pro mode.
- **What's verified:** render `GET /api/system/verification` as a plain list
  with levels (for example "Photos → 3D place: verified on real photos").
  This is the honest-claims surface the master doc asks for (§26, Appendix D).

### 6.10 Mobile `/m` (UX-12)
- Capture-first home: "Capture a place" (existing `/m/camera`) plus recent
  places.
- Camera overlay guidance: "Move sideways, keep about 60% overlap", with a
  counter against a target ("12 / 20 recommended").
- An upload queue that survives backgrounding, using the existing
  `/api/sessions/mobile-tasks` (currently unused). Show the B-02 pre-flight
  result per photo.
- Viewing a place uses Simple mode with bottom sheets.

## 7. Visual system (UX-01)

- Keep the token approach in `app/globals.css`. Add a **light theme**: tokens
  on `:root`, dark overrides under `@media (prefers-color-scheme: dark)` and
  `[data-theme="dark"]`, and an explicit `[data-theme="light"]`. Replace
  hard-coded colours such as `bg-[#08090b]` in `AppShell.tsx` with tokens.
- Typography: one UI sans-serif. Monospace only in Pro mode and for IDs.
- The accent colour is used only for the primary action and the current
  selection (fixes O2).
- Status colour is always paired with an icon or text (colour-blind safe).
- Motion: 150-200 ms ease-out for drawers and sheets. Honour
  `data-reduce-motion`.
- Viewer: neutral gradient background, grid only while Measure is active, and
  a soft ground shadow.
- Visible focus rings everywhere. Drawers are keyboard-reachable and trap
  focus. Esc closes.

## 8. Backend tasks

These are ordered by the UI phase that needs them. Every task needs tests in
`tests/` (fake backend, offline) and a status update in
`PENDING_IMPLEMENTATION.md`.

| ID | Task | Master doc | Files | Acceptance |
|---|---|---|---|---|
| B-00 | **DONE: no fix needed (O7 retracted); regression test added.** Reconstruction already runs in worker threads. `tests/test_job_progress.py::test_reads_stay_fast_while_a_job_blocks` pins reads under 0.5 s while a backend blocks. | §7.3 | `apps/api/jobs.py`, `main.py` | A test starts a job whose fake backend blocks for 3 s. `GET /api/worlds` answers in under 300 ms meanwhile. |
| B-01 | **Truthful job progress.** Fill numeric `progress` (0-100, stage-weighted, monotonic). Use terminal `stage` values (`done`, `done_with_warnings`, `failed`) to fix O8. Expose `cameras_registered` / `photos_total` when known. Add `GET /api/jobs/{id}/events` (Server-Sent Events) and keep polling as a fallback. | §7.3, §20.2 | `jobs.py`, `routes_jobs.py`, `models.py` | Progress is monotonic and ends at 100. A terminal status never has a non-terminal stage. SSE emits on every stage change. Covered by fake-backend tests. |
| B-02 | **Upload pre-flight check.** `POST /api/uploads/check` (multipart) returns per file: blur score, resolution, duplicate-of (hash), has GPS, likely evidence class. It reuses `evidence/image_check.py` and the existing classifier. Nothing is persisted. | §8.4 | `routes_misc.py`, `evidence/image_check.py` | Blurry and duplicate fixtures are flagged. No files are written. Record the measured time for 20 photos at 1080p on CPU. |
| B-03 | **Place thumbnail.** `GET /api/worlds/{id}/thumbnail` returns a JPEG. The simple version picks the placed photo with the most observations and downscales it to 480 px, so no renderer is needed. Cache it per version id. | §19.1 | `routes_worlds.py`, `storage.py` | 200 image for a built world. 404 with a reason for an empty one. A new version refreshes it. |
| B-04 | **Rename and delete a place (DONE, with one change).** `PATCH /api/worlds/{id}` `{name}`. `DELETE /api/worlds/{id}` removes the application rows (world, sessions, photos, jobs, locations, versions), **quarantines** the WorldStore version records (the store never destroys observed data), and removes the derived COLMAP state and scale references. It refuses with 409 while a job runs. Shared content-addressed photo files and pipeline artifacts are **kept** (project rule: artifacts are immutable); disk space is not reclaimed until a store garbage collector exists (not built). | §19.1, §25.2 | `routes_worlds.py`, `worldstore_service.py` | Tests cover rename, delete, delete during job = 409, and no orphan files under the storage roots. |
| B-05 | **Job logs.** `GET /api/jobs/{id}/logs?tail=N` returns stage log lines including backend stderr. Paths outside the storage root and env secrets are redacted. Pro mode only in the UI. | §7.3, §25.2 | `jobs.py`, `routes_jobs.py` | A failed fake job returns its error lines. A secret-like env value never appears. |
| B-06 | **Human review of inferences.** `POST /api/worlds/{id}/entities/{eid}/review` `{action: accept\|reject\|correct\|defer, correction?, note?}` creates a **correction version** (new immutable version, provenance = user review). Observed data is never edited in place. Also add `GET /api/worlds/{id}/review-queue`, listing low-confidence, conflicting and unknown items (the "review queue" in `REALITY_STUDIO.md`). | §11.4, §15.4 | `routes_worlds.py`, `world_ir/`, `worldstore_service.py` | A reject yields a new version with the entity marked rejected. The previous version diff is unchanged. Rollback works across it. The queue returns fixture conflicts. |
| B-07 | **Confidence for the viewer.** Verify whether `/points` and `/worldir` already carry per-entity confidence/provenance (the `/points` handler shows no confidence field). Add a `quality_band` (`good\|fair\|weak\|unknown`) per entity and an optional per-point band array. Bands come from a documented rule, not a calibrated probability (§14.2). | §14, §19.2 | `routes_worlds.py`, `docs/engineering/DESIGN_DECISIONS.md` | The rule is documented. A test pins bands for fixtures. |
| B-08 | **Saved measurements / annotations.** `GET/POST/DELETE /api/worlds/{id}/annotations` stores `{kind: measurement\|note, points, value_m?, label}` against a version, with provenance GENERATED (user). Annotations carry across versions (anchored via entity re-id, else flagged "may have moved"). | §15.4 | new `apps/api/routes_annotations.py`, `models.py` | CRUD tests pass. Annotations are excluded from exported observed geometry unless explicitly requested. |
| B-09 | **Export dry run and loss report.** Add `dry_run=true` to the export route, returning only the report. Extend the report with `fields_dropped`, `units`, `coordinate_frame_note` per format. | §18.4 | `routes_export.py`, `exporters/` | For each of the 6 core formats, the dry-run report equals the real-run report. Fields lost for IFC/CityGML are listed. |
| B-10 | **Entity search.** `GET /api/worlds/{id}/query/entities?type=&provenance=&band=&text=` searches by attribute (the existing query routes are spatial only). | §13.2 | `routes_query.py` | "INFERRED doors" and "unknown material" return correct sets on a fixture. |
| B-11 | **Place summary for cards.** Extend `GET /api/worlds` items with `photo_count`, `placed_count`, `quality_band`, `scale_state`, `active_job {id, progress}`, `updated_at`, `thumbnail_url`. This avoids N+1 calls on Home. | §19.1 | `routes_worlds.py` | One request renders Home. A contract test sits in `frontend/tests/contract`. |
| B-12 | **Registry drift fix (O12).** Inspect code and tests, then reconcile `PENDING_IMPLEMENTATION.md` (P4.20, P4.21, P7.29) with `IMPLEMENTATION_STATUS.md`. | §27 | docs only | Both files agree, and each status cites a test file. |

### Decisions for the owner (record as BLOCKED with options if unanswered)

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| D-1 | Local single user vs. hosted multi-user | (a) local single-user v1 (b) accounts, projects, access control (§25.2) | **(a)** for v1. Hide Team. Build no share links without auth. Revisit before any hosted deploy. |
| D-2 | Privacy redaction of faces/plates (§25.1) | (a) defer (b) blur on display/export (c) blur at ingest | **(a)** for local v1, **(b)** before hosting. Record as pending. |
| D-3 | Simulation / scenario workspace (§16-17, §19.1) | (a) keep out of consumer UI (b) rigid-body demo in Pro mode | **(a)**. ROADMAP rule 4 keeps P8 unwired until the mapping spine and persistence are done. |

## 9. Larger backend gaps from the master document (already in the registry)

These are not UI tasks. Follow the registry entries rather than duplicating
them here. The UI must **show them as limitations** (Quality card, What's
verified), never hide them.

| Master doc | Gap | Registry |
|---|---|---|
| §10 | Cross-source registration (drone + phone + LiDAR in one frame) | P1.10 MISSING |
| §8, §15 | Timestamp sync, real-hardware VIO | P1.6, P1.8 MISSING |
| §9.1 | Dense MVS (blocked on CUDA on this machine) | P1.11 PARTIAL |
| §11 | Doors/windows/stairs structure, material perception | P2.16, P2.17 MISSING |
| §13.3 | Cross-survey identity | P2.14 PARTIAL, PROD.2 heuristic |
| §14.3 | Uncertainty propagation, queryable provenance graph | P3.18, P3.19 PARTIAL |
| §14 | Topology/semantic/opening conflicts | CE-01 PARTIAL |
| §13.4 | Incremental integration after COLMAP; UNKNOWN→OBSERVED transitions | CE-02, CE-04 PARTIAL |
| §18.2, §23 | Large-world tiling, LOD, streaming | P7.27 MISSING |
| §18.3 | GIS georeferencing, robotics export acceptance | P7.25, P7.26, PT-02 |
| §24.2 | Real indoor room/corridor dataset | PROD.1 remaining |
| §16-17 | Validated simulation domains | P8.30, P8.31 deferred |

## 10. Execution phases (for the implementing agent)

Each phase ends with:
- `npm --prefix frontend run lint`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:contract`
- the relevant `python -m pytest -q tests/...`
- **a browser check** of every touched screen at 1440×900 and 390×844, in
  light and dark, with screenshots attached to the PR

Use the `.claude/launch.json` configs (`reality-api`,
`reality-studio-frontend`). Real data: `datasets/south_building/images`.

| Phase | Tasks | Done when |
|---|---|---|
| 0 Foundations | UX-01, `lib/copy.ts` (§4), Pro mode flag, nav cleanup (§5), B-12 | Nav shows Home / New place / Settings. Unfinished pages are gated. A test greps Simple-mode components for the banned engine terms in §4. |
| 1 Home + create | B-00, B-01, B-02, B-03, B-04, B-11, UX-02, UX-03 | Drop → check → build progress → place works with no dead ends. Home shows thumbnails and live progress. The API stays responsive during a build. |
| 2 Viewer | B-07, UX-04, UX-06 | The model is the hero, the status chip is correct, the quality overlay works, and Pro mode equals the old workstation. |
| 3 Improve loop | UX-05 | Unplaced photos show reason + tip. Add photos makes Version 2 of the same place. Delete photo works. |
| 4 Trust + output | B-05, B-06, B-08, B-09, B-10, UX-07, UX-08, UX-09, UX-10, UX-11 | Restore works. Set real size, then Measure. Export shows the report before download. Print report works. "What's verified" renders. |
| 5 Mobile | UX-12 | Capture → pre-flight → queued upload → view place, all on a 390 px viewport. |

### Rules for the implementing agent
- Do not delete Pro-mode components. Wrap them.
- Do not invent data or mock endpoints. If a backend task isn't done, the UI
  hides that control.
- Every screen has loading, empty, error and partial states.
- Keep API and route names (`world`, `evidence`). Only UI copy changes.
- Update this file's task tables and `PENDING_IMPLEMENTATION.md` P6 as tasks
  land, citing the test file as evidence.
- One commit per task, with the task ID in the message.

## 11. Progress log

### Phase 0 (2026-10-13): DONE
UX-01 light theme and tokens, `lib/copy.ts` with a banned-term contract test, Pro mode switch, navigation cleanup (unfinished
routes return 404 unless `NEXT_PUBLIC_SHOW_UNFINISHED=1`), B-12 registry reconciliation.

### Phase 1 (2026-10-13): DONE (backend verified; UI verified in the browser at 1440 px and 390 px)
| Task | Result | Evidence |
|---|---|---|
| B-00 | No fix needed; guard test added | `tests/test_job_progress.py` |
| B-01 | `progress` is stage-weighted and monotonic, 100 only on success/partial; a settled job reports `done`, `done_with_warnings`, `failed` or `cancelled` as its stage; `photos_total` / `photos_placed`; `GET /api/jobs/{id}/events` (SSE). The UI polls `GET /api/jobs/{id}`; it does not use SSE yet. | `tests/test_job_progress.py` (5 tests) |
| B-02 | `POST /api/uploads/check`: decodable, too small, duplicate, relatively soft (a ranking inside the set, 3+ photos), looks like a photo, GPS. Stores nothing. | `tests/test_place_api.py` |
| B-03 | `GET /api/worlds/{id}/thumbnail`: a 480 px JPEG of a placed photo (else the first), cached per world, version and photo. | `tests/test_place_api.py` |
| B-04 | see the B-04 row | `tests/test_place_api.py` |
| B-11 | `GET /api/worlds` carries `photo_count`, `placed_count`, `quality_band`, `scale_state`, `active_job`, `updated_at` (labelled UTC), `thumbnail_url`. `quality_band` is the coarse rule in `apps/api/place_summary.py` (not a probability). | `tests/test_place_api.py` |
| UX-02 | Home is the place library with thumbnails, quality, "not real size yet", live building state, rename and delete (typed-name confirm); empty Home is the new-place flow. | browser check |
| UX-03 | Drop, check (per-photo problems, remove, re-check), name, build with real stage steps, retry, cancel. | browser check on a real 8-photo build (ended `partial`, 0 placed) |

**Deviations from the plan text**
- The build page shows **three working steps** the engine really reports (Reading your photos, Building the 3D place, Saving this version) plus "Waiting to start", not the six sketched in 6.2. The UI never shows a step the engine does not report. Percent is not shown for the same reason.
- "Photos tick to placed live" and "a mini-viewer with appearing cameras" are **not done**: placed photos are only known after the run (`photos_placed` is null while building).
- Pro mode currently adds Sessions, Evidence and the old "All places" list to the sidebar; the full workstation swap belongs to Phase 2.

**Known limits**
- The check's "blurry" flag is relative to the other photos in the set and is not shown for sets under 3 photos.
- A first-time Next dev compile of a page can take seconds; that is not an API stall.
- The mobile layout of the new screens was checked only for horizontal overflow at 390 px (none), not visually.

### Design pass (2026-10-13, after Phase 1): DONE
Skills applied: taste-skill, ui-ux-pro-max, frontend-design, karpathy-guidelines, ui-visual-validator, wcag-audit-patterns,
react-ui-patterns, make-interfaces-feel-better, redesign-skill. Design read: a consumer product screen (not a landing page),
calm and trust-first, the user's own photos as the visual hero; dials variance 3, motion 3, density 4. taste-skill's
landing-page rules (hero, marquee, eyebrow limits) do not apply to product screens; its anti-slop and state rules do.

**Measured, then fixed (WCAG contrast computed for every token pair, both themes):**
| Finding | Before | After |
|---|---|---|
| Dark `--text-tertiary` on surfaces | 2.8 to 3.2:1 | 5.5 to 6.2:1 (`#8a8fa3`) |
| Dark error button text (white on red) | 3.8:1 | 4.99:1 (`--on-error`) |
| Light status text on tinted chips | 4.0 to 4.4:1 | 4.6 to 5.2:1 |
| Control borders (inputs, outlined buttons) | 1.2 to 1.4:1 | 3.1 to 3.4:1 (`--border-strong`) |
| Dark accent saturation | `#00e5ff` (100%) | `#4fd1e0` (desaturated, same hue) |

**Also added:** skip-to-content link; focus trap, Escape and focus restore in the delete dialog; 44 px touch targets on
coarse pointers; tactile press feedback with named transition properties; `text-wrap: balance` on headings and `pretty` on
body text; tabular numerals on counts and elapsed time; neutral inset outline on photo thumbnails; screen-reader
announcement of the current build step.

**Rejected after checking:** the ui-ux-pro-max recommendation of a "Hero + Features + CTA" landing pattern (wrong for a
product screen) and a Lora/Raleway wellness font pair (a serif, and wrong subject). The system sans stays. Its calm-cyan
primary matches the existing light accent.

**Verified:** computed colours in both themes in the browser, dialog focus behaviour, production build, 26 contract tests.
**Not verified:** a screen-reader pass (NVDA/VoiceOver), keyboard-only walk of the build page, high-contrast mode.

### Phase 2 (2026-10-13): DONE (verified in the browser at 1280 px and 390 px, dark and light, on the real South Building place)
| Task | Result | Evidence |
|---|---|---|
| B-07 | `GET /api/worlds/{id}/quality`: a band (`good`, `fair`, `weak`, `unknown`) for every entity from its stored confidence and provenance, plus the place band, why it has that band (`band_reason`), photos placed, scale state, entity counts and open conflicts. Rule in `apps/api/place_summary.py` (`entity_band`, `band_and_reason`). | `tests/test_place_api.py` (rule, reasons, endpoint) |
| UX-04 | `/worlds/[id]` is the place viewer in Simple mode: full-screen model, one status chip, a toolbar (reset view, quality colours), a selection drawer (bottom sheet on phones) showing the part's plain name, band, "how it was found" and photo thumbnails. Pro mode renders the old workstation unchanged. | browser check |
| UX-06 | Quality card (opens from the chip) and quality colours on the model with a legend. | browser check |

**Entity band rule:** `unknown` when provenance is UNKNOWN or confidence is missing (never shown as good or bad); `weak` when
provenance is CONFLICT (whatever the score) or confidence is below 0.5; `fair` from 0.5 up to 0.75; `good` from 0.75. A model
score is not a probability (master doc 14.2), so the UI shows the band, not the number.

**Deviations from the plan text**
- **No per-point quality.** The stored point cloud has no per-point confidence, so showing one would be invented. The overlay
  colours the detected parts (walls, floors and so on); points keep their own colour.
- **The viewer top bar has only Home, the place name and Pro mode.** The version pill, Add photos and Export arrive with
  Phases 3 and 4, because their backends and screens are not built; nothing is stubbed.
- **"Seen in N photos" is stated only when the engine recorded it.** Today the trace is session-level for most parts, so the
  drawer says "Built from this place's photos. The engine did not record which ones." and shows up to six of the place's photos;
  photos not placed in the model are dimmed and cannot be clicked.
- **"This is wrong" (B-06) and the one-time coach mark are not built.**
- The 3D canvas stays dark in both themes (a viewing well), with themed chrome around it.

**Known limits**
- On a weak build the model is a sparse point column inside one detected wall (South Building, 3 of 10 photos placed). That is the
  engine's honest result, but it is not a persuasive first screen; better models need better photos or the pending engine work
  in section 9.
- Selection highlight (cyan) overrides the quality colour on the selected part.
- Not verified: a screen-reader pass of the viewer, and keyboard-only operation of the 3D selection (parts can only be picked
  with a pointer today; the quality card and toolbar are keyboard-operable).
