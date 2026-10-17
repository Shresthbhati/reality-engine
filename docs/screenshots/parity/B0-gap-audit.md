# B0 — Reference vs live Studio gap audit (2026-10-18)

Owner decision: match the references **as exactly as possible; new design tokens allowed** (P6b).
Method: live app run with `launch_ui.bat` (API 8100, UI 3000) at 1536×1024, an audit world seeded through the API
(1 room + 3×3 city). Reference folder: `C:\Users\shres\Downloads\reference` (~125 images).

**Coverage, stated honestly:** I viewed 7 of ~125 reference images (intent wizard, measurement mode, asset inspector,
plot/map view, dense 3D city with properties, pipeline graph, home) and 9 live routes (3D workspace by screenshot; others by
rendered page text because screenshots timed out). The remaining ~118 references must be reviewed in slice B2 before each
slice is built. Screenshots were observed in-session, not saved to the repo.

## Shared visual language in the references (design-token target)
- Deep navy-black canvas, cyan/blue accent, green=ok / amber=review / red=error / purple=volume, thin 1px panel borders.
- Persistent **RE logo + product tagline** top-left, a **top mode bar** (Ingest/Process/Explore/Compare/Collaborate/Export or
  Explore/Capture/Analyze/Share), global search with `Ctrl K`, site/project switcher, notifications, user chip.
- A narrow **left icon rail** (Home, Ingest, Process, Map, Datasets, Settings / Layers, Evidence, Measure, Compare).
- A wide **right inspector** with tabs; a **bottom strip** (evidence thumbnails, measurements table, histogram, status footer
  with asset count, CRS `EPSG:4326`, online).
- Heavy use of real imagery (aerial/street photos, point clouds, satellite) as the visual content.

## Screen-by-screen gaps
| # | Reference | Live today | Gap | Slice |
|---|---|---|---|---|
| 1 | **Home** hero, recent worlds with thumbnails, global overview map, activity, AI assistant, quick start | `/` redirects straight into the latest world's workspace; no dashboard reachable | Missing screen; needs thumbnails from real artifacts only (honest placeholder otherwise) | B2 |
| 2 | **World Intent wizard** (4 steps: scale → capabilities → review → run; resource profile) | `/worlds/new`: bare 3-field form (name, location, sessions) | Entire wizard missing; resource estimate must be computed from real dataset facts, not invented | B3 |
| 3 | Session create/ingest | `/sessions/new`: minimal form (name, source Camera/Upload/Import, location, world) | Needs drag-drop ingest with live quality checks (reference "Ingest") | B3/B4 |
| 4 | **Asset inspector** (original vs analysis overlay, quality bars, spatial value, ACCEPTED/REJECTED + reason, similar frames) | `/evidence` list shows "No Evidence has been added yet"; evidence view exists but not compared | Overlay modes, quality/contribution panels, decision + reason; data exists in `evidence/contribution.py`, `image_check.py` | B4 |
| 5 | **Measurement mode** (tools, labelled 3D annotations, ± uncertainty, evidence strip, histogram) | World workspace has a Measure toggle only | Tool palette, uncertainty, evidence links; must refuse metres when scale unknown | B5 |
| 6 | **Plot / map 3D view** (layers, legend, plot info, evidence carousel) | `sessions/[id]/map` + Geospatial Map tab exist; not verified against reference | Compare after B2 shell; GIS data limited to real sources (OSM) | B6 |
| 7 | **3D city with Properties/Insights/AI** tabs, layer toggles, mini-map, pipeline progress list | 3D workspace: viewport + hierarchy + inspector; **no point/imagery strip, no mini-map, no Insights/AI tab** | Add tabs, layer panel, mini-map; AI tab disabled until Phase F | B2/B6 |
| 8 | **Pipeline graph** (node per stage: status, progress, confidence, CPU/GPU) | Bottom "Processing State" tab; list-style | Node graph from real job stages; failed ≠ succeeded | B7 |
| 9 | Analysis / Results / Reports | `/analysis` empty state ok; `/results` says "not served by the API yet — no mock data is shown" | Results screen is an honest stub; real results need an API (P6) | B7 |
| 10 | Mobile capture | `/m`: greeting, Open Camera, empty lists | Layout parity + guided-capture overlay unchecked | B8 |

## Defects found during the audit (not design gaps)
1. **Launcher reuses a stale server.** `launch_ui.bat` skips starting the API if port 8100 already answers, so after a code change
   the old routes (404 on `/cities`) keep serving. An orphaned multiprocessing worker also held the port after its parent was
   killed. → add "restart" guidance/option to the launcher (A2 follow-up).
2. **Axis mismatch:** Z-up grammar data vs Y-up viewport (streets show as vertical slabs). → B1.
3. The A4 lint refactors (state reset during render, `useSyncExternalStore` bookmarks) were verified by tsc/eslint/contract
   tests only; the workspace, worlds list, evidence, analysis, results and mobile routes loaded without a crash in this
   audit, but interactions (bookmarks, tree auto-expand) were not exercised.

## Build order for Phase B (unchanged from the plan, now evidence-backed)
B1 axis → B2 shell + tokens + Home → B3 intent wizard + ingest → B4 asset inspector → B5 measurement → B6 map → B7 pipeline →
B8 mobile. Each slice: review that slice's reference images first, failing test, implement, side-by-side screenshot, commit.
