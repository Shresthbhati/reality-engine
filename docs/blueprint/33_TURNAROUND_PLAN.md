# 33 — Turnaround Plan: from engine foundations to a sellable SaaS prototype

Date: 2026-11-07 · Baseline: `main` @ `45ae43e3` · Inputs: the 2026-11-07 repo audit (16 implemented /
91 partial / 13 missing) and the Master Product Constitution.
Status: **PROPOSED — needs user approval before execution.** This file orders the work; it does not
replace [30_BUILD_PLAN.md](30_BUILD_PLAN.md) task IDs — it reuses them and adds `T-*` IDs where 30 has none.

---

## 1. The one-sentence prototype

> A small or mid-sized architecture / interior design firm walks a client's room or apartment with a phone,
> and from a hosted web app gets an **as-built survey**: a measured, editable 3D model with walls, floors,
> ceilings, doors, windows (including shut ones), sockets/switches and furniture as real objects; a
> dimensioned **floor plan (PDF/DXF)**; an **IFC** model that opens in Revit/ArchiCAD; a room schedule;
> and a **read-only client link** for presenting design changes.

**Target customer: DECIDED 2026-11-07 — small-to-mid architecture and interior design firms.**
The pain we remove: a site-measure visit with tape and laser, then hours redrawing existing conditions in
CAD before any design work starts. What they will judge us on, in order: **dimensional accuracy**, correct
openings/fixtures, CAD/BIM interoperability, then speed and presentation.

Consequences for the plan versus a generic 3D tool:
- Measured-room accuracy (M3) is on the **critical path**, not a side track. A floor plan that is 5 cm off
  is worthless to this customer.
- Game-engine items (LOD chains, collision proxies, Unity preset) are **deferred**.
- Closed windows, doors and electrical fixtures are core, because every interior has them.
- Light editing (swap furniture, change finishes) + a client link come before large-world work.
- City-scale streaming, portal rendering, NeRF/splats, cloud fleets and the AI chat assistant stay out of
  the prototype.

### Customer comparison and expansion order (user, 2026-11-07)

| Customer | Prototype suitability | Reason | When | Main extra work beyond this plan |
|----------|----------------------|--------|------|----------------------------------|
| Architecture firms | Excellent | Existing-building documentation and spatial reconstruction | **Now (M0–M7)** | — |
| Interior design firms | Excellent | Room reconstruction, furniture, fixtures and measurements | **Now (M0–M7)** | — |
| Construction companies | Good | Site progress and existing-condition documentation | Next, after M7 | Repeat captures of the same site + temporal change reports (item 36), progress vs plan, rough unfinished surfaces |
| Real estate companies | Moderate | Visualisation and property presentation | After construction | Polished walk-through, shareable tours, floor-plan marketing exports (mostly reuses M4b) |
| Game studios | Moderate | Require much higher asset and reconstruction fidelity | Later | LOD, collision, clean topology, Unity/Unreal presets (deferred §4) |
| Urban planning organisations | Later | Large-scale mapping and geospatial requirements | Last | Geo accuracy, CityGML, terrain, large-world streaming (deferred §4) |

Rule: no task for a later customer enters the queue until the current customer's M7 gate has passed,
unless it is also required by the current customer.

**PROPOSED tolerance (confirm with 2–3 firms in M0):** wall length ±2 cm, room area ±1 %, ceiling height
±2 cm, opening width ±2 cm on rooms up to 6 × 6 m. Every reported dimension shows its uncertainty.

---

## 2. Effort tiers — what kind of problem costs Sonnet how much

Every task below carries a tier. The tier decides Sonnet's effort setting, who writes the brief, how hard
the review is, and whether Sonnet can finish it at all.

| Tier | Problem type | Sonnet effort | Brief written by | Review | Typical size | Can Sonnet finish it alone? |
|------|--------------|---------------|------------------|--------|--------------|-----------------------------|
| **T0** | Mechanical: fix a failing test, delete dead routes, resolve doc conflicts, CI config, rename | `low` | Sonnet self-briefs from this file | Sonnet re-runs Verify; Opus spot-check | 1 session, < 150 lines | Yes |
| **T1** | Wiring existing modules behind an existing contract (call 8.2 from the API path, expose an endpoint, add a UI panel for an existing API) | `medium` | Opus (one paragraph + file allowlist) | Opus reads diff, re-runs Verify | 1–2 sessions, < 400 lines | Yes |
| **T2** | New module with a written spec/interface (LOD exporter profile, collision proxy generator, IndexedDB upload queue, `watchPosition` trajectory) | `medium`→`high` | Opus: interface + acceptance test written **first** | Opus diff review + independent test run | 2–4 sessions | Yes, if the test exists before code |
| **T3** | Cross-cutting / stateful systems (edit transactions + persistent undo, session→world adoption, durable job queue, asset linkage) | `high`, sliced | Opus writes an ADR + contract; Sonnet implements slice by slice (≤ 400 lines each) | Opus review per slice + integration test gate | 4–10 sessions | Only slice by slice. Never one prompt. |
| **T4** | Perception / ML quality (closed-window recall, furniture recall, depth accuracy) | `medium` for harness + integration | Opus | Opus checks benchmark JSON, not prose | open-ended | **No.** Sonnet builds the eval harness and integration; the score only moves with real labelled data from the user. Coding harder does not fix recall. |
| **T5** | 3D rendering & visual UX (viewport, first-person nav, LOD switching, Studio layout) | `high` | Opus + screenshots of target | Browser screenshots at 375/1280/1920 + user eyeball | 2–5 sessions | Mostly; final "looks right" needs the user |
| **T6** | Product, architecture, licence, pricing decisions | — | **not Sonnet** | Opus proposes, user decides | — | No |
| **H** | Human-only: capture data, tape-measure rooms, label photos, talk to studios | — | user | — | — | No |

### Sonnet guard rails (apply to every tier)

1. **One task per session, own worktree, explicit file allowlist.** Two sessions never edit the same file.
2. **Test first for T2+**: the brief contains the failing test; Sonnet makes it pass.
3. **Paste real output** of Verify. "Should work" = not done. `tests/test_no_fake_backend_in_production.py`
   must stay green (no mocks in product paths).
4. **Two-strike rule**: two failed attempts at the same fix → stop, write what was tried, escalate to Opus.
5. **No scope creep**: anything outside the allowlist goes into a `FOLLOW-UP:` line, not the diff.
6. **Status lives in one place**: `30_BUILD_PLAN.md` status column + `IMPLEMENTATION_STATUS.md`, updated in
   the same PR.
7. Sonnet never decides a T6 question; it writes `BLOCKED: <question>` and moves on.

---

## 3. Milestones

Each milestone ends with a **gate**. Nothing in the next milestone starts until the gate passes, except
tasks marked *parallel-safe*.

### Progress (local branch `claude/m0-baseline`, not pushed; updated 2026-11-07)

| ID | State | Evidence |
|----|-------|----------|
| T-0.1, T-0.2, T-0.3, T-0.4 | DONE locally | CI core list 336 passed; contract tests 385 passed; docs conflict markers removed repo-wide |
| 13.1 | PARTIAL | audit + CODEOWNERS committed; owner must delete branches and protect `main` |
| T-1.1 | DONE locally | one Studio UI (decision recorded), dead proxies/files removed, artifact proxy added; tsc 0, lint 0 errors, build ok, Playwright world/routes/home/shell 34 passed |
| T-1.2 | DONE locally | catch-all proxy + same-origin browser base; Playwright 51 passed (dev and production build); token/upload/traversal checked by hand; hosted per-user identity still open (M6) |
| T-1.3 | DONE locally | `tests/test_golden_path_api_to_glb.py` (in CI core list, 8 s; plumbing only, uses the test-backend seam) and `tests/integration/test_golden_path_real_photos.py` (real COLMAP, 6 South Building photos, ~50 s). Real runs on 2026-11-07: 4 of 5 job outcomes `succeeded`, 1 `partial` (cause unknown; the test requires `succeeded`, so expect an occasional flake). Khronos glTF-Validator 2.0.0-dev.3.10: 0 errors, 0 warnings on a synthetic and a real-photo GLB |
| H-0.1, H-3.1, H-3.2 | OPEN (you) | firm interviews, closed-window photos, tape-measured rooms |

**T-1.3 findings (feed into M2):**
- The real-photo GLB is 4–5 wall planes plus a ~1,090-point sparse cloud: 12 triangles, 0 materials. Valid, but not yet a design deliverable (no floor/ceiling/openings/furniture). This is the real baseline for M3 accuracy work.
- The exporter emits an orphan mesh (`UNUSED_OBJECT /meshes/0`, Khronos INFO) in both files; tidy in T-2.1.
- Entity count and point count vary between identical real runs (4 vs 5 entities, 1,091 vs 1,089 points), so exact-value assertions on real output are not safe.

### M0 — Trustworthy baseline (≈ 1 week)

Gate: `main` CI green on 3.10/3.11/3.12; one non-conflicting status document; frontend contract tests in CI.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-0.1 | Triage the 4 failures in `tests/test_mapping_spine_perception.py` (stale expectation vs regression — decide per test, fix root cause) | T0→T1 | medium | CI green; PR says which were stale and why |
| T-0.2 | Remove merge-conflict markers in `docs/implementation/IMPLEMENTATION_STATUS.md`; reconcile rows against code | T0 | low | no conflict markers; each row cites a file or test |
| T-0.3 | Mark `docs/REALITY_ENGINE_CURRENT_STATUS.md` superseded (pointer to 30 + IMPLEMENTATION_STATUS) | T0 | low | file header says SUPERSEDED |
| T-0.4 | Run the 54 Node contract files in CI | T0 | low | CI job lists them; all pass or are deleted with reason |
| 13.1 | Branch audit + CODEOWNERS + protect `main` | T0 + **H** (user enables protection) | low | stale branches listed for user to delete; CODEOWNERS committed |
| **H-0.1** | Interview 2–3 firms: confirm tolerances, which CAD/BIM tool they use (Revit / ArchiCAD / AutoCAD / SketchUp / Vectorworks), what a site-measure costs them today | H | — | notes committed to `docs/research/customers/` |

### M1 — Golden path works end to end, locally (≈ 2–3 weeks)

Gate: in a browser, upload 20–60 photos of one room → processing shows real stages → world opens in Studio
→ furniture and openings are entities → export GLB opens in Blender. Screenshots + benchmark JSON committed.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| 8.7 | Finish single-image furniture path (already partial) | T1 | medium | spec M1 acceptance numbers in benchmark JSON |
| 8.8 | Open-vocab detector + crop pass called from the API reconstruction path | T1 (+T6: O1/O2 licence choice first) | medium | uploaded room → fixtures on wall planes in WorldIR |
| T-1.1 | Collapse Place + Workstation into one Studio route; delete the 11 gated-404 routes and 4 dead proxies (32 §2–7) | T1 | medium | one world UI; route count drops; e2e passes |
| T-1.2 | One API transport (pick direct API or Next proxy, remove the other) | T1 (+T6 pick) | medium | one client module; no duplicate fetchers |
| U.3 | Design tokens light/dark | T2 | medium | every component on tokens; contrast check passes |
| U.4 / U.5 | Home (New Session · Open World · needs-attention) and one guided Session flow with backend `stage` states | T5 | high | Playwright screenshots 375/1280/1920; no fake % |
| T-1.3 | Golden-path e2e test (photos fixture → world → GLB file exists and validates with `gltf-validator`) | T2 | medium | runs in CI on CPU fixture |

### M2 — Design-ready deliverables: the thing firms pay for (≈ 3–4 weeks)

Gate: one captured apartment produces (a) a dimensioned floor plan PDF + DXF that opens in AutoCAD with
walls, door swings, windows and socket/switch symbols on correct layers, (b) an IFC that opens in
Revit/ArchiCAD/BlenderBIM with walls, slabs, doors, windows as the right element classes, (c) a room
schedule. Dimension checks against M3 tape measurements run in CI.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| 13.5 | Floor plan export SVG/PDF **+ DXF** (layers: walls, openings, fixtures, furniture, dimensions; door swings; metric) | T2 | high | DXF parsed back with `ezdxf` in a test; layer names + wall count asserted |
| T-2.1 | IFC export through the existing IFC bridge: `IfcWall`, `IfcSlab`, `IfcDoor`, `IfcWindow`, `IfcSpace`, `IfcFurnishingElement`, openings as `IfcOpeningElement` | T2 | high | `ifcopenshell` round-trip test asserts classes + counts; **H**: user opens in Revit or ArchiCAD once |
| T-2.2 | Measure tools in Studio: wall length, room area, ceiling height, opening size, point-to-point; each shows ± uncertainty and observed/inferred | T5 | high | Playwright test measures a fixture wall; value matches WorldIR |
| T-2.3 | Room schedule (area, perimeter, ceiling height, openings, fixtures, finishes) as PDF + CSV | T1 | medium | numbers equal WorldIR values in test |
| T-2.4 | Plain mesh export for SketchUp/3ds Max workflows (OBJ/DAE/GLB, metric, per-object nodes, materials named by surface) | T1 | low | glTF validator clean; Blender headless import asserts scale |
| T-2.6 | Select furniture → "Extract as asset" → independent copy (own ID + URL), downloadable | T3 (slice: data model → API → UI) | high | extracted asset keeps geometry, materials, dimensions, provenance; original untouched (test) |
| T-2.7 | Export report in UI: what was measured vs inferred, per element, with confidence | T1 | medium | report renders per export |

Deferred for this customer: LOD chains, collision proxies, Unity preset, linked asset instances.

### M3 — Accuracy and interior completeness (≈ 3 weeks, **critical path** — starts with M1, gates M2's sign-off)

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-3.1 | Closed-window recognition: visual detector (glass/frame/mullion masks) as a second evidence source next to the gap detector in `perception/architecture/windows.py`; fuse, don't replace | T4 (harness + integration by Sonnet) | medium | closed-window test set recall reported; shut window ≠ wall in WorldIR |
| **H-3.1** | User captures ≥ 15 rooms with shut windows, glass doors, reflective surfaces; labels window boxes | H | — | data in `benchmarks/windows/` |
| 8.6 | Scene-complete eval set (≥ 20 AI + ≥ 20 real labelled photos) | T2 harness + **H** labels | medium | scorer runs; baseline numbers committed |
| **H-3.2** | Tape/laser-measure **6 real rooms** (small, large, L-shaped, sloped ceiling, many windows, a stair) — walls, openings, ceiling height, socket positions | H | — | `benchmarks/measured_rooms/` committed |
| 12.2 | Measured-room accuracy report against the §1 tolerances | T1 | medium | error table per room/element; pass/fail per tolerance |
| T-3.3 | Accuracy fixes driven by 12.2 (scale from known references, plane fitting, wall-thickness rules, ceiling-height estimation) | T4 | high | 12.2 errors inside tolerance on ≥ 5 of 6 rooms |
| T-3.4 | Sockets, switches, light fixtures, radiators placed on the correct wall with height above floor (wire existing modules, item 34) | T1 + T4 | medium | recall reported on 8.6 set; positions within ±5 cm on measured rooms |
| T-3.5 | Non-rectangular rooms: L-shapes, bay windows, chamfered corners (13.4 subset) | T4 | high | L-shaped measured room within tolerance |
| T-3.2 | Finishes: keep real surface appearance; per-surface material label (paint, tile, wood, concrete) for the schedule | T4 | medium | label accuracy reported on measured rooms |

### M4 — Rooms join into one apartment; a second visit updates it (≈ 3–4 weeks)

Firms survey one room at a time and return for a second site visit. Both must land in the same project
model. Gate: capture 3 rooms of one apartment in separate sessions → one world with connected rooms through
shared doors; then capture one room again on a different day → second session is matched to the existing world, shows
new / updated / matched / uncertain / conflicting, user accepts, world version increments, no duplicate world.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-4.1 | Adoption contract ADR: WorldID/SessionID, candidate match, alignment confidence, change set, approval states (Constitution §3) | **T6** (Opus + user) | — | ADR in `docs/engineering/DESIGN_DECISIONS.md` |
| 3.6 | Identity scenario acceptance | T2 | high | adversarial identity tests pass |
| 6.2 / 6.3 | Incremental registration + update path wired to API | T3 sliced | high | e2e from gate passes |
| U.7 | Adoption review UI | T5 | high | e2e: second session matches, not duplicates |
| T-4.2 | Late reconstruction results never overwrite accepted manual edits (rule 18) | T3 | high | test: edit → reprocess → edit survives |

### M4b — Design iteration and client presentation (≈ 3 weeks)

Gate: a designer opens the surveyed apartment, swaps two furniture pieces, changes a floor finish, undoes
and redoes after a page reload, saves a named option ("Option A"), and sends the client a read-only link
that opens on a phone with a simple walk-through view.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-6.7 | Shared edit-command API + persistent per-world undo/redo (replace the in-memory 50-entry singleton) | T3 sliced | high | reload → undo still works; undo in world A never touches B |
| T-4b.1 | Move / rotate / delete furniture; replace with an item from a small built-in catalog (glTF, licensed CC0) | T2 | high | edits stored as commands with provenance `generated` |
| T-4b.2 | Change surface finish (wall paint colour, floor material) as a derived design layer — the as-built survey is never overwritten | T2 | medium | as-built export unchanged after edits (test) |
| T-4b.3 | Named design options = saved world versions with plain-language labels; compare as-built vs option | T2 | medium | two options exportable side by side |
| T-4b.4 | Read-only client share link (token-scoped, revocable, no account needed) | T2 | medium | security test: link cannot edit or see other worlds |
| T-4b.5 | Simple walk mode (eye-height first-person, collide with walls, touch + keyboard) | T5 | high | screenshots 375/1280; works on a phone |
| T-4b.6 | Still renders (PNG) from saved camera views for client decks | T1 | medium | render endpoint returns image for a saved view |

### M5 — Capture that a normal person can do (≈ 2–3 weeks, parallel-safe with M4)

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-5.1 | `navigator.geolocation.watchPosition` route + per-photo timestamped GPS/accuracy in `CaptureScreen.tsx` | T2 | medium | session holds N fixes, not 1 |
| T-5.2 | `GET /api/sessions/{id}/trajectory` returns stored points (stop returning `available: false`) | T1 | medium | API test with fixture route |
| T-5.3 | Offline capture: IndexedDB queue + resumable upload + resume-after-reload | T2 | high | kill tab mid-upload → resumes; Playwright test |
| T-5.4 | Live coverage hint after each batch (reuse coverage analysis; no nagging) | T1 | medium | hint shown from real coverage API |
| **H-5.1** | User walks 3 captures on a real phone (indoor + stairs + outdoor) | H | — | sessions uploaded and reviewed |

VIO / indoor positioning fusion is post-prototype (T3/T4).

### M6 — SaaS shell: strangers can use it (≈ 3 weeks)

Gate: a design partner signs up on a hosted URL, uploads, waits, exports, without the developer present.

| ID | Task | Tier | Effort | Done when |
|----|------|------|--------|-----------|
| T-6.1 | Hosting decision: 1 API box + 1 GPU worker (e.g. RunPod/Lambda/Modal) + S3-compatible storage | **T6** | — | ADR with monthly cost estimate |
| T-6.2 | Durable job queue (Redis/RQ or Postgres-backed) replacing in-process jobs; one GPU worker consumes | T3 sliced | high | worker restart loses no job (test) |
| T-6.3 | Storage adapter: ArtifactStore → S3-compatible behind existing interface; local stays default | T2 | medium | same tests pass against MinIO in CI |
| T-6.4 | Account lifecycle on top of existing tenant isolation: signup, invite, delete-my-data | T2 | medium | security review pass; delete removes artifacts |
| T-6.5 | Usage metering (GPU-minutes, storage, exports per tenant) — billing later | T1 | medium | per-tenant usage endpoint |
| T-6.6 | Observability: structured job logs + error tracking + per-stage timing dashboard | T1 | medium | a failed job is diagnosable from logs alone |
| T-6.8 | Security pass (HawkScan/DAST on hosted staging) | T1 | medium | no high findings |

### M7 — Pilot and release gate (≈ 2–4 weeks, mostly human)

| ID | Task | Tier | Done when |
|----|------|------|-----------|
| H-7.1 | 3–5 design-partner firms (architecture + interior design) survey real client projects | H | written feedback per firm |
| 12.1 | Living-room demo end to end (all 12 vision steps) | T1 verify | screenshots + benchmark JSON |
| 12.3 | Mark capabilities VERIFIED with evidence links | T0 | IMPLEMENTATION_STATUS updated |
| T-7.1 | Time-saved measurement: firm's site-measure + CAD redraw hours vs Reality Engine for the same project, plus dimension error vs their own laser measure | H + T0 report | table committed — this is the sales pitch |
| T-7.2 | Pricing + Stripe checkout | T6 then T2 | only after T-7.1 shows value |

---

## 4. Explicitly deferred (do not let Sonnet start these)

City-scale streaming/HLOD (48, 116), indoor–outdoor portal rendering (29, 39, 40, 114), game-engine
export (LOD chains, collision, Unity preset), CityGML/terrain/outdoor perception (13.7), Gaussian
splats/NeRF, linked asset instances, multi-device capture, drone/360 ingestion (13.3), NL editing
assistant UI (88), cloud distributed processing, real-time collaboration. Each is a T3–T5
project that needs M4/M6 foundations first. Revisit after M7 with partner feedback.

---

## 5. How a Sonnet session runs a task

```
Task <ID>: <title>  [Tier Tn, effort <level>]
Read: docs/blueprint/README.md, 33_TURNAROUND_PLAN.md §2, <spec docs>.
Files you may change: <allowlist>. Do not touch others.
Test first (T2+): <failing test path> — make it pass.
Done when: <copy>. Verify: <command> — paste real output.
Reuse first: 20_REUSE_MAP.md. Rules: 01_PRINCIPLES.md. Two-strike rule applies.
On finish: update 30_BUILD_PLAN.md status + IMPLEMENTATION_STATUS.md in the same PR.
```

Orchestration: the main session (Opus) picks the next unblocked task, writes the brief, spawns one Sonnet
agent per task in its own worktree (max 3 concurrent, non-overlapping files), reviews the diff, re-runs
Verify itself, merges. T3 tasks: Opus writes the ADR + slice list first; each slice is its own Sonnet task.

### Rough effort budget (estimate, not measured)

| Milestone | Sonnet sessions | Opus review sessions | Human days |
|-----------|-----------------|----------------------|------------|
| M0 | 5–7 | 1–2 | 0.5 |
| M1 | 10–15 | 4–6 | 1 |
| M2 | 12–18 | 5–7 | 1–2 (CAD/BIM import checks) |
| M3 | 10–16 | 5–7 | **6–8 (measuring + labelling)** |
| M4 | 12–20 | 6–8 | 1–2 |
| M4b | 12–18 | 5–7 | 1 |
| M5 | 6–9 | 2–3 | 1–2 |
| M6 | 14–20 | 6–8 | 2 (vendor setup) |
| M7 | 3–5 | 2 | **10+ (partners)** |

The largest risk is not code volume: it is **M3/M7 human data and partner time**. Start H-3.1, H-3.2 and
partner outreach during M0, not after M2.

---

## 6. Critical path

`M0 → M1 → (M2 ∥ M3) → M4 → M4b → M6 → M7` is the shortest route to a paying pilot.
M2 cannot be signed off until M3's accuracy report passes: deliverables with wrong dimensions do not count.
`M5` (capture) runs beside `M4`/`M4b` once M1 is green. Start H-0.1, H-3.1 and H-3.2 during M0.
