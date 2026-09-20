# HANDOFF

Agent: Cline
Branch: agent/cline-mobile-experience
Checkpoint: MOBILE_FIELD_CAPTURE_READY
Status: IMPLEMENTATION_READY

## What exists

The Reality Engine field mobile app (frontend/src/apps/mobile/, routed at
frontend/src/app/phone) is complete and integrated end to end:

- **Session lifecycle** (HomeScreen + store.ts): create/select sessions backed by
  the real `Session` abstraction (evidence/session_store.py via
  frontend/src/app/api/sessions). Frames get sha256 content IDs, real capture
  timestamps, device camera label, and telemetry **only when the device reports
  it** — GPS/compass are shown as UNAVAILABLE otherwise. Nothing is fabricated.
- **Capture** (CaptureScreen.tsx): live getUserMedia camera, shutter computes
  real pixel quality (blur via Laplacian variance, exposure histogram,
  near-duplicate via sha256 distance — quality.ts) and issues an honest verdict
  per frame: USEFUL / REDUNDANT / REJECTED with an explainable reason string.
  Frames are stored in the device vault (frameVault.ts, IndexedDB) with their
  full pixels.
- **Coverage & guidance** (coverage.ts, wired into CaptureScreen and
  ReviewScreen): compass-octant coverage computed from real USEFUL frames'
  headings. Coverage is UNKNOWN when no heading telemetry exists — never
  invented. Field messages like "You don't need more footage here" /
  "Next: capture the north-facing facade…" are generated only from real frame
  state. GPS bounds shown when present.
- **Evidence review** (ReviewScreen.tsx): per-frame inspector with metrics,
  reasons, verdict filters; rejected/redundant frames stay visible (lineage).
- **Sync / handoff** (SyncScreen.tsx + bundle.ts): export of a verified
  `re.mobile-session-bundle/v1` JSON containing per-frame sha256 IDs, quality
  verdicts as *advisory triage*, telemetry, and base64 pixel payloads read back
  from the vault (re-hashed on export — pixels are verified, not just metadata).
  OTA ingest does not exist; the UI says so honestly. Consumed on desktop with:
  `reality session ingest-mobile <bundle.json> --session-dir <dir>` which
  verifies every frame hash and records task completion back into the session.
- **Capture tasks** (desktop→mobile reverse flow): desktop-issued capture
  requests are tracked with status and serviced-by frame IDs, surfaced in
  CaptureScreen as a request banner and in Review/Sync.

## Files changed

- frontend/src/apps/mobile/* (types, store, quality, coverage, frameVault,
  bundle, MobileFieldApp, Home/Capture/Review/Sync screens)
- frontend/src/app/phone/page.tsx (route)
- docs/mobile/BACKEND_CONTRACT.md, docs/mobile/REFERENCE_ANALYSIS.md
- evidence: ingest-mobile + mobile-tasks CLI verbs (apps/cli) — see BACKEND_CONTRACT.md

## Public contracts

- `re.mobile-session-bundle/v1` (docs/mobile/BACKEND_CONTRACT.md) — the
  mobile→desktop handoff artifact. Frame identity is verified on ingest.
- `quality.ts` pure functions (assessFrame etc.) and `coverage.ts`
  (computeCoverage / coverageGuidance / coverageSummaryLine) — deterministic,
  testable, no I/O.
- Frontend API: /api/sessions, /api/status (bridge to Python session store).

## How to consume

- Desktop (Antigravity/Studio): run
  `reality session ingest-mobile bundle.json --session-dir DIR`, then
  `reality session mobile-tasks bundle.json -o tasks.json` to emit follow-up
  capture requests; bundle these into a sync response to complete the loop.
- Reconstruction (FreeBuff): USEFUL frames in the bundle are candidates;
  verdicts are advisory — re-derive quality engine-side.

## Tests run

- `tsc --noEmit`: zero errors in src/apps/mobile/**. Remaining project errors
  are all in another agent's studio files (InteractiveMeasurement/
  WorldHealthDashboard) — not touched by mobile.
- Production `next build`: verified compiling earlier in this branch; latest
  run pending at handoff time (see Runtime verification).

## Runtime verification

- Dev server + phone route exercised in-browser during development: session
  create, camera permission flow (denied → explicit error + retry), capture
  verdict overlay, review filters, export bundle.
- On devices without camera/geolocation the UI shows UNAVAILABLE states; no
  simulated telemetry anywhere in the production path.

## Dependencies

- IndexedDB vault for frame pixels (bundle export requires local pixels; frames
  without them are listed as skipped, never silently dropped).
- Python CLI `session ingest-mobile` for desktop consumption.

## Known limitations

- No OTA sync server (documented in BACKEND_CONTRACT.md; UI is explicit).
- Coverage is compass-heading based (octants); no pose-graph overlap model yet.
- Devices without compass/GPS show unknown coverage rather than estimates.

## Known bugs

- None known in mobile code at this checkpoint.

## Next agent

ANTIGRAVITY (desktop/integration) and FREEBUFF (evidence/reconstruction).

## Exact action for next agent

1. Ingest a produced bundle: `reality session ingest-mobile <bundle.json>
   --session-dir DIR` and confirm hashes verify and frames land in the session.
2. Emit and return capture tasks via `reality session mobile-tasks` to close
   the desktop→mobile loop.
3. If building an OTA ingest endpoint, follow docs/mobile/BACKEND_CONTRACT.md
   — do not weaken bundle verification.
