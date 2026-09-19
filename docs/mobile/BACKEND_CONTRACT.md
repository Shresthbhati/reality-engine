# Mobile ↔ Desktop Backend Contract

Status: **PROPOSED — not implemented in the engine yet.** The mobile field app
(`frontend/src/app/phone`) runs entirely client-side today: sessions persist in
`localStorage`, quality is computed on-device, and handoff happens via the
session export file (schema `re.mobile-session/v1`). The REST contract below is
the smallest interface needed for true over-the-air sync; it is documented here
so the engine team can implement it without guessing.

## Existing (real, shipped)

- `GET/POST` none. Export/import file: `re.mobile-session/v1` JSON containing
  `sessionId` (uuid), frames with:
  - `frameId`: `frame:<sha256[0:16]>` — content-addressed, mirrors the engine's
    `folder:<hash16>` source identity rule.
  - `contentSha256`: full sha256 hex of the frame bytes.
  - `quality`: `{ sharpness (Laplacian variance), exposure, clippedRatio,
    diffVsPrevious }` — computed on-device; deterministic given the bytes.
  - `verdict` + `reasons`: on-device triage (USEFUL/REDUNDANT/REJECTED).
    **Advisory only** — the desktop pipeline re-derives its own judgment;
    mobile verdicts never mutate engine state.
  - `telemetry`: device-reported only (GNSS, compass). Absent = unknown.
  - `provenance`: origin/device/time — maps to `EvidenceSource` provenance.

Ingest on desktop today: `reality ingest <folder>` (CLI) — export the session,
place frames in `rgb/`, and the canonical `MultiSourceSession.add_source()`
path classifies and registers them. `apps/capture` (Python) remains a scaffold.

## Proposed REST contract (v1)

Base: `NEXT_PUBLIC_RE_API_BASE` (none configured = sync stays in its honest
blocked state; the UI never fakes a green sync).

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/sessions` | List sessions visible to this device (incl. desktop-created capture tasks) |
| `POST` | `/sessions` | Create/claim a session (body: mobile session record) |
| `GET`  | `/sessions/{id}` | Full session record (same schema as export file) |
| `PUT`  | `/sessions/{id}/frames/{frameId}` | Upload frame: multipart (`blob`, `meta` JSON). Server verifies sha256 and stores content-addressed |
| `POST` | `/sessions/{id}/finish` | Mark session READY_FOR_PIPELINE; triggers desktop-side ingest |

Rules the server MUST keep (mirrors engine principles):

1. Frame identity is `sha256(bytes)` — never a filename or client timestamp.
2. Mobile quality verdicts are stored as **advisory metadata**; engine-side
   quality/redundancy is recomputed and authoritative.
3. Unknown telemetry stays unknown (no zero-defaults for GNSS/heading).
4. Sessions are append-only on the evidence layer; deletions are explicit
   records, not silent removals.

## Desktop → mobile capture tasks

`GET /sessions` returns sessions with `captureTasks[]`
(`{ task_id, target_area, guidance, created_by, status }`). The mobile app
surfaces them as "requested captures" on the session card. Not implemented
client-side yet — tracked as remaining work.
