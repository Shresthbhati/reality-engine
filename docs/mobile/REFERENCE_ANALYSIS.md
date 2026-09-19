# Mobile Reference Analysis (ui ux pics → screen map)

Sources: `ui ux pics/` (120 ChatGPT-generated concept frames, 2026-09-18).
The set mixes desktop Studio frames with mobile instrument frames. The mobile
frames below directly shaped `frontend/src/apps/mobile/*`.

## Reference → implementation map

| Reference motif (observed) | Decision in code |
|---|---|
| Dark navy `#0d0e12` canvas, `#171922` panels, `#2b3040` borders; cyan `#3d8ef7` primary; green `#2ecc71` / red `#e74c3c` / amber `#f1c40f` state colors; mono microcopy for metadata | Token values reused verbatim across Home/Capture/Review/Sync screens |
| Phone-instrument frame with huge bottom shutter, camera-first layout | `CaptureScreen.tsx`: full-bleed viewfinder, single 64px shutter in thumb zone, everything else secondary |
| Top-of-frame status ribbon (session/state) | Session header with live counts: "N captured · n useful · n rejected · n redundant" |
| Verdict/quality cards (green=useful, amber=caution, red=reject) | Post-shutter banner driven by computed verdict, color-mapped identically |
| Evidence grid with per-item status dots | `ReviewScreen.tsx` 3-col grid, verdict dots, tap-through inspector with real metrics |
| Mono metadata rows (ids, telemetry) | Inspector dl: sharpness vs threshold, exposure %, clipped %, Δ previous, sha256 id, provenance line |
| Honest "unknown" states rather than placeholder numbers | All unknown telemetry renders literally as "unknown — no X data reported" |

## What was deliberately NOT copied

- Simulated RTK/PTP/±0.016m telemetry from some desktop-capture concept frames:
  the browser cannot report those, and faking them violates the no-fake-completion
  rule. Only values the device actually reports (GNSS, compass) render.
- Multi-pass SITE/STRUCTURE/FACADE/DETAIL orchestration from the desktop
  CaptureApp: requires backend guidance state (see BACKEND_CONTRACT.md
  `captureTasks`); a client-only fake would be invented data.
- Desktop Studio panels: mobile is a separate product, per the brief.

## Density / interaction decisions

- One-hand: all primary actions in the bottom third; tab bar in thumb zone.
- Glanceability: verdict = color dot + one sentence; numbers only on demand
  (inspector), never walls of telemetry.
- Low typing: session name optional (sensible default), no forms elsewhere.
