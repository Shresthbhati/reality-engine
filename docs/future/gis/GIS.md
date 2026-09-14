# GIS Integration

Status: MISSING (2026-09-14) — intentionally deferred, not a current
bottleneck (see ROADMAP phase P5)

## Purpose

Place reconstructed worlds in real-world geographic context: from
GPS-tagged captures, anchor worlds to Earth-fixed frames; support
site-scale captures (building/campus) where single-session SfM ends.

## Current state

GPS metadata is carried in manifests and preserved (never zeroed —
CLAUDE.md §44), but nothing consumes it geometrically. Worlds are in
arbitrary frames unless a metric prior gives scale; ENU/ECEF concepts
appear only in `COORDINATE_FRAMES.md` as future frames.

## Scope when implemented

1. **GNSS → world anchor.** When capture GPS has plausible accuracy,
   define the world frame as ENU at a reference epoch; camera poses
   get lat/lon/alt via the recorded transform. Accuracy (HDOP,
   device-reported) propagates as uncertainty (P3).
2. **Geo-tiled backdrops.** Map/elevation tiles (OSM/SRTM-class,
   cached, offline-first — no network at runtime) as ground-truthing
   layers, not as reconstruction inputs.
3. **Site-scale extension.** Multi-session worlds (WorldStore P4)
   + per-session GNSS anchors → site frame; drift between sessions
   handled by registration (P1), not by averaging GPS.
4. **Coordinate services.** `reality geo register`, ENU↔ECEF↔WGS84
   conversions via pyproj-class library (mature, do not hand-roll
   datum math).

## Non-goals

- Real-time SLAM-with-GPS fusion (VIO P1 handles trajectory; GPS is
  an anchor, not the primary odometry source).
- Hiding GPS quality: unknown/low-accuracy GPS stays explicit.

## Failure modes

- Urban canyon multipath → anchor uncertainty large; recorded, world
  still in ENU but with honest covariance.
- Datum confusion (WGS84 vs local) → single conversion module, tests
  against known control points.

## Acceptance criteria

- Deterministic: synthetic GNSS tags + known offsets → ENU anchors
  recovered within tolerance; covariance propagates.
- No-network guarantee: all geo layers served from local cache in
  tests (CI never fetches).

## Priority

P5 (needs WorldStore multi-session + registration first).
