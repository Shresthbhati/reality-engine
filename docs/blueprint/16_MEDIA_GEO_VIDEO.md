# 16 — Media, video and geolocation

**Scope:** photos are v1. Video, 360° and GPS anchoring are Phase 9+, built on
the same ingestion path and ledger. This doc fixes the contracts now so photo
code doesn't need rework later.

## One ingestion path for all media

```
file → store original (immutable, sha256) → probe (type, metadata) →
  photo:  1 Image
  video:  keyframes → N Images (each knows its parent video + timestamp)
  360°:   equirect/cubemap faces → N Images with a spherical camera model
  track:  GPX/NMEA → time-indexed positions attached to Images by timestamp
→ S1 quality → pipeline
```

Every derived file (resized photo, frame, cube face) links to its original.
Originals are never overwritten (P7).

## Photo metadata (v1)

Extract: dimensions, orientation, timestamp, camera make/model, focal length,
GPS (lat/lon/alt/accuracy if present). Today: Pillow EXIF in
`evidence/image_check.py`. Add ExifTool later for HEIC/RAW/video metadata.

## Video keyframes (Phase 9)

- Decode with FFmpeg / PyAV; keep presentation timestamps.
- Select frames by **information gain**, not fixed fps: camera motion,
  parallax, novelty, sharpness, coverage gain. Drop near-duplicates, but never
  drop a frame only because it looks similar (small motion = useful parallax).
- Target: a 5-minute walk-around → roughly 100–300 keyframes, not 9,000.
- Frame tracking (`perception/tracking/`) helps association within one video
  but is **not** cross-view identity; S9 still decides entities.

## Geolocation (Phase 9)

- Two coordinate systems: a **local metric** frame for geometry and a
  **geographic** anchor (lat/lon/alt/heading/CRS/accuracy) on the World. The
  transform between them is explicit and versioned. Never put lat/lon into mesh coordinates.
- GPS is a prior, not a pose. Indoors it is usually useless — say so in the UI.
- CRS transforms via PROJ (pyproj). Map display via maplibre (already in the frontend).
- Existing pieces: `evidence/sensors.py`, `evidence/clocks.py` (time sync),
  `trajectories/` (VIO/SLAM backends, subprocess-wrapped).

## Capture modes

1. **Upload** (v1).
2. **Guided capture** (Phase 6): capture requests tell the user where to stand.
3. **Live capture** (later): the phone streams frames + IMU; SLAM gives poses.
