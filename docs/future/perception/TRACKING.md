# Tracking

Status: MISSING (2026-09-14)

## Purpose

Maintain object identity across **time** within a single video source:
frame-to-frame association producing per-object tracks that anchor
multi-view identity (see `MULTI_VIEW_IDENTITY.md`) and give entities
temporal extent (first_seen, last_seen, continuity).

## Current state

No tracking stage exists. Video sources are consumed as image sets for
SfM; detections are per-image only.

## Approach

- **Association tracking (ByteTrack/OC-SORT style)** on detection
  boxes + appearance embeddings, implemented on OpenCV primitives —
  no deep tracker dependency for the foundation.
- Track state: id, per-frame detections, age, hits, time since update.
- Tracks are evidence for identity, not entities themselves; the
  identity layer decides merges across tracks/views.

## Canonical representation

`Track` (perception module): `track_id`, `source_id`,
`detections: [(image_or_frame_id, box, embedding_ref, score)]`,
`start_ts`, `end_ts`, `gaps` — persisted as artifacts with provenance.

## Failure modes

- Occlusion → track breaks; gap handling must be explicit (max-age
  config), a broken track is reported as broken, not stitched silently.
- Static camera vs moving camera: moving cameras (our primary case)
  need motion-compensated association — use per-frame relative pose
  from SfM/VIO when available; record which was used.
- ID switches: report switch count as a track-quality diagnostic.

## Acceptance criteria

- Deterministic synthetic video: known linear motion → track continuity
  with zero ID switches; occluded segment → gap recorded, switch counted.
- Tracks persist with provenance; reload reproduces identical tracks.

## Priority

P2 (with multi-view identity).
