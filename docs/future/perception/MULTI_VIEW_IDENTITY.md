# Multi-View Identity

Status: MISSING (2026-09-14)

## Purpose

Decide that the same physical object seen in multiple images (or
multiple sessions/sources) is one object — and that two different
objects are not — so the world contains **entities**, not per-view
detections. This is the foundation for persistent world state
(WorldStore, P4) and for cross-source registration via landmarks.

## Current state

Per-image detection exists (SAM-backed, `perception/`) and planes are
promoted to entities (`evidence/promote_planes.py`). Detection→entity
association across views does not exist: each image's detections are
isolated. Track IDs exist for depth-fusion provenance, not object
identity.

## Approach

Layered, deterministic first:

1. **Geometry-gated association.** Two detections in different views
   are candidates for the same entity when their 3D back-projections
   (using fused depth + poses) land within a distance/normal tolerance
   and IoU in reprojected space is plausible. Pure 2D IoU alone is
   never sufficient.
2. **Appearance embedding.** CLIP/DINOv2 embeddings compared across
   views break geometry ties (duplicate chairs) — same-backend only,
   recorded in provenance.
3. **Temporal tracking (video).** ByteTrack-style association for
   within-video identity, feeding the cross-view association as
   additional evidence — see `TRACKING.md`.
4. **Human review.** Low-confidence associations surface in Studio as
   review tasks, not silent merges (Studio P6).

## Model

Entity merge is an evidence-weighted decision:

- `association_evidence`: [(image_a, det_a, image_b, det_b, score,
  method)], each with provenance
- decision threshold is configuration, recorded in world metadata
- merged entity keeps `identity_confidence` (uncertainty, P3)
- splits are allowed later (no destructive merge: association is
  additive evidence, identity can be revised — provenance survives)

## Failure modes

- Symmetric scenes (four identical chairs) → wrong merges; appearance
  embeddings alone insufficient; require geometry + count consistency.
- Pose error → 3D association fails; degrade to UNVERIFIED identity,
  never silently guessed.
- Model availability: SAM/torch-hub breakage currently fails open in
  env — detection stage must keep its honest BACKEND_UNAVAILABLE
  behavior (pre-existing env failure in this machine's test suite).

## Acceptance criteria

- Deterministic fixture: two rendered views of the same synthetic
  object → one entity, two views of distinct objects → two entities.
- Association records are persisted artifacts with provenance.

## Priority

P2 (needs metric geometry from P1 to be reliable).
