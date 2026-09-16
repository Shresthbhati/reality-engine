# tracking

Temporal tracking (P7-02) — **PARTIAL**.

Implemented (`temporal.py`, 2026-09-16): 3D observation-level identity
across time. `TimedObservation` (observation_id, label, position,
evidence_id, capture time, measured confidence) chains per label into
`TrackRecord`s with measured path length, duration, and implied speed;
an observation joins an open track only when both the speed budget and
the gap budget pass, else it starts a new track — no forced
associations. Untimed observations are returned separately, never
mixed into timed chains. Track confidence is the minimum of member
observations' measured confidences; a track with none is reported with
confidence `None` (the InstanceTrack adapter refuses to fabricate a
number).

Still open (spec `docs/future/perception/TRACKING.md`): 2D
detection-box association (ByteTrack/OC-SORT style on boxes +
embeddings), per-frame relative-pose motion compensation for moving
cameras, ID-switch diagnostics, provenance-persisted tracks.
