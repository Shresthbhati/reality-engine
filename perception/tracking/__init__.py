"""Temporal tracking (P7-02).

See temporal.py for scope: 3D observation-level identity across time
-- TimedObservation -> TrackRecord chains with measured path/duration/
implied-speed, per-label, honest about untimed observations and about
tracks whose confidence was never measured. 2D detection-box
association (ByteTrack-style, spec TRACKING.md) is the still-open
complement; the two layers share the honesty contract.
"""
