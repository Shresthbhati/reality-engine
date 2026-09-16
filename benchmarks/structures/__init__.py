"""Structure benchmark records (directive section 33): the real
capture targets. Each record is structure-agnostic machinery -- only
identity, location, and capture requirements are named here. A record
whose capture has not happened stays CAPTURE_PENDING and cannot
produce a run (benchmarks/architectural.py refuses; nothing is
simulated).

Victoria Memorial is the FIRST benchmark, not a special case: the
same record type serves forts, skyscrapers, ordinary buildings
(directive section 34's generalization requirement).
"""

from __future__ import annotations

from benchmarks.architectural import CaptureStatus, StructureBenchmark

#: The directive's first proof target. Capture requirements follow
#: directive section 33 exactly.
VICTORIA_MEMORIAL = StructureBenchmark(
    structure_name="Victoria Memorial",
    location="Kolkata, India",
    capture_status=CaptureStatus.CAPTURE_PENDING,
    capture_requirements=[
        "multiple viewpoints with full walkaround coverage",
        "different distances: wide shots + facade detail shots",
        "facade coverage: portico, colonnades, dome, corners",
        "GNSS where available",
        "IMU where available",
        "camera metadata, timestamps, calibration",
    ],
)

#: Registered benchmarks (directive section 34: generalization list).
ALL_BENCHMARKS = (VICTORIA_MEMORIAL,)
