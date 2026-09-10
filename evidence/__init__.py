"""Evidence layer: Sessions, Datasets, and session merging (spec §3-6).

The foundational unit downstream reconstruction/WorldIR work must trace
back to. See docs/CAPABILITY_MATRIX.md for what's built on top of this
and what still isn't.
"""

from .session import (
    EvidenceKind,
    SessionStatus,
    EvidenceItem,
    ProcessingRecord,
    Session,
    DuplicateEvidenceError,
    UnknownEvidenceError,
    SessionClosedError,
)
from .dataset import Dataset, MergedContext, UnknownSessionError, DuplicateSessionError
from .promote import promote_measurement_to_entity, UnsupportedEvidenceKindError

__all__ = [
    "promote_measurement_to_entity",
    "UnsupportedEvidenceKindError",
    "EvidenceKind",
    "SessionStatus",
    "EvidenceItem",
    "ProcessingRecord",
    "Session",
    "DuplicateEvidenceError",
    "UnknownEvidenceError",
    "SessionClosedError",
    "Dataset",
    "MergedContext",
    "UnknownSessionError",
    "DuplicateSessionError",
]
