"""Evidence layer: Sessions, Datasets, and session merging (spec §3-6 & Wave 3 Block 17).

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
from .session_grouping import SessionGroup, group_evidence_by_session, haversine_distance_m
from .dedup import (
    DeduplicationResult,
    DuplicateRecord,
    compute_dhash,
    compute_phash,
    deduplicate_images,
    hamming_distance,
    hash_to_hex,
    hex_to_hash,
)
from .video import (
    SceneAwareFrameSelectionStrategy,
    SceneTransition,
    compute_frame_difference,
    detect_scene_transitions_from_frames,
    extract_video_keyframes,
)
from .cad_importer import (
    CadFloorPlan,
    CadImportRefused,
    CadRoomAnnotation,
    import_cad_floorplan_file,
    parse_dxf_text,
    promote_cad_floorplan,
)

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
    "SessionGroup",
    "group_evidence_by_session",
    "haversine_distance_m",
    "DeduplicationResult",
    "DuplicateRecord",
    "compute_dhash",
    "compute_phash",
    "deduplicate_images",
    "hamming_distance",
    "hash_to_hex",
    "hex_to_hash",
    "SceneAwareFrameSelectionStrategy",
    "SceneTransition",
    "compute_frame_difference",
    "detect_scene_transitions_from_frames",
    "extract_video_keyframes",
    "CadFloorPlan",
    "CadImportRefused",
    "CadRoomAnnotation",
    "import_cad_floorplan_file",
    "parse_dxf_text",
    "promote_cad_floorplan",
]
