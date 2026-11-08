from perception.instances.appearance import (
    AppearanceDescriptor,
    appearance_similarity,
    compute_color_histogram,
)
from perception.instances.bounding import OrientedBoundingBox
from perception.instances.dinov2_embeddings import (
    DINOV2_EMBEDDING_DIM,
    DINOV2_MODEL_NAME,
    Dinov2Descriptor,
    dinov2_similarity,
    extract_dinov2_embedding,
    fuse_appearance_cues,
)
from perception.instances.dinov2_reid import (
    DINOv2ReID,
    ReIDBenchmarkItem,
    ReIDBenchmarkReport,
    benchmark_reid_vs_color_histogram,
)
from perception.instances.object_geometry import (
    ObjectMesh,
    attach_object_mesh_to_world,
    build_chair_mesh,
    build_door_mesh,
    build_sofa_mesh,
    build_switchboard_mesh,
    build_table_mesh,
    reconstruct_object_mesh,
)
from perception.instances.track_backend import (
    MultiViewIdentityTrackBackend,
    build_images_dict,
)

__all__ = [
    "MultiViewIdentityTrackBackend",
    "build_images_dict",
    "OrientedBoundingBox",
    "ObjectMesh",
    "reconstruct_object_mesh",
    "attach_object_mesh_to_world",
    "build_sofa_mesh",
    "build_chair_mesh",
    "build_table_mesh",
    "build_door_mesh",
    "build_switchboard_mesh",
    "AppearanceDescriptor",
    "compute_color_histogram",
    "appearance_similarity",
    "Dinov2Descriptor",
    "extract_dinov2_embedding",
    "dinov2_similarity",
    "fuse_appearance_cues",
    "DINOV2_EMBEDDING_DIM",
    "DINOV2_MODEL_NAME",
    "DINOv2ReID",
    "ReIDBenchmarkItem",
    "ReIDBenchmarkReport",
    "benchmark_reid_vs_color_histogram",
]
