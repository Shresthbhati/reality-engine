"""Dense-MVS ingestion (directive P5): turn external dense output into
a REALITY ENGINE capability.

The CUDA COLMAP dense run produces fused.ply -- external evidence.
This module completes the chain that makes it the engine's own:

    fused.ply bytes
      -> parse_fused_ply (reconstruction.backend.dense_output)
      -> PointCloudData  (world_ir.geometry_data: deterministic
         float64 canonical encoding)
      -> ArtifactStore.put (content-addressed, hashed)
      -> WorldIR Geometry(type=POINTCLOUD) with dense provenance
         (sensor_type "dense_mvs_fused", source
         "colmap_stereo_fusion", per-run counts)

Honesty rules:
- provenance names the TRUE producer (multi-view stereo fusion), never
  a monocular prior -- MiDaS/monocular depth is a different, weaker
  evidence class and must never be recorded as MVS;
- a geometry without a persisted artifact is refused (an untraceable
  dense cloud would violate the provenance backbone);
- an empty cloud is refused, not recorded as a zero-point success.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from world_ir.artifact_store import ArtifactStore
from world_ir.geometry_data import PointCloudData


class DenseIngestError(ValueError):
    """Dense ingestion refused (bad input or missing store)."""


def ingest_fused_ply(
    data: bytes,
    *,
    artifact_store: Optional[ArtifactStore],
    source_evidence_ids: Sequence[str],
    scale_factor: float = 1.0,
) -> "object":
    """Parse fused.ply, encode canonically, persist, and return the
    WorldIR Geometry (POINTCLOUD) referencing the artifact.

    `scale_factor` rescales positions into meters when the caller has a
    MEASURED metric anchor (e.g. meters_per_unit from the pipeline's
    scale anchoring): COLMAP's fused output is in COLMAP's own SfM
    scale, and recording it as meters unanchored would be a scale lie.
    The factor is recorded in the geometry's observation metadata so
    the rescale is always traceable. 1.0 (no anchor) is the honest
    default -- the caller then gets COLMAP-scale coordinates and the
    record says so."""
    from reconstruction.backend.dense_output import DenseOutputError, parse_fused_ply

    if artifact_store is None:
        raise DenseIngestError(
            "no artifact_store configured -- a dense cloud without a "
            "persistent artifact would be untraceable"
        )
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise DenseIngestError(
            f"scale_factor must be a positive finite number, got {scale_factor!r}"
        )
    try:
        points, facts = parse_fused_ply(
            data, source_evidence_ids=list(source_evidence_ids), return_facts=True
        )
    except DenseOutputError:
        raise
    if not points:
        raise DenseOutputError("fused.ply carries zero points -- nothing to ingest")

    cloud = PointCloudData.from_positions(
        [
            (
                p.position[0] * scale_factor,
                p.position[1] * scale_factor,
                p.position[2] * scale_factor,
            )
            for p in points
        ]
    )
    data_uri, data_hash = artifact_store.put(cloud.to_bytes())

    xs = [p.position[0] * scale_factor for p in points]
    ys = [p.position[1] * scale_factor for p in points]
    zs = [p.position[2] * scale_factor for p in points]

    from provenance import Provenance
    from world_ir import Geometry, GeometryType, Observation, Vector3

    geometry = Geometry(
        type=GeometryType.POINTCLOUD,
        lod_level=0,
        vertex_count=len(points),
        data_uri=data_uri,
        data_hash=data_hash,
        bounds_min=Vector3(x=min(xs), y=min(ys), z=min(zs)),
        bounds_max=Vector3(x=max(xs), y=max(ys), z=max(zs)),
        provenance=Provenance.ESTIMATED,
        confidence=0.7,
        observations=[Observation(
            sensor_type="dense_mvs_fused",
            confidence=0.7,
            metadata={
                "source": "colmap_stereo_fusion",
                "n_points": len(points),
                "n_source_images": len(source_evidence_ids),
                "source_evidence_ids": list(source_evidence_ids),
                "scale_factor": scale_factor,
                "scale_note": (
                    "positions multiplied by scale_factor from the "
                    "pipeline's measured metric anchor; 1.0 means the "
                    "cloud is in COLMAP's own SfM scale, honestly "
                    "recorded as such"
                ) if scale_factor != 1.0 else
                "no metric anchor applied -- coordinates are in COLMAP's "
                "own SfM scale, recorded as such (never claimed as meters)",
                "had_colors": facts["had_colors"],
                "had_normals": facts["had_normals"],
                "encoding": "PointCloudData v1 (float64 LE)",
                "note": "true multi-view stereo fusion output; monocular "
                "depth priors are a separate, weaker evidence class and "
                "are never recorded under this source",
            },
        )],
    )
    return geometry
