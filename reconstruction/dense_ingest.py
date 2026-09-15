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
) -> "object":
    """Parse fused.ply, encode canonically, persist, and return the
    WorldIR Geometry (POINTCLOUD) referencing the artifact."""
    from reconstruction.backend.dense_output import DenseOutputError, parse_fused_ply

    if artifact_store is None:
        raise DenseIngestError(
            "no artifact_store configured -- a dense cloud without a "
            "persistent artifact would be untraceable"
        )
    try:
        points, facts = parse_fused_ply(
            data, source_evidence_ids=list(source_evidence_ids), return_facts=True
        )
    except DenseOutputError:
        raise
    if not points:
        raise DenseOutputError("fused.ply carries zero points -- nothing to ingest")

    cloud = PointCloudData.from_positions([p.position for p in points])
    data_uri, data_hash = artifact_store.put(cloud.to_bytes())

    xs = [p.position[0] for p in points]
    ys = [p.position[1] for p in points]
    zs = [p.position[2] for p in points]

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
