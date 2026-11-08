"""Routes for Multi-Epoch 4D As-Built Temporal Delta & Automated Site Change Detection (Track D).

Provides endpoints to:
- List available capture epochs across a world's lineage.
- Compute comprehensive 4D temporal deltas (added, removed, moved, renovated, unchanged).
- Detect volumetric spatial point cloud change clusters between two dates/epochs.
- Generate site progress and discrepancy audit summaries.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldVersion
from apps.api.routes_worlds import _load_world_or_409, _render_points_ply, _version_row
from engine.pipeline.temporal_delta import (
    compute_temporal_delta,
    detect_spatial_change_clusters,
)

log = logging.getLogger("reality.api.temporal")

temporal_router = APIRouter(prefix="/api/worlds", tags=["temporal"])


@temporal_router.get("/{world_id}/epochs")
async def list_world_epochs(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """List available capture epochs/versions for a world in chronological order."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldVersion)
        .where(WorldVersion.world_id == world_id)
        .order_by(WorldVersion.created_at.asc())
    )
    rows = res.scalars().all()

    epochs = []
    for r in rows:
        label = r.label or f"Epoch {r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else r.id[:8]}"
        is_current = r.id == w.current_version_id
        entity_count = 0
        try:
            wir = _load_world_or_409(r)
            entity_count = len(getattr(wir, "entities", {}) or {})
        except Exception:
            pass

        epochs.append(
            {
                "epoch_id": r.id,
                "label": label,
                "capture_date_iso": r.created_at.isoformat() if r.created_at else None,
                "parent_version_id": r.parent_version_id,
                "is_current": is_current,
                "entity_count": entity_count,
            }
        )

    return {
        "world_id": world_id,
        "current_epoch_id": w.current_version_id,
        "epochs": epochs,
    }


@temporal_router.get("/{world_id}/temporal/compare")
async def compare_world_epochs(
    world_id: str,
    from_epoch: Optional[str] = Query(None, description="Base epoch ID (default: parent of head or first version)"),
    to_epoch: Optional[str] = Query(None, description="Target epoch ID (default: current head version)"),
    tolerance_m: float = Query(0.03, ge=0.005, le=1.0, description="Movement tolerance threshold in meters"),
    include_spatial_clusters: bool = Query(False, description="Whether to compute 3D point cloud voxel change clusters"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Compute 4D temporal as-built delta between two capture epochs of a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    # Resolve target (to_epoch)
    row_to = await _version_row(db, world_id, to_epoch, auth=auth)
    world_to = _load_world_or_409(row_to)

    # Resolve base (from_epoch)
    row_from: Optional[WorldVersion] = None
    if from_epoch:
        row_from = await _version_row(db, world_id, from_epoch, auth=auth)
    else:
        # If parent exists on row_to
        if row_to.parent_version_id:
            row_from = await db.get(WorldVersion, row_to.parent_version_id)
        if row_from is None:
            # Fall back to oldest version
            res = await db.execute(
                select(WorldVersion)
                .where(WorldVersion.world_id == world_id)
                .order_by(WorldVersion.created_at.asc())
            )
            row_from = res.scalars().first()

    if row_from is None or row_from.id == row_to.id:
        # Return self-comparison (all unchanged)
        world_from = world_to
        from_id = row_to.id
        from_time = row_to.created_at.isoformat() if row_to.created_at else None
    else:
        world_from = _load_world_or_409(row_from)
        from_id = row_from.id
        from_time = row_from.created_at.isoformat() if row_from.created_at else None

    to_id = row_to.id
    to_time = row_to.created_at.isoformat() if row_to.created_at else None

    # Compute entity delta
    summary = compute_temporal_delta(
        world_from=world_from,
        world_to=world_to,
        from_epoch_id=from_id,
        to_epoch_id=to_id,
        movement_tolerance_m=tolerance_m,
        from_timestamp_iso=from_time,
        to_timestamp_iso=to_time,
    )

    # Compute point cloud spatial change clusters if requested
    if include_spatial_clusters and from_id != to_id:
        try:
            ply_from = _render_points_ply(world_from)
            ply_to = _render_points_ply(world_to)
            if ply_from and ply_to:
                pts_from = _parse_ply_points(ply_from)
                pts_to = _parse_ply_points(ply_to)
                if len(pts_from) >= 10 and len(pts_to) >= 10:
                    clusters = detect_spatial_change_clusters(
                        pts_from, pts_to, voxel_size=0.15, min_cluster_voxels=5
                    )
                    summary.spatial_clusters = clusters
        except Exception as exc:
            log.warning("Spatial change clustering skipped: %s", exc)

    return summary.to_dict()


def _parse_ply_points(ply_bytes: bytes) -> list[tuple[float, float, float]]:
    """Parse (x, y, z) point coordinates from an ASCII PLY byte buffer."""
    lines = ply_bytes.decode("ascii", errors="replace").splitlines()
    positions: list[tuple[float, float, float]] = []
    in_body = False
    for line in lines:
        if line.strip() == "end_header":
            in_body = True
            continue
        if in_body and line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    return positions
