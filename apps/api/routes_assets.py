"""Routes for extracted 3D assets (T-2.6).

Allows extracting individual furniture and architectural elements from surveyed rooms
into independent, reusable, downloadable 3D assets while keeping the source world untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import ActivityEvent, Asset, World, new_id
from apps.api.storage import resolve_artifact, store_bytes
from apps.api.worldstore_service import get_current_worldir
from engine.assets.extraction import extract_entity_asset

log = logging.getLogger("reality.api.assets")

assets_router = APIRouter(tags=["assets"])


class ExtractAssetIn(BaseModel):
    entity_id: str
    name: Optional[str] = None
    category: Optional[str] = None


@assets_router.post("/api/worlds/{world_id}/extract-asset", status_code=201)
async def extract_world_asset(
    world_id: str,
    body: ExtractAssetIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Extract a single entity from a world into an independent reusable asset."""
    world = await db.get(World, world_id)
    if not world:
        raise HTTPException(404, f"World '{world_id}' not found")

    enforce_project_isolation(world.team_id, auth)

    world_ir = await get_current_worldir(db, world_id)
    if world_ir is None:
        raise HTTPException(404, f"World '{world_id}' has no committed versions")

    try:
        extracted, asset_world, glb_bytes = extract_entity_asset(
            world_ir,
            body.entity_id,
            name=body.name,
            category=body.category,
        )
    except KeyError:
        raise HTTPException(404, f"Entity '{body.entity_id}' not found in world '{world_id}'")
    except Exception as exc:
        log.exception("Asset extraction failed for %s in %s", body.entity_id, world_id)
        raise HTTPException(422, f"Failed to extract asset: {exc}")

    # Store binary GLB
    digest, path = store_bytes(glb_bytes)
    artifact_uri = f"sha256://{digest}"

    asset_row = Asset(
        id=extracted.id,
        name=extracted.name,
        category=extracted.category,
        source_world_id=world_id,
        source_entity_id=body.entity_id,
        dimensions=extracted.dimensions,
        provenance=extracted.provenance,
        artifact_uri=artifact_uri,
        metadata_=extracted.metadata,
        owner_id=auth.user_id,
        team_id=world.team_id or (auth.tenant_id if auth.tenant_id != "default" else None),
    )
    db.add(asset_row)
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="asset.extracted",
            entity_type="asset",
            entity_id=asset_row.id,
            summary=f"Asset '{asset_row.name}' extracted from world '{world.name}'",
        )
    )
    await db.commit()
    await db.refresh(asset_row)

    return {
        "id": asset_row.id,
        "name": asset_row.name,
        "category": asset_row.category,
        "source_world_id": asset_row.source_world_id,
        "source_entity_id": asset_row.source_entity_id,
        "dimensions": asset_row.dimensions,
        "provenance": asset_row.provenance,
        "download_url": f"/api/assets/{asset_row.id}/download",
        "created_at": asset_row.created_at.isoformat(),
    }


@assets_router.get("/api/assets")
async def list_assets(
    world_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List extracted assets."""
    query = select(Asset)
    if world_id:
        query = query.where(Asset.source_world_id == world_id)
    if category:
        query = query.where(Asset.category == category)
    if auth.tenant_id != "default":
        query = query.where(Asset.team_id == auth.tenant_id)
    query = query.order_by(Asset.created_at.desc()).offset(offset).limit(limit)
    res = await db.execute(query)
    rows = res.scalars().all()
    return {
        "items": [
            {
                "id": r.id,
                "name": r.name,
                "category": r.category,
                "source_world_id": r.source_world_id,
                "source_entity_id": r.source_entity_id,
                "dimensions": r.dimensions,
                "provenance": r.provenance,
                "download_url": f"/api/assets/{r.id}/download",
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
        "count": len(rows),
    }


@assets_router.get("/api/assets/{asset_id}")
async def get_asset(
    asset_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Get metadata for an extracted asset."""
    r = await db.get(Asset, asset_id)
    if not r:
        raise HTTPException(404, f"Asset '{asset_id}' not found")
    if r.team_id and auth.tenant_id != "default" and r.team_id != auth.tenant_id:
        raise HTTPException(403, "Access denied")
    return {
        "id": r.id,
        "name": r.name,
        "category": r.category,
        "source_world_id": r.source_world_id,
        "source_entity_id": r.source_entity_id,
        "dimensions": r.dimensions,
        "provenance": r.provenance,
        "metadata": r.metadata_,
        "download_url": f"/api/assets/{r.id}/download",
        "created_at": r.created_at.isoformat(),
    }


@assets_router.get("/api/assets/{asset_id}/download")
async def download_asset(
    asset_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Download the extracted asset as a binary GLB container."""
    r = await db.get(Asset, asset_id)
    if not r:
        raise HTTPException(404, f"Asset '{asset_id}' not found")
    if r.team_id and auth.tenant_id != "default" and r.team_id != auth.tenant_id:
        raise HTTPException(403, "Access denied")
    if not r.artifact_uri:
        raise HTTPException(404, "Asset binary not found")
    path = resolve_artifact(r.artifact_uri)
    if not path or not path.exists():
        raise HTTPException(404, "Asset artifact file not found on server")
    data = path.read_bytes()
    clean_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in r.name) or "asset"
    return Response(
        content=data,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{clean_name}.glb"'},
    )
