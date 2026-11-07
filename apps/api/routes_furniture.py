"""Routes for CC0 furniture catalog (T-4b.1)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from engine.furniture.catalog import export_furniture_glb, get_catalog_item, list_catalog

log = logging.getLogger("reality.api.furniture")

furniture_router = APIRouter(prefix="/api/furniture", tags=["furniture"])


@furniture_router.get("/catalog")
async def get_furniture_catalog(category: str | None = None) -> list[dict[str, Any]]:
    """List built-in CC0 furniture models available for world editing and placement."""
    items = list_catalog(category=category)
    return [item.to_dict() for item in items]


@furniture_router.get("/catalog/{catalog_id}")
async def get_furniture_catalog_item(catalog_id: str) -> dict[str, Any]:
    """Retrieve details and CC0 license metadata for a specific furniture model."""
    item = get_catalog_item(catalog_id)
    if not item:
        raise HTTPException(404, f"Furniture catalog item '{catalog_id}' not found")
    return item.to_dict()


@furniture_router.get("/catalog/{catalog_id}/glb")
async def download_furniture_catalog_glb(catalog_id: str) -> Response:
    """Download standard binary glTF (GLB) for a catalog furniture model."""
    item = get_catalog_item(catalog_id)
    if not item:
        raise HTTPException(404, f"Furniture catalog item '{catalog_id}' not found")

    glb_bytes = export_furniture_glb(item)
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{catalog_id}.glb"',
            "X-License": item.license,
            "X-License-URL": item.license_url,
        },
    )
