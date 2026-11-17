"""World client share links router (T-4b.4 / RE-ARCH-000009).

Extracted from routes_worlds.py to provide dedicated resource routing for:
  - POST /{world_id}/shares: create client share link
  - GET /{world_id}/shares: list client share links
  - POST /{world_id}/shares/{share_id}/revoke: revoke client share link
  - GET /shared/access/{token}: access shared world via token
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import (
    AuthContext,
    Role,
    get_auth_context,
    issue_share_token,
    revoke_share_token_in_memory,
    verify_share_token,
)
from apps.api.db import get_db
from apps.api.models import World, WorldShareLink, new_id, utcnow
from apps.api.tenancy import enforce_project_isolation

shares_router = APIRouter()


class CreateShareIn(BaseModel):
    label: Optional[str] = None
    expires_in_hours: Optional[int] = None


def _is_share_link_active(link: WorldShareLink) -> bool:
    if link.revoked_at is not None:
        return False
    if link.expires_at is None:
        return True
    exp = link.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp > utcnow()


@shares_router.post("/{world_id}/shares")
async def create_world_share_link(
    world_id: str,
    body: CreateShareIn = CreateShareIn(),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Create a token-scoped, read-only client share link for this world (T-4b.4)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    if auth.role < Role.EDITOR and not auth.is_loopback:
        raise HTTPException(403, "Insufficient permissions to create share links")

    share_id = new_id("sh")
    expires_at = None
    exp_ts = None
    if body.expires_in_hours and body.expires_in_hours > 0:
        expires_at = utcnow() + timedelta(hours=body.expires_in_hours)
        exp_ts = int(expires_at.timestamp())

    token = issue_share_token(world_id=w.id, share_id=share_id, expires_at=exp_ts)
    link = WorldShareLink(
        id=share_id,
        world_id=w.id,
        token=token,
        label=body.label,
        created_by=auth.user_id,
        expires_at=expires_at,
    )
    db.add(link)
    await db.commit()

    return {
        "id": link.id,
        "world_id": w.id,
        "token": link.token,
        "share_url": f"/shared/{link.token}",
        "label": link.label,
        "expires_at": link.expires_at.isoformat() if link.expires_at else None,
        "created_at": link.created_at.isoformat() if link.created_at else None,
        "revoked": False,
    }


@shares_router.get("/{world_id}/shares")
async def list_world_share_links(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List all client share links for this world (T-4b.4)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldShareLink).where(WorldShareLink.world_id == world_id).order_by(WorldShareLink.created_at.desc())
    )
    links = res.scalars().all()
    return {
        "world_id": world_id,
        "items": [
            {
                "id": link.id,
                "world_id": link.world_id,
                "token": link.token,
                "share_url": f"/shared/{link.token}",
                "label": link.label,
                "created_at": link.created_at.isoformat() if link.created_at else None,
                "expires_at": link.expires_at.isoformat() if link.expires_at else None,
                "revoked_at": link.revoked_at.isoformat() if link.revoked_at else None,
                "is_active": _is_share_link_active(link),
            }
            for link in links
        ],
    }


@shares_router.post("/{world_id}/shares/{share_id}/revoke")
async def revoke_world_share_link(
    world_id: str,
    share_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Revoke an active client share link (T-4b.4)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    if auth.role < Role.EDITOR and not auth.is_loopback:
        raise HTTPException(403, "Insufficient permissions to revoke share links")

    res = await db.execute(
        select(WorldShareLink).where(WorldShareLink.id == share_id, WorldShareLink.world_id == world_id)
    )
    link = res.scalars().first()
    if link is None:
        raise HTTPException(404, "Share link not found")

    link.revoked_at = utcnow()
    revoke_share_token_in_memory(link.token)
    await db.commit()

    return {"status": "revoked", "id": share_id, "world_id": world_id}


@shares_router.get("/shared/access/{token}")
async def access_shared_world(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Access a shared world using a read-only token without needing an account (T-4b.4)."""
    auth_ctx = verify_share_token(token)
    if auth_ctx is None:
        raise HTTPException(401, "Invalid or expired share token")

    res = await db.execute(select(WorldShareLink).where(WorldShareLink.token == token.strip()))
    link = res.scalars().first()
    if link is None or link.revoked_at is not None:
        raise HTTPException(403, "Share link has been revoked or does not exist")
    if link.expires_at is not None:
        exp = link.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < utcnow():
            raise HTTPException(401, "Share link has expired")

    w = await db.get(World, link.world_id)
    if w is None or w.status != "active":
        raise HTTPException(404, "Shared world not found")

    from apps.api.worldstore_service import get_current_worldir
    worldir = await get_current_worldir(db, w.id)

    return {
        "world_id": w.id,
        "name": w.name,
        "description": w.description,
        "read_only": True,
        "share_id": link.id,
        "label": link.label,
        "current_version_id": w.current_version_id,
        "worldir": worldir.to_dict() if worldir else None,
    }
