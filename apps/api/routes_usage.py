"""GET /api/usage: the caller's own tenant usage (storage, compute, exports). An ADMIN may read any tenant's with ?tenant=."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, Role, get_auth_context
from apps.api.db import get_db
from apps.api.usage import tenant_usage

usage = APIRouter(prefix="/api/usage", tags=["usage"])


@usage.get("")
async def get_usage(tenant: str | None = None, since: datetime | None = None, db: AsyncSession = Depends(get_db),
                    auth: AuthContext = Depends(get_auth_context)) -> dict:
    target = tenant or auth.tenant_id
    if target != auth.tenant_id and auth.role.rank < Role.ADMIN.rank:
        raise HTTPException(403, "Only an administrator can read another tenant's usage")
    return await tenant_usage(db, target, since)
