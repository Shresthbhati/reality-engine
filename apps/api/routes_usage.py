"""Routes for tenant usage metering (T-6.5).

Provides:
- GET /api/usage: caller tenant usage (with optional tenant_id override for admin)
- GET /api/tenants/{tenant_id}/usage: per-tenant usage endpoint (isolated, 404 for cross-tenant access)
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, Role, get_auth_context
from apps.api.db import get_db
from apps.api.usage_service import get_tenant_usage

router = APIRouter(tags=["usage"])


@router.get("/api/usage")
async def get_current_tenant_usage(
    tenant_id: str | None = Query(None, description="Optional tenant override (admin only)"),
    since: datetime | None = Query(None, description="Filter usage since UTC timestamp"),
    until: datetime | None = Query(None, description="Filter usage until UTC timestamp"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve usage metrics for the caller's tenant, or for a specified tenant if caller is ADMIN."""
    target_tenant = tenant_id or auth.tenant_id
    if target_tenant != auth.tenant_id and auth.role != Role.ADMIN:
        # Prevent tenant enumeration across tenants
        raise HTTPException(status_code=404, detail="Tenant not found")

    return await get_tenant_usage(db, target_tenant, since=since, until=until)


@router.get("/api/tenants/{tenant_id}/usage")
async def get_specific_tenant_usage(
    tenant_id: str,
    since: datetime | None = Query(None, description="Filter usage since UTC timestamp"),
    until: datetime | None = Query(None, description="Filter usage until UTC timestamp"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Per-tenant usage metering endpoint (T-6.5).

    Returns GPU-minutes, compute time, storage bytes, and export counts.
    Strict tenant boundary enforced: requests for other tenants fail with 404.
    """
    if tenant_id != auth.tenant_id and auth.role != Role.ADMIN:
        # Prevent tenant enumeration across tenants
        raise HTTPException(status_code=404, detail="Tenant not found")

    return await get_tenant_usage(db, tenant_id, since=since, until=until)
