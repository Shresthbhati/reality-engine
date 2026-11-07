"""API routes for pricing, subscriptions, and Stripe billing (Task T-7.2).

Provides:
- GET /api/billing/tiers: Available pricing plans, monthly costs, and architectural survey feature sets.
- GET /api/billing/status: Current tenant subscription tier, quota usage, and payment state.
- POST /api/billing/checkout: Create Stripe Checkout Session to upgrade to Starter or Studio Pro.
- POST /api/billing/portal: Open Stripe Customer Portal for invoice history and payment methods.
- POST /api/billing/webhook: Receive and process Stripe webhook events (checkout.session.completed, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, Role, get_auth_context
from apps.api.billing_service import (
    TIERS,
    create_checkout_session,
    create_customer_portal_session,
    get_billing_status,
    process_stripe_webhook_event,
    verify_stripe_signature,
)
from apps.api.db import get_db

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    tier: str = Field(..., description="Target tier to subscribe to: 'starter', 'pro', or 'enterprise'")
    success_url: str = Field(..., description="URL to redirect user on successful checkout completion")
    cancel_url: str = Field(..., description="URL to redirect user on checkout cancellation")


class PortalRequest(BaseModel):
    return_url: str = Field(..., description="URL to return to after managing billing settings in Stripe portal")


@router.get("/tiers")
async def list_pricing_tiers() -> dict[str, Any]:
    """Retrieve all available subscription tiers and their architectural survey capabilities."""
    return {
        "tiers": list(TIERS.values()),
        "currency": "USD",
        "billing_cadence": "monthly",
    }


@router.get("/status")
async def get_current_tenant_billing_status(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Retrieve current tenant's subscription plan, active status, and monthly scanned room quota."""
    return await get_billing_status(db, auth.tenant_id)


@router.post("/checkout")
async def create_subscription_checkout(
    body: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Initiate a Stripe Checkout session to subscribe to or upgrade a paid plan."""
    if auth.role not in (Role.ADMIN, Role.EDITOR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only tenant administrators and editors can manage subscriptions and billing",
        )

    tier_key = body.tier.lower()
    if tier_key not in ("starter", "pro", "enterprise"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid tier '{body.tier}'. Available plans: starter, pro, enterprise",
        )

    try:
        res = await create_checkout_session(
            db=db,
            tenant_id=auth.tenant_id,
            target_tier=tier_key,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            user_email=getattr(auth, "email", None),
        )
        return res
    except Exception as exc:
        log.error("Failed to create checkout session: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/portal")
async def create_billing_portal_session(
    body: PortalRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Generate a Stripe Customer Portal link to manage active subscriptions, invoices, and payment methods."""
    try:
        return await create_customer_portal_session(
            db=db,
            tenant_id=auth.tenant_id,
            return_url=body.return_url,
        )
    except Exception as exc:
        log.error("Failed to create portal session: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/webhook")
async def handle_stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    stripe_signature: str | None = Header(None, alias="Stripe-Signature"),
) -> dict[str, Any]:
    """Stripe webhook endpoint for asynchronous billing and subscription lifecycle events.

    Verifies Stripe HMAC-SHA256 signature when STRIPE_WEBHOOK_SECRET is configured.
    """
    body_bytes = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    # In live mode or test mode with configured webhook secret, verify cryptographic signature
    if webhook_secret:
        if not stripe_signature:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Stripe-Signature header")
        is_valid = verify_stripe_signature(body_bytes, stripe_signature, webhook_secret)
        if not is_valid:
            log.warning("Invalid Stripe webhook signature rejected")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe webhook signature")

    try:
        event = json.loads(body_bytes.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON payload: {exc}") from exc

    return await process_stripe_webhook_event(db, event)
