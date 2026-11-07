"""Billing and subscription service with Stripe integration (Task T-7.2).

Provides:
- Tiered subscription management for Architecture & Interior Design practices:
    - free: 1 scanned room / month, standard web viewer, glTF/GLB export, 1 seat.
    - starter ($79/mo): 10 scanned rooms / month, PDF floor plans, room schedules, 1 seat.
    - pro ($249/mo): Unlimited rooms, full IFC4 CAD/BIM export, DXF floor plans,
      read-only client links, multi-session world adoptions, 5 seats.
    - enterprise (custom): Unlimited rooms & seats, custom BIM mapping, dedicated workers.
- Stripe Checkout Session creation (`POST /api/billing/checkout`).
- Stripe Customer Portal session creation (`POST /api/billing/portal`).
- Secure Stripe Webhook event processing (`POST /api/billing/webhook`) with HMAC-SHA256 signature verification.
- Monthly room scan quota enforcement.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Tenant, TenantSubscription, World, new_id, utcnow

log = logging.getLogger(__name__)

# Plan tier definitions aligned with Turnaround Plan Section 1
TIERS: dict[str, dict[str, Any]] = {
    "free": {
        "tier_id": "free",
        "name": "Community / Free",
        "price_usd_monthly": 0,
        "scanned_rooms_per_month": 1,
        "seats": 1,
        "features": [
            "1 active room project",
            "Standard Three.js web viewer",
            "glTF / GLB binary export",
            "Community support",
        ],
        "export_formats": ["glb", "gltf"],
        "stripe_price_id": None,
    },
    "starter": {
        "tier_id": "starter",
        "name": "Solo Architect / Designer",
        "price_usd_monthly": 79,
        "scanned_rooms_per_month": 10,
        "seats": 1,
        "features": [
            "10 measured room scans / month",
            "Dimensioned PDF floor plans",
            "Room schedule & takeoff PDF export",
            "glTF 2.0 / GLB binary export",
            "Standard accuracy reporting",
            "1 architect / designer seat",
        ],
        "export_formats": ["glb", "gltf", "pdf_floorplan", "pdf_schedule"],
        "stripe_price_id": os.getenv("STRIPE_PRICE_STARTER", "price_starter_monthly"),
    },
    "pro": {
        "tier_id": "pro",
        "name": "Studio Pro",
        "price_usd_monthly": 249,
        "scanned_rooms_per_month": -1,  # -1 represents unlimited
        "seats": 5,
        "features": [
            "Unlimited measured room scans",
            "Full IFC4 export (Revit & ArchiCAD native)",
            "Dimensioned 2D DXF floor plan export",
            "Read-only client presentation links with watermark",
            "Multi-session survey world adoptions",
            "Persistent edit transactions & undo/redo",
            "5 studio team seats",
            "Priority job queueing",
        ],
        "export_formats": ["glb", "gltf", "pdf_floorplan", "pdf_schedule", "dxf", "ifc"],
        "stripe_price_id": os.getenv("STRIPE_PRICE_PRO", "price_pro_monthly"),
    },
    "enterprise": {
        "tier_id": "enterprise",
        "name": "Enterprise Practice",
        "price_usd_monthly": None,
        "scanned_rooms_per_month": -1,
        "seats": -1,
        "features": [
            "Unlimited projects & team members",
            "Custom BIM parameter & classification mapping",
            "Dedicated cloud GPU worker instances",
            "Custom enterprise SSO & domain mapping",
            "Dedicated solutions engineer & 99.9% uptime SLA",
        ],
        "export_formats": ["glb", "gltf", "pdf_floorplan", "pdf_schedule", "dxf", "ifc", "cityjson", "usda"],
        "stripe_price_id": os.getenv("STRIPE_PRICE_ENTERPRISE", None),
    },
}


def get_tier_config(tier_name: str) -> dict[str, Any]:
    """Retrieve tier configuration, defaulting to 'free' if unrecognized."""
    return TIERS.get(tier_name.lower(), TIERS["free"])


async def get_or_create_subscription(db: AsyncSession, tenant_id: str) -> TenantSubscription:
    """Retrieve the subscription record for a tenant, initializing a 'free' record if missing."""
    stmt = select(TenantSubscription).where(TenantSubscription.tenant_id == tenant_id)
    sub = (await db.execute(stmt)).scalar_one_or_none()
    if sub is None:
        sub = TenantSubscription(
            id=new_id("sub"),
            tenant_id=tenant_id,
            tier="free",
            status="active",
            current_period_end=None,
            cancel_at_period_end=False,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        db.add(sub)
        await db.commit()
        await db.refresh(sub)
    return sub


async def count_tenant_scans_current_period(db: AsyncSession, tenant_id: str) -> int:
    """Count how many room scans/worlds the tenant has created in the current monthly period."""
    start_of_month = utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    stmt = (
        select(func.count(World.id))
        .where(World.team_id == tenant_id)
        .where(World.created_at >= start_of_month)
        .where(World.status != "trashed")
    )
    return (await db.execute(stmt)).scalar() or 0


async def get_billing_status(db: AsyncSession, tenant_id: str) -> dict[str, Any]:
    """Get the full billing status, tier capabilities, and quota usage for a tenant."""
    sub = await get_or_create_subscription(db, tenant_id)
    tier_info = get_tier_config(sub.tier)
    scans_used = await count_tenant_scans_current_period(db, tenant_id)
    scan_limit = tier_info["scanned_rooms_per_month"]

    return {
        "tenant_id": tenant_id,
        "subscription_id": sub.id,
        "tier": sub.tier,
        "tier_name": tier_info["name"],
        "price_usd_monthly": tier_info["price_usd_monthly"],
        "status": sub.status,
        "is_active": sub.status in ("active", "trialing"),
        "stripe_customer_id": sub.stripe_customer_id,
        "stripe_subscription_id": sub.stripe_subscription_id,
        "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
        "cancel_at_period_end": sub.cancel_at_period_end,
        "quota": {
            "scanned_rooms_limit": scan_limit,
            "scanned_rooms_used": scans_used,
            "scanned_rooms_remaining": max(0, scan_limit - scans_used) if scan_limit >= 0 else None,
            "is_unlimited": scan_limit < 0,
            "quota_exceeded": scan_limit >= 0 and scans_used >= scan_limit,
        },
        "features": tier_info["features"],
        "allowed_export_formats": tier_info["export_formats"],
    }


async def check_scan_quota(db: AsyncSession, tenant_id: str) -> tuple[bool, str | None]:
    """Check if the tenant has quota remaining to start a new reconstruction/scan."""
    status = await get_billing_status(db, tenant_id)
    quota = status["quota"]
    if quota["is_unlimited"]:
        return True, None
    if quota["quota_exceeded"]:
        limit = quota["scanned_rooms_limit"]
        used = quota["scanned_rooms_used"]
        tier = status["tier_name"]
        return False, f"Monthly scan quota exceeded ({used}/{limit} rooms on {tier}). Upgrade to Studio Pro for unlimited scans."
    return True, None


async def create_checkout_session(
    db: AsyncSession,
    tenant_id: str,
    target_tier: str,
    success_url: str,
    cancel_url: str,
    user_email: str | None = None,
) -> dict[str, Any]:
    """Create a checkout session for upgrading or subscribing to a paid tier.

    Uses Stripe's hosted Checkout Session API if STRIPE_SECRET_KEY is configured.
    In local/test environments without live Stripe credentials, generates a deterministic,
    mock-free simulation URL that redirects back to success_url with simulated checkout params.
    """
    if target_tier not in ("starter", "pro", "enterprise"):
        raise ValueError(f"Invalid tier '{target_tier}'. Available paid tiers: starter, pro, enterprise")

    sub = await get_or_create_subscription(db, tenant_id)
    tier_cfg = get_tier_config(target_tier)
    stripe_key = os.getenv("STRIPE_SECRET_KEY")

    # In live mode with Stripe API key configured:
    if stripe_key and not stripe_key.startswith("mock_"):
        import httpx

        headers = {
            "Authorization": f"Bearer {stripe_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "mode": "subscription",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "client_reference_id": f"{tenant_id}:{target_tier}",
            "metadata[tenant_id]": tenant_id,
            "metadata[target_tier]": target_tier,
            "line_items[0][price]": tier_cfg["stripe_price_id"],
            "line_items[0][quantity]": "1",
        }
        if sub.stripe_customer_id:
            data["customer"] = sub.stripe_customer_id
        elif user_email:
            data["customer_email"] = user_email

        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.stripe.com/v1/checkout/sessions", headers=headers, data=data)
            if resp.status_code not in (200, 201):
                log.error("Stripe API error: %s", resp.text)
                raise RuntimeError(f"Stripe API error: {resp.text}")
            session_data = resp.json()
            return {
                "checkout_url": session_data["url"],
                "session_id": session_data["id"],
                "tier": target_tier,
                "mode": "live",
            }

    # Deterministic test / dev mode when STRIPE_SECRET_KEY is not configured:
    session_id = f"cs_test_{new_id('sess')}"
    # Format a valid redirect URL with session token
    sep = "&" if "?" in success_url else "?"
    simulated_url = f"{success_url}{sep}session_id={session_id}&tier={target_tier}&simulated=true"

    log.info("STRIPE_SECRET_KEY not set. Generated simulated checkout URL for tenant %s tier %s", tenant_id, target_tier)
    return {
        "checkout_url": simulated_url,
        "session_id": session_id,
        "tier": target_tier,
        "mode": "test_simulated",
    }


async def create_customer_portal_session(
    db: AsyncSession,
    tenant_id: str,
    return_url: str,
) -> dict[str, Any]:
    """Create a Stripe Customer Portal session for managing payments and subscriptions."""
    sub = await get_or_create_subscription(db, tenant_id)
    stripe_key = os.getenv("STRIPE_SECRET_KEY")

    if stripe_key and sub.stripe_customer_id and not stripe_key.startswith("mock_"):
        import httpx

        headers = {
            "Authorization": f"Bearer {stripe_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "customer": sub.stripe_customer_id,
            "return_url": return_url,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.stripe.com/v1/billing_portal/sessions", headers=headers, data=data)
            if resp.status_code in (200, 201):
                return {"portal_url": resp.json()["url"], "mode": "live"}

    # Fallback / simulated portal URL
    return {
        "portal_url": f"{return_url}?portal_session=active&tenant_id={tenant_id}",
        "mode": "test_simulated",
    }


def verify_stripe_signature(payload: bytes, signature_header: str, secret: str, tolerance_s: int = 300) -> bool:
    """Verify standard Stripe Webhook HMAC-SHA256 signature (rfc-compliant).

    Header format: t=1614552224,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
    """
    if not signature_header or not secret:
        return False

    items = dict(item.split("=", 1) for item in signature_header.split(",") if "=" in item)
    t = items.get("t")
    v1 = items.get("v1")

    if not t or not v1:
        return False

    # Check timestamp tolerance to prevent replay attacks
    try:
        t_int = int(t)
        if abs(time.time() - t_int) > tolerance_s:
            log.warning("Stripe webhook timestamp %s outside tolerance %ss", t, tolerance_s)
            return False
    except ValueError:
        return False

    # Signed payload is: f"{timestamp}.{raw_payload}"
    signed_payload = f"{t}.".encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, v1)


async def process_stripe_webhook_event(db: AsyncSession, event: dict[str, Any]) -> dict[str, Any]:
    """Process verified Stripe webhook events and update tenant subscription states.

    Supported events:
    - checkout.session.completed: upgrades tenant to selected tier.
    - customer.subscription.updated: updates status, period end, cancellation flags.
    - customer.subscription.deleted: reverts tenant back to free tier.
    """
    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    log.info("Processing Stripe webhook event: %s (id: %s)", event_type, event.get("id"))

    if event_type == "checkout.session.completed":
        metadata = data_object.get("metadata", {})
        client_ref = data_object.get("client_reference_id", "")
        tenant_id = metadata.get("tenant_id")
        target_tier = metadata.get("target_tier")

        if not tenant_id and ":" in client_ref:
            tenant_id, target_tier = client_ref.split(":", 1)

        if tenant_id and target_tier:
            sub = await get_or_create_subscription(db, tenant_id)
            sub.tier = target_tier
            sub.status = "active"
            if data_object.get("customer"):
                sub.stripe_customer_id = str(data_object["customer"])
            if data_object.get("subscription"):
                sub.stripe_subscription_id = str(data_object["subscription"])
            sub.updated_at = utcnow()
            await db.commit()
            await db.refresh(sub)
            return {"status": "processed", "event": event_type, "tenant_id": tenant_id, "tier": target_tier}

    elif event_type == "customer.subscription.updated":
        sub_id = data_object.get("id")
        cust_id = data_object.get("customer")
        status = data_object.get("status", "active")
        cancel_at = data_object.get("cancel_at_period_end", False)
        current_period_end = data_object.get("current_period_end")

        stmt = select(TenantSubscription).where(
            (TenantSubscription.stripe_subscription_id == sub_id)
            | (TenantSubscription.stripe_customer_id == cust_id)
        )
        sub = (await db.execute(stmt)).scalar_one_or_none()
        if sub:
            sub.status = status
            sub.cancel_at_period_end = bool(cancel_at)
            if current_period_end:
                sub.current_period_end = datetime.fromtimestamp(current_period_end, tz=timezone.utc)
            sub.updated_at = utcnow()
            await db.commit()
            return {"status": "processed", "event": event_type, "subscription_id": sub.id, "new_status": status}

    elif event_type == "customer.subscription.deleted":
        sub_id = data_object.get("id")
        cust_id = data_object.get("customer")

        stmt = select(TenantSubscription).where(
            (TenantSubscription.stripe_subscription_id == sub_id)
            | (TenantSubscription.stripe_customer_id == cust_id)
        )
        sub = (await db.execute(stmt)).scalar_one_or_none()
        if sub:
            sub.tier = "free"
            sub.status = "canceled"
            sub.stripe_subscription_id = None
            sub.updated_at = utcnow()
            await db.commit()
            return {"status": "processed", "event": event_type, "subscription_id": sub.id, "reverted_to": "free"}

    return {"status": "ignored", "event": event_type}
