"""Tests for Task T-7.2: Pricing, Subscriptions, and Stripe Billing.

Verifies:
1. GET /api/billing/tiers returns architectural survey plans (Free, Starter $79, Studio Pro $249, Enterprise).
2. GET /api/billing/status returns caller tenant's subscription state and monthly room scan quota.
3. POST /api/billing/checkout creates a valid Stripe checkout session with redirect URL.
4. POST /api/billing/portal creates a customer portal session for subscription management.
5. POST /api/billing/webhook verifies HMAC-SHA256 signature and processes checkout.session.completed,
   customer.subscription.updated, and customer.subscription.deleted.
6. Monthly scan quota enforcement blocks excess room creations with 402 Payment Required on Free/Starter,
   while permitting unlimited reconstructions on Studio Pro.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from apps.api.billing_service import (
    TIERS,
    check_scan_quota,
    get_billing_status,
    get_or_create_subscription,
    verify_stripe_signature,
)
from apps.api.db import get_sessionmaker
from apps.api.main import app
from apps.api.models import Tenant, TenantSubscription, UserAccount, World, new_id, utcnow


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    db_file = tmp_path / "billing_test.db"
    artifacts_dir = tmp_path / "artifacts"
    worldstore_dir = tmp_path / "ws"

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(artifacts_dir))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(worldstore_dir))
    monkeypatch.setenv("REALITY_API_TOKEN", "test_billing_secret_token")

    with TestClient(app) as c:
        yield c


def _auth_header(token: str = "test_billing_secret_token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_pricing_tiers_endpoint(client: TestClient) -> None:
    """GET /api/billing/tiers returns available plans, pricing, and architectural survey capabilities."""
    r = client.get("/api/billing/tiers")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["currency"] == "USD"
    assert data["billing_cadence"] == "monthly"

    tiers_by_id = {t["tier_id"]: t for t in data["tiers"]}
    assert "free" in tiers_by_id
    assert "starter" in tiers_by_id
    assert "pro" in tiers_by_id
    assert "enterprise" in tiers_by_id

    # Starter plan: Solo Architect / Designer at $79/mo, 10 scans
    starter = tiers_by_id["starter"]
    assert starter["price_usd_monthly"] == 79
    assert starter["scanned_rooms_per_month"] == 10
    assert "pdf_floorplan" in starter["export_formats"]
    assert "pdf_schedule" in starter["export_formats"]

    # Pro plan: Studio Pro at $249/mo, unlimited scans, IFC and DXF
    pro = tiers_by_id["pro"]
    assert pro["price_usd_monthly"] == 249
    assert pro["scanned_rooms_per_month"] == -1  # unlimited
    assert "ifc" in pro["export_formats"]
    assert "dxf" in pro["export_formats"]


def test_default_tenant_billing_status(client: TestClient) -> None:
    """GET /api/billing/status initializes new tenant on the Free tier with active status."""
    r = client.get("/api/billing/status", headers=_auth_header())
    assert r.status_code == 200, r.text
    status_data = r.json()

    assert status_data["tier"] == "free"
    assert status_data["is_active"] is True
    assert status_data["status"] == "active"
    assert status_data["quota"]["scanned_rooms_limit"] == 1
    assert status_data["quota"]["scanned_rooms_used"] == 0
    assert status_data["quota"]["quota_exceeded"] is False


def test_checkout_session_creation(client: TestClient) -> None:
    """POST /api/billing/checkout generates a valid checkout session URL for upgrading."""
    payload = {
        "tier": "starter",
        "success_url": "https://app.realityengine.io/settings/billing?session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": "https://app.realityengine.io/settings/billing?cancelled=true",
    }
    r = client.post("/api/billing/checkout", json=payload, headers=_auth_header())
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["checkout_url"]
    assert res["session_id"]
    assert res["tier"] == "starter"


def test_checkout_invalid_tier_rejected(client: TestClient) -> None:
    """POST /api/billing/checkout rejects unsupported tier names with 422 Unprocessable Entity."""
    payload = {
        "tier": "mega_tier_invalid",
        "success_url": "https://app.realityengine.io/settings/billing",
        "cancel_url": "https://app.realityengine.io/settings/billing",
    }
    r = client.post("/api/billing/checkout", json=payload, headers=_auth_header())
    assert r.status_code == 422


def test_customer_portal_session(client: TestClient) -> None:
    """POST /api/billing/portal returns a valid billing portal management URL."""
    payload = {"return_url": "https://app.realityengine.io/settings/billing"}
    r = client.post("/api/billing/portal", json=payload, headers=_auth_header())
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["portal_url"]


def test_stripe_webhook_signature_verification() -> None:
    """verify_stripe_signature validates RFC-compliant Stripe HMAC-SHA256 signatures."""
    payload = b'{"id": "evt_test_123", "type": "checkout.session.completed"}'
    secret = "whsec_test_secret_key_12345"
    now_ts = int(time.time())

    # Build valid signature header: t=...,v1=...
    signed_payload = f"{now_ts}.".encode("utf-8") + payload
    v1_sig = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    valid_header = f"t={now_ts},v1={v1_sig}"

    # Verify matching signature
    assert verify_stripe_signature(payload, valid_header, secret) is True

    # Bad secret
    assert verify_stripe_signature(payload, valid_header, "whsec_wrong_secret") is False

    # Tampered payload
    tampered = payload + b" "
    assert verify_stripe_signature(tampered, valid_header, secret) is False

    # Expired timestamp (older than tolerance)
    old_ts = now_ts - 500
    old_signed = f"{old_ts}.".encode("utf-8") + payload
    old_sig = hmac.new(secret.encode("utf-8"), old_signed, hashlib.sha256).hexdigest()
    expired_header = f"t={old_ts},v1={old_sig}"
    assert verify_stripe_signature(payload, expired_header, secret, tolerance_s=300) is False


def test_stripe_webhook_checkout_completed_and_lifecycle(client: TestClient, monkeypatch) -> None:
    """POST /api/billing/webhook processes checkout.session.completed, subscription.updated, and deleted."""
    webhook_secret = "whsec_mock_stripe_key"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", webhook_secret)

    tenant_id = "ten_arch_studio_01"

    # Step 1: Simulated Stripe webhook: checkout.session.completed upgrades tenant to 'pro'
    event_payload = {
        "id": "evt_001",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_001",
                "customer": "cus_stripe_123",
                "subscription": "sub_stripe_pro_456",
                "client_reference_id": f"{tenant_id}:pro",
                "metadata": {"tenant_id": tenant_id, "target_tier": "pro"},
            }
        },
    }
    raw_body = json.dumps(event_payload).encode("utf-8")
    now_ts = int(time.time())
    sig = hmac.new(webhook_secret.encode("utf-8"), f"{now_ts}.".encode("utf-8") + raw_body, hashlib.sha256).hexdigest()
    headers = {"Stripe-Signature": f"t={now_ts},v1={sig}", "Content-Type": "application/json"}

    r = client.post("/api/billing/webhook", content=raw_body, headers=headers)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "processed"
    assert res["tier"] == "pro"

    # Step 2: Simulated webhook: customer.subscription.updated updates status to past_due
    event_update = {
        "id": "evt_002",
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_stripe_pro_456",
                "customer": "cus_stripe_123",
                "status": "past_due",
                "cancel_at_period_end": True,
            }
        },
    }
    raw_update = json.dumps(event_update).encode("utf-8")
    sig_update = hmac.new(webhook_secret.encode("utf-8"), f"{now_ts}.".encode("utf-8") + raw_update, hashlib.sha256).hexdigest()
    headers_update = {"Stripe-Signature": f"t={now_ts},v1={sig_update}", "Content-Type": "application/json"}

    r_up = client.post("/api/billing/webhook", content=raw_update, headers=headers_update)
    assert r_up.status_code == 200, r_up.text
    assert r_up.json()["new_status"] == "past_due"

    # Step 3: Simulated webhook: customer.subscription.deleted reverts tier to free
    event_del = {
        "id": "evt_003",
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_stripe_pro_456",
                "customer": "cus_stripe_123",
            }
        },
    }
    raw_del = json.dumps(event_del).encode("utf-8")
    sig_del = hmac.new(webhook_secret.encode("utf-8"), f"{now_ts}.".encode("utf-8") + raw_del, hashlib.sha256).hexdigest()
    headers_del = {"Stripe-Signature": f"t={now_ts},v1={sig_del}", "Content-Type": "application/json"}

    r_del = client.post("/api/billing/webhook", content=raw_del, headers=headers_del)
    assert r_del.status_code == 200
    assert r_del.json()["reverted_to"] == "free"


@pytest.mark.asyncio
async def test_monthly_scan_quota_logic(tmp_path, monkeypatch) -> None:
    """Monthly scan quota permits scanning within limits and rejects when quota is reached."""
    db_file = tmp_path / "quota_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        # Create base tables
        from apps.api.models import Base
        async with db.bind.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        tenant = Tenant(id="ten_test_quota_01", name="Test Architecture Studio", slug="test-arch")
        db.add(tenant)
        await db.commit()

        # Step 1: Free tier allows 1 scan
        sub = await get_or_create_subscription(db, tenant.id)
        assert sub.tier == "free"

        has_quota, _ = await check_scan_quota(db, tenant.id)
        assert has_quota is True

        # Simulate 1 room scan created this month
        w1 = World(id=new_id("wld"), name="Room 1", team_id=tenant.id, status="active")
        db.add(w1)
        await db.commit()

        # Free tier limit (1) is reached -> check_scan_quota returns False with upgrade message
        has_quota, reason = await check_scan_quota(db, tenant.id)
        assert has_quota is False
        assert "quota exceeded" in reason.lower()
        assert "upgrade to studio pro" in reason.lower()

        # Step 2: Upgrade to Studio Pro -> unlimited scans permitted
        sub.tier = "pro"
        await db.commit()

        has_quota_pro, reason_pro = await check_scan_quota(db, tenant.id)
        assert has_quota_pro is True
        assert reason_pro is None
