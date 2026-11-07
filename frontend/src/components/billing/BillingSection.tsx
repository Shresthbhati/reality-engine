"use client";

import { useEffect, useState } from "react";
import { Check, CreditCard, Sparkles, AlertCircle, ArrowUpRight } from "lucide-react";

interface Tier {
  tier_id: string;
  name: string;
  price_usd_monthly: number | null;
  scanned_rooms_per_month: number;
  seats: number;
  features: string[];
  export_formats: string[];
}

interface BillingStatus {
  tenant_id: string;
  tier: string;
  tier_name: string;
  price_usd_monthly: number | null;
  status: string;
  is_active: boolean;
  stripe_customer_id: string | null;
  stripe_subscription_id: string | null;
  current_period_end: string | null;
  cancel_at_period_end: boolean;
  quota: {
    scanned_rooms_limit: number;
    scanned_rooms_used: number;
    scanned_rooms_remaining: number | null;
    is_unlimited: boolean;
    quota_exceeded: boolean;
  };
}

export default function BillingSection() {
  const [tiers, setTiers] = useState<Tier[]>([]);
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadBilling() {
      try {
        const [tiersRes, statusRes] = await Promise.all([
          fetch("/api/billing/tiers"),
          fetch("/api/billing/status"),
        ]);

        if (tiersRes.ok) {
          const tData = await tiersRes.json();
          setTiers(tData.tiers || []);
        }
        if (statusRes.ok) {
          const sData = await statusRes.json();
          setStatus(sData);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load billing information");
      } finally {
        setLoading(false);
      }
    }
    loadBilling();
  }, []);

  const handleCheckout = async (targetTier: string) => {
    setActionLoading(targetTier);
    setError(null);
    try {
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tier: targetTier,
          success_url: `${window.location.origin}/settings?checkout=success&tier=${targetTier}`,
          cancel_url: `${window.location.origin}/settings?checkout=cancelled`,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Checkout session initialization failed");
      }

      const data = await res.json();
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Checkout failed");
      setActionLoading(null);
    }
  };

  const handlePortal = async () => {
    setActionLoading("portal");
    setError(null);
    try {
      const res = await fetch("/api/billing/portal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          return_url: `${window.location.origin}/settings`,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Could not open billing portal");
      }

      const data = await res.json();
      if (data.portal_url) {
        window.location.href = data.portal_url;
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Portal redirect failed");
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <div className="p-4 text-xs" style={{ color: "var(--text-tertiary)" }}>
        Loading subscription and plan details...
      </div>
    );
  }

  const currentTier = status?.tier || "free";

  return (
    <div className="flex flex-col gap-6" data-testid="billing-section">
      {/* Current Subscription Card */}
      <div
        className="p-4 rounded-xl border flex flex-col gap-3"
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CreditCard className="w-4 h-4 text-[var(--accent)]" />
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Current Plan: {status?.tier_name || "Community / Free"}
            </h3>
          </div>
          <span
            className="text-xs px-2.5 py-0.5 rounded-full font-medium"
            style={{
              background: status?.is_active ? "var(--accent-subtle)" : "var(--bg-surface)",
              color: status?.is_active ? "var(--accent)" : "var(--text-secondary)",
              border: "1px solid var(--border)",
            }}
          >
            {status?.status?.toUpperCase() || "ACTIVE"}
          </span>
        </div>

        {/* Quota display */}
        <div className="flex flex-col gap-1.5 text-xs pt-1 border-t" style={{ borderColor: "var(--border)" }}>
          <div className="flex justify-between items-center">
            <span style={{ color: "var(--text-secondary)" }}>Monthly Scanned Rooms</span>
            <span className="font-medium" style={{ color: "var(--text-primary)" }}>
              {status?.quota.is_unlimited
                ? "Unlimited"
                : `${status?.quota.scanned_rooms_used || 0} / ${status?.quota.scanned_rooms_limit || 1} rooms`}
            </span>
          </div>
          {!status?.quota.is_unlimited && (
            <div className="w-full h-1.5 rounded-full overflow-hidden bg-[var(--bg-surface)]">
              <div
                className="h-full bg-[var(--accent)] transition-all"
                style={{
                  width: `${Math.min(
                    100,
                    (((status?.quota.scanned_rooms_used || 0) / (status?.quota.scanned_rooms_limit || 1)) * 100)
                  )}%`,
                }}
              />
            </div>
          )}
        </div>

        {status?.stripe_customer_id && (
          <div className="pt-2 flex justify-end">
            <button
              type="button"
              onClick={handlePortal}
              disabled={actionLoading === "portal"}
              className="text-xs px-3 py-1.5 rounded-md border flex items-center gap-1.5 font-medium transition-colors"
              style={{
                background: "var(--bg-surface)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            >
              Manage Invoices & Payment Methods
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {error && (
        <div
          className="p-3 rounded-lg border text-xs flex items-center gap-2"
          style={{
            background: "rgba(239, 68, 68, 0.1)",
            borderColor: "rgba(239, 68, 68, 0.3)",
            color: "var(--text-primary)",
          }}
        >
          <AlertCircle className="w-4 h-4 text-danger shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Plan Tiers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tiers
          .filter((t) => t.tier_id === "starter" || t.tier_id === "pro")
          .map((t) => {
            const isSelected = currentTier === t.tier_id;
            const isPro = t.tier_id === "pro";

            return (
              <div
                key={t.tier_id}
                className="p-5 rounded-xl border flex flex-col justify-between gap-4 relative"
                style={{
                  background: isPro ? "var(--bg-elevated)" : "var(--bg-surface)",
                  borderColor: isPro ? "var(--accent)" : "var(--border)",
                }}
              >
                {isPro && (
                  <span
                    className="absolute -top-2.5 right-4 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
                    style={{ background: "var(--accent)", color: "#ffffff" }}
                  >
                    Recommended for Studios
                  </span>
                )}

                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    {isPro && <Sparkles className="w-4 h-4 text-[var(--accent)]" />}
                    <h4 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                      {t.name}
                    </h4>
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
                      ${t.price_usd_monthly}
                    </span>
                    <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                      / month
                    </span>
                  </div>

                  <ul className="flex flex-col gap-2 pt-3 text-xs" style={{ color: "var(--text-secondary)" }}>
                    {t.features.map((feat) => (
                      <li key={feat} className="flex items-start gap-2">
                        <Check className="w-3.5 h-3.5 text-success shrink-0 mt-0.5" />
                        <span>{feat}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <button
                  type="button"
                  onClick={() => handleCheckout(t.tier_id)}
                  disabled={isSelected || actionLoading === t.tier_id}
                  className="w-full py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors disabled:opacity-50"
                  style={{
                    background: isSelected
                      ? "var(--bg-surface)"
                      : isPro
                      ? "var(--accent)"
                      : "var(--bg-elevated)",
                    color: isSelected
                      ? "var(--text-tertiary)"
                      : isPro
                      ? "#ffffff"
                      : "var(--text-primary)",
                    border: isSelected ? "1px solid var(--border)" : "none",
                  }}
                >
                  {isSelected
                    ? "Current Active Plan"
                    : actionLoading === t.tier_id
                    ? "Redirecting to Stripe..."
                    : `Subscribe to ${t.name}`}
                </button>
              </div>
            );
          })}
      </div>
    </div>
  );
}
