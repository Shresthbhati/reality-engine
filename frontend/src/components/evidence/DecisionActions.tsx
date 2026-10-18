"use client";

import { useState } from "react";
import { Ban, Check, Eye, ShieldAlert } from "lucide-react";
import { isApiError } from "@/lib/api/client";
import { putEvidenceDecision } from "@/lib/api/evidence-quality";
import type { QualityDecision } from "@/lib/api/evidence-quality-model";

const ACTIONS: Array<{ decision: QualityDecision; label: string; icon: typeof Check; color: string }> = [
  { decision: "accepted", label: "Accept", icon: Check, color: "var(--success)" },
  { decision: "accepted_with_warning", label: "Accept with warning", icon: ShieldAlert, color: "var(--warning)" },
  { decision: "needs_review", label: "Mark for review", icon: Eye, color: "var(--purple)" },
  { decision: "rejected", label: "Reject", icon: Ban, color: "var(--error)" },
];

/** Saves the operator's decision (PUT /api/evidence/{id}/decision) and reports the saved record upward. */
export default function DecisionActions({
  evidenceId,
  current,
  initialReason,
  onSaved,
}: {
  evidenceId: string;
  current: QualityDecision | null;
  initialReason: string | null;
  onSaved: (d: { decision: QualityDecision; reason: string | null; decided_at: string | null }) => void;
}) {
  const [reason, setReason] = useState(initialReason ?? "");
  const [busy, setBusy] = useState<QualityDecision | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function save(decision: QualityDecision) {
    setBusy(decision);
    setError(null);
    try {
      const saved = await putEvidenceDecision(evidenceId, decision, reason.trim());
      onSaved({ decision: saved.decision, reason: saved.reason, decided_at: saved.decided_at });
    } catch (e) {
      setError(isApiError(e) ? e.describe() : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <input
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        maxLength={2000}
        placeholder="Reason (optional)"
        aria-label="Decision reason"
        className="h-8 px-2.5 rounded-md text-sm outline-none"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
      />
      <div className="grid grid-cols-2 gap-2">
        {ACTIONS.map(({ decision, label, icon: Icon, color }) => (
          <button
            key={decision}
            type="button"
            disabled={busy !== null}
            onClick={() => save(decision)}
            aria-pressed={current === decision}
            className="inline-flex items-center justify-center gap-1.5 h-9 rounded-md text-xs font-medium disabled:opacity-50"
            style={{
              color,
              background: current === decision ? "var(--bg-hover)" : "transparent",
              border: `1px solid ${color}`,
              outline: current === decision ? `1px solid ${color}` : undefined,
            }}
          >
            <Icon className="w-3.5 h-3.5" />
            {busy === decision ? "Saving..." : label}
          </button>
        ))}
      </div>
      {error && (
        <p className="text-xs" style={{ color: "var(--error)" }} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
