"use client";

import Link from "next/link";
import { useState } from "react";
import { AlertTriangle, CheckCircle2, CircleDashed, Lightbulb, Loader2, type LucideIcon } from "lucide-react";
import { cancelReconstruction, isApiError, modelStateCopy, retryReconstruction, type QualityLevel, type WorldStatus } from "@/lib/api";
import { worldHeadline, type Tone } from "@/lib/mobile/capture-result";

export const TONE_COLOR: Record<Tone, string> = {
  good: "var(--success)",
  warn: "var(--warning)",
  bad: "var(--error)",
  working: "var(--accent)",
  neutral: "var(--text-secondary)",
};

const SEGMENTS: Record<QualityLevel, number> = { none: 0, low: 1, moderate: 2, good: 3, unknown: 0 };

function ToneIcon({ tone }: { tone: Tone }) {
  const Icon: LucideIcon = tone === "good" ? CheckCircle2 : tone === "bad" || tone === "warn" ? AlertTriangle : tone === "working" ? Loader2 : CircleDashed;
  return <Icon className={`h-5 w-5 shrink-0 ${tone === "working" ? "animate-spin" : ""}`} style={{ color: TONE_COLOR[tone] }} aria-hidden />;
}

export interface WorldStatusCardProps {
  worldId: string;
  status: WorldStatus | null;
  error: Error | null;
  refresh: () => void;
}

/**
 * The world's REAL reconstruction state. Headline/detail come from
 * `worldHeadline` (backend fields only); the quality rows are the backend's own
 * `evidence_summary.quality` levels, shown as levels, never as percentages.
 */
export default function WorldStatusCard({ worldId, status, error, refresh }: WorldStatusCardProps) {
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setActionError(null);
    try {
      await fn();
      refresh();
    } catch (err) {
      setActionError(isApiError(err) ? err.describe() : "The request failed.");
    } finally {
      setBusy(false);
    }
  };

  const cardStyle = { background: "var(--bg-surface)", border: "1px solid var(--border)" };

  if (!status) {
    return (
      <section aria-label="Reconstruction status" className="rounded-xl p-4" style={cardStyle}>
        {error ? (
          <div role="alert" className="flex flex-col items-start gap-1">
            <p className="flex items-center gap-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              <AlertTriangle className="h-4 w-4" style={{ color: "var(--warning)" }} aria-hidden />
              Status unavailable
            </p>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {error.message} Your uploaded photos are kept.
            </p>
            <button
              type="button"
              onClick={refresh}
              className="mt-1 h-11 rounded-lg px-4 text-sm font-semibold"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              Try again
            </button>
          </div>
        ) : (
          <p role="status" className="flex items-center gap-2 text-sm" style={{ color: "var(--text-secondary)" }}>
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Reading reconstruction status…
          </p>
        )}
      </section>
    );
  }

  const h = worldHeadline(status, modelStateCopy);
  const failed = !status.in_progress && status.job && (status.state === "FAILED" || status.state === "CANCELLED");
  const quality = status.evidence_summary.quality;

  return (
    <section aria-label="Reconstruction status" className="flex flex-col gap-3 rounded-xl p-4" style={cardStyle} data-testid="world-status-card">
      <div className="flex items-start gap-3">
        <ToneIcon tone={h.tone} />
        <div className="min-w-0 flex-1" role="status">
          <p className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            {h.headline}
          </p>
          {h.detail && (
            <p className="mt-0.5 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              {h.detail}
            </p>
          )}
          {h.placed && (
            <p className="font-mono-num mt-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
              {h.placed}
            </p>
          )}
        </div>
      </div>

      {quality.length > 0 && (
        <ul className="flex flex-col gap-1.5" aria-label="Evidence quality as measured by the engine">
          {quality.map((q) => (
            <li key={q.name} title={q.basis} className="flex items-center gap-2 text-xs">
              <span className="w-28 shrink-0" style={{ color: "var(--text-secondary)" }}>
                {q.name}
              </span>
              <span className="flex gap-1" role="img" aria-label={`${q.name}: ${q.level === "unknown" ? "not measured" : q.level}`}>
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-2 w-7 rounded-sm"
                    style={{ background: i < SEGMENTS[q.level] ? "var(--accent)" : "var(--border)" }}
                  />
                ))}
              </span>
              <span style={{ color: "var(--text-tertiary)" }}>{q.level === "unknown" ? "not measured" : q.level}</span>
            </li>
          ))}
        </ul>
      )}

      {h.guidance && (
        <div className="rounded-lg p-3" style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)" }}>
          <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
            <Lightbulb className="h-3.5 w-3.5" style={{ color: "var(--warning)" }} aria-hidden /> What would help most
          </p>
          <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            {h.guidance}
          </p>
        </div>
      )}

      {actionError && (
        <p role="alert" className="text-xs" style={{ color: "var(--error)" }}>
          {actionError}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {status.has_model && (
          <Link
            href={`/m/worlds/${worldId}`}
            className="flex h-11 min-w-[44px] flex-1 items-center justify-center rounded-lg text-sm font-medium"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
          >
            View model
          </Link>
        )}
        {failed && status.job && (
          <button
            type="button"
            disabled={busy}
            onClick={() => act(() => retryReconstruction(status.job!.session_id))}
            className="h-11 min-w-[44px] flex-1 rounded-lg text-sm font-semibold disabled:opacity-60"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            Rebuild again
          </button>
        )}
        {status.in_progress && status.job && (
          <button
            type="button"
            disabled={busy}
            onClick={() => act(() => cancelReconstruction(status.job!.id))}
            className="h-11 min-w-[44px] flex-1 rounded-lg text-sm font-medium disabled:opacity-60"
            style={{ background: "var(--error-subtle)", color: "var(--error)", border: "1px solid var(--error)" }}
          >
            Cancel build
          </button>
        )}
      </div>
    </section>
  );
}
