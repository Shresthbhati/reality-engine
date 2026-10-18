"use client";

import Link from "next/link";
import { Loader2, ShieldCheck } from "lucide-react";
import { apiGet, isApiError, listEvidence, useApi } from "@/lib/api";
import { parseEvidenceQuality, resultLabel } from "@/lib/mobile/capture-result";
import { MobileEmptyState, MobileErrorState } from "./AsyncState";

const STATE_COLOR: Record<string, string> = {
  PROCESSED: "var(--success)",
  PROCESSING: "var(--accent)",
  UPLOADING: "var(--accent)",
  FAILED: "var(--error)",
};

function failureText(err: Error | null, what: string): string {
  return isApiError(err) ? err.describe() : err?.message ?? `Unexpected error loading ${what}.`;
}

/** Evidence list with the engine's own quality result per item (GET /api/evidence/quality), when the API reports one. */
export default function EvidenceQualityList({ sessionId }: { sessionId?: string | null }) {
  const evidence = useApi(() => listEvidence({ sessionId: sessionId || undefined }).then((r) => r.rows), [sessionId]);
  // Quality is an enrichment: if it is unavailable the list still renders, with the reason stated.
  const quality = useApi(async () => parseEvidenceQuality(await apiGet<unknown>("/api/evidence/quality")), []);

  if (evidence.isLoading) {
    return (
      <p role="status" className="flex items-center gap-2 py-10 text-sm" style={{ color: "var(--text-secondary)" }}>
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Loading evidence…
      </p>
    );
  }
  if (evidence.error) return <MobileErrorState message={failureText(evidence.error, "Evidence")} onRetry={evidence.refetch} />;

  const rows = evidence.data ?? [];
  if (rows.length === 0) return <MobileEmptyState icon={ShieldCheck} message="No Evidence has been added yet." />;

  const scores = quality.data;
  return (
    <div className="flex flex-col gap-2">
      {quality.error && (
        <p role="status" className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Quality results are not available right now ({failureText(quality.error, "quality")}). The list below is still current.
        </p>
      )}
      {rows.map((e) => {
        const q = scores?.get(e.id);
        return (
          <Link
            key={e.id}
            href={`/m/evidence/${e.id}`}
            className="flex min-h-[64px] items-center gap-3 rounded-xl px-4 py-2.5"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
          >
            <div className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="truncate text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {e.name}
              </span>
              {(e.location || e.sessionName) && (
                <span className="truncate text-xs" style={{ color: "var(--text-secondary)" }}>
                  {[e.location, e.sessionName].filter(Boolean).join(" · ")}
                </span>
              )}
              <span className="font-mono-num text-xs" style={{ color: "var(--text-tertiary)" }}>
                {e.capturedAt ? `Captured · ${e.capturedAt}` : "Capture time not recorded"}
              </span>
            </div>
            <div className="flex shrink-0 flex-col items-end gap-1">
              <span className="text-[11px] font-semibold" style={{ color: STATE_COLOR[e.processingStateRaw ?? e.processingState] ?? "var(--text-secondary)" }}>
                {resultLabel(e.processingStateRaw ?? e.processingState)}
              </span>
              {q && (
                <span
                  className="font-mono-num rounded-full px-2 py-0.5 text-[11px]"
                  style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
                  data-testid="quality-chip"
                >
                  {resultLabel(q.result)}
                  {q.score !== null ? ` · ${q.score}` : ""}
                </span>
              )}
            </div>
          </Link>
        );
      })}
    </div>
  );
}
