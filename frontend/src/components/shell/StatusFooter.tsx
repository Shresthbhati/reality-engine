"use client";

import { useEffect, useState } from "react";
import { getHealth, getStatus } from "@/lib/api";
import { connectionState, evidenceLabel } from "@/lib/shell/shell-logic";
import pkg from "../../../package.json";

const POLL_MS = 30_000;

const DOT_COLOR = {
  online: "var(--success)",
  offline: "var(--error)",
  checking: "var(--text-tertiary)",
} as const;
const DOT_LABEL = { online: "Online", offline: "Offline", checking: "Checking…" } as const;

/** Bottom status footer. Everything on the right comes from the live API; nothing is hard-coded. */
export default function StatusFooter({ worldName }: { worldName: string | null }) {
  const [health, setHealth] = useState<{ ok: boolean } | null>(null);
  const [evidenceCount, setEvidenceCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      getHealth().then(
        () => !cancelled && setHealth({ ok: true }),
        () => {
          if (cancelled) return;
          setHealth({ ok: false });
          setEvidenceCount(null);
        },
      );
      getStatus().then(
        (s) => !cancelled && setEvidenceCount(s.counts.evidence),
        () => !cancelled && setEvidenceCount(null),
      );
    };
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const state = connectionState(health);
  const evidence = evidenceLabel(evidenceCount);

  return (
    <footer
      className="h-8 shrink-0 flex items-center gap-4 px-4 border-t text-xs"
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)", color: "var(--text-secondary)" }}
    >
      <div className="flex items-center gap-3 min-w-0">
        <span className="shrink-0">Reality Engine v{pkg.version}</span>
        {worldName && (
          <span className="truncate" style={{ color: "var(--text-primary)" }} title={worldName}>
            {worldName}
          </span>
        )}
      </div>
      <p className="flex-1 text-center truncate hidden md:block">Cleaner evidence. Better worlds.</p>
      <div className="flex items-center gap-4 ml-auto shrink-0">
        {evidence && <span className="font-mono-num">{evidence}</span>}
        <span className="flex items-center gap-1.5" role="status" aria-live="polite">
          <span aria-hidden="true" className="w-2 h-2 rounded-full" style={{ background: DOT_COLOR[state] }} />
          {DOT_LABEL[state]}
        </span>
      </div>
    </footer>
  );
}
