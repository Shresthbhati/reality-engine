"use client";

import { statusText, type ConnectionState } from "@/lib/shell/shell-logic";
import pkg from "../../../package.json";

const DOT_COLOR: Record<ConnectionState, string> = {
  online: "var(--success)",
  offline: "var(--error)",
  checking: "var(--text-tertiary)",
};

/** Slim footer: app version, the open world's name, and the real API connection state. */
export default function StatusFooter({ worldName, connection }: { worldName: string | null; connection: ConnectionState }) {
  return (
    <footer
      className="hidden md:flex h-7 shrink-0 items-center gap-4 px-4 border-t text-xs"
      style={{ background: "var(--bg-sidebar)", borderColor: "var(--border)", color: "var(--text-secondary)" }}
    >
      <span className="shrink-0">v{pkg.version}</span>
      {worldName && (
        <span className="truncate" style={{ color: "var(--text-primary)" }} title={worldName}>
          {worldName}
        </span>
      )}
      <span className="flex items-center gap-1.5 ml-auto shrink-0" role="status" aria-live="polite">
        <span aria-hidden="true" className="w-1.5 h-1.5 rounded-full" style={{ background: DOT_COLOR[connection] }} />
        {statusText(connection)}
      </span>
    </footer>
  );
}
