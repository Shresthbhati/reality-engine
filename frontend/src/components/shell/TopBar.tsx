"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Search, Bell, Box, ChevronRight, Download } from "lucide-react";
import type { WorldRow } from "@/lib/types";
import { EXPORT_HREF, isGatedHref, statusText, type ConnectionState } from "@/lib/shell/shell-logic";
import { SHOW_UNFINISHED } from "@/lib/unfinished";
import NotificationsPanel from "./NotificationsPanel";

const DOT_COLOR: Record<ConnectionState, string> = {
  online: "var(--success)",
  offline: "var(--error)",
  checking: "var(--text-tertiary)",
};

interface TopBarProps {
  onOpenSearch: () => void;
  /** null while loading or after a failure (see worldsError). */
  worlds: WorldRow[] | null;
  worldsError: string | null;
  currentWorldId: string | null;
  connection: ConnectionState;
}

const iconBtn = "w-9 h-9 rounded-md items-center justify-center transition-colors hover:bg-[var(--bg-hover)]";

export default function TopBar({ onOpenSearch, worlds, worldsError, currentWorldId, connection }: TopBarProps) {
  const router = useRouter();
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  const switcherTitle = worldsError
    ? `Worlds could not be loaded: ${worldsError}`
    : worlds === null
      ? "Loading worlds"
      : worlds.length === 0
        ? "No worlds yet"
        : "Switch world";

  return (
    <header
      className="h-12 shrink-0 flex items-center gap-2 md:gap-3 px-3 border-b relative"
      style={{ background: "var(--bg-sidebar)", borderColor: "var(--border)" }}
    >
      <Link href="/" aria-label="Reality Engine home" className="flex items-center gap-2 shrink-0 rounded-md">
        <span
          aria-hidden="true"
          className="w-7 h-7 rounded-md flex items-center justify-center"
          style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
        >
          <Box className="w-4 h-4" />
        </span>
        <span className="hidden md:inline text-xs font-semibold tracking-[0.16em]" style={{ color: "var(--text-primary)" }}>
          REALITY ENGINE
        </span>
      </Link>

      <nav aria-label="Breadcrumb" className={`${currentWorldId ? "hidden md:flex" : "hidden"} items-center gap-1.5 min-w-0 text-sm ml-2`}>
        <Link href="/worlds" className="shrink-0 hover:underline" style={{ color: "var(--text-secondary)" }}>
          Worlds
        </Link>
        <ChevronRight className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--text-tertiary)" }} aria-hidden="true" />
        <label className="min-w-0 flex items-center">
          <span className="sr-only">Switch world</span>
          <select
            value={currentWorldId ?? ""}
            disabled={!worlds || worlds.length === 0}
            title={switcherTitle}
            onChange={(e) => {
              if (e.target.value) router.push(`/worlds/${encodeURIComponent(e.target.value)}`);
            }}
            className="h-8 max-w-48 xl:max-w-64 px-1.5 rounded-md text-sm truncate bg-transparent hover:bg-[var(--bg-hover)] disabled:opacity-60"
            style={{ border: "1px solid transparent", color: "var(--text-primary)" }}
          >
            <option value="">
              {worldsError ? "Worlds unavailable" : worlds === null ? "Loading…" : worlds.length === 0 ? "No worlds yet" : "Choose a world"}
            </option>
            {(worlds ?? []).map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </label>
      </nav>

      <span
        className="hidden md:flex items-center gap-1.5 text-xs shrink-0"
        role="status"
        aria-live="polite"
        style={{ color: "var(--text-secondary)" }}
      >
        <span aria-hidden="true" className="w-1.5 h-1.5 rounded-full" style={{ background: DOT_COLOR[connection] }} />
        {statusText(connection)}
      </span>

      <div className="flex items-center gap-1 ml-auto shrink-0">
        <button
          type="button"
          onClick={onOpenSearch}
          title="Search (Ctrl K)"
          aria-label="Search (Ctrl K)"
          className={`${iconBtn} flex`}
          style={{ color: "var(--text-secondary)" }}
        >
          <Search className="w-4 h-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          title="Notifications"
          aria-label="Notifications"
          aria-haspopup="true"
          aria-expanded={notificationsOpen}
          onClick={() => setNotificationsOpen((v) => !v)}
          className={`${iconBtn} hidden md:flex`}
          style={{ color: "var(--text-secondary)" }}
        >
          <Bell className="w-4 h-4" aria-hidden="true" />
        </button>
        <NotificationsPanel open={notificationsOpen} onClose={() => setNotificationsOpen(false)} />
        {(SHOW_UNFINISHED || !isGatedHref(EXPORT_HREF)) && (
          <Link
            href={EXPORT_HREF}
            title="Export and reports"
            aria-label="Export"
            className="ml-1 h-8 px-3 rounded-md inline-flex items-center gap-1.5 text-sm font-medium transition-colors hover:bg-[var(--primary-hover)]"
            style={{ background: "var(--primary)", color: "var(--text-primary)" }}
          >
            <Download className="w-4 h-4" aria-hidden="true" />
            <span>Export</span>
          </Link>
        )}
      </div>
    </header>
  );
}
