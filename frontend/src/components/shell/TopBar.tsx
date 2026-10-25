"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Search,
  Bell,
  User,
  Globe,
  CloudUpload,
  Cpu,
  Compass,
  GitCompare,
  BarChart3,
  Share2,
  type LucideIcon,
} from "lucide-react";
import type { WorldRow } from "@/lib/types";
import { MODE_TABS, NOT_AVAILABLE_TITLE, activeModeId } from "@/lib/shell/shell-logic";
import NotificationsPanel from "./NotificationsPanel";

const MODE_ICONS: Record<string, LucideIcon> = {
  ingest: CloudUpload,
  process: Cpu,
  explore: Compass,
  compare: GitCompare,
  analyze: BarChart3,
  export: Share2,
};

interface TopBarProps {
  onOpenSearch: () => void;
  /** null while loading or after a failure (see worldsError). */
  worlds: WorldRow[] | null;
  worldsError: string | null;
  currentWorldId: string | null;
}

export default function TopBar({ onOpenSearch, worlds, worldsError, currentWorldId }: TopBarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const activeMode = activeModeId(pathname);

  const switcherTitle = worldsError
    ? `Worlds could not be loaded: ${worldsError}`
    : worlds === null
      ? "Loading worlds"
      : worlds.length === 0
        ? "No worlds yet"
        : "Switch world";

  return (
    <header
      className="h-14 shrink-0 flex items-center gap-3 px-3 border-b relative"
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
    >
      <Link href="/" aria-label="Reality Engine home" className="flex items-center gap-2.5 shrink-0 rounded-md">
        <span
          aria-hidden="true"
          className="w-9 h-9 rounded-md flex items-center justify-center text-sm font-bold tracking-tight"
          style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
        >
          RE
        </span>
        <span className="flex flex-col leading-tight">
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Reality Engine
          </span>
          <span className="hidden lg:block text-[11px]" style={{ color: "var(--text-secondary)" }}>
            Real places. Persistent digital worlds.
          </span>
        </span>
      </Link>

      <nav aria-label="Modes" className="flex items-center gap-1 shrink-0 ml-1">
        {MODE_TABS.map((tab) => {
          const Icon = MODE_ICONS[tab.id];
          const active = tab.id === activeMode;
          const cls = "flex items-center gap-1.5 h-9 px-2.5 rounded-md border text-sm font-medium transition-colors";
          if (tab.href === null) {
            return (
              <button
                key={tab.id}
                type="button"
                disabled
                title={NOT_AVAILABLE_TITLE}
                aria-label={`${tab.label} (${NOT_AVAILABLE_TITLE.toLowerCase()})`}
                className={`${cls} cursor-not-allowed opacity-50`}
                style={{ color: "var(--text-secondary)", borderColor: "transparent" }}
              >
                <Icon className="w-4 h-4" aria-hidden="true" />
                <span className="hidden xl:inline">{tab.label}</span>
              </button>
            );
          }
          return (
            <Link
              key={tab.id}
              href={tab.href}
              title={tab.label}
              aria-label={tab.label}
              aria-current={active ? "page" : undefined}
              className={`${cls} hover:bg-[var(--bg-hover)]`}
              style={{
                color: active ? "var(--accent)" : "var(--text-secondary)",
                background: active ? "var(--accent-subtle)" : undefined,
                borderColor: active ? "var(--accent)" : "transparent",
              }}
            >
              <Icon className="w-4 h-4" aria-hidden="true" />
              <span className="hidden xl:inline">{tab.label}</span>
            </Link>
          );
        })}
      </nav>

      <button
        type="button"
        onClick={onOpenSearch}
        aria-label="Search (Ctrl K)"
        className="flex-1 min-w-0 max-w-md ml-auto flex items-center gap-2 h-9 px-3 rounded-md text-sm transition-colors"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
      >
        <Search className="w-4 h-4 shrink-0" aria-hidden="true" />
        <span className="truncate">Search evidence, locations, sessions…</span>
        <kbd
          className="ml-auto shrink-0 text-xs font-mono-num px-1.5 py-0.5 rounded"
          style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}
        >
          Ctrl K
        </kbd>
      </button>

      <label className="relative shrink-0 flex items-center">
        <span className="sr-only">Switch world</span>
        <Globe className="w-4 h-4 absolute left-2.5 pointer-events-none" style={{ color: "var(--accent)" }} aria-hidden="true" />
        <select
          value={currentWorldId ?? ""}
          disabled={!worlds || worlds.length === 0}
          title={switcherTitle}
          onChange={(e) => {
            if (e.target.value) router.push(`/worlds/${encodeURIComponent(e.target.value)}`);
          }}
          className="h-9 w-36 xl:w-48 pl-8 pr-2 rounded-md text-sm truncate disabled:opacity-60"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
        >
          <option value="">
            {worldsError ? "Worlds unavailable" : worlds === null ? "Loading…" : worlds.length === 0 ? "No worlds" : "Select world"}
          </option>
          {(worlds ?? []).map((w) => (
            <option key={w.id} value={w.id}>
              {w.name}
            </option>
          ))}
        </select>
      </label>

      <div className="flex items-center gap-1 shrink-0">
        <button
          type="button"
          title="Notifications"
          aria-label="Notifications"
          aria-haspopup="true"
          aria-expanded={notificationsOpen}
          onClick={() => setNotificationsOpen((v) => !v)}
          className="w-9 h-9 rounded-md flex items-center justify-center transition-colors hover:bg-[var(--bg-hover)]"
          style={{ color: "var(--text-secondary)" }}
        >
          <Bell className="w-4 h-4" aria-hidden="true" />
        </button>
        <NotificationsPanel open={notificationsOpen} onClose={() => setNotificationsOpen(false)} />
        {/* There is no sign-in yet, so no name or initials are invented. */}
        <span
          className="w-8 h-8 rounded-full flex items-center justify-center"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          title="Local session (no account system yet)"
          role="img"
          aria-label="Local session"
        >
          <User className="w-4 h-4" aria-hidden="true" />
        </span>
      </div>
    </header>
  );
}
