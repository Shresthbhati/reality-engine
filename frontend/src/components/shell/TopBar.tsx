"use client";

import { useState } from "react";
import { Search, HelpCircle, Bell } from "lucide-react";
import NotificationsPanel from "./NotificationsPanel";

export default function TopBar({ onOpenSearch }: { onOpenSearch: () => void }) {
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  return (
    <header
      className="h-14 shrink-0 flex items-center gap-4 px-4 border-b relative"
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-2 shrink-0">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center"
          style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
        </div>
        <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Reality Engine
        </span>
      </div>

      <button
        type="button"
        onClick={onOpenSearch}
        className="flex-1 max-w-md flex items-center gap-2 h-9 px-3 rounded-md text-sm transition-colors"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
      >
        <Search className="w-4 h-4" />
        <span>Search Reality Engine…</span>
        <kbd
          className="ml-auto text-xs font-mono-num px-1.5 py-0.5 rounded"
          style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}
        >
          ⌘K
        </kbd>
      </button>

      <div className="flex items-center gap-1 ml-auto shrink-0">
        <button
          type="button"
          title="Help"
          aria-label="Help"
          className="w-8 h-8 rounded-md flex items-center justify-center transition-colors"
          style={{ color: "var(--text-secondary)" }}
        >
          <HelpCircle className="w-4 h-4" />
        </button>
        <button
          type="button"
          title="Notifications"
          aria-label="Notifications"
          aria-haspopup="true"
          aria-expanded={notificationsOpen}
          onClick={() => setNotificationsOpen((v) => !v)}
          className="w-8 h-8 rounded-md flex items-center justify-center transition-colors relative"
          style={{ color: "var(--text-secondary)" }}
        >
          <Bell className="w-4 h-4" />
        </button>
        <NotificationsPanel open={notificationsOpen} onClose={() => setNotificationsOpen(false)} />
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold ml-1"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          title="Account"
        >
          SB
        </div>
      </div>
    </header>
  );
}
