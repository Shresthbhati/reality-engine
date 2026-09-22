"use client";

import { useEffect, useRef } from "react";
import { Bell } from "lucide-react";

/**
 * No backend notification stream is wired into this rebuild yet — the list
 * is genuinely empty. When events exist (Session completed, Evidence
 * processed, World shared, Report ready), push them here with a deep link,
 * do not fabricate sample notifications to fill the panel.
 */
export default function NotificationsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedRef = useRef<Element | null>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  useEffect(() => {
    if (open) {
      previouslyFocusedRef.current = document.activeElement;
      requestAnimationFrame(() => panelRef.current?.focus());
    } else if (previouslyFocusedRef.current instanceof HTMLElement && document.contains(previouslyFocusedRef.current)) {
      previouslyFocusedRef.current.focus();
    }
  }, [open]);

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        ref={panelRef}
        tabIndex={-1}
        className="absolute right-4 top-12 z-50 w-80 rounded-lg overflow-hidden shadow-2xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", outline: "none" }}
      >
        <div className="px-3 h-10 flex items-center text-sm font-semibold border-b" style={{ borderColor: "var(--border)", color: "var(--text-primary)" }}>
          Notifications
        </div>
        <div className="flex flex-col items-center gap-2 py-8 px-4 text-center">
          <Bell className="w-5 h-5" style={{ color: "var(--text-tertiary)" }} />
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No notifications yet.
          </p>
        </div>
      </div>
    </>
  );
}
