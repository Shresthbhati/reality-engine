"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/ui/PageHeader";

const SHORTCUTS: Array<{ keys: string; action: string }> = [
  { keys: "Ctrl/⌘ + K", action: "Open search and commands" },
  { keys: "Esc", action: "Close the open dialog or panel" },
  { keys: "Tab / Shift+Tab", action: "Move between focusable controls" },
  { keys: "↑ / ↓", action: "Move the highlight in search results and commands" },
  { keys: "Enter", action: "Activate the highlighted result or command" },
];

function readReduceMotion(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem("re-reduce-motion") === "true";
  } catch {
    return false;
  }
}

export default function SettingsPage() {
  const [reduceMotion, setReduceMotion] = useState(readReduceMotion);

  const toggleReduceMotion = () => {
    const next = !reduceMotion;
    setReduceMotion(next);
    document.documentElement.setAttribute("data-reduce-motion", String(next));
    try {
      window.localStorage.setItem("re-reduce-motion", String(next));
    } catch {
      // Private browsing / blocked storage — the toggle still works for
      // this page load, it just won't persist across visits.
    }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <PageHeader title="Settings" />

      <div className="flex flex-col gap-8 px-6 py-6 max-w-xl">
        <Section title="Appearance">
          <ToggleRow
            label="Reduce motion"
            description="Minimizes transitions and animations across the app, independent of your OS setting."
            checked={reduceMotion}
            onChange={toggleReduceMotion}
          />
        </Section>

        <Section title="Keyboard Shortcuts">
          <div className="flex flex-col gap-1.5">
            {SHORTCUTS.map((s) => (
              <div key={s.keys} className="flex items-center justify-between text-sm">
                <span style={{ color: "var(--text-secondary)" }}>{s.action}</span>
                <kbd
                  className="text-xs font-mono-num px-1.5 py-0.5 rounded"
                  style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                >
                  {s.keys}
                </kbd>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Account">
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            Account management isn&apos;t available yet — this workstation isn&apos;t connected to a real authentication backend.
          </p>
        </Section>

        <Section title="Workspace">
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No workspace-level settings are configurable yet.
          </p>
        </Section>

        <Section title="Data">
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No backend is connected — Worlds, Sessions, Evidence, Analysis, Results, and Reports are all empty until a real data source is wired in.
          </p>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          {label}
        </p>
        <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
          {description}
        </p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={onChange}
        className="w-10 h-6 rounded-full shrink-0 relative transition-colors"
        style={{ background: checked ? "var(--accent)" : "var(--bg-elevated)", border: "1px solid var(--border)" }}
      >
        <span
          className="absolute top-0.5 w-4.5 h-4.5 rounded-full transition-transform"
          style={{
            background: checked ? "var(--bg-base)" : "var(--text-tertiary)",
            transform: checked ? "translateX(19px)" : "translateX(2px)",
          }}
        />
      </button>
    </div>
  );
}
