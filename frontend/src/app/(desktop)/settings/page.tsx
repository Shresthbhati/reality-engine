"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/ui/PageHeader";
import { getSystemTools, isApiError, setColmapPath, type SystemTools } from "@/lib/api";

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

        <Section title="Reconstruction tools">
          <ReconstructionTools />
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
            Everything is stored by the Reality Engine API running on this computer; nothing leaves it.
          </p>
        </Section>
      </div>
    </div>
  );
}

function ReconstructionTools() {
  const [tools, setTools] = useState<SystemTools | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () =>
    getSystemTools()
      .then((t) => {
        setTools(t);
        setError(null);
      })
      .catch((e) => setError(isApiError(e) ? e.message : String(e)));

  useEffect(() => {
    void load();
  }, []);

  const save = async (value: string | null) => {
    setBusy(true);
    try {
      setTools(await setColmapPath(value));
      setError(null);
      setPath("");
    } catch (e) {
      setError(isApiError(e) ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (error && !tools) {
    return (
      <p role="alert" className="text-sm" style={{ color: "var(--danger, #f87171)" }}>
        Could not read tool status from the API: {error}
      </p>
    );
  }
  if (!tools) return <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>Checking this computer…</p>;
  const c = tools.colmap;
  return (
    <div className="flex flex-col gap-3 text-sm" data-testid="reconstruction-tools">
      <div>
        <p className="font-medium" style={{ color: "var(--text-primary)" }}>
          COLMAP (photo reconstruction):{" "}
          <span data-testid="colmap-status" style={{ color: c.found ? "var(--accent)" : "#fbbf24" }}>
            {c.found ? "found" : "not found"}
          </span>
        </p>
        {c.found ? (
          <p className="text-xs mt-0.5 break-all" style={{ color: "var(--text-tertiary)" }}>
            {c.path} <span>(via {c.source})</span>
          </p>
        ) : (
          <>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
              Photos cannot be reconstructed into a model until COLMAP is installed. Looked in:
            </p>
            <ul className="mt-1 list-disc pl-5 text-xs break-all" style={{ color: "var(--text-tertiary)" }}>
              {c.tried.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </>
        )}
        {c.warnings.map((w) => (
          <p key={w} className="text-xs mt-1" style={{ color: "#fbbf24" }}>
            {w}
          </p>
        ))}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void save(path.trim() || null);
        }}
      >
        <label className="sr-only" htmlFor="colmap-path">
          Path to colmap.exe or its folder
        </label>
        <input
          id="colmap-path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder={c.saved_path ?? "Full path to colmap.exe (or its folder)"}
          className="min-w-0 flex-1 rounded px-2 py-1 text-xs"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
        />
        <button type="submit" disabled={busy || !path.trim()} className="rounded px-3 py-1 text-xs" style={{ border: "1px solid var(--border)", color: "var(--text-primary)" }}>
          Save path
        </button>
        {c.saved_path && (
          <button type="button" disabled={busy} onClick={() => void save(null)} className="rounded px-3 py-1 text-xs" style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
            Clear
          </button>
        )}
      </form>
      <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
        Order: {c.env_vars.join(" / ")} environment variable, this saved path, PATH, then common install folders.
      </p>
      {error && (
        <p role="alert" className="text-xs" style={{ color: "#f87171" }}>
          {error}
        </p>
      )}
      <p className="text-xs" style={{ color: tools.video.available ? "var(--text-tertiary)" : "#fbbf24" }} data-testid="video-status">
        Video: {tools.video.available ? "supported" : "not available"} — {tools.video.detail}
      </p>
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
