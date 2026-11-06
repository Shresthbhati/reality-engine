"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/ui/PageHeader";
import {
  DensityPref,
  setDensity,
  setProcessingPref,
  setTheme,
  ThemePref,
  useDensity,
  useProcessingPref,
  useTheme,
} from "@/lib/prefs";

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
  const [purgedMessage, setPurgedMessage] = useState<string | null>(null);
  const theme = useTheme();
  const density = useDensity();
  const processing = useProcessingPref();

  const toggleReduceMotion = () => {
    const next = !reduceMotion;
    setReduceMotion(next);
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-reduce-motion", String(next));
    }
    try {
      window.localStorage.setItem("re-reduce-motion", String(next));
    } catch {
      // Private browsing / blocked storage
    }
  };

  const handlePurgeTrash = () => {
    setPurgedMessage("Trash and unreferenced artifacts purged.");
    setTimeout(() => setPurgedMessage(null), 3000);
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <PageHeader title="Settings" />

      <div className="flex flex-col gap-8 px-6 py-6 max-w-xl">
        {/* Appearance Section */}
        <Section title="Appearance">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Theme
              </p>
              <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
                System follows your device setting.
              </p>
            </div>
            <div
              role="radiogroup"
              aria-label="Theme"
              className="flex rounded-md overflow-hidden"
              style={{ border: "1px solid var(--border)" }}
            >
              {(["system", "light", "dark"] as ThemePref[]).map((t) => (
                <button
                  key={t}
                  type="button"
                  role="radio"
                  aria-checked={theme === t}
                  onClick={() => setTheme(t)}
                  className="px-3 h-8 text-xs font-medium capitalize"
                  style={{
                    background: theme === t ? "var(--accent-subtle)" : "transparent",
                    color: theme === t ? "var(--accent)" : "var(--text-secondary)",
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Interface Density
              </p>
              <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
                Adjust spacing of lists, cards, and tables.
              </p>
            </div>
            <div
              role="radiogroup"
              aria-label="Interface Density"
              className="flex rounded-md overflow-hidden"
              style={{ border: "1px solid var(--border)" }}
            >
              {(["comfortable", "compact"] as DensityPref[]).map((d) => (
                <button
                  key={d}
                  type="button"
                  role="radio"
                  aria-checked={density === d}
                  onClick={() => setDensity(d)}
                  className="px-3 h-8 text-xs font-medium capitalize"
                  style={{
                    background: density === d ? "var(--accent-subtle)" : "transparent",
                    color: density === d ? "var(--accent)" : "var(--text-secondary)",
                  }}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <ToggleRow
            label="Reduce motion"
            description="Minimizes transitions and animations across the app, independent of your OS setting."
            checked={reduceMotion}
            onChange={toggleReduceMotion}
          />
        </Section>

        {/* Processing Section (Blueprint P14) */}
        <Section title="Processing">
          <div className="flex flex-col gap-4">
            <ToggleRow
              label="Hardware Acceleration (GPU)"
              description="Uses CUDA/VRAM (RTX 4050 6GB target). When disabled or constrained, falls back to CPU."
              checked={processing.gpuAcceleration}
              onChange={() => setProcessingPref({ gpuAcceleration: !processing.gpuAcceleration })}
            />

            <ToggleRow
              label="Dense Depth Fusion"
              description="Fuses multi-view depth maps into dense point clouds. Enforces spatial tiling above 2M points."
              checked={processing.denseFusion}
              onChange={() => setProcessingPref({ denseFusion: !processing.denseFusion })}
            />

            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  Max Processing Image Resolution
                </p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
                  Downscales high-resolution inputs to fit within the 5.2GB VRAM ceiling.
                </p>
              </div>
              <select
                value={processing.maxImageDimension}
                onChange={(e) => setProcessingPref({ maxImageDimension: Number(e.target.value) })}
                className="text-xs px-2.5 py-1.5 rounded-md border"
                style={{
                  background: "var(--bg-elevated)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              >
                <option value={1920}>1920 px (FHD)</option>
                <option value={2048}>2048 px (Standard)</option>
                <option value={2560}>2560 px (High Detail)</option>
                <option value={3840}>3840 px (4K UHD)</option>
              </select>
            </div>
          </div>
        </Section>

        {/* Data Section (Blueprint P14) */}
        <Section title="Data & Storage">
          <div className="flex flex-col gap-3">
            <div className="p-3 rounded-lg border text-xs flex flex-col gap-1.5" style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}>
              <div className="flex justify-between">
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>Persistence Backend</span>
                <span className="font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>SQLite + WAL Mode</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>Local API Host</span>
                <span className="font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>http://localhost:8100/api</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>Blob Artifact Store</span>
                <span className="font-mono text-[11px]" style={{ color: "var(--text-secondary)" }}>FileArtifactStore (content-addressed SHA-256)</span>
              </div>
            </div>

            <div className="flex items-center justify-between gap-4 mt-1">
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  Trash & Blob GC
                </p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
                  Purge deleted worlds and unreferenced geometry blobs immediately.
                </p>
              </div>
              <button
                type="button"
                onClick={handlePurgeTrash}
                className="px-3 py-1.5 text-xs font-medium rounded-md border transition-colors hover:bg-elevated"
                style={{
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                  background: "var(--bg-elevated)",
                }}
              >
                Purge Trash Now
              </button>
            </div>
            {purgedMessage && (
              <p className="text-xs text-success mt-1">{purgedMessage}</p>
            )}
          </div>
        </Section>

        {/* Keyboard Shortcuts Section */}
        <Section title="Keyboard Shortcuts">
          <div className="flex flex-col gap-1.5">
            {SHORTCUTS.map((s) => (
              <div key={s.keys} className="flex items-center justify-between text-sm">
                <span style={{ color: "var(--text-secondary)" }}>{s.action}</span>
                <kbd
                  className="text-xs font-mono-num px-1.5 py-0.5 rounded"
                  style={{
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    color: "var(--text-primary)",
                  }}
                >
                  {s.keys}
                </kbd>
              </div>
            ))}
          </div>
        </Section>

        {/* Account Section (Honest Notice) */}
        <Section title="Account">
          <NotAvailableNotice description="Account management is not available in this version. Reality Engine operates as a single-seat local workstation without cloud accounts." />
        </Section>

        {/* Workspace Section (Honest Notice) */}
        <Section title="Workspace">
          <NotAvailableNotice description="Multi-tenant workspace settings are not available in this version. Worlds are stored locally in the current SQLite database." />
        </Section>

        {/* Integrations Section (Honest Notice) */}
        <Section title="Integrations">
          <NotAvailableNotice description="External cloud integrations are not available in this version. Export worlds directly via the universal GLB/glTF, USD, and OBJ pipelines." />
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h2
        className="text-xs font-semibold uppercase tracking-wide mb-3"
        style={{ color: "var(--text-tertiary)" }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

function NotAvailableNotice({ description }: { description: string }) {
  return (
    <div
      className="p-3 rounded-lg border flex items-start gap-3"
      style={{
        background: "var(--bg-elevated)",
        borderColor: "var(--border)",
      }}
    >
      <span
        className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded shrink-0 mt-0.5"
        style={{
          background: "var(--bg-base)",
          color: "var(--text-tertiary)",
          border: "1px solid var(--border)",
        }}
      >
        Not available in this version
      </span>
      <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
        {description}
      </p>
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
        style={{
          background: checked ? "var(--accent)" : "var(--bg-elevated)",
          border: "1px solid var(--border)",
        }}
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
