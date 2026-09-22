import type { ProcessingState } from "@/lib/types";

const STYLES: Record<ProcessingState, { bg: string; fg: string; label: string; pulse?: boolean }> = {
  QUEUED: { bg: "var(--bg-elevated)", fg: "var(--text-secondary)", label: "Queued" },
  PROCESSING: { bg: "var(--accent-subtle)", fg: "var(--accent)", label: "Processing", pulse: true },
  COMPLETE: { bg: "var(--success-subtle)", fg: "var(--success)", label: "Complete" },
  FAILED: { bg: "var(--error-subtle)", fg: "var(--error)", label: "Failed" },
};

export default function StatusBadge({ state }: { state: ProcessingState }) {
  const s = STYLES[state];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 h-5 rounded text-xs font-medium"
      style={{ background: s.bg, color: s.fg }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${s.pulse ? "animate-pulse" : ""}`}
        style={{ background: s.fg }}
      />
      {s.label}
    </span>
  );
}
