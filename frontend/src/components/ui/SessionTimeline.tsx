import type { SessionStage } from "@/lib/types";

const DOT_COLOR: Record<SessionStage["status"], string> = {
  COMPLETE: "var(--success)",
  ACTIVE: "var(--accent)",
  FAILED: "var(--error)",
};

export default function SessionTimeline({ stages }: { stages: SessionStage[] }) {
  if (stages.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        No processing timeline recorded for this Session yet.
      </p>
    );
  }

  return (
    <ol className="flex flex-col gap-3">
      {stages.map((stage, i) => (
        <li key={`${stage.label}-${i}`} className="flex items-start gap-3">
          <span
            className={`w-2 h-2 rounded-full mt-1 shrink-0 ${stage.status === "ACTIVE" ? "animate-pulse" : ""}`}
            style={{ background: DOT_COLOR[stage.status] }}
          />
          <div className="flex items-baseline gap-2 text-sm">
            <span className="font-mono-num" style={{ color: "var(--text-tertiary)" }}>
              {stage.timestamp}
            </span>
            <span style={{ color: "var(--text-primary)" }}>{stage.label}</span>
          </div>
        </li>
      ))}
    </ol>
  );
}
