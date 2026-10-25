import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import {
  RESULT_LABEL,
  checkTone,
  resultTone,
  scoreTone,
  type CheckStatus,
  type QualityCheck,
  type QualityResult,
  type Tone,
} from "@/lib/api/evidence-quality-model";

export const TONE_COLOR: Record<Tone, string> = {
  success: "var(--success)",
  warning: "var(--warning)",
  error: "var(--error)",
  neutral: "var(--text-tertiary)",
};
export const TONE_BG: Record<Tone, string> = {
  success: "var(--success-subtle)",
  warning: "var(--warning-subtle)",
  error: "var(--error-subtle)",
  neutral: "var(--bg-hover)",
};

export function ResultChip({ result }: { result: QualityResult }) {
  // needs_review shares the warning tone with accepted_with_warning; the purple token tells them apart
  const color = result === "needs_review" ? "var(--purple)" : TONE_COLOR[resultTone(result)];
  const bg = result === "needs_review" ? "var(--purple-subtle)" : TONE_BG[resultTone(result)];
  return (
    <span
      className="inline-flex items-center px-2 h-6 rounded text-xs font-medium whitespace-nowrap"
      style={{ color, background: bg, border: `1px solid ${color}` }}
    >
      {RESULT_LABEL[result]}
    </span>
  );
}

/** Score ring: the arc is the 0-100 score exactly as the server computed it; "n/a" when never inspected. */
export function ScoreRing({ score, size = 36 }: { score: number | null; size?: number }) {
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const color = TONE_COLOR[scoreTone(score)];
  return (
    <span className="relative inline-flex items-center justify-center shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={3} />
        {score !== null && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={3}
            strokeLinecap="round"
            strokeDasharray={`${(score / 100) * c} ${c}`}
          />
        )}
      </svg>
      <span className="absolute text-[11px] font-mono-num font-semibold" style={{ color: "var(--text-primary)" }}>
        {score === null ? "n/a" : score}
      </span>
    </span>
  );
}

const STATUS_WORD: Record<CheckStatus, string> = { pass: "Pass", warn: "Warn", fail: "Fail" };

export function CheckIcon({ status, className = "w-3.5 h-3.5" }: { status: CheckStatus; className?: string }) {
  const style = { color: TONE_COLOR[checkTone(status)] };
  const Icon = status === "pass" ? CheckCircle2 : status === "warn" ? AlertTriangle : XCircle;
  return <Icon className={className} style={style} aria-label={STATUS_WORD[status]} />;
}

/** The Quality Checks list: name, Pass/Warn/Fail, and the real measured value / message. */
export function CheckRows({ checks }: { checks: QualityCheck[] }) {
  if (checks.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        Not inspected yet: there are no measured facts for this file.
      </p>
    );
  }
  return (
    <ul className="flex flex-col">
      {checks.map((c) => (
        <li
          key={c.name}
          className="grid items-center gap-2 py-1.5 text-sm border-b last:border-b-0"
          style={{ gridTemplateColumns: "120px 56px 1fr", borderColor: "var(--border-subtle)" }}
        >
          <span style={{ color: "var(--text-secondary)" }}>{c.name}</span>
          <span className="inline-flex items-center gap-1" style={{ color: TONE_COLOR[checkTone(c.status)] }}>
            <CheckIcon status={c.status} />
            {STATUS_WORD[c.status]}
          </span>
          <span style={{ color: "var(--text-primary)" }} title={c.threshold ? `Threshold: ${c.threshold}` : undefined}>
            {c.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function Meta({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 text-sm py-0.5">
      <span style={{ color: "var(--text-tertiary)" }}>{label}</span>
      <span className="text-right font-mono-num break-all" style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}

export function Panel({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section
      className={`rounded-md p-4 ${className}`}
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <h3 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h3>
      {children}
    </section>
  );
}

export const EXPLANATION_NOTE = "Generated from the checks above, not an AI model.";
export const DECISION_NOTE =
  "Rejected photos are excluded from the next reconstruction. The other decisions are recorded for you and do not change reconstruction input. A model that was already built keeps what it used until it is rebuilt.";
