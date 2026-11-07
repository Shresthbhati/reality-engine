"use client";

/** Presentational pieces of the properties inspector. Colours come only from design tokens (globals.css). */
import { useId, useState, type ReactNode } from "react";
import { ChevronDown, Info } from "lucide-react";
import { TONE_TOKEN, type ConfidenceFacet } from "@/lib/inspector/confidence-model";
import {
  dimensionTooltip, fmtValue, stateWord, uncertaintyText, type DimensionRow, type Fact, type Swatch,
} from "@/lib/inspector/entity-facts";

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-2">
      <h4 className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-tertiary)]">{children}</h4>
      {aside}
    </div>
  );
}

export function Muted({ children }: { children: ReactNode }) {
  return <p className="text-[12px] leading-relaxed text-[var(--text-tertiary)]">{children}</p>;
}

/** Native <details>: closed by default, keyboard and screen-reader friendly. */
export function Disclosure({ title, testId, children }: { title: string; testId?: string; children: ReactNode }) {
  return (
    <details
      data-testid={testId}
      className="group border-t border-[var(--border-subtle)] pt-3 [&[open]>summary>svg]:rotate-180"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between text-[12px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] [&::-webkit-details-marker]:hidden">
        <span>{title}</span>
        <ChevronDown className="h-3.5 w-3.5 transition-transform" aria-hidden />
      </summary>
      <div className="mt-3 space-y-4">{children}</div>
    </details>
  );
}

export function InfoTip({ text, label }: { text: string; label: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
        className="rounded p-0.5 text-[var(--text-secondary)] hover:text-[var(--accent)] cursor-pointer"
      >
        <Info className="h-3.5 w-3.5" aria-hidden />
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute right-0 top-full z-30 mt-1 w-56 rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-2 text-[11px] leading-snug text-[var(--text-secondary)]"
        >
          {text}
        </span>
      )}
    </span>
  );
}

export function FactList({ facts }: { facts: Fact[] }) {
  return (
    <dl className="space-y-2 text-[12px]">
      {facts.map((f) => (
        <div key={f.label} className="flex items-baseline justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">{f.label}</dt>
          <dd className={`text-right font-mono-num ${f.muted ? "text-[var(--text-tertiary)]" : "text-[var(--text-primary)]"}`}>{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ConfidenceBars({ facets }: { facets: ConfidenceFacet[] }) {
  return (
    <ul className="space-y-2.5" data-testid="confidence-bars">
      {facets.map(({ key, label, reading }) => {
        const token = TONE_TOKEN[reading.tone];
        return (
          <li
            key={key}
            data-tone={reading.tone}
            className="grid grid-cols-[76px_96px_1fr] items-center gap-2 text-[12px]"
          >
            <span className="text-[var(--text-secondary)]">{label}</span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ background: token }} />
              <span className={reading.tone === "unknown" ? "text-[var(--text-tertiary)]" : "text-[var(--text-primary)]"}>{reading.label}</span>
            </span>
            <span
              className="h-1.5 overflow-hidden rounded-full bg-[var(--border-subtle)]"
              role={reading.ratio === null ? undefined : "meter"}
              aria-label={`${label} confidence`}
              aria-valuemin={reading.ratio === null ? undefined : 0}
              aria-valuemax={reading.ratio === null ? undefined : 1}
              aria-valuenow={reading.ratio ?? undefined}
            >
              {reading.ratio !== null && (
                <span className="block h-full rounded-full" style={{ width: `${Math.max(4, reading.ratio * 100)}%`, background: token }} />
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function DimensionRows({
  rows,
  onCorrect,
}: {
  rows: DimensionRow[];
  onCorrect?: (key: string) => void;
}) {
  return (
    <ul className="divide-y divide-[var(--border-subtle)]" data-testid="dimension-rows">
      {rows.map((r) => (
        <li key={r.key} className="flex items-center gap-3 py-2.5 text-[12px]">
          <span className="w-28 shrink-0 text-[var(--text-secondary)]">{r.label}</span>
          <span className="min-w-0 flex-1">
            <span className="block font-mono-num text-[var(--text-primary)]">
              {fmtValue(r.value, r.unit)}
              {uncertaintyText(r) && <span className="ml-1.5 text-[11px] text-[var(--text-tertiary)]" data-testid="dimension-uncertainty">{uncertaintyText(r)}</span>}
            </span>
            <span className="block text-[11px] text-[var(--text-tertiary)]">
              {r.assumedDefault ? "Assumed default, not measured" : stateWord(r.state)}
            </span>
          </span>
          {onCorrect && r.correctable && (
            <button
              type="button"
              onClick={() => onCorrect(r.key)}
              className="rounded px-1.5 py-0.5 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-subtle)] cursor-pointer"
            >
              Correct
            </button>
          )}
          <InfoTip text={dimensionTooltip(r)} label={`How ${r.label.toLowerCase()} was determined`} />
        </li>
      ))}
    </ul>
  );
}

const hex = (rgb: number[]) => "#" + rgb.map((n) => Math.round(n).toString(16).padStart(2, "0")).join("");

export function MaterialCards({ swatches, confidenceLabel, confidenceToken }: { swatches: Swatch[]; confidenceLabel: string; confidenceToken: string }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <ul className="divide-y divide-[var(--border-subtle)]" data-testid="material-cards">
      {swatches.map((s) => {
        const expanded = open === s.title;
        return (
          <li key={s.title}>
            <button
              type="button"
              aria-expanded={expanded}
              onClick={() => setOpen(expanded ? null : s.title)}
              className="flex w-full items-center gap-3 py-3 text-left cursor-pointer"
            >
              <span
                aria-hidden
                className="h-10 w-10 shrink-0 rounded border border-[var(--border)]"
                style={{ background: `rgb(${s.rgb.join(",")})` }}
              />
              <span className="min-w-0 flex-1">
                <span className="block text-[11px] text-[var(--text-tertiary)]">{s.title}</span>
                <span className="block text-[13px] text-[var(--text-primary)]">{s.name}</span>
                <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)]">
                  <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: confidenceToken }} />
                  Confidence: {confidenceLabel}
                </span>
              </span>
              <ChevronDown className={`h-4 w-4 shrink-0 text-[var(--text-tertiary)] transition-transform ${expanded ? "rotate-180" : ""}`} aria-hidden />
            </button>
            {expanded && (
              <p className="pb-3 pl-[52px] text-[11px] leading-relaxed text-[var(--text-secondary)]">
                <span className="font-mono-num text-[var(--text-primary)]">{hex(s.rgb)}</span> &middot; rgb({s.rgb.map((n) => Math.round(n)).join(", ")}).
                Average colour of the reconstructed surface. It is not a material identification.
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function EntityChip({ label, hint, onClick }: { label: string; hint?: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint}
      className="max-w-full truncate rounded border border-[var(--border)] px-2 py-0.5 text-[11px] text-[var(--text-secondary)] hover:border-[var(--accent-border)] hover:text-[var(--accent)] cursor-pointer"
    >
      {label}
    </button>
  );
}

export function ChipGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] text-[var(--text-tertiary)]">{title}</div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

/** Row used inside Details: a label and a mono value. */
export function KV({ k, v, wrap }: { k: string; v: ReactNode; wrap?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-[12px]">
      <span className="shrink-0 text-[var(--text-secondary)]">{k}</span>
      <span className={`text-right font-mono text-[var(--text-primary)] ${wrap ? "break-all" : "truncate"}`}>{v}</span>
    </div>
  );
}
