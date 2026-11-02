"use client";

import React from "react";
import { AlertCircle, Check, Loader2, Lock } from "lucide-react";
import { buildFlowSteps, CAPTURE_TIPS, type FlowStatusInput, type FlowStep, type StepCta } from "@/lib/reconstruct/flow-model";
import { cn } from "@/lib/utils";

export interface GuidedFlowProps {
  /** null = no world yet (the user is about to add the first photos) */
  status: FlowStatusInput | null | undefined;
  onCta?: (kind: StepCta) => void;
  /** one-row version for the workspace status bar: only the current step's reason is spelled out */
  compact?: boolean;
  /** show the capture tips under step 1 (the reconstruct page) */
  showTips?: boolean;
}

const STATE_LABEL: Record<FlowStep["state"], string> = {
  done: "Done",
  active: "In progress",
  action: "Needs action",
  unavailable: "Not available",
};

function Marker({ step }: { step: FlowStep }) {
  const base = "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold";
  if (step.state === "done") return <span className={cn(base, "border-emerald-400/60 bg-emerald-400/15 text-emerald-300")}><Check className="h-3 w-3" aria-hidden /></span>;
  if (step.state === "active") return <span className={cn(base, "border-[var(--accent)]/60 text-[var(--accent)]")}><Loader2 className="h-3 w-3 animate-spin" aria-hidden /></span>;
  if (step.state === "unavailable") return <span className={cn(base, "border-neutral-700 text-neutral-500")}><Lock className="h-2.5 w-2.5" aria-hidden /></span>;
  return <span className={cn(base, step.current ? "border-amber-300 bg-amber-300/15 text-amber-200" : "border-neutral-500 text-neutral-300")}>{step.n}</span>;
}

/** Linear 5-step flow: photos -> reconstruct -> scale -> review -> measure. State is derived, never stored. */
export default function GuidedFlow({ status, onCta, compact = false, showTips = false }: GuidedFlowProps) {
  const steps = buildFlowSteps(status);
  const current = steps.find((s) => s.current) ?? null;

  return (
    <nav aria-label="Reconstruction steps" data-testid="guided-flow">
      <ol className={cn("grid gap-1.5", compact ? "grid-cols-5" : "grid-cols-1 sm:grid-cols-5 sm:gap-2")}>
        {steps.map((s) => (
          <li
            key={s.id}
            data-testid={`flow-step-${s.id}`}
            data-state={s.state}
            data-current={s.current ? "true" : undefined}
            aria-current={s.current ? "step" : undefined}
            title={`${s.title}: ${STATE_LABEL[s.state]}. ${s.reason}`}
            className={cn(
              "min-w-0 rounded-md border px-2 py-1.5",
              s.current ? "border-amber-300/50 bg-amber-300/5" : "border-[var(--border)]",
              s.state === "unavailable" && "opacity-60",
            )}
          >
            <div className="flex items-center gap-1.5">
              <Marker step={s} />
              <span className={cn("truncate text-[11px] font-semibold", s.state === "unavailable" ? "text-neutral-400" : "text-neutral-100")}>
                <span className="sr-only">Step {s.n}, {STATE_LABEL[s.state]}: </span>{s.title}
              </span>
            </div>
            {!compact && (
              <>
                <p className="mt-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400">{STATE_LABEL[s.state]}</p>
                <p className="mt-0.5 text-[11px] leading-snug text-neutral-300">{s.reason}</p>
                {s.cta && onCta && (
                  <button
                    type="button"
                    onClick={() => onCta(s.cta!.kind)}
                    data-testid={`flow-cta-${s.cta.kind}`}
                    aria-label={`${s.cta.label}, step ${s.n}: ${s.title}`}
                    className="mt-1.5 rounded border border-[var(--accent)]/40 px-2 py-0.5 text-[11px] font-medium text-[var(--accent)] hover:bg-[var(--accent)]/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
                  >
                    {s.cta.label}
                  </button>
                )}
              </>
            )}
          </li>
        ))}
      </ol>

      {compact && current && (
        <p className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-neutral-300" data-testid="flow-current">
          <AlertCircle className="h-3 w-3 shrink-0 text-amber-300" aria-hidden />
          <span>
            <strong className="text-neutral-100">Step {current.n}, {current.title}:</strong> {current.reason}
          </span>
          {current.cta && onCta && (
            <button
              type="button"
              onClick={() => onCta(current.cta!.kind)}
              data-testid={`flow-cta-${current.cta.kind}`}
              aria-label={`${current.cta.label}, step ${current.n}: ${current.title}`}
              className="rounded border border-[var(--accent)]/40 px-2 py-0 text-[10px] font-medium text-[var(--accent)] hover:bg-[var(--accent)]/10"
            >
              {current.cta.label}
            </button>
          )}
        </p>
      )}

      {showTips && (
        <div className="mt-3 rounded-md border border-[var(--border)] p-3" data-testid="capture-tips">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">Capture tips</h3>
          <ul className="mt-1 space-y-1">
            {CAPTURE_TIPS.map((t) => (
              <li key={t} className="text-[11px] leading-relaxed text-neutral-300">
                • {t}
              </li>
            ))}
          </ul>
        </div>
      )}
    </nav>
  );
}
