"use client";

import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CircleDashed,
  Loader2,
  Lightbulb,
  Plus,
} from "lucide-react";
import type {
  EvidenceStatus,
  QualityBar,
  QualityLevel,
  ReconstructionState,
  WorldStatus,
} from "@/lib/api";
import { modelStateCopy } from "@/lib/api";
import type { WorldIR } from "@/types/worldir";
import { cn } from "@/lib/utils";

const STATE_STYLE: Record<ReconstructionState, string> = {
  CAPTURING: "border-neutral-600 text-neutral-300",
  ANALYZING: "border-[#00e5ff]/50 text-[#00e5ff]",
  BUILDING_ROUGH_MODEL: "border-[#00e5ff]/50 text-[#00e5ff]",
  RECONSTRUCTING: "border-[#00e5ff]/50 text-[#00e5ff]",
  REFINING: "border-[#00e5ff]/50 text-[#00e5ff]",
  PARTIALLY_COMPLETE: "border-amber-400/60 text-amber-300",
  READY_TO_INSPECT: "border-emerald-400/60 text-emerald-300",
  NEEDS_MORE_EVIDENCE: "border-amber-400/60 text-amber-300",
  FAILED: "border-red-400/60 text-red-300",
  CANCELLED: "border-neutral-500 text-neutral-300",
};

const LEVEL_SEGMENTS: Record<QualityLevel, number> = { none: 0, low: 1, moderate: 2, good: 3, unknown: 0 };

const CONTRIBUTION_COPY: Record<string, string> = {
  first: "first view",
  new_view: "adds a new viewpoint",
  redundant: "mostly repeats an existing view",
  disconnected: "shares nothing verifiable with the rest",
  unknown: "not analysed",
};

function QualityRow({ bar }: { bar: QualityBar }) {
  const filled = LEVEL_SEGMENTS[bar.level];
  const unknown = bar.level === "unknown";
  return (
    <div title={bar.basis} className="flex items-center gap-2 text-[11px]">
      <span className="w-28 shrink-0 text-neutral-400">{bar.name}</span>
      <div
        className="flex gap-0.5"
        role="img"
        aria-label={`${bar.name}: ${unknown ? "not measured" : bar.level}`}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={cn(
              "h-2 w-6 rounded-sm",
              unknown ? "bg-neutral-800" : i < filled ? "bg-[#00e5ff]" : "bg-neutral-800",
            )}
          />
        ))}
      </div>
      <span className="text-neutral-300">{unknown ? "not measured" : bar.level}</span>
    </div>
  );
}

/** Plain-language failure line; a user-requested cancel is not an error. */
function failureCopy(status: WorldStatus): string {
  const f = status.failure;
  if (!f) return "";
  if (f.kind === "cancelled") return "Cancelled. Your photos are kept and the previous model is unchanged.";
  if (f.kind === "insufficient_evidence") return `Needs more evidence: ${f.message ?? "no usable photographs"}`;
  return `Could not finish: ${f.message ?? "no details were recorded"}`;
}

/** Observed / Inferred / Unknown: how much of this model was seen, deduced, or is simply not known. */
function provenanceCounts(world: WorldIR | null | undefined) {
  const counts = { observed: 0, reconstructed: 0, inferred: 0, unknown: 0 };
  if (!world?.entities) return counts;
  for (const e of Object.values(world.entities)) {
    const p = String((e as { provenance?: string }).provenance ?? "UNKNOWN").toUpperCase();
    if (p === "OBSERVED") counts.observed += 1;
    else if (p === "RECONSTRUCTED") counts.reconstructed += 1;
    else if (p === "INFERRED" || p === "ESTIMATED") counts.inferred += 1;
    else counts.unknown += 1;
  }
  return counts;
}

export interface ReconstructionStatusBarProps {
  status: WorldStatus | null;
  error: Error | null;
  world: WorldIR | null | undefined;
  /** version being inspected; null = current */
  selectedVersionId: string | null;
  onSelectVersion: (versionId: string | null) => void;
  onAddPhotos: () => void;
  onCompareVersions?: (base?: string, head?: string) => void;
  /** Stop the running reconstruction. Any version already adopted stands. */
  onCancel?: (jobId: string) => void;
  /** Re-run a failed/cancelled reconstruction (the photos are kept). */
  onRetry?: (sessionId: string) => void;
}

export default function ReconstructionStatusBar({
  status,
  error,
  world,
  selectedVersionId,
  onSelectVersion,
  onAddPhotos,
  onCompareVersions,
  onCancel,
  onRetry,
}: ReconstructionStatusBarProps) {
  const [showEvidence, setShowEvidence] = useState(false);
  // null = automatic: open while there is no model yet (nothing else to look at),
  // collapsed to a one-line strip once a model exists so the viewport keeps the space.
  const [expandedPref, setExpandedPref] = useState<boolean | null>(null);
  const counts = useMemo(() => provenanceCounts(world), [world]);

  if (!status) {
    return (
      <div className="flex h-10 items-center gap-2 px-3 text-xs text-neutral-400" role="status">
        {error ? (
          <>
            <AlertTriangle className="h-3.5 w-3.5 text-amber-300" aria-hidden />
            Reconstruction status unavailable: {error.message}
          </>
        ) : (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> Reading reconstruction status…
          </>
        )}
      </div>
    );
  }

  const model = status.model;
  const copy = modelStateCopy(model?.model_state);
  const StateIcon = status.in_progress
    ? Loader2
    : status.state === "READY_TO_INSPECT"
      ? CheckCircle2
      : status.state === "FAILED" || status.state === "NEEDS_MORE_EVIDENCE"
        ? AlertTriangle
        : CircleDashed;
  const inspecting = selectedVersionId
    ? status.versions.find((v) => v.id === selectedVersionId) ?? null
    : status.versions.find((v) => v.is_current) ?? null;
  const viewingOld = Boolean(inspecting && !inspecting.is_current);
  const expanded = expandedPref ?? !status.has_model;
  const topGuidance = status.guidance[0]?.message ?? null;

  return (
    <section aria-label="Reconstruction status" className="bg-[#0c0d12] text-neutral-200">
      {/* one-line strip: always visible */}
      <div className="flex h-9 items-center gap-2 px-3 text-[11px]">
        <span
          data-testid="reconstruction-state"
          className={cn(
            "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-semibold tracking-wide",
            STATE_STYLE[status.state],
          )}
        >
          <StateIcon className={cn("h-3.5 w-3.5", status.in_progress && "animate-spin")} aria-hidden />
          {status.state_label}
        </span>
        {model && (
          <span className="shrink-0 text-neutral-400" data-testid="model-state">
            {copy.label} · {model.images_registered}/{model.images_used} photos placed
          </span>
        )}
        {status.versions.length > 0 && (
          <span className="flex shrink-0 items-center gap-1" aria-label="Versions">
            {[...status.versions]
              .sort((a, b) => a.number - b.number)
              .map((v) => {
                const active = inspecting?.id === v.id;
                return (
                  <button
                    key={v.id}
                    type="button"
                    onClick={() => onSelectVersion(v.is_current ? null : v.id)}
                    aria-pressed={active}
                    data-testid={`version-${v.label}`}
                    title={`${v.images_used} photo${v.images_used === 1 ? "" : "s"}${v.is_current ? " · current" : ""}`}
                    className={cn(
                      "rounded border px-1.5 py-0 text-[10px] font-medium",
                      active
                        ? "border-[#00e5ff] bg-[#00e5ff]/15 text-[#00e5ff]"
                        : "border-[#2a2f3a] text-neutral-300 hover:bg-neutral-800",
                    )}
                  >
                    {v.label}
                  </button>
                );
              })}
          </span>
        )}
        <span className="min-w-0 flex-1 truncate text-neutral-400" title={topGuidance ?? undefined}>
          {viewingOld ? (
            <span className="text-amber-300">Viewing {inspecting?.label}, an earlier version</span>
          ) : status.failure ? (
            failureCopy(status)
          ) : (
            topGuidance ?? (status.in_progress ? "Working…" : "")
          )}
        </span>
        {!status.in_progress && status.job && onRetry && (status.state === "FAILED" || status.state === "CANCELLED") && (
          <button
            type="button"
            onClick={() => onRetry(status.job!.session_id)}
            data-testid="retry-reconstruction"
            className="shrink-0 rounded border border-[#00e5ff]/40 px-2 py-0 text-[10px] font-medium text-[#00e5ff] hover:bg-[#00e5ff]/10"
          >
            Try again
          </button>
        )}
        {status.in_progress && status.job && onCancel && (
          <button
            type="button"
            onClick={() => onCancel(status.job!.id)}
            data-testid="cancel-reconstruction"
            className="shrink-0 rounded border border-red-400/40 px-2 py-0 text-[10px] font-medium text-red-300 hover:bg-red-500/10"
          >
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={() => setExpandedPref(!expanded)}
          aria-expanded={expanded}
          className="inline-flex shrink-0 items-center gap-1 text-neutral-400 hover:text-white"
        >
          {expanded ? "Hide details" : "Details"}
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />}
        </button>
      </div>

      {expanded && (
      <div className="max-h-[42vh] overflow-y-auto border-t border-[#1f222b]">
      <div className="flex flex-col gap-3 px-3 py-2.5 md:flex-row md:items-start md:gap-6">
        {/* state + model honesty */}
        <div className="min-w-0 md:w-[30%]">
          <div className="flex flex-wrap items-center gap-2">
            <span
              data-testid="reconstruction-state"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold tracking-wide",
                STATE_STYLE[status.state],
              )}
            >
              <StateIcon className={cn("h-3.5 w-3.5", status.in_progress && "animate-spin")} aria-hidden />
              {status.state_label}
            </span>
            {model && (
              <span className="text-[11px] text-neutral-400" data-testid="model-state">
                Current spatial reconstruction · <strong className="text-neutral-200">{copy.label}</strong>
              </span>
            )}
          </div>
          {model ? (
            <>
              <p className="mt-1 text-[11px] leading-relaxed text-neutral-400">{copy.detail}</p>
              <p className="mt-1 text-[11px] text-neutral-400">
                {model.images_registered} of {model.images_used} photos placed in the model
                {model.scale?.state?.toLowerCase() === "relative" ? " · relative scale (no real-world units)" : ""}
              </p>
              <p
                className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px]"
                title="Observed = seen directly. Inferred = deduced by the engine. Unknown = not seen, so not guessed."
              >
                <span className="text-emerald-300">Observed {counts.observed + counts.reconstructed}</span>
                <span className="text-amber-300">Inferred {counts.inferred}</span>
                <span className="text-neutral-400">Unknown {counts.unknown}</span>
              </p>
            </>
          ) : (
            <p className="mt-1 text-[11px] text-neutral-400">
              {status.in_progress ? "The first model appears here as soon as it is ready." : "No model has been built yet."}
            </p>
          )}
          {status.failure && (
            <div
              role="alert"
              className="mt-2 rounded border border-red-500/40 bg-red-500/10 px-2 py-1.5 text-[11px] text-red-200"
            >
              {failureCopy(status)}
              {status.has_model ? " The previous model is still shown." : ""}
            </div>
          )}
        </div>

        {/* evidence quality */}
        <div className="min-w-0 md:w-[28%]">
          <div className="mb-1 flex items-center justify-between">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">Evidence</h3>
            <button
              type="button"
              onClick={() => setShowEvidence((v) => !v)}
              className="inline-flex items-center gap-1 text-[11px] text-neutral-400 hover:text-white"
              aria-expanded={showEvidence}
            >
              {status.evidence_summary.count} image{status.evidence_summary.count === 1 ? "" : "s"}
              {showEvidence ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
          </div>
          {status.evidence_summary.headline && (
            <p className="mb-1.5 text-[11px] text-neutral-300">{status.evidence_summary.headline}</p>
          )}
          <div className="space-y-1">
            {status.evidence_summary.quality.map((q) => (
              <QualityRow key={q.name} bar={q} />
            ))}
            {status.evidence_summary.quality.length === 0 && (
              <p className="text-[11px] text-neutral-500">Quality is measured once the photos are analysed.</p>
            )}
          </div>
        </div>

        {/* guidance */}
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center justify-between gap-2">
            <h3 className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
              <Lightbulb className="h-3 w-3 text-amber-300" aria-hidden /> What would help most
            </h3>
            <button
              type="button"
              onClick={onAddPhotos}
              className="inline-flex items-center gap-1 rounded border border-[#00e5ff]/40 bg-[#00e5ff]/10 px-2 py-0.5 text-[11px] font-medium text-[#00e5ff] hover:bg-[#00e5ff]/20"
            >
              <Plus className="h-3 w-3" aria-hidden /> Add evidence
            </button>
          </div>
          <ul className="space-y-1" data-testid="guidance">
            {status.guidance.slice(0, 4).map((g, i) => (
              <li key={`${g.kind}-${i}`} title={`Based on: ${g.basis}`} className="text-[11px] leading-relaxed text-neutral-300">
                • {g.message}
              </li>
            ))}
            {status.guidance.length === 0 && (
              <li className="text-[11px] text-neutral-500">
                {status.in_progress ? "Suggestions appear when the model is ready." : "No suggestions yet."}
              </li>
            )}
          </ul>
        </div>
      </div>

      {/* versions */}
      {status.versions.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-[#1f222b] px-3 py-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">Versions</span>
          {[...status.versions]
            .sort((a, b) => a.number - b.number)
            .map((v) => {
              const active = inspecting?.id === v.id;
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => onSelectVersion(v.is_current ? null : v.id)}
                  aria-pressed={active}
                  title={`${v.images_used} photo${v.images_used === 1 ? "" : "s"} · ${
                    v.model_state ? modelStateCopy(v.model_state).label.toLowerCase() : "state not recorded"
                  }${v.is_current ? " · current" : ""}`}
                  className={cn(
                    "rounded-md border px-2 py-0.5 text-[11px] font-medium",
                    active
                      ? "border-[#00e5ff] bg-[#00e5ff]/15 text-[#00e5ff]"
                      : "border-[#2a2f3a] text-neutral-300 hover:bg-neutral-800",
                  )}
                >
                  {v.label}
                  <span className="ml-1 text-[10px] text-neutral-500">{v.images_used} photos</span>
                </button>
              );
            })}
          {status.versions.length > 1 && onCompareVersions && inspecting && (
            <button
              type="button"
              onClick={() => onCompareVersions(inspecting.parent_version_id ?? undefined, inspecting.id)}
              className="text-[11px] text-neutral-400 underline-offset-2 hover:text-white hover:underline"
            >
              What changed in {inspecting.label}?
            </button>
          )}
          {viewingOld && (
            <span className="text-[11px] text-amber-300" role="status">
              Viewing an earlier version ({inspecting?.label}).{" "}
              <button type="button" className="underline" onClick={() => onSelectVersion(null)}>
                Back to current
              </button>
            </span>
          )}
        </div>
      )}

      {showEvidence && (
        <ul className="max-h-40 divide-y divide-[#1f222b] overflow-y-auto border-t border-[#1f222b] text-[11px]">
          {status.evidence.map((e: EvidenceStatus) => (
            <li key={e.id} className="flex items-center gap-3 px-3 py-1">
              <span className="w-48 shrink-0 truncate text-neutral-200" title={e.name}>
                {e.name}
              </span>
              <span className="w-32 shrink-0 text-neutral-400" title={e.class_basis ?? undefined}>
                {e.evidence_class ?? "not analysed"}
                {e.used_for_geometry === false ? " · context only" : ""}
              </span>
              <span className="text-neutral-400">
                {e.contribution ? CONTRIBUTION_COPY[e.contribution] ?? e.contribution : "waiting for analysis"}
                {e.registered === false ? " · not placed in the model" : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
      </div>
      )}
    </section>
  );
}
