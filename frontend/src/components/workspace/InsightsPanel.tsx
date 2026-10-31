"use client";

import React, { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { insightsStatus, type InsightCard, type InsightsModel } from "@/lib/workspace/insights-model";

const MAX_THUMBS = 5;

interface InsightsPanelProps {
  model: InsightsModel;
  worldId: string;
  /** Controlled: the workstation resolves the default (open only when there is something to show). */
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectEntity: (id: string) => void;
  onSelectEvidence: (id: string) => void;
  /** "+N more" opens the evidence view. */
  onOpenEvidence: () => void;
}

function Thumb({ worldId, id, onClick }: { worldId: string; id: string; onClick: () => void }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <button
      type="button"
      onClick={onClick}
      title={`Photograph ${id}`}
      aria-label={`Show photograph ${id}`}
      className="h-10 w-14 shrink-0 overflow-hidden rounded border border-[var(--border)] bg-[var(--bg-base)] hover:border-[var(--accent-border)] cursor-pointer"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`/api/worlds/${encodeURIComponent(worldId)}/evidence/${encodeURIComponent(id)}/image`}
        alt=""
        loading="lazy"
        onError={() => setFailed(true)}
        className="h-full w-full object-cover"
      />
    </button>
  );
}

export default function InsightsPanel({ model, worldId, open, onOpenChange, onSelectEntity, onSelectEvidence, onOpenEvidence }: InsightsPanelProps) {
  const status = insightsStatus(model);
  const photos = model.photoIds.slice(0, MAX_THUMBS);
  const more = model.photoIds.length - photos.length;

  const warningCount = model.cards.filter((c) => c.tone === "warning").length;
  const summary = warningCount > 0 ? `${warningCount} to review` : status === "clear" ? "Nothing needs attention" : model.hasInsights ? `${model.cards.length} found` : "None yet";

  const renderCard = (c: InsightCard) => {
    const warn = c.tone === "warning";
    const Icon = warn ? AlertTriangle : Info;
    return (
      <button
        key={c.id}
        type="button"
        onClick={() => c.entityIds[0] && onSelectEntity(c.entityIds[0])}
        title={c.detail}
        data-testid={`insight-${c.id}`}
        className={cn(
          "flex h-8 shrink-0 items-center gap-1.5 rounded border px-2.5 text-xs cursor-pointer",
          warn
            ? "border-[var(--warning)] bg-[var(--warning-subtle)] text-[var(--warning)]"
            : "border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]",
        )}
      >
        {c.rgb ? (
          <span aria-hidden className="h-3 w-3 rounded-sm border border-[var(--border)]" style={{ backgroundColor: `rgb(${c.rgb.join(",")})` }} />
        ) : (
          <Icon className="h-3.5 w-3.5 shrink-0" />
        )}
        <span className="whitespace-nowrap">{c.label}</span>
      </button>
    );
  };

  return (
    <section aria-label="Reconstruction insights" data-testid="insights-panel" className="shrink-0 border-t border-[var(--border)] bg-[var(--bg-sidebar)]">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
        data-testid="insights-toggle"
        className="flex h-8 w-full items-center gap-2 px-3 text-left cursor-pointer"
      >
        <span className="text-xs font-semibold text-[var(--text-primary)]">Reconstruction insights</span>
        <span className={cn("text-[11px]", warningCount > 0 ? "text-[var(--warning)]" : "text-[var(--text-tertiary)]")}>{summary}</span>
        <span className="ml-auto text-[var(--text-tertiary)]">{open ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}</span>
      </button>
      {open && (
        <div className="flex items-center gap-2 overflow-x-auto px-3 pb-2.5" data-testid="insights-body">
          {model.cards.map(renderCard)}
          {photos.length > 0 && (
            <div className="flex shrink-0 items-center gap-1.5 border-l border-[var(--border-subtle)] pl-2" aria-label="Supporting photographs">
              {photos.map((id) => (
                <Thumb key={id} worldId={worldId} id={id} onClick={() => onSelectEvidence(id)} />
              ))}
              {more > 0 && (
                <button
                  type="button"
                  onClick={onOpenEvidence}
                  data-testid="insights-more"
                  className="h-10 shrink-0 rounded border border-[var(--border)] px-2 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  +{more} more
                </button>
              )}
            </div>
          )}
          {!model.hasInsights && <span className="text-xs text-[var(--text-tertiary)]">Nothing has been detected in this model yet.</span>}
          {model.hasInsights && model.cards.length === 0 && status === "clear" && <span className="text-xs text-[var(--text-tertiary)]">Nothing needs attention.</span>}
        </div>
      )}
    </section>
  );
}
