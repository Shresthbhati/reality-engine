"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Ruler } from "lucide-react";
import {
  acceptScaleSuggestion,
  describeScale,
  getScaleSuggestion,
  rejectScaleSuggestion,
  suggestionSentence,
  type ScaleStageDto,
  type ScaleSuggestionResponse,
} from "@/lib/api/scale";

export interface ScaleCardProps {
  worldId: string;
  /** `status.model.scale` of the current version */
  scale: ScaleStageDto | null | undefined;
  /** called after Accept / Reject so the owner can refresh the reconstruction status */
  onChanged?: () => void;
}

/**
 * Scale call to action. RELATIVE world: "Dimensions are not in metres. Enter one known distance." and, when a
 * door was detected, a typical-size SUGGESTION the user must accept. Nothing is applied until Accept; a metric
 * world scaled that way is labelled ESTIMATED.
 */
export default function ScaleCard({ worldId, scale, onChanged }: ScaleCardProps) {
  const view = describeScale(scale);
  const [data, setData] = useState<ScaleSuggestionResponse | null>(null);
  const [busy, setBusy] = useState<"accept" | "reject" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const relative = view.kind === "relative";
  useEffect(() => {
    if (!relative) {
      // No-op when not relative
      return;
    }
    let live = true;
    getScaleSuggestion(worldId).then(
      (r) => live && setData(r),
      () => live && setData(null),
    );
    return () => {
      live = false;
    };
  }, [worldId, relative]);

  const run = useCallback(
    async (kind: "accept" | "reject") => {
      const s = data?.suggestion;
      if (!s) return;
      setBusy(kind);
      setError(null);
      try {
        if (kind === "accept") {
          await acceptScaleSuggestion(worldId, s);
          setMessage("Scale accepted. The model is being re-evaluated in metres; dimensions will be labelled ESTIMATED.");
        } else {
          await rejectScaleSuggestion(worldId, s.source_entity_id);
          setData({ ...data, rejected: true });
          setMessage("Suggestion dismissed. The model stays in model units.");
        }
        onChanged?.();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not apply that. Nothing was changed.");
      } finally {
        setBusy(null);
      }
    },
    [data, worldId, onChanged],
  );

  if (view.kind === "unknown") return null;

  if (view.kind === "estimated" || view.kind === "measured") {
    return (
      <p
        className={view.kind === "estimated" ? "mt-1 flex items-center gap-1.5 text-[11px] text-warning" : "mt-1 flex items-center gap-1.5 text-[11px] text-success"}
        data-testid="scale-state-label"
        data-scale-state={view.kind}
      >
        <Ruler className="h-3.5 w-3.5" aria-hidden />
        {view.label}
        {view.kind === "estimated" ? " · enter a real distance to replace this estimate" : ""}
      </p>
    );
  }

  const s = data?.suggestion && !data.rejected ? data.suggestion : null;
  return (
    <div className="mt-2 rounded border border-warning/30 bg-warning-subtle p-2" data-testid="scale-card">
      <p className="flex items-center gap-1.5 text-[11px] font-semibold text-warning" data-testid="scale-cta">
        <Ruler className="h-3.5 w-3.5" aria-hidden />
        {view.cta}
      </p>
      {s ? (
        <div className="mt-1.5" data-testid="scale-suggestion">
          <p className="text-[11px] leading-relaxed text-fg">{suggestionSentence(s)}</p>
          <p className="mt-0.5 text-[10px] text-fg-muted">
            Source: {s.source_kind} <code>{s.source_entity_id}</code> · implies {s.implied_scale.toFixed(3)} m per model unit ·
            dimensions would be labelled ESTIMATED, not measured.
          </p>
          <div className="mt-1.5 flex gap-2">
            <button
              type="button"
              onClick={() => void run("accept")}
              disabled={busy !== null}
              className="rounded border border-warning/60 px-2 py-0.5 text-[11px] font-medium text-warning hover:bg-warning-subtle disabled:opacity-50"
            >
              {busy === "accept" ? "Accepting…" : "Accept"}
            </button>
            <button
              type="button"
              onClick={() => void run("reject")}
              disabled={busy !== null}
              className="rounded border border-[var(--border)] px-2 py-0.5 text-[11px] text-fg-muted hover:bg-hover disabled:opacity-50"
            >
              {busy === "reject" ? "Dismissing…" : "Reject"}
            </button>
          </div>
        </div>
      ) : data && !data.rejected ? (
        <p className="mt-1 text-[10px] text-fg-muted" data-testid="scale-no-suggestion">
          No typical size to suggest: {data.reason ?? "no standard object was detected"}.
        </p>
      ) : null}
      {message && (
        <p className="mt-1 text-[10px] text-fg-muted" role="status">
          {message}
        </p>
      )}
      {error && (
        <p className="mt-1 text-[10px] text-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
