"use client";

import React, { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Lightbulb, Ruler } from "lucide-react";
import CaptureCoverage from "@/components/reconstruct/CaptureCoverage";
import { getCaptureCoverage, type CaptureCoverageDto } from "@/lib/api/capture-coverage";
import { buildRoomSummary, type SummaryExtras } from "@/lib/reconstruct/summary-model";
import { captureBadge } from "@/lib/viewport/structure-points-model";
import type { WorldIR } from "@/types/worldir";
import { cn } from "@/lib/utils";

export interface RoomSummaryCardProps extends Omit<SummaryExtras, "coverage"> {
  world: WorldIR | null | undefined;
  /** the world whose CURRENT model's wall coverage is shown; omit when inspecting an older version */
  worldId?: string | null;
  /** change (e.g. the current version id) to re-read coverage after a rebuild */
  refreshKey?: string | number | null;
  /** extra photos were added from the coverage map */
  onPhotosAdded?: () => void;
}

const STATE_STYLE: Record<string, string> = {
  MEASURED: "border-emerald-500/40 text-emerald-300",
  ESTIMATED: "border-amber-400/50 text-amber-300",
  INFERRED: "border-sky-400/40 text-sky-300",
};

const VERDICT_STYLE = {
  sufficient: "border-emerald-500/40 text-emerald-300",
  partial: "border-amber-400/50 text-amber-300",
  insufficient: "border-red-400/60 text-red-300",
} as const;

/**
 * Results summary for one world, read from its WorldIR. A relative world shows "model units", never "m".
 *
 * Wall coverage comes from GET /api/worlds/{id}/capture-coverage (not-seen / weak counts and its hints); the full
 * top-down map (CaptureCoverage) opens under "Wall coverage map". If the endpoint has nothing, the card falls back to
 * what the WorldIR itself says (UNOBSERVED walls) and to unplaced photos.
 */
export default function RoomSummaryCard({ world, worldId, refreshKey, onPhotosAdded, scale, imagesUsed, imagesRegistered, captureQuality }: RoomSummaryCardProps) {
  const [dto, setDto] = useState<CaptureCoverageDto | null>(null);
  const [mapOpen, setMapOpen] = useState(false);
  useEffect(() => {
    if (!worldId) return;
    let live = true;
    getCaptureCoverage(worldId).then(
      (r) => live && setDto(r),
      () => live && setDto(null),
    );
    return () => {
      live = false;
    };
  }, [worldId, refreshKey]);
  const coverage = useMemo(
    () =>
      worldId && dto?.available
        ? { unobserved_walls: dto.summary?.not_seen ?? 0, weak_walls: dto.summary?.weak ?? 0, hint: dto.hints[0] ?? null }
        : null,
    [dto, worldId],
  );
  const s = useMemo(
    () => buildRoomSummary(world, { scale, imagesUsed, imagesRegistered, coverage, captureQuality }),
    [world, scale, imagesUsed, imagesRegistered, coverage, captureQuality],
  );
  if (!s) return null;

  return (
    <section aria-label="Room summary" data-testid="room-summary" className="px-3 py-2">
      <div className="flex flex-wrap items-start gap-x-6 gap-y-2">
        {s.dimensions.length > 0 ? (
          <dl className="flex flex-wrap gap-x-4 gap-y-1" data-testid="summary-dimensions">
            {s.dimensions.map((d) => (
              <div key={d.key} data-testid={`summary-dim-${d.key}`} className="min-w-0">
                <dt className="text-[10px] uppercase tracking-wider text-neutral-500">{d.label}</dt>
                <dd className="flex items-baseline gap-1.5 text-[12px] text-neutral-100">
                  {d.value === null ? (
                    <span className="text-neutral-500">not available</span>
                  ) : (
                    <>
                      <span className="font-semibold tabular-nums">{d.value}</span>
                      <span className="text-[10px] text-neutral-400">{d.unit}</span>
                      {d.state && (
                        <span
                          data-testid={`summary-dim-state-${d.key}`}
                          className={cn("rounded border px-1 text-[9px] font-semibold", STATE_STYLE[d.state] ?? "border-neutral-600 text-neutral-300")}
                        >
                          {d.state}
                        </span>
                      )}
                    </>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-[11px] text-neutral-400" data-testid="summary-no-room">
            No room outline measured yet: dimensions are not available.
          </p>
        )}

        <ul className="flex flex-wrap items-center gap-1.5 text-[11px]" aria-label="What was found" data-testid="summary-counts">
          {(
            [
              ["walls", s.counts.walls, undefined],
              ["doors", s.counts.doors, undefined],
              ["windows", s.counts.windows, undefined],
              ["furniture", s.counts.furniture, "Furniture candidates are unlabelled objects the engine found resting on the floor."],
            ] as const
          ).map(([name, n, tip]) => (
            <li
              key={name}
              data-testid={`summary-count-${name}`}
              className="rounded border border-[var(--border)] px-1.5 py-0.5 text-neutral-200"
              title={s.countsReliable ? tip : "Not counted: the photos could not be placed in one 3D space, so nothing here describes the room."}
            >
              {s.countsReliable ? n : "—"} {name}
            </li>
          ))}
        </ul>

        <p
          data-testid="summary-scale"
          data-scale-kind={s.scaleKind}
          className={cn(
            "flex items-center gap-1 text-[11px]",
            s.scaleKind === "measured" ? "text-emerald-300" : s.scaleKind === "estimated" ? "text-amber-300" : "text-neutral-400",
          )}
        >
          <Ruler className="h-3.5 w-3.5" aria-hidden />
          {s.scaleLabel}
        </p>
      </div>

      {s.unitNote && (
        <p className="mt-1 text-[11px] text-amber-300" data-testid="summary-unit-note">
          {s.unitNote}
        </p>
      )}

      {s.captureQuality && (
        <div
          data-testid="summary-capture-verdict"
          data-state={s.captureQuality.state}
          role={s.captureQuality.state === "insufficient" ? "alert" : "status"}
          className="mt-1.5 flex flex-wrap items-start gap-x-2 gap-y-0.5 text-[11px]"
        >
          <span className={cn("rounded border px-1.5 py-0 text-[10px] font-semibold uppercase tracking-wide", VERDICT_STYLE[s.captureQuality.state])}>
            {captureBadge(s.captureQuality)}
          </span>
          {s.captureQuality.message && <span className="min-w-0 flex-1 text-neutral-200">{s.captureQuality.message}</span>}
        </div>
      )}

      {s.warnings.length > 0 && (
        <ul className="mt-1.5 space-y-0.5" data-testid="summary-warnings">
          {s.warnings.map((w) => (
            <li key={w} className="flex items-start gap-1.5 text-[11px] text-amber-300">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
              {w}
            </li>
          ))}
        </ul>
      )}
      {s.captureHint && (
        <p className="mt-1 flex items-start gap-1.5 text-[11px] text-neutral-300" data-testid="summary-capture-hint">
          <Lightbulb className="mt-0.5 h-3 w-3 shrink-0 text-[var(--accent)]" aria-hidden />
          <span>
            <strong className="text-neutral-100">Capture next:</strong> {s.captureHint}
          </span>
        </p>
      )}
      {worldId && dto?.available && (
        <details className="mt-1.5" onToggle={(e) => setMapOpen((e.currentTarget as HTMLDetailsElement).open)}>
          <summary className="cursor-pointer text-[11px] text-neutral-400 hover:text-neutral-200" data-testid="coverage-toggle">
            Wall coverage map
          </summary>
          {mapOpen && (
            <div className="mt-1">
              <CaptureCoverage worldId={worldId} refreshKey={refreshKey} onUploaded={onPhotosAdded} />
            </div>
          )}
        </details>
      )}
    </section>
  );
}
