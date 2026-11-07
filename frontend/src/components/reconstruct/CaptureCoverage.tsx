"use client";

import React, { useCallback, useEffect, useId, useRef, useState } from "react";
import { Camera } from "lucide-react";
import {
  STATUS_LABEL,
  STATUS_STYLE,
  coverageHeadline,
  getCaptureCoverage,
  mapGeometry,
  type CaptureCoverageDto,
} from "@/lib/api/capture-coverage";
import { createReconstruction } from "@/lib/api/reconstructions";

export interface CaptureCoverageProps {
  worldId: string;
  /** change this (e.g. the current version id) to re-fetch after a rebuild */
  refreshKey?: string | number | null;
  /** called after extra photos were accepted; the owner should refresh the reconstruction status */
  onUploaded?: () => void;
}

/**
 * Top-down map of which walls the photos covered (seen / seen weakly / not seen) with "take N more photos"
 * hints. Walls are numbered around the footprint, never by compass. Extra photos go through the existing
 * refining path (POST /api/reconstructions with this world_id): same world, rebuilt from all evidence.
 */
export default function CaptureCoverage({ worldId, refreshKey, onUploaded }: CaptureCoverageProps) {
  const [data, setData] = useState<CaptureCoverageDto | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const inputId = useId();

  useEffect(() => {
    let live = true;
    getCaptureCoverage(worldId).then(
      (r) => live && (setData(r), setError(null)),
      (e) => live && setError(e instanceof Error ? e.message : "Could not load wall coverage."),
    );
    return () => {
      live = false;
    };
  }, [worldId, refreshKey]);

  const upload = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setBusy(true);
      setMessage(null);
      try {
        const r = (await createReconstruction(Array.from(files), { worldId })) as Awaited<ReturnType<typeof createReconstruction>> & {
          capture_quality?: { summary: { warn: number; will_be_dropped: number; headline: string } } | null;
        };
        const q = r.capture_quality?.summary;
        setMessage(
          `${r.evidence.length} file(s) added; the model is being refined.` +
            (q && (q.warn > 0 || q.will_be_dropped > 0) ? ` ${q.headline}` : ""),
        );
        onUploaded?.();
      } catch (e) {
        setMessage(e instanceof Error ? e.message : "Upload failed. Nothing was changed.");
      } finally {
        setBusy(false);
        if (input.current) input.current.value = "";
      }
    },
    [worldId, onUploaded],
  );

  if (error) {
    return (
      <p className="text-[11px] text-error" role="alert" data-testid="coverage-error">
        {error}
      </p>
    );
  }
  if (!data) return <p className="text-[11px] text-fg-muted">Checking wall coverage…</p>;

  const geo = mapGeometry(data);
  return (
    <section className="rounded border border-[var(--border)] p-2" aria-label="Wall coverage" data-testid="capture-coverage">
      <h3 className="text-[11px] font-semibold text-fg" data-testid="coverage-headline">
        {coverageHeadline(data)}
      </h3>
      {geo && (
        <svg
          viewBox={`0 0 ${geo.width} ${geo.height}`}
          className="mt-1 w-full max-w-[360px]"
          role="img"
          aria-label={`Top-down map: ${coverageHeadline(data)}`}
        >
          {geo.walls.map((w) => {
            const st = STATUS_STYLE[w.status];
            return (
              <g key={w.index} data-testid={`coverage-wall-${w.index}`} data-status={w.status}>
                <title>{`${w.label}: ${STATUS_LABEL[w.status]}`}</title>
                <line x1={w.x1} y1={w.y1} x2={w.x2} y2={w.y2} stroke={st.stroke} strokeWidth={4} strokeDasharray={st.dash} strokeLinecap="round" />
                <text x={w.mx} y={w.my} fontSize={10} fill="currentColor" textAnchor="middle" dy={-6} className="text-fg-muted">
                  {w.index}
                </text>
              </g>
            );
          })}
          {geo.corners.map((c) => (
            <text key={c.label} x={c.x} y={c.y} fontSize={9} fill="currentColor" textAnchor="middle" dy={3} className="text-fg-subtle">
              {c.label}
            </text>
          ))}
          {geo.cameras.map((c, i) => (
            <g key={i}>
              <circle cx={c.x} cy={c.y} r={3} style={{ fill: "var(--info)" }} />
              <line x1={c.x} y1={c.y} x2={c.tx} y2={c.ty} style={{ stroke: "var(--info)" }} strokeWidth={1.5} />
            </g>
          ))}
        </svg>
      )}
      {geo && (
        <ul className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-fg-muted" aria-label="Legend">
          {(["seen", "weak", "not_seen"] as const).map((k) => (
            <li key={k} className="flex items-center gap-1">
              <svg width="18" height="6" aria-hidden>
                <line x1="1" y1="3" x2="17" y2="3" stroke={STATUS_STYLE[k].stroke} strokeWidth="3" strokeDasharray={STATUS_STYLE[k].dash} />
              </svg>
              {STATUS_LABEL[k]}
            </li>
          ))}
        </ul>
      )}
      {data.hints.length > 0 && (
        <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[11px] text-fg" data-testid="coverage-hints">
          {data.hints.map((h) => (
            <li key={h}>{h}</li>
          ))}
        </ul>
      )}
      {data.available && (data.notes ?? []).map((n) => (
        <p key={n} className="mt-1 text-[10px] text-fg-subtle">
          {n}
        </p>
      ))}
      {data.available && (
        <div className="mt-2">
          <input id={inputId} ref={input} type="file" accept="image/*,video/*" multiple className="sr-only" onChange={(e) => void upload(e.target.files)} />
          <label
            htmlFor={inputId}
            className={`inline-flex cursor-pointer items-center gap-1.5 rounded border border-[var(--border)] px-2 py-0.5 text-[11px] text-fg hover:bg-hover focus-within:ring-1 ${busy ? "pointer-events-none opacity-50" : ""}`}
          >
            <Camera className="h-3.5 w-3.5" aria-hidden />
            {busy ? "Adding…" : "Add photos to refine this model"}
          </label>
        </div>
      )}
      {message && (
        <p className="mt-1 text-[10px] text-fg-muted" role="status">
          {message}
        </p>
      )}
    </section>
  );
}
