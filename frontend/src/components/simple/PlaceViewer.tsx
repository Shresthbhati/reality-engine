"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2, Maximize2, Palette, ShieldCheck, Wrench, X } from "lucide-react";
import World3DViewport from "@/components/viewport/World3DViewport";
import {
  apiBaseUrl,
  getWorldQuality,
  useApi,
  useWorldCameras,
  useWorldIR,
  useWorldPoints,
  useWorldStatus,
  type Band,
} from "@/lib/api";
import { fetchEntityProvenance } from "@/lib/api/provenance";
import { PROVENANCE, QUALITY, SCALE, VIEWER, type ProvenanceKey } from "@/lib/copy";
import { parseMeshPly, parsePly } from "@/lib/viewport/loaders";
import { setProMode } from "@/lib/prefs";
import { entityLabel, statusChip, traceMessage, type StatusChipKind } from "@/lib/places";
import { BAND_COLOR, BAND_MARK } from "./bandStyle";
import QualityCard from "./QualityCard";

const CHIP_COLOR: Record<StatusChipKind, string> = {
  building: "var(--accent)",
  unplaced: "var(--warning)",
  size: "var(--warning)",
  weak: "var(--error)",
  ok: "var(--success)",
};

/** Scene defaults for Simple mode: the model and its parts, without the camera frusta an engineer wants. */
const SIMPLE_LAYERS = { cameras: false, topology: false, oversized: false } as const;

/**
 * The place viewer (Simple mode): the model is the whole screen. One status chip says the most important thing,
 * a small toolbar resets the view and shows quality colours, and selecting a part opens a drawer with what it is,
 * how sure the engine is, and which photos it came from. Engineer overlays stay in Pro mode.
 */
export default function PlaceViewer({ worldId }: { worldId: string }) {
  const { status, error: statusError, refresh } = useWorldStatus(worldId);
  const modelKey = status?.model?.version_id ?? null;
  const hasModel = Boolean(status?.has_model);

  const { data: worldIR, isLoading: irLoading, error: irError, refetch } = useWorldIR(worldId, null, modelKey);
  const { data: pointsBuffer } = useWorldPoints(worldId, null, modelKey);
  const { data: cameras } = useWorldCameras(worldId, null, modelKey);
  const { data: quality } = useApi(() => (hasModel ? getWorldQuality(worldId) : Promise.resolve(null)), [worldId, hasModel, modelKey]);

  const points = useMemo(() => {
    if (!pointsBuffer) return null;
    try {
      return parsePly(new Uint8Array(pointsBuffer));
    } catch {
      return null;
    }
  }, [pointsBuffer]);
  const mesh = useMemo(() => {
    if (!pointsBuffer) return null;
    try {
      return parseMeshPly(new Uint8Array(pointsBuffer));
    } catch {
      return null;
    }
  }, [pointsBuffer]);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [overlayOn, setOverlayOn] = useState(false);
  const [cardOpen, setCardOpen] = useState(false);
  const [frameSignal, setFrameSignal] = useState(0);

  const bandOverlay = useMemo<Record<string, string> | null>(() => {
    if (!overlayOn || !quality) return null;
    return Object.fromEntries(Object.entries(quality.entities).map(([id, e]) => [id, e.band]));
  }, [overlayOn, quality]);

  const { data: trace } = useApi(
    () => (selectedId ? fetchEntityProvenance(worldId, selectedId) : Promise.resolve(null)),
    [worldId, selectedId],
  );

  const selected = selectedId && quality ? quality.entities[selectedId] ?? null : null;
  const placedEvidence = useMemo(() => new Set((cameras?.cameras ?? []).map((c) => c.evidence_id).filter(Boolean)), [cameras]);

  const chip = quality
    ? statusChip({
        building: Boolean(status?.in_progress),
        placed: quality.photos_placed,
        total: quality.photos_total,
        scaleState: quality.scale_state,
        weakCount: quality.entity_counts.weak,
      })
    : status?.in_progress
      ? statusChip({ building: true, placed: 0, total: 0, scaleState: null, weakCount: 0 })
      : null;

  const flyToPhoto = useCallback((evidenceId: string) => {
    window.dispatchEvent(new CustomEvent("frame-camera", { detail: { id: evidenceId } }));
  }, []);

  const loadFailed = (irError || statusError) && !worldIR;
  const noModel = status && !status.has_model && !status.in_progress;

  return (
    <div className="flex h-full w-full flex-col" style={{ background: "var(--bg-base)" }}>
      <header
        className="flex h-14 shrink-0 items-center gap-3 px-4"
        style={{ background: "var(--bg-surface)", borderBottom: "1px solid var(--border)" }}
      >
        <Link
          href="/"
          className="flex h-9 items-center gap-1.5 rounded-md pr-2 text-sm pointer-coarse:h-11"
          style={{ color: "var(--text-secondary)" }}
        >
          <ArrowLeft className="h-4 w-4" aria-hidden /> {VIEWER.back}
        </Link>
        <h1 className="min-w-0 flex-1 truncate text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          {status?.name ?? ""}
        </h1>
        <button
          type="button"
          onClick={() => setProMode(true)}
          className="btn-press flex h-9 items-center gap-1.5 rounded-md px-3 text-xs pointer-coarse:h-11"
          style={{ color: "var(--text-secondary)", border: "1px solid var(--border-strong)" }}
        >
          <Wrench className="h-3.5 w-3.5" aria-hidden /> {VIEWER.openPro}
        </button>
      </header>

      <div className="relative min-h-0 flex-1">
        <World3DViewport
          world={worldIR ?? null}
          points={points}
          cameras={cameras ?? null}
          mesh={mesh}
          selectedEntityId={selectedId}
          onSelectEntity={setSelectedId}
          isLoading={!status || irLoading}
          loadingMessage={VIEWER.loading}
          chrome={false}
          initialLayers={SIMPLE_LAYERS}
          bandOverlay={bandOverlay}
          frameSignal={frameSignal}
        />

        {chip && (
          <div className="absolute left-3 top-3 z-10 flex flex-col items-start gap-2">
            <button
              type="button"
              disabled={!quality}
              aria-expanded={cardOpen}
              onClick={() => setCardOpen((v) => !v)}
              className="btn-press flex h-9 items-center gap-2 rounded-full px-3 text-xs font-medium pointer-coarse:h-11"
              style={{
                background: "color-mix(in srgb, var(--bg-surface) 92%, transparent)",
                border: "1px solid " + CHIP_COLOR[chip.kind],
                color: "var(--text-primary)",
              }}
            >
              {chip.kind === "building" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: CHIP_COLOR[chip.kind] }} aria-hidden />
              ) : (
                <ShieldCheck className="h-3.5 w-3.5" style={{ color: CHIP_COLOR[chip.kind] }} aria-hidden />
              )}
              {chip.text}
            </button>
            {cardOpen && quality && (
              <QualityCard
                quality={quality}
                overlayOn={overlayOn}
                onToggleOverlay={() => setOverlayOn((v) => !v)}
                onClose={() => setCardOpen(false)}
              />
            )}
          </div>
        )}

        {hasModel && (
          <div
            role="toolbar"
            aria-label="View"
            className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-full p-1"
            style={{ background: "color-mix(in srgb, var(--bg-surface) 92%, transparent)", border: "1px solid var(--border)" }}
          >
            <ToolButton icon={<Maximize2 className="h-4 w-4" aria-hidden />} label={VIEWER.resetView} onClick={() => setFrameSignal((n) => n + 1)} />
            <ToolButton
              icon={<Palette className="h-4 w-4" aria-hidden />}
              label={overlayOn ? VIEWER.qualityHide : VIEWER.qualityShow}
              pressed={overlayOn}
              disabled={!quality}
              onClick={() => setOverlayOn((v) => !v)}
            />
          </div>
        )}

        {overlayOn && (
          <ul
            aria-label={VIEWER.legend}
            className="absolute bottom-4 left-3 z-10 flex flex-col gap-1 rounded-md px-3 py-2 text-xs"
            style={{ background: "color-mix(in srgb, var(--bg-surface) 92%, transparent)", border: "1px solid var(--border)" }}
          >
            {(["good", "fair", "weak", "unknown"] as Band[]).map((b) => (
              <li key={b} className="flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
                <span aria-hidden style={{ color: BAND_COLOR[b] }}>
                  {BAND_MARK[b]}
                </span>
                {QUALITY[b]}
              </li>
            ))}
          </ul>
        )}

        {selectedId && selected && (
          <aside
            aria-label={VIEWER.selected}
            className="absolute z-10 flex flex-col gap-3 overflow-y-auto rounded-lg p-4 shadow-lg max-sm:inset-x-2 max-sm:bottom-24 max-sm:max-h-[50%] sm:bottom-16 sm:right-3 sm:top-3 sm:w-80"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {entityLabel(selected.type)}
                </h2>
                <p className="mt-1 flex items-center gap-1.5 text-xs font-medium" style={{ color: BAND_COLOR[selected.band] }}>
                  <span aria-hidden>{BAND_MARK[selected.band]}</span>
                  {QUALITY[selected.band]}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedId(null)}
                aria-label={VIEWER.closeDrawer}
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md pointer-coarse:h-11 pointer-coarse:w-11"
                style={{ color: "var(--text-tertiary)" }}
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>

            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {PROVENANCE[(selected.provenance as ProvenanceKey) in PROVENANCE ? (selected.provenance as ProvenanceKey) : "UNKNOWN"]}
            </p>

            <div>
              <h3 className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
                {VIEWER.builtFrom}
              </h3>
              <p className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
                {traceMessage(trace?.trace_level ?? null)}
              </p>
              {trace && trace.evidence.length > 0 && (
                <ul className="mt-2 grid grid-cols-3 gap-1.5">
                  {trace.evidence.slice(0, 6).map((ev) => {
                    const placed = placedEvidence.has(ev.evidence_id);
                    return (
                      <li key={ev.evidence_id}>
                        <button
                          type="button"
                          disabled={!placed}
                          onClick={() => flyToPhoto(ev.evidence_id)}
                          title={placed ? ev.evidence_name : ev.evidence_name + ": not placed in the model"}
                          className="block w-full overflow-hidden rounded disabled:cursor-default disabled:opacity-50"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={apiBaseUrl() + "/api/evidence/" + encodeURIComponent(ev.evidence_id) + "/artifact"}
                            alt={ev.evidence_name}
                            loading="lazy"
                            className="photo-outline aspect-square w-full object-cover"
                          />
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
              {quality?.scale_state === "relative" && (
                <p className="mt-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
                  {SCALE.relative}
                </p>
              )}
            </div>
          </aside>
        )}

        {loadFailed && (
          <CenterMessage
            title={VIEWER.loadError}
            actionLabel="Try again"
            onAction={() => {
              refresh();
              refetch();
            }}
          >
            {statusError?.message ?? irError?.message}
          </CenterMessage>
        )}
        {noModel && !loadFailed && <CenterMessage title={VIEWER.noModelTitle}>{VIEWER.noModelBody}</CenterMessage>}
      </div>
    </div>
  );
}

function ToolButton({
  icon,
  label,
  onClick,
  pressed,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  pressed?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={pressed}
      title={label}
      className="btn-press flex h-9 w-9 items-center justify-center rounded-full disabled:opacity-40 pointer-coarse:h-11 pointer-coarse:w-11"
      style={{
        color: pressed ? "var(--accent)" : "var(--text-primary)",
        background: pressed ? "var(--accent-subtle)" : "transparent",
      }}
    >
      {icon}
    </button>
  );
}

function CenterMessage({
  title,
  children,
  actionLabel,
  onAction,
}: {
  title: string;
  children?: React.ReactNode;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center p-6" style={{ background: "color-mix(in srgb, var(--bg-base) 88%, transparent)" }}>
      <div className="max-w-sm text-center">
        <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          {title}
        </h2>
        {children && (
          <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
            {children}
          </p>
        )}
        {actionLabel && onAction && (
          <button type="button" onClick={onAction} className="btn-press mt-4 h-9 rounded-md px-4 text-sm font-semibold" style={{ background: "var(--accent)", color: "var(--bg-base)" }}>
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}
