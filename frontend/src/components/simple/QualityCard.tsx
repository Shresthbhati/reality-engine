"use client";

import { X } from "lucide-react";
import type { WorldQuality } from "@/lib/api";
import { BAND_REASON, VIEWER, photosPlaced } from "@/lib/copy";
import { BAND_COLOR, BAND_MARK } from "./bandStyle";

/**
 * The place's quality in plain words: how many photos were placed, whether sizes are real, which parts are weak.
 * It states only what the quality endpoint reported. The one action is toggling the colours on the model.
 */
export default function QualityCard({
  quality,
  overlayOn,
  onToggleOverlay,
  onClose,
}: {
  quality: WorldQuality;
  overlayOn: boolean;
  onToggleOverlay: () => void;
  onClose: () => void;
}) {
  const weak = quality.entity_counts.weak;
  const unknown = quality.entity_counts.unknown;
  return (
    <section
      aria-label={VIEWER.cardTitle}
      className="w-80 max-w-[calc(100vw-2rem)] rounded-lg p-4 shadow-lg"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-start justify-between gap-2">
        <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          {VIEWER.cardTitle}
        </h2>
        <button
          type="button"
          onClick={onClose}
          aria-label={VIEWER.closeDrawer}
          className="flex h-7 w-7 items-center justify-center rounded-md pointer-coarse:h-11 pointer-coarse:w-11"
          style={{ color: "var(--text-tertiary)" }}
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
      </div>

      <p className="mt-1 flex items-center gap-1.5 text-xs font-medium" style={{ color: BAND_COLOR[quality.band] }}>
        <span aria-hidden>{BAND_MARK[quality.band]}</span>
        {quality.band === "unknown" ? VIEWER.cardUnknown : bandSentence(quality.band)}
      </p>

      {quality.band_reason && BAND_REASON[quality.band_reason] && (
        <p className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
          {BAND_REASON[quality.band_reason]}
        </p>
      )}

      <dl className="mt-3 flex flex-col gap-2.5 text-xs">
        <Row label={VIEWER.cardPhotos} value={photosPlaced(quality.photos_placed, quality.photos_total)} />
        <Row label={VIEWER.cardSize} value={quality.scale_state === "relative" ? VIEWER.cardSizeNo : VIEWER.cardSizeYes} />
        <Row label={VIEWER.cardWeak} value={weak > 0 ? String(weak) : VIEWER.cardWeakNone} />
        {unknown > 0 && <Row label={VIEWER.cardUnknown} value={String(unknown)} />}
        {quality.open_conflicts > 0 && <Row label={VIEWER.cardConflicts} value={String(quality.open_conflicts)} />}
      </dl>

      <button
        type="button"
        onClick={onToggleOverlay}
        aria-pressed={overlayOn}
        className="btn-press mt-4 h-9 w-full rounded-md text-sm font-semibold"
        style={{ background: "var(--accent)", color: "var(--bg-base)" }}
      >
        {overlayOn ? VIEWER.cardHide : VIEWER.cardShow}
      </button>
    </section>
  );
}

function bandSentence(band: WorldQuality["band"]): string {
  return band === "good" ? "Good overall" : band === "fair" ? "Fair overall" : "Weak overall";
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt style={{ color: "var(--text-tertiary)" }}>{label}</dt>
      <dd className="tabular-nums" style={{ color: "var(--text-primary)" }}>
        {value}
      </dd>
    </div>
  );
}
