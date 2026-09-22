import Link from "next/link";
import { Image as ImageIcon, Video, Box, FileText, Radio, type LucideIcon } from "lucide-react";
import type { EvidenceRow, EvidenceType, EvidenceProcessingState } from "@/lib/types";

export const EVIDENCE_TYPE_ICONS: Record<EvidenceType, LucideIcon> = {
  IMAGE: ImageIcon,
  VIDEO: Video,
  POINT_CLOUD: Box,
  DOCUMENT: FileText,
  SENSOR_DATA: Radio,
};

export function humanizeEvidenceType(type: EvidenceType): string {
  return type
    .toLowerCase()
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

const PROCESSING_STYLES: Record<
  EvidenceProcessingState,
  { bg: string; fg: string; label: string; pulse?: boolean }
> = {
  UPLOADING: { bg: "var(--accent-subtle)", fg: "var(--accent)", label: "Uploading", pulse: true },
  PROCESSING: { bg: "var(--accent-subtle)", fg: "var(--accent)", label: "Processing", pulse: true },
  PROCESSED: { bg: "var(--success-subtle)", fg: "var(--success)", label: "Processed" },
  FAILED: { bg: "var(--error-subtle)", fg: "var(--error)", label: "Failed" },
};

// Local pill — EvidenceProcessingState is a different enum than StatusBadge's
// ProcessingState, so this intentionally doesn't reuse that shared component.
export function EvidenceStatusPill({ state }: { state: EvidenceProcessingState }) {
  const s = PROCESSING_STYLES[state];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 h-5 rounded text-xs font-medium shrink-0"
      style={{ background: s.bg, color: s.fg }}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${s.pulse ? "animate-pulse" : ""}`} style={{ background: s.fg }} />
      {s.label}
    </span>
  );
}

export default function EvidenceCard({ evidence }: { evidence: EvidenceRow }) {
  const Icon = EVIDENCE_TYPE_ICONS[evidence.type];

  return (
    <Link
      href={`/evidence/${evidence.id}`}
      className="flex flex-col rounded-lg overflow-hidden transition-colors"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="h-32 flex items-center justify-center shrink-0" style={{ background: "var(--bg-elevated)" }}>
        <Icon className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
      </div>

      <div className="flex flex-col gap-1 p-4">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>
            {evidence.name}
          </span>
          <EvidenceStatusPill state={evidence.processingState} />
        </div>

        <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
          {humanizeEvidenceType(evidence.type)}
        </span>

        {evidence.location && (
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {evidence.location}
          </span>
        )}

        {evidence.sessionName && (
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {evidence.sessionName}
          </span>
        )}

        {evidence.capturedAt && (
          <span className="text-xs font-mono-num mt-1" style={{ color: "var(--text-tertiary)" }}>
            Captured · {evidence.capturedAt}
          </span>
        )}
      </div>
    </Link>
  );
}
