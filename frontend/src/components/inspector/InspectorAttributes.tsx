"use client";

import { useId, useState } from "react";
import { Info, Tag, CheckCircle2, HelpCircle, User } from "lucide-react";
import { Disclosure, SectionTitle } from "./InspectorParts";

export interface AttributeEntry {
  key: string;
  label: string;
  value: string | number | boolean | null;
  basis: "visual" | "user_stated" | "inferred" | "unknown";
  confidence?: number | null;
  cropUrl?: string | null;
  notes?: string | null;
}

interface InspectorAttributesProps {
  attributes: Record<string, unknown> | null | undefined;
  testId?: string;
}

const BASIS_BADGES: Record<
  AttributeEntry["basis"],
  { label: string; bg: string; text: string; icon: typeof Tag }
> = {
  visual: {
    label: "Observed",
    bg: "bg-[var(--accent-subtle)]",
    text: "text-[var(--accent)]",
    icon: CheckCircle2,
  },
  user_stated: {
    label: "User stated",
    bg: "bg-[var(--info-subtle)]",
    text: "text-[var(--info)]",
    icon: User,
  },
  inferred: {
    label: "Inferred",
    bg: "bg-[var(--warning-subtle)]",
    text: "text-[var(--warning)]",
    icon: Tag,
  },
  unknown: {
    label: "Not seen",
    bg: "bg-[var(--border-subtle)]",
    text: "text-[var(--text-tertiary)]",
    icon: HelpCircle,
  },
};

function parseAttributes(raw: Record<string, unknown> | null | undefined): AttributeEntry[] {
  if (!raw || typeof raw !== "object") return [];
  const entries: AttributeEntry[] = [];

  for (const [key, val] of Object.entries(raw)) {
    if (key.startsWith("_") || key === "detail_cells") continue;
    const label = key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

    if (val && typeof val === "object" && !Array.isArray(val)) {
      const obj = val as Record<string, unknown>;
      const v = obj.value !== undefined ? (obj.value as string | number | boolean | null) : null;
      const basis = (obj.basis as AttributeEntry["basis"]) || (v ? "visual" : "unknown");
      entries.push({
        key,
        label,
        value: v,
        basis: v === null || v === "unknown" ? "unknown" : basis,
        confidence: typeof obj.p === "number" ? obj.p : typeof obj.confidence === "number" ? obj.confidence : null,
        cropUrl: typeof obj.crop_url === "string" ? obj.crop_url : typeof obj.image_crop === "string" ? obj.image_crop : null,
        notes: typeof obj.notes === "string" ? obj.notes : null,
      });
    } else {
      const v = (val as string | number | boolean | null) ?? null;
      entries.push({
        key,
        label,
        value: v,
        basis: v === null || v === "unknown" ? "unknown" : "visual",
      });
    }
  }

  return entries.sort((a, b) => a.label.localeCompare(b.label));
}

export function InspectorAttributes({ attributes, testId = "inspector-attributes" }: InspectorAttributesProps) {
  const entries = parseAttributes(attributes);

  if (entries.length === 0) {
    return (
      <Disclosure title="Attributes" testId={testId}>
        <p className="text-[12px] leading-relaxed text-[var(--text-tertiary)]">
          No attributes identified for this entity.
        </p>
      </Disclosure>
    );
  }

  return (
    <Disclosure title={`Attributes (${entries.length})`} testId={testId}>
      <ul className="divide-y divide-[var(--border-subtle)]">
        {entries.map((attr) => {
          const badge = BASIS_BADGES[attr.basis] || BASIS_BADGES.unknown;
          const BadgeIcon = badge.icon;
          const isUnknown = attr.value === null || attr.value === undefined || attr.value === "unknown";

          return (
            <li key={attr.key} className="py-2.5 space-y-1.5 text-[12px]">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-[var(--text-secondary)]">{attr.label}</span>
                <span
                  className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${badge.bg} ${badge.text}`}
                  title={`Evidence basis: ${attr.basis}`}
                >
                  <BadgeIcon className="h-3 w-3" aria-hidden />
                  {badge.label}
                </span>
              </div>

              <div className="flex items-baseline justify-between gap-3">
                {isUnknown ? (
                  <span className="text-[11px] italic text-[var(--text-tertiary)]">
                    Unknown — not visible in any photo
                  </span>
                ) : (
                  <span className="font-mono-num font-medium text-[var(--text-primary)]">
                    {String(attr.value)}
                  </span>
                )}

                {attr.confidence !== null && attr.confidence !== undefined && (
                  <span className="text-[10px] text-[var(--text-tertiary)] font-mono-num">
                    {(attr.confidence * 100).toFixed(0)}% conf
                  </span>
                )}
              </div>

              {attr.cropUrl && (
                <div className="pt-1">
                  <img
                    src={attr.cropUrl}
                    alt={`${attr.label} evidence crop`}
                    className="h-12 w-auto max-w-[120px] rounded border border-[var(--border)] object-cover"
                  />
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </Disclosure>
  );
}

export default InspectorAttributes;
