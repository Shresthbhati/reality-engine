"use client";

import { Box, Clock, FileSearch, Layers, Ruler, Target } from "lucide-react";
import type { ComponentType } from "react";
import { CAPABILITIES, type CapabilityId } from "@/lib/intent/model";

const ICONS: Record<CapabilityId, ComponentType<{ className?: string }>> = {
  geometry: Box,
  semantic_understanding: Layers,
  measurement: Ruler,
  evidence_provenance: FileSearch,
  temporal_state: Clock,
  high_precision: Target,
};

export default function CapabilityList({
  value,
  onToggle,
  disabled = false,
}: {
  value: readonly CapabilityId[];
  onToggle: (id: CapabilityId) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0" disabled={disabled}>
      <legend className="sr-only">What level of reconstruction is required?</legend>
      <div className="grid grid-cols-2 gap-2.5 max-[1023px]:grid-cols-1">
        {CAPABILITIES.map((c) => {
          const Icon = ICONS[c.id];
          const checked = value.includes(c.id);
          return (
            <label
              key={c.id}
              className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2"
              style={{
                background: "var(--bg-surface)",
                border: `1px solid ${checked ? "var(--accent-border)" : "var(--border)"}`,
                outlineColor: "var(--accent)",
              }}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(c.id)}
                className="h-4 w-4 shrink-0"
                style={{ accentColor: "var(--accent)" }}
              />
              <span
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md"
                style={{ background: "var(--bg-elevated)", color: "var(--accent)" }}
                aria-hidden
              >
                <Icon className="h-5 w-5" />
              </span>
              <span className="flex min-w-0 flex-col">
                <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {c.label}
                </span>
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {c.blurb}
                </span>
                <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                  {c.example}
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
