"use client";

import { Building2, DoorOpen, Landmark, LayoutGrid, Map as MapIcon, MapPinned, Route } from "lucide-react";
import type { ComponentType } from "react";
import { SCALES, type ScaleId } from "@/lib/intent/model";

// Illustration tiles, not photographs: the product never shows imagery it did not capture.
const ICONS: Record<ScaleId, ComponentType<{ className?: string }>> = {
  interior_space: DoorOpen,
  building: Building2,
  street: Route,
  neighborhood: LayoutGrid,
  locality: MapPinned,
  district: MapIcon,
  city: Landmark,
};

export default function ScaleCards({
  value,
  onChange,
  disabled = false,
}: {
  value: ScaleId | null;
  onChange: (id: ScaleId) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0" disabled={disabled}>
      <legend className="sr-only">What are you trying to construct?</legend>
      <div className="grid grid-cols-7 gap-2.5 max-[1279px]:grid-cols-4 max-[1023px]:grid-cols-2">
        {SCALES.map((s, i) => {
          const Icon = ICONS[s.id];
          const selected = value === s.id;
          return (
            <label
              key={s.id}
              className="group relative flex cursor-pointer flex-col overflow-hidden rounded-lg transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2"
              style={{
                background: "var(--bg-surface)",
                border: `1px solid ${selected ? "var(--accent)" : "var(--border)"}`,
                boxShadow: selected ? "0 0 0 1px var(--accent-border)" : undefined,
                outlineColor: "var(--accent)",
              }}
            >
              <input
                type="radio"
                name="world-scale"
                value={s.id}
                checked={selected}
                // a radio group with no selection is tabbable on its first card
                tabIndex={value === null ? (i === 0 ? 0 : -1) : selected ? 0 : -1}
                onChange={() => onChange(s.id)}
                className="peer sr-only"
                aria-label={`${s.label}: ${s.blurb}`}
              />
              <span
                className="flex h-20 items-center justify-center"
                style={{ background: selected ? "var(--accent-subtle)" : "var(--bg-elevated)" }}
                aria-hidden
              >
                <Icon className="h-9 w-9" />
              </span>
              <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {s.label}
                </span>
                <span className="text-xs leading-snug" style={{ color: "var(--text-secondary)" }}>
                  {s.blurb}
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
