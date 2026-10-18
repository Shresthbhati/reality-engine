"use client";

import { Plus, Trash2 } from "lucide-react";
import {
  WALLS,
  newOpening,
  validateOpenings,
  wallLength,
  type OpeningDraft,
  type Wall,
} from "@/lib/procedural/generated-model";

const cell =
  "w-full bg-transparent outline-none text-xs px-1.5 h-7 rounded border border-[var(--border)] text-[var(--text-primary,#fff)]";

/** Door / window list for a room. Inline validation only; the backend stays authoritative. */
export default function OpeningsEditor({
  openings,
  room,
  onChange,
}: {
  openings: OpeningDraft[];
  room: { width: number; depth: number };
  onChange: (next: OpeningDraft[]) => void;
}) {
  const errors = validateOpenings(openings, room);
  const set = (i: number, patch: Partial<OpeningDraft>) =>
    onChange(openings.map((o, j) => (j === i ? { ...o, ...patch } : o)));

  return (
    <div className="flex flex-col gap-2" data-testid="openings-editor">
      <div className="flex items-center justify-between text-xs text-neutral-400">
        <span>Doors and windows ({openings.length})</span>
        <div className="flex gap-1.5">
          {(["door", "window"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => onChange([...openings, newOpening(k)])}
              className="flex items-center gap-1 px-2 h-6 rounded border border-[var(--border)] text-[11px] text-neutral-300 hover:text-white cursor-pointer"
            >
              <Plus className="w-3 h-3" /> {k}
            </button>
          ))}
        </div>
      </div>

      {openings.length === 0 && <p className="text-[11px] text-neutral-500">No openings: solid walls.</p>}

      {openings.map((o, i) => (
        <div key={i} className="rounded border border-[var(--border)] p-2 flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[11px] capitalize text-neutral-300 w-12">{o.kind}</span>
            <select
              aria-label={`Opening ${i + 1} wall`}
              value={o.wall}
              onChange={(e) => set(i, { wall: e.target.value as Wall })}
              className={cell}
            >
              {WALLS.map((w) => (
                <option key={w} value={w}>
                  {w} wall ({wallLength(w, room)} m)
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-label={`Remove opening ${i + 1}`}
              onClick={() => onChange(openings.filter((_, j) => j !== i))}
              className="cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5 text-neutral-400 hover:text-[var(--error)]" />
            </button>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {(
              [
                ["lateral_offset", "Offset"],
                ["width", "Width"],
                ["bottom", "Bottom"],
                ["top", "Top"],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="flex flex-col gap-0.5 text-[10px] text-neutral-500">
                {label} (m)
                <input
                  type="number"
                  step={0.1}
                  aria-label={`Opening ${i + 1} ${label.toLowerCase()}`}
                  value={Number.isFinite(o[key]) ? o[key] : ""}
                  onChange={(e) => set(i, { [key]: e.target.value === "" ? NaN : Number(e.target.value) })}
                  className={cell}
                />
              </label>
            ))}
          </div>
          {errors[i].map((m) => (
            <p key={m} role="alert" className="text-[11px] text-[var(--error)]">
              {m}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}
