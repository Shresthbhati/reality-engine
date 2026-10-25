"use client";

import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import {
  APARTMENT_KINDS,
  APARTMENT_TEMPLATES,
  nextRoomId,
  templateAdjacency,
  templateRows,
  type AdjacencyDraft,
  type AdjacencyLevel,
  type ApartmentKind,
  type RoomRow,
} from "@/lib/procedural/apartment-model";

export const inputCls =
  "w-full bg-transparent outline-none text-sm px-2 h-8 rounded border border-[var(--border)] text-[var(--text-primary)]";

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-[var(--text-secondary)]">
      {label}
      {children}
    </label>
  );
}

function NumCell({ value, label, onChange }: { value: number; label: string; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      step={0.5}
      value={Number.isFinite(value) ? value : ""}
      aria-label={label}
      onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      className={inputCls}
    />
  );
}

/** Room rows (id / kind / min-max area), entrance and adjacency inputs. Shared by the
 * Apartment and Building tabs of the Build dialog; the parent owns all state. */
export default function RoomListEditor({
  rows,
  setRows,
  entrance,
  setEntrance,
  adjacency,
  setAdjacency,
  onTemplate,
}: {
  rows: RoomRow[];
  setRows: (r: RoomRow[]) => void;
  /** Effective entrance id (see effectiveEntrance). */
  entrance: string;
  setEntrance: (id: string) => void;
  adjacency: AdjacencyDraft[];
  setAdjacency: (a: AdjacencyDraft[]) => void;
  onTemplate?: () => void;
}) {
  const [adjA, setAdjA] = useState("");
  const [adjB, setAdjB] = useState("");
  const [adjLevel, setAdjLevel] = useState<AdjacencyLevel>("required");
  const ids = rows.map((r) => r.id.trim()).filter(Boolean);

  function setRow(i: number, patch: Partial<RoomRow>) {
    setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  }

  function removeRow(i: number) {
    const gone = rows[i].id.trim();
    setRows(rows.filter((_, j) => j !== i));
    // Adjacency to a removed room would only fail validation; drop it with the room.
    setAdjacency(adjacency.filter((a) => a.a !== gone && a.b !== gone));
  }

  const adjPairTaken = adjacency.some(
    (a) => (a.a === adjA && a.b === adjB) || (a.a === adjB && a.b === adjA),
  );
  const canAddAdj = adjA !== "" && adjB !== "" && adjA !== adjB && !adjPairTaken;

  return (
    <>
      <Field label="Start from a template (fills rooms and adjacency; edit freely)">
        <select
          value=""
          aria-label="Apartment template"
          onChange={(e) => {
            const next = templateRows(e.target.value);
            if (next.length === 0) return;
            setRows(next);
            setAdjacency(templateAdjacency(next));
            setEntrance("");
            onTemplate?.();
          }}
          className={inputCls}
        >
          <option value="">Choose a template…</option>
          {APARTMENT_TEMPLATES.map((t) => (
            <option key={t.key} value={t.key}>{t.label}</option>
          ))}
        </select>
      </Field>

      <fieldset className="flex flex-col gap-1.5">
        <legend className="text-xs text-[var(--text-secondary)] mb-1">Rooms (areas in m²)</legend>
        {rows.length === 0 && (
          <p className="text-xs text-[var(--text-tertiary)]">No rooms yet. Pick a template or add a room.</p>
        )}
        {rows.map((r, i) => (
          <div key={i} className="grid grid-cols-[1fr_6.5rem_4rem_4rem_2rem] gap-1.5 items-center" data-testid="apartment-room-row">
            <input value={r.id} onChange={(e) => setRow(i, { id: e.target.value })} aria-label={`Room ${i + 1} id`} className={inputCls} />
            <select value={r.kind} onChange={(e) => setRow(i, { kind: e.target.value as ApartmentKind })} aria-label={`Room ${i + 1} kind`} className={inputCls}>
              {APARTMENT_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <NumCell value={r.min_area} label={`Room ${i + 1} min area`} onChange={(v) => setRow(i, { min_area: v })} />
            <NumCell value={r.max_area} label={`Room ${i + 1} max area`} onChange={(v) => setRow(i, { max_area: v })} />
            <button type="button" onClick={() => removeRow(i)} aria-label={`Remove room ${r.id || i + 1}`} className="h-8 grid place-items-center rounded border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => setRows([...rows, { id: nextRoomId(rows, "bedroom"), kind: "bedroom", min_area: 9, max_area: 14 }])}
          className="self-start flex items-center gap-1 h-7 px-2 rounded text-xs border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" /> Add room
        </button>
      </fieldset>

      <Field label="Entrance room">
        <select value={entrance} onChange={(e) => setEntrance(e.target.value)} aria-label="Entrance room" disabled={rows.length === 0} className={inputCls}>
          {ids.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </Field>

      <fieldset className="flex flex-col gap-1.5">
        <legend className="text-xs text-[var(--text-secondary)] mb-1">Adjacency (rooms that should share a wall)</legend>
        <div className="grid grid-cols-[1fr_1fr_6.5rem_auto] gap-1.5 items-center">
          <select value={adjA} onChange={(e) => setAdjA(e.target.value)} aria-label="Adjacency first room" className={inputCls}>
            <option value="">Room…</option>
            {ids.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
          <select value={adjB} onChange={(e) => setAdjB(e.target.value)} aria-label="Adjacency second room" className={inputCls}>
            <option value="">Room…</option>
            {ids.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
          <select value={adjLevel} onChange={(e) => setAdjLevel(e.target.value as AdjacencyLevel)} aria-label="Adjacency level" className={inputCls}>
            <option value="required">required</option>
            <option value="preferred">preferred</option>
          </select>
          <button
            type="button"
            disabled={!canAddAdj}
            onClick={() => {
              setAdjacency([...adjacency, { a: adjA, b: adjB, level: adjLevel }]);
              setAdjA("");
              setAdjB("");
            }}
            className="h-8 px-2 rounded text-xs border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
          >
            Add
          </button>
        </div>
        {adjacency.length > 0 && (
          <ul className="flex flex-col gap-1" data-testid="apartment-adjacency-list">
            {adjacency.map((a, i) => (
              <li key={`${a.a}|${a.b}`} className="flex items-center gap-2 text-xs text-[var(--text-primary)]">
                <span>{a.a} – {a.b}</span>
                <span className={a.level === "required" ? "text-[var(--accent)]" : "text-[var(--text-tertiary)]"}>{a.level}</span>
                <button type="button" onClick={() => setAdjacency(adjacency.filter((_, j) => j !== i))} aria-label={`Remove adjacency ${a.a} – ${a.b}`} className="ml-auto text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </fieldset>
    </>
  );
}
