"use client";

import { useState } from "react";
import { RotateCcw, RotateCw, Trash2, TriangleAlert } from "lucide-react";
import {
  clampScale,
  footprintRect,
  normalizeDeg,
  overlapPartners,
  parseCell,
  rotateBy,
  type Placement,
} from "@/lib/sites/site-grid-model";
import type { UpdatePlacementInput } from "@/lib/api";

interface Props {
  placement: Placement;
  placements: Placement[];
  overlaps: [string, string][];
  gridCellM: number;
  busy: boolean;
  onChange: (patch: UpdatePlacementInput) => void;
  onRemoveRequest: () => void;
}

const inputStyle = { background: "var(--bg-base)", borderColor: "var(--border)", color: "var(--text-primary)" } as const;

/** Text field that commits on blur / Enter and re-syncs when the server value changes (via `key`). */
function CommitField({ id, label, value, onCommit, inputMode = "decimal", validate }: {
  id: string;
  label: string;
  value: string;
  onCommit: (raw: string) => void;
  inputMode?: "decimal" | "numeric" | "text";
  validate?: (raw: string) => boolean;
}) {
  const [draft, setDraft] = useState(value);
  const bad = validate ? !validate(draft) : false;
  const commit = () => {
    if (draft === value) return;
    if (bad) {
      setDraft(value);
      return;
    }
    onCommit(draft);
  };
  return (
    <div>
      <label htmlFor={id} className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>
        {label}
      </label>
      <input
        id={id}
        value={draft}
        inputMode={inputMode}
        aria-invalid={bad}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.currentTarget as HTMLInputElement).blur()}
        className="w-full h-9 px-3 rounded-md border text-sm"
        style={{ ...inputStyle, borderColor: bad ? "var(--error)" : "var(--border)" }}
      />
    </div>
  );
}

const isNum = (raw: string) => raw.trim() !== "" && Number.isFinite(Number(raw));

export default function PlacementInspector({ placement: p, placements, overlaps, gridCellM, busy, onChange, onRemoveRequest }: Props) {
  const partners = overlapPartners(p.id, overlaps)
    .map((id) => placements.find((q) => q.id === id)?.label)
    .filter((x): x is string => !!x);
  const rect = footprintRect(p, gridCellM);
  const k = (name: string, v: number | string) => `${p.id}:${name}:${v}`; // re-sync a field when the saved value changes

  return (
    <div className="flex flex-col gap-4" aria-busy={busy}>
      <div>
        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>World</p>
        <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{p.world_name}</p>
        <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
          {rect.isFallback
            ? "Footprint unknown: drawn as a dashed stand-in box, not the real size."
            : `Footprint ${rect.width_m.toFixed(1)} × ${rect.depth_m.toFixed(1)} m (after scale)`}
        </p>
      </div>

      {partners.length > 0 && (
        <div role="status" className="flex items-start gap-2 rounded-md px-3 py-2 text-xs" style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}>
          <TriangleAlert className="w-4 h-4 shrink-0 mt-px" aria-hidden="true" />
          <span>Overlaps {partners.join(", ")}. Overlap is allowed.</span>
        </div>
      )}

      <CommitField key={k("label", p.label)} id="pi-label" label="Name" inputMode="text" value={p.label} validate={(r) => r.trim().length > 0} onCommit={(r) => onChange({ label: r.trim() })} />

      <div className="grid grid-cols-2 gap-3">
        <CommitField key={k("cx", p.cell_x)} id="pi-cx" label="Cell X" inputMode="numeric" value={String(p.cell_x)} validate={(r) => parseCell(r) !== null} onCommit={(r) => onChange({ cell_x: parseCell(r)! })} />
        <CommitField key={k("cz", p.cell_z)} id="pi-cz" label="Cell Z" inputMode="numeric" value={String(p.cell_z)} validate={(r) => parseCell(r) !== null} onCommit={(r) => onChange({ cell_z: parseCell(r)! })} />
        <CommitField key={k("ox", p.offset_x_m)} id="pi-ox" label="Offset X (m)" value={String(p.offset_x_m)} validate={isNum} onCommit={(r) => onChange({ offset_x_m: Number(r) })} />
        <CommitField key={k("oz", p.offset_z_m)} id="pi-oz" label="Offset Z (m)" value={String(p.offset_z_m)} validate={isNum} onCommit={(r) => onChange({ offset_z_m: Number(r) })} />
      </div>
      <p className="text-[11px] -mt-2" style={{ color: "var(--text-tertiary)" }}>
        Centre: {p.center_m.x.toFixed(1)}, {p.center_m.z.toFixed(1)} m. Arrow keys nudge a quarter cell (Shift: fine).
      </p>

      <div>
        <span className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Rotation</span>
        <div className="flex items-center gap-2">
          <button type="button" aria-label="Rotate 15 degrees counter-clockwise" onClick={() => onChange({ rotation_deg: rotateBy(p.rotation_deg, -15) })} className="w-9 h-9 rounded-md border flex items-center justify-center hover:bg-[var(--bg-hover)]" style={{ ...inputStyle, color: "var(--text-secondary)" }}>
            <RotateCcw className="w-4 h-4" aria-hidden="true" />
          </button>
          <div className="flex-1">
            <CommitField key={k("rot", p.rotation_deg)} id="pi-rot" label="Degrees (clockwise from above)" value={String(p.rotation_deg)} validate={isNum} onCommit={(r) => onChange({ rotation_deg: normalizeDeg(Number(r)) })} />
          </div>
          <button type="button" aria-label="Rotate 15 degrees clockwise" onClick={() => onChange({ rotation_deg: rotateBy(p.rotation_deg, 15) })} className="w-9 h-9 rounded-md border flex items-center justify-center hover:bg-[var(--bg-hover)] self-end" style={{ ...inputStyle, color: "var(--text-secondary)" }}>
            <RotateCw className="w-4 h-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div>
        <CommitField key={k("scale", p.scale)} id="pi-scale" label="Scale (0.1 to 10)" value={String(p.scale)} validate={(r) => isNum(r) && Number(r) > 0} onCommit={(r) => onChange({ scale: clampScale(Number(r)) })} />
        <div className="flex gap-2 mt-2">
          {[0.5, 2].map((f) => (
            <button key={f} type="button" onClick={() => onChange({ scale: clampScale(p.scale * f) })} className="h-8 px-3 rounded-md border text-xs hover:bg-[var(--bg-hover)]" style={{ ...inputStyle, color: "var(--text-secondary)" }}>
              ×{f}
            </button>
          ))}
          <button type="button" onClick={() => onChange({ scale: 1 })} className="h-8 px-3 rounded-md border text-xs hover:bg-[var(--bg-hover)]" style={{ ...inputStyle, color: "var(--text-secondary)" }}>
            Reset
          </button>
        </div>
      </div>

      <button
        type="button"
        onClick={onRemoveRequest}
        className="h-9 rounded-md border text-sm flex items-center justify-center gap-2 hover:bg-[var(--error-subtle)]"
        style={{ borderColor: "var(--error)", color: "var(--error)" }}
      >
        <Trash2 className="w-4 h-4" aria-hidden="true" />
        Remove from site
      </button>
    </div>
  );
}
