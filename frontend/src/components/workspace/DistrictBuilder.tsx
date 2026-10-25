"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { createDistrict } from "@/lib/api/procedural";
import { isApiError } from "@/lib/api/client";
import { Field, inputCls } from "@/components/workspace/RoomListEditor";
import {
  MAX_FAMILIES,
  defaultDistrictDraft,
  districtNote,
  effectiveBlocksY,
  toDistrictBody,
  validateDistrict,
  type DistrictDraft,
  type DistrictLayout,
} from "@/lib/procedural/district-model";

const toNum = (s: string) => (s === "" ? NaN : Number(s));
const show = (n: number) => (Number.isFinite(n) ? n : "");

/** District tab of the Build dialog: layout + grid + lot/street sizes + families + detail policy
 * -> POST .../districts -> new WorldStore version. Server 409/422 text is shown verbatim. */
export default function DistrictBuilder({
  worldId,
  onCreated,
  onClose,
}: {
  worldId: string;
  onCreated: () => void;
  onClose: () => void;
}) {
  const [d, setD] = useState<DistrictDraft>(defaultDistrictDraft);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<DistrictDraft>) => setD((cur) => ({ ...cur, ...patch }));
  const problems = validateDistrict(d);
  const note = districtNote(d);
  const spine = d.layout === "spine";

  async function build() {
    setBusy(true);
    setError(null);
    try {
      await createDistrict(worldId, toDistrictBody(d));
      onCreated();
      onClose();
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Failed to create district.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3" data-testid="district-builder">
      <div className="flex items-center justify-between text-[10px] text-[var(--text-tertiary)]">
        <span>All dimensions in metres (m).</span>
        <span className="font-semibold text-[var(--warning)] border border-[var(--warning)] rounded px-1" data-testid="district-provenance">GENERATED</span>
      </div>

      <Field label="Name">
        <input value={d.name} onChange={(e) => set({ name: e.target.value })} aria-label="District name" className={inputCls} />
      </Field>

      <Field label="Layout">
        <select value={d.layout} onChange={(e) => set({ layout: e.target.value as DistrictLayout })} aria-label="Layout" className={inputCls}>
          <option value="grid">Grid (a street on every parcel line)</option>
          <option value="spine">Spine (one main street, a row of parcels each side)</option>
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-2">
        <Field label="Parcels east-west">
          <input type="number" step={1} value={show(d.blocksX)} onChange={(e) => set({ blocksX: toNum(e.target.value) })} aria-label="Parcels east-west" className={inputCls} />
        </Field>
        <Field label={spine ? "Parcels north-south (fixed at 2 for spine)" : "Parcels north-south"}>
          <input type="number" step={1} value={show(effectiveBlocksY(d))} disabled={spine} onChange={(e) => set({ blocksY: toNum(e.target.value) })} aria-label="Parcels north-south" className={inputCls} />
        </Field>
        <Field label="Lot size (m, 8-200)">
          <input type="number" step={1} value={show(d.lot)} onChange={(e) => set({ lot: toNum(e.target.value) })} aria-label="Lot size (m)" className={inputCls} />
        </Field>
        <Field label="Street width (m, 1-60)">
          <input type="number" step={0.5} value={show(d.streetWidth)} onChange={(e) => set({ streetWidth: toNum(e.target.value) })} aria-label="Street width (m)" className={inputCls} />
        </Field>
        {spine && (
          <Field label="Parcels between cross streets">
            <input type="number" step={1} value={show(d.crossEvery)} onChange={(e) => set({ crossEvery: toNum(e.target.value) })} aria-label="Parcels between cross streets" className={inputCls} />
          </Field>
        )}
        <Field label={`Building families (1-${MAX_FAMILIES}; 3 = seeded defaults)`}>
          <input type="number" step={1} value={show(d.families)} onChange={(e) => set({ families: toNum(e.target.value) })} aria-label="Families" className={inputCls} />
        </Field>
        <Field label="Fully detailed buildings (empty = all)">
          <input
            type="number"
            step={1}
            min={0}
            value={d.maxFull === null ? "" : show(d.maxFull)}
            onChange={(e) => set({ maxFull: e.target.value === "" ? null : Number(e.target.value) })}
            aria-label="Fully detailed buildings"
            className={inputCls}
          />
        </Field>
        <Field label="Seed (same seed = same district)">
          <input type="number" step={1} value={show(d.seed)} onChange={(e) => set({ seed: toNum(e.target.value) })} aria-label="Seed" className={inputCls} />
        </Field>
      </div>

      <p className="text-[11px] text-neutral-500" data-testid="district-note">
        {note || "Fix the highlighted values to see what will be generated."}
      </p>

      {problems.length > 0 && (
        <ul className="text-xs text-[var(--warning)] list-disc pl-4" data-testid="district-problems">
          {problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}

      {error && <p role="alert" className="text-xs text-[var(--error)]">{error}</p>}

      <button
        type="button"
        onClick={() => void build()}
        disabled={busy || problems.length > 0}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold bg-[var(--primary)] text-[var(--text-primary)] hover:bg-[var(--primary-hover)] disabled:opacity-40 cursor-pointer"
      >
        {busy && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {busy ? "Building…" : "Build district"}
      </button>
    </div>
  );
}
