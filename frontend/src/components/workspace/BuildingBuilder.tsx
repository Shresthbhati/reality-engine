"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { createBuilding } from "@/lib/api/procedural";
import { isApiError } from "@/lib/api/client";
import RoomListEditor, { Field, inputCls } from "@/components/workspace/RoomListEditor";
import { effectiveEntrance, type AdjacencyDraft, type RoomRow } from "@/lib/procedural/apartment-model";
import { toBuildingBody, validateBuilding, type BuildingDraft } from "@/lib/procedural/building-model";
import { H_MAX, H_MIN, MAX_UI_STOREYS } from "@/lib/procedural/generated-model";

/** Building tab of the Build dialog: footprint + storeys (+ an optional floor layout built
 * from the same room/adjacency inputs as Apartment) -> POST .../buildings -> new WorldStore
 * version. Server 409/422 text is shown verbatim. */
export default function BuildingBuilder({
  worldId,
  onCreated,
  onClose,
}: {
  worldId: string;
  onCreated: () => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("building-1");
  const [width, setWidth] = useState(12);
  const [depth, setDepth] = useState(10);
  const [storeys, setStoreys] = useState(3);
  const [height, setHeight] = useState(3);
  const [seed, setSeed] = useState(0);
  const [rows, setRows] = useState<RoomRow[]>([]);
  const [entrance, setEntrance] = useState("");
  const [adjacency, setAdjacency] = useState<AdjacencyDraft[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const draft: BuildingDraft = {
    name, width, depth, storeys, height, rows,
    entrance: effectiveEntrance(rows, entrance), adjacency, seed,
  };
  const problems = validateBuilding(draft);

  async function build() {
    setBusy(true);
    setError(null);
    try {
      await createBuilding(worldId, toBuildingBody(draft));
      onCreated();
      onClose();
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Failed to create building.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3" data-testid="building-builder">
      <div className="flex items-center justify-between text-[10px] text-[var(--text-tertiary)]">
        <span>All dimensions in metres (m); areas in m².</span>
        <span className="font-semibold text-[var(--warning)] border border-[var(--warning)] rounded px-1" data-testid="building-provenance">GENERATED</span>
      </div>

      <Field label="Name">
        <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Building name" className={inputCls} />
      </Field>

      <div className="grid grid-cols-2 gap-2">
        <Field label="Footprint width (m)">
          <input type="number" step={0.5} value={Number.isFinite(width) ? width : ""} onChange={(e) => setWidth(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Footprint width (m)" className={inputCls} />
        </Field>
        <Field label="Footprint depth (m)">
          <input type="number" step={0.5} value={Number.isFinite(depth) ? depth : ""} onChange={(e) => setDepth(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Footprint depth (m)" className={inputCls} />
        </Field>
        <Field label={`Storeys (1-${MAX_UI_STOREYS})`}>
          <input type="number" step={1} min={1} max={MAX_UI_STOREYS} value={Number.isFinite(storeys) ? storeys : ""} onChange={(e) => setStoreys(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Storeys" className={inputCls} />
        </Field>
        <Field label={`Storey height (m, ${H_MIN}-${H_MAX})`}>
          <input type="number" step={0.1} value={Number.isFinite(height) ? height : ""} onChange={(e) => setHeight(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Storey height (m)" className={inputCls} />
        </Field>
      </div>

      <Field label="Seed (same seed = same building)">
        <input type="number" step={1} value={Number.isFinite(seed) ? seed : ""} onChange={(e) => setSeed(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Seed" className={inputCls} />
      </Field>

      <details className="flex flex-col gap-3" open={rows.length > 0}>
        <summary className="cursor-pointer text-xs text-[var(--text-secondary)]">
          Floor layout (optional): rooms repeated on every storey. Leave empty for a shell.
        </summary>
        <div className="flex flex-col gap-3 mt-3">
          <RoomListEditor
            rows={rows}
            setRows={setRows}
            entrance={draft.entrance}
            setEntrance={setEntrance}
            adjacency={adjacency}
            setAdjacency={setAdjacency}
          />
        </div>
      </details>

      {problems.length > 0 && (
        <ul className="text-xs text-[var(--warning)] list-disc pl-4" data-testid="building-problems">
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
        {busy ? "Building…" : "Build building"}
      </button>
    </div>
  );
}
