"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { createApartment, solveApartment, type ApartmentSolveResult } from "@/lib/api/procedural";
import { isApiError } from "@/lib/api/client";
import RoomListEditor, { Field, inputCls } from "@/components/workspace/RoomListEditor";
import {
  adjacencySummary,
  effectiveEntrance,
  planToSvg,
  toRequestBody,
  validateApartment,
  type AdjacencyDraft,
  type ApartmentDraft,
  type RoomRow,
} from "@/lib/procedural/apartment-model";

/** Apartment tab of the Build dialog: describe rooms + adjacency, ask the real
 * constraint solver (POST .../apartments/solve) for candidate floor plans, then commit
 * the chosen one (POST .../apartments -> new WorldStore version). */
export default function ApartmentBuilder({
  worldId,
  onCreated,
  onClose,
}: {
  worldId: string;
  onCreated: () => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("apartment-1");
  const [rows, setRows] = useState<RoomRow[]>([]);
  const [entrance, setEntrance] = useState("");
  const [auto, setAuto] = useState(true);
  const [fw, setFw] = useState(10);
  const [fd, setFd] = useState(8);
  const [adjacency, setAdjacency] = useState<AdjacencyDraft[]>([]);
  const [seed, setSeed] = useState(0);

  const [solving, setSolving] = useState(false);
  const [building, setBuilding] = useState(false);
  const [solved, setSolved] = useState<{ key: string; result: ApartmentSolveResult } | null>(null);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const draft: ApartmentDraft = {
    name,
    rows,
    entrance: effectiveEntrance(rows, entrance),
    footprint: auto ? null : { width: fw, depth: fd },
    adjacency,
    seed,
  };
  const problems = validateApartment(draft);
  const body = toRequestBody(draft);
  const bodyKey = JSON.stringify(body);
  const current = solved && solved.key === bodyKey ? solved.result : null;
  const stale = solved !== null && current === null;
  const kinds = Object.fromEntries(rows.map((r) => [r.id.trim(), r.kind]));

  function describe(err: unknown, fallback: string): string {
    return isApiError(err) ? err.describe() : fallback;
  }

  async function findLayouts() {
    setSolving(true);
    setError(null);
    try {
      const result = await solveApartment(worldId, body);
      setSolved({ key: bodyKey, result });
      setSelected(0);
    } catch (err) {
      setSolved(null);
      setError(describe(err, "Could not reach the layout solver."));
    } finally {
      setSolving(false);
    }
  }

  async function build() {
    setBuilding(true);
    setError(null);
    try {
      await createApartment(worldId, body, selected);
      onCreated();
      onClose();
    } catch (err) {
      setError(describe(err, "Failed to create apartment."));
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="flex flex-col gap-3" data-testid="apartment-builder">
      <Field label="Name">
        <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Apartment name" className={inputCls} />
      </Field>

      <RoomListEditor
        rows={rows}
        setRows={setRows}
        entrance={draft.entrance}
        setEntrance={setEntrance}
        adjacency={adjacency}
        setAdjacency={setAdjacency}
        onTemplate={() => setSolved(null)}
      />

      <Field label="Seed (same seed = same layouts)">
        <input type="number" step={1} value={Number.isFinite(seed) ? seed : ""} onChange={(e) => setSeed(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Seed" className={inputCls} />
      </Field>

      <div className="flex flex-col gap-1.5">
        <label className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} aria-label="Auto footprint" />
          Footprint: auto (solver picks a size from the room areas)
        </label>
        {!auto && (
          <div className="grid grid-cols-2 gap-2">
            <Field label="Footprint width (m)">
              <input type="number" step={0.1} value={Number.isFinite(fw) ? fw : ""} onChange={(e) => setFw(e.target.value === "" ? NaN : Number(e.target.value))} className={inputCls} />
            </Field>
            <Field label="Footprint depth (m)">
              <input type="number" step={0.1} value={Number.isFinite(fd) ? fd : ""} onChange={(e) => setFd(e.target.value === "" ? NaN : Number(e.target.value))} className={inputCls} />
            </Field>
          </div>
        )}
      </div>


      {problems.length > 0 && (
        <ul className="text-xs text-[var(--warning)] list-disc pl-4" data-testid="apartment-problems">
          {problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}

      <button
        type="button"
        onClick={() => void findLayouts()}
        disabled={solving || problems.length > 0}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold border border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--accent)] disabled:opacity-40 cursor-pointer"
      >
        {solving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {solving ? "Solving…" : "Find layouts"}
      </button>

      {current && !current.ok && (
        <div role="alert" className="rounded border border-[var(--error)] bg-[var(--error-subtle)] p-2 text-xs text-[var(--text-primary)]" data-testid="apartment-violations">
          <p className="font-semibold mb-1">No valid layout for this brief:</p>
          <ul className="list-disc pl-4">
            {current.violations.map((v, i) => <li key={i}>{v}</li>)}
          </ul>
        </div>
      )}

      {stale && (
        <p className="text-xs text-[var(--text-tertiary)]">The inputs changed since the last solve. Find layouts again.</p>
      )}

      {current?.ok && (
        <div className="flex flex-col gap-2" data-testid="apartment-plans">
          <p className="text-xs text-[var(--text-secondary)]">
            {current.plans.length} candidate layout{current.plans.length === 1 ? "" : "s"}
            {current.footprint ? ` - footprint ${current.footprint[0]} x ${current.footprint[1]} m` : ""}. Pick one.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {current.plans.map((p, i) => {
              const svg = planToSvg(p, kinds, current.footprint, 220);
              const on = selected === i;
              return (
                <button
                  key={i}
                  type="button"
                  data-testid="apartment-plan"
                  aria-pressed={on}
                  onClick={() => setSelected(i)}
                  className={`rounded border p-1.5 text-left flex flex-col gap-1 cursor-pointer ${
                    on ? "border-[var(--accent)] bg-[var(--accent-subtle)]" : "border-[var(--border)] hover:bg-[var(--bg-hover)]"
                  }`}
                >
                  <svg viewBox={`0 0 ${svg.width} ${svg.height}`} className="w-full h-auto" role="img" aria-label={`Floor plan ${i + 1}`}>
                    {svg.rooms.map((r) => (
                      <g key={r.id}>
                        <rect
                          x={r.x} y={r.y} width={r.w} height={r.h}
                          fill={r.circulation ? "var(--purple-subtle)" : "var(--bg-elevated)"}
                          stroke="var(--accent-border)" strokeWidth={1}
                        />
                        <text x={r.cx} y={r.cy - (r.tiny ? 0 : 2)} textAnchor="middle" dominantBaseline="middle" fontSize={r.tiny ? 7 : 9} fill="var(--text-primary)">
                          {r.id}
                        </text>
                        {!r.tiny && (
                          <text x={r.cx} y={r.cy + 9} textAnchor="middle" dominantBaseline="middle" fontSize={8} fill="var(--text-secondary)">
                            {r.areaText}
                          </text>
                        )}
                      </g>
                    ))}
                  </svg>
                  <span className="text-[11px] text-[var(--text-primary)]">Plan {i + 1} - score {Math.round(p.score * 100) / 100}</span>
                  <span className="text-[11px] text-[var(--text-secondary)]">{adjacencySummary(p, adjacency)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {error && <p role="alert" className="text-xs text-[var(--error)]">{error}</p>}

      <button
        type="button"
        onClick={() => void build()}
        disabled={building || !current?.ok || current.plans.length === 0}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold bg-[var(--primary)] text-[var(--text-primary)] hover:bg-[var(--primary-hover)] disabled:opacity-40 cursor-pointer"
      >
        {building && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {building ? "Building…" : "Build apartment"}
      </button>
    </div>
  );
}
