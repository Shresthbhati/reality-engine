"use client";

import { useMemo, useRef, useState } from "react";
import { Globe, X } from "lucide-react";
import { isApiError, useWorlds } from "@/lib/api";
import { cellLabel, defaultLabel, parseCell, type Placement } from "@/lib/sites/site-grid-model";

interface Props {
  gridCellM: number;
  /** Placements already on the site, shown as occupied cells on the picker. */
  placements: Placement[];
  onClose: () => void;
  onAdd: (input: { world_id: string; label: string; cell_x: number; cell_z: number }) => Promise<void>;
}

/** Picker window: cells shown around the existing content; any cell can also be typed. */
const PAD = 1;
const MIN_SPAN = 3;

export default function AddWorldDialog({ gridCellM, placements, onClose, onAdd }: Props) {
  const { data: worlds, isLoading, error, refetch } = useWorlds();
  const [worldId, setWorldId] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [cx, setCx] = useState("0");
  const [cz, setCz] = useState("0");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const world = worlds?.find((w) => w.id === worldId) ?? null;
  const cellX = parseCell(cx);
  const cellZ = parseCell(cz);

  const window_ = useMemo(() => {
    const xs = [...placements.map((p) => p.cell_x), cellX ?? 0];
    const zs = [...placements.map((p) => p.cell_z), cellZ ?? 0];
    const minX = Math.min(...xs) - PAD;
    const minZ = Math.min(...zs) - PAD;
    const spanX = Math.max(MIN_SPAN + 2, Math.max(...xs) + PAD - minX + 1);
    const spanZ = Math.max(MIN_SPAN + 2, Math.max(...zs) + PAD - minZ + 1);
    return { minX, minZ, spanX: Math.min(spanX, 15), spanZ: Math.min(spanZ, 15) };
  }, [placements, cellX, cellZ]);

  const occupied = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const p of placements) {
      const k = `${p.cell_x},${p.cell_z}`;
      m.set(k, [...(m.get(k) ?? []), p.label]);
    }
    return m;
  }, [placements]);

  const canSubmit = world !== null && cellX !== null && cellZ !== null && !busy;

  async function submit() {
    if (!world || cellX === null || cellZ === null) return;
    setBusy(true);
    setFailure(null);
    try {
      await onAdd({ world_id: world.id, label: label.trim() || defaultLabel(world.name), cell_x: cellX, cell_z: cellZ });
      onClose();
    } catch (err) {
      setFailure(isApiError(err) ? err.describe() : (err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "color-mix(in srgb, var(--bg-base) 70%, transparent)" }}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-world-title"
        className="w-full max-w-3xl max-h-full overflow-y-auto rounded-xl border"
        style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 h-14 border-b" style={{ borderColor: "var(--border)" }}>
          <h2 id="add-world-title" className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Add world to site
          </h2>
          <button type="button" onClick={onClose} aria-label="Close" className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="grid gap-5 p-5 md:grid-cols-2">
          <section aria-labelledby="pick-world">
            <h3 id="pick-world" className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-secondary)" }}>
              1. Pick a world
            </h3>
            {isLoading && <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Loading worlds…</p>}
            {error && (
              <div role="alert" className="text-sm rounded-md px-3 py-2" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
                {isApiError(error) ? error.describe() : error.message}{" "}
                <button type="button" className="underline" onClick={refetch}>
                  Retry
                </button>
              </div>
            )}
            {worlds && worlds.length === 0 && (
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                No worlds exist yet. Reconstruct a world first, then come back to place it.
              </p>
            )}
            {worlds && worlds.length > 0 && (
              <ul className="flex flex-col gap-1 max-h-64 overflow-y-auto" role="listbox" aria-label="Worlds">
                {worlds.map((w) => {
                  const on = w.id === worldId;
                  return (
                    <li key={w.id} role="option" aria-selected={on}>
                      <button
                        type="button"
                        onClick={() => setWorldId(w.id)}
                        className="w-full flex items-center gap-2 px-3 h-10 rounded-md border text-left text-sm hover:bg-[var(--bg-hover)]"
                        style={{
                          borderColor: on ? "var(--accent-border)" : "var(--border-subtle)",
                          background: on ? "var(--accent-subtle)" : undefined,
                          color: on ? "var(--accent)" : "var(--text-primary)",
                        }}
                      >
                        <Globe className="w-4 h-4 shrink-0" aria-hidden="true" />
                        <span className="truncate">{w.name}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}

            <label className="block mt-4 text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-secondary)" }} htmlFor="placement-label">
              Label (optional)
            </label>
            <input
              id="placement-label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder={world ? defaultLabel(world.name) : "Defaults to the world name"}
              className="mt-1 w-full h-9 px-3 rounded-md border text-sm"
              style={{ background: "var(--bg-base)", borderColor: "var(--border)", color: "var(--text-primary)" }}
            />
          </section>

          <section aria-labelledby="pick-cell">
            <h3 id="pick-cell" className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-secondary)" }}>
              2. Pick a grid cell ({gridCellM} m each)
            </h3>
            <p className="text-[11px] mb-1" style={{ color: "var(--text-tertiary)" }}>Back</p>
            <div
              className="grid gap-1"
              style={{ gridTemplateColumns: `repeat(${window_.spanX}, minmax(0, 1fr))` }}
              role="grid"
              aria-label="Grid cells; back row at the top"
            >
              {Array.from({ length: window_.spanZ }).flatMap((_, zi) =>
                Array.from({ length: window_.spanX }).map((__, xi) => {
                  const x = window_.minX + xi;
                  const z = window_.minZ + zi;
                  const sel = x === cellX && z === cellZ;
                  const taken = occupied.get(`${x},${z}`);
                  return (
                    <button
                      key={`${x},${z}`}
                      type="button"
                      role="gridcell"
                      aria-selected={sel}
                      aria-label={`Cell ${cellLabel(x, z)}${taken ? `, occupied by ${taken.join(", ")}` : ""}`}
                      title={taken ? `Occupied: ${taken.join(", ")}` : cellLabel(x, z)}
                      onClick={() => {
                        setCx(String(x));
                        setCz(String(z));
                      }}
                      className="aspect-square min-h-8 rounded text-[10px] border flex items-center justify-center overflow-hidden"
                      style={{
                        borderColor: sel ? "var(--accent)" : "var(--border)",
                        background: sel ? "var(--accent-subtle)" : taken ? "var(--bg-elevated)" : "var(--bg-base)",
                        color: sel ? "var(--accent)" : "var(--text-tertiary)",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {taken ? "●" : `${x},${z}`}
                    </button>
                  );
                }),
              )}
            </div>
            <p className="text-[11px] mt-1" style={{ color: "var(--text-tertiary)" }}>Front</p>

            <div className="grid grid-cols-2 gap-3 mt-3">
              {[
                { id: "cell-x", label: "Cell X", value: cx, set: setCx, bad: cellX === null },
                { id: "cell-z", label: "Cell Z (larger = nearer the front)", value: cz, set: setCz, bad: cellZ === null },
              ].map((f) => (
                <div key={f.id}>
                  <label htmlFor={f.id} className="block text-xs" style={{ color: "var(--text-secondary)" }}>
                    {f.label}
                  </label>
                  <input
                    id={f.id}
                    inputMode="numeric"
                    value={f.value}
                    onChange={(e) => f.set(e.target.value)}
                    aria-invalid={f.bad}
                    className="mt-1 w-full h-9 px-3 rounded-md border text-sm"
                    style={{ background: "var(--bg-base)", borderColor: f.bad ? "var(--error)" : "var(--border)", color: "var(--text-primary)" }}
                  />
                </div>
              ))}
            </div>
          </section>
        </div>

        {failure && (
          <div role="alert" className="mx-5 mb-3 text-sm rounded-md px-3 py-2" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
            {failure}
          </div>
        )}

        <div className="flex justify-end gap-2 px-5 h-16 items-center border-t" style={{ borderColor: "var(--border)" }}>
          <button type="button" onClick={onClose} className="h-9 px-3 rounded-md text-sm hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>
            Cancel
          </button>
          <button
            type="button"
            disabled={!canSubmit}
            onClick={submit}
            className="h-9 px-4 rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            {busy ? "Placing…" : "Place on grid"}
          </button>
        </div>
      </div>
    </div>
  );
}
