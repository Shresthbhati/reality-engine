"use client";

import { useEffect, useState } from "react";
import { Armchair, RotateCcw, RotateCw, Trash2 } from "lucide-react";
import {
  type DesignResult, catalogLabel, degreesToRadians, deleteConfirmation, deleteFurniture, getFurnitureCatalog, moveFurniture, normaliseDegrees,
  parseNumber, replaceBlockedReason, replaceFurniture, REPLACE_NOTE, rotateFurniture, yawDegrees, type CatalogItem,
} from "@/lib/api/design";

interface FurnitureLike {
  id: string;
  name?: string;
  type?: string;
  custom_properties?: Record<string, unknown> | null;
  transform?: { position?: { x: number; y: number; z: number } } | null;
}

const field = "h-7 w-full rounded border border-[var(--border)] bg-[var(--surface-sunken)] px-1.5 text-xs tabular-nums";
const btn = "flex h-7 items-center justify-center gap-1 rounded border border-[var(--border)] px-2 text-xs hover:bg-[var(--surface-hover)] disabled:opacity-40 cursor-pointer";

/** Move / turn / swap / delete one piece of furniture. Each action is one saved version, so Undo (Ctrl+Z) reverses it. */
export default function FurnitureDesignPanel({ worldId, entity, parentVersion, metricScale, onChanged }: {
  worldId: string; entity: FurnitureLike; parentVersion?: string; metricScale: boolean; onChanged: () => void;
}) {
  const pos = entity.transform?.position;
  const [x, setX] = useState(pos ? String(Math.round(pos.x * 100) / 100) : "");
  const [z, setZ] = useState(pos ? String(Math.round(pos.z * 100) / 100) : "");
  const [deg, setDeg] = useState(String(yawDegrees(entity)));
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  // Each save returns the new version; the next action builds on it right away instead of waiting for the model to reload. Once the
  // model has reloaded (or changed some other way, e.g. Undo) the version the screen shows wins again.
  const [saved, setSaved] = useState<{ version: string; basedOn: string | undefined } | null>(null);
  const parent = saved && saved.basedOn === parentVersion ? saved.version : parentVersion;
  const blocked = replaceBlockedReason(metricScale);

  // The caller keys this panel by piece + version, so a new selection or a saved edit remounts it with fresh numbers.
  useEffect(() => {
    let live = true;
    getFurnitureCatalog().then((items) => { if (live) setCatalog(items); }).catch(() => { if (live) setCatalog([]); });
    return () => { live = false; };
  }, []);

  async function run(action: () => Promise<DesignResult>) {
    setBusy(true);
    setError(null);
    try {
      const res = await action();
      if (res?.version_id) setSaved({ version: res.version_id, basedOn: parentVersion });
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save that change");
    } finally {
      setBusy(false);
    }
  }

  const nx = parseNumber(x), nz = parseNumber(z), nd = parseNumber(deg);

  return (
    <section aria-label="Furniture design" className="space-y-3 rounded border border-[var(--border)] p-3 text-xs" data-testid="furniture-design">
      <h4 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
        <Armchair className="h-3.5 w-3.5" aria-hidden /> Design
      </h4>

      <div className="space-y-1">
        <div className="text-[11px] text-[var(--text-secondary)]">Position (x, z)</div>
        <div className="flex items-center gap-1.5">
          <input aria-label="Position x" className={field} value={x} onChange={(e) => setX(e.target.value)} inputMode="decimal" />
          <input aria-label="Position z" className={field} value={z} onChange={(e) => setZ(e.target.value)} inputMode="decimal" />
          <button type="button" className={btn} disabled={busy || nx === null || nz === null}
            onClick={() => run(() => moveFurniture(worldId, entity.id, nx as number, nz as number, parent))}>Move</button>
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-[11px] text-[var(--text-secondary)]">Turn (degrees)</div>
        <div className="flex items-center gap-1.5">
          <button type="button" aria-label="Turn left 90 degrees" className={btn} disabled={busy}
            onClick={() => run(() => rotateFurniture(worldId, entity.id, degreesToRadians(yawDegrees(entity) - 90), parent))}><RotateCcw className="h-3.5 w-3.5" /></button>
          <button type="button" aria-label="Turn right 90 degrees" className={btn} disabled={busy}
            onClick={() => run(() => rotateFurniture(worldId, entity.id, degreesToRadians(yawDegrees(entity) + 90), parent))}><RotateCw className="h-3.5 w-3.5" /></button>
          <input aria-label="Turn angle" className={field} value={deg} onChange={(e) => setDeg(e.target.value)} inputMode="decimal" />
          <button type="button" className={btn} disabled={busy || nd === null}
            onClick={() => run(() => rotateFurniture(worldId, entity.id, degreesToRadians(normaliseDegrees(nd as number)), parent))}>Turn</button>
        </div>
      </div>

      <div className="space-y-1">
        <label className="text-[11px] text-[var(--text-secondary)]" htmlFor={`swap-${entity.id}`}>Swap for a catalog piece</label>
        <div className="flex items-center gap-1.5">
          <select id={`swap-${entity.id}`} className={field} value={pick} onChange={(e) => setPick(e.target.value)} disabled={blocked !== null || catalog.length === 0}>
            <option value="">Choose a piece...</option>
            {catalog.map((c) => <option key={c.id} value={c.id}>{catalogLabel(c)}</option>)}
          </select>
          <button type="button" className={btn} disabled={busy || !pick || blocked !== null}
            onClick={() => run(() => replaceFurniture(worldId, entity.id, pick, parent))}>Swap</button>
        </div>
        <p className="text-[10px] text-[var(--text-secondary)]">{blocked ?? REPLACE_NOTE}</p>
      </div>

      <div>
        {confirmDelete ? (
          <div role="alert" className="space-y-1.5">
            <p>{deleteConfirmation(entity.name ?? "")}</p>
            <div className="flex gap-1.5">
              <button type="button" className={`${btn} text-[var(--danger,#e0524a)]`} disabled={busy}
                onClick={() => run(() => deleteFurniture(worldId, entity.id, parent))}>Delete</button>
              <button type="button" className={btn} onClick={() => setConfirmDelete(false)}>Keep it</button>
            </div>
          </div>
        ) : (
          <button type="button" className={btn} onClick={() => setConfirmDelete(true)} disabled={busy}>
            <Trash2 className="h-3.5 w-3.5" aria-hidden /> Delete piece
          </button>
        )}
      </div>

      {error && <p role="alert" className="text-[var(--danger,#e0524a)]">{error}</p>}
    </section>
  );
}
