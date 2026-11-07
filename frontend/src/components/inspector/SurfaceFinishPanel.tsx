"use client";

import { useEffect, useState } from "react";
import { Paintbrush } from "lucide-react";
import {
  applyAllLabel, clearFinish, currentFinish, FINISH_NOTE, finishSummary, getFinishOptions, hexToRgb, offersMaterials, rgbToHex, setFinish,
  surfaceType, type FinishOptions, type FinishResult,
} from "@/lib/api/finishes";

interface SurfaceLike { id: string; type?: string; custom_properties?: Record<string, unknown> | null }

const btn = "flex h-7 items-center justify-center gap-1 rounded border border-[var(--border)] px-2 text-xs hover:bg-[var(--surface-hover)] disabled:opacity-40 cursor-pointer";

/** Paint a wall or ceiling, or choose a floor material. A design layer: the survey colour is kept. One saved version per change. */
export default function SurfaceFinishPanel({ worldId, entity, sameTypeIds, parentVersion, onChanged }: {
  worldId: string; entity: SurfaceLike; sameTypeIds: string[]; parentVersion?: string; onChanged: () => void;
}) {
  const type = surfaceType(entity);
  const finish = currentFinish(entity);
  const [options, setOptions] = useState<FinishOptions | null>(null);
  const [custom, setCustom] = useState(rgbToHex(finish?.colour_rgb ?? [200, 200, 198]));
  const [all, setAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // the version each save returns; the next change builds on it at once, until the screen shows a newer one (reload, Undo)
  const [saved, setSaved] = useState<{ version: string; basedOn: string | undefined } | null>(null);
  const parent = saved && saved.basedOn === parentVersion ? saved.version : parentVersion;
  const allLabel = applyAllLabel(type, sameTypeIds.length);
  const also = all ? sameTypeIds.filter((i) => i !== entity.id) : [];

  useEffect(() => {
    let live = true;
    getFinishOptions().then((o) => { if (live) setOptions(o); }).catch(() => { if (live) setOptions(null); });
    return () => { live = false; };
  }, []);

  async function run(action: () => Promise<FinishResult>) {
    setBusy(true);
    setError(null);
    try {
      const res = await action();
      if (res?.version_id) setSaved({ version: res.version_id, basedOn: parentVersion });
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save that finish");
    } finally {
      setBusy(false);
    }
  }

  const customRgb = hexToRgb(custom);
  const swatch = (id: string, name: string, rgb: number[], act: () => Promise<FinishResult>) => (
    <button key={id} type="button" title={name} aria-label={name} disabled={busy} onClick={() => run(act)}
      className="h-6 w-6 rounded border border-[var(--border)] cursor-pointer disabled:opacity-40" style={{ background: rgbToHex(rgb) }} />
  );

  return (
    <section aria-label="Surface finish" className="space-y-3 rounded border border-[var(--border)] p-3 text-xs" data-testid="surface-finish">
      <h4 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
        <Paintbrush className="h-3.5 w-3.5" aria-hidden /> Finish
      </h4>
      <p data-testid="finish-current">{finishSummary(finish)}</p>

      {allLabel && (
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> {allLabel}
        </label>
      )}

      <div className="space-y-1">
        <div className="text-[11px] text-[var(--text-secondary)]">Paint</div>
        <div className="flex flex-wrap gap-1.5">
          {(options?.paints ?? []).map((p) => swatch(p.id, p.name, p.rgb, () => setFinish(worldId, entity.id, { paint: p.id }, also, parent)))}
        </div>
      </div>

      {offersMaterials(type) && (
        <div className="space-y-1">
          <div className="text-[11px] text-[var(--text-secondary)]">Floor material</div>
          <div className="flex flex-wrap gap-1.5">
            {(options?.floor_materials ?? []).map((m) => swatch(m.id, m.name, m.rgb, () => setFinish(worldId, entity.id, { material: m.id }, also, parent)))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-1.5">
        <input aria-label="Custom colour" type="color" value={customRgb ? custom : "#c8c8c6"} onChange={(e) => setCustom(e.target.value)}
          className="h-7 w-10 cursor-pointer rounded border border-[var(--border)] bg-transparent" />
        <button type="button" className={btn} disabled={busy || !customRgb}
          onClick={() => run(() => setFinish(worldId, entity.id, { colour_rgb: customRgb as [number, number, number] }, also, parent))}>Use this colour</button>
        <button type="button" className={btn} disabled={busy || !finish}
          onClick={() => run(() => clearFinish(worldId, entity.id, also, parent))}>Clear finish</button>
      </div>

      <p className="text-[10px] text-[var(--text-secondary)]">{options?.note ?? ""} {FINISH_NOTE}</p>
      {error && <p role="alert" className="text-[var(--danger,#e0524a)]">{error}</p>}
    </section>
  );
}
