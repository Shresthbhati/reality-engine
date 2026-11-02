"use client";

import { useState } from "react";
import { TriangleAlert } from "lucide-react";
import { isApiError, loadGeoContext, saveAnchor, type Site } from "@/lib/api";
import {
  MAP_COVERAGE_NOTE,
  MAP_CONTEXT_LABEL,
  OSM_ATTRIBUTION,
  RADIUS_DEFAULT_M,
  RADIUS_MAX_M,
  RADIUS_MIN_M,
  buildGeoContextPayload,
  draftFromAnchor,
  geoErrorMessage,
  parseLatLonPaste,
  summariseCounts,
  validateAnchor,
  type AnchorDraft,
  type MapContextResult,
} from "@/lib/sites/compose-model";

interface Props {
  site: Site;
  /** Adds the returned world at cell (0,0) with the Map context label. */
  onAddMapWorld: (worldId: string) => Promise<void>;
}

const field = { background: "var(--bg-base)", borderColor: "var(--border)", color: "var(--text-primary)" } as const;
const err = (m?: string) => (m ? <p role="alert" className="text-xs mt-1" style={{ color: "var(--error)" }}>{m}</p> : null);

export default function AnchorMapPanel({ site, onAddMapWorld }: Props) {
  const [draft, setDraft] = useState<AnchorDraft>(() => draftFromAnchor(site.anchor));
  const [paste, setPaste] = useState("");
  const [pasteError, setPasteError] = useState<string | null>(null);
  const [showErrors, setShowErrors] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [radius, setRadius] = useState(RADIUS_DEFAULT_M);
  const [loading, setLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapResult, setMapResult] = useState<MapContextResult | null>(null);

  const check = validateAnchor(draft);
  const hasMapPlacement = site.placements.some((p) => p.label === MAP_CONTEXT_LABEL);

  function applyPaste() {
    const p = parseLatLonPaste(paste);
    if (!p) {
      setPasteError("Paste coordinates like 22.5757, 88.4312 (latitude, longitude).");
      return;
    }
    setPasteError(null);
    setDraft((d) => ({ ...d, lat: String(p.lat), lon: String(p.lon) }));
    setPaste("");
  }

  async function save() {
    setShowErrors(true);
    setSaved(null);
    if (!check.anchor) return;
    try {
      const a = await saveAnchor(site.id, check.anchor);
      setSaveError(null);
      setSaved(`Saved ${a.lat}, ${a.lon} · ${a.rotation_deg}°`);
    } catch (e) {
      setSaveError(isApiError(e) ? (e.status === 404 ? "Anchoring is not available on this server yet." : e.describe()) : (e as Error).message);
    }
  }

  async function loadMap() {
    setShowErrors(true);
    if (!check.anchor) return;
    setLoading(true);
    setMapError(null);
    try {
      const r = await loadGeoContext(buildGeoContextPayload(check.anchor, radius, `${site.name} map context`));
      setMapResult(r);
      if (!hasMapPlacement) await onAddMapWorld(r.world_id);
    } catch (e) {
      setMapError(isApiError(e) ? geoErrorMessage(e.status, e.message) : (e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section aria-label="Anchor and map context" className="flex flex-col gap-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-secondary)" }}>Anchor and map context</h2>

      <div>
        <label htmlFor="anchor-paste" className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Paste “lat, lon” from a map</label>
        <div className="flex gap-2">
          <input id="anchor-paste" value={paste} onChange={(e) => setPaste(e.target.value)} onKeyDown={(e) => e.key === "Enter" && applyPaste()} placeholder="22.5757, 88.4312" className="flex-1 min-w-0 h-9 px-3 rounded-md border text-sm" style={field} />
          <button type="button" onClick={applyPaste} className="h-9 px-3 rounded-md text-sm border hover:bg-[var(--bg-hover)]" style={{ borderColor: "var(--border)", color: "var(--text-primary)" }}>Use</button>
        </div>
        {err(pasteError ?? undefined)}
      </div>

      <div className="grid grid-cols-3 gap-2">
        {([["lat", "Latitude"], ["lon", "Longitude"], ["rotation", "Rotation °"]] as const).map(([k, label]) => (
          <div key={k}>
            <label htmlFor={`anchor-${k}`} className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>{label}</label>
            <input id={`anchor-${k}`} inputMode="decimal" value={draft[k]} aria-invalid={showErrors && !!check.errors[k]} onChange={(e) => { setSaved(null); setDraft({ ...draft, [k]: e.target.value }); }} className="w-full h-9 px-2 rounded-md border text-sm" style={{ ...field, borderColor: showErrors && check.errors[k] ? "var(--error)" : "var(--border)" }} />
          </div>
        ))}
      </div>
      {showErrors && [check.errors.lat, check.errors.lon, check.errors.rotation].map((m, i) => <span key={i}>{err(m)}</span>)}
      <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>Rotation turns the grid clockwise from north. North is the top (back) of the grid.</p>

      <div className="flex items-center gap-2">
        <button type="button" onClick={save} className="h-8 px-3 rounded-md text-sm font-medium" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>Save anchor</button>
        {saved && <span role="status" className="text-xs" style={{ color: "var(--success)" }}>{saved}</span>}
      </div>
      {err(saveError ?? undefined)}

      <div className="border-t pt-3" style={{ borderColor: "var(--border-subtle)" }}>
        <label htmlFor="map-radius" className="flex justify-between text-xs mb-1" style={{ color: "var(--text-secondary)" }}>
          <span>Map radius</span><span>{radius} m</span>
        </label>
        <input id="map-radius" type="range" min={RADIUS_MIN_M} max={RADIUS_MAX_M} step={25} value={radius} onChange={(e) => setRadius(Number(e.target.value))} className="w-full" />
        <button type="button" onClick={loadMap} disabled={loading} className="mt-2 h-8 px-3 rounded-md text-sm font-medium disabled:opacity-50" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
          {loading ? "Loading map data…" : "Load map context"}
        </button>
        {hasMapPlacement && <p className="text-xs mt-1" style={{ color: "var(--text-tertiary)" }}>A “{MAP_CONTEXT_LABEL}” placement already exists; loading again adds a new world version only.</p>}
      </div>

      {mapError && (
        <div role="alert" className="flex items-start gap-2 rounded-md px-3 py-2 text-sm" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
          <span className="flex-1">{mapError}</span>
          <button type="button" className="underline" onClick={loadMap}>Retry</button>
        </div>
      )}

      {mapResult && (
        <div role="status" className="text-xs rounded-md border p-2" style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}>
          <p>Imported {mapResult.element_count} map elements{summariseCounts(mapResult.counts_by_kind) ? `: ${summariseCounts(mapResult.counts_by_kind)}` : ""}. {mapResult.estimated_count} {mapResult.estimated_count === 1 ? "is" : "are"} estimated from incomplete tags. Source: {mapResult.source === "cache" ? "cached" : "live"} map service.</p>
        </div>
      )}

      <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
        {MAP_COVERAGE_NOTE} Map data {OSM_ATTRIBUTION}.
      </p>
    </section>
  );
}
