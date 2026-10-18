"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Eye,
  Layers,
  Map as MapIcon,
  MousePointer2,
  PenLine,
  Route,
  Ruler,
  Satellite,
  SquareSplitHorizontal,
  Image as ImageIcon,
} from "lucide-react";
import GeoLayerMap, { GEO_KIND_TOKEN, type GeoKind, type GeoPoint } from "@/components/map/GeoLayerMap";
import { getEvidence } from "@/lib/api/evidence";
import {
  entityFootprints,
  formatAreaValue,
  formatLatLon,
  formatLength,
  isGeoreferenced,
  planExtent,
  resolveUnitContext,
  unionArea,
} from "@/lib/map/plot-model";
import type { EvidenceRow, SessionRow } from "@/lib/types";
import type { WorldIR } from "@/types/worldir";

interface CameraLike {
  position_m: [number, number, number];
  evidence_id?: string;
}

interface Props {
  worldId: string;
  worldName?: string;
  world: WorldIR | null;
  cameras: { cameras: CameraLike[] } | null;
  worldLat: number | null;
  worldLng: number | null;
  sessions: SessionRow[];
  evidence: EvidenceRow[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  selectedEvidenceId: string | null;
  onSelectEvidence: (id: string | null) => void;
  onOpen3D: () => void;
}

type LayerKey = "entities" | "cameras" | "origin" | "session" | "evidence";

/** Single source for the layer checklist AND the legend, so they cannot drift apart. */
const LAYERS: { key: LayerKey; label: string; token: string; shape: "box" | "dot" }[] = [
  { key: "entities", label: "Entity footprints", token: "--accent", shape: "box" },
  { key: "cameras", label: "Camera positions", token: "--success", shape: "dot" },
  { key: "origin", label: "World origin", token: GEO_KIND_TOKEN.origin, shape: "dot" },
  { key: "session", label: "Session locations", token: GEO_KIND_TOKEN.session, shape: "dot" },
  { key: "evidence", label: "Evidence GPS", token: GEO_KIND_TOKEN.evidence, shape: "dot" },
];

const MAX_GPS_LOOKUPS = 40;

const DISABLED_TOOLS = [
  { label: "Measure", icon: Ruler, title: "Measure lives in the 3D workspace (it carries uncertainty)" },
  { label: "Annotate", icon: PenLine, title: "Not available in the plan view yet" },
  { label: "Point", icon: Crosshair, title: "Not available in the plan view yet" },
  { label: "Route", icon: Route, title: "Not available in the plan view yet" },
];

const niceStep = (span: number) => {
  const raw = span / 8 || 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
};

export default function MapPlotView({
  worldId, worldName, world, cameras, worldLat, worldLng, sessions, evidence,
  selectedEntityId, onSelectEntity, selectedEvidenceId, onSelectEvidence, onOpen3D,
}: Props) {
  const geo = isGeoreferenced(worldLat, worldLng);
  const footprints = useMemo(() => entityFootprints(world), [world]);
  const camList = useMemo(() => cameras?.cameras ?? [], [cameras]);
  const extent = useMemo(() => planExtent(footprints, camList as never), [footprints, camList]);
  const unit = useMemo(() => resolveUnitContext(world), [world]);
  const occupied = useMemo(() => unionArea(footprints), [footprints]);
  const hasPlan = footprints.length > 0 || camList.length > 0;

  const [mode, setMode] = useState<"map" | "split">(geo && hasPlan ? "split" : "map");
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    entities: true, cameras: true, origin: true, session: true, evidence: true,
  });
  const [layersOpen, setLayersOpen] = useState(true);
  const [cursor, setCursor] = useState<string | null>(null);
  const [tab, setTab] = useState<"all" | "IMAGE" | "VIDEO" | "POINT_CLOUD">("all");
  const carouselRef = useRef<HTMLDivElement | null>(null);

  // Real per-evidence GPS: the list endpoint omits location, the detail endpoint has it.
  const sessionIds = useMemo(() => new Set(sessions.map((s) => s.id)), [sessions]);
  const worldEvidence = useMemo(
    () => evidence.filter((e) => e.sessionId && sessionIds.has(e.sessionId)),
    [evidence, sessionIds],
  );
  const [evidenceGps, setEvidenceGps] = useState<GeoPoint[]>([]);
  useEffect(() => {
    let cancelled = false;
    const ids = worldEvidence.slice(0, MAX_GPS_LOOKUPS);
    Promise.allSettled(ids.map((e) => getEvidence(e.id))).then((res) => {
      if (cancelled) return;
      const pts: GeoPoint[] = [];
      res.forEach((r, i) => {
        const loc = r.status === "fulfilled" ? r.value.location : null;
        if (loc && isGeoreferenced(loc.latitude, loc.longitude)) {
          pts.push({ kind: "evidence", lat: loc.latitude, lng: loc.longitude, label: ids[i].name });
        }
      });
      setEvidenceGps(pts);
    });
    return () => { cancelled = true; };
  }, [worldEvidence]);

  const geoPoints = useMemo<GeoPoint[]>(() => {
    const pts: GeoPoint[] = [];
    if (geo) pts.push({ kind: "origin", lat: worldLat as number, lng: worldLng as number, label: worldName ?? "World origin" });
    sessions.forEach((s) => {
      if (isGeoreferenced(s.lat, s.lng)) pts.push({ kind: "session", lat: s.lat as number, lng: s.lng as number, label: s.name });
    });
    return [...pts, ...evidenceGps];
  }, [geo, worldLat, worldLng, worldName, sessions, evidenceGps]);
  const hasGeoData = geoPoints.length > 0;

  const geoVisible: Record<GeoKind, boolean> = { origin: layers.origin, session: layers.session, evidence: layers.evidence };
  const showGeo = hasGeoData && (mode === "split" || !hasPlan);
  const showPlan = hasPlan && (mode === "split" || !hasGeoData);

  const filtered = worldEvidence.filter((e) => tab === "all" || e.type === tab);
  const count = (t: EvidenceRow["type"]) => worldEvidence.filter((e) => e.type === t).length;
  const reconstruction = world?.metadata?.reconstruction;
  const toggle = (k: LayerKey) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  const switchBtn = (active: boolean, disabled = false) =>
    `flex items-center gap-1.5 px-3 h-8 rounded text-xs font-medium border transition-colors ${
      active
        ? "bg-[var(--accent-subtle)] text-[var(--accent)] border-[var(--accent-border)]"
        : "text-[var(--text-secondary)] border-transparent"
    } ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer hover:text-[var(--text-primary)]"}`;

  return (
    <div className="absolute inset-0 bg-[var(--bg-base)] text-[var(--text-primary)] overflow-hidden" data-testid="map-plot-view">
      {/* Canvas */}
      <div className="absolute inset-0 right-[300px] bottom-7 flex">
        {showPlan && (
          <div className="relative flex-1 min-w-0 border-r border-[var(--border)]">
            <PlanCanvas
              key={extent ? `${extent.minX}:${extent.maxX}:${extent.minZ}:${extent.maxZ}` : "empty"}
              footprints={footprints}
              cameras={camList}
              extent={extent}
              unit={unit}
              layers={layers}
              selectedEntityId={selectedEntityId}
              onSelect={onSelectEntity}
              onCursor={setCursor}
            />
          </div>
        )}
        {showGeo && (
          <div className="relative flex-1 min-w-0">
            <GeoLayerMap
              points={geoPoints}
              visible={geoVisible}
              onCursor={(c) => setCursor(c ? formatLatLon(c.lat, c.lng) : null)}
            />
          </div>
        )}
        {!showPlan && !showGeo && <EmptyState />}
      </div>

      {/* View switch */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1 p-1 rounded-md bg-[var(--bg-surface)] border border-[var(--border)]">
        <button type="button" className={switchBtn(mode === "map")} onClick={() => setMode("map")}>
          <MapIcon className="w-3.5 h-3.5" /> 2D Map
        </button>
        <button type="button" className={switchBtn(false)} onClick={onOpen3D}>
          <Box className="w-3.5 h-3.5" /> 3D View
        </button>
        <button
          type="button"
          disabled={!(hasPlan && hasGeoData)}
          title={hasPlan && hasGeoData ? "Plan and geographic map side by side" : "Split needs both geometry and a GPS location"}
          className={switchBtn(mode === "split", !(hasPlan && hasGeoData))}
          onClick={() => setMode("split")}
        >
          <SquareSplitHorizontal className="w-3.5 h-3.5" /> Split View
        </button>
        <button type="button" disabled title="No satellite imagery source is configured (no basemap)" className={switchBtn(false, true)}>
          <Satellite className="w-3.5 h-3.5" /> Satellite
        </button>
      </div>

      {/* Tool rail */}
      <div className="absolute top-16 left-3 z-10 flex flex-col gap-1 p-1 rounded-md bg-[var(--bg-surface)] border border-[var(--border)] w-14">
        <RailButton icon={MousePointer2} label="Select" active title="Click a footprint to select it" />
        <RailButton icon={Layers} label="Layers" active={layersOpen} onClick={() => setLayersOpen((v) => !v)} title="Toggle layer list" />
        {DISABLED_TOOLS.map((t) => <RailButton key={t.label} icon={t.icon} label={t.label} disabled title={t.title} />)}
        <RailButton icon={Eye} label="View" onClick={() => window.dispatchEvent(new CustomEvent("plot-fit"))} title="Fit to extent" />
      </div>

      {/* Layer checklist */}
      {layersOpen && (
        <div className="absolute top-3 right-[316px] z-10 w-52 p-3 rounded-md bg-[var(--bg-surface)] border border-[var(--border)]" data-testid="plot-layers">
          <div className="text-xs font-semibold mb-2">Layers</div>
          {LAYERS.map((l) => (
            <label key={l.key} className="flex items-center gap-2 py-1 text-xs text-[var(--text-secondary)] cursor-pointer">
              <input type="checkbox" checked={layers[l.key]} onChange={() => toggle(l.key)} className="accent-[var(--accent)]" />
              {l.label}
            </label>
          ))}
        </div>
      )}

      {/* Legend (generated from LAYERS) */}
      <div className="absolute bottom-[calc(1.75rem+0.75rem)] left-3 z-10 p-3 rounded-md bg-[var(--bg-surface)] border border-[var(--border)]" data-testid="plot-legend">
        <div className="text-xs font-semibold mb-1.5">Legend</div>
        {LAYERS.map((l) => (
          <div key={l.key} className="flex items-center gap-2 py-0.5 text-[11px] text-[var(--text-secondary)]">
            <span
              className={`inline-block w-3 h-3 ${l.shape === "dot" ? "rounded-full" : "rounded-sm border"}`}
              style={{ background: l.shape === "dot" ? `var(${l.token})` : "transparent", borderColor: `var(${l.token})` }}
            />
            {l.label}
          </div>
        ))}
      </div>

      {/* Evidence carousel */}
      {worldEvidence.length > 0 && (
        <div className="absolute bottom-[calc(1.75rem+0.75rem)] left-48 right-[316px] z-10 p-2 rounded-md bg-[var(--bg-surface)] border border-[var(--border)]" data-testid="plot-carousel">
          <div className="flex items-center gap-1 mb-2 text-[11px]">
            {([
              ["all", `All Evidence (${worldEvidence.length})`],
              ["IMAGE", `Images (${count("IMAGE")})`],
              ["VIDEO", `Videos (${count("VIDEO")})`],
              ["POINT_CLOUD", `Point Cloud (${count("POINT_CLOUD")})`],
            ] as const).map(([k, label]) => (
              <button key={k} type="button" onClick={() => setTab(k)}
                className={`px-2 h-6 rounded border cursor-pointer ${tab === k ? "text-[var(--accent)] border-[var(--accent-border)] bg-[var(--accent-subtle)]" : "text-[var(--text-secondary)] border-transparent"}`}>
                {label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <button type="button" aria-label="Scroll evidence left" className="p-1 text-[var(--text-secondary)] cursor-pointer" onClick={() => carouselRef.current?.scrollBy({ left: -240, behavior: "smooth" })}>
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div ref={carouselRef} className="flex gap-2 overflow-x-auto flex-1 scroll-smooth">
              {filtered.map((e) => (
                <button key={e.id} type="button" title={e.name} onClick={() => onSelectEvidence(e.id)}
                  className={`shrink-0 w-24 h-16 rounded border overflow-hidden flex items-center justify-center bg-[var(--bg-elevated)] cursor-pointer ${selectedEvidenceId === e.id ? "border-[var(--accent)]" : "border-[var(--border)]"}`}
                  style={e.type === "IMAGE" ? { backgroundImage: `url(/api/worlds/${encodeURIComponent(worldId)}/evidence/${encodeURIComponent(e.id)}/image)`, backgroundSize: "cover", backgroundPosition: "center" } : undefined}
                  role={e.type === "IMAGE" ? "img" : undefined}
                  aria-label={e.name}>
                  {e.type !== "IMAGE" && <ImageIcon className="w-4 h-4 text-[var(--text-tertiary)]" />}
                </button>
              ))}
              {filtered.length === 0 && <span className="text-[11px] text-[var(--text-tertiary)] py-5 px-2">No evidence of this type.</span>}
            </div>
            <button type="button" aria-label="Scroll evidence right" className="p-1 text-[var(--text-secondary)] cursor-pointer" onClick={() => carouselRef.current?.scrollBy({ left: 240, behavior: "smooth" })}>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Info panel */}
      <aside className="absolute top-0 right-0 bottom-7 w-[300px] overflow-y-auto border-l border-[var(--border)] bg-[var(--bg-surface)] p-4" data-testid="plot-info">
        <div className="text-base font-semibold truncate">{worldName ?? "World"}</div>
        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          {geo ? formatLatLon(worldLat as number, worldLng as number) : "Local frame - not georeferenced"}
        </div>

        <div className="mt-5 text-sm font-semibold">Plot information</div>
        <dl className="mt-2 text-xs space-y-2">
          <Row label="Entities with geometry" value={String(footprints.length)} />
          <Row label="Occupied footprint" value={footprints.length ? formatAreaValue(occupied, unit) : "-"} title={`Union of entity bounding boxes on the x/z plane. ${unit.reason}`} />
          <Row label="Extent (x by z)" value={extent ? `${formatLength(extent.width, unit)} by ${formatLength(extent.depth, unit)}` : "-"} title={unit.reason} />
          <Row label="Cameras placed" value={String(camList.length)} />
          <Row label="Units" value={unit.unit} title={unit.reason} />
        </dl>
        {unit.unit === "model units" && hasPlan && (
          <p className="mt-2 text-[11px] text-[var(--warning)]">{unit.reason} Sizes are relative, not metres.</p>
        )}

        <div className="mt-5 text-sm font-semibold">Evidence and reconstruction</div>
        <dl className="mt-2 text-xs space-y-2">
          <Row label="Sessions" value={String(sessions.length)} />
          <Row label="Images" value={String(count("IMAGE"))} />
          <Row label="Videos" value={String(count("VIDEO"))} />
          <Row label="Point clouds" value={String(count("POINT_CLOUD"))} />
          <Row label="With GPS" value={`${evidenceGps.length} of ${Math.min(worldEvidence.length, MAX_GPS_LOOKUPS)} checked`} />
          {reconstruction?.cameras_input != null && (
            <Row label="Registered views" value={`${reconstruction.cameras_registered ?? 0} of ${reconstruction.cameras_input}`} />
          )}
        </dl>
      </aside>

      {/* Status bar */}
      <div className="absolute left-0 right-[300px] bottom-0 h-7 px-3 flex items-center gap-4 text-[11px] border-t border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-secondary)]" data-testid="plot-status">
        <span className="font-mono">{cursor ?? (geo ? "Move over the map for coordinates" : "Local frame - not georeferenced")}</span>
        <span className="ml-auto">Basemap: none</span>
        <span>{geo ? "Geographic: lat/lon degrees" : "Frame: local metric-or-model, +Y up"}</span>
      </div>
    </div>
  );
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="flex justify-between gap-3" title={title}>
      <dt className="text-[var(--text-secondary)]">{label}</dt>
      <dd className="font-mono text-right">{value}</dd>
    </div>
  );
}

function RailButton({ icon: Icon, label, active, disabled, onClick, title }: {
  icon: typeof Eye; label: string; active?: boolean; disabled?: boolean; onClick?: () => void; title?: string;
}) {
  return (
    <button type="button" disabled={disabled} onClick={onClick} title={title}
      className={`flex flex-col items-center gap-0.5 py-1.5 rounded text-[10px] ${active ? "text-[var(--accent)] bg-[var(--accent-subtle)]" : "text-[var(--text-secondary)]"} ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer hover:text-[var(--text-primary)]"}`}>
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-md text-center text-xs leading-relaxed text-[var(--text-secondary)]" data-testid="plot-empty">
        <div className="text-sm font-semibold text-[var(--text-primary)] mb-2">Nothing to plot yet</div>
        This world has no reconstructed geometry and no GPS location. A plan needs entities with geometry
        (add photographs and let the reconstruction finish). A geographic map needs a GPS origin on the world,
        or a session or photo captured with location.
      </div>
    </div>
  );
}

function PlanCanvas({ footprints, cameras, extent, unit, layers, selectedEntityId, onSelect, onCursor }: {
  footprints: ReturnType<typeof entityFootprints>;
  cameras: CameraLike[];
  extent: ReturnType<typeof planExtent>;
  unit: ReturnType<typeof resolveUnitContext>;
  layers: Record<LayerKey, boolean>;
  selectedEntityId: string | null;
  onSelect: (id: string | null) => void;
  onCursor: (s: string | null) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const fitBox = useMemo(() => {
    if (!extent) return { x: -5, y: -5, w: 10, h: 10 };
    const w = Math.max(extent.width, 1e-3), h = Math.max(extent.depth, 1e-3);
    const pad = Math.max(w, h) * 0.12;
    return { x: extent.minX - pad, y: extent.minZ - pad, w: w + 2 * pad, h: h + 2 * pad };
  }, [extent]);
  const [vb, setVb] = useState(fitBox);
  useEffect(() => {
    const fit = () => setVb(fitBox);
    window.addEventListener("plot-fit", fit);
    return () => window.removeEventListener("plot-fit", fit);
  }, [fitBox]);

  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const toLocal = (e: { clientX: number; clientY: number }) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return { x: p.x, z: p.y };
  };

  const step = niceStep(Math.max(vb.w, vb.h));
  const gx: number[] = [], gz: number[] = [];
  for (let x = Math.ceil(vb.x / step) * step; x <= vb.x + vb.w; x += step) gx.push(x);
  for (let z = Math.ceil(vb.y / step) * step; z <= vb.y + vb.h; z += step) gz.push(z);
  const stroke = Math.max(vb.w, vb.h) / 600;

  return (
    <svg
      ref={svgRef}
      className="w-full h-full touch-none select-none"
      viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
      data-testid="plan-canvas"
      role="img"
      aria-label="Plan view of entity footprints and camera positions"
      onWheel={(e) => {
        const p = toLocal(e);
        if (!p) return;
        const k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
        setVb((v) => ({ x: p.x - (p.x - v.x) * k, y: p.z - (p.z - v.y) * k, w: v.w * k, h: v.h * k }));
      }}
      onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; (e.target as Element).setPointerCapture?.(e.pointerId); }}
      onPointerMove={(e) => {
        const p = toLocal(e);
        if (p) onCursor(`x ${formatLength(p.x, unit)}   z ${formatLength(p.z, unit)}`);
        const d = drag.current;
        const rect = svgRef.current?.getBoundingClientRect();
        if (!d || !rect) return;
        const dx = e.clientX - d.x, dy = e.clientY - d.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
        if (!d.moved) return;
        d.x = e.clientX; d.y = e.clientY;
        setVb((v) => ({ ...v, x: v.x - (dx * v.w) / rect.width, y: v.y - (dy * v.h) / rect.height }));
      }}
      onPointerLeave={() => onCursor(null)}
      onPointerUp={() => { const moved = drag.current?.moved; drag.current = null; if (!moved) onSelect(null); }}
    >
      <g style={{ stroke: "var(--border-subtle)", strokeWidth: stroke }}>
        {gx.map((x) => <line key={`x${x}`} x1={x} x2={x} y1={vb.y} y2={vb.y + vb.h} />)}
        {gz.map((z) => <line key={`z${z}`} y1={z} y2={z} x1={vb.x} x2={vb.x + vb.w} />)}
      </g>
      {layers.entities && footprints.map((f) => {
        const sel = f.id === selectedEntityId;
        return (
          <rect key={f.id} x={f.minX} y={f.minZ} width={f.maxX - f.minX} height={f.maxZ - f.minZ}
            style={{ fill: sel ? "var(--primary)" : "var(--accent)", fillOpacity: sel ? 0.45 : 0.18, stroke: sel ? "var(--primary)" : "var(--accent)", strokeWidth: stroke * (sel ? 2.5 : 1.2), cursor: "pointer" }}
            onPointerUp={(e) => { if (!drag.current?.moved) { e.stopPropagation(); drag.current = null; onSelect(f.id); } }}>
            <title>{`${f.name} (${f.type})`}</title>
          </rect>
        );
      })}
      {layers.cameras && cameras.map((c, i) => (
        <circle key={c.evidence_id ?? i} cx={c.position_m[0]} cy={c.position_m[2]} r={stroke * 4}
          style={{ fill: "var(--success)", stroke: "var(--bg-surface)", strokeWidth: stroke }}>
          <title>{c.evidence_id ? `Camera for ${c.evidence_id}` : "Camera"}</title>
        </circle>
      ))}
      <text x={vb.x + vb.w * 0.01} y={vb.y + vb.h * 0.03} style={{ fill: "var(--text-tertiary)", fontSize: Math.max(vb.w, vb.h) / 55 }}>
        {`grid ${formatLength(step, unit)} - +x right, +z down`}
      </text>
    </svg>
  );
}
