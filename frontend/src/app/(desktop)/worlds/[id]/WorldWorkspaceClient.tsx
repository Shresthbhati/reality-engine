"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Plus, Camera, ShieldCheck, Workflow, FileText, History } from "lucide-react";
import type { Map as MapLibreMap } from "maplibre-gl";
import WorldMap from "@/components/map/WorldMap";
import WorldInspector from "@/components/ui/WorldInspector";
import StatusBadge from "@/components/ui/StatusBadge";
import EmptyState from "@/components/ui/EmptyState";
import ActivityFeed from "@/components/ui/ActivityFeed";
import { getActivity } from "@/lib/activity";
import type {
  AnalysisRow,
  EvidenceRow,
  PlaceRow,
  ResultRow,
  SessionRow,
  WorldRow,
  WorldVersionRow,
} from "@/lib/types";

const NAV = ["Overview", "Sessions", "Evidence", "Places", "Layers", "Analysis", "Results", "Versions", "Reports"] as const;
const BOTTOM_TABS = ["Timeline", "Activity", "Coverage"] as const;

// MapLibre's style JSON is consumed outside the DOM, so paint properties can't
// reference a CSS custom property like var(--accent). This is the one place a
// literal hex is acceptable — it mirrors the --accent token in globals.css.
const MARKER_COLOR = "#00e5ff";

type LayerKey = "worldCoverage" | "sessions" | "evidence" | "places" | "analysis" | "results";

const LAYER_LABELS: Record<LayerKey, string> = {
  worldCoverage: "World Coverage",
  sessions: "Sessions",
  evidence: "Evidence",
  places: "Places",
  analysis: "Analysis",
  results: "Results",
};

interface WorldWorkspaceClientProps {
  world: WorldRow;
  sessions: SessionRow[];
  evidence: EvidenceRow[];
  places: PlaceRow[];
  versions: WorldVersionRow[];
  results: ResultRow[];
  analysis: AnalysisRow[];
}

export default function WorldWorkspaceClient({
  world,
  sessions,
  evidence,
  places,
  versions,
  results,
  analysis,
}: WorldWorkspaceClientProps) {
  const [nav, setNav] = useState<(typeof NAV)[number]>("Overview");
  const [bottomTab, setBottomTab] = useState<(typeof BOTTOM_TABS)[number]>("Activity");
  const [layerToggles, setLayerToggles] = useState<Record<LayerKey, boolean>>({
    worldCoverage: false,
    sessions: true,
    evidence: true,
    places: true,
    analysis: false,
    results: false,
  });

  const mapRef = useRef<MapLibreMap | null>(null);

  // The map only stays mounted while the user is on Overview/Places (the only
  // tabs that render it). Once it unmounts, WorldMap's own effect calls
  // map.remove(); clear our copy of the reference so we never call a method
  // on a removed map instance from the Layers tab's checkboxes.
  useEffect(() => {
    if (nav !== "Overview" && nav !== "Places") {
      mapRef.current = null;
    }
  }, [nav]);

  const handleMapLoad = useCallback(
    (map: MapLibreMap) => {
      mapRef.current = map;

      const pointsFor: Record<LayerKey, Array<{ lng: number; lat: number }>> = {
        worldCoverage: world.lat != null && world.lng != null ? [{ lng: world.lng, lat: world.lat }] : [],
        sessions: sessions
          .filter((s): s is SessionRow & { lat: number; lng: number } => s.lat != null && s.lng != null)
          .map((s) => ({ lng: s.lng, lat: s.lat })),
        // EvidenceRow carries no lat/lng in the data schema, so this layer's
        // source is always empty until the backend adds evidence coordinates.
        evidence: [],
        places: places.map((p) => ({ lng: p.lng, lat: p.lat })),
        // AnalysisRow carries no lat/lng in the data schema either.
        analysis: [],
        results: results
          .filter((r): r is ResultRow & { lat: number; lng: number } => r.lat != null && r.lng != null)
          .map((r) => ({ lng: r.lng, lat: r.lat })),
      };

      (Object.keys(pointsFor) as LayerKey[]).forEach((key) => {
        const sourceId = `world-${key}-source`;
        const layerId = `world-${key}-layer`;
        if (map.getSource(sourceId)) return;
        map.addSource(sourceId, {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: pointsFor[key].map((p) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [p.lng, p.lat] },
              properties: {},
            })),
          },
        });
        map.addLayer({
          id: layerId,
          type: "circle",
          source: sourceId,
          paint: { "circle-radius": 5, "circle-color": MARKER_COLOR },
          layout: { visibility: layerToggles[key] ? "visible" : "none" },
        });
      });
    },
    [world.lat, world.lng, sessions, places, results, layerToggles]
  );

  const toggleLayer = (key: LayerKey) => {
    setLayerToggles((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      const map = mapRef.current;
      const layerId = `world-${key}-layer`;
      if (map && map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", next[key] ? "visible" : "none");
      }
      return next;
    });
  };

  const flyTo = useCallback((lng: number, lat: number) => {
    mapRef.current?.flyTo({ center: [lng, lat], zoom: 12 });
  }, []);

  const relevantHrefs = new Set<string>([
    ...sessions.map((s) => `/sessions/${s.id}`),
    ...evidence.map((e) => `/evidence/${e.id}`),
    ...analysis.map((a) => `/analysis/${a.id}`),
    ...results.map((r) => `/results/${r.id}`),
  ]);
  const worldActivity = getActivity().filter((e) => relevantHrefs.has(e.href));

  const mapEl = (
    <WorldMap
      center={world.lat != null && world.lng != null ? [world.lng, world.lat] : undefined}
      zoom={world.lat != null ? 12 : undefined}
      onLoad={handleMapLoad}
    />
  );

  return (
    <div className="flex flex-col h-full">
      {/* Context bar */}
      <div className="flex items-center gap-4 px-6 h-14 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <h1 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          {world.name}
        </h1>
        {world.location && (
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            {world.location}
          </span>
        )}
        <div className="flex items-center gap-3 ml-auto text-sm font-mono-num" style={{ color: "var(--text-secondary)" }}>
          {world.coverageKm2 != null && <span>{world.coverageKm2} km²</span>}
          <span>{world.sessionCount} Sessions</span>
          <span>{world.evidenceCount} Evidence</span>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* World nav */}
        <nav className="w-44 shrink-0 border-r flex flex-col gap-0.5 p-2" style={{ borderColor: "var(--border)" }}>
          {NAV.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setNav(item)}
              className="text-left px-2.5 h-8 rounded-md text-sm font-medium transition-colors"
              style={{
                color: nav === item ? "var(--accent)" : "var(--text-secondary)",
                background: nav === item ? "var(--accent-subtle)" : "transparent",
              }}
            >
              {item}
            </button>
          ))}
        </nav>

        {/* Center content */}
        <div className="flex-1 min-w-0">
          {nav === "Overview" ? (
            mapEl
          ) : nav === "Places" ? (
            <div className="flex h-full">
              <PlacesTab places={places} onSelect={flyTo} />
              <div className="flex-1 min-w-0">{mapEl}</div>
            </div>
          ) : nav === "Sessions" ? (
            <SessionsTab sessions={sessions} />
          ) : nav === "Evidence" ? (
            <EvidenceTab evidence={evidence} />
          ) : nav === "Layers" ? (
            <LayersTab toggles={layerToggles} onToggle={toggleLayer} />
          ) : nav === "Analysis" ? (
            <AnalysisTab analysis={analysis} />
          ) : nav === "Results" ? (
            <ResultsTab results={results} />
          ) : nav === "Versions" ? (
            <VersionsTab versions={versions} />
          ) : (
            <NotYetAvailable label={nav} />
          )}
        </div>

        {/* Inspector */}
        <WorldInspector world={world} />
      </div>

      {/* Bottom panel */}
      <div className="shrink-0 border-t" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-1 px-4 h-10 border-b" style={{ borderColor: "var(--border-subtle)" }}>
          {BOTTOM_TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setBottomTab(tab)}
              className="px-2.5 h-7 rounded text-sm font-medium transition-colors"
              style={{
                color: bottomTab === tab ? "var(--accent)" : "var(--text-secondary)",
                background: bottomTab === tab ? "var(--accent-subtle)" : "transparent",
              }}
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="px-6 py-4 text-sm max-h-56 overflow-y-auto" style={{ color: "var(--text-tertiary)" }}>
          {bottomTab === "Coverage" ? (
            <CoverageTab sessions={sessions} />
          ) : bottomTab === "Activity" ? (
            <ActivityFeed events={worldActivity} limit={10} />
          ) : (
            <p>No timeline recorded for this World yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function SessionsTab({ sessions }: { sessions: SessionRow[] }) {
  if (sessions.length === 0) {
    return <EmptyState icon={Camera} message="No Sessions belong to this World yet." actionLabel="Add Session" actionHref="/sessions/new" />;
  }
  return (
    <div className="flex flex-col gap-1 p-4">
      {sessions.map((s) => (
        <Link
          key={s.id}
          href={`/sessions/${s.id}`}
          className="flex items-center justify-between px-3 h-11 rounded-md text-sm transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <span style={{ color: "var(--text-primary)" }}>{s.name}</span>
          <StatusBadge state={s.state} />
        </Link>
      ))}
    </div>
  );
}

function EvidenceTab({ evidence }: { evidence: EvidenceRow[] }) {
  if (evidence.length === 0) {
    return <EmptyState icon={ShieldCheck} message="No Evidence belongs to this World yet." />;
  }
  return (
    <div className="flex flex-col gap-1 p-4">
      {evidence.map((e) => (
        <Link
          key={e.id}
          href={`/evidence/${e.id}`}
          className="flex items-center justify-between px-3 h-11 rounded-md text-sm transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <div className="flex flex-col">
            <span style={{ color: "var(--text-primary)" }}>{e.name}</span>
            <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>{e.type}</span>
          </div>
          <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
            {e.processingState}
          </span>
        </Link>
      ))}
    </div>
  );
}

function PlacesTab({ places, onSelect }: { places: PlaceRow[]; onSelect: (lng: number, lat: number) => void }) {
  if (places.length === 0) {
    return (
      <div className="w-64 shrink-0 border-r p-4" style={{ borderColor: "var(--border)" }}>
        <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
          Saved places are not stored by the backend yet.
        </p>
      </div>
    );
  }
  return (
    <div className="w-64 shrink-0 border-r overflow-y-auto p-2 flex flex-col gap-1" style={{ borderColor: "var(--border)" }}>
      {places.map((p) => (
        <button
          key={p.id}
          type="button"
          onClick={() => onSelect(p.lng, p.lat)}
          className="flex flex-col items-start gap-0.5 px-2.5 py-2 rounded-md text-left text-sm transition-colors"
          style={{ color: "var(--text-primary)" }}
        >
          <span>{p.name}</span>
          <span className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
            {p.lat.toFixed(4)}, {p.lng.toFixed(4)}
          </span>
        </button>
      ))}
    </div>
  );
}

function LayersTab({ toggles, onToggle }: { toggles: Record<LayerKey, boolean>; onToggle: (key: LayerKey) => void }) {
  return (
    <div className="p-4 flex flex-col gap-1 max-w-xs">
      {(Object.keys(LAYER_LABELS) as LayerKey[]).map((key) => (
        <label
          key={key}
          className="flex items-center gap-2.5 px-2.5 h-9 rounded-md text-sm cursor-pointer transition-colors"
          style={{ color: "var(--text-primary)" }}
        >
          <input
            type="checkbox"
            checked={toggles[key]}
            onChange={() => onToggle(key)}
            style={{ accentColor: "var(--accent)" }}
            className="w-4 h-4"
          />
          {LAYER_LABELS[key]}
        </label>
      ))}
    </div>
  );
}

function AnalysisTab({ analysis }: { analysis: AnalysisRow[] }) {
  if (analysis.length === 0) {
    return <EmptyState icon={Workflow} message="No Analysis has been run for this World yet." />;
  }
  return (
    <div className="flex flex-col gap-1 p-4">
      {analysis.map((a) => (
        <Link
          key={a.id}
          href={`/analysis/${a.id}`}
          className="flex items-center justify-between px-3 h-11 rounded-md text-sm transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <span style={{ color: "var(--text-primary)" }}>{a.name}</span>
          <StatusBadge state={a.state} />
        </Link>
      ))}
    </div>
  );
}

function ResultsTab({ results }: { results: ResultRow[] }) {
  if (results.length === 0) {
    return <EmptyState icon={FileText} message="No Results have been generated for this World yet." />;
  }
  return (
    <div className="flex flex-col gap-1 p-4">
      {results.map((r) => (
        <Link
          key={r.id}
          href={`/results/${r.id}`}
          className="flex items-center justify-between px-3 h-11 rounded-md text-sm transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <span style={{ color: "var(--text-primary)" }}>{r.title}</span>
          <span className="font-mono-num text-xs" style={{ color: "var(--text-tertiary)" }}>
            {r.generatedAt ?? "Unavailable"}
          </span>
        </Link>
      ))}
    </div>
  );
}

function VersionsTab({ versions }: { versions: WorldVersionRow[] }) {
  if (versions.length === 0) {
    return <EmptyState icon={History} message="No versions recorded for this World yet." />;
  }
  return (
    <div className="flex flex-col gap-1.5 p-4">
      {versions.map((v) => (
        <div
          key={v.id}
          className="flex flex-col gap-1 px-3 py-2.5 rounded-md"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              {v.label}
            </span>
            <div className="flex items-center gap-2 shrink-0">
              {v.isCurrent && (
                <span
                  className="text-xs font-medium px-2 h-5 rounded flex items-center"
                  style={{ background: "var(--success-subtle)", color: "var(--success)" }}
                >
                  Current
                </span>
              )}
              <span className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                {v.createdAt}
              </span>
            </div>
          </div>
          {v.changeSummary && (
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {v.changeSummary}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function CoverageTab({ sessions }: { sessions: SessionRow[] }) {
  if (sessions.length === 0) {
    return <p>No coverage recorded for this World yet.</p>;
  }
  return (
    <div className="flex flex-col gap-1.5">
      {sessions.map((s) => (
        <div key={s.id} className="flex items-center justify-between">
          <span style={{ color: "var(--text-primary)" }}>{s.name}</span>
          <span className="font-mono-num" style={{ color: "var(--text-secondary)" }}>
            {s.coverageKm2 != null ? `${s.coverageKm2} km²` : "Unavailable"}
          </span>
        </div>
      ))}
    </div>
  );
}

function NotYetAvailable({ label }: { label: string }) {
  return (
    <div className="h-full flex flex-col items-center justify-center gap-2 text-center px-6">
      <Plus className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        {label} isn&apos;t populated for this World yet.
      </p>
    </div>
  );
}
