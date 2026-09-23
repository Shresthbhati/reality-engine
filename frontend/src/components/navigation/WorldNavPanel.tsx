"use client";

import { useState, useMemo } from "react";
import {
  Globe,
  Camera,
  ShieldCheck,
  MapPin,
  GitBranch,
  Box,
  ChevronDown,
  Search,
  Maximize2,
  Filter,
  Download,
  Workflow,
  Sparkles,
  Layers,
  Sliders,
  Check,
  AlertCircle,
} from "lucide-react";
import type { WorldIR, Entity } from "@/types/worldir";
import type {
  EvidenceRow,
  PlaceRow,
  SessionRow,
  WorldRow,
  WorldVersionRow,
} from "@/lib/types";
import { TYPE_COLORS } from "@/lib/viewport/three-scene";

export interface SpatialQueryFilter {
  type?: string;
  minConfidence?: number;
  search?: string;
}

interface WorldNavPanelProps {
  worlds: WorldRow[];
  activeWorldId: string;
  onSelectWorld: (id: string) => void;
  worldIR: WorldIR | null;
  sessions: SessionRow[];
  evidence: EvidenceRow[];
  places?: PlaceRow[];
  versions: WorldVersionRow[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
  selectedEvidenceId?: string | null;
  onSelectEvidence?: (id: string | null) => void;
  onCompareVersions?: (base?: string, head?: string) => void;
  onSelectSession?: (id: string) => void;
  onSelectPlace?: (place: { name: string; position: [number, number, number] }) => void;
  activeQuery?: SpatialQueryFilter | null;
  onUpdateQuery?: (query: SpatialQueryFilter | null) => void;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
}

type NavSection =
  | "entities"
  | "evidence"
  | "sessions"
  | "versions"
  | "query"
  | "pipeline"
  | "export"
  | "anchors";

export default function WorldNavPanel({
  worlds,
  activeWorldId,
  onSelectWorld,
  worldIR,
  sessions,
  evidence,
  places = [],
  versions,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
  selectedEvidenceId,
  onSelectEvidence,
  onCompareVersions,
  onSelectSession,
  onSelectPlace,
  activeQuery = null,
  onUpdateQuery,
  onOpenRoomConstruction,
  onExport,
}: WorldNavPanelProps) {
  const [section, setSection] = useState<NavSection>("entities");
  const [filterText, setFilterText] = useState(activeQuery?.search || "");
  const [typeFilter, setTypeFilter] = useState<string>(activeQuery?.type || "all");
  const [minConfFilter, setMinConfFilter] = useState<number>(activeQuery?.minConfidence || 0);

  const entitiesList = useMemo(
    () => Object.values(worldIR?.entities || {}),
    [worldIR]
  );

  // Group entities into spatial hierarchy
  const categorizedEntities = useMemo(() => {
    const structuralPlanes: Entity[] = [];
    const spatialObjects: Entity[] = [];

    entitiesList.forEach((e) => {
      const t = e.type.toLowerCase();
      if (t === "floor" || t === "wall" || t === "ceiling") {
        structuralPlanes.push(e);
      } else {
        spatialObjects.push(e);
      }
    });

    return { structuralPlanes, spatialObjects };
  }, [entitiesList]);

  // Filtered entities matching current query
  const filteredEntities = useMemo(() => {
    return entitiesList
      .filter((e) => {
        const matchesText =
          !filterText ||
          e.id.toLowerCase().includes(filterText.toLowerCase()) ||
          e.type.toLowerCase().includes(filterText.toLowerCase()) ||
          (e.semantic_labels || []).some((l) =>
            l.toLowerCase().includes(filterText.toLowerCase())
          );
        const matchesType = typeFilter === "all" || e.type === typeFilter;
        const matchesConf = (e.confidence ?? 0) >= minConfFilter;
        return matchesText && matchesType && matchesConf;
      })
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  }, [entitiesList, filterText, typeFilter, minConfFilter]);

  const typesCount = useMemo(() => {
    return entitiesList.reduce<Record<string, number>>((acc, e) => {
      acc[e.type] = (acc[e.type] || 0) + 1;
      return acc;
    }, {});
  }, [entitiesList]);

  // Derived authentic evidence list
  const depthViews = worldIR?.metadata?.depth?.per_view || [];
  const effectiveEvidence = useMemo(() => {
    if (evidence.length > 0) {
      return evidence.map((e) => ({
        id: e.id,
        name: e.name,
        type: e.type,
        status: e.processingState,
        residual: null as string | null,
        inlierFraction: null as number | null,
      }));
    }
    if (depthViews.length > 0) {
      return depthViews.map((dv: any) => ({
        id: dv.evidence_id,
        name: `Pose ${dv.evidence_id}`,
        type: "Calibrated Perspective Pose",
        status: dv.inlier_fraction
          ? `${(dv.inlier_fraction * 100).toFixed(0)}% inliers`
          : "Calibrated",
        residual: dv.residual_median_m
          ? `${(dv.residual_median_m * 1000).toFixed(0)} mm res`
          : null,
        inlierFraction: dv.inlier_fraction ?? null,
      }));
    }
    return [];
  }, [evidence, depthViews]);

  // Derived authentic spatial anchors (no fake mock places)
  const authenticAnchors = useMemo(() => {
    const list: { name: string; position: [number, number, number]; tag: string }[] = [];
    
    // ENU Origin Anchor
    list.push({
      name: "Metric ENU Origin",
      position: [0, 0, 0],
      tag: "World Origin (0,0,0)",
    });

    // Check bounds from metadata or entities
    if (entitiesList.length > 0) {
      const sum = entitiesList.reduce(
        (acc, e) => {
          if (e.transform?.position) {
            acc.x += e.transform.position.x;
            acc.y += e.transform.position.y;
            acc.z += e.transform.position.z;
            acc.count++;
          }
          return acc;
        },
        { x: 0, y: 0, z: 0, count: 0 }
      );
      if (sum.count > 0) {
        list.push({
          name: "Entity Cluster Centroid",
          position: [sum.x / sum.count, sum.y / sum.count, sum.z / sum.count],
          tag: `${sum.count} entities centroid`,
        });
      }
    }

    return list;
  }, [entitiesList]);

  const handleApplyQuery = (type?: string, minConf?: number, text?: string) => {
    const nextType = type !== undefined ? type : typeFilter;
    const nextConf = minConf !== undefined ? minConf : minConfFilter;
    const nextText = text !== undefined ? text : filterText;
    
    setTypeFilter(nextType);
    setMinConfFilter(nextConf);
    setFilterText(nextText);

    if (onUpdateQuery) {
      if (nextType === "all" && nextConf === 0 && !nextText) {
        onUpdateQuery(null);
      } else {
        onUpdateQuery({
          type: nextType === "all" ? undefined : nextType,
          minConfidence: nextConf > 0 ? nextConf : undefined,
          search: nextText || undefined,
        });
      }
    }
  };

  const handleClearQuery = () => {
    setTypeFilter("all");
    setMinConfFilter(0);
    setFilterText("");
    onUpdateQuery?.(null);
  };

  return (
    <nav
      className="w-full h-full flex flex-col overflow-hidden bg-[#0e1013] select-none"
      style={{ borderColor: "var(--border)" }}
    >
      {/* World Selector Header */}
      <div
        className="p-3 border-b shrink-0 flex flex-col gap-1.5 bg-[#12141a]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
            Spatial World
          </label>
          <span className="text-[10px] font-mono text-[#00e5ff] px-1.5 py-0.2 rounded bg-[#00e5ff]/10">
            {String(worldIR?.coordinate_frame || worldIR?.coordinate_system || "metric_enu")}
          </span>
        </div>
        <div className="relative">
          <select
            value={activeWorldId}
            onChange={(e) => onSelectWorld(e.target.value)}
            className="w-full h-8 pl-2.5 pr-7 rounded bg-[#181b24] border border-[#232734] text-xs text-white font-medium appearance-none focus:border-[#00e5ff] focus:outline-none cursor-pointer truncate"
          >
            {worlds.map((w) => (
              <option key={w.id} value={w.id} className="bg-[#181b24] text-white">
                {w.name} {w.id === "world-compiled-seed42" ? "★" : ""}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-neutral-400 absolute right-2.5 top-2.5 pointer-events-none" />
        </div>
      </div>

      {/* Navigation Sub-sections Tabs */}
      <div
        className="flex items-center px-2 py-1.5 border-b gap-1 shrink-0 overflow-x-auto text-xs bg-[#101217]"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <NavTabButton
          active={section === "entities"}
          onClick={() => setSection("entities")}
          icon={Box}
          label="Entities"
          count={entitiesList.length}
        />
        <NavTabButton
          active={section === "evidence"}
          onClick={() => setSection("evidence")}
          icon={ShieldCheck}
          label="Evidence"
          count={effectiveEvidence.length}
        />
        <NavTabButton
          active={section === "sessions"}
          onClick={() => setSection("sessions")}
          icon={Camera}
          label="Sessions"
          count={sessions.length}
        />
        <NavTabButton
          active={section === "query"}
          onClick={() => setSection("query")}
          icon={Filter}
          label="Query"
          count={activeQuery ? filteredEntities.length : undefined}
        />
        <NavTabButton
          active={section === "pipeline"}
          onClick={() => setSection("pipeline")}
          icon={Workflow}
          label="Pipeline"
        />
        <NavTabButton
          active={section === "versions"}
          onClick={() => setSection("versions")}
          icon={GitBranch}
          label="Lineage"
          count={versions.length}
        />
        <NavTabButton
          active={section === "anchors"}
          onClick={() => setSection("anchors")}
          icon={MapPin}
          label="Anchors"
          count={authenticAnchors.length}
        />
        <NavTabButton
          active={section === "export"}
          onClick={() => setSection("export")}
          icon={Download}
          label="Export"
        />
      </div>

      {/* Section Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {/* ENTITIES SECTION */}
        {section === "entities" && (
          <div className="flex flex-col h-full">
            {/* Search & Filter Bar */}
            <div className="p-2 border-b space-y-1.5" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-neutral-500 absolute left-2.5 top-2" />
                <input
                  type="text"
                  placeholder="Filter entities..."
                  value={filterText}
                  onChange={(e) => handleApplyQuery(undefined, undefined, e.target.value)}
                  className="w-full h-7 pl-7 pr-2 rounded bg-[#151821] border border-[#1f222b] text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none"
                />
              </div>

              <div className="flex items-center gap-1 overflow-x-auto pb-0.5 text-[11px]">
                <button
                  type="button"
                  onClick={() => handleApplyQuery("all")}
                  className={`px-1.5 py-0.5 rounded transition-colors ${
                    typeFilter === "all"
                      ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-semibold"
                      : "text-neutral-400 hover:text-white"
                  }`}
                >
                  All ({entitiesList.length})
                </button>
                {Object.entries(typesCount).map(([type, count]) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => handleApplyQuery(type)}
                    className={`px-1.5 py-0.5 rounded transition-colors whitespace-nowrap ${
                      typeFilter === type
                        ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-semibold"
                        : "text-neutral-400 hover:text-white"
                    }`}
                  >
                    {type} ({count})
                  </button>
                ))}
              </div>
            </div>

            {/* Entities List */}
            <div className="flex-1 p-1.5 space-y-0.5 overflow-y-auto">
              {filteredEntities.length === 0 ? (
                <div className="p-6 text-center text-xs text-neutral-500 italic space-y-2">
                  <p>{entitiesList.length === 0 ? "No entities compiled in this World." : "No matching entities for query."}</p>
                  {entitiesList.length === 0 && onOpenRoomConstruction && (
                    <button
                      type="button"
                      onClick={onOpenRoomConstruction}
                      className="px-3 py-1.5 rounded text-xs bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/30 hover:bg-[#00e5ff]/30 transition-colors"
                    >
                      Open Room Construction →
                    </button>
                  )}
                </div>
              ) : (
                filteredEntities.map((e) => {
                  const isSel = e.id === selectedEntityId;
                  const hexColor = (TYPE_COLORS[e.type] || TYPE_COLORS.default).toString(16).padStart(6, "0");
                  return (
                    <div
                      key={e.id}
                      onClick={() => onSelectEntity(e.id)}
                      className={`group flex items-center justify-between px-2.5 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                        isSel
                          ? "bg-[#151821] border border-[#00e5ff]/60 shadow-sm"
                          : "hover:bg-[#151821]/70 border border-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: `#${hexColor}` }} />
                        <div className="flex flex-col min-w-0">
                          <span className={`font-mono truncate ${isSel ? "text-white font-medium" : "text-neutral-300"}`}>
                            {e.id}
                          </span>
                          <span className="text-[10px] text-neutral-500 font-mono">
                            {e.type}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0 ml-2">
                        <span className="text-[10px] font-mono-num text-neutral-400">
                          {e.confidence != null ? `${(e.confidence * 100).toFixed(0)}%` : "—"}
                        </span>
                        <button
                          type="button"
                          onClick={(evt) => {
                            evt.stopPropagation();
                            onSelectEntity(e.id);
                            onFrameEntity(e.id);
                          }}
                          title="Frame in 3D [F]"
                          className="opacity-0 group-hover:opacity-100 p-0.5 text-neutral-400 hover:text-[#00e5ff] transition-opacity"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* EVIDENCE SECTION */}
        {section === "evidence" && (
          <div className="p-2 space-y-1.5">
            {effectiveEvidence.length === 0 ? (
              <p className="p-4 text-center text-xs text-neutral-500">No evidence attached to this world.</p>
            ) : (
              effectiveEvidence.map((ev) => {
                const isSel = ev.id === selectedEvidenceId;
                return (
                  <div
                    key={ev.id}
                    onClick={() => {
                      onSelectEvidence?.(ev.id);
                      window.dispatchEvent(new CustomEvent("frame-camera", { detail: { id: ev.id } }));
                    }}
                    className={`p-2 rounded-md border text-xs flex items-center justify-between cursor-pointer transition-colors ${
                      isSel
                        ? "bg-[#151821] border-[#00e5ff] text-white"
                        : "bg-[#151821] border-[#1f222b] hover:border-neutral-600 text-neutral-300"
                    }`}
                  >
                    <div className="flex flex-col min-w-0">
                      <div className="flex items-center gap-1.5">
                        <Camera className="w-3 h-3 text-[#35d07f] shrink-0" />
                        <span className="font-mono font-medium truncate">{ev.name}</span>
                      </div>
                      <span className="text-[10px] text-neutral-500 font-mono mt-0.5">{ev.type}</span>
                    </div>
                    <div className="flex flex-col items-end shrink-0 ml-2">
                      <span className="text-[10px] px-1.5 py-0.5 rounded text-[#2ecc71] bg-[#2ecc71]/10 font-mono">
                        {ev.status}
                      </span>
                      {ev.residual && (
                        <span className="text-[9px] text-neutral-500 font-mono mt-0.5">{ev.residual}</span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* SESSIONS SECTION */}
        {section === "sessions" && (
          <div className="p-2 space-y-1.5">
            {sessions.length === 0 ? (
              <p className="p-4 text-center text-xs text-neutral-500">No sessions attached to this world.</p>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => onSelectSession?.(s.id)}
                  className="p-2.5 rounded-md border border-[#1f222b] bg-[#151821] text-xs space-y-1 hover:border-[#00e5ff]/50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white truncate">{s.name}</span>
                    <span
                      className={`text-[10px] px-1 rounded font-mono ${
                        s.state === "COMPLETE"
                          ? "text-[#2ecc71] bg-[#2ecc71]/10"
                          : "text-[#f5a623] bg-[#f5a623]/10"
                      }`}
                    >
                      {s.state}
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px] text-neutral-400">
                    <span>{s.evidenceCount} Captures</span>
                    <span>{s.capturedAt ?? "Recent"}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* SPATIAL QUERY SECTION */}
        {section === "query" && (
          <div className="p-3 space-y-4 text-xs">
            <div>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400 block mb-1">
                Semantic Classification Query
              </span>
              <select
                value={typeFilter}
                onChange={(e) => handleApplyQuery(e.target.value)}
                className="w-full h-8 px-2 rounded bg-[#151821] border border-[#1f222b] text-white focus:border-[#00e5ff] focus:outline-none"
              >
                <option value="all">All Entity Types</option>
                {Object.keys(typesCount).map((t) => (
                  <option key={t} value={t}>
                    {t} ({typesCount[t]})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div className="flex justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400 mb-1">
                <span>Minimum Confidence Filter</span>
                <span className="font-mono text-[#00e5ff]">{(minConfFilter * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.95"
                step="0.05"
                value={minConfFilter}
                onChange={(e) => handleApplyQuery(undefined, parseFloat(e.target.value))}
                className="w-full accent-[#00e5ff] cursor-pointer"
              />
              <div className="flex justify-between text-[9px] text-neutral-500 font-mono mt-0.5">
                <span>0% (All)</span>
                <span>50% (Standard)</span>
                <span>80% (Verified)</span>
              </div>
            </div>

            <div className="p-3 rounded-md bg-[#151821] border border-[#1f222b] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-neutral-400">Matching Entities</span>
                <span className="font-mono font-bold text-[#00e5ff]">
                  {filteredEntities.length} / {entitiesList.length}
                </span>
              </div>
              <button
                type="button"
                onClick={handleClearQuery}
                className="w-full py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors text-[11px]"
              >
                Reset Spatial Query
              </button>
            </div>
          </div>
        )}

        {/* PIPELINE / ROOM CONSTRUCTION SECTION */}
        {section === "pipeline" && (
          <div className="p-3 space-y-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-neutral-400 font-semibold">
                Room Construction
              </span>
              <span className="text-[10px] font-mono text-[#2ecc71] bg-[#2ecc71]/10 px-1 rounded">
                CANONICAL
              </span>
            </div>

            <div className="space-y-2">
              {[
                { label: "1. Evidence Ingestion", status: effectiveEvidence.length > 0 ? "DONE" : "AWAITING", detail: `${effectiveEvidence.length} camera poses` },
                { label: "2. Camera Calibration (SfM)", status: worldIR ? "CALIBRATED" : "PENDING", detail: worldIR?.metadata?.reconstruction?.registration_status ?? "Verified" },
                { label: "3. Depth Metricization", status: worldIR?.metadata?.depth ? "FUSED" : "SKIPPED", detail: worldIR?.metadata?.depth?.model ?? "DPT / MiDaS" },
                { label: "4. Structural Plane Promotion", status: entitiesList.length > 0 ? "PROMOTED" : "AWAITING", detail: `${entitiesList.length} planes compiled` },
                { label: "5. WorldIR Compilation", status: worldIR ? "COMPILED" : "PENDING", detail: "Canonical Intermediate Representation" },
              ].map((st, idx) => (
                <div key={idx} className="p-2 rounded bg-[#151821] border border-[#1f222b] space-y-1">
                  <div className="flex items-center justify-between font-medium">
                    <span className="text-white">{st.label}</span>
                    <span className="text-[10px] font-mono text-[#00e5ff]">{st.status}</span>
                  </div>
                  <div className="text-[11px] text-neutral-400">{st.detail}</div>
                </div>
              ))}
            </div>

            {onOpenRoomConstruction && (
              <button
                type="button"
                onClick={onOpenRoomConstruction}
                className="w-full py-2 rounded font-medium text-xs bg-[#00e5ff] text-black hover:bg-[#33ebff] transition-colors flex items-center justify-center gap-1.5 cursor-pointer mt-2"
              >
                <Workflow className="w-3.5 h-3.5" />
                <span>Launch Construction Job</span>
              </button>
            )}
          </div>
        )}

        {/* VERSIONS (LINEAGE) SECTION */}
        {section === "versions" && (
          <div className="p-2 space-y-2">
            <div className="flex items-center justify-between px-1">
              <span className="text-[10px] uppercase tracking-wider text-neutral-400 font-semibold">Lineage History</span>
              {onCompareVersions && (
                <button
                  type="button"
                  onClick={() => onCompareVersions()}
                  className="text-[11px] font-medium text-[#00e5ff] hover:underline flex items-center gap-1 cursor-pointer"
                >
                  Compare Versions →
                </button>
              )}
            </div>

            {versions.length === 0 ? (
              <div className="p-3 text-xs bg-[#151821] rounded border border-[#1f222b] text-neutral-400 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white font-mono">v1.0.0 (Root)</span>
                  <span className="text-[10px] font-mono px-1 rounded text-[#00e5ff] bg-[#00e5ff]/10">ACTIVE</span>
                </div>
                <div className="text-[11px] text-neutral-500">Immutable canonical baseline</div>
              </div>
            ) : (
              versions.map((v) => (
                <div
                  key={v.id}
                  className={`p-2.5 rounded-md border text-xs space-y-1.5 ${
                    v.isCurrent ? "bg-[#151821] border-[#00e5ff]/70" : "bg-[#151821] border-[#1f222b]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white font-mono">{v.label}</span>
                    {v.isCurrent && (
                      <span className="text-[10px] font-mono px-1 rounded text-[#00e5ff] bg-[#00e5ff]/10">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-neutral-400 leading-relaxed">
                    {v.changeSummary || "Compiled representation"}
                  </p>
                  {v.createdAt && (
                    <div className="text-[10px] font-mono text-neutral-500 pt-1 border-t border-neutral-800">
                      {v.createdAt}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {/* AUTHENTIC ANCHORS SECTION */}
        {section === "anchors" && (
          <div className="p-2 space-y-1.5">
            {authenticAnchors.map((p, idx) => (
              <div
                key={idx}
                onClick={() => {
                  onSelectPlace?.({ name: p.name, position: p.position });
                  window.dispatchEvent(
                    new CustomEvent("frame-position", { detail: { x: p.position[0], y: p.position[1], z: p.position[2] } })
                  );
                }}
                className="p-2.5 rounded-md border border-[#1f222b] bg-[#151821] text-xs flex items-center justify-between cursor-pointer hover:border-[#00e5ff]/50 transition-colors group"
              >
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-[#00e5ff] shrink-0" />
                    <span className="font-medium text-white group-hover:text-[#00e5ff] transition-colors">{p.name}</span>
                  </div>
                  <span className="text-[10px] text-neutral-400 font-mono mt-0.5">{p.tag}</span>
                </div>
                <Maximize2 className="w-3 h-3 text-neutral-500 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            ))}
          </div>
        )}

        {/* EXPORT SECTION */}
        {section === "export" && (
          <div className="p-3 space-y-2 text-xs">
            <span className="text-[10px] uppercase tracking-wider text-neutral-400 font-semibold block mb-2">
              Export Canonical Artifacts
            </span>

            {[
              { format: "worldir" as const, label: "Export WorldIR (JSON)", desc: "Canonical Intermediate Representation" },
              { format: "ply" as const, label: "Export Point Cloud (PLY)", desc: "Binary float32 3D geometry" },
              { format: "cameras" as const, label: "Export Camera Poses (JSON)", desc: "Oriented perspective viewpoints" },
              { format: "report" as const, label: "Export Pipeline Report (JSON)", desc: "Calibration residuals and facts" },
            ].map((exp) => (
              <button
                key={exp.format}
                type="button"
                onClick={() => onExport?.(exp.format)}
                className="w-full p-2.5 rounded-md border border-[#1f222b] bg-[#151821] hover:border-[#00e5ff]/50 hover:bg-[#1a1f2c] transition-all text-left flex items-start gap-2.5 cursor-pointer"
              >
                <Download className="w-4 h-4 text-[#00e5ff] shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">{exp.label}</div>
                  <div className="text-[11px] text-neutral-400">{exp.desc}</div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}

function NavTabButton({
  active,
  onClick,
  icon: Icon,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Box;
  label: string;
  count?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1 px-2 py-1 rounded transition-colors whitespace-nowrap cursor-pointer ${
        active
          ? "text-[#00e5ff] font-medium bg-[rgba(0,229,255,0.12)]"
          : "text-neutral-400 hover:text-white"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
      {count != null && count > 0 && (
        <span className="text-[10px] font-mono text-neutral-500">({count})</span>
      )}
    </button>
  );
}
