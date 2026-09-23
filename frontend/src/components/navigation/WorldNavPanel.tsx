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
  Eye,
  Plus,
  Compass,
  Building,
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
  onSelectSession?: (id: string) => void;
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

// Canonical 5 Left-Panel Sections
type NavSection = "hierarchy" | "evidence" | "sessions" | "versions" | "query";

// Canonical Architectural Primitives
const ARCHITECTURAL_PRIMITIVES = [
  { type: "room", label: "Room", icon: Building },
  { type: "wall", label: "Wall", icon: Box },
  { type: "door", label: "Door", icon: Box },
  { type: "window", label: "Window", icon: Box },
  { type: "floor", label: "Floor", icon: Layers },
  { type: "ceiling", label: "Ceiling", icon: Layers },
  { type: "stair", label: "Stair", icon: Box },
  { type: "corridor", label: "Corridor", icon: Box },
  { type: "roof", label: "Roof", icon: Layers },
  { type: "column", label: "Column", icon: Box },
  { type: "beam", label: "Beam", icon: Box },
] as const;

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
  onSelectSession,
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
  const [section, setSection] = useState<NavSection>("hierarchy");
  const [filterText, setFilterText] = useState(activeQuery?.search || "");
  const [typeFilter, setTypeFilter] = useState<string>(activeQuery?.type || "all");
  const [minConfFilter, setMinConfFilter] = useState<number>(activeQuery?.minConfidence || 0);

  // Architectural Hierarchy State
  const [selectedLevel, setSelectedLevel] = useState<string>("all");
  const [selectedSpace, setSelectedSpace] = useState<string>("all");
  const [selectedPrimitive, setSelectedPrimitive] = useState<string>("all");

  const entitiesList = useMemo(
    () => Object.values(worldIR?.entities || {}),
    [worldIR]
  );

  // Group entities by primitive type
  const primitiveCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    ARCHITECTURAL_PRIMITIVES.forEach((p) => {
      counts[p.type] = 0;
    });
    counts["other"] = 0;

    entitiesList.forEach((e) => {
      const t = e.type.toLowerCase();
      if (counts[t] !== undefined) {
        counts[t]++;
      } else {
        counts["other"]++;
      }
    });
    return counts;
  }, [entitiesList]);

  // Filter entities according to Hierarchy tab selections (Level / Space / Element)
  const hierarchyEntities = useMemo(() => {
    return entitiesList.filter((e) => {
      // Element/Primitive filter
      if (selectedPrimitive !== "all" && e.type.toLowerCase() !== selectedPrimitive) {
        return false;
      }
      // Level filter
      if (selectedLevel !== "all" && e.transform?.position) {
        const z = e.transform.position.z;
        if (selectedLevel === "ground" && (z < -1.5 || z > 1.5)) return false;
        if (selectedLevel === "upper" && z <= 1.5) return false;
      }
      return true;
    }).sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  }, [entitiesList, selectedPrimitive, selectedLevel]);

  // Filtered entities matching Query tab
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
        const matchesType = typeFilter === "all" || e.type.toLowerCase() === typeFilter.toLowerCase();
        const matchesConf = (e.confidence ?? 0) >= minConfFilter;
        return matchesText && matchesType && matchesConf;
      })
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  }, [entitiesList, filterText, typeFilter, minConfFilter]);

  // Derived authentic evidence list
  const depthViews = worldIR?.metadata?.depth?.per_view || [];
  const effectiveEvidence = useMemo(() => {
    if (evidence.length > 0) {
      return evidence.map((e) => ({
        id: e.id,
        name: e.name,
        type: e.type,
        status: e.processingState || "Captured",
        residual: null as string | null,
        inlierFraction: null as number | null,
      }));
    }
    if (depthViews.length > 0) {
      return depthViews.map((dv: any) => ({
        id: dv.evidence_id,
        name: `Pose ${dv.evidence_id}`,
        type: "Calibrated Camera Pose",
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
          <span className="text-[10px] font-mono text-[#00e5ff] px-1.5 py-0.2 rounded bg-[#00e5ff]/10 border border-[#00e5ff]/20">
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
                {w.name}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-neutral-400 absolute right-2.5 top-2.5 pointer-events-none" />
        </div>
      </div>

      {/* Primary 5 Desktop Nav Tabs */}
      <div
        className="flex items-center px-2 py-1.5 border-b gap-1 shrink-0 overflow-x-auto text-xs bg-[#101217]"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <NavTabButton
          active={section === "hierarchy"}
          onClick={() => setSection("hierarchy")}
          icon={Layers}
          label="Hierarchy"
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
          active={section === "versions"}
          onClick={() => setSection("versions")}
          icon={GitBranch}
          label="Versions"
          count={versions.length}
        />
        <NavTabButton
          active={section === "query"}
          onClick={() => setSection("query")}
          icon={Filter}
          label="Query"
          count={activeQuery ? filteredEntities.length : undefined}
        />
      </div>

      {/* Section Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {/* ============================================================ */}
        {/* 1. HIERARCHY: WORLD → LEVEL → SPACE → ELEMENT                */}
        {/* ============================================================ */}
        {section === "hierarchy" && (
          <div className="p-3 space-y-3">
            {/* Level & Space Slices */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
                <span>Spatial Hierarchy</span>
                <span className="text-neutral-500 font-mono">WORLD → LEVEL → SPACE → ELEMENT</span>
              </div>

              {/* Level Slices */}
              <div className="grid grid-cols-3 gap-1">
                <button
                  type="button"
                  onClick={() => setSelectedLevel("all")}
                  className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                    selectedLevel === "all"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                      : "bg-[#151821] text-neutral-400 hover:text-white"
                  }`}
                >
                  All Levels
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedLevel("ground")}
                  className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                    selectedLevel === "ground"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                      : "bg-[#151821] text-neutral-400 hover:text-white"
                  }`}
                >
                  Level 0 (ENU)
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedLevel("upper")}
                  className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                    selectedLevel === "upper"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                      : "bg-[#151821] text-neutral-400 hover:text-white"
                  }`}
                >
                  Upper Levels
                </button>
              </div>
            </div>

            {/* Architectural Primitives Breakdown */}
            <div className="space-y-1.5 pt-2 border-t border-[#1f222b]">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
                <span>Architectural Elements</span>
                <span className="font-mono text-neutral-500">
                  {selectedPrimitive === "all" ? "All Elements" : selectedPrimitive}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-1 max-h-48 overflow-y-auto pr-1">
                <button
                  type="button"
                  onClick={() => setSelectedPrimitive("all")}
                  className={`flex items-center justify-between px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                    selectedPrimitive === "all"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                      : "bg-[#151821] text-neutral-400 hover:text-white"
                  }`}
                >
                  <span>All Primitives</span>
                  <span className="text-[10px] font-semibold">{entitiesList.length}</span>
                </button>

                {ARCHITECTURAL_PRIMITIVES.map((p) => {
                  const count = primitiveCounts[p.type] || 0;
                  const isReconstructed = count > 0;
                  return (
                    <button
                      key={p.type}
                      type="button"
                      onClick={() => setSelectedPrimitive(p.type)}
                      className={`flex items-center justify-between px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                        selectedPrimitive === p.type
                          ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                          : "bg-[#151821] text-neutral-400 hover:text-white"
                      }`}
                    >
                      <span className="capitalize">{p.label}</span>
                      <span
                        className={`text-[10px] px-1 py-0.2 rounded font-semibold ${
                          isReconstructed
                            ? "bg-[#00e5ff]/15 text-[#00e5ff]"
                            : "bg-neutral-800 text-neutral-500"
                        }`}
                      >
                        {count}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Elements List */}
            <div className="space-y-1.5 pt-2 border-t border-[#1f222b]">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
                <span>Compiled Entities ({hierarchyEntities.length})</span>
                {onOpenRoomConstruction && (
                  <button
                    type="button"
                    onClick={onOpenRoomConstruction}
                    className="text-[#00e5ff] hover:underline flex items-center gap-1 font-mono lowercase"
                  >
                    <Workflow className="w-3 h-3" />
                    <span>reconstruct</span>
                  </button>
                )}
              </div>

              {hierarchyEntities.length === 0 ? (
                <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-1.5">
                  <p className="font-medium text-white">
                    {selectedPrimitive === "all"
                      ? "No entities compiled in this World"
                      : `No '${selectedPrimitive}' elements reconstructed`}
                  </p>
                  <p className="text-[11px] text-neutral-500">
                    {selectedPrimitive === "all"
                      ? "Reconstruct attached session evidence or load an authentic WorldIR."
                      : "Primitive classifier not detected in current SfM/depth pass. Showing honest unavailable state."}
                  </p>
                </div>
              ) : (
                <div className="space-y-1">
                  {hierarchyEntities.map((e) => {
                    const isSelected = e.id === selectedEntityId;
                    const hexColor = (TYPE_COLORS[e.type] || TYPE_COLORS.default)
                      .toString(16)
                      .padStart(6, "0");
                    const conf = typeof e.confidence === "number" ? e.confidence : 0.5;

                    return (
                      <div
                        key={e.id}
                        onClick={() => onSelectEntity(isSelected ? null : e.id)}
                        className={`group p-2 rounded border transition-colors cursor-pointer flex items-center justify-between ${
                          isSelected
                            ? "bg-[#182030] border-[#00e5ff] shadow-sm"
                            : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
                        }`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ background: `#${hexColor}` }}
                          />
                          <div className="min-w-0">
                            <div className="font-mono text-xs font-medium text-neutral-200 truncate group-hover:text-white">
                              {e.name || e.id}
                            </div>
                            <div className="flex items-center gap-1.5 text-[10px] text-neutral-400">
                              <span className="capitalize">{e.type}</span>
                              <span>·</span>
                              <span className="font-mono-num">{(conf * 100).toFixed(0)}% conf</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            type="button"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              onFrameEntity(e.id);
                            }}
                            title="Frame Entity [F]"
                            className="p-1 rounded text-neutral-400 hover:text-[#00e5ff] hover:bg-neutral-800 transition-colors"
                          >
                            <Maximize2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* 2. EVIDENCE: CAMERA POSES & INLIERS                          */}
        {/* ============================================================ */}
        {section === "evidence" && (
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Calibrated Viewpoints ({effectiveEvidence.length})</span>
              <span className="text-[#00e5ff] font-mono">Multi-View SfM</span>
            </div>

            {effectiveEvidence.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400">
                No calibrated camera evidence available.
              </div>
            ) : (
              <div className="space-y-1">
                {effectiveEvidence.map((ev) => {
                  const isSelected = ev.id === selectedEvidenceId;
                  return (
                    <div
                      key={ev.id}
                      onClick={() => onSelectEvidence?.(isSelected ? null : ev.id)}
                      className={`group p-2 rounded border transition-colors cursor-pointer flex items-center justify-between ${
                        isSelected
                          ? "bg-[#182030] border-[#00e5ff]"
                          : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <Camera className="w-3.5 h-3.5 text-[#00e5ff] shrink-0" />
                        <div className="min-w-0">
                          <div className="font-mono text-xs font-medium text-neutral-200 truncate group-hover:text-white">
                            {ev.name}
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px] text-neutral-400 font-mono">
                            <span>{ev.status}</span>
                            {ev.residual && (
                              <>
                                <span>·</span>
                                <span className="text-[#35d07f]">{ev.residual}</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectEvidence?.(ev.id);
                        }}
                        title="Frame Camera Viewpoint"
                        className="p-1 rounded text-neutral-400 hover:text-[#00e5ff] hover:bg-neutral-800 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 3. SESSIONS: ATTACHED CAPTURE SESSIONS                       */}
        {/* ============================================================ */}
        {section === "sessions" && (
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Attached Sessions ({sessions.length})</span>
              <span className="text-[#00e5ff] font-mono">Capture Ingestion</span>
            </div>

            {sessions.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-2">
                <p>No capture sessions attached to this World yet.</p>
                <p className="text-[11px] text-neutral-500">
                  Attach an existing session or start a new mobile capture.
                </p>
              </div>
            ) : (
              <div className="space-y-1.5">
                {sessions.map((s) => (
                  <div
                    key={s.id}
                    className="p-2.5 rounded border border-[#1f222b] bg-[#14161f] hover:border-neutral-700 transition-colors cursor-pointer space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <button
                        type="button"
                        onClick={() => onSelectSession?.(s.id)}
                        className="font-semibold text-xs text-white text-left hover:text-[#00e5ff]"
                      >
                        {s.name}
                      </button>
                      <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-neutral-800 text-neutral-300">
                        {s.locationSource || "WGS84"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-neutral-400 font-mono">
                      <span>{s.capturedAt ? s.capturedAt.slice(0, 10) : "Recent"}</span>
                      <span>{s.evidenceCount ?? 0} items</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 4. VERSIONS: LINEAGE & DIFF                                  */}
        {/* ============================================================ */}
        {section === "versions" && (
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>WorldStore Lineage</span>
              <span className="text-[#00e5ff] font-mono">Immutable Snapshots</span>
            </div>

            {/* Version Diff Launcher */}
            {onCompareVersions && versions.length > 0 && (
              <button
                type="button"
                onClick={() => onCompareVersions(versions[0]?.id, "latest")}
                className="w-full py-1.5 rounded font-medium text-xs bg-[#182030] hover:bg-[#1f2b42] text-[#00e5ff] border border-[#00e5ff]/30 transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <GitBranch className="w-3.5 h-3.5" />
                <span>Compare Versions (Diff)</span>
              </button>
            )}

            {versions.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-1.5">
                <p className="font-medium text-white">No version snapshots recorded yet</p>
                <p className="text-[11px] text-neutral-500">
                  Reconstruction or correction commit required to create an initial WorldStore version snapshot.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {versions.map((v, i) => (
                  <div
                    key={v.id}
                    className="p-2.5 rounded border border-[#1f222b] bg-[#14161f] space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            v.isCurrent ? "bg-[#00e5ff] animate-pulse" : "bg-neutral-600"
                          }`}
                        />
                        <span className="font-mono text-xs font-semibold text-white truncate">
                          {v.label}
                        </span>
                      </div>
                      {v.isCurrent && (
                        <span className="text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40">
                          HEAD
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-neutral-400 line-clamp-2">
                      {v.changeSummary}
                    </p>
                    <div className="text-[10px] font-mono text-neutral-500 pt-1 border-t border-neutral-800">
                      {v.createdAt} · ID: {v.id}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 5. QUERY: SPATIAL QUERY & FILTERING                          */}
        {/* ============================================================ */}
        {section === "query" && (
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Spatial Query Engine</span>
              {activeQuery && (
                <button
                  type="button"
                  onClick={handleClearQuery}
                  className="text-neutral-400 hover:text-white text-[11px] font-mono underline"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Search Box */}
            <div className="relative">
              <input
                type="text"
                value={filterText}
                onChange={(e) => handleApplyQuery(undefined, undefined, e.target.value)}
                placeholder="Query ID, semantic label, type..."
                className="w-full h-8 pl-8 pr-2.5 rounded bg-[#181b24] border border-[#232734] text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none font-mono"
              />
              <Search className="w-3.5 h-3.5 text-neutral-400 absolute left-2.5 top-2.5 pointer-events-none" />
            </div>

            {/* Confidence Threshold */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-neutral-400">
                <span>Min Confidence:</span>
                <span className="font-mono text-white">{(minConfFilter * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.9"
                step="0.05"
                value={minConfFilter}
                onChange={(e) => handleApplyQuery(undefined, parseFloat(e.target.value), undefined)}
                className="w-full accent-[#00e5ff] cursor-pointer"
              />
            </div>

            {/* Matching Result Count */}
            <div className="pt-2 border-t border-[#1f222b] text-[11px] font-mono text-neutral-400 flex justify-between">
              <span>Matching Entities:</span>
              <span className="text-[#00e5ff] font-semibold">{filteredEntities.length}</span>
            </div>

            {/* Filtered Entity Results */}
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {filteredEntities.map((e) => (
                <div
                  key={e.id}
                  onClick={() => onSelectEntity(e.id)}
                  className="p-1.5 rounded bg-[#14161f] border border-[#1f222b] hover:border-neutral-700 transition-colors flex items-center justify-between text-xs cursor-pointer"
                >
                  <span className="font-mono text-neutral-200 truncate">{e.id}</span>
                  <span className="text-[10px] font-mono text-neutral-400">
                    {e.type} · {((e.confidence ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
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
  icon: any;
  label: string;
  count?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors shrink-0 cursor-pointer ${
        active
          ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-semibold border border-[#00e5ff]/30"
          : "text-neutral-400 hover:text-white hover:bg-neutral-800/40"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
      {count !== undefined && (
        <span
          className={`text-[10px] px-1 py-0.2 rounded font-mono ${
            active ? "bg-[#00e5ff]/20 text-[#00e5ff]" : "bg-neutral-800 text-neutral-400"
          }`}
        >
          {count}
        </span>
      )}
    </button>
  );
}
