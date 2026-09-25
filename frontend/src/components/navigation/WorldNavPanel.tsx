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
  ChevronRight,
  Search,
  Maximize2,
  Filter,
  Workflow,
  Layers,
  Building,
  Eye,
  Crosshair,
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
  onSelectPlace?: (place: { name: string; position: [number, number, number] }) => void;
  activeQuery?: SpatialQueryFilter | null;
  onUpdateQuery?: (query: SpatialQueryFilter | null) => void;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
}

// Canonical Left-Panel Navigation Sections
type NavSection = "hierarchy" | "evidence" | "sessions" | "places" | "versions" | "query";

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
  onSelectPlace,
  activeQuery = null,
  onUpdateQuery,
  onOpenRoomConstruction,
}: WorldNavPanelProps) {
  const [section, setSection] = useState<NavSection>("hierarchy");
  const [filterText, setFilterText] = useState(activeQuery?.search || "");
  const [typeFilter, setTypeFilter] = useState<string>(activeQuery?.type || "all");
  const [minConfFilter, setMinConfFilter] = useState<number>(activeQuery?.minConfidence || 0);

  // Tree node expansion state
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({
    root: true,
    "level-0": true,
    "level-upper": true,
    "unenclosed-0": true,
    "spaces-0": true,
  });

  const toggleNode = (nodeKey: string) => {
    setExpandedNodes((prev) => ({
      ...prev,
      [nodeKey]: !prev[nodeKey],
    }));
  };

  const entitiesList = useMemo(
    () => Object.values(worldIR?.entities || {}),
    [worldIR]
  );

  // --------------------------------------------------------------------------
  // CANONICAL HIERARCHY MODEL:
  // WORLD
  //  ├── LEVEL
  //  │    ├── SPACE
  //  │    │    ├── ELEMENT
  //  │    │    └── ELEMENT
  //  │    └── SPACE
  //  ├── EVIDENCE
  //  ├── SESSIONS
  //  └── VERSIONS
  // --------------------------------------------------------------------------
  const hierarchyTree = useMemo(() => {
    if (!worldIR || entitiesList.length === 0) return null;

    // 1. Identify Spaces (Rooms / Corridors / Enclosures)
    const spaceEntities = entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      return t === "room" || t === "space" || t === "corridor";
    });

    // 2. Identify Levels
    const explicitLevels = entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      return t === "level" || t === "floor_level";
    });

    // 3. Map element-to-space containment
    const spaceChildrenMap: Record<string, Entity[]> = {};
    spaceEntities.forEach((s) => {
      spaceChildrenMap[s.id] = [];
    });

    const unassignedElements: Entity[] = [];

    entitiesList.forEach((e) => {
      const t = e.type.toLowerCase();
      if (t === "room" || t === "space" || t === "corridor" || t === "level" || t === "floor_level") {
        return;
      }

      // Check relationships for containment
      let assignedSpaceId: string | null = null;
      for (const rel of e.relationships || []) {
        const kind = (rel.type || (rel as { kind?: string }).kind || "").toLowerCase();
        if (kind === "part_of" && spaceChildrenMap[rel.target_id]) {
          assignedSpaceId = rel.target_id;
          break;
        }
      }

      // Check reverse containment on spaces
      if (!assignedSpaceId) {
        for (const s of spaceEntities) {
          for (const rel of s.relationships || []) {
            const kind = (rel.type || (rel as { kind?: string }).kind || "").toLowerCase();
            if (kind === "contains" && rel.target_id === e.id) {
              assignedSpaceId = s.id;
              break;
            }
          }
          if (assignedSpaceId) break;
        }
      }

      if (assignedSpaceId && spaceChildrenMap[assignedSpaceId]) {
        spaceChildrenMap[assignedSpaceId].push(e);
      } else {
        unassignedElements.push(e);
      }
    });

    // Partition elements into levels
    const groundElements = unassignedElements.filter((e) => {
      const z = e.transform?.position?.z ?? 0;
      return z <= 1.5;
    });

    const upperElements = unassignedElements.filter((e) => {
      const z = e.transform?.position?.z ?? 0;
      return z > 1.5;
    });

    return {
      worldId: worldIR.id || activeWorldId,
      worldName: worldIR.name || activeWorldId,
      coordinateSystem: String(worldIR.coordinate_frame || worldIR.coordinate_system || "metric_enu"),
      explicitLevels,
      spaces: spaceEntities.map((s) => ({
        space: s,
        elements: spaceChildrenMap[s.id] || [],
      })),
      groundElements,
      upperElements,
      totalElements: entitiesList.length,
    };
  }, [worldIR, entitiesList, activeWorldId]);

  // Derived authentic evidence list
  const effectiveEvidence = useMemo(() => {
    const views = (worldIR?.metadata?.depth?.per_view as Array<{
      evidence_id: string;
      residual_median_m?: number;
      inlier_fraction?: number;
    }>) || [];
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
    if (views.length > 0) {
      return views.map((dv) => ({
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
  }, [evidence, worldIR?.metadata?.depth?.per_view]);

interface SpatialAnchor {
  id: string;
  name: string;
  worldId: string;
  position: [number, number, number];
  createdAt?: string;
}

  // Authentic Spatial Anchors & Places
  const effectivePlaces = useMemo<SpatialAnchor[]>(() => {
    const list: SpatialAnchor[] = places.map((p) => ({
      id: p.id,
      name: p.name,
      worldId: p.worldId,
      position: [p.lng, p.lat, 0],
      createdAt: "Saved Place",
    }));

    list.unshift({
      id: "anchor-enu-origin",
      name: "Metric ENU Origin [0, 0, 0]",
      worldId: activeWorldId,
      position: [0, 0, 0],
      createdAt: "System Coordinate Anchor",
    });

    if (entitiesList.length > 0) {
      list.push({
        id: "anchor-reconstruction-center",
        name: "Reconstruction Center",
        worldId: activeWorldId,
        position: [0, 1.2, 0],
        createdAt: "Spatial Bounding Center",
      });
    }
    return list;
  }, [places, activeWorldId, entitiesList]);

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

      {/* Primary Left Nav Tabs */}
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
          active={section === "places"}
          onClick={() => setSection("places")}
          icon={MapPin}
          label="Places"
          count={effectivePlaces.length}
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
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>World Hierarchy</span>
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

            {!hierarchyTree ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-1.5 font-sans">
                <p className="font-medium text-white">No WorldIR entities loaded</p>
                <p className="text-[11px] text-neutral-500">
                  This world has no compiled entities yet. Reconstruct attached session evidence to promote structural geometry.
                </p>
              </div>
            ) : (
              <div className="space-y-1.5 border border-[#1f222b] rounded-lg p-2 bg-[#12141a]">
                {/* 1. Root: WORLD Node */}
                <div className="flex items-center justify-between text-neutral-200 py-1">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <button
                      type="button"
                      onClick={() => toggleNode("root")}
                      className="text-neutral-500 hover:text-white"
                    >
                      {expandedNodes["root"] ? (
                        <ChevronDown className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5" />
                      )}
                    </button>
                    <Globe className="w-3.5 h-3.5 text-[#00e5ff] shrink-0" />
                    <span className="font-semibold text-white truncate">
                      {hierarchyTree.worldName}
                    </span>
                  </div>
                  <span className="text-[10px] text-neutral-500 font-mono">
                    {hierarchyTree.totalElements} entities
                  </span>
                </div>

                {expandedNodes["root"] && (
                  <div className="pl-4 space-y-2 border-l border-neutral-800 ml-2">
                    {/* LEVEL 0 (ENU Ground Plane) */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between py-0.5 text-neutral-300">
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => toggleNode("level-0")}
                            className="text-neutral-500 hover:text-white"
                          >
                            {expandedNodes["level-0"] ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                          </button>
                          <Layers className="w-3.5 h-3.5 text-[#4da3ff] shrink-0" />
                          <span className="font-medium text-neutral-200">
                            Level 0 (ENU 0.0m)
                          </span>
                        </div>
                        <span className="text-[10px] text-neutral-500">
                          {hierarchyTree.groundElements.length +
                            hierarchyTree.spaces.reduce((acc, s) => acc + s.elements.length, 0)}{" "}
                          items
                        </span>
                      </div>

                      {expandedNodes["level-0"] && (
                        <div className="pl-4 space-y-1.5 border-l border-neutral-800 ml-2">
                          {/* SPACES / ROOMS */}
                          {hierarchyTree.spaces.length > 0 ? (
                            hierarchyTree.spaces.map(({ space, elements }) => {
                              const isSpaceExpanded = expandedNodes[space.id] ?? true;
                              const isSelected = space.id === selectedEntityId;
                              return (
                                <div key={space.id} className="space-y-1">
                                  <div
                                    onClick={() => onSelectEntity(isSelected ? null : space.id)}
                                    className={`flex items-center justify-between p-1 rounded transition-colors cursor-pointer ${
                                      isSelected
                                        ? "bg-[#182030] text-[#00e5ff]"
                                        : "hover:bg-neutral-800/60 text-neutral-300"
                                    }`}
                                  >
                                    <div className="flex items-center gap-1.5 min-w-0">
                                      <button
                                        type="button"
                                        onClick={(ev) => {
                                          ev.stopPropagation();
                                          toggleNode(space.id);
                                        }}
                                        className="text-neutral-500 hover:text-white"
                                      >
                                        {isSpaceExpanded ? (
                                          <ChevronDown className="w-3 h-3" />
                                        ) : (
                                          <ChevronRight className="w-3 h-3" />
                                        )}
                                      </button>
                                      <Building className="w-3.5 h-3.5 text-[#35d07f] shrink-0" />
                                      <span className="truncate">{space.name || space.id}</span>
                                    </div>
                                    <div className="flex items-center gap-1 shrink-0">
                                      <span className="text-[10px] text-neutral-500">
                                        {elements.length} elements
                                      </span>
                                      <button
                                        type="button"
                                        onClick={(ev) => {
                                          ev.stopPropagation();
                                          onFrameEntity(space.id);
                                        }}
                                        title="Frame Room [F]"
                                        className="p-0.5 text-neutral-400 hover:text-[#00e5ff]"
                                      >
                                        <Maximize2 className="w-3 h-3" />
                                      </button>
                                    </div>
                                  </div>

                                  {/* Space Elements */}
                                  {isSpaceExpanded && (
                                    <div className="pl-4 space-y-0.5 border-l border-neutral-800 ml-2">
                                      {elements.map((e) => (
                                        <HierarchyElementRow
                                          key={e.id}
                                          entity={e}
                                          isSelected={e.id === selectedEntityId}
                                          onSelect={() => onSelectEntity(e.id === selectedEntityId ? null : e.id)}
                                          onFrame={() => onFrameEntity(e.id)}
                                        />
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })
                          ) : (
                            <div className="p-1.5 rounded bg-[#14161f] border border-dashed border-neutral-800 text-[11px] text-neutral-500 space-y-1">
                              <div className="flex items-center justify-between">
                                <span className="flex items-center gap-1">
                                  <Building className="w-3 h-3 text-neutral-600" />
                                  <span>Spaces: 0 detected</span>
                                </span>
                                <span className="text-[10px] text-[#00e5ff]/70 font-mono">
                                  unclosed rings
                                </span>
                              </div>
                              <p className="text-[10px] text-neutral-600 font-sans">
                                Raw planar geometry available below. Room boundary closure required for space promotion.
                              </p>
                            </div>
                          )}

                          {/* UNENCLOSED / PLANAR STRUCTURAL ELEMENTS */}
                          {hierarchyTree.groundElements.length > 0 && (
                            <div className="space-y-1 pt-1">
                              <div className="flex items-center justify-between text-[11px] text-neutral-400">
                                <span className="flex items-center gap-1">
                                  <Box className="w-3 h-3 text-neutral-500" />
                                  <span>Structural Elements ({hierarchyTree.groundElements.length})</span>
                                </span>
                              </div>
                              <div className="space-y-0.5 max-h-60 overflow-y-auto pr-1">
                                {hierarchyTree.groundElements.map((e) => (
                                  <HierarchyElementRow
                                    key={e.id}
                                    entity={e}
                                    isSelected={e.id === selectedEntityId}
                                    onSelect={() => onSelectEntity(e.id === selectedEntityId ? null : e.id)}
                                    onFrame={() => onFrameEntity(e.id)}
                                  />
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* UPPER LEVELS (if any) */}
                    {hierarchyTree.upperElements.length > 0 && (
                      <div className="space-y-1 pt-1 border-t border-neutral-800/80">
                        <div className="flex items-center justify-between py-0.5 text-neutral-300">
                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={() => toggleNode("level-upper")}
                              className="text-neutral-500 hover:text-white"
                            >
                              {expandedNodes["level-upper"] ? (
                                <ChevronDown className="w-3 h-3" />
                              ) : (
                                <ChevronRight className="w-3 h-3" />
                              )}
                            </button>
                            <Layers className="w-3.5 h-3.5 text-[#b28dff] shrink-0" />
                            <span className="font-medium text-neutral-200">Upper Elevation</span>
                          </div>
                          <span className="text-[10px] text-neutral-500">
                            {hierarchyTree.upperElements.length} items
                          </span>
                        </div>

                        {expandedNodes["level-upper"] && (
                          <div className="pl-4 space-y-0.5 border-l border-neutral-800 ml-2">
                            {hierarchyTree.upperElements.map((e) => (
                              <HierarchyElementRow
                                key={e.id}
                                entity={e}
                                isSelected={e.id === selectedEntityId}
                                onSelect={() => onSelectEntity(e.id === selectedEntityId ? null : e.id)}
                                onFrame={() => onFrameEntity(e.id)}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 2. EVIDENCE: CAMERA POSES & INLIERS                          */}
        {/* ============================================================ */}
        {section === "evidence" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Calibrated Viewpoints ({effectiveEvidence.length})</span>
              <span className="text-[#00e5ff] font-mono">Multi-View SfM</span>
            </div>

            {effectiveEvidence.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 font-sans">
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
                          window.dispatchEvent(
                            new CustomEvent("frame-camera", { detail: { id: ev.id } })
                          );
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
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Attached Sessions ({sessions.length})</span>
              <span className="text-[#00e5ff] font-mono">Capture Ingestion</span>
            </div>

            {sessions.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-2 font-sans">
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
                    onClick={() => onSelectSession?.(s.id)}
                    className="p-2.5 rounded border border-[#1f222b] bg-[#14161f] hover:border-neutral-700 transition-colors cursor-pointer space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-white hover:text-[#00e5ff]">
                        {s.name}
                      </span>
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
        {/* 4. PLACES & SPATIAL ANCHORS                                  */}
        {/* ============================================================ */}
        {section === "places" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Spatial Anchors ({effectivePlaces.length})</span>
              <span className="text-[#00e5ff] font-mono">ENU Coordinate Frame</span>
            </div>

            <div className="space-y-1.5">
              {effectivePlaces.map((pl) => (
                <div
                  key={pl.id}
                  onClick={() => {
                    const pos = pl.position || [0, 0, 0];
                    onSelectPlace?.({ name: pl.name, position: pos as [number, number, number] });
                    window.dispatchEvent(
                      new CustomEvent("frame-position", {
                        detail: { x: pos[0], y: pos[1], z: pos[2] },
                      })
                    );
                  }}
                  className="p-2.5 rounded border border-[#1f222b] bg-[#14161f] hover:border-[#00e5ff]/50 transition-colors cursor-pointer space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-white font-medium">
                      <MapPin className="w-3.5 h-3.5 text-[#00e5ff]" />
                      <span>{pl.name}</span>
                    </div>
                    <Crosshair className="w-3 h-3 text-neutral-500 hover:text-white" />
                  </div>
                  <div className="flex justify-between text-[10px] text-neutral-400">
                    <span>Pos: [{pl.position.map((v: number) => v.toFixed(1)).join(", ")}] m</span>
                    <span className="text-neutral-500">{pl.createdAt}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* 5. VERSIONS: LINEAGE & DIFF                                  */}
        {/* ============================================================ */}
        {section === "versions" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>WorldStore Lineage</span>
              <span className="text-[#00e5ff] font-mono">Immutable Snapshots</span>
            </div>

            {/* Version Diff Launcher */}
            {onCompareVersions && versions.length > 0 && (
              <button
                type="button"
                onClick={() => onCompareVersions(versions[0]?.id, "latest")}
                className="w-full py-1.5 rounded font-medium text-xs bg-[#182030] hover:bg-[#1f2b42] text-[#00e5ff] border border-[#00e5ff]/30 transition-colors flex items-center justify-center gap-1.5 cursor-pointer font-sans"
              >
                <GitBranch className="w-3.5 h-3.5" />
                <span>Compare Versions (Diff)</span>
              </button>
            )}

            {versions.length === 0 ? (
              <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 space-y-1.5 font-sans">
                <p className="font-medium text-white">No version snapshots recorded yet</p>
                <p className="text-[11px] text-neutral-500">
                  Reconstruction or correction commit required to create an initial WorldStore version snapshot.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {versions.map((v) => (
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
                    <p className="text-[11px] text-neutral-400 line-clamp-2 font-sans">
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
        {/* 6. QUERY: SPATIAL QUERY & FILTERING                          */}
        {/* ============================================================ */}
        {section === "query" && (
          <div className="p-3 space-y-3 font-mono text-xs">
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

function HierarchyElementRow({
  entity,
  isSelected,
  onSelect,
  onFrame,
}: {
  entity: Entity;
  isSelected: boolean;
  onSelect: () => void;
  onFrame: () => void;
}) {
  const hexColor = (TYPE_COLORS[entity.type] || TYPE_COLORS.default)
    .toString(16)
    .padStart(6, "0");
  const conf = typeof entity.confidence === "number" ? entity.confidence : 0.5;

  return (
    <div
      onClick={onSelect}
      className={`group p-1.5 rounded border transition-colors cursor-pointer flex items-center justify-between text-[11px] font-mono ${
        isSelected
          ? "bg-[#182030] border-[#00e5ff]"
          : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
      }`}
    >
      <div className="flex items-center gap-1.5 min-w-0">
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ background: `#${hexColor}` }}
        />
        <span className="text-neutral-200 truncate group-hover:text-white">
          {entity.name || entity.id}
        </span>
      </div>

      <div className="flex items-center gap-1 shrink-0">
        <span className="text-[10px] text-neutral-500 font-mono-num">
          {(conf * 100).toFixed(0)}%
        </span>
        <button
          type="button"
          onClick={(ev) => {
            ev.stopPropagation();
            onFrame();
          }}
          title="Frame Entity [F]"
          className="p-0.5 text-neutral-400 hover:text-[#00e5ff] rounded"
        >
          <Maximize2 className="w-3 h-3" />
        </button>
      </div>
    </div>
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
  icon: React.ComponentType<{ className?: string }>;
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
