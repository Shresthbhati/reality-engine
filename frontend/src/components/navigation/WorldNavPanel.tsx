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
  AlertTriangle,
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

// Canonical 8 Left-Panel Navigation Sections
export type NavSection =
  | "worlds"
  | "sessions"
  | "evidence"
  | "levels"
  | "rooms"
  | "corridors"
  | "places"
  | "versions"
  | "query";

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
  const [section, setSection] = useState<NavSection>("levels");
  const [filterText, setFilterText] = useState(activeQuery?.search || "");
  const [typeFilter, setTypeFilter] = useState<string>(activeQuery?.type || "all");
  const [minConfFilter, setMinConfFilter] = useState<number>(activeQuery?.minConfidence || 0);
  const [activeLevelFilter, setActiveLevelFilter] = useState<number | null>(null);

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

  // Derived interior entities
  const roomsList = useMemo(() => {
    return entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      const sub = String((e.custom_properties as Record<string, unknown> | undefined)?.subtype || "").toLowerCase();
      const labels = (e.semantic_labels || []).map((l) => l.toLowerCase());
      return (t === "room" || t === "space") && sub !== "corridor" && !labels.includes("corridor");
    });
  }, [entitiesList]);

  const corridorsList = useMemo(() => {
    return entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      const sub = String((e.custom_properties as Record<string, unknown> | undefined)?.subtype || "").toLowerCase();
      const labels = (e.semantic_labels || []).map((l) => l.toLowerCase());
      return t === "corridor" || sub === "corridor" || labels.includes("corridor");
    });
  }, [entitiesList]);

  const wallsList = useMemo(() => {
    return entitiesList.filter((e) => e.type.toLowerCase() === "wall");
  }, [entitiesList]);

  const openingsList = useMemo(() => {
    return entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      return t === "door" || t === "window";
    });
  }, [entitiesList]);

  const getEntityDimensions = (e: Entity) => {
    const geomId = (e.geometry_ids || [])[0];
    const g = geomId && worldIR?.geometries ? worldIR.geometries[geomId] : null;
    if (g && g.bounds_min && g.bounds_max) {
      const dx = Math.abs(g.bounds_max.x - g.bounds_min.x);
      const dy = Math.abs(g.bounds_max.y - g.bounds_min.y);
      const dz = Math.abs(g.bounds_max.z - g.bounds_min.z);
      const length = Math.max(dx, dz);
      const width = Math.min(dx, dz);
      const height = dy;
      const area = length * width;
      return { length, width, height, area };
    }
    return null;
  };

  const getSpaceMetrics = (spaceId: string) => {
    let boundaryCount = 0;
    let openingCount = 0;
    entitiesList.forEach((e) => {
      const isContained = e.parent_id === spaceId;
      const isRel = (e.relationships || []).some(
        (r) =>
          r.target_id === spaceId ||
          (r as { target_entity_id?: string }).target_entity_id === spaceId
      );
      if (isContained || isRel) {
        if (e.type.toLowerCase() === "wall") boundaryCount++;
        if (e.type.toLowerCase() === "door" || e.type.toLowerCase() === "window")
          openingCount++;
      }
    });
    return { boundaryCount, openingCount };
  };

  const handleIsolateSpace = (spaceId: string) => {
    window.dispatchEvent(new CustomEvent("isolate-space", { detail: { id: spaceId } }));
  };

  const handleFilterLevelAction = (levelIndex: number | null) => {
    setActiveLevelFilter(levelIndex);
    window.dispatchEvent(new CustomEvent("filter-level", { detail: { levelIndex } }));
  };

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

  // Derived building levels
  const levelsList = useMemo(() => {
    const explicit = entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      return t === "level" || t === "floor_level";
    });
    if (explicit.length > 0) {
      return explicit.map((lvl, idx) => ({
        index: idx,
        id: lvl.id,
        name: lvl.name || `Level ${idx}`,
        elevationY: lvl.transform?.position?.y ?? 0,
        elementCount: entitiesList.filter((e) => {
          const l =
            (e.custom_properties as Record<string, unknown> | undefined)?.level ??
            (e.custom_properties as Record<string, unknown> | undefined)?.floor_level;
          return l !== undefined && Number(l) === idx;
        }).length,
      }));
    }
    const groundElements = entitiesList.filter((e) => {
      const y = e.transform?.position?.y ?? 0;
      return y <= 2.2;
    });
    const upperElements = entitiesList.filter((e) => {
      const y = e.transform?.position?.y ?? 0;
      return y > 2.2;
    });
    const list = [
      {
        index: 0,
        id: "level-ground",
        name: "Level 0 (Ground)",
        elevationY: 0.0,
        elementCount: groundElements.length,
      },
    ];
    if (upperElements.length > 0) {
      list.push({
        index: 1,
        id: "level-upper",
        name: "Level 1 (Upper)",
        elevationY: 2.8,
        elementCount: upperElements.length,
      });
    }
    return list;
  }, [entitiesList]);

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
          active={section === "worlds"}
          onClick={() => setSection("worlds")}
          icon={Globe}
          label="Worlds"
          count={worlds.length}
        />
        <NavTabButton
          active={section === "sessions"}
          onClick={() => setSection("sessions")}
          icon={Camera}
          label="Sessions"
          count={sessions.length}
        />
        <NavTabButton
          active={section === "evidence"}
          onClick={() => setSection("evidence")}
          icon={ShieldCheck}
          label="Evidence"
          count={effectiveEvidence.length}
        />
        <NavTabButton
          active={section === "levels"}
          onClick={() => setSection("levels")}
          icon={Layers}
          label="Levels"
          count={levelsList.length}
        />
        <NavTabButton
          active={section === "rooms"}
          onClick={() => setSection("rooms")}
          icon={Building}
          label="Rooms"
          count={roomsList.length}
        />
        <NavTabButton
          active={section === "corridors"}
          onClick={() => setSection("corridors")}
          icon={Workflow}
          label="Corridors"
          count={corridorsList.length}
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
        {/* 1. WORLDS: ACTIVE & DISCOVERED SPATIAL WORLDS                */}
        {/* ============================================================ */}
        {section === "worlds" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Spatial Worlds ({worlds.length})</span>
              <span className="text-[#00e5ff]">WorldIR</span>
            </div>

            <div className="space-y-1.5">
              {worlds.map((w) => {
                const isActive = w.id === activeWorldId;
                return (
                  <div
                    key={w.id}
                    onClick={() => onSelectWorld(w.id)}
                    className={`p-2.5 rounded-lg border transition-colors cursor-pointer space-y-1.5 ${
                      isActive
                        ? "bg-[#182030] border-[#00e5ff] shadow-sm"
                        : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <Globe className="w-3.5 h-3.5 text-[#00e5ff] shrink-0" />
                        <span className="font-semibold text-white truncate text-xs">
                          {w.name}
                        </span>
                      </div>
                      {isActive && (
                        <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-[10px] text-neutral-400 font-mono">
                      <span>Sessions: {w.sessionCount ?? 0}</span>
                      <span>Evidence: {w.evidenceCount ?? 0}</span>
                    </div>
                    <div className="text-[10px] text-neutral-500 truncate pt-1 border-t border-neutral-800">
                      ID: {w.id}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* 2. LEVELS: STOREYS & ELEVATION SLICES                        */}
        {/* ============================================================ */}
        {section === "levels" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Storeys & Levels ({levelsList.length})</span>
              {activeLevelFilter !== null && (
                <button
                  type="button"
                  onClick={() => handleFilterLevelAction(null)}
                  className="text-amber-400 hover:underline text-[10px] cursor-pointer"
                >
                  Reset Filter
                </button>
              )}
            </div>

            <div className="space-y-2">
              {levelsList.map((lvl) => {
                const isFiltered = activeLevelFilter === lvl.index;
                return (
                  <div
                    key={lvl.id}
                    className={`p-2.5 rounded-lg border space-y-2 ${
                      isFiltered
                        ? "bg-[#182030] border-[#38bdf8]"
                        : "bg-[#14161f] border-[#1f222b]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <Layers className="w-3.5 h-3.5 text-[#38bdf8] shrink-0" />
                        <span className="font-semibold text-white truncate text-xs">
                          {lvl.name}
                        </span>
                      </div>
                      <span className="text-[10px] text-[#38bdf8] font-mono px-1.5 py-0.2 rounded bg-[#38bdf8]/10">
                        Elev {lvl.elevationY.toFixed(1)}m
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-neutral-400">
                      <span>{lvl.elementCount} elements on level</span>
                      <button
                        type="button"
                        onClick={() =>
                          handleFilterLevelAction(isFiltered ? null : lvl.index)
                        }
                        className={`px-2 py-0.5 rounded text-[10px] border transition-colors cursor-pointer ${
                          isFiltered
                            ? "bg-[#38bdf8] text-black font-semibold border-[#38bdf8]"
                            : "bg-[#38bdf8]/10 text-[#38bdf8] hover:bg-[#38bdf8]/20 border-[#38bdf8]/30"
                        }`}
                      >
                        {isFiltered ? "Active Filter" : "Filter 3D View"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Embedded Level Hierarchy Tree */}
            {hierarchyTree && (
              <div className="space-y-2 pt-2 border-t border-neutral-800">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
                  Level Spatial Structure
                </div>
                {hierarchyTree.spaces.length > 0 && (
                  <div className="space-y-1">
                    {hierarchyTree.spaces.map(({ space, elements }) => {
                      const isExpanded = expandedNodes[space.id] ?? true;
                      const isSelected = space.id === selectedEntityId;
                      return (
                        <div key={space.id} className="space-y-1">
                          <div
                            onClick={() => onSelectEntity(isSelected ? null : space.id)}
                            className={`flex items-center justify-between p-1.5 rounded transition-colors cursor-pointer ${
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
                                {isExpanded ? (
                                  <ChevronDown className="w-3 h-3" />
                                ) : (
                                  <ChevronRight className="w-3 h-3" />
                                )}
                              </button>
                              <Building className="w-3.5 h-3.5 text-[#35d07f] shrink-0" />
                              <span className="truncate">{space.name || space.id}</span>
                            </div>
                            <span className="text-[10px] text-neutral-500 font-mono">
                              {elements.length} elems
                            </span>
                          </div>

                          {isExpanded && (
                            <div className="pl-4 space-y-0.5 border-l border-neutral-800 ml-2">
                              {elements.map((e) => (
                                <HierarchyElementRow
                                  key={e.id}
                                  entity={e}
                                  isSelected={e.id === selectedEntityId}
                                  onSelect={() =>
                                    onSelectEntity(e.id === selectedEntityId ? null : e.id)
                                  }
                                  onFrame={() => onFrameEntity(e.id)}
                                />
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 3. ROOMS: RECONSTRUCTED ENCLOSED SPACES & BOUNDARIES         */}
        {/* ============================================================ */}
        {section === "rooms" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Reconstructed Rooms ({roomsList.length})</span>
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

            {roomsList.length > 0 ? (
              <div className="space-y-2">
                {roomsList.map((room) => {
                  const dims = getEntityDimensions(room);
                  const metrics = getSpaceMetrics(room.id);
                  const conf = typeof room.confidence === "number" ? room.confidence : 0.5;
                  const isSelected = room.id === selectedEntityId;
                  const isLowConf = conf < 0.6;

                  return (
                    <div
                      key={room.id}
                      onClick={() => onSelectEntity(isSelected ? null : room.id)}
                      className={`p-2.5 rounded-lg border transition-all cursor-pointer space-y-2 ${
                        isSelected
                          ? "bg-[#182030] border-[#00e5ff] shadow-md shadow-[#00e5ff]/10"
                          : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <Building className="w-3.5 h-3.5 text-[#3b82f6] shrink-0" />
                          <span className="font-semibold text-white truncate text-xs">
                            {room.name || room.id}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-[#00e5ff] px-1.5 py-0.5 rounded bg-[#00e5ff]/10">
                          {dims ? `${dims.area.toFixed(1)} m²` : "Room"}
                        </span>
                      </div>

                      {/* Dimensions & Boundaries */}
                      <div className="grid grid-cols-2 gap-1 text-[10px] text-neutral-400 pt-1 border-t border-neutral-800">
                        <div>
                          <span className="text-neutral-500">Size: </span>
                          <span className="text-neutral-200">
                            {dims
                              ? `${dims.length.toFixed(1)} × ${dims.width.toFixed(1)} m`
                              : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="text-neutral-500">Height: </span>
                          <span className="text-neutral-200">
                            {dims ? `${dims.height.toFixed(1)} m` : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="text-neutral-500">Walls: </span>
                          <span className="text-neutral-200">
                            {metrics.boundaryCount} bound
                          </span>
                        </div>
                        <div>
                          <span className="text-neutral-500">Openings: </span>
                          <span className="text-neutral-200">
                            {metrics.openingCount} doors/win
                          </span>
                        </div>
                      </div>

                      {/* Low Confidence Warning */}
                      {isLowConf && (
                        <div className="flex items-center gap-1.5 p-1.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400 text-[10px] font-sans">
                          <AlertTriangle className="w-3 h-3 shrink-0" />
                          <span>Room inferred with low confidence ({(conf * 100).toFixed(0)}%)</span>
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex items-center justify-between pt-1 border-t border-neutral-800 text-[10px]">
                        <span className="text-neutral-500">
                          Conf: {(conf * 100).toFixed(0)}%
                        </span>
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              handleIsolateSpace(room.id);
                            }}
                            className="px-2 py-0.5 rounded bg-[#3b82f6]/20 text-[#3b82f6] hover:bg-[#3b82f6]/30 border border-[#3b82f6]/40 cursor-pointer"
                          >
                            Isolate
                          </button>
                          <button
                            type="button"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              onFrameEntity(room.id);
                            }}
                            className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800"
                            title="Frame in Viewport [F]"
                          >
                            <Maximize2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="space-y-3">
                {wallsList.length > 0 ? (
                  <div className="p-3 rounded-lg border border-amber-500/40 bg-amber-500/10 text-xs font-sans space-y-2">
                    <div className="flex items-center gap-2 text-amber-400 font-semibold">
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                      <span>Wall geometry available; room topology unresolved</span>
                    </div>
                    <p className="text-[11px] text-neutral-300 leading-relaxed">
                      Detected {wallsList.length} planar wall segments in point cloud reconstruction, but space boundary loop has not yet been resolved into closed 3D room volumes.
                    </p>
                    {onOpenRoomConstruction && (
                      <button
                        type="button"
                        onClick={onOpenRoomConstruction}
                        className="px-3 py-1.5 rounded bg-amber-500 text-black font-semibold text-xs hover:bg-amber-400 transition-colors cursor-pointer w-full text-center"
                      >
                        Open Room Construction Workflow
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="p-4 rounded-md border border-[#1f222b] bg-[#12141a] text-center text-xs text-neutral-400 font-sans">
                    No room volumes or wall boundaries reconstructed yet.
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* 4. CORRIDORS: CIRCULATION ELEMENTS & INTERCONNECTS           */}
        {/* ============================================================ */}
        {section === "corridors" && (
          <div className="p-3 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
              <span>Circulation Corridors ({corridorsList.length})</span>
            </div>

            {corridorsList.length > 0 ? (
              <div className="space-y-2">
                {corridorsList.map((corridor) => {
                  const dims = getEntityDimensions(corridor);
                  const metrics = getSpaceMetrics(corridor.id);
                  const isSelected = corridor.id === selectedEntityId;
                  const lengthM = dims ? dims.length.toFixed(1) : "—";

                  return (
                    <div
                      key={corridor.id}
                      onClick={() => onSelectEntity(isSelected ? null : corridor.id)}
                      className={`p-2.5 rounded-lg border transition-all cursor-pointer space-y-2 ${
                        isSelected
                          ? "bg-[#182030] border-[#00e5ff]"
                          : "bg-[#14161f] border-[#1f222b] hover:border-neutral-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <Workflow className="w-3.5 h-3.5 text-[#06b6d4] shrink-0" />
                          <span className="font-semibold text-white truncate text-xs">
                            {corridor.name || corridor.id}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-[#06b6d4] px-1.5 py-0.5 rounded bg-[#06b6d4]/10">
                          {lengthM} m route
                        </span>
                      </div>

                      <div className="flex justify-between text-[10px] text-neutral-400 pt-1 border-t border-neutral-800">
                        <span>Openings: {metrics.openingCount}</span>
                        <span>Boundaries: {metrics.boundaryCount}</span>
                      </div>

                      <div className="flex items-center justify-end gap-1.5 pt-1 border-t border-neutral-800 text-[10px]">
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            handleIsolateSpace(corridor.id);
                          }}
                          className="px-2 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4] hover:bg-[#06b6d4]/30 border border-[#06b6d4]/40 cursor-pointer"
                        >
                          Isolate
                        </button>
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            onFrameEntity(corridor.id);
                          }}
                          className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-3 rounded-lg border border-neutral-800 bg-[#12141a] text-xs font-sans space-y-1.5">
                <div className="text-neutral-300 font-semibold flex items-center gap-1.5">
                  <Workflow className="w-3.5 h-3.5 text-neutral-500" />
                  <span>No Circulation Corridors Segmented</span>
                </div>
                <p className="text-[11px] text-neutral-400 leading-relaxed">
                  Reconstruction currently represents direct space interconnects or open-plan circulation. No dedicated circulation corridor entities found in WorldIR.
                </p>
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

            {onCompareVersions && versions.length > 0 && versions[0]?.parentVersionId && (
              <button
                type="button"
                onClick={() => onCompareVersions(versions[0].parentVersionId ?? undefined, versions[0].id)}
                className="w-full py-1.5 rounded font-medium text-xs bg-[#182030] hover:bg-[#1f2b42] text-[#00e5ff] border border-[#00e5ff]/30 transition-colors flex items-center justify-center gap-1.5 cursor-pointer font-sans"
              >
                <GitBranch className="w-3.5 h-3.5" />
                <span>Compare HEAD vs Parent</span>
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
                    <div className="flex items-center justify-between text-[10px] font-mono text-neutral-500 pt-1 border-t border-neutral-800">
                      <span>{v.createdAt} · ID: {v.id}</span>
                      {onCompareVersions && v.parentVersionId && (
                        <button
                          type="button"
                          onClick={() => onCompareVersions(v.parentVersionId ?? undefined, v.id)}
                          className="text-[#00e5ff] hover:underline font-sans shrink-0 cursor-pointer"
                        >
                          vs parent
                        </button>
                      )}
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
