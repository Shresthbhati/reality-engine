"use client";

import { useState, useMemo, useEffect } from "react";
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
  Network,
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
export type NavSection =
  | "hierarchy"
  | "topology"
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
  const [section, setSection] = useState<NavSection>("hierarchy");
  const [filterText, setFilterText] = useState(activeQuery?.search || "");
  const [typeFilter, setTypeFilter] = useState<string>(activeQuery?.type || "all");
  const [minConfFilter, setMinConfFilter] = useState<number>(activeQuery?.minConfidence || 0);
  const [activeLevelFilter, setActiveLevelFilter] = useState<number | null>(null);

  // Tree node expansion state. Only the root starts expanded: pre-seeding
  // keys like "level-0"/"level-upper" assumed storey ids the data may not
  // contain, and real level ids are not known until WorldIR loads.
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({
    root: true,
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

  // Building levels, derived ONLY from canonical data.
  //
  // This used to fall back to slicing entities at a hard-coded y = 2.2 m and
  // inventing two storeys with the literal ids "level-ground" and
  // "level-upper" (elevations 0.0 / 2.8). A world with no storey
  // information therefore displayed a building the reconstruction never
  // produced -- a frontend-only building model. Levels now come from the
  // compiler's own interior space graph, or from explicit level entities;
  // when neither exists, the list is empty and the UI says so.
  const levelsList = useMemo(() => {
    const metaLevels = ((worldIR?.metadata?.interior_space_graph as Record<string, unknown> | undefined)?.levels as Array<Record<string, unknown>> | undefined) || [];
    if (metaLevels.length > 0) {
      return metaLevels
        .filter((lvl) => typeof lvl.level_id === "string" && lvl.level_id.length > 0)
        .map((lvl, idx) => ({
          index: idx,
          id: String(lvl.level_id),
          name: lvl.name ? String(lvl.name) : `Level ${idx}`,
          elevationY: Number(lvl.elevation_m ?? 0),
          elementCount:
            ((lvl.room_ids as string[] | undefined) || []).length +
            ((lvl.corridor_ids as string[] | undefined) || []).length,
        }));
    }
    const explicit = entitiesList.filter((e) => {
      const t = e.type.toLowerCase();
      return t === "level" || t === "floor_level";
    });
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
  }, [entitiesList, worldIR]);

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
    return places.map((p) => ({
      id: p.id,
      name: p.name,
      worldId: p.worldId,
      position: [p.lng, p.lat, 0],
      createdAt: "Saved Place",
    }));
  }, [places]);

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
          icon={Network}
          label="Hierarchy"
        />
        <NavTabButton
          active={section === "topology"}
          onClick={() => setSection("topology")}
          icon={Workflow}
          label="Topology"
        />
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
        {/* 0. HIERARCHY: BUILDING → LEVELS → ROOMS → CORRIDORS → ...   */}
        {/* ============================================================ */}
        {section === "hierarchy" && (
          <HierarchyExplorer
            worldIR={worldIR}
            entitiesList={entitiesList}
            selectedEntityId={selectedEntityId}
            onSelectEntity={onSelectEntity}
            onFrameEntity={onFrameEntity}
          />
        )}

        {/* ============================================================ */}
        {/* 0B. TOPOLOGY: ROOM ADJACENCY, CORRIDORS, STAIRS, OPENINGS   */}
        {/* ============================================================ */}
        {section === "topology" && (
          <BuildingTopologyInspector
            worldIR={worldIR}
            entitiesList={entitiesList}
            selectedEntityId={selectedEntityId}
            onSelectEntity={onSelectEntity}
            onFrameEntity={onFrameEntity}
          />
        )}

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
              {levelsList.length === 0 ? (
                <p className="text-[11px] text-neutral-500 leading-relaxed">
                  No storeys recorded. This reconstruction produced no interior space
                  graph and no level entities, so there is no canonical level structure
                  to show.
                </p>
              ) : (
                levelsList.map((lvl) => {
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
              }))}
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
                            onClick={() => {
                              const targetId = isSelected ? null : space.id;
                              onSelectEntity(targetId);
                              if (targetId) onFrameEntity(targetId);
                            }}
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
                  const conf =
                    typeof room.confidence === "number" && Number.isFinite(room.confidence)
                      ? room.confidence
                      : null;
                  const isSelected = room.id === selectedEntityId;
                  const isLowConf = conf !== null && conf < 0.6;

                  return (
                    <div
                      key={room.id}
                      onClick={() => {
                        const targetId = isSelected ? null : room.id;
                        onSelectEntity(targetId);
                        if (targetId) onFrameEntity(targetId);
                      }}
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
                          Conf: {conf !== null ? `${(conf * 100).toFixed(0)}%` : "Not available"}
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
                      onClick={() => {
                        const targetId = isSelected ? null : corridor.id;
                        onSelectEntity(targetId);
                        if (targetId) onFrameEntity(targetId);
                      }}
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
  const conf =
    typeof entity.confidence === "number" && Number.isFinite(entity.confidence)
      ? entity.confidence
      : null;

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
          {conf !== null ? `${(conf * 100).toFixed(0)}%` : "Not available"}
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

function HierarchyExplorer({
  worldIR,
  entitiesList,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
}: {
  worldIR: WorldIR | null;
  entitiesList: Entity[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    building: true,
  });

  const toggle = (key: string) => {
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // 1. Building Entity or envelope
  const buildingEntity =
    entitiesList.find((e) => e.type === "building" || e.id.startsWith("building")) || null;

  // 2. Levels (Storeys)
  const levelEntities = entitiesList.filter(
    (e) => e.type === "storey" || e.type === "level" || e.id.startsWith("storey")
  );

  const levels: Array<{ id: string; name: string; entity?: Entity }> =
    levelEntities.length > 0
      ? levelEntities.map((lvl) => ({ id: lvl.id, name: lvl.name || lvl.id, entity: lvl }))
      : [
          {
            id: "level-0",
            name: "Level 0 (Ground)",
          },
        ];

  // 3. Rooms
  const rooms = entitiesList.filter((e) => {
    const t = e.type.toLowerCase();
    const sub = String((e.custom_properties as Record<string, unknown> | undefined)?.subtype || "").toLowerCase();
    const labels = (e.semantic_labels || []).map((l) => l.toLowerCase());
    return (t === "room" || t === "space") && sub !== "corridor" && !labels.includes("corridor");
  });

  // 4. Corridors
  const corridors = entitiesList.filter((e) => {
    const t = e.type.toLowerCase();
    const sub = String((e.custom_properties as Record<string, unknown> | undefined)?.subtype || "").toLowerCase();
    const labels = (e.semantic_labels || []).map((l) => l.toLowerCase());
    return t === "corridor" || sub === "corridor" || labels.includes("corridor");
  });

  // 5. Stairs
  const stairs = entitiesList.filter(
    (e) => e.type === "stairs" || e.type === "stair" || e.id.startsWith("stair")
  );

  // Helper to get boundary walls of a room
  const getRoomWalls = (room: Entity) => {
    const boundaryIds = new Set<string>([
      ...(Array.isArray(room.custom_properties?.boundary_element_ids)
        ? (room.custom_properties.boundary_element_ids as string[])
        : []),
      ...(Array.isArray(room.custom_properties?.wall_ids)
        ? (room.custom_properties.wall_ids as string[])
        : []),
    ]);

    return entitiesList.filter(
      (e) =>
        e.type === "wall" &&
        (boundaryIds.has(e.id) ||
          e.parent_id === room.id ||
          (room.relationships || []).some((r) => r.target_id === e.id))
    );
  };

  // Helper to get doors of a room
  const getRoomDoors = (room: Entity) => {
    const doorIds = new Set<string>(
      Array.isArray(room.custom_properties?.door_ids) ? (room.custom_properties.door_ids as string[]) : []
    );
    return entitiesList.filter(
      (e) =>
        e.type === "door" &&
        (doorIds.has(e.id) ||
          e.parent_id === room.id ||
          (room.relationships || []).some((r) => r.target_id === e.id))
    );
  };

  // Helper to get windows of a room
  const getRoomWindows = (room: Entity) => {
    const winIds = new Set<string>(
      Array.isArray(room.custom_properties?.window_ids)
        ? (room.custom_properties.window_ids as string[])
        : []
    );
    return entitiesList.filter(
      (e) =>
        e.type === "window" &&
        (winIds.has(e.id) ||
          e.parent_id === room.id ||
          (room.relationships || []).some((r) => r.target_id === e.id))
    );
  };

  // Helper to get corridor openings
  const getCorridorOpenings = (corridor: Entity) => {
    return entitiesList.filter(
      (e) =>
        (e.type === "door" || e.type === "window") &&
        (e.parent_id === corridor.id ||
          (corridor.relationships || []).some((r) => r.target_id === e.id) ||
          (e.relationships || []).some((r) => r.target_id === corridor.id))
    );
  };

  // Auto-expand tree branch to reveal selected entity
  useEffect(() => {
    if (!selectedEntityId) return;
    setExpanded((prev) => {
      const next: Record<string, boolean> = { ...prev, building: true };
      for (const lvl of levels) {
        next[`lvl-${lvl.id}`] = true;
      }
      for (const room of rooms) {
        if (room.id === selectedEntityId) {
          next[`room-${room.id}`] = true;
        } else {
          const walls = getRoomWalls(room);
          const doors = getRoomDoors(room);
          const windows = getRoomWindows(room);
          if (
            walls.some((w) => w.id === selectedEntityId) ||
            doors.some((d) => d.id === selectedEntityId) ||
            windows.some((win) => win.id === selectedEntityId)
          ) {
            next[`room-${room.id}`] = true;
          }
        }
      }
      for (const c of corridors) {
        if (c.id === selectedEntityId) {
          next[`corridor-${c.id}`] = true;
        } else {
          const ops = getCorridorOpenings(c);
          if (ops.some((op) => op.id === selectedEntityId)) {
            next[`corridor-${c.id}`] = true;
          }
        }
      }
      return next;
    });
  }, [selectedEntityId, entitiesList]);

  // Loose structural elements not assigned to a room
  const assignedWallIds = new Set<string>();
  rooms.forEach((r) => getRoomWalls(r).forEach((w) => assignedWallIds.add(w.id)));
  const unassignedWalls = entitiesList.filter(
    (e) => e.type === "wall" && !assignedWallIds.has(e.id)
  );

  return (
    <div className="p-3 space-y-3 font-mono text-xs select-none">
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
        <span className="flex items-center gap-1.5">
          <Network className="w-3.5 h-3.5 text-[#00e5ff]" />
          <span>World Hierarchy</span>
        </span>
        <span className="text-neutral-500 font-mono">
          {entitiesList.length} entities
        </span>
      </div>

      <div className="space-y-1 text-[11px]">
        {/* ROOT: BUILDING */}
        <div className="rounded-lg border border-[#f59e0b]/40 bg-[#f59e0b]/5 overflow-hidden">
          <div
            onClick={() => {
              if (buildingEntity) {
                onSelectEntity(buildingEntity.id);
                onFrameEntity(buildingEntity.id);
              }
            }}
            className="flex items-center justify-between p-2 hover:bg-[#f59e0b]/10 cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-1.5 min-w-0">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  toggle("building");
                }}
                className="p-0.5 text-neutral-400 hover:text-white cursor-pointer"
              >
                {expanded.building ? (
                  <ChevronDown className="w-3.5 h-3.5" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5" />
                )}
              </button>
              <Building className="w-3.5 h-3.5 text-[#f59e0b] shrink-0" />
              <span className="font-semibold text-white truncate">
                {buildingEntity?.name || worldIR?.name || "Building Envelope"}
              </span>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <span className="text-[10px] text-[#f59e0b] px-1 rounded bg-[#f59e0b]/20">
                {levels.length} Level{levels.length > 1 ? "s" : ""}
              </span>
              {buildingEntity && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onFrameEntity(buildingEntity.id);
                  }}
                  title="Frame Building [F]"
                  className="p-1 text-neutral-400 hover:text-white"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>

          {/* BUILDING CHILDREN: LEVELS */}
          {expanded.building && (
            <div className="pl-4 pr-2 pb-2 space-y-2 border-t border-[#f59e0b]/20 pt-2">
              {levels.map((lvl) => {
                const lvlKey = `lvl-${lvl.id}`;
                const isLvlExpanded = expanded[lvlKey] ?? true;

                // Elements belonging to this level
                const lvlRooms =
                  levels.length === 1
                    ? rooms
                    : rooms.filter(
                        (r) =>
                          r.parent_id === lvl.id ||
                          (r.custom_properties?.storey_id as string) === lvl.id ||
                          (r.custom_properties?.level_id as string) === lvl.id ||
                          (r.relationships || []).some((rel) => rel.target_id === lvl.id)
                      );

                const lvlCorridors =
                  levels.length === 1
                    ? corridors
                    : corridors.filter(
                        (c) =>
                          c.parent_id === lvl.id ||
                          (c.custom_properties?.storey_id as string) === lvl.id ||
                          (c.relationships || []).some((rel) => rel.target_id === lvl.id)
                      );

                const lvlStairs = stairs;

                return (
                  <div
                    key={lvl.id}
                    className="rounded border border-[#8b5cf6]/30 bg-[#8b5cf6]/5 overflow-hidden"
                  >
                    {/* LEVEL HEADER */}
                    <div
                      onClick={() => {
                        if (lvl.entity) {
                          onSelectEntity(lvl.entity.id);
                          onFrameEntity(lvl.entity.id);
                        }
                      }}
                      className="flex items-center justify-between p-1.5 hover:bg-[#8b5cf6]/10 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center gap-1.5 min-w-0">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggle(lvlKey);
                          }}
                          className="p-0.5 text-neutral-400 hover:text-white cursor-pointer"
                        >
                          {isLvlExpanded ? (
                            <ChevronDown className="w-3 h-3" />
                          ) : (
                            <ChevronRight className="w-3 h-3" />
                          )}
                        </button>
                        <Layers className="w-3.5 h-3.5 text-[#8b5cf6] shrink-0" />
                        <span className="font-semibold text-neutral-200 truncate">
                          {lvl.name}
                        </span>
                      </div>

                      <div className="flex items-center gap-1 text-[10px] text-neutral-400 shrink-0">
                        <span>
                          {lvlRooms.length}R · {lvlCorridors.length}C
                        </span>
                        {lvl.entity && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onFrameEntity(lvl.entity!.id);
                            }}
                            title="Frame Level [F]"
                            className="p-0.5 text-neutral-400 hover:text-white"
                          >
                            <Maximize2 className="w-2.5 h-2.5" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* LEVEL CHILDREN: ROOMS, CORRIDORS, STAIRS */}
                    {isLvlExpanded && (
                      <div className="pl-3 pr-1.5 pb-1.5 pt-1 space-y-2 border-t border-[#8b5cf6]/20">
                        {/* 1. ROOMS OF THIS LEVEL */}
                        {lvlRooms.length > 0 && (
                          <div className="space-y-1">
                            <span className="text-[9px] text-[#3b82f6] uppercase tracking-wider block font-semibold">
                              Rooms ({lvlRooms.length})
                            </span>
                            <div className="space-y-1 pl-1">
                              {lvlRooms.map((room) => {
                                const roomKey = `room-${room.id}`;
                                const isRoomExpanded = expanded[roomKey] ?? false;
                                const isSelected = selectedEntityId === room.id;
                                const walls = getRoomWalls(room);
                                const doors = getRoomDoors(room);
                                const windows = getRoomWindows(room);
                                const area = room.custom_properties?.floor_area_m2 as number | undefined;

                                return (
                                  <div
                                    key={room.id}
                                    className={`rounded border transition-colors ${
                                      isSelected
                                        ? "border-[#00e5ff] bg-[#00e5ff]/10"
                                        : "border-[#3b82f6]/20 bg-[#3b82f6]/5"
                                    }`}
                                  >
                                    <div
                                      onClick={() => {
                                        onSelectEntity(room.id);
                                        onFrameEntity(room.id);
                                      }}
                                      className="flex items-center justify-between p-1 hover:bg-[#3b82f6]/10 cursor-pointer"
                                    >
                                      <div className="flex items-center gap-1 min-w-0">
                                        <button
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            toggle(roomKey);
                                          }}
                                          className="p-0.5 text-neutral-400 hover:text-white cursor-pointer"
                                        >
                                          {isRoomExpanded ? (
                                            <ChevronDown className="w-2.5 h-2.5" />
                                          ) : (
                                            <ChevronRight className="w-2.5 h-2.5" />
                                          )}
                                        </button>
                                        <Building className="w-3 h-3 text-[#3b82f6] shrink-0" />
                                        <span className="text-white truncate">
                                          {room.name || room.id}
                                        </span>
                                      </div>

                                      <div className="flex items-center gap-1 shrink-0 text-[10px]">
                                        {area && (
                                          <span className="text-neutral-400 font-mono">
                                            {area.toFixed(1)}m²
                                          </span>
                                        )}
                                        <button
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            onFrameEntity(room.id);
                                          }}
                                          title="Frame Room [F]"
                                          className="p-0.5 text-neutral-400 hover:text-[#00e5ff]"
                                        >
                                          <Maximize2 className="w-2.5 h-2.5" />
                                        </button>
                                      </div>
                                    </div>

                                    {/* ROOM INTERIORS: WALLS, DOORS, WINDOWS */}
                                    {isRoomExpanded && (
                                      <div className="pl-4 pr-1 pb-1 pt-1 space-y-1 border-t border-[#3b82f6]/15 text-[10px]">
                                        {/* Walls */}
                                        {walls.length > 0 && (
                                          <div>
                                            <span className="text-neutral-500 uppercase tracking-wider text-[8px] block">
                                              Walls ({walls.length})
                                            </span>
                                            <div className="flex flex-wrap gap-1 mt-0.5">
                                              {walls.map((w) => (
                                                <button
                                                  key={w.id}
                                                  type="button"
                                                  onClick={() => {
                                                    onSelectEntity(w.id);
                                                    onFrameEntity(w.id);
                                                  }}
                                                  className={`px-1 py-0.2 rounded border font-mono truncate max-w-[100px] cursor-pointer ${
                                                    selectedEntityId === w.id
                                                      ? "bg-[#f59e0b] text-black border-[#f59e0b]"
                                                      : "bg-[#f59e0b]/15 text-[#f59e0b] border-[#f59e0b]/30 hover:bg-[#f59e0b]/25"
                                                  }`}
                                                >
                                                  {w.id}
                                                </button>
                                              ))}
                                            </div>
                                          </div>
                                        )}

                                        {/* Doors */}
                                        {doors.length > 0 && (
                                          <div>
                                            <span className="text-neutral-500 uppercase tracking-wider text-[8px] block">
                                              Doors ({doors.length})
                                            </span>
                                            <div className="flex flex-wrap gap-1 mt-0.5">
                                              {doors.map((d) => (
                                                <button
                                                  key={d.id}
                                                  type="button"
                                                  onClick={() => {
                                                    onSelectEntity(d.id);
                                                    onFrameEntity(d.id);
                                                  }}
                                                  className={`px-1 py-0.2 rounded border font-mono truncate max-w-[100px] cursor-pointer ${
                                                    selectedEntityId === d.id
                                                      ? "bg-[#10b981] text-black border-[#10b981]"
                                                      : "bg-[#10b981]/15 text-[#10b981] border-[#10b981]/30 hover:bg-[#10b981]/25"
                                                  }`}
                                                >
                                                  {d.id}
                                                </button>
                                              ))}
                                            </div>
                                          </div>
                                        )}

                                        {/* Windows */}
                                        {windows.length > 0 && (
                                          <div>
                                            <span className="text-neutral-500 uppercase tracking-wider text-[8px] block">
                                              Windows ({windows.length})
                                            </span>
                                            <div className="flex flex-wrap gap-1 mt-0.5">
                                              {windows.map((win) => (
                                                <button
                                                  key={win.id}
                                                  type="button"
                                                  onClick={() => {
                                                    onSelectEntity(win.id);
                                                    onFrameEntity(win.id);
                                                  }}
                                                  className={`px-1 py-0.2 rounded border font-mono truncate max-w-[100px] cursor-pointer ${
                                                    selectedEntityId === win.id
                                                      ? "bg-[#06b6d4] text-black border-[#06b6d4]"
                                                      : "bg-[#06b6d4]/15 text-[#06b6d4] border-[#06b6d4]/30 hover:bg-[#06b6d4]/25"
                                                  }`}
                                                >
                                                  {win.id}
                                                </button>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* 2. CORRIDORS OF THIS LEVEL */}
                        {lvlCorridors.length > 0 && (
                          <div className="space-y-1">
                            <span className="text-[9px] text-[#06b6d4] uppercase tracking-wider block font-semibold">
                              Corridors ({lvlCorridors.length})
                            </span>
                            <div className="space-y-1 pl-1">
                              {lvlCorridors.map((corridor) => {
                                const cKey = `corridor-${corridor.id}`;
                                const isCExpanded = expanded[cKey] ?? false;
                                const isSelected = selectedEntityId === corridor.id;
                                const openings = getCorridorOpenings(corridor);

                                return (
                                  <div
                                    key={corridor.id}
                                    className={`rounded border transition-colors ${
                                      isSelected
                                        ? "border-[#00e5ff] bg-[#00e5ff]/10"
                                        : "border-[#06b6d4]/20 bg-[#06b6d4]/5"
                                    }`}
                                  >
                                    <div
                                      onClick={() => {
                                        onSelectEntity(corridor.id);
                                        onFrameEntity(corridor.id);
                                      }}
                                      className="flex items-center justify-between p-1 hover:bg-[#06b6d4]/10 cursor-pointer"
                                    >
                                      <div className="flex items-center gap-1 min-w-0">
                                        <button
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            toggle(cKey);
                                          }}
                                          className="p-0.5 text-neutral-400 hover:text-white cursor-pointer"
                                        >
                                          {isCExpanded ? (
                                            <ChevronDown className="w-2.5 h-2.5" />
                                          ) : (
                                            <ChevronRight className="w-2.5 h-2.5" />
                                          )}
                                        </button>
                                        <Workflow className="w-3 h-3 text-[#06b6d4] shrink-0" />
                                        <span className="text-white truncate">
                                          {corridor.name || corridor.id}
                                        </span>
                                      </div>

                                      <div className="flex items-center gap-1 shrink-0 text-[10px]">
                                        <button
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            onFrameEntity(corridor.id);
                                          }}
                                          title="Frame Corridor [F]"
                                          className="p-0.5 text-neutral-400 hover:text-[#00e5ff]"
                                        >
                                          <Maximize2 className="w-2.5 h-2.5" />
                                        </button>
                                      </div>
                                    </div>

                                    {isCExpanded && openings.length > 0 && (
                                      <div className="pl-4 pr-1 pb-1 pt-1 space-y-1 border-t border-[#06b6d4]/15 text-[10px]">
                                        <span className="text-neutral-500 uppercase tracking-wider text-[8px] block">
                                          Openings ({openings.length})
                                        </span>
                                        <div className="flex flex-wrap gap-1">
                                          {openings.map((op) => (
                                            <button
                                              key={op.id}
                                              type="button"
                                              onClick={() => {
                                                onSelectEntity(op.id);
                                                onFrameEntity(op.id);
                                              }}
                                              className="px-1 py-0.2 rounded border border-[#10b981]/30 bg-[#10b981]/15 text-[#10b981] font-mono text-[9px] cursor-pointer"
                                            >
                                              {op.id}
                                            </button>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* 3. STAIRS OF THIS LEVEL */}
                        {lvlStairs.length > 0 && (
                          <div className="space-y-1">
                            <span className="text-[9px] text-[#a855f7] uppercase tracking-wider block font-semibold">
                              Vertical Circulation ({lvlStairs.length})
                            </span>
                            <div className="space-y-1 pl-1">
                              {lvlStairs.map((stair) => {
                                const isSelected = selectedEntityId === stair.id;
                                return (
                                  <div
                                    key={stair.id}
                                    onClick={() => {
                                      onSelectEntity(stair.id);
                                      onFrameEntity(stair.id);
                                    }}
                                    className={`flex items-center justify-between p-1 rounded border transition-colors cursor-pointer ${
                                      isSelected
                                        ? "border-[#00e5ff] bg-[#00e5ff]/10"
                                        : "border-[#a855f7]/20 bg-[#a855f7]/5 hover:bg-[#a855f7]/10"
                                    }`}
                                  >
                                    <div className="flex items-center gap-1.5 min-w-0">
                                      <Layers className="w-3 h-3 text-[#a855f7] shrink-0" />
                                      <span className="text-white truncate">
                                        {stair.name || stair.id}
                                      </span>
                                    </div>
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        onFrameEntity(stair.id);
                                      }}
                                      title="Frame Stair [F]"
                                      className="p-0.5 text-neutral-400 hover:text-[#00e5ff]"
                                    >
                                      <Maximize2 className="w-2.5 h-2.5" />
                                    </button>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              {/* UNASSIGNED STRUCTURAL ELEMENTS (if any) */}
              {unassignedWalls.length > 0 && (
                <div className="rounded border border-neutral-800 bg-neutral-900/40 p-1.5 space-y-1">
                  <span className="text-[9px] text-neutral-500 uppercase tracking-wider block font-semibold">
                    Unenclosed Walls ({unassignedWalls.length})
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {unassignedWalls.slice(0, 15).map((w) => (
                      <button
                        key={w.id}
                        type="button"
                        onClick={() => {
                          onSelectEntity(w.id);
                          onFrameEntity(w.id);
                        }}
                        className="px-1 py-0.2 rounded border border-neutral-700 bg-neutral-800 text-neutral-300 font-mono text-[9px] hover:text-white cursor-pointer"
                      >
                        {w.id}
                      </button>
                    ))}
                    {unassignedWalls.length > 15 && (
                      <span className="text-[9px] text-neutral-500 py-0.2">
                        +{unassignedWalls.length - 15} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// TOPOLOGY INSPECTOR: ROOM ADJACENCY, CORRIDORS, STAIRS, OPENINGS
// ============================================================================

interface SpaceGraphNode {
  id: string;
  type: string;
  name?: string;
  level_id?: string;
  confidence?: number;
}

interface SpaceGraphData {
  world_id: string;
  nodes: SpaceGraphNode[];
  edges: { source: string; target: string; type: string }[];
  adjacencies: Record<string, string[]>;
  corridor_connections: Record<string, string[]>;
  stair_connections: Record<string, { connected_levels: string[]; name?: string }>;
}

export function BuildingTopologyInspector({
  worldIR,
  entitiesList,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
}: {
  worldIR: WorldIR | null;
  entitiesList: Entity[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
}) {
  const [graphData, setGraphData] = useState<SpaceGraphData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!worldIR?.id) {
      setGraphData(null);
      return;
    }
    let isCancelled = false;
    setLoading(true);

    fetch(`/api/worlds/${encodeURIComponent(worldIR.id)}/space-graph`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!isCancelled && data && data.nodes) {
          setGraphData(data as SpaceGraphData);
        }
      })
      .catch(() => {
        // Fallback to local derivation from entitiesList
      })
      .finally(() => {
        if (!isCancelled) setLoading(false);
      });

    return () => {
      isCancelled = true;
    };
  }, [worldIR?.id]);

  // Fallback / enhanced derivation from entitiesList
  const derivedData = useMemo(() => {
    const rooms = entitiesList.filter((e) => e.type === "room" || e.type === "space");
    const corridors = entitiesList.filter((e) => e.type === "corridor");
    const levels = entitiesList.filter((e) => e.type === "level" || e.type === "storey");
    const stairs = entitiesList.filter((e) => e.type === "stairs" || e.type === "stair");
    const openings = entitiesList.filter((e) => e.type === "door" || e.type === "window" || e.type === "opening");

    const adjacencies: Record<string, string[]> = { ...(graphData?.adjacencies || {}) };
    const corridorConns: Record<string, string[]> = { ...(graphData?.corridor_connections || {}) };
    const stairConns: Record<string, { connected_levels: string[]; name?: string }> = {
      ...(graphData?.stair_connections || {}),
    };

    // Fill missing corridor connections
    for (const c of corridors) {
      if (!corridorConns[c.id]) {
        const connected = new Set<string>();
        const customRooms = (c.custom_properties?.connected_room_ids as string[]) || [];
        customRooms.forEach((rid) => connected.add(rid));
        for (const rel of c.relationships || []) {
          if (rel.kind === "connects" || rel.kind === "connected_room") {
            connected.add(rel.target_id);
          }
        }
        corridorConns[c.id] = Array.from(connected);
      }
    }

    // Fill missing stair connections
    for (const s of stairs) {
      if (!stairConns[s.id]) {
        const conLevels = (s.custom_properties?.connected_level_ids as string[]) || [];
        stairConns[s.id] = {
          connected_levels: conLevels,
          name: s.name || s.id,
        };
      }
    }

    return {
      rooms,
      corridors,
      levels,
      stairs,
      openings,
      adjacencies,
      corridorConns,
      stairConns,
    };
  }, [entitiesList, graphData]);

  const selectedEntity = useMemo(() => {
    if (!selectedEntityId) return null;
    return entitiesList.find((e) => e.id === selectedEntityId) || null;
  }, [selectedEntityId, entitiesList]);

  // Selected entity specific relations
  const selectedRelations = useMemo(() => {
    if (!selectedEntity) return null;
    const directTargetIds = new Set<string>();
    for (const r of selectedEntity.relationships || []) {
      directTargetIds.add(r.target_id);
    }
    // Also entities that point to selected
    const inbound = entitiesList.filter((e) =>
      (e.relationships || []).some((r) => r.target_id === selectedEntity.id)
    );

    // If room: connected corridors, adjacent rooms
    const connectedCorridors = derivedData.corridors.filter((c) =>
      (derivedData.corridorConns[c.id] || []).includes(selectedEntity.id)
    );
    const adjacentRooms = (derivedData.adjacencies[selectedEntity.id] || []).map((id) =>
      entitiesList.find((e) => e.id === id)
    ).filter(Boolean) as Entity[];

    // If corridor: connected rooms
    const corridorRooms = (derivedData.corridorConns[selectedEntity.id] || []).map((id) =>
      entitiesList.find((e) => e.id === id)
    ).filter(Boolean) as Entity[];

    // If stair: connected levels
    const stairLevels = derivedData.stairConns[selectedEntity.id]?.connected_levels || [];

    return {
      directTargetIds: Array.from(directTargetIds),
      inbound,
      connectedCorridors,
      adjacentRooms,
      corridorRooms,
      stairLevels,
    };
  }, [selectedEntity, entitiesList, derivedData]);

  return (
    <div className="p-3 space-y-4 font-mono text-xs select-none">
      {/* Top Banner */}
      <div className="flex items-center justify-between border-b border-[#00e5ff]/20 pb-2">
        <div className="flex items-center gap-1.5">
          <Workflow className="w-4 h-4 text-[#00e5ff]" />
          <span className="font-semibold text-white uppercase tracking-wider text-[11px]">
            Spatial Topology
          </span>
        </div>
        <div className="flex items-center gap-2">
          {loading && <span className="text-[10px] text-neutral-500 animate-pulse">Syncing...</span>}
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/30">
            Graph View
          </span>
        </div>
      </div>

      {/* Network Metrics KPIs */}
      <div className="grid grid-cols-4 gap-1.5 text-center">
        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
          <div className="text-[10px] text-neutral-400">Rooms</div>
          <div className="text-sm font-semibold text-[#3b82f6]">{derivedData.rooms.length}</div>
        </div>
        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
          <div className="text-[10px] text-neutral-400">Routes</div>
          <div className="text-sm font-semibold text-[#06b6d4]">{derivedData.corridors.length}</div>
        </div>
        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
          <div className="text-[10px] text-neutral-400">Stairs</div>
          <div className="text-sm font-semibold text-[#a855f7]">{derivedData.stairs.length}</div>
        </div>
        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
          <div className="text-[10px] text-neutral-400">Portals</div>
          <div className="text-sm font-semibold text-[#10b981]">{derivedData.openings.length}</div>
        </div>
      </div>

      {/* 1. ACTIVE SELECTION RELATIONSHIP FOCUS */}
      {selectedEntity && selectedRelations ? (
        <div className="rounded-lg border border-[#00e5ff]/40 bg-[#00e5ff]/5 p-2.5 space-y-2">
          <div className="flex items-center justify-between border-b border-[#00e5ff]/20 pb-1.5">
            <span className="text-[10px] font-semibold text-[#00e5ff] uppercase tracking-wider flex items-center gap-1">
              <Crosshair className="w-3 h-3 text-[#00e5ff]" />
              Topology Context: {selectedEntity.name || selectedEntity.id}
            </span>
            <span className="text-[9px] uppercase px-1 rounded bg-[#00e5ff]/20 text-[#00e5ff]">
              {selectedEntity.type}
            </span>
          </div>

          {/* Room Specific Topology */}
          {(selectedEntity.type === "room" || selectedEntity.type === "space") && (
            <div className="space-y-2 text-[10px]">
              <div>
                <span className="text-neutral-400 block mb-1">Adjacent Rooms:</span>
                {selectedRelations.adjacentRooms.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {selectedRelations.adjacentRooms.map((adj) => (
                      <button
                        key={adj.id}
                        type="button"
                        onClick={() => {
                          onSelectEntity(adj.id);
                          onFrameEntity(adj.id);
                        }}
                        className="px-1.5 py-0.5 rounded border border-[#3b82f6]/40 bg-[#3b82f6]/15 text-[#3b82f6] hover:bg-[#3b82f6]/25 cursor-pointer truncate max-w-[120px]"
                      >
                        {adj.name || adj.id}
                      </button>
                    ))}
                  </div>
                ) : (
                  <span className="text-neutral-500 italic">No direct wall adjacencies detected</span>
                )}
              </div>

              <div>
                <span className="text-neutral-400 block mb-1">Connected Corridors:</span>
                {selectedRelations.connectedCorridors.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {selectedRelations.connectedCorridors.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => {
                          onSelectEntity(c.id);
                          onFrameEntity(c.id);
                        }}
                        className="px-1.5 py-0.5 rounded border border-[#06b6d4]/40 bg-[#06b6d4]/15 text-[#06b6d4] hover:bg-[#06b6d4]/25 cursor-pointer truncate max-w-[120px]"
                      >
                        {c.name || c.id}
                      </button>
                    ))}
                  </div>
                ) : (
                  <span className="text-neutral-500 italic">No corridor route connection</span>
                )}
              </div>
            </div>
          )}

          {/* Corridor Specific Topology */}
          {selectedEntity.type === "corridor" && (
            <div className="space-y-2 text-[10px]">
              <span className="text-neutral-400 block mb-1">Connected Room Destinations:</span>
              {selectedRelations.corridorRooms.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {selectedRelations.corridorRooms.map((rm) => (
                    <button
                      key={rm.id}
                      type="button"
                      onClick={() => {
                        onSelectEntity(rm.id);
                        onFrameEntity(rm.id);
                      }}
                      className="px-1.5 py-0.5 rounded border border-[#3b82f6]/40 bg-[#3b82f6]/15 text-[#3b82f6] hover:bg-[#3b82f6]/25 cursor-pointer truncate max-w-[120px]"
                    >
                      {rm.name || rm.id}
                    </button>
                  ))}
                </div>
              ) : (
                <span className="text-neutral-500 italic">No connected room endpoints</span>
              )}
            </div>
          )}

          {/* Stair Specific Topology */}
          {(selectedEntity.type === "stairs" || selectedEntity.type === "stair") && (
            <div className="space-y-2 text-[10px]">
              <span className="text-neutral-400 block mb-1">Spanned Vertical Levels:</span>
              {selectedRelations.stairLevels.length > 0 ? (
                <div className="flex items-center gap-1.5">
                  {selectedRelations.stairLevels.map((lvlId, idx) => (
                    <span key={lvlId} className="flex items-center gap-1">
                      {idx > 0 && <span className="text-[#a855f7]">⮂</span>}
                      <button
                        type="button"
                        onClick={() => {
                          onSelectEntity(lvlId);
                          onFrameEntity(lvlId);
                        }}
                        className="px-1.5 py-0.5 rounded border border-[#8b5cf6]/40 bg-[#8b5cf6]/15 text-[#8b5cf6] hover:bg-[#8b5cf6]/25 cursor-pointer font-semibold"
                      >
                        {lvlId}
                      </button>
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-neutral-500 italic">No level transitions registered</span>
              )}
            </div>
          )}

          {/* General Inbound & Outbound Relationships */}
          {selectedRelations.directTargetIds.length > 0 && (
            <div className="pt-1.5 border-t border-[#00e5ff]/10">
              <span className="text-neutral-500 uppercase text-[9px] block mb-1">
                Linked Entities ({selectedRelations.directTargetIds.length})
              </span>
              <div className="flex flex-wrap gap-1">
                {selectedRelations.directTargetIds.slice(0, 10).map((tid) => (
                  <button
                    key={tid}
                    type="button"
                    onClick={() => {
                      onSelectEntity(tid);
                      onFrameEntity(tid);
                    }}
                    className="px-1 py-0.2 rounded border border-neutral-700 bg-neutral-800 text-neutral-300 hover:text-white text-[9px] cursor-pointer"
                  >
                    {tid}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-2.5 rounded-lg border border-neutral-800 bg-neutral-900/40 text-neutral-400 text-[11px] flex items-center gap-2">
          <Eye className="w-4 h-4 text-neutral-500 shrink-0" />
          <span>Select any room, corridor, or stair to reveal its spatial relationships.</span>
        </div>
      )}

      {/* 2. SECTION: ROOM ADJACENCIES & REACHABILITY */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
          <span>Room Adjacencies ({derivedData.rooms.length})</span>
          <span className="text-[#3b82f6]">Interior</span>
        </div>
        <div className="space-y-1">
          {derivedData.rooms.map((room) => {
            const adjs = derivedData.adjacencies[room.id] || [];
            const connectedCorrs = derivedData.corridors.filter((c) =>
              (derivedData.corridorConns[c.id] || []).includes(room.id)
            );
            const isSelected = selectedEntityId === room.id;

            return (
              <div
                key={room.id}
                onClick={() => {
                  onSelectEntity(room.id);
                  onFrameEntity(room.id);
                }}
                className={`p-2 rounded border transition-colors cursor-pointer space-y-1 ${
                  isSelected
                    ? "bg-[#182030] border-[#00e5ff]"
                    : "bg-[#14161f] border-neutral-800 hover:border-neutral-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Building className="w-3.5 h-3.5 text-[#3b82f6] shrink-0" />
                    <span className="font-semibold text-white truncate text-[11px]">
                      {room.name || room.id}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFrameEntity(room.id);
                    }}
                    className="p-0.5 text-neutral-500 hover:text-white"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>

                {/* Adjacencies & Connected Corridors */}
                <div className="flex flex-wrap gap-1 pt-1">
                  {adjs.map((adjId) => (
                    <button
                      key={adjId}
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEntity(adjId);
                        onFrameEntity(adjId);
                      }}
                      className="px-1 py-0.2 rounded border border-[#3b82f6]/30 bg-[#3b82f6]/10 text-[#3b82f6] text-[9px] hover:bg-[#3b82f6]/20 cursor-pointer"
                    >
                      adj: {adjId}
                    </button>
                  ))}
                  {connectedCorrs.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEntity(c.id);
                        onFrameEntity(c.id);
                      }}
                      className="px-1 py-0.2 rounded border border-[#06b6d4]/30 bg-[#06b6d4]/10 text-[#06b6d4] text-[9px] hover:bg-[#06b6d4]/20 cursor-pointer"
                    >
                      route: {c.name || c.id}
                    </button>
                  ))}
                  {adjs.length === 0 && connectedCorrs.length === 0 && (
                    <span className="text-neutral-500 text-[9px] italic">Isolated room enclosure</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. SECTION: CORRIDOR CIRCULATION ROUTES */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
          <span>Corridor Circulation Routes ({derivedData.corridors.length})</span>
          <span className="text-[#06b6d4]">Routes</span>
        </div>
        <div className="space-y-1">
          {derivedData.corridors.map((c) => {
            const connectedRooms = derivedData.corridorConns[c.id] || [];
            const isSelected = selectedEntityId === c.id;

            return (
              <div
                key={c.id}
                onClick={() => {
                  onSelectEntity(c.id);
                  onFrameEntity(c.id);
                }}
                className={`p-2 rounded border transition-colors cursor-pointer space-y-1 ${
                  isSelected
                    ? "bg-[#182030] border-[#00e5ff]"
                    : "bg-[#14161f] border-neutral-800 hover:border-neutral-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Workflow className="w-3.5 h-3.5 text-[#06b6d4] shrink-0" />
                    <span className="font-semibold text-white truncate text-[11px]">
                      {c.name || c.id}
                    </span>
                  </div>
                  <span className="text-[9px] text-[#06b6d4] font-mono">
                    {connectedRooms.length} room{connectedRooms.length === 1 ? "" : "s"}
                  </span>
                </div>

                <div className="flex flex-wrap gap-1 pt-1">
                  {connectedRooms.map((rmId) => (
                    <button
                      key={rmId}
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEntity(rmId);
                        onFrameEntity(rmId);
                      }}
                      className="px-1 py-0.2 rounded border border-[#06b6d4]/30 bg-[#06b6d4]/10 text-[#06b6d4] text-[9px] hover:bg-[#06b6d4]/20 cursor-pointer"
                    >
                      room: {rmId}
                    </button>
                  ))}
                  {connectedRooms.length === 0 && (
                    <span className="text-neutral-500 text-[9px] italic">No room egress points</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. SECTION: VERTICAL TRANSITIONS (STAIRS) */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
          <span>Vertical Level Transitions ({derivedData.stairs.length})</span>
          <span className="text-[#a855f7]">Stairs</span>
        </div>
        <div className="space-y-1">
          {derivedData.stairs.map((stair) => {
            const conn = derivedData.stairConns[stair.id];
            const levels = conn?.connected_levels || [];
            const isSelected = selectedEntityId === stair.id;

            return (
              <div
                key={stair.id}
                onClick={() => {
                  onSelectEntity(stair.id);
                  onFrameEntity(stair.id);
                }}
                className={`p-2 rounded border transition-colors cursor-pointer space-y-1.5 ${
                  isSelected
                    ? "bg-[#182030] border-[#00e5ff]"
                    : "bg-[#14161f] border-neutral-800 hover:border-neutral-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Layers className="w-3.5 h-3.5 text-[#a855f7] shrink-0" />
                    <span className="font-semibold text-white truncate text-[11px]">
                      {stair.name || stair.id}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFrameEntity(stair.id);
                    }}
                    className="p-0.5 text-neutral-500 hover:text-white"
                  >
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>

                <div className="flex items-center gap-1 text-[10px]">
                  {levels.length > 0 ? (
                    levels.map((lvl, idx) => (
                      <span key={lvl} className="flex items-center gap-1">
                        {idx > 0 && <span className="text-[#a855f7]">⮂</span>}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectEntity(lvl);
                            onFrameEntity(lvl);
                          }}
                          className="px-1.5 py-0.5 rounded border border-[#8b5cf6]/30 bg-[#8b5cf6]/10 text-[#8b5cf6] text-[9px] hover:bg-[#8b5cf6]/20 cursor-pointer"
                        >
                          {lvl}
                        </button>
                      </span>
                    ))
                  ) : (
                    <span className="text-neutral-500 text-[9px] italic">No level bounds set</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 5. SECTION: OPENINGS & PORTALS */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
          <span>Portals & Openings ({derivedData.openings.length})</span>
          <span className="text-[#10b981]">Doors & Windows</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {derivedData.openings.slice(0, 20).map((op) => {
            const isSelected = selectedEntityId === op.id;
            return (
              <button
                key={op.id}
                type="button"
                onClick={() => {
                  onSelectEntity(op.id);
                  onFrameEntity(op.id);
                }}
                className={`px-1.5 py-0.5 rounded border font-mono text-[9px] truncate max-w-[120px] cursor-pointer ${
                  isSelected
                    ? "bg-[#10b981] text-black border-[#10b981]"
                    : "bg-[#10b981]/15 text-[#10b981] border-[#10b981]/30 hover:bg-[#10b981]/25"
                }`}
              >
                {op.id}
              </button>
            );
          })}
          {derivedData.openings.length > 20 && (
            <span className="text-[10px] text-neutral-500 py-0.5">
              +{derivedData.openings.length - 20} more
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
