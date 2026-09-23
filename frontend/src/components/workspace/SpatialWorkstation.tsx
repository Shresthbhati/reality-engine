"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import WorldNavPanel, { type SpatialQueryFilter } from "@/components/navigation/WorldNavPanel";
import World3DViewport from "@/components/viewport/World3DViewport";
import AdaptiveInspector from "@/components/inspector/AdaptiveInspector";
import BottomContextBar from "@/components/context/BottomContextBar";
import VersionDiffModal from "@/components/versions/VersionDiffModal";
import RoomConstructionModal from "@/components/workspace/RoomConstructionModal";
import WorldMap from "@/components/map/WorldMap";
import { 
  useWorlds, 
  useSessions, 
  useEvidence, 
  useWorldIR, 
  useWorldPoints, 
  useWorldCameras,
  useWorldVersions,
} from "@/lib/api";
import { commitWorldCorrection } from "@/lib/api/worlds";
import { parsePly, parseMeshPly } from "@/lib/viewport/loaders";
import type { WorldIR, Entity, CamerasPayload } from "@/types/worldir";
import type { WorldRow, SessionRow, EvidenceRow, PlaceRow, WorldVersionRow } from "@/lib/types";
import { 
  Loader2, 
  Globe, 
  Box, 
  Map as MapIcon, 
  Sliders, 
  Layers, 
  PanelLeftClose, 
  PanelLeftOpen, 
  PanelRightClose, 
  PanelRightOpen,
  Workflow,
  Search,
  Download,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SpatialWorkstationProps {
  worldId: string;
}

export default function SpatialWorkstation({ worldId }: SpatialWorkstationProps) {
  const router = useRouter();
  
  // Workspace UI State
  const [navOpen, setNavOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [bottomOpen, setBottomOpen] = useState(true);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"3d" | "map">("3d");
  const [diffModalOpen, setDiffModalOpen] = useState(false);
  const [constructionModalOpen, setConstructionModalOpen] = useState(false);
  const [diffVersions, setDiffVersions] = useState<{ base?: string; head?: string }>({});
  
  // Spatial Query Filter State
  const [activeQuery, setActiveQuery] = useState<SpatialQueryFilter | null>(null);

  // API Hooks
  const { data: worlds = [] } = useWorlds();
  const { data: allSessions = [] } = useSessions();
  const { data: evidence = [] } = useEvidence();
  const { data: worldIR, isLoading: isIrLoading, refetch: refetchIr } = useWorldIR(worldId);
  const { data: pointsBuffer } = useWorldPoints(worldId);
  const { data: camerasPayload } = useWorldCameras(worldId);
  const { data: versionsData = [], refetch: refetchVersions } = useWorldVersions(worldId);

  // Derived 3D artifacts
  const points = useMemo(() => {
    if (!pointsBuffer) return null;
    try {
      return parsePly(new Uint8Array(pointsBuffer));
    } catch (e) {
      console.error("Failed to parse points PLY", e);
      return null;
    }
  }, [pointsBuffer]);

  const mesh = useMemo(() => {
    if (!pointsBuffer) return null;
    try {
      return parseMeshPly(new Uint8Array(pointsBuffer));
    } catch (e) {
      return null;
    }
  }, [pointsBuffer]);

  const cameras = camerasPayload || null;
  const places: PlaceRow[] = [];
  
  // WorldStore immutable version lineage
  const versions: WorldVersionRow[] = useMemo(() => {
    if (versionsData && versionsData.length > 0) {
      return versionsData.map((v: any) => ({
        id: v.id,
        worldId: v.world_id || worldId,
        label: v.label || v.id,
        createdAt: v.created_at ? v.created_at.slice(0, 10) : "2026-09-22",
        changeSummary: v.changeSummary || v.change_summary || "Compiled representation",
        isCurrent: Boolean(v.is_current),
      }));
    }
    return [
      {
        id: `v1-${worldId}`,
        worldId,
        label: "v1.0.0 (Canonical Baseline)",
        createdAt: "2026-09-22",
        changeSummary: "Canonical compilation baseline",
        isCurrent: true,
      },
    ];
  }, [versionsData, worldId]);
  
  // Filter sessions by worldId
  const sessions = useMemo(() => 
    (allSessions || []).filter(s => s.worldId === worldId), 
    [allSessions, worldId]
  );
  
  const currentWorldRow = useMemo(() => 
    (worlds || []).find(w => w.id === worldId) || {
      id: worldId,
      name: worldId === "world-compiled-seed42" ? "Room Capture (Compiled Vertical Slice)" : worldId,
      location: "Verified Spatial Environment",
      coverageKm2: 0.05,
      sessionCount: sessions.length,
      evidenceCount: 25,
      timeRangeStart: null,
      timeRangeEnd: null,
      updatedAt: "Current",
    },
    [worlds, worldId, sessions]
  );

  const selectedEntity = useMemo(() => {
    if (!worldIR || !selectedEntityId) return null;
    return worldIR.entities[selectedEntityId] || null;
  }, [worldIR, selectedEntityId]);

  // Spatial query filtering calculation
  const queryMatchingIds = useMemo(() => {
    if (!activeQuery || !worldIR?.entities) return null;
    const matching = new Set<string>();
    for (const [id, e] of Object.entries(worldIR.entities)) {
      const matchesText = !activeQuery.search ||
        id.toLowerCase().includes(activeQuery.search.toLowerCase()) ||
        e.type.toLowerCase().includes(activeQuery.search.toLowerCase()) ||
        (e.semantic_labels || []).some(l => l.toLowerCase().includes(activeQuery.search!.toLowerCase()));
      const matchesType = !activeQuery.type || activeQuery.type === "all" || e.type === activeQuery.type;
      const matchesConf = (e.confidence ?? 0) >= (activeQuery.minConfidence ?? 0);
      if (matchesText && matchesType && matchesConf) {
        matching.add(id);
      }
    }
    return matching;
  }, [activeQuery, worldIR]);

  // Selection & Framing actions
  const handleSelectWorld = useCallback((id: string) => {
    if (id !== worldId) {
      router.push(`/worlds/${id}`);
    }
  }, [router, worldId]);

  const handleSelectEntity = useCallback((id: string | null) => {
    setSelectedEntityId(id);
    if (id !== null) {
      setInspectorOpen(true);
    }
  }, []);

  const handleSelectEvidence = useCallback((id: string | null) => {
    setSelectedEvidenceId(id);
    if (id !== null) {
      setInspectorOpen(true);
      window.dispatchEvent(new CustomEvent('frame-camera', { detail: { id } }));
    }
  }, []);

  const handleFrameEntity = useCallback((id: string) => {
    window.dispatchEvent(new CustomEvent('frame-entity', { detail: { id } }));
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedEntityId(null);
    setSelectedEvidenceId(null);
  }, []);

  const handleTraceEvidence = useCallback((evidenceId: string) => {
    setSelectedEvidenceId(evidenceId);
    window.dispatchEvent(new CustomEvent('frame-camera', { detail: { id: evidenceId } }));
  }, []);

  const handleCommitCorrection = useCallback(async (entityId: string, changes: any, commitMessage: string) => {
    await commitWorldCorrection(worldId, {
      entityId,
      changes,
      parentVersionId: versions[0]?.id || `v1-${worldId}`,
      commitMessage,
    });
    refetchVersions();
    refetchIr();
  }, [worldId, versions, refetchVersions, refetchIr]);

  const handleOpenDiff = useCallback((base?: string, head?: string) => {
    setDiffVersions({ base, head });
    setDiffModalOpen(true);
  }, []);

  // Export artifact handler
  const handleExport = useCallback(async (format: "worldir" | "ply" | "cameras" | "report") => {
    let url = "";
    let filename = "";

    if (format === "worldir") {
      url = `/api/worlds/${worldId}/worldir`;
      filename = `worldir-${worldId}.json`;
    } else if (format === "ply") {
      url = `/api/worlds/${worldId}/points`;
      filename = `points-${worldId}.ply`;
    } else if (format === "cameras") {
      url = `/api/worlds/${worldId}/cameras`;
      filename = `cameras-${worldId}.json`;
    } else if (format === "report") {
      url = `/api/worlds/${worldId}/report`;
      filename = `report-${worldId}.json`;
    }

    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Export failed (${res.status})`);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch (e) {
      console.error("Export error", e);
    }
  }, [worldId]);

  // Global Keyboard Shortcuts and Custom Event Listeners
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input or textarea
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (e.key === "[") {
        setNavOpen(prev => !prev);
      } else if (e.key === "]") {
        setInspectorOpen(prev => !prev);
      } else if (e.key === "\\") {
        setBottomOpen(prev => !prev);
      } else if (e.key === "Escape") {
        handleClearSelection();
      } else if (e.key.toLowerCase() === "f" && selectedEntityId) {
        handleFrameEntity(selectedEntityId);
      } else if (e.key.toLowerCase() === "m") {
        setViewMode(prev => (prev === "3d" ? "map" : "3d"));
      }
    };

    const handleApplyQueryEvent = (e: any) => {
      if (e.detail) {
        setActiveQuery(prev => ({ ...(prev || {}), ...e.detail }));
      }
    };

    const handleResetQueryEvent = () => {
      setActiveQuery(null);
    };

    const handleOpenDiffEvent = () => {
      setDiffModalOpen(true);
    };

    const handleOpenConstructionEvent = () => {
      setConstructionModalOpen(true);
    };

    const handleTriggerExportEvent = (e: any) => {
      if (e.detail?.format) {
        handleExport(e.detail.format);
      }
    };
    
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("apply-spatial-query", handleApplyQueryEvent as EventListener);
    window.addEventListener("reset-spatial-query", handleResetQueryEvent as EventListener);
    window.addEventListener("open-version-diff", handleOpenDiffEvent as EventListener);
    window.addEventListener("open-room-construction", handleOpenConstructionEvent as EventListener);
    window.addEventListener("trigger-export", handleTriggerExportEvent as EventListener);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("apply-spatial-query", handleApplyQueryEvent as EventListener);
      window.removeEventListener("reset-spatial-query", handleResetQueryEvent as EventListener);
      window.removeEventListener("open-version-diff", handleOpenDiffEvent as EventListener);
      window.removeEventListener("open-room-construction", handleOpenConstructionEvent as EventListener);
      window.removeEventListener("trigger-export", handleTriggerExportEvent as EventListener);
    };
  }, [selectedEntityId, handleClearSelection, handleFrameEntity, handleExport]);

  // Loading state
  if (isIrLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[#08090b] text-[var(--text-primary)] select-none">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-7 w-7 animate-spin text-[#00e5ff]" />
          <p className="text-xs font-mono text-neutral-400">Consuming canonical WorldIR stream...</p>
        </div>
      </div>
    );
  }

  const has3DContent = Boolean(worldIR && (Object.keys(worldIR.entities || {}).length > 0 || points));

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#08090b] text-[var(--text-primary)] select-none">
      {/* Sleek Persistent Studio Header */}
      <header className="h-10 shrink-0 flex items-center justify-between px-3 border-b border-[#1f222b] bg-[#0c0d12] z-30">
        {/* Left: Studio Identity & Active World Info */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2 h-2 rounded-full bg-[#00e5ff] animate-pulse" />
            <span className="text-xs font-semibold text-white tracking-wide">Reality Studio</span>
          </div>

          <span className="text-neutral-700">/</span>

          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xs font-medium text-neutral-200 truncate max-w-[200px] sm:max-w-[300px]">
              {currentWorldRow.name}
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-neutral-700 shrink-0">
              {String(worldIR?.coordinate_frame || worldIR?.coordinate_system || "metric_enu")}
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/30 shrink-0">
              {versions[0]?.label ? versions[0].label.split(" ")[0] : "v1.0.0"}
            </span>
          </div>
        </div>

        {/* Center: Viewport Mode Switcher Pill */}
        <div className="flex items-center p-0.5 rounded-md bg-[#151821] border border-[#1f222b]">
          <button
            type="button"
            onClick={() => setViewMode("3d")}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
              viewMode === "3d"
                ? "bg-[#00e5ff] text-black font-semibold shadow-sm"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Box className="w-3.5 h-3.5" />
            <span>3D Workspace</span>
          </button>
          <button
            type="button"
            onClick={() => setViewMode("map")}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
              viewMode === "map"
                ? "bg-[#00e5ff] text-black font-semibold shadow-sm"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <MapIcon className="w-3.5 h-3.5" />
            <span>Geospatial Map</span>
          </button>
        </div>

        {/* Right: Quick Studio Controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          {/* Command Palette Trigger */}
          <button
            type="button"
            onClick={() => window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, ctrlKey: true }))}
            title="Command Palette (⌘K)"
            className="flex items-center gap-1.5 px-2.5 h-7 rounded text-xs font-mono text-neutral-400 hover:text-white bg-[#151821] hover:bg-neutral-800 border border-[#1f222b] transition-colors cursor-pointer"
          >
            <Search className="w-3 h-3 text-[#00e5ff]" />
            <span className="hidden sm:inline">Commands</span>
            <kbd className="text-[10px] px-1 py-0.2 rounded bg-neutral-900 border border-neutral-700 text-neutral-400 font-mono-num">
              ⌘K
            </kbd>
          </button>

          {/* Room Construction Launcher */}
          <button
            type="button"
            onClick={() => setConstructionModalOpen(true)}
            title="Room Construction Pipeline"
            className="flex items-center gap-1 px-2.5 h-7 rounded text-xs font-medium bg-[#151821] hover:bg-neutral-800 text-neutral-300 hover:text-white border border-[#1f222b] transition-colors cursor-pointer"
          >
            <Workflow className="w-3.5 h-3.5 text-[#00e5ff]" />
            <span className="hidden sm:inline">Construction</span>
          </button>

          {/* Quick Layout Toggles */}
          <div className="flex items-center p-0.5 rounded bg-[#151821] border border-[#1f222b]">
            <button
              type="button"
              onClick={() => setNavOpen(v => !v)}
              title={navOpen ? "Collapse Navigation Panel [ [ ]" : "Expand Navigation Panel [ [ ]"}
              className={`p-1 rounded transition-colors ${
                navOpen ? "text-[#00e5ff] bg-neutral-800" : "text-neutral-400 hover:text-white"
              }`}
            >
              {navOpen ? <PanelLeftClose className="w-3.5 h-3.5" /> : <PanelLeftOpen className="w-3.5 h-3.5" />}
            </button>
            <button
              type="button"
              onClick={() => setBottomOpen(v => !v)}
              title={bottomOpen ? "Collapse Bottom Telemetry [ \\ ]" : "Expand Bottom Telemetry [ \\ ]"}
              className={`p-1 rounded transition-colors ${
                bottomOpen ? "text-[#00e5ff] bg-neutral-800" : "text-neutral-400 hover:text-white"
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setInspectorOpen(v => !v)}
              title={inspectorOpen ? "Collapse Inspector [ ] ]" : "Expand Inspector [ ] ]"}
              className={`p-1 rounded transition-colors ${
                inspectorOpen ? "text-[#00e5ff] bg-neutral-800" : "text-neutral-400 hover:text-white"
              }`}
            >
              {inspectorOpen ? <PanelRightClose className="w-3.5 h-3.5" /> : <PanelRightOpen className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </header>

      {/* Main Persistent Workspace (Panels + Viewport) */}
      <div className="flex flex-1 min-h-0 overflow-hidden relative">
        {/* Left Nav Panel */}
        <div 
          className={cn(
            "h-full flex-shrink-0 transition-all duration-200 ease-in-out border-r border-[#1f222b] z-20",
            navOpen ? "w-[280px]" : "w-0 overflow-hidden border-r-0"
          )}
        >
          {navOpen && (
            <WorldNavPanel
              worlds={worlds || []}
              activeWorldId={worldId}
              onSelectWorld={handleSelectWorld}
              worldIR={worldIR}
              sessions={sessions}
              evidence={evidence || []}
              places={places}
              versions={versions}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              onFrameEntity={handleFrameEntity}
              selectedEvidenceId={selectedEvidenceId}
              onSelectEvidence={handleSelectEvidence}
              onCompareVersions={handleOpenDiff}
              onSelectSession={(sid) => router.push(`/sessions/${sid}`)}
              activeQuery={activeQuery}
              onUpdateQuery={setActiveQuery}
              onOpenRoomConstruction={() => setConstructionModalOpen(true)}
              onExport={handleExport}
            />
          )}
        </div>
        
        {/* Center Viewport */}
        <div className="flex-1 relative h-full min-w-0 bg-[#08090b] overflow-hidden">
          {viewMode === "3d" ? (
            <World3DViewport
              world={worldIR}
              points={points}
              cameras={cameras}
              mesh={mesh}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              highlightedEvidenceId={selectedEvidenceId}
              onSelectEvidence={setSelectedEvidenceId}
              queryMatchingIds={queryMatchingIds}
              hasNoWorldData={!has3DContent}
              onTriggerSampleWorld={() => handleSelectWorld("world-compiled-seed42")}
            />
          ) : (
            <div className="w-full h-full relative">
              <WorldMap
                center={[-122.4194, 37.7749]}
                zoom={14}
                pitch={40}
              />
              <div className="absolute bottom-4 left-4 z-10 px-3 py-2 rounded-md bg-[#0e1013]/90 border border-[#1f222b] text-xs text-neutral-300 backdrop-blur-md font-mono">
                <div className="text-white font-semibold">Geospatial Coordinate Anchor</div>
                <div className="text-neutral-400 text-[11px] mt-0.5">Lat: 37.7749° N · Lon: -122.4194° W · ENU Origin</div>
              </div>
            </div>
          )}
        </div>
        
        {/* Right Adaptive Inspector */}
        <div 
          className={cn(
            "h-full flex-shrink-0 transition-all duration-200 ease-in-out border-l border-[#1f222b] z-20",
            inspectorOpen ? "w-[340px]" : "w-0 overflow-hidden border-l-0"
          )}
        >
          {inspectorOpen && (
            <AdaptiveInspector
              world={worldIR}
              worldId={worldId}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              onFrameEntity={handleFrameEntity}
              onTraceEvidence={handleTraceEvidence}
              onCommitCorrection={handleCommitCorrection}
              onOpenRoomConstruction={() => setConstructionModalOpen(true)}
              onExport={handleExport}
            />
          )}
        </div>
      </div>
      
      {/* Bottom Context Bar */}
      {bottomOpen && (
        <div className="h-auto flex-shrink-0 border-t border-[#1f222b] z-30">
          <BottomContextBar
            world={worldIR}
            selectedEntity={selectedEntity}
            sessions={sessions}
            onFrameSelected={() => {
              if (selectedEntityId) handleFrameEntity(selectedEntityId);
            }}
            onClearSelection={handleClearSelection}
          />
        </div>
      )}

      {/* Version Comparison / Diff Modal */}
      <VersionDiffModal
        worldId={worldId}
        isOpen={diffModalOpen}
        onClose={() => setDiffModalOpen(false)}
        baseVersion={diffVersions.base || versions[0]?.id || `v1-${worldId}`}
        headVersion={diffVersions.head || "latest"}
        onSelectEntity={(eid) => {
          handleSelectEntity(eid);
          handleFrameEntity(eid);
        }}
      />

      {/* Room Construction Pipeline Modal */}
      <RoomConstructionModal
        worldId={worldId}
        isOpen={constructionModalOpen}
        onClose={() => setConstructionModalOpen(false)}
        onReconstructionSuccess={() => {
          refetchIr();
          refetchVersions();
        }}
      />
    </div>
  );
}
