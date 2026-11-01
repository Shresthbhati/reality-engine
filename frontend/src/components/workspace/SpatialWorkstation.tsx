"use client";

import React, { useState, useEffect, useMemo, useCallback, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import WorldNavPanel, { type SpatialQueryFilter } from "@/components/navigation/WorldNavPanel";
import World3DViewportBase from "@/components/viewport/World3DViewport";
import AdaptiveInspector from "@/components/inspector/AdaptiveInspector";
import BottomContextBar from "@/components/context/BottomContextBar";
import VersionDiffModal from "@/components/versions/VersionDiffModal";
import RoomConstructionModal from "@/components/workspace/RoomConstructionModal";
import BuildModal from "@/components/workspace/BuildModal";
import GeneratedPanel from "@/components/workspace/GeneratedPanel";
import CompleteInteriorPanel from "@/components/workspace/CompleteInteriorPanel";
import SceneTree from "@/components/workspace/SceneTree";
import InsightsPanel from "@/components/workspace/InsightsPanel";
import ActionsMenu from "@/components/workspace/ActionsMenu";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";
import ReconstructionStatusBar from "@/components/reconstruct/ReconstructionStatusBar";
import NoGeometryExplanation from "@/components/reconstruct/NoGeometryExplanation";
import QueryPanel from "@/components/workspace/QueryPanel";
import ExportPanel from "@/components/workspace/ExportPanel";
import MapPlotView from "@/components/map/MapPlotView";
import {
  useWorlds,
  useSessions,
  useEvidence,
  useWorldIR,
  useWorldPoints,
  useWorldCameras,
  useWorldVersions,
  useWorldStatus,
  cancelReconstruction,
  retryReconstruction,
  type CreateReconstructionResult,
} from "@/lib/api";
import { commitWorldCorrection } from "@/lib/api/worlds";
import { downloadArtifact } from "@/lib/api/download";
import { isApiError } from "@/lib/api/client";
import { parsePly, parseMeshPly } from "@/lib/viewport/loaders";
import { parseCaptureQuality, parseStructurePoints } from "@/lib/viewport/structure-points-model";
import type { MeasurementResult } from "@/lib/viewport/three-scene";
import type { PlaceRow, WorldVersionRow } from "@/lib/types";
import { buildFlowSteps } from "@/lib/reconstruct/flow-model";
import { buildSceneTree, toggleHidden } from "@/lib/workspace/scene-tree-model";
import { buildInsights, withKnownPhotos } from "@/lib/workspace/insights-model";
import { Loader2, Box, Map as MapIcon, ImagePlus, X, PanelLeft, PanelRight } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The viewport agent is adding optional props to World3DViewport. Until they land in its own
 * interface this local shim types them. TODO: drop the shim once `hiddenEntityIds` is declared on
 * World3DViewportProps (also coming: `mode`, `onModeChange`, `cutaway`, which the workstation does not drive).
 */
const World3DViewport = World3DViewportBase as React.ComponentType<
  React.ComponentProps<typeof World3DViewportBase> & { hiddenEntityIds?: Set<string> }
>;

type Layout = "mobile" | "tablet" | "desktop";
type MobilePane = "scene" | "view" | "details";

function subscribeLayout(cb: () => void) {
  const qs = [window.matchMedia("(min-width: 768px)"), window.matchMedia("(min-width: 1024px)")];
  qs.forEach((q) => q.addEventListener("change", cb));
  return () => qs.forEach((q) => q.removeEventListener("change", cb));
}
function layoutSnapshot(): Layout {
  if (window.matchMedia("(min-width: 1024px)").matches) return "desktop";
  return window.matchMedia("(min-width: 768px)").matches ? "tablet" : "mobile";
}
/** <768 mobile (segmented switcher), 768-1023 tablet (slide-over drawers), >=1024 desktop (3 columns). */
function useLayout(): Layout {
  return useSyncExternalStore(subscribeLayout, layoutSnapshot, () => "desktop");
}

interface SpatialWorkstationProps {
  worldId?: string;
}

export default function SpatialWorkstation({ worldId: propWorldId }: SpatialWorkstationProps) {
  const router = useRouter();
  
  // Workspace UI State
  const layout = useLayout();
  const [navOpen, setNavOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  // tablet: which panel is slid over the viewport; mobile: which pane is showing
  const [drawer, setDrawer] = useState<"scene" | "inspector" | null>(null);
  const [mobilePane, setMobilePane] = useState<MobilePane>("view");
  const [browseOpen, setBrowseOpen] = useState(false);
  // null = follow the default (insights strip open only when there is something in it)
  // Timeline / telemetry bar: off by default so the viewport keeps the space.
  const [bottomOpen, setBottomOpen] = useState(false);
  const [insightsPref, setInsightsPref] = useState<boolean | null>(null);
  // Entities hidden from the viewport through the Scene tree eye toggles.
  const [hiddenIds, setHiddenIds] = useState<Set<string>>(() => new Set());
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"3d" | "map">("3d");
  const [diffModalOpen, setDiffModalOpen] = useState(false);
  const [constructionModalOpen, setConstructionModalOpen] = useState(false);
  const [buildModalOpen, setBuildModalOpen] = useState(false);
  const [queryPanelOpen, setQueryPanelOpen] = useState(false);
  const [exportPanelOpen, setExportPanelOpen] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [diffVersions, setDiffVersions] = useState<{ base?: string; head?: string }>({});
  const [measurement, setMeasurement] = useState<MeasurementResult | null>(null);
  const [isolateSpaceId, setIsolateSpaceId] = useState<string | null>(null);
  const [activeLevelIndex, setActiveLevelIndex] = useState<number | null>(null);
  
  // Spatial Query Filter State
  const [activeQuery, setActiveQuery] = useState<SpatialQueryFilter | null>(null);

  // API Hooks
  const { data: rawWorlds, isLoading: worldsLoading } = useWorlds();
  const worlds = useMemo(() => rawWorlds ?? [], [rawWorlds]);
  const { data: rawSessions } = useSessions();
  const allSessions = useMemo(() => rawSessions ?? [], [rawSessions]);
  const { data: rawEvidence } = useEvidence();
  const evidence = useMemo(() => rawEvidence ?? [], [rawEvidence]);

  // Resolve active world: prop -> first discovered world -> empty string
  const worldId = propWorldId || (worlds.length > 0 ? worlds[0].id : "");

  // Version being inspected (null = the world's current version).
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [addPhotosOpen, setAddPhotosOpen] = useState(false);

  // Live reconstruction status: polls while the engine works. Geometry is
  // refetched whenever a NEW current version lands (modelVersionKey below).
  const { status: reconStatus, error: statusError, refresh: refreshStatus } = useWorldStatus(worldId || null);
  const modelVersionKey = reconStatus?.model?.version_id ?? null;

  const { data: worldIR, isLoading: isIrLoading, error: irError, refetch: refetchIr } = useWorldIR(worldId || null, selectedVersionId, modelVersionKey);
  const { data: pointsBuffer } = useWorldPoints(worldId || null, selectedVersionId, modelVersionKey);
  const { data: camerasPayload } = useWorldCameras(worldId || null, selectedVersionId, modelVersionKey);
  const { data: versionsData = [], refetch: refetchVersions } = useWorldVersions(worldId || null, modelVersionKey);

  // Capture-quality verdict and structure points belong to the CURRENT version's report: not passed when an older version is inspected.
  const currentModel = reconStatus?.model ?? null;
  const viewingCurrent = selectedVersionId === null || selectedVersionId === currentModel?.version_id;
  const captureQuality = useMemo(() => (viewingCurrent ? parseCaptureQuality(currentModel?.capture_quality) : null), [viewingCurrent, currentModel?.capture_quality]);
  const structurePoints = useMemo(() => (viewingCurrent ? parseStructurePoints(currentModel?.structure_points) : []), [viewingCurrent, currentModel?.structure_points]);

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
    } catch {
      return null;
    }
  }, [pointsBuffer]);

  const cameras = camerasPayload || null;
  const places: PlaceRow[] = [];
  
  // WorldStore immutable version lineage (honest: empty until computational commits exist)
  const versions: WorldVersionRow[] = useMemo(() => {
    return versionsData || [];
  }, [versionsData]);
  
  // Filter sessions by worldId
  const sessions = useMemo(() => 
    (allSessions || []).filter(s => s.worldId === worldId), 
    [allSessions, worldId]
  );

  // Session the room-construction job is enqueued for. Defaults to the first
  // attached session; the modal can switch it. Falls back if it goes away.
  const [constructionSessionId, setConstructionSessionId] = useState<string | null>(null);
  const effectiveConstructionSessionId = useMemo(() => {
    if (constructionSessionId && sessions.some(s => s.id === constructionSessionId)) {
      return constructionSessionId;
    }
    return sessions.length > 0 ? sessions[0].id : null;
  }, [constructionSessionId, sessions]);
  const constructionSessions = useMemo(
    () => sessions.map(s => ({ id: s.id, name: s.name || s.id })),
    [sessions]
  );

  
  // The selected world's real row, or null when no world is selected/loaded.
  // This used to fall back to an invented row (location "Physical Coordinate
  // Space", coverage 0.05 km², updatedAt "Current") so the header always had
  // something to print; the header now shows the absence instead.
  const currentWorldRow = useMemo(
    () => (worlds || []).find(w => w.id === worldId) ?? null,
    [worlds, worldId]
  );

  const currentVersionId = versions[0]?.id ?? null;

  const selectedEntity = useMemo(() => {
    if (!worldIR || !selectedEntityId) return null;
    return worldIR.entities[selectedEntityId] || null;
  }, [worldIR, selectedEntityId]);

  // Scene tree + insights come from the WorldIR of the active world only (never a stale one).
  const sceneWorld = worldIR && worldIR.id === worldId ? worldIR : null;
  const sceneTree = useMemo(() => buildSceneTree(sceneWorld), [sceneWorld]);
  const insightsModel = useMemo(
    () => withKnownPhotos(buildInsights(sceneWorld), new Set(evidence.map((ev) => ev.id))),
    [sceneWorld, evidence],
  );

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

  const handleToggleHidden = useCallback((leafIds: string[]) => {
    setHiddenIds((prev) => toggleHidden(prev, leafIds));
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

  const handleCommitCorrection = useCallback(async (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => {
    if (!worldId) return;
    const parentVersionId = versions[0]?.id || "";
    const res = await commitWorldCorrection(worldId, {
      entity_id: entityId,
      changes,
      parent_version_id: parentVersionId,
      commit_message: commitMessage,
    });
    refetchVersions();
    refetchIr();
    return { ...res, parent_version_id: parentVersionId };
  }, [worldId, versions, refetchVersions, refetchIr]);

  const handleOpenDiff = useCallback((base?: string, head?: string) => {
    setDiffVersions({ base, head });
    setDiffModalOpen(true);
  }, []);

  // Photos were added: same world -> refresh in place; new world -> open it.
  const handlePhotosCreated = useCallback((res: CreateReconstructionResult) => {
    setAddPhotosOpen(false);
    setSelectedVersionId(null);
    if (res.world_id !== worldId) {
      router.push(`/worlds/${res.world_id}`);
    } else {
      refreshStatus();
    }
  }, [router, worldId, refreshStatus]);

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

    setExportError(null);
    try {
      // same-origin Next proxy route: pass it absolute so downloadArtifact doesn't re-point it at the API host
      await downloadArtifact(`${window.location.origin}${url}`, filename);
    } catch (e) {
      setExportError(isApiError(e) ? e.describe() : `Export of ${format} failed.`);
    }
  }, [worldId]);

  // Reset transient workspace selection & filters whenever active world changes
  const [resetForWorldId, setResetForWorldId] = useState(worldId);
  if (resetForWorldId !== worldId) {
    setResetForWorldId(worldId);
    setSelectedEntityId(null);
    setSelectedEvidenceId(null);
    setActiveQuery(null);
    setSelectedVersionId(null);
    setAddPhotosOpen(false);
    setHiddenIds(new Set());
    setBrowseOpen(false);
  }

  // Global Keyboard Shortcuts and Custom Event Listeners
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input or textarea
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (e.key === "[") {
        if (layout === "desktop") setNavOpen(prev => !prev);
        else if (layout === "tablet") setDrawer(d => (d === "scene" ? null : "scene"));
        else setMobilePane(p => (p === "scene" ? "view" : "scene"));
      } else if (e.key === "]") {
        if (layout === "desktop") setInspectorOpen(prev => !prev);
        else if (layout === "tablet") setDrawer(d => (d === "inspector" ? null : "inspector"));
        else setMobilePane(p => (p === "details" ? "view" : "details"));
      } else if (e.key === "\\") {
        setInsightsPref(prev => !(prev ?? insightsModel.hasInsights));
      } else if (e.key === "Escape") {
        handleClearSelection();
      } else if (e.key.toLowerCase() === "f" && selectedEntityId) {
        handleFrameEntity(selectedEntityId);
      }
    };

    const handleApplyQueryEvent = (e: Event) => {
      const customEvent = e as CustomEvent<SpatialQueryFilter>;
      if (customEvent.detail) {
        setActiveQuery(prev => ({ ...(prev || {}), ...customEvent.detail }));
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

    const handleOpenQueryEvent = () => {
      setQueryPanelOpen(true);
    };

    const handleOpenExportPanelEvent = () => {
      setExportPanelOpen(true);
    };

    const handleTriggerExportEvent = (e: Event) => {
      const customEvent = e as CustomEvent<{ format: "worldir" | "ply" | "cameras" | "report" }>;
      if (customEvent.detail?.format) {
        handleExport(customEvent.detail.format);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("apply-spatial-query", handleApplyQueryEvent as EventListener);
    window.addEventListener("reset-spatial-query", handleResetQueryEvent as EventListener);
    window.addEventListener("open-version-diff", handleOpenDiffEvent as EventListener);
    window.addEventListener("open-room-construction", handleOpenConstructionEvent as EventListener);
    window.addEventListener("open-spatial-query", handleOpenQueryEvent as EventListener);
    window.addEventListener("open-export-panel", handleOpenExportPanelEvent as EventListener);
    window.addEventListener("trigger-export", handleTriggerExportEvent as EventListener);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("apply-spatial-query", handleApplyQueryEvent as EventListener);
      window.removeEventListener("reset-spatial-query", handleResetQueryEvent as EventListener);
      window.removeEventListener("open-version-diff", handleOpenDiffEvent as EventListener);
      window.removeEventListener("open-room-construction", handleOpenConstructionEvent as EventListener);
      window.removeEventListener("open-spatial-query", handleOpenQueryEvent as EventListener);
      window.removeEventListener("open-export-panel", handleOpenExportPanelEvent as EventListener);
      window.removeEventListener("trigger-export", handleTriggerExportEvent as EventListener);
    };
  }, [selectedEntityId, layout, insightsModel.hasInsights, handleClearSelection, handleFrameEntity, handleExport]);

  // First-ever world: the only thing to do is to drop photos.
  if (!propWorldId && !worldsLoading && worlds.length === 0) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center bg-[var(--bg-base)] p-6 text-[var(--text-primary)]">
        <h1 className="mb-1 text-xl font-semibold text-[var(--text-primary)]">Create a reconstruction</h1>
        <p className="mb-6 max-w-lg text-center text-xs text-[var(--text-secondary)]">
          Give Reality Engine photographs of a real place. It builds the best spatial model the
          evidence supports, tells you what it is unsure about, and improves the same model as you add more.
        </p>
        <div className="w-full max-w-xl">
          <PhotoDropZone onCreated={handlePhotosCreated} />
        </div>
      </div>
    );
  }

  // Loading state (first load only; later refetches keep the Studio mounted)
  if (isIrLoading && !worldIR) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[var(--bg-base)] text-[var(--text-primary)] select-none">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-7 w-7 animate-spin text-[var(--accent)]" />
          <p className="text-xs font-mono text-[var(--text-secondary)]">Consuming canonical WorldIR stream...</p>
        </div>
      </div>
    );
  }

  const isWorldMatching = Boolean(worldIR && worldIR.id === worldId);
  const activeWorldIR = isWorldMatching ? worldIR : null;
  const activePoints = isWorldMatching ? points : null;
  const activeCameras = isWorldMatching ? cameras : null;
  const activeMesh = isWorldMatching ? mesh : null;
  const has3DContent = Boolean(activeWorldIR && (Object.keys(activeWorldIR.entities || {}).length > 0 || activePoints));

  // ONE primary action: the guided flow's current step (add photos / retry / set scale), else Add evidence.
  const flowCta = buildFlowSteps(reconStatus).find((st) => st.current)?.cta;
  const primary: { label: string; run: () => void } =
    flowCta?.kind === "retry" && reconStatus?.job
      ? { label: flowCta.label, run: () => void retryReconstruction(reconStatus.job!.session_id).then(refreshStatus, refreshStatus) }
      : flowCta?.kind === "set_scale"
        ? { label: flowCta.label, run: () => window.dispatchEvent(new CustomEvent("reconstruction-expand")) }
        : { label: "Add evidence", run: () => setAddPhotosOpen(true) };

  const toggleScene = () => {
    if (layout === "desktop") setNavOpen((v) => !v);
    else setDrawer((d) => (d === "scene" ? null : "scene"));
  };
  const toggleInspector = () => {
    if (layout === "desktop") setInspectorOpen((v) => !v);
    else setDrawer((d) => (d === "inspector" ? null : "inspector"));
  };
  // phones: collapsed unless asked for, so the viewport keeps the screen
  const insightsOpen = insightsPref ?? (layout !== "mobile" && insightsModel.hasInsights);
  const sceneOn = layout === "desktop" ? navOpen : drawer === "scene";
  const inspectorOn = layout === "desktop" ? inspectorOpen : drawer === "inspector";

  const sceneEl = (
    <SceneTree
      tree={sceneTree}
      selectedEntityId={selectedEntityId}
      hiddenIds={hiddenIds}
      onSelect={(id) => {
        handleSelectEntity(id);
        if (layout === "tablet") setDrawer(null);
        if (layout === "mobile") setMobilePane("view");
      }}
      onToggleVisibility={handleToggleHidden}
      onBrowse={() => setBrowseOpen(true)}
      onClose={layout === "tablet" ? () => setDrawer(null) : undefined}
      loading={isIrLoading}
    />
  );

  const inspectorEl = (
    <AdaptiveInspector
      world={worldIR}
      worldId={worldId}
      activeVersion={currentVersionId ?? undefined}
      selectedEntityId={selectedEntityId}
      onSelectEntity={handleSelectEntity}
      onFrameEntity={handleFrameEntity}
      onTraceEvidence={handleTraceEvidence}
      onCommitCorrection={handleCommitCorrection}
      onOpenDiff={handleOpenDiff}
      onOpenRoomConstruction={() => setConstructionModalOpen(true)}
      onExport={handleExport}
      loadError={!worldIR && irError ? irError.message : null}
      onRetryLoad={refetchIr}
    />
  );

  const showCentre = layout !== "mobile" || mobilePane === "view";

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[var(--bg-base)] text-[var(--text-primary)] select-none">
      <header className="z-30 flex h-10 shrink-0 items-center justify-between gap-2 border-b border-[var(--border)] bg-[var(--bg-surface)] px-3">
        {/* Left: studio identity, world, version */}
        <div className="flex min-w-0 items-center gap-2">
          <span className="hidden shrink-0 text-xs font-semibold tracking-wide text-[var(--text-primary)] sm:inline">Reality Studio</span>
          <span className="hidden text-[var(--text-tertiary)] sm:inline">/</span>
          <span className="max-w-[160px] truncate text-xs font-medium text-[var(--text-primary)] sm:max-w-[260px]">
            {currentWorldRow?.name ?? "No World Selected"}
          </span>
          <span className="hidden shrink-0 rounded border border-[var(--accent-border)] bg-[var(--accent-subtle)] px-1.5 font-mono text-[10px] text-[var(--accent)] md:inline">
            {reconStatus?.versions.find((v) => v.id === (selectedVersionId ?? reconStatus.model?.version_id))?.label
              ?? currentVersionId ?? "no version"}
          </span>
        </div>

        {/* Centre: 3D / Map */}
        <div className="flex items-center rounded border border-[var(--border)] bg-[var(--bg-base)] p-0.5">
          {([["3d", "3D Studio", Box], ["map", "Map", MapIcon]] as const).map(([mode, label, Icon]) => (
            <button
              key={mode}
              type="button"
              onClick={() => setViewMode(mode)}
              aria-pressed={viewMode === mode}
              className={cn(
                "flex h-6 items-center gap-1.5 rounded px-2.5 text-xs font-medium cursor-pointer",
                viewMode === mode ? "bg-[var(--accent-subtle)] text-[var(--accent)]" : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>

        {/* Right: panel toggles, the one primary action, the one Actions menu */}
        <div className="flex shrink-0 items-center gap-1.5">
          {layout !== "mobile" && (
            <div className="flex items-center rounded border border-[var(--border)] bg-[var(--bg-base)] p-0.5">
              <button
                type="button"
                onClick={toggleScene}
                aria-label="Toggle scene panel"
                aria-pressed={sceneOn}
                title="Scene panel [ [ ]"
                className={cn("rounded p-1 cursor-pointer", sceneOn ? "text-[var(--accent)]" : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]")}
              >
                <PanelLeft className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={toggleInspector}
                aria-label="Toggle inspector"
                aria-pressed={inspectorOn}
                title="Inspector [ ] ]"
                className={cn("rounded p-1 cursor-pointer", inspectorOn ? "text-[var(--accent)]" : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]")}
              >
                <PanelRight className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
          <button
            type="button"
            onClick={primary.run}
            data-testid="add-photos"
            title={primary.label === "Add evidence" ? "Add photographs to improve this model" : primary.label}
            className="flex h-7 items-center gap-1 rounded bg-[var(--primary)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--primary-hover)] cursor-pointer"
          >
            <ImagePlus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{primary.label}</span>
          </button>
          <ActionsMenu
            onAddEvidence={() => setAddPhotosOpen(true)}
            onBuild={() => setBuildModalOpen(true)}
            onGenerated={() => window.dispatchEvent(new Event("open-generated-panel"))}
            onCompleteInterior={() => window.dispatchEvent(new Event("open-complete-interior"))}
            onConstruction={() => setConstructionModalOpen(true)}
            onQuery={() => setQueryPanelOpen(true)}
            onExport={() => setExportPanelOpen(true)}
            onBrowse={() => setBrowseOpen(true)}
            onCommands={() => window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, ctrlKey: true }))}
            onToggleScene={toggleScene}
            onToggleInspector={toggleInspector}
            onToggleInsights={() => setInsightsPref(!insightsOpen)}
            onToggleTimeline={() => setBottomOpen((v) => !v)}
          />
          {/* Dialog hosts: opened from the Actions menu */}
          <GeneratedPanel hideTrigger worldId={worldId} onChanged={() => { Promise.all([refetchIr(), refetchVersions()]); }} />
          <CompleteInteriorPanel hideTrigger worldId={worldId} onChanged={() => { Promise.all([refetchIr(), refetchVersions()]); }} />
        </div>
      </header>

      {/* Studio: Scene tree | viewport | inspector */}
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {layout === "desktop" && navOpen && (
          <aside aria-label="Scene" className="z-20 h-full w-[248px] shrink-0 border-r border-[var(--border)]">{sceneEl}</aside>
        )}
        {layout === "mobile" && mobilePane === "scene" && <div className="h-full w-full">{sceneEl}</div>}

        <main className={cn("relative h-full min-w-0 flex-1 overflow-hidden bg-[var(--bg-base)]", !showCentre && "hidden")}>
          {viewMode === "3d" ? (
            <World3DViewport
              world={activeWorldIR}
              points={activePoints}
              cameras={activeCameras}
              mesh={activeMesh}
              structurePoints={structurePoints}
              captureQuality={captureQuality}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              highlightedEvidenceId={selectedEvidenceId}
              onSelectEvidence={setSelectedEvidenceId}
              queryMatchingIds={queryMatchingIds}
              hiddenEntityIds={hiddenIds}
              isLoading={isIrLoading}
              hasNoWorldData={!has3DContent}
              onMeasurementChange={setMeasurement}
              worldId={worldId || null}
              onOpenRoomConstruction={() => setConstructionModalOpen(true)}
            />
          ) : (
            <MapPlotView
              worldId={worldId}
              worldName={currentWorldRow?.name}
              world={activeWorldIR}
              cameras={activeCameras}
              worldLat={currentWorldRow?.lat ?? null}
              worldLng={currentWorldRow?.lng ?? null}
              sessions={sessions}
              evidence={evidence}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              selectedEvidenceId={selectedEvidenceId}
              onSelectEvidence={setSelectedEvidenceId}
              onOpen3D={() => setViewMode("3d")}
            />
          )}

          {!has3DContent && viewMode === "3d" && (
            <NoGeometryExplanation status={reconStatus} worldId={worldId || null} onCreated={handlePhotosCreated} />
          )}

          {queryPanelOpen && (
            <QueryPanel
              worldId={worldId}
              selectedEntityId={selectedEntityId}
              onClose={() => setQueryPanelOpen(false)}
              onSelectEntity={handleSelectEntity}
              onFrameEntity={handleFrameEntity}
            />
          )}

          {exportPanelOpen && <ExportPanel worldId={worldId} onClose={() => setExportPanelOpen(false)} />}

          {exportError && (
            <div
              role="alert"
              className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 rounded-md px-3 py-2 text-xs shadow-xl"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--error)", color: "var(--error)" }}
            >
              <span>{exportError}</span>
              <button type="button" onClick={() => setExportError(null)} aria-label="Dismiss export error">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </main>

        {layout === "desktop" && inspectorOpen && (
          <aside aria-label="Inspector" className="z-20 h-full w-[320px] shrink-0 border-l border-[var(--border)]">{inspectorEl}</aside>
        )}
        {layout === "mobile" && mobilePane === "details" && <div className="h-full w-full">{inspectorEl}</div>}

        {/* Tablet: slide-over drawers */}
        {layout === "tablet" && drawer && (
          <>
            <button type="button" aria-label="Close panel" onClick={() => setDrawer(null)} className="absolute inset-0 z-30 cursor-default bg-[var(--bg-base)]/70" />
            <aside
              aria-label={drawer === "scene" ? "Scene" : "Inspector"}
              className={cn(
                "absolute inset-y-0 z-40 w-[320px] max-w-[85%] border-[var(--border)] bg-[var(--bg-base)]",
                drawer === "scene" ? "left-0 border-r" : "right-0 border-l",
              )}
            >
              {drawer === "scene" ? sceneEl : inspectorEl}
            </aside>
          </>
        )}

        {/* Worlds, sessions, evidence, versions, places, queries (the previous navigation panel) */}
        {browseOpen && (
          <>
            <button type="button" aria-label="Close browser" onClick={() => setBrowseOpen(false)} className="absolute inset-0 z-40 cursor-default bg-[var(--bg-base)]/70" />
            <aside aria-label="Worlds and data" className="absolute inset-y-0 left-0 z-50 flex w-[340px] max-w-full flex-col border-r border-[var(--border)] bg-[var(--bg-base)]">
              <div className="flex h-9 shrink-0 items-center justify-between border-b border-[var(--border-subtle)] px-3">
                <h2 className="text-xs font-semibold text-[var(--text-primary)]">Worlds, sessions, evidence</h2>
                <button type="button" onClick={() => setBrowseOpen(false)} aria-label="Close" className="rounded p-1 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="min-h-0 flex-1">
                <WorldNavPanel
                  worlds={worlds || []}
                  activeWorldId={worldId}
                  onSelectWorld={(id) => {
                    setBrowseOpen(false);
                    handleSelectWorld(id);
                  }}
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
                  onSelectPlace={(place) => {
                    window.dispatchEvent(
                      new CustomEvent("frame-position", {
                        detail: { x: place.position[0], y: place.position[1], z: place.position[2] },
                      })
                    );
                  }}
                  activeQuery={activeQuery}
                  onUpdateQuery={setActiveQuery}
                  onOpenRoomConstruction={() => setConstructionModalOpen(true)}
                  onExport={handleExport}
                />
              </div>
            </aside>
          </>
        )}
      </div>

      {/* Reconstruction insights: real findings + supporting photographs */}
      {showCentre && (
        <InsightsPanel
          model={insightsModel}
          worldId={worldId}
          open={insightsOpen}
          onOpenChange={setInsightsPref}
          onSelectEntity={handleSelectEntity}
          onSelectEvidence={handleSelectEvidence}
          onOpenEvidence={() => router.push("/evidence")}
        />
      )}

      {/* Reconstruction state, versions, scale */}
      <div className="z-30 max-h-[14vh] shrink-0 overflow-y-auto md:max-h-[24vh] border-t border-[var(--border)]">
        <ReconstructionStatusBar
          status={reconStatus}
          error={statusError}
          world={activeWorldIR}
          selectedVersionId={selectedVersionId}
          onSelectVersion={setSelectedVersionId}
          onAddPhotos={() => setAddPhotosOpen(true)}
          onCompareVersions={handleOpenDiff}
          onCancel={(jobId) => {
            void cancelReconstruction(jobId).then(refreshStatus, refreshStatus);
          }}
          onRetry={(sessionId) => {
            void retryReconstruction(sessionId).then(refreshStatus, refreshStatus);
          }}
          onScaleChanged={refreshStatus}
        />
      </div>

      {/* Mobile: Scene | View | Details */}
      {layout === "mobile" && (
        <div role="tablist" aria-label="Workspace panes" className="z-30 grid h-11 shrink-0 grid-cols-3 border-t border-[var(--border)] bg-[var(--bg-surface)]">
          {([["scene", "Scene"], ["view", "View"], ["details", "Details"]] as const).map(([pane, label]) => (
            <button
              key={pane}
              type="button"
              role="tab"
              aria-selected={mobilePane === pane}
              onClick={() => setMobilePane(pane)}
              data-testid={`mobile-pane-${pane}`}
              className={cn(
                "text-xs font-medium cursor-pointer",
                mobilePane === pane ? "border-t-2 border-[var(--accent)] text-[var(--accent)]" : "border-t-2 border-transparent text-[var(--text-secondary)]",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {addPhotosOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--bg-base)]/80 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Add photographs"
        >
          <div className="relative w-full max-w-lg rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-4">
            <button
              type="button"
              onClick={() => setAddPhotosOpen(false)}
              aria-label="Close"
              className="absolute right-2 top-2 rounded p-1 text-[var(--text-tertiary)] hover:text-[var(--text-primary)]"
            >
              <X className="h-4 w-4" />
            </button>
            <h2 className="mb-3 text-sm font-semibold text-[var(--text-primary)]">Add photographs to this model</h2>
            <PhotoDropZone worldId={worldId || null} onCreated={handlePhotosCreated} />
          </div>
        </div>
      )}

      {/* Bottom context bar (selection, measurement, version) */}
      {layout !== "mobile" && bottomOpen && (
        <div className="z-30 h-auto max-h-[30vh] shrink-0 overflow-y-auto border-t border-[var(--border)]">
          <BottomContextBar
            world={worldIR}
            selectedEntity={selectedEntity}
            sessions={sessions}
            onFrameSelected={() => {
              if (selectedEntityId) handleFrameEntity(selectedEntityId);
            }}
            onClearSelection={handleClearSelection}
            measurement={measurement}
            activeVersion={currentVersionId ?? undefined}
            worldId={worldId || undefined}
          />
        </div>
      )}

      {/* Version Comparison / Diff Modal */}
      <VersionDiffModal
        worldId={worldId}
        isOpen={diffModalOpen}
        onClose={() => setDiffModalOpen(false)}
        baseVersion={diffVersions.base || versions[0]?.parentVersionId || ""}
        headVersion={diffVersions.head || versions[0]?.id || ""}
        onSelectEntity={(eid) => {
          handleSelectEntity(eid);
          handleFrameEntity(eid);
        }}
      />

      <BuildModal
        worldId={worldId}
        rooms={Object.values(worldIR?.entities ?? {})
          .filter((e) => e.type === "room")
          .map((e) => ({ id: e.id, name: e.name || e.id }))}
        isOpen={buildModalOpen}
        onClose={() => setBuildModalOpen(false)}
        onCreated={() => {
          // Frame only once the new version's WorldIR has actually loaded.
          Promise.all([refetchIr(), refetchVersions()]).then(() =>
            window.dispatchEvent(new CustomEvent("frame-all")),
          );
        }}
      />

      {/* Room Construction Pipeline Modal */}
      <RoomConstructionModal
        worldId={worldId}
        sessions={constructionSessions}
        sessionId={effectiveConstructionSessionId}
        onSelectSession={setConstructionSessionId}
        isOpen={constructionModalOpen}
        onClose={() => setConstructionModalOpen(false)}
        onReconstructionSuccess={() => {
          refetchIr();
          refetchVersions();
          window.dispatchEvent(new CustomEvent("frame-all"));
        }}
      />
    </div>
  );
}
