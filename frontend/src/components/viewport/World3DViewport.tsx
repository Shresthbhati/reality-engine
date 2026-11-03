"use client";

/**
 * World3DViewport: the dark architectural-studio 3D viewport (mode tabs above, tool column right, toolbar bottom).
 *
 * Props (all optional ones are backwards compatible; omit them and the viewport manages that state itself):
 *  - mode / onModeChange: "realistic" | "wireframe" | "confidence". Controlled when `mode` is given, otherwise the
 *    choice is internal and remembered in localStorage (failures ignored).
 *  - cutaway / onCutawayChange: dollhouse cutaway (ceiling + camera-facing walls hidden). Default ON. A selected
 *    entity is always revealed, so choosing a cut-away wall from the hierarchy shows it.
 *  - hiddenEntityIds: Set<string> of entity ids to hide (shell parts and bounding boxes), e.g. hierarchy eye toggles.
 *  - onSelectEntity / selectedEntityId / queryMatchingIds / onMeasurementChange / worldId: unchanged.
 */

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  Maximize2,
  Eye,
  Camera,
  Box,
  AlertTriangle,
  Sparkles,
  Ruler,
  Grid,
  X,
  Layers,
  Workflow,
  MousePointer2,
  Hand,
  Orbit,
  ZoomIn,
  ZoomOut,
  Crosshair,
  Scissors,
  Download,
  MoveHorizontal,
  Compass,
  History,
  Globe,
  Cpu,
} from "lucide-react";
import {
  WorldSceneController,
  type ViewportLayers,
  type ViewPreset,
  type MeasurementResult,
  type ViewportStatistics,
  type PointerTool,
  type CameraInfo,
} from "@/lib/viewport/three-scene";
import type {
  CamerasPayload,
  WorldIR,
} from "@/types/worldir";
import { legendFor } from "@/lib/viewport/room-shell-model";
import { outdoorSummary } from "@/lib/viewport/outdoor-model";
import {
  VIEW_MODES,
  axisGizmo,
  formatMeters,
  loadStoredMode,
  niceScale,
  storeMode,
  viewModeToRender,
  type ViewMode,
} from "@/lib/viewport/view-model";
import { resolveUnitContext } from "@/lib/viewport/measure-model";
import { captureTitle, isUnplaced, type CaptureQuality, type StructurePoint } from "@/lib/viewport/structure-points-model";
import MeasurementPanel from "./MeasurementPanel";
import TransformGizmoToolbar from "@/components/studio/TransformGizmoToolbar";
import { CapturePlanPanel } from "@/components/studio/CapturePlanPanel";
import type { CapturePlanSummary } from "@/lib/studio/capture-plan-model";
import { StoreySelector } from "@/components/studio/StoreySelector";
import type { StoreyLevelItem } from "@/lib/studio/storey-model";
import { TemporalDiffSlider } from "@/components/studio/TemporalDiffSlider";
import { SpatialIntelligencePanel } from "@/components/studio/SpatialIntelligencePanel";
import { TechnologiesPanel } from "@/components/studio/TechnologiesPanel";
import type { SpatialIntelligenceSummary } from "@/lib/studio/spatial-intelligence-model";
import type { TemporalDiffSummary } from "@/lib/studio/temporal-model";
import { DEFAULT_SNAP_CONFIG, type TransformGizmoMode, type SnapConfig } from "@/lib/studio/transform-types";

const NO_SHELL_REASON =
  "No reconstructed room shell yet (no measured footprint). Only legacy bounding boxes can be drawn, so the view modes are unavailable.";

const PANEL = { background: "var(--bg-surface)", border: "1px solid var(--border)" } as const;

interface World3DViewportProps {
  /** Engineer overlays (stats, layer toggles, compass, empty/failed explanations). Default on; Simple mode turns it off. */
  chrome?: boolean;
  /** Layer visibility the scene starts with (merged over the defaults). */
  initialLayers?: Partial<ViewportLayers> | null;
  /** Entity id -> quality band. When set, entities are coloured by band instead of by type. */
  bandOverlay?: Record<string, string> | boolean | null;
  /** Increment to frame the whole model. */
  frameSignal?: number;
  world: WorldIR | null;
  points: Float32Array | null;
  cameras: CamerasPayload | null;
  mesh?: { positions: Float32Array; indices: Uint32Array } | null;
  /** Room corners / opening corners / dimension points (the cloud's frame). Shown instead of the dense cloud unless "Full point cloud" is on. */
  structurePoints?: StructurePoint[] | null;
  /** Capture-quality verdict. When "insufficient" the photos were not placed in one 3D space: no room is drawn, an explanation is shown. */
  captureQuality?: CaptureQuality | null;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  highlightedEvidenceId?: string | null;
  onSelectEvidence?: (id: string | null) => void;
  queryMatchingIds?: Set<string> | null;
  isLoading?: boolean;
  loadingMessage?: string;
  hasNoWorldData?: boolean;
  onMeasurementChange?: (measurement: MeasurementResult | null) => void;
  onOpenRoomConstruction?: () => void;
  /** Needed to save/list measurements; without it measuring still works but cannot persist. */
  worldId?: string | null;
  mode?: ViewMode;
  onModeChange?: (mode: ViewMode) => void;
  hiddenEntityIds?: Set<string>;
  cutaway?: boolean;
  onCutawayChange?: (on: boolean) => void;
  /** Next-best-view capture plan recommendations. */
  capturePlan?: CapturePlanSummary | null;
  onSelectCaptureViewpoint?: (viewpointId: string) => void;
  /** Multi-storey building vertical levels. */
  storeys?: StoreyLevelItem[] | null;
  activeStoreyId?: string | "all";
  onSelectStorey?: (storeyId: string | "all") => void;
  /** Multi-epoch temporal diff between baseline and current state. */
  temporalDiff?: TemporalDiffSummary | null;
  onSelectTemporalEntity?: (entityId: string) => void;
  /** Spatial intelligence summary (georeference, building envelope, architectural style, physics, lighting). */
  spatialIntelligence?: SpatialIntelligenceSummary | null;
}

function IconButton({
  label, active = false, disabled = false, onClick, children,
}: { label: string; active?: boolean; disabled?: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded transition-colors disabled:cursor-not-allowed disabled:opacity-35 cursor-pointer"
      style={{
        color: active ? "var(--accent)" : "var(--text-secondary)",
        background: active ? "var(--accent-subtle)" : "transparent",
      }}
    >
      {children}
    </button>
  );
}

export default function World3DViewport({
  world,
  points,
  cameras,
  mesh = null,
  structurePoints = null,
  captureQuality = null,
  selectedEntityId,
  onSelectEntity,
  highlightedEvidenceId,
  queryMatchingIds = null,
  isLoading = false,
  loadingMessage = "Initializing 3D spatial viewport...",
  hasNoWorldData = false,
  onMeasurementChange,
  onOpenRoomConstruction,
  worldId,
  mode: modeProp,
  onModeChange,
  hiddenEntityIds,
  cutaway: cutawayProp,
  onCutawayChange,
  capturePlan: capturePlanProp,
  onSelectCaptureViewpoint: onSelectCaptureViewpointProp,
  storeys: storeysProp,
  activeStoreyId: activeStoreyIdProp,
  onSelectStorey: onSelectStoreyProp,
  temporalDiff: temporalDiffProp,
  onSelectTemporalEntity: onSelectTemporalEntityProp,
  spatialIntelligence: spatialIntelligenceProp,
  chrome = true,
  initialLayers = null,
  bandOverlay = null,
  frameSignal = 0,
}: World3DViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<WorldSceneController | null>(null);
  const getController = useCallback(() => controllerRef.current, []);
  // Latest callbacks, so the controller is created once and never torn down by an unstable parent callback.
  const onSelectRef = useRef(onSelectEntity);
  const onMeasureRef = useRef(onMeasurementChange);
  onSelectRef.current = onSelectEntity;
  onMeasureRef.current = onMeasurementChange;

  const unplaced = isUnplaced(captureQuality);
  const [layers, setLayers] = useState<ViewportLayers>({
    points: true,
    fullCloud: false,
    entities: true,
    cameras: true,
    mesh: true,
    oversized: false,
    uncertainty: false,
    walls: true,
    topology: true,
    shell: true,
    ...(initialLayers ?? {}),
  });
  const [modeState, setModeState] = useState<ViewMode>("realistic");
  const mode = modeProp ?? modeState;
  const [cutawayState, setCutawayState] = useState(true);
  const cutaway = cutawayProp ?? cutawayState;
  const [hasShell, setHasShell] = useState(false);
  const [outdoorInfo, setOutdoorInfo] = useState<{ summary: string; total: number; imported: number; generated: number; skipped: number } | null>(null);
  // the view modes apply to the room shell AND to map-imported / generated outdoor parts
  const hasModes = hasShell || outdoorInfo !== null;
  const [tool, setTool] = useState<PointerTool>("select");
  const [gizmoMode, setGizmoMode] = useState<TransformGizmoMode>("translate");
  const [snapConfig, setSnapConfig] = useState<SnapConfig>(DEFAULT_SNAP_CONFIG);

  // Studio Intelligence state fallbacks (from props or world.metadata)
  const resolvedCapturePlan = capturePlanProp ?? ((world?.metadata as any)?.capture_plan as CapturePlanSummary | null) ?? null;
  const resolvedStoreys = storeysProp ?? ((world?.metadata as any)?.storeys as StoreyLevelItem[] | null) ?? ((world?.metadata as any)?.vertical_topology?.storeys as StoreyLevelItem[] | null) ?? null;
  const resolvedTemporalDiff = temporalDiffProp ?? ((world?.metadata as any)?.temporal_diff as TemporalDiffSummary | null) ?? null;

  const [showCapturePlan, setShowCapturePlan] = useState(false);
  const [selectedCaptureViewpointId, setSelectedCaptureViewpointId] = useState<string | null>(null);
  const [showCameraCones, setShowCameraCones] = useState(true);

  const [showStoreys, setShowStoreys] = useState(false);
  const [activeStoreyIdState, setActiveStoreyIdState] = useState<string | "all">("all");
  const activeStorey = activeStoreyIdProp ?? activeStoreyIdState;
  const [isolateStorey, setIsolateStorey] = useState(false);

  const [showTemporalDiff, setShowTemporalDiff] = useState(false);
  const [temporalSplitRatio, setTemporalSplitRatio] = useState(0.5);

  const resolvedSpatialIntelligence = useMemo<SpatialIntelligenceSummary | null>(() => {
    if (spatialIntelligenceProp) return spatialIntelligenceProp;
    if (!world) return null;
    const meta = (world.metadata ?? {}) as any;
    const georef = meta.georeference ?? null;
    const style = meta.design_style ?? null;
    const env = (world.environment ?? {}) as any;
    const lighting = env.lighting ?? null;
    const plausibility = meta.physical_plausibility ?? null;
    const completeness = meta.scene_completeness ?? null;
    const envelope = meta.building_envelope ?? null;
    const sharedWalls = meta.shared_walls ?? null;
    const transforms = world.transforms ?? {};

    if (!georef && !style && !lighting && !plausibility && !completeness && !envelope && Object.keys(transforms).length === 0) {
      return null;
    }
    return {
      world_id: world.id,
      coordinate_frame: String((world as any).coordinate_frame || world.coordinate_system || "WORLD"),
      transforms_count: Object.keys(transforms).length,
      georeference: georef,
      design_style: style,
      building_envelope: envelope,
      physical_plausibility: plausibility,
      scene_completeness: completeness,
      lighting: lighting,
      shared_walls: sharedWalls,
    };
  }, [spatialIntelligenceProp, world]);

  const [showSpatialIntelligence, setShowSpatialIntelligence] = useState(false);
  const [showTechnologies, setShowTechnologies] = useState(false);

  const handleGizmoModeChange = useCallback((m: TransformGizmoMode) => {
    setGizmoMode(m);
    controllerRef.current?.setTransformMode(m);
  }, []);

  const handleSnapConfigChange = useCallback((c: Partial<SnapConfig>) => {
    setSnapConfig((prev) => {
      const next = { ...prev, ...c };
      controllerRef.current?.setGizmoSnapConfig(next);
      return next;
    });
  }, []);

  const handleUndo = useCallback(() => {
    controllerRef.current?.getTransformGizmo()?.undo();
  }, []);

  const handleRedo = useCallback(() => {
    controllerRef.current?.getTransformGizmo()?.redo();
  }, []);
  const [ortho, setOrtho] = useState(false);
  const [showDims, setShowDims] = useState(true);
  const [layersOpen, setLayersOpen] = useState(false);

  const [gridVisible, setGridVisible] = useState(true);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [isContextLost, setIsContextLost] = useState(false);
  const [stats, setStats] = useState<ViewportStatistics>({
    points: 0,
    canonicalPoints: 0,
    renderedPoints: 0,
    isLODSampled: false,
    pointMode: "none",
    cameras: 0,
    entities: 0,
    isContextLost: false,
  });
  const [isolatedSpaceId, setIsolatedSpaceId] = useState<string | null>(null);
  const [activeLevel, setActiveLevel] = useState<number | null>(null);
  const [camInfo, setCamInfo] = useState<CameraInfo | null>(null);

  const recon = world?.metadata?.reconstruction;
  const reconStatus = recon?.registration_status?.toLowerCase();
  const isReconFailed = reconStatus === "failed";
  const isReconPartial =
    reconStatus === "partial" ||
    ((recon?.cameras_input ?? 0) > (recon?.cameras_registered ?? 0) &&
      (recon?.cameras_registered ?? 0) > 0);
  const isUnscaled =
    world?.metadata?.scale?.state === "unscaled" ||
    world?.metadata?.scale?.state === "nominal";
  // Metres are only claimed (scale bar, dimension labels) when the world states a metric or authored scale.
  // Map-imported worlds (metadata.geo) are metric by construction: OSM coordinates are metres from the anchor.
  const metersKnown = resolveUnitContext(world).basis !== "unknown" || Boolean((world?.metadata as { geo?: unknown } | undefined)?.geo);

  // Initialize Three.js Controller (once; callbacks go through refs)
  useEffect(() => {
    if (!containerRef.current) return;

    const controller = new WorldSceneController(containerRef.current);
    controllerRef.current = controller;

    controller.setCallbacks({
      onSelect: (id) => onSelectRef.current(id),
      onMeasure: (res) => {
        onMeasureRef.current?.(res);
      },
      onCameraChange: setCamInfo,
      onContextLossChange: (lost) => {
        setIsContextLost(lost);
      },
    });

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, []);

  // Remember the chosen mode (storage failures are swallowed inside view-model)
  useEffect(() => {
    if (modeProp === undefined) setModeState(loadStoredMode());
  }, [modeProp]);

  const changeMode = useCallback((next: ViewMode) => {
    setModeState(next);
    storeMode(next);
    onModeChange?.(next);
  }, [onModeChange]);

  const changeCutaway = useCallback((on: boolean) => {
    setCutawayState(on);
    onCutawayChange?.(on);
  }, [onCutawayChange]);

  // Sync data into scene
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;

    if (world) {
      controller.loadWorldData(world, points, cameras, mesh, { structurePoints, suppressLegacy: unplaced });
      setStats(controller.getStatistics());
      setHasShell(controller.hasShell());
      if (controller.hasOutdoor()) {
        const o = controller.getOutdoorSummary();
        setOutdoorInfo({ summary: outdoorSummary(o.parts), total: o.parts.length, imported: o.imported, generated: o.generated, skipped: o.skipped });
      } else {
        setOutdoorInfo(null);
      }
    }
  }, [world, points, cameras, mesh, structurePoints, unplaced]);

  // The point layer is swapped inside the controller when "Full point cloud" changes: refresh the HUD counts.
  useEffect(() => {
    const controller = controllerRef.current;
    if (controller) setStats(controller.getStatistics());
  }, [layers.fullCloud]);

  // View mode / cutaway / hidden entities (re-applied after every data load so a rebuilt shell keeps them)
  useEffect(() => {
    controllerRef.current?.setRenderMode(viewModeToRender(mode));
  }, [mode, world, points, cameras, mesh, structurePoints, unplaced]);

  useEffect(() => {
    controllerRef.current?.setCutaway(cutaway);
  }, [cutaway, world, points, cameras, mesh, structurePoints, unplaced]);

  useEffect(() => {
    controllerRef.current?.setHiddenEntities(hiddenEntityIds ?? null);
  }, [hiddenEntityIds, world, points, cameras, mesh, structurePoints, unplaced]);

  useEffect(() => {
    controllerRef.current?.setShowDimensions(showDims && metersKnown);
  }, [showDims, metersKnown, world, points, cameras, mesh, structurePoints, unplaced]);

  // Sync spatial query filtering
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    controller.applyEntityFilter(queryMatchingIds ?? null);
  }, [queryMatchingIds]);

  // Sync external selection (e.g. from nav panel or inspector)
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    controller.selectEntity(selectedEntityId, true);
  }, [selectedEntityId]);

  // Sync external camera/evidence focus
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller || !highlightedEvidenceId) return;
    controller.flyToCamera(highlightedEvidenceId);
  }, [highlightedEvidenceId]);

  // Quality overlay: colour entities by band. Re-applied after every scene rebuild, cleared when off.
  useEffect(() => {
    controllerRef.current?.setBandOverlay(typeof bandOverlay === "object" && bandOverlay !== null ? bandOverlay as Record<string, string> : null);
  }, [bandOverlay, world, points, cameras, mesh]);

  // The first layers (e.g. cameras off in Simple mode) must reach the scene, which starts from its own defaults.
  useEffect(() => {
    if (initialLayers) controllerRef.current?.setLayerVisibility({ ...layers });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world]);

  useEffect(() => {
    if (frameSignal > 0) controllerRef.current?.frameAll();
  }, [frameSignal]);

  // Listen for custom frame and spatial manipulation events
  useEffect(() => {
    const handleFlyToEntity = (e: Event) => {
      const customEvent = e as CustomEvent<{ id: string }>;
      if (customEvent.detail?.id) controllerRef.current?.flyToEntity(customEvent.detail.id);
    };
    const handleFlyToCamera = (e: Event) => {
      const customEvent = e as CustomEvent<{ id: string }>;
      if (customEvent.detail?.id) controllerRef.current?.flyToCamera(customEvent.detail.id);
    };
    const handleFlyToPosition = (e: Event) => {
      const customEvent = e as CustomEvent<{ x: number; y: number; z: number }>;
      if (customEvent.detail) controllerRef.current?.flyToPosition(customEvent.detail.x, customEvent.detail.y, customEvent.detail.z);
    };
    const handleIsolateSpace = (e: Event) => {
      const customEvent = e as CustomEvent<{ id: string | null }>;
      const id = customEvent.detail?.id ?? null;
      controllerRef.current?.isolateSpace(id);
      setIsolatedSpaceId(id);
    };
    const handleFilterLevel = (e: Event) => {
      const customEvent = e as CustomEvent<{ levelIndex: number | null }>;
      const lvl = customEvent.detail?.levelIndex ?? null;
      controllerRef.current?.setLevelFilter(lvl);
      setActiveLevel(lvl);
    };
    const handlePreviewCorrection = (e: Event) => {
      const customEvent = e as CustomEvent<{
        entityId: string;
        type?: string;
        confidence?: number | null;
        transform?: { position?: { x: number; y: number; z: number } };
      }>;
      if (customEvent.detail) {
        controllerRef.current?.setCorrectionPreview(
          customEvent.detail.entityId,
          customEvent.detail.type,
          customEvent.detail.confidence,
          customEvent.detail.transform
        );
      }
    };
    const handleClearCorrectionPreview = () => {
      controllerRef.current?.clearCorrectionPreview();
    };
    const handleRequestCameraPose = () => {
      const pose = controllerRef.current?.getCameraPose();
      if (pose) {
        window.dispatchEvent(new CustomEvent("viewport-camera-pose-response", { detail: pose }));
      }
    };
    const handleFlyToBookmark = (e: Event) => {
      const customEvent = e as CustomEvent<{
        position: [number, number, number];
        target: [number, number, number];
      }>;
      if (customEvent.detail?.position && customEvent.detail?.target) {
        controllerRef.current?.flyToBookmark(customEvent.detail.position, customEvent.detail.target);
      }
    };

    window.addEventListener("frame-entity", handleFlyToEntity as EventListener);
    window.addEventListener("frame-camera", handleFlyToCamera as EventListener);
    window.addEventListener("frame-position", handleFlyToPosition as EventListener);
    window.addEventListener("isolate-space", handleIsolateSpace as EventListener);
    window.addEventListener("filter-level", handleFilterLevel as EventListener);
    window.addEventListener("preview-correction", handlePreviewCorrection as EventListener);
    window.addEventListener("clear-correction-preview", handleClearCorrectionPreview as EventListener);
    window.addEventListener("request-viewport-camera-pose", handleRequestCameraPose as EventListener);
    window.addEventListener("fly-to-bookmark", handleFlyToBookmark as EventListener);

    return () => {
      window.removeEventListener("frame-entity", handleFlyToEntity as EventListener);
      window.removeEventListener("frame-camera", handleFlyToCamera as EventListener);
      window.removeEventListener("frame-position", handleFlyToPosition as EventListener);
      window.removeEventListener("isolate-space", handleIsolateSpace as EventListener);
      window.removeEventListener("filter-level", handleFilterLevel as EventListener);
      window.removeEventListener("preview-correction", handlePreviewCorrection as EventListener);
      window.removeEventListener("clear-correction-preview", handleClearCorrectionPreview as EventListener);
      window.removeEventListener("request-viewport-camera-pose", handleRequestCameraPose as EventListener);
      window.removeEventListener("fly-to-bookmark", handleFlyToBookmark as EventListener);
    };
  }, []);

  // Toggle Layer Helper
  const toggleLayer = useCallback((key: keyof ViewportLayers) => {
    setLayers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      controllerRef.current?.setLayerVisibility(next);
      return next;
    });
  }, []);

  const toggleGrid = useCallback(() => {
    setGridVisible((prev) => {
      const next = !prev;
      controllerRef.current?.setGridVisible(next);
      return next;
    });
  }, []);

  const toggleMeasure = useCallback(() => {
    setIsMeasuring((prev) => {
      const next = !prev;
      controllerRef.current?.setMeasurementMode(next);
      if (!next) {
        onMeasurementChange?.(null);
      }
      return next;
    });
  }, [onMeasurementChange]);

  const handlePreset = useCallback((preset: ViewPreset) => {
    controllerRef.current?.setViewPreset(preset);
  }, []);

  const handleFit = useCallback(() => controllerRef.current?.frameAll(), []);

  const handleFocus = useCallback(() => {
    if (selectedEntityId) controllerRef.current?.flyToEntity(selectedEntityId);
  }, [selectedEntityId]);

  const handleFrameAll = useCallback(() => {
    if (selectedEntityId) {
      controllerRef.current?.flyToEntity(selectedEntityId);
    } else {
      controllerRef.current?.frameAll();
    }
  }, [selectedEntityId]);

  const pickTool = useCallback((next: PointerTool) => {
    setTool(next);
    controllerRef.current?.setPointerTool(next);
  }, []);

  const changeProjection = useCallback((nextOrtho: boolean) => {
    setOrtho(nextOrtho);
    controllerRef.current?.setOrthographic(nextOrtho);
  }, []);

  const handleScreenshot = useCallback(() => {
    const url = controllerRef.current?.captureImage();
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${worldId ?? "reality-viewport"}.png`;
    a.click();
  }, [worldId]);

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid intercepting input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        handleFrameAll();
      } else if (e.key === "1") {
        toggleLayer("points");
      } else if (e.key === "2") {
        toggleLayer("entities");
      } else if (e.key === "3") {
        toggleLayer("cameras");
      } else if (e.key === "w" || e.key === "W") {
        toggleLayer("walls");
      } else if (e.key === "t" || e.key === "T") {
        toggleLayer("topology");
      } else if (e.key === "g" || e.key === "G") {
        toggleGrid();
      } else if (e.key === "m" || e.key === "M") {
        toggleMeasure();
      } else if (e.key === "o" || e.key === "O") {
        toggleLayer("oversized");
      } else if (e.key === "u" || e.key === "U") {
        toggleLayer("uncertainty");
      } else if (e.key === "c" || e.key === "C") {
        changeCutaway(!cutaway);
      } else if (e.key === "Escape") {
        if (isMeasuring) {
          toggleMeasure();
        } else if (isolatedSpaceId) {
          controllerRef.current?.isolateSpace(null);
          setIsolatedSpaceId(null);
        } else {
          onSelectEntity(null);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleFrameAll, toggleLayer, toggleGrid, toggleMeasure, changeCutaway, cutaway, isMeasuring, isolatedSpaceId, onSelectEntity]);

  // Scale bar + axis gizmo (derived from the camera the controller reports)
  const scale = useMemo(() => (camInfo && metersKnown ? niceScale(camInfo.metersPerPx, 110) : null), [camInfo, metersKnown]);
  const axes = useMemo(() => (camInfo ? axisGizmo(camInfo.right, camInfo.up) : null), [camInfo]);
  const legend = hasModes ? legendFor(viewModeToRender(mode)).filter((item) => hasShell || !item.hatch) : [];

  return (
    <div className="flex h-full w-full select-none flex-col overflow-hidden bg-[var(--bg-base)]">
      {/* Mode tabs: Realistic | Wireframe | Confidence. One canonical world, three materials. */}
      {chrome && (
      <div className="flex shrink-0 items-center justify-center px-3 py-2" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
        <div
          role="group"
          aria-label="Viewport mode"
          data-testid="render-mode-group"
          title={hasModes ? undefined : NO_SHELL_REASON}
          className="flex items-center gap-1 rounded-lg p-1"
          style={PANEL}
        >
          {VIEW_MODES.map(({ id, label }) => {
            const selected = hasModes && mode === id;
            return (
              <button
                key={id}
                type="button"
                data-testid={`render-mode-${id}`}
                aria-pressed={selected}
                disabled={!hasModes}
                onClick={() => changeMode(id)}
                className="h-8 min-w-[104px] rounded-md px-4 text-sm font-medium transition-colors disabled:cursor-not-allowed cursor-pointer"
                style={{
                  color: !hasModes ? "var(--text-tertiary)" : selected ? "var(--accent)" : "var(--text-secondary)",
                  background: selected ? "var(--accent-subtle)" : "transparent",
                  border: `1px solid ${selected ? "var(--accent-border)" : "transparent"}`,
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>
      )}

      <div className="relative min-h-0 flex-1">
        {/* 3D Canvas Mount Point */}
        <div
          ref={containerRef}
          className={`w-full h-full ${
            isMeasuring ? "cursor-crosshair" : tool === "pan" ? "cursor-move" : "cursor-grab active:cursor-grabbing"
          }`}
        />

        {/* Top left: stats + state badges */}
        <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-2 max-w-[calc(100%-6rem)] pointer-events-none">
          <div
            className="flex items-center gap-3 px-3 py-1.5 rounded-md text-xs font-mono-num"
            style={{ ...PANEL, color: "var(--text-secondary)" }}
          >
            {stats.pointMode === "structure" ? (
              <span
                data-testid="point-mode-structure"
                title={`${stats.renderedPoints.toLocaleString()} structure points (room corners, opening corners, dimension points). The dense cloud${stats.canonicalPoints > 0 ? ` (${stats.canonicalPoints.toLocaleString()} points)` : ""} is hidden: turn on "Full point cloud" to see it.`}
              >
                {stats.renderedPoints.toLocaleString()} structure pts
              </span>
            ) : stats.pointMode === "decimated" ? (
              <span
                className="text-[var(--info)]"
                data-testid="point-mode-decimated"
                title={`Cloud thinned to ${stats.renderedPoints.toLocaleString()} of ${stats.canonicalPoints.toLocaleString()} points. No structure points were reported. Turn on "Full point cloud" to see all of it.`}
              >
                {stats.renderedPoints.toLocaleString()} / {stats.canonicalPoints.toLocaleString()} pts (thinned)
              </span>
            ) : stats.isLODSampled ? (
              <span
                className="text-[var(--info)]"
                title={`${stats.renderedPoints.toLocaleString()} rendered / ${stats.canonicalPoints.toLocaleString()} canonical points (LOD strided for interactive WebGL rendering)`}
              >
                {stats.renderedPoints.toLocaleString()} / {stats.canonicalPoints.toLocaleString()} pts (LOD)
              </span>
            ) : (
              <span>{stats.points.toLocaleString()} pts</span>
            )}
            <span>{stats.entities} entities</span>
            <span>{stats.cameras} cameras</span>
          </div>

          {stats.canonicalPoints > 0 && (
            <button
              type="button"
              data-testid="full-point-cloud-toggle"
              aria-pressed={layers.fullCloud}
              onClick={() => toggleLayer("fullCloud")}
              title="Show every point of the dense cloud instead of only the structure points"
              className="pointer-events-auto flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-mono cursor-pointer"
              style={{
                ...PANEL,
                color: layers.fullCloud ? "var(--accent)" : "var(--text-secondary)",
                background: layers.fullCloud ? "var(--accent-subtle)" : PANEL.background,
              }}
            >
              <Eye className="h-3.5 w-3.5" aria-hidden />
              Full point cloud
            </button>
          )}

          {outdoorInfo && (
            <div
              className="flex flex-col gap-0.5 px-3 py-1.5 rounded-md text-xs font-mono-num"
              style={{ ...PANEL, color: "var(--text-secondary)" }}
              data-testid="outdoor-hud"
              title="Massing proxies from map data: footprints are real map outlines, heights and furniture sizes are defaults unless tagged."
            >
              <span>{outdoorInfo.summary}</span>
              {outdoorInfo.imported > 0 && (
                <span className="text-[10px] text-[var(--text-tertiary)]">approximate - imported from OpenStreetMap (c) OpenStreetMap contributors</span>
              )}
              {outdoorInfo.generated > 0 && (
                <span className="text-[10px] text-[var(--text-tertiary)]">{outdoorInfo.generated} generated (not captured), drawn translucent</span>
              )}
              {outdoorInfo.skipped > 0 && (
                <span className="text-[10px] text-[var(--warning)]">{outdoorInfo.skipped} without geometry not drawn</span>
              )}
            </div>
          )}

          {queryMatchingIds !== null && (
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono bg-[var(--accent-subtle)] border border-[var(--accent-border)] text-[var(--accent)]">
              <span>Query: {queryMatchingIds.size} matching</span>
            </div>
          )}

          {isolatedSpaceId && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-mono bg-[var(--info-subtle)] border border-[var(--info)] text-[var(--info)] pointer-events-auto">
              <span>Isolated Space: {isolatedSpaceId}</span>
              <button
                type="button"
                onClick={() => {
                  controllerRef.current?.isolateSpace(null);
                  setIsolatedSpaceId(null);
                }}
                className="text-[var(--text-primary)] ml-1 underline cursor-pointer text-[10px]"
              >
                Exit [Esc]
              </button>
            </div>
          )}

          {activeLevel !== null && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-mono bg-[var(--info-subtle)] border border-[var(--info)] text-[var(--info)] pointer-events-auto">
              <span>Level {activeLevel} Filter</span>
              <button
                type="button"
                onClick={() => {
                  controllerRef.current?.setLevelFilter(null);
                  setActiveLevel(null);
                }}
                className="text-[var(--text-primary)] ml-1 underline cursor-pointer text-[10px]"
              >
                Reset
              </button>
            </div>
          )}

          {isMeasuring && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-mono bg-[var(--accent-subtle)] border border-[var(--accent)] text-[var(--accent)] pointer-events-auto">
              <Ruler className="w-3.5 h-3.5" />
              <span>Measuring: pick points in the viewport</span>
              <button type="button" onClick={toggleMeasure} className="text-[var(--text-secondary)] ml-1 cursor-pointer" aria-label="Stop measuring">
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          {isReconPartial && (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono bg-[var(--warning-subtle)] border border-[var(--warning)] text-[var(--warning)] pointer-events-auto"
              role="status"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>
                Partial Reconstruction: {recon?.cameras_registered ?? 0}/{recon?.cameras_input ?? 0} cams registered
              </span>
            </div>
          )}

          {isUnscaled && !isReconFailed && (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono"
              style={{ ...PANEL, color: "var(--text-secondary)" }}
              title="Spatial model scale is unconstrained or nominal. Relative proportions are accurate, absolute dimensions carry scale variance."
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--warning)]" />
              <span>Scale Unconstrained</span>
            </div>
          )}
        </div>

        {/* Interactive 3D Transform Gizmo Toolbar */}
        {selectedEntityId && tool === "select" && (
          <div className="absolute top-14 left-1/2 -translate-x-1/2 z-20 pointer-events-auto">
            <TransformGizmoToolbar
              mode={gizmoMode}
              onModeChange={handleGizmoModeChange}
              snapConfig={snapConfig}
              onSnapConfigChange={handleSnapConfigChange}
              selectedEntityId={selectedEntityId}
              onUndo={handleUndo}
              onRedo={handleRedo}
            />
          </div>
        )}

        {/* Interactive Measurement HUD Card */}
        {isMeasuring && (
          <MeasurementPanel
            worldId={worldId ?? null}
            world={world}
            getController={getController}
            onClose={toggleMeasure}
          />
        )}

        {/* Multi-Storey Building Level Selector Overlay */}
        {showStoreys && resolvedStoreys && (
          <div className="absolute top-20 left-4 z-20 w-56 pointer-events-auto">
            <StoreySelector
              storeys={resolvedStoreys}
              activeStoreyId={activeStorey}
              onSelectStorey={(id) => {
                setActiveStoreyIdState(id);
                onSelectStoreyProp?.(id);
              }}
              isolateActiveStorey={isolateStorey}
              onToggleIsolate={setIsolateStorey}
            />
          </div>
        )}

        {/* Next-Best-View Intelligent Capture Planner Overlay */}
        {showCapturePlan && resolvedCapturePlan && (
          <div className="absolute top-20 right-14 z-20 w-80 max-h-[75vh] overflow-y-auto pointer-events-auto">
            <CapturePlanPanel
              plan={resolvedCapturePlan}
              selectedViewpointId={selectedCaptureViewpointId}
              onSelectViewpoint={(id) => {
                setSelectedCaptureViewpointId(id);
                onSelectCaptureViewpointProp?.(id);
              }}
              showCameraCones={showCameraCones}
              onToggleCameraCones={setShowCameraCones}
            />
          </div>
        )}

        {/* Multi-Epoch Temporal Diff Comparison Slider Overlay */}
        {showTemporalDiff && resolvedTemporalDiff && (
          <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-20 w-[480px] max-w-[calc(100%-2rem)] pointer-events-auto">
            <TemporalDiffSlider
              diff={resolvedTemporalDiff}
              splitRatio={temporalSplitRatio}
              onSplitRatioChange={setTemporalSplitRatio}
              selectedEntityId={selectedEntityId}
              onSelectEntity={(id) => {
                onSelectEntity(id);
                onSelectTemporalEntityProp?.(id);
              }}
            />
          </div>
        )}

        {/* Spatial Intelligence Overlay */}
        {showSpatialIntelligence && resolvedSpatialIntelligence && (
          <div
            className={`absolute top-14 ${showStoreys && resolvedStoreys && resolvedStoreys.length > 0 ? "left-48" : "left-3"} z-20 w-[360px] max-w-[calc(100%-2rem)] pointer-events-auto`}
          >
            <SpatialIntelligencePanel summary={resolvedSpatialIntelligence} />
          </div>
        )}

        {/* 50 Technologies Evaluation Overlay */}
        {showTechnologies && (
          <div className="absolute top-14 right-3 z-30 w-[420px] max-w-[calc(100%-2rem)] pointer-events-auto">
            <TechnologiesPanel onClose={() => setShowTechnologies(false)} />
          </div>
        )}

        {/* Top right: XYZ axis gizmo */}
        {axes && (
          <svg
            width="64" height="64" viewBox="-32 -32 64 64"
            className="absolute top-3 right-3 z-10 pointer-events-none"
            role="img" aria-label="Axis gizmo"
          >
            {(["x", "y", "z"] as const).map((k) => {
              const [dx, dy] = axes[k];
              const color = k === "x" ? "var(--error)" : k === "y" ? "var(--success)" : "var(--info)";
              return (
                <g key={k}>
                  <line x1="0" y1="0" x2={dx * 22} y2={-dy * 22} stroke={color} strokeWidth="1.5" />
                  <text x={dx * 28} y={-dy * 28 + 3} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill={color}>
                    {k.toUpperCase()}
                  </text>
                </g>
              );
            })}
          </svg>
        )}

        {/* Right edge: slim vertical tool column */}
        <div
          className="absolute right-3 top-20 z-10 flex flex-col items-center gap-0.5 rounded-lg p-1"
          style={PANEL}
          role="toolbar"
          aria-label="Viewport tools"
          aria-orientation="vertical"
        >
          <IconButton label="Select (click to inspect, drag to orbit)" active={tool === "select"} onClick={() => pickTool("select")}>
            <MousePointer2 className="w-4 h-4" />
          </IconButton>
          <IconButton label="Pan (drag to move the view)" active={tool === "pan"} onClick={() => pickTool("pan")}>
            <Hand className="w-4 h-4" />
          </IconButton>
          <IconButton label="Orbit (drag to rotate, no selection)" active={tool === "orbit"} onClick={() => pickTool("orbit")}>
            <Orbit className="w-4 h-4" />
          </IconButton>
          <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
          <IconButton label="Zoom in" onClick={() => controllerRef.current?.zoomBy(0.75)}>
            <ZoomIn className="w-4 h-4" />
          </IconButton>
          <IconButton label="Zoom out" onClick={() => controllerRef.current?.zoomBy(1 / 0.75)}>
            <ZoomOut className="w-4 h-4" />
          </IconButton>
          <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
          <IconButton label="Camera frustums [3]" active={layers.cameras} onClick={() => toggleLayer("cameras")}>
            <Camera className="w-4 h-4" />
          </IconButton>
          <IconButton label="Measure [M]" active={isMeasuring} onClick={toggleMeasure}>
            <Ruler className="w-4 h-4" />
          </IconButton>
          {resolvedStoreys && resolvedStoreys.length > 0 && (
            <>
              <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
              <IconButton
                label="Storey levels selector"
                active={showStoreys}
                onClick={() => setShowStoreys((v) => !v)}
              >
                <Layers className="w-4 h-4 text-indigo-400" />
              </IconButton>
            </>
          )}
          {resolvedCapturePlan && resolvedCapturePlan.recommended_viewpoints.length > 0 && (
            <>
              <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
              <IconButton
                label="Next-best-view capture plan"
                active={showCapturePlan}
                onClick={() => setShowCapturePlan((v) => !v)}
              >
                <Compass className="w-4 h-4 text-amber-400" />
              </IconButton>
            </>
          )}
          {resolvedTemporalDiff && resolvedTemporalDiff.changes.length > 0 && (
            <>
              <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
              <IconButton
                label="Temporal epoch diffing"
                active={showTemporalDiff}
                onClick={() => setShowTemporalDiff((v) => !v)}
              >
                <History className="w-4 h-4 text-emerald-400" />
              </IconButton>
            </>
          )}
          {resolvedSpatialIntelligence && (
            <>
              <div className="my-0.5 h-px w-5" style={{ background: "var(--border)" }} />
              <IconButton
                label="Spatial Intelligence (georeference, style, envelope, physics)"
                active={showSpatialIntelligence}
                onClick={() => setShowSpatialIntelligence((v) => !v)}
              >
                <Globe className="w-4 h-4 text-emerald-400" />
              </IconButton>
              <IconButton
                label="50 Open-Source Technologies Catalog & Benchmarking"
                active={showTechnologies}
                onClick={() => setShowTechnologies((v) => !v)}
              >
                <Cpu className="w-4 h-4 text-indigo-400" />
              </IconButton>
            </>
          )}
        </div>

        {/* Bottom centre: compact toolbar */}
        <div className="absolute bottom-3 left-1/2 z-10 -translate-x-1/2 max-w-[calc(100%-1.5rem)]">
          {layersOpen && (
            <div
              className="mb-2 mx-auto w-[320px] max-w-full rounded-lg p-2 text-xs"
              style={{ ...PANEL, color: "var(--text-secondary)" }}
              role="group"
              aria-label="Layers and views"
            >
              <div className="mb-1.5 flex items-center gap-1">
                <span className="mr-1 text-[var(--text-tertiary)]">View</span>
                {([["isometric", "Iso"], ["top", "Top"], ["front", "Front"], ["side", "Side"]] as const).map(([p, l]) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => handlePreset(p)}
                    className="h-6 rounded px-2 font-mono hover:bg-[var(--bg-hover)] cursor-pointer"
                  >
                    {l}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                {([
                  ["points", "Points [1]", layers.points, () => toggleLayer("points"), Eye],
                  ["entities", "Entities [2]", layers.entities, () => toggleLayer("entities"), Box],
                  ["walls", "Walls [W]", layers.walls, () => toggleLayer("walls"), Layers],
                  ["topology", "Topology [T]", layers.topology, () => toggleLayer("topology"), Workflow],
                  ["grid", "Grid [G]", gridVisible, toggleGrid, Grid],
                  ["oversized", "Noise [O]", layers.oversized, () => toggleLayer("oversized"), AlertTriangle],
                  ["uncertainty", "Uncertainty [U]", layers.uncertainty, () => toggleLayer("uncertainty"), Sparkles],
                ] as const).map(([key, label, on, fn, Icon]) => (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={on}
                    onClick={fn}
                    className="flex h-7 items-center gap-1.5 rounded px-1.5 text-left cursor-pointer hover:bg-[var(--bg-hover)]"
                    style={{ color: on ? "var(--accent)" : "var(--text-tertiary)" }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="flex items-center justify-center gap-0.5 rounded-lg p-1" style={PANEL} role="toolbar" aria-label="Camera and scene">
            <IconButton label="Isometric view" onClick={() => handlePreset("isometric")}>
              <Box className="w-4 h-4" />
            </IconButton>
            <IconButton label="Fit to scene [F]" onClick={handleFit}>
              <Maximize2 className="w-4 h-4" />
            </IconButton>
            <IconButton label={selectedEntityId ? "Focus selected [F]" : "Focus selected (select something first)"} disabled={!selectedEntityId} onClick={handleFocus}>
              <Crosshair className="w-4 h-4" />
            </IconButton>
            <IconButton label="Save screenshot (PNG)" onClick={handleScreenshot}>
              <Download className="w-4 h-4" />
            </IconButton>
            <div className="mx-0.5 h-5 w-px" style={{ background: "var(--border)" }} />
            <IconButton
              label={hasShell ? "Cutaway: hide ceiling and camera-facing walls [C]" : "Cutaway needs a reconstructed room shell"}
              active={hasShell && cutaway}
              disabled={!hasShell}
              onClick={() => changeCutaway(!cutaway)}
            >
              <Scissors className="w-4 h-4" />
            </IconButton>
            {mode === "wireframe" && hasShell && metersKnown && (
              <IconButton label="Room dimensions (bounding length, width, height)" active={showDims} onClick={() => setShowDims((v) => !v)}>
                <MoveHorizontal className="w-4 h-4" />
              </IconButton>
            )}
            <IconButton label="Layers and views" active={layersOpen} onClick={() => setLayersOpen((v) => !v)}>
              <Layers className="w-4 h-4" />
            </IconButton>
            <div className="mx-0.5 h-5 w-px" style={{ background: "var(--border)" }} />
            <select
              aria-label="Camera projection"
              value={ortho ? "ortho" : "persp"}
              onChange={(e) => changeProjection(e.target.value === "ortho")}
              className="h-8 rounded border-0 bg-transparent px-1 text-xs cursor-pointer"
              style={{ color: "var(--text-secondary)" }}
            >
              <option value="persp" style={{ background: "var(--bg-surface)" }}>Perspective</option>
              <option value="ortho" style={{ background: "var(--bg-surface)" }}>Orthographic</option>
            </select>
          </div>
        </div>

        {/* Bottom left: legend / honest no-shell notice */}
        {!hasNoWorldData && !unplaced && (!hasModes || legend.length > 0) && (
          <div
            className="absolute bottom-3 left-3 z-10 max-w-[240px] rounded-md px-2.5 py-2 text-[11px] leading-snug pointer-events-none"
            style={PANEL}
            data-testid="render-mode-legend"
          >
            {!hasModes ? (
              <span className="text-[var(--warning)]">{NO_SHELL_REASON}</span>
            ) : (
              <ul className="space-y-1">
                {legend.map((item) => (
                  <li key={item.label} className="flex items-center gap-2 text-[var(--text-secondary)]">
                    <span
                      className="inline-block h-3 w-5 shrink-0 rounded-sm border border-[var(--border)]"
                      style={{
                        background: item.hatch
                          ? `repeating-linear-gradient(45deg, rgb(${item.color.join(",")}) 0 3px, rgba(0,0,0,0.55) 3px 5px)`
                          : `rgb(${item.color.join(",")})`,
                      }}
                    />
                    {item.label}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Bottom right: metric scale bar (+ uncertainty legend when on) */}
        <div className="absolute bottom-3 right-3 z-10 flex flex-col items-end gap-2 pointer-events-none">
          {layers.uncertainty && (
            <div className="px-2.5 py-1.5 rounded-md text-[10px] font-mono flex items-center gap-2" style={{ ...PANEL, color: "var(--text-secondary)" }}>
              <span>Confidence</span>
              <span className="h-2 w-2 rounded-full" style={{ background: "var(--conf-low)" }} />
              <span>low</span>
              <span className="h-2 w-2 rounded-full" style={{ background: "var(--conf-high)" }} />
              <span>high</span>
            </div>
          )}
          {scale ? (
            <div className="flex flex-col items-end gap-1 text-[10px] font-mono" style={{ color: "var(--text-secondary)" }} aria-label="Scale bar">
              <span>{formatMeters(scale.meters)}</span>
              <div style={{ width: scale.px, height: 6, borderLeft: "1px solid var(--text-secondary)", borderRight: "1px solid var(--text-secondary)", borderBottom: "1px solid var(--text-secondary)" }} />
            </div>
          ) : !metersKnown ? (
            <span className="text-[10px] font-mono text-[var(--text-tertiary)]">scale unknown (model units)</span>
          ) : null}
        </div>

      {/* WebGL Context Loss Banner Overlay */}
      {isContextLost && (
        <div
          className="absolute inset-0 z-30 flex flex-col items-center justify-center p-8 text-center bg-[var(--bg-base)]/95 pointer-events-auto"
          role="alert"
          aria-live="assertive"
        >
          <div className="w-14 h-14 rounded-full flex items-center justify-center bg-amber-950/40 border border-amber-500/40 text-amber-400 mb-4 shadow-inner">
            <AlertTriangle className="w-7 h-7 text-amber-400 animate-pulse" />
          </div>
          <h3 className="text-base font-semibold text-white mb-2 tracking-wide">
            WebGL Context Lost
          </h3>
          <p className="text-xs text-neutral-400 max-w-lg mb-4 leading-relaxed">
            The GPU graphics context was reclaimed or interrupted by the operating system.
            Interaction is temporarily paused to preserve spatial data integrity.
          </p>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-neutral-900 border border-neutral-800 text-xs font-mono text-neutral-300">
            <div className="w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
            <span>Awaiting WebGL context restoration...</span>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[var(--bg-base)]/80 backdrop-blur-sm">
          <div className="w-10 h-10 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-sm font-medium text-white">{loadingMessage}</p>
          <span className="text-xs text-neutral-500 mt-1 font-mono">Consuming canonical WorldIR stream</span>
        </div>
      )}

      {/* Explicit Failed Reconstruction State Overlay */}
      {!isLoading && isReconFailed && (
        <div
          className="absolute inset-0 z-10 flex flex-col items-center justify-center p-8 text-center bg-[var(--bg-base)]/92 pointer-events-auto"
          role="alert"
        >
          <div className="w-14 h-14 rounded-full flex items-center justify-center bg-red-950/40 border border-red-500/40 text-red-400 mb-4 shadow-inner">
            <AlertTriangle className="w-7 h-7 text-red-400" />
          </div>
          <h3 className="text-base font-semibold text-white mb-2 tracking-wide">
            Spatial Reconstruction Failed
          </h3>
          <p className="text-xs text-neutral-400 max-w-lg mb-6 leading-relaxed">
            The canonical reconstruction pipeline did not produce calibrated 3D geometry for this world.
            {(recon as { note?: string } | undefined)?.note
              ? ` Diagnostic reason: ${(recon as { note?: string }).note}`
              : " Multi-view camera registration or feature matching failed to converge."}
          </p>
          {onOpenRoomConstruction && (
            <button
              type="button"
              onClick={onOpenRoomConstruction}
              className="flex items-center gap-2 px-4 py-2 rounded-md bg-red-500 hover:bg-red-400 text-black font-semibold text-xs transition-all cursor-pointer shadow-lg shadow-red-500/20"
            >
              <Workflow className="w-4 h-4" />
              <span>Open Construction Pipeline</span>
            </button>
          )}
        </div>
      )}

      {/* The photos could not be placed in one 3D space: say so, with the engine's own reasons and advice */}
      {!isLoading && !isReconFailed && unplaced && captureQuality && (
        <div
          className="absolute inset-0 z-10 flex flex-col items-center justify-center p-8 text-center bg-[var(--bg-base)]/90 pointer-events-auto"
          role="alert"
          data-testid="capture-insufficient"
        >
          <div className="w-14 h-14 rounded-full flex items-center justify-center bg-amber-950/40 border border-amber-500/40 text-amber-400 mb-4 shadow-inner">
            <AlertTriangle className="w-7 h-7 text-amber-400" />
          </div>
          <h3 className="text-base font-semibold text-white mb-2 tracking-wide">
            {captureTitle(captureQuality)}
          </h3>
          {captureQuality.message && (
            <p className="text-xs text-neutral-300 max-w-lg mb-3 leading-relaxed" data-testid="capture-insufficient-message">
              {captureQuality.message}
            </p>
          )}
          {captureQuality.advice.length > 0 && (
            <ul className="max-w-lg space-y-1 text-left text-xs text-neutral-300 list-disc pl-5" data-testid="capture-insufficient-advice" aria-label="What to do next">
              {captureQuality.advice.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Explicit Empty State when no world data exists (honest, never fake) */}
      {!isLoading && !isReconFailed && !unplaced && (hasNoWorldData || (!world && stats.points === 0)) && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-8 text-center bg-[var(--bg-base)]/90 pointer-events-auto">
          <div className="w-14 h-14 rounded-full flex items-center justify-center bg-neutral-900 border border-neutral-800 text-neutral-400 mb-4 shadow-inner">
            <Box className="w-7 h-7 text-[var(--accent)]" />
          </div>
          <h3 className="text-base font-semibold text-white mb-2 tracking-wide">
            No 3D Reconstructed World Available
          </h3>
          <p className="text-xs text-neutral-400 max-w-lg mb-6 leading-relaxed">
            Reality Engine never synthesizes fake world geometry. A 3D world is generated via the
            canonical spatial pipeline: <strong className="text-neutral-200">Capture Session</strong> → <strong className="text-neutral-200">SfM Perception</strong> → <strong className="text-neutral-200">Interior Compilation</strong> → <strong className="text-neutral-200">Reality Studio</strong>.
          </p>
          {onOpenRoomConstruction && (
            <div className="flex flex-col sm:flex-row items-center gap-3">
              <button
                type="button"
                onClick={onOpenRoomConstruction}
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-[var(--primary)] text-white font-semibold text-xs hover:bg-[var(--primary-hover)] transition-all cursor-pointer shadow-lg shadow-[var(--primary)]/20"
              >
                <Layers className="w-4 h-4" />
                <span>Construct Space / Compile IR</span>
              </button>
            </div>
          )}
        </div>
      )}

      </div>
    </div>
  );
}
