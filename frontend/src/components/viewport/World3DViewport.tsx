"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Maximize2,
  Eye,
  Camera,
  Box,
  AlertTriangle,
  Compass,
  Sparkles,
  Ruler,
  Grid,
  X,
  Layers,
  Workflow,
} from "lucide-react";
import {
  WorldSceneController,
  type ViewportLayers,
  type ViewPreset,
  type MeasurementResult,
} from "@/lib/viewport/three-scene";
import type {
  CamerasPayload,
  WorldIR,
} from "@/types/worldir";

interface World3DViewportProps {
  world: WorldIR | null;
  points: Float32Array | null;
  cameras: CamerasPayload | null;
  mesh?: { positions: Float32Array; indices: Uint32Array } | null;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  highlightedEvidenceId?: string | null;
  onSelectEvidence?: (id: string | null) => void;
  queryMatchingIds?: Set<string> | null;
  isLoading?: boolean;
  loadingMessage?: string;
  hasNoWorldData?: boolean;
  onTriggerSampleWorld?: () => void;
  onMeasurementChange?: (measurement: MeasurementResult | null) => void;
}

export default function World3DViewport({
  world,
  points,
  cameras,
  mesh = null,
  selectedEntityId,
  onSelectEntity,
  highlightedEvidenceId,
  queryMatchingIds = null,
  isLoading = false,
  loadingMessage = "Initializing 3D spatial viewport...",
  hasNoWorldData = false,
  onTriggerSampleWorld,
  onMeasurementChange,
}: World3DViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<WorldSceneController | null>(null);

  const [layers, setLayers] = useState<ViewportLayers>({
    points: true,
    entities: true,
    cameras: true,
    mesh: true,
    oversized: false,
    uncertainty: false,
    walls: true,
    topology: true,
  });

  const [gridVisible, setGridVisible] = useState(true);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [measurement, setMeasurement] = useState<MeasurementResult | null>(null);
  const [stats, setStats] = useState({ points: 0, cameras: 0, entities: 0 });
  const [isolatedSpaceId, setIsolatedSpaceId] = useState<string | null>(null);
  const [activeLevel, setActiveLevel] = useState<number | null>(null);

  // Initialize Three.js Controller
  useEffect(() => {
    if (!containerRef.current) return;

    const controller = new WorldSceneController(containerRef.current);
    controllerRef.current = controller;

    controller.setCallbacks({
      onSelect: (id) => onSelectEntity(id),
      onMeasure: (res) => {
        setMeasurement(res);
        onMeasurementChange?.(res);
      },
    });

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, [onSelectEntity, onMeasurementChange]);

  // Sync data into scene
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;

    if (world) {
      controller.loadWorldData(world, points, cameras, mesh);
      setStats(controller.getStatistics());
    }
  }, [world, points, cameras, mesh]);

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
      const customEvent = e as CustomEvent<{ entityId: string; type: string; confidence?: number }>;
      if (customEvent.detail) {
        controllerRef.current?.setCorrectionPreview(
          customEvent.detail.entityId,
          customEvent.detail.type,
          customEvent.detail.confidence
        );
      }
    };
    const handleClearCorrectionPreview = () => {
      controllerRef.current?.clearCorrectionPreview();
    };

    window.addEventListener("frame-entity", handleFlyToEntity as EventListener);
    window.addEventListener("frame-camera", handleFlyToCamera as EventListener);
    window.addEventListener("frame-position", handleFlyToPosition as EventListener);
    window.addEventListener("isolate-space", handleIsolateSpace as EventListener);
    window.addEventListener("filter-level", handleFilterLevel as EventListener);
    window.addEventListener("preview-correction", handlePreviewCorrection as EventListener);
    window.addEventListener("clear-correction-preview", handleClearCorrectionPreview as EventListener);

    return () => {
      window.removeEventListener("frame-entity", handleFlyToEntity as EventListener);
      window.removeEventListener("frame-camera", handleFlyToCamera as EventListener);
      window.removeEventListener("frame-position", handleFlyToPosition as EventListener);
      window.removeEventListener("isolate-space", handleIsolateSpace as EventListener);
      window.removeEventListener("filter-level", handleFilterLevel as EventListener);
      window.removeEventListener("preview-correction", handlePreviewCorrection as EventListener);
      window.removeEventListener("clear-correction-preview", handleClearCorrectionPreview as EventListener);
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
        setMeasurement(null);
        onMeasurementChange?.(null);
      }
      return next;
    });
  }, [onMeasurementChange]);

  const handleClearMeasurement = useCallback(() => {
    controllerRef.current?.clearMeasurement();
    setMeasurement(null);
    onMeasurementChange?.(null);
  }, [onMeasurementChange]);

  const handlePreset = useCallback((preset: ViewPreset) => {
    controllerRef.current?.setViewPreset(preset);
  }, []);

  const handleFrameAll = useCallback(() => {
    if (selectedEntityId) {
      controllerRef.current?.flyToEntity(selectedEntityId);
    } else {
      controllerRef.current?.frameAll();
    }
  }, [selectedEntityId]);

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid intercepting input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
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
  }, [handleFrameAll, toggleLayer, toggleGrid, toggleMeasure, isMeasuring, isolatedSpaceId, onSelectEntity]);

  return (
    <div className="relative w-full h-full overflow-hidden select-none bg-[#08090b]">
      {/* 3D Canvas Mount Point */}
      <div
        ref={containerRef}
        className={`w-full h-full ${
          isMeasuring ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"
        }`}
      />

      {/* Floating Viewport HUD: Top Left Stats */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 pointer-events-none">
        <div
          className="flex items-center gap-3 px-3 py-1.5 rounded-md backdrop-blur-md text-xs font-mono-num"
          style={{
            background: "rgba(14, 16, 19, 0.78)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
          }}
        >
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#00e5ff] animate-pulse" />
            <span className="font-semibold text-white tracking-wide">3D WORLDIR</span>
          </div>
          <span className="text-neutral-600">|</span>
          <span>{stats.points.toLocaleString()} pts</span>
          <span>{stats.entities} entities</span>
          <span>{stats.cameras} cameras</span>
        </div>

        {queryMatchingIds !== null && (
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md backdrop-blur-md text-xs font-mono bg-[#00e5ff]/15 border border-[#00e5ff]/40 text-[#00e5ff] shadow-md">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00e5ff] animate-ping" />
            <span>Query: {queryMatchingIds.size} matching</span>
          </div>
        )}

        {isolatedSpaceId && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md backdrop-blur-md text-xs font-mono bg-[#3b82f6]/20 border border-[#3b82f6] text-[#3b82f6] shadow-lg pointer-events-auto">
            <span>Isolated Space: {isolatedSpaceId}</span>
            <button
              type="button"
              onClick={() => {
                controllerRef.current?.isolateSpace(null);
                setIsolatedSpaceId(null);
              }}
              className="text-white hover:text-red-300 ml-1 underline cursor-pointer text-[10px]"
            >
              Exit [Esc]
            </button>
          </div>
        )}

        {activeLevel !== null && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md backdrop-blur-md text-xs font-mono bg-[#38bdf8]/20 border border-[#38bdf8] text-[#38bdf8] shadow-lg pointer-events-auto">
            <span>Level {activeLevel} Filter</span>
            <button
              type="button"
              onClick={() => {
                controllerRef.current?.setLevelFilter(null);
                setActiveLevel(null);
              }}
              className="text-white hover:text-red-300 ml-1 underline cursor-pointer text-[10px]"
            >
              Reset
            </button>
          </div>
        )}

        {isMeasuring && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md backdrop-blur-md text-xs font-mono bg-[#00e5ff]/20 border border-[#00e5ff] text-[#00e5ff] shadow-lg pointer-events-auto">
            <Ruler className="w-3.5 h-3.5" />
            <span>Measuring 3D Distance (click 2 points)</span>
            <button
              type="button"
              onClick={toggleMeasure}
              className="text-neutral-300 hover:text-white ml-1 cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>

      {/* Interactive Measurement HUD Card */}
      {isMeasuring && measurement && measurement.distance > 0 && (
        <div
          className="absolute top-14 left-3 z-10 p-3 rounded-lg backdrop-blur-md text-xs font-mono border border-[#00e5ff]/40 bg-[#0e1013]/90 text-white space-y-1.5 shadow-2xl pointer-events-auto"
          style={{ minWidth: "220px" }}
        >
          <div className="flex items-center justify-between text-[#00e5ff] font-semibold text-[11px] uppercase tracking-wider">
            <span>Spatial Measurement</span>
            <button
              type="button"
              onClick={handleClearMeasurement}
              className="text-neutral-400 hover:text-white text-[10px] underline"
            >
              Reset
            </button>
          </div>
          <div className="text-lg font-bold text-white">
            {measurement.distance.toFixed(3)} <span className="text-xs text-[#00e5ff]">m</span>
          </div>
          <div className="text-[11px] text-neutral-400 space-y-0.5 pt-1 border-t border-neutral-800">
            <div>ΔX: {measurement.delta[0].toFixed(3)} m</div>
            <div>ΔY: {measurement.delta[1].toFixed(3)} m</div>
            <div>ΔZ: {measurement.delta[2].toFixed(3)} m</div>
          </div>
        </div>
      )}

      {/* Floating Viewport HUD: Top Right Toolbar Controls */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 pointer-events-auto">
        {/* Layer Visibility Toggles */}
        <div
          className="flex items-center gap-1 p-1 rounded-md backdrop-blur-md"
          style={{
            background: "rgba(14, 16, 19, 0.85)",
            border: "1px solid var(--border)",
          }}
        >
          <button
            type="button"
            onClick={() => toggleLayer("points")}
            title="Toggle Points [1]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.points ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Pts</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("entities")}
            title="Toggle Entities [2]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.entities ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Box className="w-3.5 h-3.5" />
            <span>Ent</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("walls")}
            title="Toggle Walls [W] (Peel back to see inside rooms)"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.walls ? "text-[#f59e0b] bg-[rgba(245,158,11,0.15)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Walls</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("topology")}
            title="Toggle 3D Spatial Topology Connections [T]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.topology ? "text-[#00e5ff] bg-[rgba(0,229,255,0.15)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Workflow className="w-3.5 h-3.5" />
            <span>Topo</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("cameras")}
            title="Toggle Cameras [3]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.cameras ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Cam</span>
          </button>

          <button
            type="button"
            onClick={toggleGrid}
            title="Toggle ENU Ground Grid [G]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              gridVisible ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Grid className="w-3.5 h-3.5" />
            <span>Grid</span>
          </button>

          <button
            type="button"
            onClick={toggleMeasure}
            title="Interactive 3D Measurement [M]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              isMeasuring ? "text-[#00e5ff] bg-[rgba(0,229,255,0.2)] border border-[#00e5ff]/50" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Ruler className="w-3.5 h-3.5" />
            <span>Measure</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("oversized")}
            title="Toggle Oversized Depth Noise [O]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.oversized ? "text-[#f5a623] bg-[rgba(245,166,35,0.15)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Noise</span>
          </button>

          <button
            type="button"
            onClick={() => toggleLayer("uncertainty")}
            title="Toggle Uncertainty Heatmap [U]"
            className={`flex items-center gap-1 px-2 h-7 rounded text-xs font-medium transition-colors ${
              layers.uncertainty ? "text-[#2ecc71] bg-[rgba(46,204,113,0.15)]" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Uncert</span>
          </button>
        </div>

        {/* Camera Views & Framing */}
        <div
          className="flex items-center gap-1 p-1 rounded-md backdrop-blur-md"
          style={{
            background: "rgba(14, 16, 19, 0.85)",
            border: "1px solid var(--border)",
          }}
        >
          <button
            type="button"
            onClick={() => handlePreset("isometric")}
            title="Isometric View"
            className="p-1.5 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            <Compass className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => handlePreset("top")}
            title="Top-down View"
            className="px-1.5 h-7 rounded text-xs font-mono text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            TOP
          </button>
          <button
            type="button"
            onClick={() => handlePreset("front")}
            title="Front View"
            className="px-1.5 h-7 rounded text-xs font-mono text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            FRNT
          </button>
          <button
            type="button"
            onClick={handleFrameAll}
            title="Frame Selection / All [F]"
            className="flex items-center gap-1 px-2 h-7 rounded text-xs font-medium text-[#00e5ff] hover:bg-[rgba(0,229,255,0.12)] transition-colors"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span>[F]</span>
          </button>
        </div>
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#08090b]/80 backdrop-blur-sm">
          <div className="w-10 h-10 border-2 border-[#00e5ff] border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-sm font-medium text-white">{loadingMessage}</p>
          <span className="text-xs text-neutral-500 mt-1 font-mono">Consuming canonical WorldIR stream</span>
        </div>
      )}

      {/* Explicit Empty State when no world data exists (honest, never fake) */}
      {!isLoading && (hasNoWorldData || (!world && stats.points === 0)) && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-8 text-center bg-[#08090b]/90 pointer-events-auto">
          <div className="w-14 h-14 rounded-full flex items-center justify-center bg-neutral-900 border border-neutral-800 text-neutral-400 mb-4">
            <Box className="w-7 h-7" />
          </div>
          <h3 className="text-base font-semibold text-white mb-1">
            No 3D Reconstructed World Available
          </h3>
          <p className="text-xs text-neutral-400 max-w-md mb-5 leading-relaxed">
            This World does not have a compiled 3D representation yet. Reality Engine never synthesizes fake world geometry. Reconstruct attached capture sessions or open an authentic dataset.
          </p>
          {onTriggerSampleWorld && (
            <button
              type="button"
              onClick={onTriggerSampleWorld}
              className="px-4 py-2 rounded-md text-xs font-semibold bg-[#00e5ff] text-black hover:bg-[#33ebff] transition-colors cursor-pointer"
            >
              Open Verified Dataset World
            </button>
          )}
        </div>
      )}

      {/* Bottom overlay: View navigation hint */}
      <div className="absolute bottom-3 left-3 z-10 pointer-events-none text-[11px] font-mono text-neutral-500">
        Left-click: Orbit · Right-click: Pan · Scroll: Zoom · Click entity to inspect · [M] Measure · [G] Grid
      </div>
    </div>
  );
}
