"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Maximize2,
  Eye,
  Camera,
  Layers,
  Box,
  AlertTriangle,
  Compass,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import {
  WorldSceneController,
  type ViewportLayers,
  type ViewPreset,
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
  isLoading?: boolean;
  loadingMessage?: string;
  hasNoWorldData?: boolean;
  onTriggerSampleWorld?: () => void;
}

export default function World3DViewport({
  world,
  points,
  cameras,
  mesh = null,
  selectedEntityId,
  onSelectEntity,
  isLoading = false,
  loadingMessage = "Initializing 3D spatial viewport...",
  hasNoWorldData = false,
  onTriggerSampleWorld,
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
  });

  const [stats, setStats] = useState({ points: 0, cameras: 0, entities: 0 });

  // Initialize Three.js Controller
  useEffect(() => {
    if (!containerRef.current) return;

    const controller = new WorldSceneController(containerRef.current);
    controllerRef.current = controller;

    controller.setCallbacks({
      onSelect: (id) => onSelectEntity(id),
    });

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, [onSelectEntity]);

  // Sync data into scene
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;

    if (world) {
      controller.loadWorldData(world, points, cameras, mesh);
      setStats(controller.getStatistics());
    }
  }, [world, points, cameras, mesh]);

  // Sync external selection (e.g. from nav panel or inspector)
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    controller.selectEntity(selectedEntityId, true);
  }, [selectedEntityId]);

  // Toggle Layer Helper
  const toggleLayer = useCallback((key: keyof ViewportLayers) => {
    setLayers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      controllerRef.current?.setLayerVisibility(next);
      return next;
    });
  }, []);

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
      } else if (e.key === "o" || e.key === "O") {
        toggleLayer("oversized");
      } else if (e.key === "u" || e.key === "U") {
        toggleLayer("uncertainty");
      } else if (e.key === "Escape") {
        onSelectEntity(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleFrameAll, toggleLayer, onSelectEntity]);

  return (
    <div className="relative w-full h-full overflow-hidden select-none bg-[#08090b]">
      {/* 3D Canvas Mount Point */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

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
      </div>

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
            This World does not have a compiled 3D representation yet. Reality Engine never synthesizes fake world geometry. Compile a capture session or open an existing verified dataset.
          </p>
          {onTriggerSampleWorld && (
            <button
              type="button"
              onClick={onTriggerSampleWorld}
              className="px-4 py-2 rounded-md text-xs font-semibold bg-[#00e5ff] text-black hover:bg-[#33ebff] transition-colors"
            >
              Open Verified Room Capture World
            </button>
          )}
        </div>
      )}

      {/* Bottom overlay: View navigation hint */}
      <div className="absolute bottom-3 left-3 z-10 pointer-events-none text-[11px] font-mono text-neutral-500">
        Left-click: Orbit · Right-click: Pan · Scroll: Zoom · Click entity to inspect
      </div>
    </div>
  );
}
