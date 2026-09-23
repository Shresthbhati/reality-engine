"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import WorldNavPanel from "@/components/navigation/WorldNavPanel";
import World3DViewport from "@/components/viewport/World3DViewport";
import AdaptiveInspector from "@/components/inspector/AdaptiveInspector";
import BottomContextBar from "@/components/context/BottomContextBar";
import RoomBuilderPanel from "@/components/workspace/RoomBuilderPanel";
import { Plus } from "lucide-react";
import {
  useWorlds,
  useSessions,
  useEvidence,
  useWorldIR,
  useWorldPoints,
  useWorldCameras,
  useWorldReport,
  useWorldVersions,
} from "@/lib/api";
import { getActivity } from "@/lib/api/activity";
import type { ActivityEvent } from "@/lib/api/activity";
import { parsePly, parseMeshPly } from "@/lib/viewport/loaders";
import type { WorldIR, Entity, CamerasPayload } from "@/types/worldir";
import type { WorldRow, SessionRow, EvidenceRow } from "@/lib/types";
import { Loader2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface SpatialWorkstationProps {
  worldId: string;
}

export default function SpatialWorkstation({ worldId }: SpatialWorkstationProps) {
  const router = useRouter();
  
  // State
  const [navOpen, setNavOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [roomBuilderOpen, setRoomBuilderOpen] = useState(false);
  
  // API Hooks
  const { data: worlds = [] } = useWorlds();
  const { data: allSessions = [] } = useSessions();
  const { data: evidence = [] } = useEvidence();
  const { data: worldIR, isLoading: isIrLoading, error: irError, refetch: refetchIr } = useWorldIR(worldId);
  const { data: pointsBuffer } = useWorldPoints(worldId);
  const { data: camerasPayload } = useWorldCameras(worldId);
  const { data: report } = useWorldReport(worldId);
  const { data: versionsData } = useWorldVersions(worldId);
  const versions = versionsData ?? [];
  const [activityItems, setActivityItems] = useState<ActivityEvent[]>([]);
  useEffect(() => {
    let cancelled = false;
    getActivity().then((items) => {
      if (!cancelled) setActivityItems(items);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Derived state
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

// Filter sessions by worldId
  const sessions = useMemo(() => 
    (allSessions || []).filter(s => s.worldId === worldId), 
    [allSessions, worldId]
  );
  
  const selectedEntity = useMemo(() => {
    if (!worldIR || !selectedEntityId) return null;
    return worldIR.entities[selectedEntityId] || null;
  }, [worldIR, selectedEntityId]);

  // Actions
  const handleSelectWorld = useCallback((id: string) => {
    router.push(`/worlds/${id}`);
  }, [router]);

  const handleSelectEntity = useCallback((id: string | null) => {
    setSelectedEntityId(id);
    if (id !== null) {
      setInspectorOpen(true);
    } else {
      setInspectorOpen(false);
    }
  }, []);

  const handleFrameEntity = useCallback((id: string) => {
    // Custom event to trigger framing in the viewport, as typically standard in such 3D views
    window.dispatchEvent(new CustomEvent('frame-entity', { detail: { id } }));
  }, []);

  const handleClearSelection = useCallback(() => {
    handleSelectEntity(null);
  }, [handleSelectEntity]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (e.key === "[") {
        setNavOpen(prev => !prev);
      } else if (e.key === "Escape") {
        handleClearSelection();
      } else if (e.key.toLowerCase() === "f" && selectedEntityId) {
        handleFrameEntity(selectedEntityId);
      }
    };
    
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedEntityId, handleClearSelection, handleFrameEntity]);

  // Render states
  if (isIrLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[var(--bg-base)] text-[var(--text-primary)]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
          <p className="text-sm text-[var(--text-secondary)]">Loading World Representation...</p>
        </div>
      </div>
    );
  }

  if (irError) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[var(--bg-base)] text-[var(--text-primary)]">
        <div className="flex max-w-md flex-col items-center gap-4 text-center">
          <AlertCircle className="h-10 w-10 text-[var(--error)]" />
          <h2 className="text-lg font-semibold">Failed to load world</h2>
          <p className="text-sm text-[var(--text-secondary)]">{String(irError)}</p>
          <button 
            onClick={() => refetchIr()}
            className="mt-4 rounded-md bg-[var(--accent-subtle)] px-4 py-2 text-[var(--accent)] hover:bg-[var(--accent-border)] transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!worldIR) {
    return (
      <div className="relative flex h-screen w-full items-center justify-center bg-[var(--bg-base)] text-[var(--text-primary)]">
        <div className="max-w-md text-center flex flex-col items-center gap-4">
          <p className="text-lg text-[var(--text-secondary)]">
            No reconstruction available for this world yet. Run the reconstruction pipeline, or build a room directly.
          </p>
          <button
            type="button"
            onClick={() => setRoomBuilderOpen(true)}
            className="flex items-center gap-1.5 h-9 px-4 rounded-md text-sm font-medium transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create Room
          </button>
        </div>
        {roomBuilderOpen && (
          <RoomBuilderPanel
            worldId={worldId}
            onClose={() => setRoomBuilderOpen(false)}
            onCreated={refetchIr}
          />
        )}
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-[var(--bg-base)] text-[var(--text-primary)]">
      <div className="flex flex-1 overflow-hidden">
        {/* Left Nav */}
        <div 
          className={cn(
            "h-full flex-shrink-0 transition-all duration-300 ease-in-out border-r border-[var(--border)]",
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
              versions={versions}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              onFrameEntity={handleFrameEntity}
            />
          )}
        </div>
        
        {/* Center Viewport */}
        <div className="flex-1 relative h-full min-w-0 bg-black">
          <World3DViewport
            world={worldIR}
            points={points}
            cameras={cameras}
            mesh={mesh}
            selectedEntityId={selectedEntityId}
            onSelectEntity={handleSelectEntity}
          />

          <button
            type="button"
            onClick={() => setRoomBuilderOpen(true)}
            className="absolute top-4 left-4 z-10 flex items-center gap-1.5 h-8 px-3 rounded-md text-sm font-medium transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create Room
          </button>

          {roomBuilderOpen && (
            <RoomBuilderPanel
              worldId={worldId}
              onClose={() => setRoomBuilderOpen(false)}
              onCreated={refetchIr}
            />
          )}
        </div>
        
        {/* Right Inspector */}
        <div 
          className={cn(
            "h-full flex-shrink-0 transition-all duration-300 ease-in-out border-l border-[var(--border)]",
            inspectorOpen ? "w-[320px]" : "w-0 overflow-hidden border-l-0"
          )}
        >
          {inspectorOpen && (
            <AdaptiveInspector
              world={worldIR}
              selectedEntityId={selectedEntityId}
              onSelectEntity={handleSelectEntity}
              onFrameEntity={handleFrameEntity}
            />
          )}
        </div>
      </div>
      
      {/* Bottom Context Bar */}
      <div className="h-auto flex-shrink-0 border-t border-[var(--border)] z-10">
        <BottomContextBar
          world={worldIR}
          selectedEntity={selectedEntity}
          sessions={sessions}
          report={report ?? null}
          activityItems={activityItems}
          onFrameSelected={() => {
            if (selectedEntityId) handleFrameEntity(selectedEntityId);
          }}
          onClearSelection={handleClearSelection}
        />
      </div>
    </div>
  );
}
