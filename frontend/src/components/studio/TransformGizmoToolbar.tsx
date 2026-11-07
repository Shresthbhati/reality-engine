"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  Move,
  RotateCw,
  Maximize,
  Magnet,
  ShieldCheck,
  Undo2,
  Redo2,
  SlidersHorizontal,
} from "lucide-react";
import type { TransformGizmoMode, SnapConfig } from "@/lib/studio/transform-types";
import { getStudioHistory } from "@/lib/viewport/history-stack";

export interface TransformGizmoToolbarProps {
  mode: TransformGizmoMode;
  onModeChange: (mode: TransformGizmoMode) => void;
  snapConfig: SnapConfig;
  onSnapConfigChange: (config: Partial<SnapConfig>) => void;
  selectedEntityId: string | null;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  className?: string;
}

export function TransformGizmoToolbar({
  mode,
  onModeChange,
  snapConfig,
  onSnapConfigChange,
  selectedEntityId,
  onUndo,
  onRedo,
  canUndo: controlledCanUndo,
  canRedo: controlledCanRedo,
  className = "",
}: TransformGizmoToolbarProps) {
  const history = getStudioHistory();
  const [internalCanUndo, setInternalCanUndo] = useState(history.canUndo());
  const [internalCanRedo, setInternalCanRedo] = useState(history.canRedo());

  // Keep history buttons synced with HistoryStack events
  useEffect(() => {
    const unsubscribe = history.subscribe(() => {
      setInternalCanUndo(history.canUndo());
      setInternalCanRedo(history.canRedo());
    });
    return unsubscribe;
  }, [history]);

  const canUndo = controlledCanUndo ?? internalCanUndo;
  const canRedo = controlledCanRedo ?? internalRedoSupported(canUndo, internalCanRedo);

  function internalRedoSupported(u: boolean, r: boolean) {
    return r;
  }

  const handleUndo = useCallback(() => {
    if (onUndo) {
      onUndo();
    } else {
      history.undo();
    }
  }, [onUndo, history]);

  const handleRedo = useCallback(() => {
    if (onRedo) {
      onRedo();
    } else {
      history.redo();
    }
  }, [onRedo, history]);

  // Global hotkeys for gizmo modes and undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept when user is typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        handleRedo();
      } else if (selectedEntityId && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.key.toLowerCase() === "w") {
          onModeChange("translate");
        } else if (e.key.toLowerCase() === "e") {
          onModeChange("rotate");
        } else if (e.key.toLowerCase() === "r") {
          onModeChange("scale");
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleUndo, handleRedo, onModeChange, selectedEntityId]);

  if (!selectedEntityId) {
    return null;
  }

  return (
    <div
      className={`flex items-center gap-1 rounded-md px-2 py-1 shadow-md text-xs backdrop-blur-md select-none transition-all ${className}`}
      style={{
        background: "rgba(23, 23, 23, 0.85)",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        color: "var(--text-primary)",
      }}
      role="toolbar"
      aria-label="3D Transform Gizmo Controls"
    >
      {/* Mode Selectors */}
      <div className="flex items-center gap-0.5 border-r border-line pr-1.5 mr-0.5">
        <button
          type="button"
          onClick={() => onModeChange("translate")}
          title="Translate / Move (W)"
          aria-label="Translate Mode"
          className={`flex h-7 items-center gap-1.5 px-2 rounded font-medium transition-colors cursor-pointer ${
            mode === "translate"
              ? "bg-info/80 text-fg font-semibold"
              : "text-fg-muted hover:text-fg hover:bg-hover"
          }`}
        >
          <Move className="h-3.5 w-3.5" />
          <span>Move</span>
        </button>

        <button
          type="button"
          onClick={() => onModeChange("rotate")}
          title="Rotate (E)"
          aria-label="Rotate Mode"
          className={`flex h-7 items-center gap-1.5 px-2 rounded font-medium transition-colors cursor-pointer ${
            mode === "rotate"
              ? "bg-info/80 text-fg font-semibold"
              : "text-fg-muted hover:text-fg hover:bg-hover"
          }`}
        >
          <RotateCw className="h-3.5 w-3.5" />
          <span>Rotate</span>
        </button>

        <button
          type="button"
          onClick={() => onModeChange("scale")}
          title="Scale (R)"
          aria-label="Scale Mode"
          className={`flex h-7 items-center gap-1.5 px-2 rounded font-medium transition-colors cursor-pointer ${
            mode === "scale"
              ? "bg-info/80 text-fg font-semibold"
              : "text-fg-muted hover:text-fg hover:bg-hover"
          }`}
        >
          <Maximize className="h-3.5 w-3.5" />
          <span>Scale</span>
        </button>
      </div>

      {/* Boundary Snapping Toggle */}
      <div className="flex items-center gap-0.5 border-r border-line pr-1.5 mr-0.5">
        <button
          type="button"
          onClick={() => onSnapConfigChange({ enabled: !snapConfig.enabled })}
          title={snapConfig.enabled ? "Wall & Grid Snapping Active" : "Snapping Disabled"}
          aria-label="Toggle Snapping"
          aria-pressed={snapConfig.enabled}
          className={`flex h-7 items-center gap-1 px-2 rounded transition-colors cursor-pointer ${
            snapConfig.enabled
              ? "bg-success/70 text-fg font-semibold"
              : "text-fg-muted hover:text-fg hover:bg-hover"
          }`}
        >
          <Magnet className="h-3.5 w-3.5" />
          <span>Snap</span>
        </button>

        <button
          type="button"
          onClick={() =>
            onSnapConfigChange({ preventWallPenetration: !snapConfig.preventWallPenetration })
          }
          title={
            snapConfig.preventWallPenetration
              ? "Room Boundary Collision Active"
              : "Collision Prevention Disabled"
          }
          aria-label="Toggle Collision Prevention"
          aria-pressed={snapConfig.preventWallPenetration}
          className={`flex h-7 items-center gap-1 px-2 rounded transition-colors cursor-pointer ${
            snapConfig.preventWallPenetration
              ? "bg-accent/70 text-fg font-semibold"
              : "text-fg-muted hover:text-fg hover:bg-hover"
          }`}
        >
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>Collision</span>
        </button>
      </div>

      {/* Undo / Redo Actions */}
      <div className="flex items-center gap-0.5">
        <button
          type="button"
          onClick={handleUndo}
          disabled={!canUndo}
          title="Undo Transform (Ctrl+Z)"
          aria-label="Undo Transform"
          className="flex h-7 w-7 items-center justify-center rounded text-fg-muted hover:text-fg hover:bg-hover disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-colors"
        >
          <Undo2 className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          onClick={handleRedo}
          disabled={!canRedo}
          title="Redo Transform (Ctrl+Y)"
          aria-label="Redo Transform"
          className="flex h-7 w-7 items-center justify-center rounded text-fg-muted hover:text-fg hover:bg-hover disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-colors"
        >
          <Redo2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
export default TransformGizmoToolbar;
