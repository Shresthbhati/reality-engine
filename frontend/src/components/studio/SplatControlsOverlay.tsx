"use client";

import React, { useRef, useState } from "react";
import {
  Sparkles,
  Eye,
  EyeOff,
  Upload,
  RefreshCw,
  Layers,
  Box,
  Image as ImageIcon,
  X,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { SplatMetadata } from "@/lib/api/splats";
import { formatSplatCount, formatByteSize, type HybridRenderMode } from "@/lib/viewport/splat-model";

interface SplatControlsOverlayProps {
  metadata: SplatMetadata | null;
  loadedCount: number;
  isLoaded: boolean;
  isLoading: boolean;
  visibility: boolean;
  opacity: number;
  scaleMultiplier: number;
  hybridMode: HybridRenderMode;
  onToggleVisibility: (visible: boolean) => void;
  onSetOpacity: (opacity: number) => void;
  onSetScaleMultiplier: (multiplier: number) => void;
  onSetHybridMode: (mode: HybridRenderMode) => void;
  onGenerateSplats: () => Promise<void>;
  onUploadSplats: (file: File) => Promise<void>;
  onClose?: () => void;
}

export default function SplatControlsOverlay({
  metadata,
  loadedCount,
  isLoaded,
  isLoading,
  visibility,
  opacity,
  scaleMultiplier,
  hybridMode,
  onToggleVisibility,
  onSetOpacity,
  onSetScaleMultiplier,
  onSetHybridMode,
  onGenerateSplats,
  onUploadSplats,
  onClose,
}: SplatControlsOverlayProps) {
  const [minimized, setMinimized] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const displayCount = loadedCount > 0 ? loadedCount : metadata?.splat_count ?? 0;
  const displayBytes = metadata?.byte_size ?? displayCount * 32;

  const handleGenerate = async () => {
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      await onGenerateSplats();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to generate Gaussian splats");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setErrorMsg(null);
    try {
      await onUploadSplats(file);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to upload Gaussian splats");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div
      data-testid="splat-controls-overlay"
      className="absolute top-16 right-4 z-20 flex flex-col w-80 rounded-lg shadow-xl backdrop-blur-md bg-[var(--bg-surface)] border border-[var(--border)] text-[var(--text-primary)] text-xs overflow-hidden transition-all duration-200"
    >
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-[var(--bg-elevated)] border-b border-[var(--border-subtle)]">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-[var(--accent)]" />
          <span className="font-semibold tracking-wide text-[var(--text-primary)]">3D Gaussian Splats</span>
          {displayCount > 0 && (
            <span
              data-testid="splat-count-badge"
              className="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent-border)]"
            >
              {formatSplatCount(displayCount)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setMinimized(!minimized)}
            title={minimized ? "Expand" : "Collapse"}
            className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
          >
            {minimized ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              title="Close"
              className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {!minimized && (
        <div className="p-3 flex flex-col gap-3">
          {/* Metadata pill */}
          <div className="flex items-center justify-between text-[11px] text-[var(--text-secondary)] px-2 py-1.5 rounded bg-[var(--bg-base)] border border-[var(--border-subtle)]">
            <span>{isLoaded ? "Rasterizer active" : "Memory footprint"}</span>
            <span className="font-mono text-[var(--text-primary)]">{formatByteSize(displayBytes)}</span>
          </div>

          {/* Mode Switcher */}
          <div className="flex flex-col gap-1.5">
            <span className="text-[11px] font-medium text-[var(--text-secondary)]">Viewport Representation</span>
            <div className="grid grid-cols-3 gap-1 p-0.5 rounded bg-[var(--bg-base)] border border-[var(--border-subtle)]">
              <button
                type="button"
                data-testid="splat-mode-hybrid"
                onClick={() => onSetHybridMode("hybrid")}
                className={`flex items-center justify-center gap-1 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer ${
                  hybridMode === "hybrid"
                    ? "bg-[var(--primary)] text-[var(--on-primary)] shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                <Layers className="w-3 h-3" />
                <span>Hybrid</span>
              </button>

              <button
                type="button"
                data-testid="splat-mode-splats"
                onClick={() => onSetHybridMode("splats_only")}
                className={`flex items-center justify-center gap-1 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer ${
                  hybridMode === "splats_only"
                    ? "bg-[var(--primary)] text-[var(--on-primary)] shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                <ImageIcon className="w-3 h-3" />
                <span>Splats</span>
              </button>

              <button
                type="button"
                data-testid="splat-mode-cad"
                onClick={() => onSetHybridMode("cad_only")}
                className={`flex items-center justify-center gap-1 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer ${
                  hybridMode === "cad_only"
                    ? "bg-[var(--primary)] text-[var(--on-primary)] shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                <Box className="w-3 h-3" />
                <span>BIM CAD</span>
              </button>
            </div>
          </div>

          {/* Visibility and sliders */}
          <div className="flex flex-col gap-2 pt-1 border-t border-[var(--border-subtle)]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-[var(--text-secondary)]">Splat Layer</span>
              <button
                type="button"
                data-testid="toggle-splat-visibility"
                onClick={() => onToggleVisibility(!visibility)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium transition-colors cursor-pointer ${
                  visibility
                    ? "bg-[var(--bg-elevated)] text-[var(--accent)] border border-[var(--accent-border)]"
                    : "bg-[var(--bg-base)] text-[var(--text-tertiary)] border border-[var(--border-subtle)]"
                }`}
              >
                {visibility ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                <span>{visibility ? "Visible" : "Hidden"}</span>
              </button>
            </div>

            {/* Opacity Slider */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-[11px] text-[var(--text-secondary)]">
                <span>Opacity</span>
                <span className="font-mono text-[var(--text-primary)]">{Math.round(opacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => onSetOpacity(parseFloat(e.target.value))}
                className="w-full h-1 bg-[var(--bg-base)] rounded-lg appearance-none cursor-pointer accent-[var(--accent)]"
              />
            </div>

            {/* Scale Multiplier Slider */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-[11px] text-[var(--text-secondary)]">
                <span>Scale Multiplier</span>
                <span className="font-mono text-[var(--text-primary)]">{scaleMultiplier.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="3.0"
                step="0.1"
                value={scaleMultiplier}
                onChange={(e) => onSetScaleMultiplier(parseFloat(e.target.value))}
                className="w-full h-1 bg-[var(--bg-base)] rounded-lg appearance-none cursor-pointer accent-[var(--accent)]"
              />
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-subtle)]">
            {displayCount === 0 && metadata?.can_generate && (
              <button
                type="button"
                data-testid="generate-splats-btn"
                onClick={handleGenerate}
                disabled={isGenerating || isLoading}
                className="flex items-center justify-center gap-1.5 w-full py-1.5 rounded font-medium bg-[var(--primary)] text-[var(--on-primary)] hover:bg-[var(--primary-hover)] disabled:opacity-50 transition-colors cursor-pointer"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isGenerating ? "animate-spin" : ""}`} />
                <span>{isGenerating ? "Synthesizing Splats..." : "Generate from Point Cloud"}</span>
              </button>
            )}

            <div className="flex items-center gap-2">
              <input
                type="file"
                ref={fileInputRef}
                accept=".splat,.ply"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                data-testid="upload-splats-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center justify-center gap-1.5 flex-1 py-1 rounded font-medium bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] border border-[var(--border-subtle)] disabled:opacity-50 transition-colors cursor-pointer"
              >
                <Upload className={`w-3 h-3 ${isUploading ? "animate-pulse" : ""}`} />
                <span>{isUploading ? "Uploading..." : "Upload .splat / .ply"}</span>
              </button>
            </div>
          </div>

          {/* Error notice */}
          {errorMsg && (
            <div className="p-2 rounded bg-[var(--error-subtle)] border border-[var(--error)] text-[var(--error)] text-[11px]">
              {errorMsg}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
