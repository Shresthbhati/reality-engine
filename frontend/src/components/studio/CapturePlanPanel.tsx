"use client";

import React, { useState } from "react";
import {
  Camera,
  ChevronDown,
  ChevronUp,
  Compass,
  Crosshair,
  Eye,
  EyeOff,
  Sparkles,
  HelpCircle,
} from "lucide-react";
import {
  type RecommendedPose,
  type CapturePlanSummary,
  formatCompassHeading,
  formatPitchAngle,
  calculateCoverageImprovementPct,
  filterAndSortViewpoints,
  describeCameraPose,
} from "@/lib/studio/capture-plan-model";

export interface CapturePlanPanelProps {
  plan: CapturePlanSummary | null;
  selectedViewpointId: string | null;
  onSelectViewpoint: (viewpointId: string) => void;
  showCameraCones: boolean;
  onToggleCameraCones: (show: boolean) => void;
  className?: string;
}

export function CapturePlanPanel({
  plan,
  selectedViewpointId,
  onSelectViewpoint,
  showCameraCones,
  onToggleCameraCones,
  className = "",
}: CapturePlanPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!plan || plan.recommended_viewpoints.length === 0) {
    return null;
  }

  const coveragePct = calculateCoverageImprovementPct(
    plan.initial_uncertainty,
    plan.uncertainty_reduction_ratio
  );
  const sortedViewpoints = filterAndSortViewpoints(plan.recommended_viewpoints, 0.05, 8);

  return (
    <div
      className={`bg-zinc-900/90 backdrop-blur-md border border-zinc-700/80 rounded-xl shadow-2xl overflow-hidden transition-all duration-200 text-zinc-200 text-xs ${className}`}
      data-testid="capture-plan-panel"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2.5 bg-zinc-800/60 border-b border-zinc-700/50 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <div className="p-1 rounded-md bg-indigo-500/20 text-indigo-400">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-zinc-100 flex items-center gap-1.5">
              Capture Assistant
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                +{coveragePct}% Gain
              </span>
            </div>
            <div className="text-[10px] text-zinc-400">
              {plan.unobserved_patches_count} unobserved surface patches detected
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1 text-zinc-400 hover:text-zinc-200">
          <button
            type="button"
            className="p-1 hover:bg-zinc-700/50 rounded transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              onToggleCameraCones(!showCameraCones);
            }}
            title={showCameraCones ? "Hide camera cones" : "Show camera cones"}
          >
            {showCameraCones ? <Eye className="w-3.5 h-3.5 text-indigo-400" /> : <EyeOff className="w-3.5 h-3.5 text-zinc-500" />}
          </button>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>

      {/* Body */}
      {isExpanded && (
        <div className="p-2 space-y-2 max-h-[380px] overflow-y-auto">
          <div className="text-[11px] text-zinc-300 px-1 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-amber-400 shrink-0" />
            <span>Recommended capture positions to eliminate reconstruction gaps:</span>
          </div>

          <div className="space-y-1.5">
            {sortedViewpoints.map((vp) => {
              const isSelected = selectedViewpointId === vp.viewpoint_id;
              const compass = formatCompassHeading(vp.yaw_deg);
              const pitch = formatPitchAngle(vp.pitch_deg);

              return (
                <div
                  key={vp.viewpoint_id}
                  onClick={() => onSelectViewpoint(vp.viewpoint_id)}
                  className={`p-2 rounded-lg border transition-all cursor-pointer ${
                    isSelected
                      ? "bg-indigo-950/40 border-indigo-500/80 shadow-md ring-1 ring-indigo-500/40"
                      : "bg-zinc-800/40 border-zinc-700/40 hover:bg-zinc-800/70 hover:border-zinc-600/60"
                  }`}
                  data-testid={`viewpoint-item-${vp.viewpoint_id}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-300 font-bold text-[10px] flex items-center justify-center">
                        {vp.priority_rank}
                      </span>
                      <span className="font-medium text-zinc-200">
                        Viewpoint {vp.priority_rank}
                      </span>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-700/50 text-zinc-300">
                      IG: {(vp.information_gain * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="text-[11px] text-zinc-300 mb-1 leading-snug">
                    {vp.instruction}
                  </div>

                  <div className="flex items-center gap-2 text-[10px] text-zinc-400">
                    <span className="flex items-center gap-0.5">
                      <Crosshair className="w-3 h-3 text-zinc-500" />
                      ({vp.position[0].toFixed(1)}, {vp.position[1].toFixed(1)}, {vp.position[2].toFixed(1)}m)
                    </span>
                    <span className="flex items-center gap-0.5">
                      <Compass className="w-3 h-3 text-zinc-500" />
                      {compass} ({pitch})
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
