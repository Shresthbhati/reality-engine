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
      className={`bg-surface/90 backdrop-blur-md border border-line/80 rounded-xl shadow-2xl overflow-hidden transition-all duration-200 text-fg text-xs ${className}`}
      data-testid="capture-plan-panel"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2.5 bg-elevated/60 border-b border-line/50 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <div className="p-1 rounded-md bg-accent-subtle text-accent">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-fg flex items-center gap-1.5">
              Capture Assistant
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-success-subtle text-success border border-success/30">
                +{coveragePct}% Gain
              </span>
            </div>
            <div className="text-[10px] text-fg-muted">
              {plan.unobserved_patches_count} unobserved surface patches detected
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1 text-fg-muted hover:text-fg">
          <button
            type="button"
            className="p-1 hover:bg-line/50 rounded transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              onToggleCameraCones(!showCameraCones);
            }}
            title={showCameraCones ? "Hide camera cones" : "Show camera cones"}
          >
            {showCameraCones ? <Eye className="w-3.5 h-3.5 text-accent" /> : <EyeOff className="w-3.5 h-3.5 text-fg-subtle" />}
          </button>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>

      {/* Body */}
      {isExpanded && (
        <div className="p-2 space-y-2 max-h-[380px] overflow-y-auto">
          <div className="text-[11px] text-fg-muted px-1 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-warning shrink-0" />
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
                      ? "bg-accent/40 border-accent/80 shadow-md ring-1 ring-accent/40"
                      : "bg-elevated/40 border-line/40 hover:bg-elevated/70 hover:border-line-strong/60"
                  }`}
                  data-testid={`viewpoint-item-${vp.viewpoint_id}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span className="w-4 h-4 rounded-full bg-accent-subtle text-accent font-bold text-[10px] flex items-center justify-center">
                        {vp.priority_rank}
                      </span>
                      <span className="font-medium text-fg">
                        Viewpoint {vp.priority_rank}
                      </span>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-line/50 text-fg-muted">
                      IG: {(vp.information_gain * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="text-[11px] text-fg-muted mb-1 leading-snug">
                    {vp.instruction}
                  </div>

                  <div className="flex items-center gap-2 text-[10px] text-fg-muted">
                    <span className="flex items-center gap-0.5">
                      <Crosshair className="w-3 h-3 text-fg-subtle" />
                      ({vp.position[0].toFixed(1)}, {vp.position[1].toFixed(1)}, {vp.position[2].toFixed(1)}m)
                    </span>
                    <span className="flex items-center gap-0.5">
                      <Compass className="w-3 h-3 text-fg-subtle" />
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
