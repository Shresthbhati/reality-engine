"use client";

import React, { useState } from "react";
import {
  Globe,
  Compass,
  Building,
  ShieldCheck,
  ShieldAlert,
  Sun,
  Layers,
  ChevronDown,
  ChevronUp,
  Sparkles,
  MapPin,
  Maximize2,
} from "lucide-react";
import {
  type SpatialIntelligenceSummary,
  formatCoordinates,
  formatStyleDisplay,
  formatPlausibilityStatus,
  isGeoreferenced,
} from "@/lib/studio/spatial-intelligence-model";

export interface SpatialIntelligencePanelProps {
  summary: SpatialIntelligenceSummary | null;
  className?: string;
}

export function SpatialIntelligencePanel({
  summary,
  className = "",
}: SpatialIntelligencePanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!summary) {
    return null;
  }

  const georeferenced = isGeoreferenced(summary);
  const plausibilityBadge = formatPlausibilityStatus(summary.physical_plausibility);
  const completenessPct = summary.scene_completeness?.overall_completeness_pct ?? 0;

  return (
    <div
      className={`bg-zinc-900/90 backdrop-blur-md border border-zinc-700/80 rounded-xl shadow-2xl overflow-hidden transition-all duration-200 text-zinc-200 text-xs ${className}`}
      data-testid="spatial-intelligence-panel"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2.5 bg-zinc-800/60 border-b border-zinc-700/50 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <div className="p-1 rounded-md bg-emerald-500/20 text-emerald-400">
            <Globe className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-zinc-100 flex items-center gap-1.5">
              Spatial Intelligence
              {georeferenced && (
                <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  ENU Georeferenced
                </span>
              )}
            </div>
            <div className="text-[10px] text-zinc-400">
              {summary.coordinate_frame} Frame • {summary.transforms_count} Transforms
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 text-zinc-400">
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>

      {isExpanded && (
        <div className="p-3 space-y-3">
          {/* Georeference Origin Card */}
          <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-lg p-2.5">
            <div className="flex items-center gap-1.5 text-zinc-300 font-medium mb-1">
              <MapPin className="w-3.5 h-3.5 text-emerald-400" />
              <span>Geodetic Origin</span>
            </div>
            <div className="text-[11px] text-zinc-300 font-mono">
              {formatCoordinates(summary.georeference?.origin)}
            </div>
            {summary.georeference?.diagnostics && (
              <div className="text-[10px] text-zinc-400 mt-1 flex gap-3">
                <span>H-RMSE: {summary.georeference.diagnostics.rmse_horizontal_m.toFixed(2)}m</span>
                <span>V-RMSE: {summary.georeference.diagnostics.rmse_vertical_m.toFixed(2)}m</span>
                <span>Scale: {summary.georeference.diagnostics.scale_factor.toFixed(4)}x</span>
              </div>
            )}
          </div>

          {/* Architectural Style & Envelope */}
          <div className="grid grid-cols-2 gap-2">
            {/* Style */}
            <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-zinc-300 font-medium mb-1">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Design Style</span>
              </div>
              <div className="text-[11px] font-medium text-zinc-200">
                {formatStyleDisplay(summary.design_style?.primary_style)}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">
                Ornamentation: {summary.design_style?.ornamentation || "Standard"}
              </div>
            </div>

            {/* Envelope */}
            <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-zinc-300 font-medium mb-1">
                <Building className="w-3.5 h-3.5 text-cyan-400" />
                <span>Envelope</span>
              </div>
              <div className="text-[11px] font-medium text-zinc-200">
                {summary.building_envelope?.total_storeys || 1} Storey(s)
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">
                GFA: {summary.building_envelope?.gross_floor_area_m2 || 0} m²
              </div>
            </div>
          </div>

          {/* Physical Plausibility & Completeness */}
          <div className="grid grid-cols-2 gap-2">
            {/* Plausibility */}
            <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-zinc-300 font-medium mb-1">
                {plausibilityBadge.variant === "success" ? (
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                )}
                <span>Plausibility</span>
              </div>
              <div
                className={`text-[11px] font-medium ${
                  plausibilityBadge.variant === "success"
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}
              >
                {plausibilityBadge.label}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">
                Collisions: {summary.physical_plausibility?.collisions_count || 0}
              </div>
            </div>

            {/* Completeness */}
            <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-zinc-300 font-medium mb-1">
                <Maximize2 className="w-3.5 h-3.5 text-indigo-400" />
                <span>Completeness</span>
              </div>
              <div className="text-[11px] font-medium text-zinc-200">
                {completenessPct.toFixed(1)}%
              </div>
              <div className="w-full bg-zinc-700/50 rounded-full h-1 mt-1.5 overflow-hidden">
                <div
                  className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, completenessPct))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Environmental Lighting & Shared Walls */}
          <div className="flex items-center justify-between px-2.5 py-1.5 bg-zinc-800/30 border border-zinc-700/30 rounded-lg text-[10px] text-zinc-400">
            <div className="flex items-center gap-1.5">
              <Sun className="w-3 h-3 text-amber-400" />
              <span>
                Light:{" "}
                {summary.lighting?.dominant_light_direction
                  ? `[${summary.lighting.dominant_light_direction.map((d) => d.toFixed(1)).join(", ")}]`
                  : "Omni"}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Layers className="w-3 h-3 text-blue-400" />
              <span>Partitions: {summary.shared_walls?.consolidated_wall_count || 0} consolidated</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
