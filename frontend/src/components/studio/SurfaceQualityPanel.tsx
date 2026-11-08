"use client";

import React, { useState } from "react";
import {
  Download,
  Eye,
  EyeOff,
  RefreshCw,
  Ruler,
  ShieldAlert,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  type BeamDeflectionReport,
  type FloorFlatnessReport,
  type HeatmapData,
  type SurfaceQualityReport,
  formatDeviationMm,
  formatFFNumber,
  getASTMClassificationMeta,
  getDeflectionCodeMeta,
} from "@/lib/studio/surface-quality-model";

export interface SurfaceQualityPanelProps {
  report: SurfaceQualityReport | null;
  heatmap: HeatmapData | null;
  isHeatmapVisible: boolean;
  onToggleHeatmap: (visible: boolean) => void;
  onSelectFloor?: (floor: FloorFlatnessReport) => void;
  onSelectBeam?: (beam: BeamDeflectionReport) => void;
  onRefreshQuality?: () => void;
  onDownloadCertificate?: () => void;
  onClose: () => void;
  isLoading?: boolean;
  className?: string;
}

export function SurfaceQualityPanel({
  report,
  heatmap,
  isHeatmapVisible,
  onToggleHeatmap,
  onSelectFloor,
  onSelectBeam,
  onRefreshQuality,
  onDownloadCertificate,
  onClose,
  isLoading = false,
  className = "",
}: SurfaceQualityPanelProps) {
  const [activeTab, setActiveTab] = useState<"floors" | "beams" | "heatmap">("floors");
  const [selectedFloorId, setSelectedFloorId] = useState<string | null>(null);
  const [selectedBeamId, setSelectedBeamId] = useState<string | null>(null);

  const floors = report?.floors ?? [];
  const beams = report?.beams ?? [];
  const overallPass = report?.overall_compliance ?? true;

  return (
    <div
      className={`fixed top-16 right-4 w-96 max-h-[calc(100vh-5rem)] bg-panel border border-border rounded-xl shadow-2xl flex flex-col z-30 overflow-hidden backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-surface/50">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-accent/10 border border-accent/20 text-accent">
            <Ruler className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-fg">Surface Quality & ASTM E1155</h2>
            <p className="text-[11px] text-muted-fg">Floor Flatness (FF/FL) & Deflection</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {onRefreshQuality && (
            <button
              onClick={onRefreshQuality}
              disabled={isLoading}
              title="Rescan surfaces"
              className="p-1.5 rounded-md hover:bg-surface text-muted-fg hover:text-fg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            </button>
          )}
          <button
            onClick={onClose}
            className="p-1.5 rounded-md hover:bg-surface text-muted-fg hover:text-fg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Executive Summary Card */}
      <div className="p-3 border-b border-border bg-surface/20">
        <div className="flex items-center justify-between p-2.5 rounded-lg border border-border bg-panel">
          <div className="flex items-center gap-2.5">
            {overallPass ? (
              <div className="p-1 rounded-md bg-success/10 text-success">
                <ShieldCheck className="w-5 h-5" />
              </div>
            ) : (
              <div className="p-1 rounded-md bg-danger/10 text-danger">
                <ShieldAlert className="w-5 h-5" />
              </div>
            )}
            <div>
              <div className="text-xs font-semibold text-fg">
                {overallPass ? "Tolerances Compliant" : "Review Required"}
              </div>
              <div className="text-[10px] text-muted-fg">
                {floors.length} floor slabs, {beams.length} structural members
              </div>
            </div>
          </div>
          {onDownloadCertificate && (
            <button
              onClick={onDownloadCertificate}
              className="px-2.5 py-1 text-[11px] font-medium rounded-md bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1 transition-opacity shadow-sm"
            >
              <Download className="w-3 h-3" />
              Certificate
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border bg-surface/40 px-2 pt-1.5">
        <button
          onClick={() => setActiveTab("floors")}
          className={`flex-1 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "floors"
              ? "border-accent text-accent"
              : "border-transparent text-muted-fg hover:text-fg"
          }`}
        >
          Floor Slabs ({floors.length})
        </button>
        <button
          onClick={() => setActiveTab("beams")}
          className={`flex-1 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "beams"
              ? "border-accent text-accent"
              : "border-transparent text-muted-fg hover:text-fg"
          }`}
        >
          Beams ({beams.length})
        </button>
        <button
          onClick={() => setActiveTab("heatmap")}
          className={`flex-1 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "heatmap"
              ? "border-accent text-accent"
              : "border-transparent text-muted-fg hover:text-fg"
          }`}
        >
          Heatmap
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {activeTab === "floors" && (
          <>
            {floors.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-fg">
                No floor slab entities analyzed in this model.
              </div>
            ) : (
              floors.map((flr) => {
                const meta = getASTMClassificationMeta(flr.classification);
                const isSelected = selectedFloorId === flr.entity_id;
                return (
                  <div
                    key={flr.entity_id}
                    onClick={() => {
                      setSelectedFloorId(flr.entity_id);
                      onSelectFloor?.(flr);
                    }}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? "border-accent bg-accent/5 ring-1 ring-accent"
                        : "border-border bg-panel hover:bg-surface/50"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-fg">{flr.surface_name}</span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${meta.badgeClass}`}
                      >
                        {meta.label}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                      <div className="p-2 rounded bg-surface border border-border/50">
                        <div className="text-[10px] text-muted-fg">Overall FF (Flatness)</div>
                        <div className="text-sm font-bold text-fg">
                          {formatFFNumber(flr.overall_ff)}{" "}
                          <span className="text-[10px] font-normal text-muted-fg">
                            / target {flr.target_ff}
                          </span>
                        </div>
                      </div>
                      <div className="p-2 rounded bg-surface border border-border/50">
                        <div className="text-[10px] text-muted-fg">Overall FL (Levelness)</div>
                        <div className="text-sm font-bold text-fg">
                          {formatFFNumber(flr.overall_fl)}{" "}
                          <span className="text-[10px] font-normal text-muted-fg">
                            / target {flr.target_fl}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-muted-fg pt-1 border-t border-border/40">
                      <span>Area: {flr.area_sqm} m²</span>
                      <span>Runs: {flr.runs.length} lines</span>
                      <span className={flr.compliance_passed ? "text-success" : "text-danger"}>
                        {flr.compliance_passed ? "PASSED" : "FAILED"}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </>
        )}

        {activeTab === "beams" && (
          <>
            {beams.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-fg">
                No structural beam members analyzed in this model.
              </div>
            ) : (
              beams.map((beam) => {
                const meta = getDeflectionCodeMeta(beam.achieved_code);
                const isSelected = selectedBeamId === beam.entity_id;
                return (
                  <div
                    key={beam.entity_id}
                    onClick={() => {
                      setSelectedBeamId(beam.entity_id);
                      onSelectBeam?.(beam);
                    }}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? "border-accent bg-accent/5 ring-1 ring-accent"
                        : "border-border bg-panel hover:bg-surface/50"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-fg">{beam.beam_name}</span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${meta.badgeClass}`}
                      >
                        {meta.label}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                      <div className="p-2 rounded bg-surface border border-border/50">
                        <div className="text-[10px] text-muted-fg">Max Deflection</div>
                        <div className="text-sm font-bold text-fg">
                          {beam.max_deflection_mm.toFixed(1)} mm
                        </div>
                      </div>
                      <div className="p-2 rounded bg-surface border border-border/50">
                        <div className="text-[10px] text-muted-fg">Deflection Ratio</div>
                        <div className="text-sm font-bold text-fg">
                          {beam.deflection_ratio_str}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-muted-fg pt-1 border-t border-border/40">
                      <span>Span: {beam.span_length_m} m</span>
                      <span className={beam.is_compliant ? "text-success" : "text-danger"}>
                        {beam.is_compliant ? "COMPLIANT" : "EXCESSIVE SAG"}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </>
        )}

        {activeTab === "heatmap" && (
          <div className="space-y-3">
            <div className="p-3 rounded-lg border border-border bg-surface/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-fg">3D Viewport Deviation Mesh</span>
                <button
                  onClick={() => onToggleHeatmap(!isHeatmapVisible)}
                  className={`px-2 py-1 text-[11px] font-medium rounded-md border flex items-center gap-1 transition-colors ${
                    isHeatmapVisible
                      ? "bg-accent/10 border-accent/30 text-accent"
                      : "bg-surface border-border text-muted-fg hover:text-fg"
                  }`}
                >
                  {isHeatmapVisible ? (
                    <>
                      <Eye className="w-3.5 h-3.5" /> Visible
                    </>
                  ) : (
                    <>
                      <EyeOff className="w-3.5 h-3.5" /> Hidden
                    </>
                  )}
                </button>
              </div>

              {heatmap ? (
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between text-muted-fg">
                    <span>Analyzed Points</span>
                    <span className="font-medium text-fg">{heatmap.point_count}</span>
                  </div>
                  <div className="flex justify-between text-muted-fg">
                    <span>RMS Deviation</span>
                    <span className="font-medium text-fg">{heatmap.rms_deviation_mm.toFixed(2)} mm</span>
                  </div>
                  <div className="flex justify-between text-muted-fg">
                    <span>Max Depression</span>
                    <span className="font-medium text-accent">
                      {formatDeviationMm(heatmap.min_deviation_mm)}
                    </span>
                  </div>
                  <div className="flex justify-between text-muted-fg">
                    <span>Max High Spot</span>
                    <span className="font-medium text-danger">
                      {formatDeviationMm(heatmap.max_deviation_mm)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-4 text-center text-xs text-muted-fg">
                  Click below to generate surface deviation heatmap.
                </div>
              )}
            </div>

            {/* Diverging Colormap Legend */}
            <div className="p-3 rounded-lg border border-border bg-surface/30">
              <div className="text-[11px] font-medium text-fg mb-1.5">Deviation Colormap</div>
              <div
                className="h-3 rounded-full mb-1.5"
                style={{ background: "linear-gradient(to right, #2563eb, #10b981, #ef4444)" }}
              />
              <div className="flex justify-between text-[10px] text-muted-fg">
                <span>-25 mm (Depression)</span>
                <span>0 mm (Flush)</span>
                <span>+25 mm (Bulge)</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
