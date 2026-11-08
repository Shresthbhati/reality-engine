"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Layers,
  Network,
  RefreshCw,
  Sliders,
  X,
  Zap,
} from "lucide-react";
import {
  type MepFacilityTopology,
  type MepSegment,
  type MepSystemType,
  formatMepLength,
  formatMepSize,
  getMepSystemHex,
} from "@/lib/studio/mep-topology-model";
import { downloadMepGeoJson, downloadMepIfc } from "@/lib/api/mep";

export interface MepTopologyPanelProps {
  worldId: string;
  topology: MepFacilityTopology | null;
  selectedSegmentId: string | null;
  onSelectSegment: (segment: MepSegment | null) => void;
  onRefresh?: () => void;
  onClose: () => void;
  isOpen: boolean;
  isLoading?: boolean;
}

export function MepTopologyPanel({
  worldId,
  topology,
  selectedSegmentId,
  onSelectSegment,
  onRefresh,
  onClose,
  isOpen,
  isLoading = false,
}: MepTopologyPanelProps) {
  const [systemFilter, setSystemFilter] = useState<MepSystemType | "all">("all");
  const [isExportingIfc, setIsExportingIfc] = useState(false);
  const [isExportingGeoJson, setIsExportingGeoJson] = useState(false);

  if (!isOpen) return null;

  const systems = topology?.systems ?? [];
  const allSegments = systems.flatMap((s) => s.segments);
  const filteredSegments = systemFilter === "all"
    ? allSegments
    : allSegments.filter((seg) => seg.system_type === systemFilter);

  const violations = topology?.clearance_violations ?? [];

  const handleExportIfc = async () => {
    setIsExportingIfc(true);
    try {
      await downloadMepIfc(worldId);
    } catch (err) {
      console.warn("Failed to export IFC4x3:", err);
    } finally {
      setIsExportingIfc(false);
    }
  };

  const handleExportGeoJson = async () => {
    setIsExportingGeoJson(true);
    try {
      await downloadMepGeoJson(worldId);
    } catch (err) {
      console.warn("Failed to export GeoJSON:", err);
    } finally {
      setIsExportingGeoJson(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-label="MEP & Conduit Routing Topology"
      className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-border bg-surface text-fg shadow-2xl transition-all duration-300 sm:w-96"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md border border-border bg-base text-fg">
            <Network className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-fg">
              MEP Facility Topology
            </h2>
            <p className="text-xs text-fg-muted">
              IFC4x3 Building Services & Routing
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading}
              title="Refresh MEP Perception"
              className="flex h-7 w-7 items-center justify-center rounded border border-border bg-base text-fg-muted hover:text-fg disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`} />
            </button>
          )}
          <button
            onClick={onClose}
            aria-label="Close MEP Panel"
            className="flex h-7 w-7 items-center justify-center rounded border border-border bg-base text-fg-muted hover:text-fg"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {/* Metric Cards Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-lg border border-border bg-base p-2.5">
            <div className="flex items-center justify-between text-fg-muted">
              <span>Total Network</span>
              <Layers className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 text-base font-bold text-fg">
              {topology ? formatMepLength(topology.summary.total_run_length_m) : "0.00 m"}
            </div>
            <div className="mt-0.5 text-[10px] text-fg-muted">
              {allSegments.length} runs across {systems.length} systems
            </div>
          </div>

          <div className="rounded-lg border border-border bg-base p-2.5">
            <div className="flex items-center justify-between text-fg-muted">
              <span>Code Clearance</span>
              {violations.length === 0 ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
              ) : (
                <AlertTriangle className="h-3.5 w-3.5 text-warning" />
              )}
            </div>
            <div className="mt-1 text-base font-bold text-fg">
              {violations.length === 0 ? "Compliant" : `${violations.length} Conflicts`}
            </div>
            <div className="mt-0.5 text-[10px] text-fg-muted">
              NEC / IPC 150mm separation
            </div>
          </div>

          <div className="rounded-lg border border-border bg-base p-2.5">
            <div className="flex items-center justify-between text-fg-muted">
              <span>Electrical Conduits</span>
              <Zap className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 text-sm font-semibold text-fg">
              {topology ? formatMepLength(topology.total_conduit_length_m) : "0.00 m"}
            </div>
            <div className="mt-0.5 text-[10px] text-fg-muted">
              Power & lighting branch
            </div>
          </div>

          <div className="rounded-lg border border-border bg-base p-2.5">
            <div className="flex items-center justify-between text-fg-muted">
              <span>Pipes & Ducts</span>
              <Sliders className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 text-sm font-semibold text-fg">
              {topology
                ? formatMepLength(topology.total_pipe_length_m + topology.total_duct_length_m)
                : "0.00 m"}
            </div>
            <div className="mt-0.5 text-[10px] text-fg-muted">
              Plumbing water & HVAC
            </div>
          </div>
        </div>

        {/* Clearance Violation Notice */}
        {violations.length > 0 && (
          <div className="rounded-lg border border-border bg-base p-3 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-warning">
              <AlertTriangle className="h-4 w-4" />
              <span>Clearance Separation Warning</span>
            </div>
            <div className="mt-1.5 space-y-1.5">
              {violations.map((v) => (
                <div
                  key={v.violation_id}
                  className="rounded border border-border bg-surface p-2 text-[11px] text-fg-muted"
                >
                  <p className="font-medium text-fg">{v.description}</p>
                  <p className="mt-0.5 text-[10px]">
                    Actual: {(v.actual_clearance_m * 1000).toFixed(1)} mm | Required:{" "}
                    {(v.required_clearance_m * 1000).toFixed(1)} mm
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* System Filter Tabs */}
        <div>
          <div className="text-[11px] font-medium text-fg-muted">Filter by Trade System</div>
          <div className="mt-1 flex flex-wrap gap-1">
            {(
              [
                { id: "all", label: "All Trades" },
                { id: "electrical", label: "Electrical" },
                { id: "plumbing", label: "Plumbing" },
                { id: "hvac", label: "HVAC" },
              ] as const
            ).map((tab) => {
              const active = systemFilter === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setSystemFilter(tab.id)}
                  className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                    active
                      ? "border border-border bg-base text-fg"
                      : "border border-transparent text-fg-muted hover:text-fg"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Distribution Runs List */}
        <div>
          <div className="flex items-center justify-between text-[11px] font-medium text-fg-muted">
            <span>Perceived Distribution Runs ({filteredSegments.length})</span>
          </div>

          <div className="mt-1.5 space-y-1.5">
            {filteredSegments.length === 0 ? (
              <div className="rounded-lg border border-border bg-base p-4 text-center text-xs text-fg-muted">
                No runs perceived for selected system filter.
              </div>
            ) : (
              filteredSegments.map((seg) => {
                const isSelected = selectedSegmentId === seg.segment_id;
                const hex = getMepSystemHex(seg.system_type);
                return (
                  <div
                    key={seg.segment_id}
                    onClick={() => onSelectSegment(isSelected ? null : seg)}
                    className={`cursor-pointer rounded-lg border p-2.5 transition-colors ${
                      isSelected
                        ? "border-fg bg-base shadow-sm"
                        : "border-border bg-base hover:border-border-focus"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: hex }}
                        />
                        <span className="text-xs font-semibold text-fg">
                          {seg.name}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-fg-muted">
                        {formatMepSize(seg)}
                      </span>
                    </div>

                    <div className="mt-1 flex items-center justify-between text-[11px] text-fg-muted">
                      <span>Length: {formatMepLength(seg.length_m)}</span>
                      <span>{seg.waypoints.length} waypoints</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="space-y-2 border-t border-border p-4">
        <button
          onClick={handleExportIfc}
          disabled={isExportingIfc || !topology}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-base py-2 text-xs font-medium text-fg transition-colors hover:border-border-focus hover:bg-surface disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{isExportingIfc ? "Generating IFC..." : "Export IFC4x3 Building Services (.ifc)"}</span>
        </button>

        <button
          onClick={handleExportGeoJson}
          disabled={isExportingGeoJson || !topology}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-base py-2 text-xs font-medium text-fg transition-colors hover:border-border-focus hover:bg-surface disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{isExportingGeoJson ? "Exporting GeoJSON..." : "Export 3D GeoJSON Network"}</span>
        </button>
      </div>
    </div>
  );
}
