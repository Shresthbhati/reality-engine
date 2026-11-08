"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  Download,
  Eye,
  Filter,
  RefreshCw,
  Search,
  ShieldAlert,
  X,
  Layers,
  ArrowRight,
} from "lucide-react";
import {
  type ClashItem,
  type ClashReportSummary,
  type ClashSeverity,
  type ClashType,
  getClashSeverityColor,
  getClashTypeColor,
  countClashesBySeverity,
  filterClashes,
} from "@/lib/studio/clash-bcf-model";

export interface ClashDetectionPanelProps {
  report: ClashReportSummary | null;
  selectedClashId: string | null;
  onSelectClash: (clash: ClashItem | null) => void;
  onExportBcf: () => void;
  onRefreshClashes?: () => void;
  onClose: () => void;
  isLoading?: boolean;
  className?: string;
}

export function ClashDetectionPanel({
  report,
  selectedClashId,
  onSelectClash,
  onExportBcf,
  onRefreshClashes,
  onClose,
  isLoading = false,
  className = "",
}: ClashDetectionPanelProps) {
  const [severityFilter, setSeverityFilter] = useState<ClashSeverity | "all">("all");
  const [typeFilter, setTypeFilter] = useState<ClashType | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [isExporting, setIsExporting] = useState(false);

  const items = report?.items ?? [];
  const counts = countClashesBySeverity(items);
  const filtered = filterClashes(items, severityFilter, typeFilter, searchQuery);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      await onExportBcf();
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div
      className={`bg-surface/95 backdrop-blur-md border border-line-strong rounded-xl shadow-2xl p-3.5 text-fg text-xs w-[440px] max-w-[calc(100%-2rem)] flex flex-col max-h-[85vh] ${className}`}
      data-testid="clash-detection-panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-line-subtle shrink-0">
        <div className="flex items-center gap-1.5 font-semibold text-fg">
          <ShieldAlert className="w-4 h-4 text-error" />
          <span>Automated Issue & Clash Detection</span>
        </div>
        <div className="flex items-center gap-1.5">
          {report && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium bg-error-subtle text-error border border-error">
              {report.total_clashes} Clashes
            </span>
          )}
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded text-fg-muted hover:text-fg hover:bg-elevated transition-colors"
            title="Close panel"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Severity counter chips */}
      <div className="grid grid-cols-4 gap-1.5 mb-2.5 shrink-0">
        <button
          type="button"
          onClick={() => setSeverityFilter(severityFilter === "critical" ? "all" : "critical")}
          className={`flex flex-col items-center py-1 px-1.5 rounded-md border text-[10px] font-medium transition-colors ${
            severityFilter === "critical"
              ? "bg-error-subtle text-error border-error"
              : "bg-elevated/40 text-error border-error/30 hover:bg-elevated"
          }`}
        >
          <span className="text-[9px] uppercase tracking-wider text-fg-subtle">Critical</span>
          <span className="font-mono font-bold text-[12px]">{counts.critical}</span>
        </button>

        <button
          type="button"
          onClick={() => setSeverityFilter(severityFilter === "major" ? "all" : "major")}
          className={`flex flex-col items-center py-1 px-1.5 rounded-md border text-[10px] font-medium transition-colors ${
            severityFilter === "major"
              ? "bg-warning-subtle text-warning border-warning"
              : "bg-elevated/40 text-warning border-warning/30 hover:bg-elevated"
          }`}
        >
          <span className="text-[9px] uppercase tracking-wider text-fg-subtle">Major</span>
          <span className="font-mono font-bold text-[12px]">{counts.major}</span>
        </button>

        <button
          type="button"
          onClick={() => setSeverityFilter(severityFilter === "minor" ? "all" : "minor")}
          className={`flex flex-col items-center py-1 px-1.5 rounded-md border text-[10px] font-medium transition-colors ${
            severityFilter === "minor"
              ? "bg-info-subtle text-info border-info"
              : "bg-elevated/40 text-info border-info/30 hover:bg-elevated"
          }`}
        >
          <span className="text-[9px] uppercase tracking-wider text-fg-subtle">Minor</span>
          <span className="font-mono font-bold text-[12px]">{counts.minor}</span>
        </button>

        <button
          type="button"
          onClick={() => setSeverityFilter("all")}
          className={`flex flex-col items-center py-1 px-1.5 rounded-md border text-[10px] font-medium transition-colors ${
            severityFilter === "all"
              ? "bg-accent text-on-accent border-accent"
              : "bg-elevated/40 text-fg-muted border-line hover:text-fg hover:bg-elevated"
          }`}
        >
          <span className="text-[9px] uppercase tracking-wider">Total</span>
          <span className="font-mono font-bold text-[12px]">{items.length}</span>
        </button>
      </div>

      {/* Clash type filters */}
      <div className="flex items-center gap-1.5 mb-2 shrink-0">
        {(["all", "hard", "clearance", "duplicate"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTypeFilter(t)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider border transition-colors ${
              typeFilter === t
                ? "bg-surface border-line-strong text-fg font-bold"
                : "bg-elevated/30 border-line-subtle text-fg-muted hover:text-fg"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Search box */}
      <div className="mb-2 shrink-0">
        <div className="relative">
          <Search className="w-3 h-3 absolute left-2 top-2 text-fg-subtle" />
          <input
            type="text"
            placeholder="Search clashes by name, element, or trade..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-elevated/40 border border-line-subtle rounded pl-7 pr-2 py-1 text-[11px] text-fg placeholder:text-fg-subtle focus:outline-none focus:border-accent"
          />
        </div>
      </div>

      {/* Clash items scroll list */}
      <div className="space-y-1.5 overflow-y-auto pr-0.5 flex-1 min-h-[140px]">
        {isLoading ? (
          <div className="text-fg-muted text-center py-6 text-[11px] animate-pulse">
            Analyzing multi-trade geometry & clearance...
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-fg-subtle text-center py-6 text-[11px]">
            {items.length === 0 ? "No clashes detected in this model." : "No clashes match the selected filters."}
          </div>
        ) : (
          filtered.map((clash) => {
            const isSelected = selectedClashId === clash.id;
            const sevColor = getClashSeverityColor(clash.severity);
            const typeColor = getClashTypeColor(clash.clash_type);

            return (
              <div
                key={clash.id}
                onClick={() => onSelectClash(isSelected ? null : clash)}
                className={`p-2 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-elevated border-accent ring-1 ring-accent/60"
                    : "bg-elevated/30 border-line-subtle hover:bg-elevated/60"
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <div className="font-semibold text-fg text-[11px] leading-tight">
                    {clash.title}
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <span
                      className={`text-[9px] px-1.5 py-0.2 rounded font-medium border uppercase tracking-wider ${typeColor.bg} ${typeColor.text} ${typeColor.border}`}
                    >
                      {clash.clash_type}
                    </span>
                    <span
                      className={`text-[9px] px-1.5 py-0.2 rounded font-medium border uppercase tracking-wider ${sevColor.bg} ${sevColor.text} ${sevColor.border}`}
                    >
                      {clash.severity}
                    </span>
                  </div>
                </div>

                {/* Trade pair badges */}
                <div className="flex items-center gap-1 text-[10px] text-fg-muted mb-1 font-mono">
                  <span className="px-1 py-0.5 rounded bg-surface border border-line-subtle uppercase">
                    {clash.trade_a}
                  </span>
                  <ArrowRight className="w-2.5 h-2.5 text-fg-subtle" />
                  <span className="px-1 py-0.5 rounded bg-surface border border-line-subtle uppercase">
                    {clash.trade_b}
                  </span>
                  {clash.penetration_depth_m > 0 && (
                    <span className="text-error font-medium ml-1">
                      Overlap: {(clash.penetration_depth_m * 100).toFixed(1)} cm
                    </span>
                  )}
                  {clash.clearance_distance_m > 0 && (
                    <span className="text-warning font-medium ml-1">
                      Gap: {(clash.clearance_distance_m * 100).toFixed(1)} cm
                    </span>
                  )}
                </div>

                {/* Description */}
                <div className="text-[10px] text-fg-subtle line-clamp-2">
                  {clash.description}
                </div>

                {/* Viewpoint fly action button */}
                {isSelected && clash.viewpoint && (
                  <div className="mt-2 pt-1.5 border-t border-line-subtle flex items-center justify-between">
                    <span className="text-[10px] text-accent font-medium flex items-center gap-1">
                      <Eye className="w-3 h-3" />
                      Viewpoint Focused
                    </span>
                    <span className="text-[10px] text-fg-subtle font-mono">
                      Pos: [{clash.clash_point.map((p) => p.toFixed(1)).join(", ")}]
                    </span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer Actions */}
      <div className="pt-2.5 mt-2 border-t border-line-subtle flex items-center justify-between shrink-0">
        {onRefreshClashes && (
          <button
            type="button"
            onClick={onRefreshClashes}
            disabled={isLoading}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-line bg-elevated/40 text-fg-muted hover:text-fg hover:bg-elevated transition-colors text-[11px] disabled:opacity-50"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? "animate-spin" : ""}`} />
            <span>Rescan</span>
          </button>
        )}

        <button
          type="button"
          onClick={handleExport}
          disabled={isExporting || items.length === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent text-on-accent font-semibold transition-opacity hover:opacity-90 disabled:opacity-50 text-[11px] ml-auto shadow-md"
        >
          <Download className="w-3.5 h-3.5" />
          <span>{isExporting ? "Exporting..." : "Export BCF 2.1 (.bcfzip)"}</span>
        </button>
      </div>
    </div>
  );
}
