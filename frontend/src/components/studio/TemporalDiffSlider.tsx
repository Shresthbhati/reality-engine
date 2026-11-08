"use client";

import React, { useState } from "react";
import {
  History,
  Calendar,
  Layers,
  PlusCircle,
  MinusCircle,
  Move,
  Wrench,
  Boxes,
} from "lucide-react";
import {
  type TemporalDiffSummary,
  type TemporalEpochSummary,
  type TemporalChangeKind,
  type TradeCategory,
  getTemporalChangeColor,
  countChangesByKind,
  filterTemporalChanges,
} from "@/lib/studio/temporal-model";

export interface TemporalDiffSliderProps {
  diff: TemporalDiffSummary | null;
  splitRatio: number;
  onSplitRatioChange: (ratio: number) => void;
  selectedEntityId: string | null;
  onSelectEntity: (entityId: string) => void;
  availableEpochs?: TemporalEpochSummary[];
  onSelectEpochs?: (fromEpochId: string, toEpochId: string) => void;
  isLoading?: boolean;
  className?: string;
}

const TRADES: { id: TradeCategory | "all"; label: string }[] = [
  { id: "all", label: "All Trades" },
  { id: "structure", label: "Structure" },
  { id: "architecture", label: "Arch" },
  { id: "mep", label: "MEP" },
  { id: "finishes", label: "Finishes" },
  { id: "equipment", label: "Equipment" },
];

export function TemporalDiffSlider({
  diff,
  splitRatio,
  onSplitRatioChange,
  selectedEntityId,
  onSelectEntity,
  availableEpochs = [],
  onSelectEpochs,
  isLoading = false,
  className = "",
}: TemporalDiffSliderProps) {
  const [activeFilter, setActiveFilter] = useState<TemporalChangeKind | "all">("all");
  const [activeTrade, setActiveTrade] = useState<TradeCategory | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");

  if (!diff && !isLoading) {
    return null;
  }

  const counts = diff ? countChangesByKind(diff.changes) : null;
  const filteredChanges = diff
    ? filterTemporalChanges(diff.changes, activeFilter, searchQuery, activeTrade)
    : [];

  const residualError =
    diff?.residual_alignment_error_m ?? diff?.residual_error_m ?? 0;
  const netVolume = diff?.net_volume_change_m3 ?? 0;
  const clusterCount = diff?.spatial_clusters?.length ?? 0;

  return (
    <div
      className={`bg-surface/95 backdrop-blur-md border border-line-strong rounded-xl shadow-2xl p-3.5 text-fg text-xs w-96 ${className}`}
      data-testid="temporal-diff-slider"
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-line-subtle">
        <div className="flex items-center gap-1.5 font-semibold text-fg">
          <History className="w-4 h-4 text-warning" />
          <span>4D Site As-Built Delta</span>
        </div>
        {diff && (
          <div className="flex items-center gap-1">
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded font-mono border ${
                diff.is_aligned
                  ? "bg-success-subtle text-success border-success"
                  : "bg-warning-subtle text-warning border-warning"
              }`}
              title={`Static anchors: ${diff.static_anchor_count}, Residual error: ${residualError.toFixed(4)}m`}
            >
              {diff.is_aligned ? "Aligned" : "Drift"} ({(residualError * 1000).toFixed(1)}mm)
            </span>
          </div>
        )}
      </div>

      {/* Epoch selection dropdowns if available */}
      {availableEpochs.length >= 2 && onSelectEpochs && diff && (
        <div className="grid grid-cols-2 gap-2 mb-2.5 bg-elevated/40 p-1.5 rounded-lg border border-line-subtle">
          <div>
            <label className="text-[10px] text-fg-muted block mb-0.5 font-medium">Base Epoch (From)</label>
            <select
              value={diff.from_epoch_id}
              onChange={(e) => onSelectEpochs(e.target.value, diff.to_epoch_id)}
              className="w-full bg-surface text-fg border border-line rounded px-1.5 py-1 text-[11px] focus:outline-none focus:border-accent"
            >
              {availableEpochs.map((ep) => (
                <option key={ep.epoch_id} value={ep.epoch_id}>
                  {ep.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-fg-muted block mb-0.5 font-medium">Target Epoch (To)</label>
            <select
              value={diff.to_epoch_id}
              onChange={(e) => onSelectEpochs(diff.from_epoch_id, e.target.value)}
              className="w-full bg-surface text-fg border border-line rounded px-1.5 py-1 text-[11px] focus:outline-none focus:border-accent"
            >
              {availableEpochs.map((ep) => (
                <option key={ep.epoch_id} value={ep.epoch_id}>
                  {ep.label} {ep.is_current ? "(Current)" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Metric badges: Net volume and spatial clusters */}
      {diff && (
        <div className="flex items-center justify-between text-[11px] bg-elevated/30 px-2 py-1.5 rounded-md border border-line-subtle mb-2.5">
          <div className="flex items-center gap-1.5 text-fg-muted">
            <Boxes className="w-3.5 h-3.5 text-accent" />
            <span>Net Volume Delta:</span>
            <span
              className={`font-mono font-medium ${
                netVolume >= 0 ? "text-success" : "text-error"
              }`}
            >
              {netVolume >= 0 ? `+${netVolume.toFixed(2)}` : netVolume.toFixed(2)} m³
            </span>
          </div>
          {clusterCount > 0 && (
            <div className="flex items-center gap-1 text-[10px] text-info font-medium">
              <Layers className="w-3 h-3 text-info" />
              <span>{clusterCount} voxel clusters</span>
            </div>
          )}
        </div>
      )}

      {/* Change counter chips */}
      {counts && (
        <div className="grid grid-cols-4 gap-1.5 mb-2.5">
          <button
            type="button"
            onClick={() => setActiveFilter(activeFilter === "added" ? "all" : "added")}
            className={`flex items-center justify-center gap-1 px-1.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
              activeFilter === "added"
                ? "bg-success-subtle text-success border-success"
                : "bg-elevated/40 text-success border-success/30 hover:bg-elevated"
            }`}
          >
            <PlusCircle className="w-3 h-3" />
            <span>+{counts.added}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveFilter(activeFilter === "removed" ? "all" : "removed")}
            className={`flex items-center justify-center gap-1 px-1.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
              activeFilter === "removed"
                ? "bg-error-subtle text-error border-error"
                : "bg-elevated/40 text-error border-error/30 hover:bg-elevated"
            }`}
          >
            <MinusCircle className="w-3 h-3" />
            <span>-{counts.removed}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveFilter(activeFilter === "moved" ? "all" : "moved")}
            className={`flex items-center justify-center gap-1 px-1.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
              activeFilter === "moved"
                ? "bg-warning-subtle text-warning border-warning"
                : "bg-elevated/40 text-warning border-warning/30 hover:bg-elevated"
            }`}
          >
            <Move className="w-3 h-3" />
            <span>Δ {counts.moved}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveFilter(activeFilter === "renovated" ? "all" : "renovated")}
            className={`flex items-center justify-center gap-1 px-1.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
              activeFilter === "renovated"
                ? "bg-info-subtle text-info border-info"
                : "bg-elevated/40 text-info border-info/30 hover:bg-elevated"
            }`}
          >
            <Wrench className="w-3 h-3" />
            <span>{counts.renovated}</span>
          </button>
        </div>
      )}

      {/* Trade filters */}
      <div className="flex items-center gap-1 overflow-x-auto pb-1 mb-2 scrollbar-none">
        {TRADES.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveTrade(t.id)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium whitespace-nowrap transition-colors border ${
              activeTrade === t.id
                ? "bg-accent text-on-accent border-accent"
                : "bg-elevated/40 text-fg-muted border-line hover:text-fg hover:bg-elevated"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Split comparison slider */}
      <div className="bg-elevated/50 p-2 rounded-lg border border-line-subtle mb-2.5">
        <div className="flex items-center justify-between text-[10px] text-fg-muted mb-1">
          <span className="flex items-center gap-1">
            <Calendar className="w-2.5 h-2.5" />
            <span>Before (Base)</span>
          </span>
          <span className="font-mono text-fg font-medium">{Math.round(splitRatio * 100)}%</span>
          <span className="flex items-center gap-1">
            <span>After (Target)</span>
            <Calendar className="w-2.5 h-2.5" />
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={splitRatio}
          onChange={(e) => onSplitRatioChange(parseFloat(e.target.value))}
          className="w-full accent-warning cursor-pointer h-1.5 bg-line rounded-lg"
        />
      </div>

      {/* Quick Search */}
      <div className="mb-2">
        <input
          type="text"
          placeholder="Filter changes by name or ID..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full bg-elevated/40 border border-line-subtle rounded px-2 py-1 text-[11px] text-fg placeholder:text-fg-subtle focus:outline-none focus:border-accent"
        />
      </div>

      {/* Change items list */}
      <div className="space-y-1 max-h-48 overflow-y-auto pr-0.5">
        {isLoading ? (
          <div className="text-fg-muted text-center py-4 text-[11px] animate-pulse">
            Computing 4D temporal delta...
          </div>
        ) : filteredChanges.length === 0 ? (
          <div className="text-fg-subtle text-center py-2 text-[11px]">
            No changes match current filter
          </div>
        ) : (
          filteredChanges.map((c) => {
            const isSelected = selectedEntityId === c.entity_id;
            const colors = getTemporalChangeColor(c.change_kind);

            return (
              <div
                key={c.entity_id}
                onClick={() => onSelectEntity(c.entity_id)}
                className={`p-1.5 rounded-md border flex items-center justify-between cursor-pointer transition-colors ${
                  isSelected
                    ? "bg-elevated border-accent ring-1 ring-accent/50"
                    : "bg-elevated/30 border-line-subtle hover:bg-elevated/60"
                }`}
              >
                <div className="min-w-0 pr-2">
                  <div className="font-medium text-fg text-[11px] truncate">{c.name}</div>
                  <div className="flex items-center gap-1.5 text-[10px] text-fg-muted">
                    {c.trade && <span className="capitalize">{c.trade}</span>}
                    {c.displacement_m !== undefined && c.displacement_m > 0 && (
                      <span>• {c.displacement_m.toFixed(2)}m</span>
                    )}
                    {c.volume_delta_m3 !== undefined && c.volume_delta_m3 !== 0 && (
                      <span>
                        • {c.volume_delta_m3 > 0 ? "+" : ""}
                        {c.volume_delta_m3.toFixed(2)} m³
                      </span>
                    )}
                  </div>
                  {c.details && (
                    <div className="text-[10px] text-fg-subtle truncate">{c.details}</div>
                  )}
                </div>

                <span
                  className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded font-medium border uppercase tracking-wider ${colors.bg} ${colors.text} ${colors.border}`}
                >
                  {c.change_kind}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
