"use client";

import React, { useState } from "react";
import {
  History,
  Calendar,
  Split,
  PlusCircle,
  MinusCircle,
  Move,
  Wrench,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";
import {
  type TemporalEntityChangeItem,
  type TemporalDiffSummary,
  type TemporalChangeKind,
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
  className?: string;
}

export function TemporalDiffSlider({
  diff,
  splitRatio,
  onSplitRatioChange,
  selectedEntityId,
  onSelectEntity,
  className = "",
}: TemporalDiffSliderProps) {
  const [activeFilter, setActiveFilter] = useState<TemporalChangeKind | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");

  if (!diff) {
    return null;
  }

  const counts = countChangesByKind(diff.changes);
  const filteredChanges = filterTemporalChanges(diff.changes, activeFilter, searchQuery);

  return (
    <div
      className={`bg-surface/90 backdrop-blur-md border border-line/80 rounded-xl shadow-2xl p-3 text-fg text-xs w-80 ${className}`}
      data-testid="temporal-diff-slider"
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-line-subtle">
        <div className="flex items-center gap-1.5 font-semibold text-fg">
          <History className="w-4 h-4 text-warning" />
          <span>Temporal Epoch Comparison</span>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-elevated text-fg-muted border border-line font-mono">
          {diff.from_epoch_id} ➔ {diff.to_epoch_id}
        </span>
      </div>

      {/* Change counter chips */}
      <div className="grid grid-cols-3 gap-1.5 mb-2.5">
        <button
          type="button"
          onClick={() => setActiveFilter(activeFilter === "added" ? "all" : "added")}
          className={`flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-medium transition-colors ${
            activeFilter === "added"
              ? "bg-success-subtle text-success border-success"
              : "bg-elevated/40 text-success border-success/30 hover:bg-elevated"
          }`}
        >
          <PlusCircle className="w-3 h-3" />
          <span>Added: {counts.added}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveFilter(activeFilter === "removed" ? "all" : "removed")}
          className={`flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-medium transition-colors ${
            activeFilter === "removed"
              ? "bg-error-subtle text-error border-error"
              : "bg-elevated/40 text-error border-error/30 hover:bg-elevated"
          }`}
        >
          <MinusCircle className="w-3 h-3" />
          <span>Removed: {counts.removed}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveFilter(activeFilter === "moved" ? "all" : "moved")}
          className={`flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-medium transition-colors ${
            activeFilter === "moved"
              ? "bg-warning-subtle text-warning border-warning"
              : "bg-elevated/40 text-warning border-warning/30 hover:bg-elevated"
          }`}
        >
          <Move className="w-3 h-3" />
          <span>Moved: {counts.moved}</span>
        </button>
      </div>

      {/* Split comparison slider */}
      <div className="bg-elevated/50 p-2 rounded-lg border border-line/40 mb-2.5">
        <div className="flex items-center justify-between text-[10px] text-fg-muted mb-1">
          <span className="flex items-center gap-1">
            <Calendar className="w-2.5 h-2.5" />
            <span>Before (Past)</span>
          </span>
          <span className="font-mono text-fg-muted">{Math.round(splitRatio * 100)}%</span>
          <span className="flex items-center gap-1">
            <span>After (Current)</span>
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

      {/* Change items list */}
      <div className="space-y-1 max-h-48 overflow-y-auto pr-0.5">
        {filteredChanges.length === 0 ? (
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
                    : "bg-elevated/30 border-line/40 hover:bg-elevated/60"
                }`}
              >
                <div>
                  <div className="font-medium text-fg text-[11px]">{c.name}</div>
                  {c.displacement_m !== undefined && (
                    <div className="text-[10px] text-fg-muted">
                      Displaced {c.displacement_m.toFixed(2)}m
                    </div>
                  )}
                  {c.details && (
                    <div className="text-[10px] text-fg-muted">{c.details}</div>
                  )}
                </div>

                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-medium border uppercase tracking-wider ${colors.bg} ${colors.text} ${colors.border}`}
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
