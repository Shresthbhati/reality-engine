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
      className={`bg-zinc-900/90 backdrop-blur-md border border-zinc-700/80 rounded-xl shadow-2xl p-3 text-zinc-200 text-xs w-80 ${className}`}
      data-testid="temporal-diff-slider"
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-800">
        <div className="flex items-center gap-1.5 font-semibold text-zinc-100">
          <History className="w-4 h-4 text-amber-400" />
          <span>Temporal Epoch Comparison</span>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700 font-mono">
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
              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500"
              : "bg-zinc-800/40 text-emerald-400 border-emerald-500/30 hover:bg-zinc-800"
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
              ? "bg-rose-500/20 text-rose-300 border-rose-500"
              : "bg-zinc-800/40 text-rose-400 border-rose-500/30 hover:bg-zinc-800"
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
              ? "bg-amber-500/20 text-amber-300 border-amber-500"
              : "bg-zinc-800/40 text-amber-400 border-amber-500/30 hover:bg-zinc-800"
          }`}
        >
          <Move className="w-3 h-3" />
          <span>Moved: {counts.moved}</span>
        </button>
      </div>

      {/* Split comparison slider */}
      <div className="bg-zinc-800/50 p-2 rounded-lg border border-zinc-700/40 mb-2.5">
        <div className="flex items-center justify-between text-[10px] text-zinc-400 mb-1">
          <span className="flex items-center gap-1">
            <Calendar className="w-2.5 h-2.5" />
            <span>Before (Past)</span>
          </span>
          <span className="font-mono text-zinc-300">{Math.round(splitRatio * 100)}%</span>
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
          className="w-full accent-amber-500 cursor-pointer h-1.5 bg-zinc-700 rounded-lg"
        />
      </div>

      {/* Change items list */}
      <div className="space-y-1 max-h-48 overflow-y-auto pr-0.5">
        {filteredChanges.length === 0 ? (
          <div className="text-zinc-500 text-center py-2 text-[11px]">
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
                    ? "bg-zinc-800 border-indigo-500 ring-1 ring-indigo-500/50"
                    : "bg-zinc-800/30 border-zinc-700/40 hover:bg-zinc-800/60"
                }`}
              >
                <div>
                  <div className="font-medium text-zinc-200 text-[11px]">{c.name}</div>
                  {c.displacement_m !== undefined && (
                    <div className="text-[10px] text-zinc-400">
                      Displaced {c.displacement_m.toFixed(2)}m
                    </div>
                  )}
                  {c.details && (
                    <div className="text-[10px] text-zinc-400">{c.details}</div>
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
