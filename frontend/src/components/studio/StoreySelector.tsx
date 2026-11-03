"use client";

import React from "react";
import { Layers, Eye, Filter, ArrowUpDown } from "lucide-react";
import {
  type StoreyLevelItem,
  sortStoreysByElevation,
  formatStoreyBadge,
} from "@/lib/studio/storey-model";

export interface StoreySelectorProps {
  storeys: StoreyLevelItem[];
  activeStoreyId: string | "all";
  onSelectStorey: (storeyId: string | "all") => void;
  isolateActiveStorey: boolean;
  onToggleIsolate: (isolate: boolean) => void;
  className?: string;
}

export function StoreySelector({
  storeys,
  activeStoreyId,
  onSelectStorey,
  isolateActiveStorey,
  onToggleIsolate,
  className = "",
}: StoreySelectorProps) {
  if (storeys.length === 0) {
    return null;
  }

  // Display top levels at the top of the stack
  const reversedStoreys = [...sortStoreysByElevation(storeys)].reverse();

  return (
    <div
      className={`bg-zinc-900/90 backdrop-blur-md border border-zinc-700/80 rounded-xl shadow-2xl p-2 text-zinc-200 text-xs ${className}`}
      data-testid="storey-selector"
    >
      <div className="flex items-center justify-between px-1.5 pb-2 mb-1.5 border-b border-zinc-800">
        <div className="flex items-center gap-1.5 text-zinc-300 font-semibold text-[11px]">
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
          <span>Building Storeys</span>
        </div>

        <button
          type="button"
          onClick={() => onToggleIsolate(!isolateActiveStorey)}
          className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors border ${
            isolateActiveStorey
              ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
              : "bg-zinc-800 text-zinc-400 border-zinc-700 hover:text-zinc-200"
          }`}
          title="Isolate active floor cutaway"
        >
          <Filter className="w-2.5 h-2.5" />
          <span>{isolateActiveStorey ? "Isolated" : "All Visible"}</span>
        </button>
      </div>

      <div className="flex flex-col gap-1">
        {/* All Floors Button */}
        <button
          type="button"
          onClick={() => onSelectStorey("all")}
          className={`px-2.5 py-1.5 rounded-lg text-left transition-all flex items-center justify-between border ${
            activeStoreyId === "all"
              ? "bg-indigo-600/30 border-indigo-500/80 text-zinc-100 font-semibold shadow-sm ring-1 ring-indigo-500/30"
              : "bg-zinc-800/40 border-zinc-700/30 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          }`}
        >
          <span>All Levels Combined</span>
          <span className="text-[10px] text-zinc-500 font-normal">Full Model</span>
        </button>

        {/* Individual Storey Buttons */}
        {reversedStoreys.map((st) => {
          const isActive = activeStoreyId === st.storey_id;
          return (
            <button
              key={st.storey_id}
              type="button"
              onClick={() => onSelectStorey(st.storey_id)}
              className={`px-2.5 py-1.5 rounded-lg text-left transition-all flex items-center justify-between border ${
                isActive
                  ? "bg-indigo-600/30 border-indigo-500/80 text-zinc-100 font-semibold shadow-sm ring-1 ring-indigo-500/30"
                  : "bg-zinc-800/40 border-zinc-700/30 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
              }`}
              data-testid={`storey-btn-${st.storey_id}`}
            >
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 rounded bg-zinc-700/60 text-[10px] flex items-center justify-center font-bold text-zinc-300">
                  {st.level_index >= 0 ? `L${st.level_index}` : `B${Math.abs(st.level_index)}`}
                </span>
                <span>{st.name}</span>
              </div>
              <span className="text-[10px] text-zinc-400 font-mono">
                {st.floor_elevation_m >= 0 ? `+${st.floor_elevation_m.toFixed(1)}m` : `${st.floor_elevation_m.toFixed(1)}m`}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
