'use client';

import React, { useState } from 'react';
import { Clock, TrendingUp, AlertTriangle } from 'lucide-react';

export function SimTimeline() {
  const [scrubberPos, setScrubberPos] = useState(65); // percentage

  // Mock data for sparkline SVG paths
  const floodPoints = "0,45 40,42 80,38 120,32 160,25 200,20 240,16 280,12 320,10 360,8 400,6";
  const agentPoints = "0,10 40,14 80,18 120,24 160,30 200,34 240,38 280,41 320,44 360,45 400,46";

  return (
    <div className="flex flex-col h-[180px] bg-[#121215] text-[#e8e8f0] px-4 py-2.5 select-none justify-between">
      {/* ── Timeline Header with Scrubber Controls ──────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-[11px] font-semibold tracking-wider uppercase text-[#e8e8f0]">
            TEMPORAL PROPAGATION TIMELINE
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-[#5c5c78]">STEP: 100ms</span>
          <span className="text-[#3d8ef7] font-semibold">T+02h 36m (65%)</span>
          <span className="text-[#9898b0]">Total: 04h 00m</span>
        </div>
      </div>

      {/* ── Scrubber Track ──────────────────────────────────────── */}
      <div className="relative w-full my-2">
        {/* Time markers */}
        <div className="flex justify-between text-[9px] font-mono text-[#5c5c78] mb-1">
          <span>T+00h:00m (Rainfall Peak)</span>
          <span>T+01h:00m (Runoff Peak)</span>
          <span>T+02h:00m (River Crest)</span>
          <span>T+03h:00m (Peak Inundation)</span>
          <span>T+04h:00m (Recession)</span>
        </div>

        {/* Interactive Track bar */}
        <div 
          className="relative h-6 bg-[#17171c] rounded border border-[#272733] cursor-pointer overflow-hidden flex items-center"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const pos = ((e.clientX - rect.left) / rect.width) * 100;
            setScrubberPos(Math.max(0, Math.min(100, pos)));
          }}
        >
          {/* Progress fill */}
          <div 
            className="absolute left-0 top-0 bottom-0 bg-[#3d8ef7]/20 border-r-2 border-[#3d8ef7]"
            style={{ width: `${scrubberPos}%` }}
          />

          {/* Key milestone ticks */}
          {[15, 35, 55, 75, 90].map((t) => (
            <div 
              key={t}
              className="absolute top-0 bottom-0 w-[1px] bg-[#2a2a36]"
              style={{ left: `${t}%` }}
            />
          ))}

          {/* Draggable thumb cursor */}
          <div 
            className="absolute w-3 h-8 -top-1 bg-[#3d8ef7] rounded shadow-[0_0_8px_rgba(61,142,247,0.8)] cursor-ew-resize flex items-center justify-center pointer-events-none"
            style={{ left: `calc(${scrubberPos}% - 6px)` }}
          >
            <div className="w-[1px] h-4 bg-white" />
          </div>
        </div>
      </div>

      {/* ── Metric Sparklines & Live Gauges ─────────────────────── */}
      <div className="grid grid-cols-3 gap-3 pt-1 border-t border-[#1e1e28]">
        {/* Metric 1: Inundation Area */}
        <div className="bg-[#17171c] px-3 py-1.5 rounded border border-[#272733] flex items-center justify-between">
          <div>
            <div className="text-[10px] text-[#5c5c78] uppercase font-mono">Inundated Area</div>
            <div className="text-sm font-semibold font-mono text-[#5edaff]">2.41 km²</div>
            <div className="text-[9px] text-[#9898b0] font-mono">+0.18 km²/h rate</div>
          </div>
          <svg className="w-24 h-10 overflow-visible">
            <polyline
              fill="none"
              stroke="#5edaff"
              strokeWidth="2"
              points={floodPoints}
            />
          </svg>
        </div>

        {/* Metric 2: Displaced Agents */}
        <div className="bg-[#17171c] px-3 py-1.5 rounded border border-[#272733] flex items-center justify-between">
          <div>
            <div className="text-[10px] text-[#5c5c78] uppercase font-mono">Evacuated Agents</div>
            <div className="text-sm font-semibold font-mono text-[#34c76f]">8,912 / 12,402</div>
            <div className="text-[9px] text-green-400 font-mono">71.8% clearance</div>
          </div>
          <svg className="w-24 h-10 overflow-visible">
            <polyline
              fill="none"
              stroke="#34c76f"
              strokeWidth="2"
              points={agentPoints}
            />
          </svg>
        </div>

        {/* Metric 3: Critical Asset Warnings */}
        <div className="bg-[#17171c] px-3 py-1.5 rounded border border-[#272733] flex items-center justify-between">
          <div>
            <div className="text-[10px] text-[#5c5c78] uppercase font-mono">Critical Assets at Risk</div>
            <div className="text-sm font-semibold font-mono text-[#f5a623] flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" />
              Substation 04
            </div>
            <div className="text-[9px] text-yellow-400 font-mono">Water clearance: 14 cm</div>
          </div>
          <div className="text-right">
            <span className="text-[9px] font-mono px-1.5 py-0.5 bg-yellow-500/10 text-yellow-400 border border-yellow-500/20 rounded">
              DEFENSE ACTIVE
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
