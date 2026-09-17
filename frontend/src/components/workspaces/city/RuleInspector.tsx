'use client';

import React, { useState } from 'react';
import { Sliders, Sparkles, Building, ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';

export function RuleInspector() {
  const [minHeight, setMinHeight] = useState(14);
  const [maxHeight, setMaxHeight] = useState(32);
  const [floors, setFloors] = useState('4-8');
  const [setback, setSetback] = useState(5.0);
  const [density, setDensity] = useState(0.74);
  const [style, setStyle] = useState('Industrial Brick / Concrete');
  const [copied, setCopied] = useState(false);

  return (
    <div className="flex flex-col h-full bg-[#121215] text-[#e8e8f0] select-none text-xs">
      {/* Panel Header */}
      <div className="flex items-center justify-between px-3 h-[32px] bg-[#17171c] border-b border-[#1e1e28]">
        <div className="flex items-center gap-1.5 font-semibold text-[11px] uppercase tracking-wider text-[#9898b0]">
          <Sliders className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span>Procedural Rule Inspector</span>
        </div>
        <span className="text-[10px] font-mono text-[#34c76f] bg-green-500/10 px-1.5 py-0.5 rounded border border-green-500/20">
          Rule V2.4
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Selected Context */}
        <div className="p-2.5 rounded bg-[#17171c] border border-[#272733] space-y-1">
          <div className="text-[10px] text-[#5c5c78] uppercase tracking-wider">Target District / Parcel</div>
          <div className="text-[12px] font-semibold text-[#e8e8f0]">Sector 04 — Heavy Industry Zone</div>
          <div className="text-[10px] font-mono text-[#3d8ef7]">Area: 48,200 m² | Max Permissible FAR: 3.5</div>
        </div>

        {/* Building Grammar Rule Configuration */}
        <div className="space-y-3">
          <div className="text-[10px] font-semibold text-[#9898b0] uppercase tracking-wider flex items-center gap-1.5">
            <Building className="w-3.5 h-3.5 text-[#3d8ef7]" />
            <span>Building Envelope Constraints</span>
          </div>

          <div className="space-y-2.5 bg-[#17171c] p-3 rounded border border-[#272733]">
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#9898b0]">Height Envelope</span>
                <span className="font-mono text-[#3d8ef7]">{minHeight}m – {maxHeight}m</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="6"
                  max="80"
                  value={maxHeight}
                  onChange={(e) => setMaxHeight(Number(e.target.value))}
                  className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#9898b0]">Floor Count Range</span>
                <span className="font-mono text-[#3d8ef7]">{floors} Storeys</span>
              </div>
              <input
                type="text"
                value={floors}
                onChange={(e) => setFloors(e.target.value)}
                className="w-full bg-[#0d0d0f] border border-[#272733] rounded px-2 py-1 text-xs font-mono text-[#e8e8f0]"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#9898b0]">Parcel Perimeter Setback</span>
                <span className="font-mono text-[#3d8ef7]">{setback.toFixed(1)} m</span>
              </div>
              <input
                type="range"
                min="0"
                max="15"
                step="0.5"
                value={setback}
                onChange={(e) => setSetback(Number(e.target.value))}
                className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#9898b0]">Lot Coverage Density</span>
                <span className="font-mono text-[#3d8ef7]">{Math.round(density * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.02"
                value={density}
                onChange={(e) => setDensity(Number(e.target.value))}
                className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
              />
            </div>

            <div>
              <div className="text-[11px] text-[#9898b0] mb-1">Architectural Grammar Typology</div>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value)}
                className="w-full bg-[#0d0d0f] border border-[#272733] text-[#e8e8f0] text-xs rounded px-2 py-1 cursor-pointer"
              >
                <option>Industrial Brick / Concrete</option>
                <option>Modern Glass Curtain Wall</option>
                <option>Modular Steel Truss Warehousing</option>
                <option>Historical Masonry Residential</option>
              </select>
            </div>
          </div>
        </div>

        {/* Synthesis & Generation Actions */}
        <div className="space-y-2 pt-1">
          <button 
            className="w-full py-2 bg-[#3d8ef7] hover:bg-[#5ba3fa] text-white rounded text-xs font-medium flex items-center justify-center gap-1.5 transition-colors shadow-[0_0_12px_rgba(61,142,247,0.3)]"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Generate 4 Procedural Variants</span>
          </button>

          <button 
            onClick={() => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
            className="w-full py-1.5 bg-[#17171c] hover:bg-[#22222c] text-[#9898b0] hover:text-[#e8e8f0] rounded border border-[#272733] text-xs font-mono flex items-center justify-center gap-1.5 transition-colors"
          >
            {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Rule Copied to Clipboard' : 'Copy Rule JSON'}</span>
          </button>
        </div>

        <div className="text-[9px] text-[#5c5c78] font-mono italic px-1">
          {"// WIRE: POST /api/worlds/{worldId}/procedural/generate"}
        </div>
      </div>
    </div>
  );
}
