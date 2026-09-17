'use client';

import React from 'react';
import { 
  MousePointer, 
  Milestone, 
  Building2, 
  Grid, 
  Mountain, 
  Paintbrush, 
  ScatterChart, 
  Binary, 
  Ruler, 
  Undo2, 
  Redo2, 
  Wand2 
} from 'lucide-react';

export type CityTool = 
  | 'SELECT' 
  | 'ROAD' 
  | 'BUILDING' 
  | 'PARCEL' 
  | 'TERRAIN' 
  | 'PAINT' 
  | 'SCATTER' 
  | 'PROCEDURAL' 
  | 'MEASURE';

interface CityToolPaletteProps {
  activeTool: CityTool;
  onSelectTool: (tool: CityTool) => void;
}

export function CityToolPalette({ activeTool, onSelectTool }: CityToolPaletteProps) {
  const toolGroups: { name: string; tools: { id: CityTool; label: string; icon: React.ComponentType<{ className?: string }>; shortcut: string }[] }[] = [
    {
      name: 'Selection',
      tools: [
        { id: 'SELECT', label: 'Select (S)', icon: MousePointer, shortcut: 'S' },
      ],
    },
    {
      name: 'Authoring',
      tools: [
        { id: 'ROAD', label: 'Road Network (R)', icon: Milestone, shortcut: 'R' },
        { id: 'BUILDING', label: 'Building Footprint (B)', icon: Building2, shortcut: 'B' },
        { id: 'PARCEL', label: 'Zoning Parcel (P)', icon: Grid, shortcut: 'P' },
      ],
    },
    {
      name: 'Environment',
      tools: [
        { id: 'TERRAIN', label: 'Terrain Sculpt (T)', icon: Mountain, shortcut: 'T' },
        { id: 'PAINT', label: 'Surface Paint', icon: Paintbrush, shortcut: 'M' },
        { id: 'SCATTER', label: 'Vegetation Scatter', icon: ScatterChart, shortcut: 'V' },
      ],
    },
    {
      name: 'Procedural',
      tools: [
        { id: 'PROCEDURAL', label: 'Grammar Rule (G)', icon: Binary, shortcut: 'G' },
        { id: 'MEASURE', label: 'Measure Survey', icon: Ruler, shortcut: 'X' },
      ],
    },
  ];

  return (
    <div className="flex items-center justify-between px-3 h-[38px] bg-[#121215] border-b border-[#1e1e28] select-none">
      {/* Tool Groups */}
      <div className="flex items-center gap-1">
        {toolGroups.map((group, gIdx) => (
          <React.Fragment key={group.name}>
            <div className="flex items-center gap-0.5">
              {group.tools.map((t) => {
                const Icon = t.icon;
                const isActive = activeTool === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => onSelectTool(t.id)}
                    title={t.label}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-all ${
                      isActive 
                        ? 'bg-[#3d8ef7] text-white font-medium shadow-[0_0_8px_rgba(61,142,247,0.4)]' 
                        : 'text-[#9898b0] hover:text-[#e8e8f0] hover:bg-[#1c1c23]'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span className="hidden md:inline text-[11px]">{t.id}</span>
                  </button>
                );
              })}
            </div>

            {gIdx < toolGroups.length - 1 && (
              <div className="w-[1px] h-4 bg-[#272733] mx-1.5" />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Right side actions: Quick Generate & Undo/Redo */}
      <div className="flex items-center gap-2">
        <button 
          title="Undo (Ctrl+Z)"
          className="p-1.5 text-[#5c5c78] hover:text-[#e8e8f0] hover:bg-[#1c1c23] rounded transition-colors"
        >
          <Undo2 className="w-3.5 h-3.5" />
        </button>
        <button 
          title="Redo (Ctrl+Y)"
          className="p-1.5 text-[#5c5c78] hover:text-[#e8e8f0] hover:bg-[#1c1c23] rounded transition-colors"
        >
          <Redo2 className="w-3.5 h-3.5" />
        </button>

        <div className="w-[1px] h-4 bg-[#272733]" />

        <button 
          className="flex items-center gap-1.5 px-2.5 py-1 bg-[#22222c] hover:bg-[#2a2a36] text-[#3d8ef7] hover:text-[#5ba3fa] border border-[#3d8ef7]/30 rounded text-[11px] font-medium transition-colors"
        >
          <Wand2 className="w-3 h-3" />
          <span>Synthesize District</span>
        </button>
      </div>
    </div>
  );
}
