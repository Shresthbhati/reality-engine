'use client';

import React, { useState } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import { CityToolPalette, type CityTool } from './CityToolPalette';
import { WorldTree } from './WorldTree';
import { CityMapView } from './CityMapView';
import { RuleInspector } from './RuleInspector';

export function CityBuilderWorkspace() {
  const [activeTool, setActiveTool] = useState<CityTool>('SELECT');

  return (
    <div className="flex flex-col w-full h-full overflow-hidden bg-[#0d0d0f] text-[#e8e8f0]">
      {/* ── Top Tool Palette (38px) ─────────────────────────────── */}
      <CityToolPalette 
        activeTool={activeTool} 
        onSelectTool={setActiveTool} 
      />

      {/* ── Resizable 3-Column Layout: WorldTree | MapView | RuleInspector ── */}
      <div className="flex-1 min-h-0 w-full">
        <Group orientation="horizontal">
          {/* Left Panel: City Assembly Outliner & Sessions Compositor */}
          <Panel defaultSize={22} minSize={16} maxSize={35}>
            <div className="h-full w-full bg-[#121215] border-r border-[#1e1e28] overflow-hidden">
              <WorldTree />
            </div>
          </Panel>

          <Separator className="w-[3px] bg-[#1e1e28] hover:bg-[#3d8ef7] transition-colors cursor-col-resize active:bg-[#3d8ef7]" />

          {/* Center: 2D GIS / Top-Down City Authoring Map */}
          <Panel defaultSize={54} minSize={30}>
            <div className="h-full w-full relative bg-[#0d0d0f] overflow-hidden">
              <CityMapView activeTool={activeTool} />
            </div>
          </Panel>

          <Separator className="w-[3px] bg-[#1e1e28] hover:bg-[#3d8ef7] transition-colors cursor-col-resize active:bg-[#3d8ef7]" />

          {/* Right Panel: Procedural Grammar & Rules Inspector */}
          <Panel defaultSize={24} minSize={18} maxSize={38}>
            <div className="h-full w-full bg-[#121215] border-l border-[#1e1e28] overflow-hidden">
              <RuleInspector />
            </div>
          </Panel>
        </Group>
      </div>
    </div>
  );
}

export default CityBuilderWorkspace;
