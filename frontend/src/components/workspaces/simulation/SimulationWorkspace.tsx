'use client';

/**
 * SimulationWorkspace — Main simulation control room layout.
 * Inspired by Unreal's play mode + scientific control room + monitoring dashboard.
 *
 * Layout:
 *  ┌──────────────────────────────────────────────────┐
 *  │ SimControls (top bar ~48px)                      │
 *  ├──────────────────┬───────────────────────────────┤
 *  │ ScenarioConfig   │ SimWorldView (canvas)          │
 *  │ (left 300px)     │                               │
 *  ├──────────────────┴───────────────────────────────┤
 *  │ SimTimeline (bottom 180px)                       │
 *  └──────────────────────────────────────────────────┘
 */

import { SimControls } from './SimControls';
import { ScenarioConfig } from './ScenarioConfig';
import { SimWorldView } from './SimWorldView';
import { SimTimeline } from './SimTimeline';

export function SimulationWorkspace() {
  return (
    <div
      className="flex flex-col w-full h-full overflow-hidden"
      style={{ background: '#0d0d0f' }}
    >
      {/* ── Top: Transport controls ─────────────────────────── */}
      <SimControls />

      {/* ── Middle: Config panel + World view ───────────────── */}
      <div className="flex flex-1 min-h-0">
        {/* Left config panel — fixed width */}
        <div
          className="flex-none w-[300px] border-r border-[#1e1e28] overflow-y-auto overflow-x-hidden"
          style={{ background: '#121215' }}
        >
          <ScenarioConfig />
        </div>

        {/* Center world view — grows to fill */}
        <div className="flex-1 min-w-0 relative">
          <SimWorldView />
        </div>
      </div>

      {/* ── Bottom: Timeline + metrics ───────────────────────── */}
      <div
        className="flex-none border-t border-[#1e1e28]"
        style={{ height: 180, background: '#121215' }}
      >
        <SimTimeline />
      </div>
    </div>
  );
}

export default SimulationWorkspace;
