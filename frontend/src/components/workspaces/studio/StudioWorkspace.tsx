'use client';

/**
 * StudioWorkspace — Spatial Workstation for Reality Reconstruction.
 * - Collapsible Outliner (⌘B)
 * - Collapsible Inspector (⌘I)
 * - Collapsed-by-default Bottom Drawer (⌘J) with minimal 28px docking bar
 * - 3D Viewport visually dominates the screen
 */

import React, { useEffect } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import { useREStore } from '@/store/re-store';
import { ViewportToolbar } from './ViewportToolbar';
import { WorldOutliner } from './WorldOutliner';
import { Viewport3D } from './Viewport3D';
import { EntityInspector } from './EntityInspector';
import EvidencePanel from './EvidencePanel';
import {
  ChevronUp,
  ChevronDown,
  Activity,
  Layers,
  Terminal,
  Cpu,
  Clock,
  PanelLeft,
  PanelRight,
} from 'lucide-react';

export default function StudioWorkspace() {
  const {
    outlinerCollapsed,
    toggleOutliner,
    inspectorCollapsed,
    toggleInspector,
    bottomDrawerOpen,
    bottomDrawerTab,
    setBottomDrawerTab,
    toggleBottomDrawer,
  } = useREStore();

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleOutliner();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'i') {
        e.preventDefault();
        toggleInspector();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault();
        toggleBottomDrawer();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleOutliner, toggleInspector, toggleBottomDrawer]);

  const bottomTabs: Array<{
    id: typeof bottomDrawerTab;
    label: string;
    icon: React.ReactNode;
  }> = [
    { id: 'jobs', label: 'Jobs (0)', icon: <Activity className="w-3 h-3" /> },
    { id: 'pipeline', label: 'Pipeline & Timeline', icon: <Clock className="w-3 h-3" /> },
    { id: 'evidence', label: 'Evidence Graph', icon: <Layers className="w-3 h-3" /> },
    { id: 'diagnostics', label: 'Diagnostics', icon: <Cpu className="w-3 h-3" /> },
    { id: 'console', label: 'Engine Console', icon: <Terminal className="w-3 h-3" /> },
  ];

  return (
    <div className="flex flex-col w-full h-full overflow-hidden bg-[#0d0d0f] text-[#e8e8f0]">
      {/* ── Top Viewport Controls Bar (32px) ──────────────────────── */}
      <ViewportToolbar />

      {/* ── Main Workstation Resizable Grid ───────────────────────── */}
      <div className="flex-1 min-h-0 w-full flex flex-col overflow-hidden">
        {/* If bottom drawer is open, split vertically; otherwise, viewport takes 100% height */}
        {bottomDrawerOpen ? (
          <Group orientation="vertical">
            {/* Top Row: Outliner | 3D Viewport | Inspector */}
            <Panel defaultSize={74} minSize={40}>
              <Group orientation="horizontal">
                {/* Left Outliner */}
                {!outlinerCollapsed && (
                  <>
                    <Panel defaultSize={20} minSize={14} maxSize={35}>
                      <div className="h-full w-full bg-[#101217] border-r border-[#1f222b] overflow-hidden">
                        <WorldOutliner />
                      </div>
                    </Panel>
                    <Separator className="w-[2px] bg-[#1f222b] hover:bg-[#3d8ef7] transition-colors cursor-col-resize active:bg-[#3d8ef7]" />
                  </>
                )}

                {/* Center 3D Viewport (Expands to fill available space) */}
                <Panel minSize={30}>
                  <div className="h-full w-full relative bg-[#08090b] overflow-hidden">
                    <Viewport3D />
                  </div>
                </Panel>

                {/* Right Inspector */}
                {!inspectorCollapsed && (
                  <>
                    <Separator className="w-[2px] bg-[#1f222b] hover:bg-[#3d8ef7] transition-colors cursor-col-resize active:bg-[#3d8ef7]" />
                    <Panel defaultSize={24} minSize={18} maxSize={40}>
                      <div className="h-full w-full bg-[#101217] border-l border-[#1f222b] overflow-hidden">
                        <EntityInspector />
                      </div>
                    </Panel>
                  </>
                )}
              </Group>
            </Panel>

            <Separator className="h-[2px] bg-[#1f222b] hover:bg-[#3d8ef7] transition-colors cursor-row-resize active:bg-[#3d8ef7]" />

            {/* Bottom Panel Drawer */}
            <Panel defaultSize={26} minSize={15} maxSize={50}>
              <div className="h-full w-full bg-[#101217] overflow-hidden flex flex-col">
                <EvidencePanel />
              </div>
            </Panel>
          </Group>
        ) : (
          // Maximized Viewport Layout (Bottom Drawer Collapsed)
          <div className="flex-1 w-full h-full min-h-0 flex overflow-hidden">
            {/* Left Outliner */}
            {!outlinerCollapsed && (
              <div className="w-64 shrink-0 h-full border-r border-[#1f222b] bg-[#101217] overflow-hidden">
                <WorldOutliner />
              </div>
            )}

            {/* Center 3D Viewport — The Dominant Hero */}
            <div className="flex-1 h-full min-w-0 relative bg-[#08090b] overflow-hidden">
              <Viewport3D />

              {/* Floating Quick Restore Buttons when panels are collapsed */}
              {outlinerCollapsed && (
                <button
                  type="button"
                  onClick={toggleOutliner}
                  title="Show Outliner (⌘B)"
                  className="absolute top-12 left-3 z-30 p-1.5 rounded-md bg-[#0f1014]/90 border border-[#1f222b] text-[#9296a6] hover:text-[#ededf2] hover:border-[#3d8ef7]/50 shadow-xl transition-all"
                >
                  <PanelLeft className="w-3.5 h-3.5" />
                </button>
              )}

              {inspectorCollapsed && (
                <button
                  type="button"
                  onClick={toggleInspector}
                  title="Show Inspector (⌘I)"
                  className="absolute top-12 right-3 z-30 p-1.5 rounded-md bg-[#0f1014]/90 border border-[#1f222b] text-[#9296a6] hover:text-[#ededf2] hover:border-[#3d8ef7]/50 shadow-xl transition-all"
                >
                  <PanelRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Right Inspector */}
            {!inspectorCollapsed && (
              <div className="w-72 shrink-0 h-full border-l border-[#1f222b] bg-[#101217] overflow-hidden">
                <EntityInspector />
              </div>
            )}
          </div>
        )}

        {/* ── Sleek 28px Bottom Docking Bar (Always Visible) ───────── */}
        <div className="h-7 px-2.5 flex items-center justify-between border-t border-[#1f222b] bg-[#0c0d11] shrink-0 text-[11px] font-mono select-none">
          {/* Left: Tab Triggers with click-to-expand */}
          <div className="flex items-center gap-1">
            {bottomTabs.map((tab) => {
              const isActive = bottomDrawerOpen && bottomDrawerTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => toggleBottomDrawer(tab.id)}
                  className={`flex items-center gap-1.5 px-2 py-0.5 rounded transition-colors ${
                    isActive
                      ? 'bg-[#1f222b] text-[#3d8ef7] font-bold'
                      : 'text-[#54596b] hover:text-[#ededf2] hover:bg-[#14161f]'
                  }`}
                >
                  {tab.icon}
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Right: Quick Engine Status & Toggle Chevron */}
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-[#54596b] hidden sm:inline num-tabular">
              VRAM: 6.3/8.0 GB • OptiX RTX • 60 FPS
            </span>
            <button
              type="button"
              onClick={() => toggleBottomDrawer()}
              title={bottomDrawerOpen ? 'Collapse Drawer (⌘J)' : 'Expand Drawer (⌘J)'}
              className="flex items-center gap-1 text-[10px] text-[#9296a6] hover:text-[#ededf2] p-0.5"
            >
              <span>{bottomDrawerOpen ? 'Collapse' : 'Expand'}</span>
              {bottomDrawerOpen ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronUp className="w-3 h-3" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
