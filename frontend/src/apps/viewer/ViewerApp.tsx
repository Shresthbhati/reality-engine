'use client';

/**
 * ViewerApp — Reality Engine Extreme-Fidelity Reality Reconstruction Studio & Viewer.
 * 
 * Core Capabilities:
 * 1. Multi-Scale Reality Navigation:
 *    WORLD -> SITE -> BUILDING -> STRUCTURE -> FACADE -> COMPONENT -> DETAIL -> MICRO DETAIL
 * 2. 8 Reconstruction View Modes:
 *    WORLD, GEOMETRY, DETAIL, TEXTURE, SEMANTICS, EVIDENCE, CONFIDENCE, PROVENANCE
 * 3. Spatial Detail Coverage Overlay:
 *    Fine-grained hierarchical coverage percentages (e.g. Building 87%, Facade 94%, Entrance 99%, Ornament 42%)
 * 4. Evidence-to-Detail Traceability Inspector:
 *    Inspects the exact chain: Detail -> Observations -> Registered Cameras -> Passes -> Depth -> Mesh Fusion
 * 5. 3D Viewport Hero Workspace with 1-click collapsible panels (⌘B, ⌘I, ⌘J)
 * 6. Direct access to the 40-target architectural benchmark validation suite.
 */

import React, { useState, useEffect } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import { useREStore, ScaleLevel, ReconstructionViewMode } from '@/store/re-store';
import { ViewportToolbar } from '@/components/workspaces/studio/ViewportToolbar';
import { WorldOutliner } from '@/components/workspaces/studio/WorldOutliner';
import { Viewport3D } from '@/components/workspaces/studio/Viewport3D';
import { EntityInspector } from '@/components/workspaces/studio/EntityInspector';
import EvidencePanel from '@/components/workspaces/studio/EvidencePanel';
import WorldNavPanel from '@/components/shell/WorldNavPanel';
import ContextInspectorPanel from '@/components/shell/ContextInspectorPanel';
import ContextualBottomDrawer from '@/components/shell/ContextualBottomDrawer';
import {
  Activity,
  Layers,
  Terminal,
  Cpu,
  Clock,
  PanelLeft,
  PanelRight,
  Sparkles,
  Search,
  Eye,
  ChevronRight,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Camera,
  Globe,
  X,
  ExternalLink,
  Sliders,
} from 'lucide-react';

export default function ViewerApp() {
  const {
    outlinerCollapsed,
    toggleOutliner,
    inspectorCollapsed,
    toggleInspector,
    bottomDrawerOpen,
    bottomDrawerTab,
    setBottomDrawerTab,
    toggleBottomDrawer,
    activeScaleLevel,
    setScaleLevel,
    reconstructionViewMode,
    setReconstructionViewMode,
    spatialCoverageNodes,
    selectedDetailProvenance,
    setActiveWorkspace,
    selection,
    deselectAll,
    toggleEntityVisibility,
    isolateEntity,
    builds,
    loadWorldFromBackend,
  } = useREStore();

  useEffect(() => {
    loadWorldFromBackend();
  }, [loadWorldFromBackend]);

  const [showCoverageOverlay, setShowCoverageOverlay] = useState(false);
  const [showProvenanceDrawer, setShowProvenanceDrawer] = useState(false);
  const [leftTab, setLeftTab] = useState<'nav' | 'entities'>('nav');
  const [rightTab, setRightTab] = useState<'inspector' | 'details'>('inspector');

  // Global Keyboard Shortcuts (Professional Workstation Standards)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept when user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target as HTMLElement).isContentEditable
      ) {
        return;
      }

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleOutliner();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'i') {
        e.preventDefault();
        toggleInspector();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault();
        toggleBottomDrawer();
      } else if (e.key === 'Escape') {
        deselectAll();
      } else if (e.key.toLowerCase() === 'h' && !e.metaKey && !e.ctrlKey) {
        const selectedId = selection.selectedEntityIds[0];
        if (selectedId) {
          e.preventDefault();
          toggleEntityVisibility(selectedId);
        }
      } else if (e.key.toLowerCase() === 'i' && !e.metaKey && !e.ctrlKey) {
        const selectedId = selection.selectedEntityIds[0];
        if (selectedId) {
          e.preventDefault();
          isolateEntity(selectedId);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    toggleOutliner,
    toggleInspector,
    toggleBottomDrawer,
    deselectAll,
    toggleEntityVisibility,
    isolateEntity,
    selection.selectedEntityIds,
  ]);

  const scaleLevels: Array<{ level: ScaleLevel; label: string }> = [
    { level: 'WORLD', label: 'World' },
    { level: 'SITE', label: 'Site' },
    { level: 'BUILDING', label: 'Building' },
    { level: 'STRUCTURE', label: 'Structure' },
    { level: 'FACADE', label: 'Facade' },
    { level: 'COMPONENT', label: 'Component' },
    { level: 'DETAIL', label: 'Detail' },
    { level: 'MICRO_DETAIL', label: 'Micro Detail' },
  ];

  const viewModes: Array<{ mode: ReconstructionViewMode; label: string }> = [
    { mode: 'WORLD', label: 'World' },
    { mode: 'GEOMETRY', label: 'Geometry' },
    { mode: 'DETAIL', label: 'Detail' },
    { mode: 'TEXTURE', label: 'Texture' },
    { mode: 'SEMANTICS', label: 'Semantics' },
    { mode: 'EVIDENCE', label: 'Evidence' },
    { mode: 'CONFIDENCE', label: 'Confidence' },
    { mode: 'PROVENANCE', label: 'Provenance' },
  ];

  const activeJobCount = builds.length > 0 ? builds[0].stages.filter((s) => s.status === 'RUNNING').length || 3 : 3;

  const bottomTabs: Array<{
    id: typeof bottomDrawerTab;
    label: string;
    icon: React.ReactNode;
  }> = [
    { id: 'pipeline', label: 'Pipeline (10 Stg)', icon: <Clock className="w-3 h-3" /> },
    { id: 'jobs', label: `Jobs (${activeJobCount})`, icon: <Activity className="w-3 h-3" /> },
    { id: 'evidence', label: 'Evidence Lineage', icon: <Layers className="w-3 h-3" /> },
    { id: 'diagnostics', label: 'Diagnostics', icon: <Cpu className="w-3 h-3" /> },
    { id: 'console', label: 'Engine Console', icon: <Terminal className="w-3 h-3" /> },
  ];

  return (
    <div className="flex flex-col w-full h-full overflow-hidden bg-[#0a0b0e] text-[#ededf2] select-none font-sans">
      {/* ── 1. Extreme-Fidelity Multi-Scale & 8 View Modes Ribbon (42px) ── */}
      <div className="h-10.5 px-3 bg-[#0e1015] border-b border-[#1f222b] flex items-center justify-between gap-2 overflow-x-auto shrink-0 text-xs font-mono">
        {/* Scale Hierarchy Selector */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] uppercase text-[#54596b] mr-1">Scale:</span>
          {scaleLevels.map((s, idx) => {
            const active = activeScaleLevel === s.level;
            return (
              <React.Fragment key={s.level}>
                <button
                  type="button"
                  onClick={() => setScaleLevel(s.level)}
                  className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                    active
                      ? 'bg-[#3d8ef7] text-[#08090b] shadow-sm'
                      : 'bg-[#14161f] text-[#9296a6] hover:text-[#f0f1f6] border border-[#222633]'
                  }`}
                >
                  {s.label}
                </button>
                {idx < scaleLevels.length - 1 && (
                  <ChevronRight className="w-2.5 h-2.5 text-[#303545]" />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* 8 Reconstruction View Modes Segmented Control */}
        <div className="flex items-center gap-1 bg-[#14161f] p-0.5 rounded-lg border border-[#222633]">
          <span className="text-[10px] uppercase text-[#54596b] px-1.5 hidden sm:inline">View Mode:</span>
          {viewModes.map((v) => {
            const active = reconstructionViewMode === v.mode;
            return (
              <button
                key={v.mode}
                type="button"
                onClick={() => setReconstructionViewMode(v.mode)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                  active
                    ? 'bg-[#3d8ef7] text-[#08090b]'
                    : 'text-[#9296a6] hover:text-[#ededf2]'
                }`}
              >
                {v.label}
              </button>
            );
          })}
        </div>

        {/* Floating Overlays Triggers (Detail Coverage & Evidence Trace) */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowCoverageOverlay(!showCoverageOverlay)}
            className={`px-2 py-1 rounded text-[10px] flex items-center gap-1.5 border transition-all ${
              showCoverageOverlay
                ? 'bg-[#2ecc71]/20 border-[#2ecc71] text-[#2ecc71]'
                : 'bg-[#14161f] border-[#222633] text-[#9296a6] hover:text-[#f0f1f6]'
            }`}
          >
            <ShieldCheck className="w-3 h-3" />
            <span>Detail Coverage (%)</span>
          </button>

          <button
            type="button"
            onClick={() => setShowProvenanceDrawer(!showProvenanceDrawer)}
            className={`px-2 py-1 rounded text-[10px] flex items-center gap-1.5 border transition-all ${
              showProvenanceDrawer
                ? 'bg-[#a855f7]/20 border-[#a855f7] text-[#a855f7]'
                : 'bg-[#14161f] border-[#222633] text-[#9296a6] hover:text-[#f0f1f6]'
            }`}
          >
            <Sparkles className="w-3 h-3" />
            <span>Evidence Trace</span>
          </button>
        </div>
      </div>

      {/* ── 2. Viewport Toolbar (32px) ────────────────────────────────── */}
      <ViewportToolbar />

      {/* ── 3. Main Workstation Resizable Grid ─────────────────────────── */}
      <div className="flex-1 min-h-0 w-full flex flex-col overflow-hidden relative">
        {bottomDrawerOpen ? (
          <Group orientation="vertical">
            {/* Top Row: Navigation/Outliner | 3D Viewport | Context Inspector */}
            <Panel defaultSize={74} minSize={40}>
              <Group orientation="horizontal">
                {!outlinerCollapsed && (
                  <>
                    <Panel defaultSize={20} minSize={14} maxSize={35}>
                      <div className="flex flex-col h-full w-full bg-[#101217] overflow-hidden">
                        {/* Left Tab Switcher */}
                        <div className="h-6 px-2 bg-[#0c0d11] border-b border-[#1f222b] flex items-center gap-1 shrink-0 text-[10px] font-mono">
                          <button
                            type="button"
                            onClick={() => setLeftTab('nav')}
                            className={`px-2 py-0.5 rounded transition-all ${
                              leftTab === 'nav'
                                ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                                : 'text-[#54596b] hover:text-[#9296a6]'
                            }`}
                          >
                            WORLD NAV
                          </button>
                          <button
                            type="button"
                            onClick={() => setLeftTab('entities')}
                            className={`px-2 py-0.5 rounded transition-all ${
                              leftTab === 'entities'
                                ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                                : 'text-[#54596b] hover:text-[#9296a6]'
                            }`}
                          >
                            ENTITIES
                          </button>
                        </div>
                        <div className="flex-1 overflow-hidden">
                          {leftTab === 'nav' ? <WorldNavPanel /> : <WorldOutliner />}
                        </div>
                      </div>
                    </Panel>
                    <Separator className="w-1 bg-[#1f222b] hover:bg-[#00e5ff]/50 transition-colors cursor-col-resize" />
                  </>
                )}

                <Panel defaultSize={outlinerCollapsed && inspectorCollapsed ? 100 : 58} minSize={30}>
                  <div className="h-full w-full relative bg-[#08090b] overflow-hidden">
                    <Viewport3D />
                  </div>
                </Panel>

                {!inspectorCollapsed && (
                  <>
                    <Separator className="w-1 bg-[#1f222b] hover:bg-[#00e5ff]/50 transition-colors cursor-col-resize" />
                    <Panel defaultSize={22} minSize={16} maxSize={40}>
                      <div className="flex flex-col h-full w-full bg-[#101217] overflow-hidden">
                        {/* Right Tab Switcher */}
                        <div className="h-6 px-2 bg-[#0c0d11] border-b border-[#1f222b] flex items-center gap-1 shrink-0 text-[10px] font-mono">
                          <button
                            type="button"
                            onClick={() => setRightTab('inspector')}
                            className={`px-2 py-0.5 rounded transition-all ${
                              rightTab === 'inspector'
                                ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                                : 'text-[#54596b] hover:text-[#9296a6]'
                            }`}
                          >
                            CONTEXT INSPECTOR
                          </button>
                          <button
                            type="button"
                            onClick={() => setRightTab('details')}
                            className={`px-2 py-0.5 rounded transition-all ${
                              rightTab === 'details'
                                ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                                : 'text-[#54596b] hover:text-[#9296a6]'
                            }`}
                          >
                            ATTRIBUTES
                          </button>
                        </div>
                        <div className="flex-1 overflow-hidden">
                          {rightTab === 'inspector' ? <ContextInspectorPanel /> : <EntityInspector />}
                        </div>
                      </div>
                    </Panel>
                  </>
                )}
              </Group>
            </Panel>

            <Separator className="h-1 bg-[#1f222b] hover:bg-[#00e5ff]/50 transition-colors cursor-row-resize" />

            {/* Bottom Panel Drawer */}
            <Panel defaultSize={26} minSize={15} maxSize={50}>
              <div className="h-full w-full bg-[#101217] overflow-hidden flex flex-col">
                <ContextualBottomDrawer />
              </div>
            </Panel>
          </Group>
        ) : (
          // Maximized Viewport Layout (Bottom Drawer Collapsed)
          <div className="flex-1 w-full h-full min-h-0 flex overflow-hidden">
            {!outlinerCollapsed && (
              <div className="w-64 shrink-0 h-full border-r border-[#1f222b] bg-[#101217] overflow-hidden flex flex-col">
                <div className="h-6 px-2 bg-[#0c0d11] border-b border-[#1f222b] flex items-center gap-1 shrink-0 text-[10px] font-mono">
                  <button
                    type="button"
                    onClick={() => setLeftTab('nav')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftTab === 'nav'
                        ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                        : 'text-[#54596b] hover:text-[#9296a6]'
                    }`}
                  >
                    WORLD NAV
                  </button>
                  <button
                    type="button"
                    onClick={() => setLeftTab('entities')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      leftTab === 'entities'
                        ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                        : 'text-[#54596b] hover:text-[#9296a6]'
                    }`}
                  >
                    ENTITIES
                  </button>
                </div>
                <div className="flex-1 overflow-hidden">
                  {leftTab === 'nav' ? <WorldNavPanel /> : <WorldOutliner />}
                </div>
              </div>
            )}

            <div className="flex-1 h-full min-w-0 relative bg-[#08090b] overflow-hidden">
              <Viewport3D />

              {/* Restore Pill when Outliner is collapsed */}
              {outlinerCollapsed && (
                <button
                  type="button"
                  onClick={toggleOutliner}
                  title="Show Outliner (⌘B)"
                  className="absolute top-12 left-3 z-30 p-1.5 rounded-md bg-[#0f1014]/90 border border-[#1f222b] text-[#9296a6] hover:text-[#ededf2] hover:border-[#00e5ff]/50 shadow-xl transition-all"
                >
                  <PanelLeft className="w-3.5 h-3.5" />
                </button>
              )}

              {/* Restore Pill when Inspector is collapsed */}
              {inspectorCollapsed && (
                <button
                  type="button"
                  onClick={toggleInspector}
                  title="Show Inspector (⌘I)"
                  className="absolute top-12 right-3 z-30 p-1.5 rounded-md bg-[#0f1014]/90 border border-[#1f222b] text-[#9296a6] hover:text-[#ededf2] hover:border-[#00e5ff]/50 shadow-xl transition-all"
                >
                  <PanelRight className="w-3.5 h-3.5" />
                </button>
              )}

              {/* Active Scale & View Mode HUD Indicator on Viewport */}
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 pointer-events-none flex items-center gap-2 bg-black/70 backdrop-blur-md px-3 py-1 rounded-full border border-white/10 text-[10px] font-mono">
                <span className="text-[#00e5ff] font-bold">SCALE: {activeScaleLevel}</span>
                <span className="text-[#54596b]">•</span>
                <span className="text-[#2ecc71] font-bold">MODE: {reconstructionViewMode}</span>
              </div>
            </div>

            {!inspectorCollapsed && (
              <div className="w-80 shrink-0 h-full border-l border-[#1f222b] bg-[#101217] overflow-hidden flex flex-col">
                <div className="h-6 px-2 bg-[#0c0d11] border-b border-[#1f222b] flex items-center gap-1 shrink-0 text-[10px] font-mono">
                  <button
                    type="button"
                    onClick={() => setRightTab('inspector')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightTab === 'inspector'
                        ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                        : 'text-[#54596b] hover:text-[#9296a6]'
                    }`}
                  >
                    CONTEXT INSPECTOR
                  </button>
                  <button
                    type="button"
                    onClick={() => setRightTab('details')}
                    className={`px-2 py-0.5 rounded transition-all ${
                      rightTab === 'details'
                        ? 'bg-[#00e5ff]/20 text-[#00e5ff] font-bold'
                        : 'text-[#54596b] hover:text-[#9296a6]'
                    }`}
                  >
                    ATTRIBUTES
                  </button>
                </div>
                <div className="flex-1 overflow-hidden">
                  {rightTab === 'inspector' ? <ContextInspectorPanel /> : <EntityInspector />}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Floating Overlay 1: Spatial Detail Coverage Breakdown ────── */}
        {showCoverageOverlay && (
          <div className="absolute top-14 right-80 z-40 w-96 rounded-xl bg-[#101218]/95 backdrop-blur-xl border border-[#2ecc71]/40 shadow-2xl p-3.5 font-mono space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <div className="flex items-center gap-1.5 text-xs font-bold text-[#2ecc71]">
                <ShieldCheck className="w-4 h-4" />
                <span>Spatial Detail Coverage (%)</span>
              </div>
              <button
                type="button"
                onClick={() => setShowCoverageOverlay(false)}
                className="text-[#9296a6] hover:text-white"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="text-[10px] text-[#9296a6] leading-relaxed">
              Spatial completeness per architectural partition. Sub-millimeter evidence is evaluated independently from building massing.
            </div>

            {/* Hierarchical Coverage List */}
            <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
              <div className="p-2 rounded bg-[#151922] border border-[#222634] space-y-1">
                <div className="flex justify-between text-[11px] font-bold">
                  <span className="text-[#f0f1f6]">Victoria Memorial Edifice</span>
                  <span className="text-[#2ecc71]">87%</span>
                </div>
                <div className="text-[9px] text-[#9296a6]">Ground Resolution: 4.2 mm/px</div>
                <div className="h-1.5 w-full bg-[#1e2230] rounded-full overflow-hidden">
                  <div className="h-full bg-[#2ecc71]" style={{ width: '87%' }} />
                </div>
              </div>

              <div className="p-2 ml-3 rounded bg-[#151922] border border-[#222634] space-y-1">
                <div className="flex justify-between text-[11px] font-bold">
                  <span className="text-[#3d8ef7]">↳ North Facade</span>
                  <span className="text-[#2ecc71]">94%</span>
                </div>
                <div className="text-[9px] text-[#9296a6]">Ground Resolution: 2.1 mm/px</div>
                <div className="h-1.5 w-full bg-[#1e2230] rounded-full overflow-hidden">
                  <div className="h-full bg-[#2ecc71]" style={{ width: '94%' }} />
                </div>
              </div>

              <div className="p-2 ml-6 rounded bg-[#151922] border border-[#222634] space-y-1">
                <div className="flex justify-between text-[11px] font-bold">
                  <span className="text-[#3d8ef7]">↳ Central Portico & Columns</span>
                  <span className="text-[#2ecc71]">99%</span>
                </div>
                <div className="text-[9px] text-[#9296a6]">Ground Resolution: 1.2 mm/px</div>
                <div className="h-1.5 w-full bg-[#1e2230] rounded-full overflow-hidden">
                  <div className="h-full bg-[#2ecc71]" style={{ width: '99%' }} />
                </div>
              </div>

              <div className="p-2 ml-9 rounded bg-[#151922] border border-[#222634] space-y-1">
                <div className="flex justify-between text-[11px] font-bold">
                  <span className="text-[#a855f7]">↳ Carved Floral Relief</span>
                  <span className="text-[#2ecc71]">91%</span>
                </div>
                <div className="text-[9px] text-[#9296a6]">Ground Resolution: 0.35 mm/px</div>
                <div className="h-1.5 w-full bg-[#1e2230] rounded-full overflow-hidden">
                  <div className="h-full bg-[#2ecc71]" style={{ width: '91%' }} />
                </div>
              </div>

              <div className="p-2 ml-6 rounded bg-[#1a1417] border border-[#e74c3c]/40 space-y-1">
                <div className="flex justify-between text-[11px] font-bold">
                  <span className="text-[#fca5a5]">↳ Northwest Ornament</span>
                  <span className="text-[#e74c3c]">42%</span>
                </div>
                <div className="text-[9px] text-[#9296a6]">Ground Resolution: 8.5 mm/px (Sparse)</div>
                <div className="h-1.5 w-full bg-[#1e2230] rounded-full overflow-hidden">
                  <div className="h-full bg-[#e74c3c]" style={{ width: '42%' }} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Floating Overlay 2: Evidence-to-Detail Provenance Inspector ── */}
        {showProvenanceDrawer && selectedDetailProvenance && (
          <div className="absolute top-14 right-80 z-40 w-[420px] rounded-xl bg-[#101218]/95 backdrop-blur-xl border border-[#a855f7]/40 shadow-2xl p-4 font-mono space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <div className="flex items-center gap-1.5 text-xs font-bold text-[#a855f7]">
                <Sparkles className="w-4 h-4" />
                <span>Evidence-to-Detail Traceability</span>
              </div>
              <button
                type="button"
                onClick={() => setShowProvenanceDrawer(false)}
                className="text-[#9296a6] hover:text-white"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="p-2.5 rounded bg-[#151322] border border-[#a855f7]/30 space-y-1">
              <div className="text-xs font-bold text-[#f0f1f6]">
                {selectedDetailProvenance.detailName}
              </div>
              <div className="text-[10px] text-[#9296a6]">
                Component: {selectedDetailProvenance.parentComponent}
              </div>
              <div className="text-[10px] text-[#a855f7] font-bold">
                LOD: {selectedDetailProvenance.geometryLOD}
              </div>
            </div>

            {/* Traceability Chain */}
            <div className="space-y-1.5 text-[10px]">
              <div className="p-2 rounded bg-[#151922] border border-[#222634] flex items-center justify-between">
                <span className="text-[#9296a6]">Registered Cameras:</span>
                <span className="text-[#f0f1f6] font-bold">{selectedDetailProvenance.registeredCamerasCount} Cameras</span>
              </div>
              <div className="p-2 rounded bg-[#151922] border border-[#222634] flex items-center justify-between">
                <span className="text-[#9296a6]">Source Observations:</span>
                <span className="text-[#f0f1f6] font-bold">{selectedDetailProvenance.observationsCount} Keyframes</span>
              </div>
              <div className="p-2 rounded bg-[#151922] border border-[#222634] flex items-center justify-between">
                <span className="text-[#9296a6]">Ground Resolution:</span>
                <span className="text-[#2ecc71] font-bold">{selectedDetailProvenance.groundResolutionMm} mm / pixel</span>
              </div>
              <div className="p-2 rounded bg-[#151922] border border-[#222634] flex items-center justify-between">
                <span className="text-[#9296a6]">Confidence Score:</span>
                <span className="text-[#2ecc71] font-bold">{(selectedDetailProvenance.confidenceScore * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="space-y-1 text-[10px]">
              <div className="text-[#9296a6]">Reconstruction Passes Applied:</div>
              <div className="flex flex-wrap gap-1">
                {selectedDetailProvenance.passes.map((pass) => (
                  <span
                    key={pass}
                    className="px-1.5 py-0.5 rounded bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/30 text-[9px]"
                  >
                    {pass}
                  </span>
                ))}
              </div>
            </div>

            <div className="space-y-1 text-[10px] border-t border-white/10 pt-2">
              <div className="text-[#9296a6]">Dense Depth Method:</div>
              <div className="text-[#ededf2] font-mono text-[9px] bg-black/40 p-1.5 rounded">
                {selectedDetailProvenance.depthMethod}
              </div>
            </div>

            <div className="space-y-1 text-[10px]">
              <div className="text-[#9296a6]">Mesh Fusion Method:</div>
              <div className="text-[#ededf2] font-mono text-[9px] bg-black/40 p-1.5 rounded">
                {selectedDetailProvenance.fusionMethod}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── 4. Minimal Docking Tab Bar (28px) ─────────────────────────── */}
      <div className="h-7 border-t border-[#1f222b] bg-[#0d0e12] px-2 flex items-center justify-between shrink-0 select-none text-[11px] font-mono">
        <div className="flex items-center gap-1">
          {bottomTabs.map((tab) => {
            const isActive = bottomDrawerOpen && bottomDrawerTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => toggleBottomDrawer(tab.id)}
                className={`h-5.5 px-2.5 rounded flex items-center gap-1.5 text-[10px] font-medium transition-all ${
                  isActive
                    ? 'bg-[#1b1e27] text-[#3d8ef7] font-semibold border border-[#3d8ef7]/40 shadow-sm'
                    : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#15171e]'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-3 text-[10px] text-[#54596b]">
          <span className="hidden md:inline">⌘B: Outliner • ⌘I: Inspector • ⌘J: Dock</span>
          <button
            type="button"
            onClick={() => setActiveWorkspace('benchmarks')}
            className="text-[#3d8ef7] hover:underline flex items-center gap-1"
          >
            <Globe className="w-3 h-3" />
            <span>40 Benchmarks</span>
          </button>
        </div>
      </div>
    </div>
  );
}
