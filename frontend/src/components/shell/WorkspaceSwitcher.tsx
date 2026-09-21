'use client';

import React, { useCallback, useRef } from 'react';
import { useREStore } from '@/store/re-store';
import type { WorkspaceId } from '@/types/reality-engine';
import {
  Camera,
  ArrowDownToLine,
  Box,
  Zap,
  Building2,
  Cpu,
  Settings,
  Workflow,
  ShieldCheck,
} from 'lucide-react';

interface TabItem {
  id: WorkspaceId;
  label: string;
  group: 'CORE' | 'DOWNSTREAM' | 'SYSTEM';
  badge?: number | string;
  icon: React.ReactNode;
}

const TABS: TabItem[] = [
  // ── 1. Core Reality Engine Applications (Canonical Pipeline)
  { id: 'capture', label: 'Capture', group: 'CORE', icon: <Camera className="w-3.5 h-3.5" /> },
  { id: 'loader', label: 'Loader (Ingest)', group: 'CORE', badge: 4, icon: <ArrowDownToLine className="w-3.5 h-3.5" /> },
  { id: 'build', label: 'Build (Pipeline)', group: 'CORE', badge: '10 Stg', icon: <Workflow className="w-3.5 h-3.5 text-[#3d8ef7]" /> },
  { id: 'studio', label: 'Studio (3D)', group: 'CORE', icon: <Box className="w-3.5 h-3.5" /> },
  { id: 'evidence', label: 'Evidence', group: 'CORE', icon: <ShieldCheck className="w-3.5 h-3.5 text-[#2ecc71]" /> },
  { id: 'benchmarks', label: 'Benchmarks', group: 'CORE', badge: 40, icon: <Zap className="w-3.5 h-3.5 text-amber-400" /> },

  // ── 2. Downstream Consumers (Consuming WorldIR)
  { id: 'city', label: 'City Builder', group: 'DOWNSTREAM', icon: <Building2 className="w-3.5 h-3.5" /> },

  // ── 3. System & Diagnostics
  { id: 'diagnostics', label: 'Diagnostics', group: 'SYSTEM', icon: <Cpu className="w-3.5 h-3.5" /> },
  { id: 'settings', label: 'Settings', group: 'SYSTEM', icon: <Settings className="w-3.5 h-3.5" /> },
];

export default function WorkspaceSwitcher() {
  const activeWorkspace = useREStore((s) => s.activeWorkspace);
  const setActiveWorkspace = useREStore((s) => s.setActiveWorkspace);
  const densityMode = useREStore((s) => s.densityMode);
  const setDensityMode = useREStore((s) => s.setDensityMode);
  const tabsRef = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        const next = (index + 1) % TABS.length;
        tabsRef.current[next]?.focus();
        setActiveWorkspace(TABS[next].id);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const prev = (index - 1 + TABS.length) % TABS.length;
        tabsRef.current[prev]?.focus();
        setActiveWorkspace(TABS[prev].id);
      }
    },
    [setActiveWorkspace],
  );

  return (
    <nav
      role="tablist"
      aria-label="Workspace switcher"
      className="flex items-center justify-between shrink-0 px-2 select-none"
      style={{
        height: '32px',
        background: 'var(--re-bg-elevated)',
        borderBottom: '1px solid var(--re-border-default)',
      }}
    >
      <div className="flex items-center h-full gap-0.5">
        {/* Core Platform Header Tag */}
        <span className="text-[9px] font-mono font-bold tracking-wider text-[#54596b] px-2 uppercase">
          Core Suite:
        </span>

        {TABS.map((tab, i) => {
          const isActive = tab.id === activeWorkspace;
          const isDownstream = tab.group === 'DOWNSTREAM';
          const isSystem = tab.group === 'SYSTEM';

          return (
            <React.Fragment key={tab.id}>
              {/* Visual Divider before Downstream Consumers */}
              {tab.id === 'city' && (
                <div className="flex items-center mx-1.5 h-4 text-[#54596b]">
                  <span className="w-px h-full bg-[#1f222b] mr-2" />
                  <span className="text-[9px] font-mono tracking-wider uppercase">
                    WorldIR Consumers:
                  </span>
                </div>
              )}

              {/* Visual Divider before System Tools */}
              {tab.id === 'diagnostics' && (
                <div className="flex items-center mx-1.5 h-4 text-[#54596b]">
                  <span className="w-px h-full bg-[#1f222b] mr-2" />
                  <span className="text-[9px] font-mono tracking-wider uppercase">
                    System:
                  </span>
                </div>
              )}

              <button
                ref={(el) => {
                  tabsRef.current[i] = el;
                }}
                role="tab"
                aria-selected={isActive}
                type="button"
                onClick={() => setActiveWorkspace(tab.id)}
                onKeyDown={(e) => handleKeyDown(e, i)}
                className={`relative flex items-center gap-1.5 px-2.5 h-7 rounded text-[11px] font-medium transition-colors focus:outline-none ${
                  isActive
                    ? 'bg-[#1a1d26] text-[#3d8ef7] font-semibold border border-[#3d8ef7]/35 shadow-sm'
                    : isDownstream
                    ? 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 border border-transparent'
                    : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 border border-transparent'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>

                {tab.badge !== undefined && (
                  <span
                    className={`flex items-center justify-center rounded-full px-1.5 text-[9px] font-mono font-bold leading-none ${
                      isActive
                        ? 'bg-[#3d8ef7]/20 text-[#3d8ef7]'
                        : 'bg-white/5 text-[#9296a6]'
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            </React.Fragment>
          );
        })}
      </div>

      {/* Right Density Selector */}
      <div className="flex items-center gap-1 text-[10px] font-mono text-[#54596b]">
        <span>DENSITY:</span>
        {(['comfortable', 'compact', 'dense'] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            onClick={() => setDensityMode(mode)}
            className={`px-1.5 py-0.5 rounded uppercase transition-colors ${
              densityMode === mode
                ? 'bg-[#1f222b] text-[#3d8ef7] font-bold'
                : 'text-[#54596b] hover:text-[#9296a6]'
            }`}
          >
            {mode.slice(0, 4)}
          </button>
        ))}
      </div>
    </nav>
  );
}
