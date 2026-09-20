'use client';

import React from 'react';
import { useREStore } from '@/store/re-store';
import type { GlobalMode, SpatialScale } from '@/types/reality-engine';
import {
  Search,
  Bell,
  Settings,
  Globe2,
  ChevronDown,
  Layers,
  Sparkles,
} from 'lucide-react';

const GLOBAL_MODES: Array<{ mode: GlobalMode; label: string }> = [
  { mode: 'EXPLORE', label: 'EXPLORE' },
  { mode: 'INSPECT', label: 'INSPECT' },
  { mode: 'EDIT', label: 'EDIT' },
  { mode: 'CAPTURE', label: 'CAPTURE' },
  { mode: 'BUILD', label: 'BUILD' },
  { mode: 'REVIEW', label: 'REVIEW' },
];

const SPATIAL_SCALES: Array<{ scale: SpatialScale; label: string }> = [
  { scale: 'ROOM', label: 'ROOM' },
  { scale: 'BUILDING', label: 'BUILDING' },
  { scale: 'STREET', label: 'STREET' },
  { scale: 'PLOT', label: 'PLOT' },
  { scale: 'BLOCK', label: 'BLOCK' },
  { scale: 'MULTI-BLOCK', label: 'MULTI-BLOCK' },
  { scale: 'LOCALITY', label: 'LOCALITY' },
  { scale: 'WARD', label: 'WARD' },
  { scale: 'DISTRICT', label: 'DISTRICT' },
  { scale: 'CITY', label: 'CITY' },
];

export default function TitleBar() {
  const globalMode = useREStore((s) => s.globalMode);
  const setGlobalMode = useREStore((s) => s.setGlobalMode);
  const spatialScale = useREStore((s) => s.spatialScale);
  const setSpatialScale = useREStore((s) => s.setSpatialScale);
  const setCommandPaletteOpen = useREStore((s) => s.setCommandPaletteOpen);
  const setActiveWorkspace = useREStore((s) => s.setActiveWorkspace);
  const notifications = useREStore((s) => s.notifications);
  const activeWorldVersion = useREStore((s) => s.activeWorldVersion);
  const worldVersions = useREStore((s) => s.worldVersions);
  // Real mounted version from the backend store; no invented world names.
  const activeWorld = worldVersions.find((w) => w.version_id === activeWorldVersion);
  const activeWorldLabel = activeWorld
    ? `${activeWorld.name || activeWorld.world_id} | ${activeWorld.version_id}`
    : activeWorldVersion || 'No world mounted';

  return (
    <header className="flex flex-col shrink-0 select-none bg-[#0a0b0e] border-b border-[#1f222b] text-[#ededf2]">
      {/* ── Main Top Bar (44px) ── */}
      <div className="h-11 px-3 flex items-center justify-between gap-3">
        {/* LEFT: Branding & Logo */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="w-6 h-6 rounded-lg bg-[#00e5ff]/10 border border-[#00e5ff]/40 flex items-center justify-center text-[#00e5ff] shadow-[0_0_12px_rgba(0,229,255,0.2)]">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-xs tracking-wider text-[#f0f1f6] font-sans">
                Reality Engine
              </span>
              <span className="text-[9px] font-mono px-1 rounded bg-[#151821] text-[#38bdf8] border border-[#222838]">
                v2.3.1
              </span>
            </div>
            <span className="text-[9px] text-[#54596b] tracking-tight hidden sm:inline">
              Real places. Persistent digital worlds.
            </span>
          </div>
        </div>

        {/* CENTER: Global Modes Switcher */}
        <div className="flex items-center gap-1 bg-[#101217] p-1 rounded-lg border border-[#1f222b] shadow-inner">
          <span className="text-[9px] font-mono font-bold tracking-wider text-[#54596b] px-2 uppercase hidden lg:inline">
            Global Modes:
          </span>
          {GLOBAL_MODES.map((item) => {
            const isActive = globalMode === item.mode;
            return (
              <button
                key={item.mode}
                type="button"
                onClick={() => setGlobalMode(item.mode)}
                className={`px-2.5 py-1 rounded text-[11px] font-mono font-bold transition-all ${
                  isActive
                    ? 'bg-[#00e5ff] text-[#08090b] shadow-[0_0_10px_rgba(0,229,255,0.35)]'
                    : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {/* RIGHT: Search, World Selector, Status & Profile */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Command Palette Trigger */}
          <button
            type="button"
            onClick={() => setCommandPaletteOpen(true)}
            className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#14161f] hover:bg-[#1a1d28] border border-[#1f222b] text-xs text-[#9296a6] transition-all"
            title="Search locations, objects, or commands (Ctrl+K)"
          >
            <Search className="w-3.5 h-3.5 text-[#54596b]" />
            <span className="text-[11px] hidden md:inline text-[#54596b]">Search…</span>
            <kbd className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-white/5 text-[#54596b]">
              Ctrl+K
            </kbd>
          </button>

          {/* Active World Selector */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#14161f] border border-[#1f222b] text-[11px] font-mono text-[#f0f1f6]">
            <Globe2 className="w-3.5 h-3.5 text-[#00e5ff]" />
            <span className="max-w-[120px] truncate">{activeWorldLabel}</span>
            <ChevronDown className="w-3 h-3 text-[#54596b]" />
          </div>

          {/* Engine Connectivity Status Badge */}
          <div
            title="Reality Engine Daemon: Local GPU Worker (CUDA 12.4 active)"
            className="hidden xl:flex items-center gap-1.5 px-2 py-1 rounded-lg bg-[#101915] border border-[#1a3826] text-[10px] font-mono text-[#2ecc71]"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71] animate-pulse" />
            <span className="font-semibold">ONLINE</span>
          </div>

          {/* Notifications Bell */}
          <button
            type="button"
            onClick={() => setCommandPaletteOpen(true)}
            className="relative p-1.5 rounded-lg text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
            title={`${notifications.length} notifications`}
          >
            <Bell className="w-4 h-4" />
            {notifications.length > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-[#00e5ff] shadow-[0_0_6px_#00e5ff]" />
            )}
          </button>

          {/* Settings */}
          <button
            type="button"
            onClick={() => setActiveWorkspace('settings')}
            className="p-1.5 rounded-lg text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
            title="Workstation Settings"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* User Avatar */}
          <div
            className="w-7 h-7 rounded-full bg-[#00e5ff]/20 border border-[#00e5ff]/40 flex items-center justify-center text-[10px] font-bold text-[#00e5ff] font-mono"
            title="Logged in as Shresth Bhati"
          >
            SB
          </div>
        </div>
      </div>

      {/* ── Secondary Ribbon: Global Spatial Scale Selector (28px) ── */}
      <div className="h-7 px-3 bg-[#0e1015] border-t border-[#171922] flex items-center justify-between text-[10px] font-mono overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max">
          <span className="text-[#54596b] uppercase font-bold tracking-wider mr-1">
            Global Spatial Scale:
          </span>
          {SPATIAL_SCALES.map((item, idx) => {
            const isActive = spatialScale === item.scale;
            return (
              <React.Fragment key={item.scale}>
                <button
                  type="button"
                  onClick={() => setSpatialScale(item.scale)}
                  className={`px-2 py-0.5 rounded transition-all font-semibold ${
                    isActive
                      ? 'bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/50 shadow-sm'
                      : 'text-[#9296a6] hover:text-[#f0f1f6] border border-transparent'
                  }`}
                >
                  {item.label}
                </button>
                {idx < SPATIAL_SCALES.length - 1 && (
                  <span className="text-[#2b3040] select-none">→</span>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Benchmark Corpus Shortcut */}
        <div className="flex items-center gap-2 pl-4 shrink-0">
          <button
            type="button"
            onClick={() => setActiveWorkspace('benchmarks')}
            className="text-[10px] font-mono text-[#38bdf8] hover:underline flex items-center gap-1"
          >
            <Sparkles className="w-3 h-3" />
            <span>40 Benchmark Targets</span>
          </button>
        </div>
      </div>
    </header>
  );
}
