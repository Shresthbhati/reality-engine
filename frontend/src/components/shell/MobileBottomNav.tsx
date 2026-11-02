'use client';

/**
 * Mobile replacement for WorkspaceSwitcher's desktop top-tab pattern.
 * Fixed bottom tab bar with the 4 most-used workspaces plus a "More"
 * sheet for the rest. Respects safe-area-inset-bottom and 44px+ touch
 * targets.
 */

import { useState } from 'react';
import { useREStore } from '@/store/re-store';
import type { WorkspaceId } from '@/types/reality-engine';
import { Box, Camera, ArrowDownToLine, Activity, Menu, Building2, Cpu, Settings, Workflow, ShieldCheck, Zap } from 'lucide-react';
import MobileSheet from './MobileSheet';

const PRIMARY: Array<{ id: WorkspaceId; label: string; icon: React.ReactNode }> = [
  { id: 'studio', label: 'Studio', icon: <Box className="w-5 h-5" /> },
  { id: 'capture', label: 'Capture', icon: <Camera className="w-5 h-5" /> },
  { id: 'loader', label: 'Loader', icon: <ArrowDownToLine className="w-5 h-5" /> },
  { id: 'evidence', label: 'Evidence', icon: <ShieldCheck className="w-5 h-5" /> },
];

const MORE: Array<{ id: WorkspaceId; label: string; icon: React.ReactNode }> = [
  { id: 'build', label: 'Build (Pipeline)', icon: <Workflow className="w-4 h-4" /> },
  { id: 'benchmarks', label: 'Benchmarks', icon: <Zap className="w-4 h-4" /> },
  { id: 'city', label: 'City Builder', icon: <Building2 className="w-4 h-4" /> },
  { id: 'simulation', label: 'Simulation', icon: <Activity className="w-4 h-4" /> },
  { id: 'diagnostics', label: 'Diagnostics', icon: <Cpu className="w-4 h-4" /> },
  { id: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> },
];

export default function MobileBottomNav() {
  const activeWorkspace = useREStore((s) => s.activeWorkspace);
  const setActiveWorkspace = useREStore((s) => s.setActiveWorkspace);
  const [moreOpen, setMoreOpen] = useState(false);

  const isMoreActive = MORE.some((m) => m.id === activeWorkspace);

  return (
    <>
      <nav
        aria-label="Workspace navigation"
        className="flex items-stretch shrink-0 select-none"
        style={{
          background: 'var(--re-bg-elevated)',
          borderTop: '1px solid var(--re-border-default)',
          paddingBottom: 'env(safe-area-inset-bottom, 0px)',
        }}
      >
        {PRIMARY.map((tab) => {
          const isActive = tab.id === activeWorkspace;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveWorkspace(tab.id)}
              aria-current={isActive ? 'page' : undefined}
              className="flex flex-col items-center justify-center gap-0.5 flex-1 min-h-[52px] text-[10px] font-medium transition-colors"
              style={{ color: isActive ? 'var(--re-accent-blue, #3d8ef7)' : 'var(--re-text-secondary)' }}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          );
        })}

        <button
          type="button"
          onClick={() => setMoreOpen(true)}
          aria-haspopup="dialog"
          aria-expanded={moreOpen}
          className="flex flex-col items-center justify-center gap-0.5 flex-1 min-h-[52px] text-[10px] font-medium transition-colors"
          style={{ color: isMoreActive ? 'var(--re-accent-blue, #3d8ef7)' : 'var(--re-text-secondary)' }}
        >
          <Menu className="w-5 h-5" />
          <span>More</span>
        </button>
      </nav>

      <MobileSheet open={moreOpen} onClose={() => setMoreOpen(false)} title="More Workspaces" heightFraction={0.6}>
        <div className="p-2">
          {MORE.map((tab) => {
            const isActive = tab.id === activeWorkspace;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => {
                  setActiveWorkspace(tab.id);
                  setMoreOpen(false);
                }}
                className={`flex items-center gap-3 w-full min-h-[48px] px-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-white/10 text-[#3d8ef7]' : 'text-[#e8e8f0] hover:bg-white/5'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </MobileSheet>
    </>
  );
}
