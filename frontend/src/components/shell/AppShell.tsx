'use client';

import { useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useREStore } from '@/store/re-store';
import type { WorkspaceId } from '@/types/reality-engine';
import TitleBar from './TitleBar';
import WorkspaceSwitcher from './WorkspaceSwitcher';
import StatusBar from './StatusBar';
import CommandPalette from './CommandPalette';

// ─── Lazy workspace imports ───────────────────────────────────────────────────

const CaptureWorkspace = dynamic(
  () => import('@/apps/capture/CaptureApp'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Capture" /> },
);

const BenchmarkExplorer = dynamic(
  () => import('@/components/workspaces/benchmarks/BenchmarkExplorer'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Benchmarks" /> },
);

const StudioWorkspace = dynamic(
  () => import('@/apps/viewer/ViewerApp'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Studio & Viewer" /> },
);

const BuildWorkspace = dynamic(
  () => import('@/components/workspaces/build/BuildWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Build" /> },
);

const LoaderWorkspace = dynamic(
  () => import('@/apps/loader'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Loader" /> },
);

const CityBuilderWorkspace = dynamic(
  () => import('@/components/workspaces/city/CityBuilderWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="City" /> },
);

const EvidenceWorkspace = dynamic(
  () => import('@/components/workspaces/evidence/EvidenceWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Evidence" /> },
);

const SimulationWorkspace = dynamic(
  () => import('@/components/workspaces/simulation/SimulationWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Simulate" /> },
);

const DiagnosticsWorkspace = dynamic(
  () => import('@/components/workspaces/diagnostics/DiagnosticsWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Diagnostics" /> },
);

const SettingsWorkspace = dynamic(
  () => import('@/components/workspaces/settings/SettingsWorkspace'),
  { ssr: false, loading: () => <WorkspacePlaceholder label="Settings" /> },
);

// Placeholder for workspaces loading
function WorkspacePlaceholder({ label }: { label: string }) {
  return (
    <div
      className="flex flex-col items-center justify-center flex-1 select-none"
      style={{ background: 'var(--re-bg-base)' }}
    >
      <div className="w-4 h-4 border-2 border-[#3d8ef7] border-t-transparent rounded-full animate-spin mb-3" />
      <span
        style={{
          fontSize: '11px',
          letterSpacing: '0.12em',
          color: 'var(--re-text-secondary)',
          fontFamily: 'JetBrains Mono, monospace',
        }}
      >
        LOADING {label.toUpperCase()} WORKSPACE...
      </span>
    </div>
  );
}

// ─── Workspace registry ───────────────────────────────────────────────────────

function ActiveWorkspace({ id }: { id: WorkspaceId }) {
  switch (id) {
    case 'capture':
      return <CaptureWorkspace />;
    case 'loader':
      return <LoaderWorkspace />;
    case 'studio':
      return <StudioWorkspace />;
    case 'benchmarks':
      return <BenchmarkExplorer />;
    case 'build':
      return <BuildWorkspace />;
    case 'city':
      return <CityBuilderWorkspace />;
    case 'evidence':
      return <EvidenceWorkspace />;
    case 'simulation':
      return <SimulationWorkspace />;
    case 'diagnostics':
      return <DiagnosticsWorkspace />;
    case 'settings':
      return <SettingsWorkspace />;
    default: {
      const _exhaustive: never = id;
      return <WorkspacePlaceholder label={String(_exhaustive)} />;
    }
  }
}

// ─── CSS vars injection ───────────────────────────────────────────────────────
// Injects the RE design tokens so they are available globally.
// In production these should live in globals.css; this ensures they're always
// present regardless of import order.
const RE_CSS_VARS = `
  :root {
    --re-bg-base:      #0d0d0f;
    --re-bg-elevated:  #121215;
    --re-bg-surface:   #17171c;
    --re-accent:       #3d8ef7;
    --re-text-primary:   #e8e8f0;
    --re-text-secondary: #9898b0;
    --re-text-tertiary:  #5c5c78;
    --re-border-default: #272733;
  }
`;

// ─── Root shell ───────────────────────────────────────────────────────────────

export default function AppShell() {
  const loadMockData = useREStore((s) => s.loadMockData);
  const activeWorkspace = useREStore((s) => s.activeWorkspace);

  // Load mock data once on mount
  useEffect(() => {
    loadMockData();
  }, [loadMockData]);

  return (
    <>
      {/* Inject design tokens */}
      <style>{RE_CSS_VARS}</style>

      <div
        className="flex flex-col"
        style={{
          width: '100vw',
          height: '100vh',
          overflow: 'hidden',
          background: 'var(--re-bg-base)',
          fontFamily: 'Inter, system-ui, sans-serif',
        }}
      >
        {/* ── Top chrome */}
        <TitleBar />
        <WorkspaceSwitcher />

        {/* ── Workspace area */}
        <main className="flex flex-1 overflow-hidden">
          <ActiveWorkspace id={activeWorkspace} />
        </main>

        {/* ── Bottom chrome */}
        <StatusBar />

        {/* ── Overlays */}
        <CommandPalette />
      </div>
    </>
  );
}
