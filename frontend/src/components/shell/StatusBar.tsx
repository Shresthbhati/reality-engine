'use client';

import React from 'react';
import { useREStore } from '@/store/re-store';

function Divider() {
  return (
    <span
      className="shrink-0"
      style={{
        width: '1px',
        height: '12px',
        background: 'var(--re-border-default, #1f222b)',
        margin: '0 8px',
      }}
    />
  );
}

function Seg({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="flex items-center gap-1.5 whitespace-nowrap"
      style={{ color: 'var(--re-text-tertiary, #54596b)' }}
    >
      {children}
    </span>
  );
}

export default function StatusBar() {
  const world = useREStore((s) => s.world);
  const entities = useREStore((s) => s.entities);
  const sessions = useREStore((s) => s.sessions);
  const selection = useREStore((s) => s.selection);
  const computeMetrics = useREStore((s) => s.computeMetrics);
  const globalMode = useREStore((s) => s.globalMode);
  const spatialScale = useREStore((s) => s.spatialScale);
  const toggleBottomDrawer = useREStore((s) => s.toggleBottomDrawer);
  const bottomDrawerOpen = useREStore((s) => s.bottomDrawerOpen);

  const backendConnected = useREStore((s) => s.backendConnected);
  const loadedFromBackend = useREStore((s) => s.loadedFromBackend);
  const activeWorldVersion = useREStore((s) => s.activeWorldVersion);

  const entityCount = entities.size;
  const sessionCount = sessions.length;
  const gpuPct = computeMetrics.gpuUsage;
  const isGpuWarning = gpuPct > 90;

  const focusedEntity =
    selection.focusedEntityId != null
      ? entities.get(selection.focusedEntityId)
      : undefined;

  const statusColor = !backendConnected && !loadedFromBackend
    ? '#e54d4d' // Red: Offline
    : loadedFromBackend
    ? '#2ecc71' // Green: Real backend mounted
    : '#f5a623'; // Amber: Demo corpus active

  return (
    <footer
      className="flex items-center justify-between shrink-0 px-3 select-none bg-[#0a0b0e] border-t border-[#1f222b] font-mono text-[10px] text-[#9296a6]"
      style={{ height: '24px' }}
      aria-label="Application Status Bar"
    >
      {/* ── LEFT SEGMENTS ── */}
      <div className="flex items-center">
        {/* RE Badge */}
        <Seg>
          <span className="font-bold text-[#f0f1f6]">RE</span>
          <span className="text-[#54596b]">v2.3.1</span>
        </Seg>

        <Divider />

        {/* Global Mode & Spatial Scale */}
        <Seg>
          <span className="text-[#54596b]">MODE:</span>
          <span className="text-[#00e5ff] font-bold">{globalMode}</span>
          <span className="text-[#54596b] mx-0.5">•</span>
          <span className="text-[#54596b]">SCALE:</span>
          <span className="text-[#38bdf8] font-bold">{spatialScale}</span>
        </Seg>

        <Divider />

        {/* World Status & Backend Link */}
        <Seg>
          <span
            className="rounded-full shrink-0"
            style={{ width: '6px', height: '6px', background: statusColor }}
          />
          <span className="font-semibold text-[#ededf2]">
            {loadedFromBackend ? `MOUNTED [${activeWorldVersion ?? 'v1'}]` : 'DEMO CORPUS'}
          </span>
        </Seg>

        <Divider />

        <Seg>
          <span>{entityCount} entities</span>
          <span className="text-[#54596b]">·</span>
          <span>{sessionCount} sessions</span>
        </Seg>
      </div>

      {/* ── CENTER SEGMENT ── */}
      <div className="flex items-center justify-center truncate px-2 text-[#9296a6]">
        {focusedEntity != null ? (
          <span className="flex items-center gap-1.5 truncate">
            <span className="text-[#54596b] text-[9px] uppercase tracking-wider">
              {focusedEntity.type}
            </span>
            <span className="text-[#1f222b]">›</span>
            <span className="text-[#f0f1f6] font-semibold truncate">
              {focusedEntity.name}
            </span>
          </span>
        ) : (
          <span className="text-[#54596b] truncate">
            {world?.name ?? 'Reality Engine Spatial Workstation'}
          </span>
        )}
      </div>

      {/* ── RIGHT SEGMENTS ── */}
      <div className="flex items-center">
        {/* Dock Drawer Button */}
        <button
          type="button"
          onClick={() => toggleBottomDrawer()}
          className={`px-1.5 py-0.5 rounded text-[9px] font-mono transition-colors ${
            bottomDrawerOpen
              ? 'bg-[#00e5ff]/15 text-[#00e5ff] border border-[#00e5ff]/30'
              : 'text-[#54596b] hover:text-[#ededf2]'
          }`}
          title="Toggle Contextual Dock (⌘J)"
        >
          DOCK: {bottomDrawerOpen ? 'OPEN' : 'COLLAPSED'}
        </button>

        <Divider />

        {/* GPU Metric */}
        <Seg>
          <span style={{ color: isGpuWarning ? '#f5a623' : '#54596b' }}>
            GPU
          </span>
          <span
            className="font-num"
            style={{
              color: isGpuWarning ? '#f5a623' : '#9296a6',
              fontWeight: isGpuWarning ? 700 : 400,
            }}
          >
            {gpuPct}%
          </span>
          {isGpuWarning && <span className="text-[#f5a623]">⚠</span>}
        </Seg>

        <Divider />

        {/* Spatial Coordinates & Frame */}
        <Seg>
          <span className="text-[#00e5ff] font-semibold">
            {loadedFromBackend ? 'WORLDIR' : 'EPSG:32645'}
          </span>
          <span className="text-[#1f222b] mx-0.5">·</span>
          <span className="font-num">{loadedFromBackend ? 'METRIC_FRAME' : '22.5721° N'}</span>
          <span className="text-[#1f222b] mx-0.5">·</span>
          <span className="font-num">{loadedFromBackend ? 'SURVEY' : '88.3639° E'}</span>
        </Seg>

        <Divider />

        {/* FPS Counter */}
        <Seg>
          <span className="text-[#2ecc71] font-bold font-num">60</span>
          <span>fps</span>
        </Seg>
      </div>
    </footer>
  );
}
