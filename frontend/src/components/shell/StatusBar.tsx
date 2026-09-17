'use client';

import { useREStore } from '@/store/re-store';

function Divider() {
  return (
    <span
      className="shrink-0"
      style={{
        width: '1px',
        height: '12px',
        background: 'var(--re-border-default)',
        margin: '0 6px',
      }}
    />
  );
}

function Seg({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="flex items-center gap-1.5 whitespace-nowrap"
      style={{ color: 'var(--re-text-tertiary)' }}
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

  const entityCount = entities.size;
  const sessionCount = sessions.length;
  const gpuPct = computeMetrics.gpuUsage;
  const isGpuWarning = gpuPct > 90;

  const focusedEntity =
    selection.focusedEntityId != null
      ? entities.get(selection.focusedEntityId)
      : undefined;

  const statusColor =
    world?.status === 'READY'
      ? '#27c370'
      : world?.status === 'PROCESSING'
      ? '#f0a050'
      : world?.status === 'ERROR'
      ? '#e05050'
      : '#5c5c78';

  return (
    <footer
      className="flex items-center shrink-0 px-2 select-none"
      style={{
        height: '22px',
        background: 'var(--re-bg-base)',
        borderTop: '1px solid #1e1e28',
        fontFamily: "'JetBrains Mono', 'Geist Mono', monospace",
        fontSize: '10px',
      }}
    >
      {/* LEFT */}
      <div className="flex items-center">
        <Seg>
          <span
            className="rounded-full shrink-0"
            style={{ width: '6px', height: '6px', background: statusColor }}
          />
          <span style={{ letterSpacing: '0.06em', fontWeight: 500 }}>
            {world?.status ?? 'NO WORLD'}
          </span>
        </Seg>

        <Divider />

        <Seg>
          <span>{entityCount} entities</span>
        </Seg>

        <Divider />

        <Seg>
          <span>{sessionCount} sessions</span>
        </Seg>
      </div>

      {/* CENTER */}
      <div
        className="flex-1 flex items-center justify-center"
        style={{ color: 'var(--re-text-secondary)' }}
      >
        {focusedEntity != null && (
          <span className="flex items-center gap-1.5">
            <span
              style={{
                color: 'var(--re-text-tertiary)',
                fontSize: '9px',
                letterSpacing: '0.06em',
              }}
            >
              {focusedEntity.type}
            </span>
            <span style={{ color: 'var(--re-border-default)' }}>›</span>
            <span>{focusedEntity.name}</span>
          </span>
        )}
      </div>

      {/* RIGHT */}
      <div className="flex items-center">
        <Seg>
          <span style={{ color: isGpuWarning ? '#f0a050' : 'var(--re-text-tertiary)' }}>
            GPU
          </span>
          <span
            style={{
              color: isGpuWarning ? '#f0a050' : 'var(--re-text-secondary)',
              fontWeight: isGpuWarning ? 600 : 400,
            }}
          >
            {gpuPct}%
          </span>
          {isGpuWarning && <span style={{ color: '#f0a050' }}>⚠</span>}
        </Seg>

        <Divider />

        <Seg>
          <span className="text-[#3d8ef7] font-semibold">EPSG:32645</span>
          <span style={{ color: 'var(--re-border-default)', margin: '0 1px' }}>·</span>
          <span>22.5448° N</span>
          <span style={{ color: 'var(--re-border-default)', margin: '0 1px' }}>·</span>
          <span>88.3426° E</span>
          <span style={{ color: 'var(--re-border-default)', margin: '0 1px' }}>·</span>
          <span>+12.4m</span>
        </Seg>

        <Divider />

        <Seg>
          <span style={{ color: '#27c370', fontWeight: 500 }}>60</span>
          <span>fps</span>
        </Seg>
      </div>
    </footer>
  );
}
