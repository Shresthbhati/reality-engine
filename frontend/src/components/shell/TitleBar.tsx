'use client';

import { useREStore } from '@/store/re-store';

export default function TitleBar() {
  const project = useREStore((s) => s.project);
  const world = useREStore((s) => s.world);
  const setActiveWorkspace = useREStore((s) => s.setActiveWorkspace);

  return (
    <header
      className="flex items-center justify-between shrink-0 px-3 select-none"
      style={{
        height: '40px',
        background: 'var(--re-bg-base)',
        borderBottom: '1px solid #1e1e28',
      }}
    >
      {/* LEFT — Logo */}
      <div className="flex items-center gap-2">
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path
            d="M7 1L9.5 5.5H13L9.5 8.5L10.5 13L7 10.5L3.5 13L4.5 8.5L1 5.5H4.5L7 1Z"
            fill="var(--re-accent)"
            stroke="var(--re-accent)"
            strokeWidth="0.5"
            strokeLinejoin="round"
          />
        </svg>
        <span
          className="font-semibold"
          style={{
            fontSize: '11px',
            letterSpacing: '0.15em',
            color: 'var(--re-text-primary)',
            fontFamily: 'Inter, sans-serif',
          }}
        >
          REALITY ENGINE
        </span>
      </div>

      {/* CENTER — Project / World */}
      <div
        className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5"
        style={{ fontSize: '13px' }}
      >
        <span style={{ color: 'var(--re-text-secondary)' }}>
          {project?.name ?? '—'}
        </span>
        <span style={{ color: 'var(--re-text-tertiary)' }}>/</span>
        <span style={{ color: 'var(--re-text-tertiary)' }}>
          {world?.name ?? '—'}
        </span>
        {world?.status === 'READY' && (
          <span
            className="ml-1.5 rounded-sm px-1 py-px"
            style={{
              fontSize: '9px',
              letterSpacing: '0.08em',
              fontWeight: 600,
              background: 'rgba(39,195,112,0.12)',
              color: '#27c370',
              border: '1px solid rgba(39,195,112,0.25)',
            }}
          >
            READY
          </span>
        )}
        <button
          type="button"
          onClick={() => setActiveWorkspace('benchmarks')}
          title="Browse 40 Architectural Benchmarks"
          className="ml-2 flex items-center gap-1 px-2 py-0.5 rounded bg-[#151720] hover:bg-[#1e2230] border border-[#1f222b] text-[10px] font-mono text-[#3d8ef7] transition-colors"
        >
          <span>Corpus: 40 Targets</span>
        </button>
      </div>

      {/* RIGHT — Controls */}
      <div className="flex items-center gap-2">
        {/* Engine Connectivity Status Badge */}
        <div
          title="Reality Engine Daemon: Local GPU Worker (CUDA 12.4 active)"
          className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#131720] border border-[#222838] text-[10px] font-mono text-[#2ecc71]"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71] animate-pulse" />
          <span className="font-semibold">ENGINE ONLINE</span>
        </div>

        {/* Task count badge */}
        <button
          type="button"
          onClick={() => setActiveWorkspace('build')}
          title="Open Active Build Pipeline Tasks"
          className="flex items-center gap-1.5 rounded px-2 py-0.5 transition-colors"
          style={{
            fontSize: '11px',
            color: 'var(--re-text-tertiary)',
            border: '1px solid var(--re-border-default)',
            background: 'transparent',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'var(--re-bg-surface)';
            (e.currentTarget as HTMLButtonElement).style.color = 'var(--re-text-secondary)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
            (e.currentTarget as HTMLButtonElement).style.color = 'var(--re-text-tertiary)';
          }}
          aria-label="3 background tasks"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
            <circle cx="5" cy="5" r="3.5" stroke="currentColor" strokeWidth="1.2" />
            <path d="M5 2.5V5L6.5 6.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
          <span>3 tasks</span>
        </button>

        {/* Settings gear */}
        <button
          type="button"
          onClick={() => setActiveWorkspace('settings')}
          title="Workstation Settings & Downstream Exporters"
          className="flex items-center justify-center rounded transition-colors"
          style={{
            width: '26px',
            height: '26px',
            color: 'var(--re-text-tertiary)',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'var(--re-bg-surface)';
            (e.currentTarget as HTMLButtonElement).style.color = 'var(--re-text-secondary)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
            (e.currentTarget as HTMLButtonElement).style.color = 'var(--re-text-tertiary)';
          }}
          aria-label="Settings"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <circle cx="7" cy="7" r="2.2" stroke="currentColor" strokeWidth="1.1" />
            <path
              d="M7 1.5v1M7 11.5v1M1.5 7h1M11.5 7h1M3.1 3.1l.7.7M10.2 10.2l.7.7M10.2 3.8l-.7.7M3.8 10.2l-.7.7"
              stroke="currentColor"
              strokeWidth="1.1"
              strokeLinecap="round"
            />
          </svg>
        </button>

        {/* User avatar */}
        <button
          type="button"
          className="flex items-center justify-center rounded-full shrink-0"
          style={{
            width: '24px',
            height: '24px',
            background: 'var(--re-accent)',
            color: '#fff',
            fontSize: '9px',
            fontWeight: 700,
            letterSpacing: '0.04em',
            border: 'none',
            cursor: 'pointer',
          }}
          aria-label="User menu"
        >
          SR
        </button>
      </div>
    </header>
  );
}
