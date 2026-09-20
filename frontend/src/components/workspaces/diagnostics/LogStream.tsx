'use client';

/**
 * Log stream fed by the REAL reconstruction run records (the E2E
 * pipeline writes datasets/<dataset>/runs/run_*.json with measured stages, runtimes
 * and failure reasons; /api/run-log reads the newest one).
 *
 * The previous version GENERATED random fake log lines ("CUDA kernel
 * panic", invented timestamps) on the client — fabricated diagnostics.
 * Now: if the bridge has a run record, its measured stage timeline is
 * rendered; if not, the panel says LOGS: UNAVAILABLE and explains why.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export interface LogEntry {
  id: string;
  level: LogLevel;
  timestamp: string;
  module: string;
  message: string;
}

interface RunLogPayload {
  available: boolean;
  reason?: string;
  source?: string;
  entries?: Array<{
    level: LogLevel;
    timestamp: string;
    module: string;
    message: string;
  }>;
}

const ALL_LEVEL_OPTIONS: Array<'ALL' | LogLevel> = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const getLevelStyles = (level: LogLevel) => {
  switch (level) {
    case 'DEBUG':
      return 'text-[#5a5a7a] bg-[#1a1a22]';
    case 'INFO':
      return 'text-[#3d8ef7] bg-[#0d1525]';
    case 'WARNING':
      return 'text-yellow-400 bg-yellow-900/30';
    case 'ERROR':
      return 'text-red-400 bg-red-900/30';
    case 'CRITICAL':
      return 'text-red-300 bg-red-900/50 animate-pulse';
    default:
      return 'text-[#5a5a7a] bg-[#1a1a22]';
  }
};

export function LogStream() {
  const [payload, setPayload] = useState<RunLogPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedLevel, setSelectedLevel] = useState<'ALL' | LogLevel>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/run-log', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data: RunLogPayload) => {
        if (!cancelled) setPayload(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const logs: LogEntry[] = useMemo(
    () =>
      (payload?.entries ?? []).map((e, i) => ({
        id: `runlog-${i}`,
        level: e.level,
        timestamp: e.timestamp,
        module: e.module,
        message: e.message,
      })),
    [payload]
  );

  const filteredLogs = useMemo(() => {
    if (selectedLevel === 'ALL') return logs;
    return logs.filter((log) => log.level === selectedLevel);
  }, [logs, selectedLevel]);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  return (
    <div className="flex flex-col h-full bg-[#0a0a0e]">
      {/* Header */}
      <div className="h-8 border-b border-[#1e1e24] px-3 flex items-center gap-2 flex-shrink-0">
        <span className="uppercase tracking-widest text-[10px] text-[#5a5a7a]">Logs</span>
        <span className="font-mono text-[9px] bg-[#1e1e2e] px-1 rounded text-[#3d8ef7]">
          {filteredLogs.length}
        </span>
        {payload?.source && (
          <span className="font-mono text-[9px] text-[#5a5a7a] truncate max-w-[160px]" title={payload.source}>
            {payload.source}
          </span>
        )}
        <div className="flex-1" />

        <select
          value={selectedLevel}
          onChange={(e) => setSelectedLevel(e.target.value as 'ALL' | LogLevel)}
          className="bg-[#111118] border border-[#1e1e24] text-[#8a8aaa] text-[10px] font-mono outline-none px-1 py-0.5 rounded focus:border-[#3d8ef7]"
        >
          {ALL_LEVEL_OPTIONS.map((lvl) => (
            <option key={lvl} value={lvl}>{lvl}</option>
          ))}
        </select>

        <button
          onClick={() => setAutoScroll((a) => !a)}
          className={`text-[9px] font-mono transition-colors ${autoScroll ? 'text-[#3d8ef7]' : 'text-[#3a3a5a] hover:text-[#8a8aaa]'}`}
          title="Toggle auto-scroll"
        >
          AUTO
        </button>
      </div>

      {/* Content */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-2 font-mono text-[11px]">
        {loadError ? (
          <div className="p-3 text-[#8a8aaa]">
            LOGS: UNAVAILABLE — the log bridge returned an error ({loadError}). No lines are invented.
          </div>
        ) : payload === null ? (
          <div className="p-3 text-[#5a5a7a]">Reading backend run record…</div>
        ) : !payload.available ? (
          <div className="p-3 text-[#8a8aaa]">
            LOGS: UNAVAILABLE — {payload.reason ?? 'no reconstruction run record exists yet'}. Run the
            pipeline (Build workspace or `scripts/run_south_building_e2e.py`) to produce real logs;
            none are synthesized here.
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="p-3 text-[#5a5a7a]">No entries at level {selectedLevel}.</div>
        ) : (
          <div className="flex flex-col space-y-1">
            {filteredLogs.map((log) => (
              <div key={log.id} className="flex items-start space-x-2">
                <span className="text-[#5a5a7a] shrink-0">{log.timestamp}</span>
                <span className={`px-1 rounded text-[9px] shrink-0 ${getLevelStyles(log.level)}`}>
                  {log.level}
                </span>
                <span className="text-[#c77df5] shrink-0">{log.module}</span>
                <span className="text-gray-400 break-words">{log.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default LogStream;
