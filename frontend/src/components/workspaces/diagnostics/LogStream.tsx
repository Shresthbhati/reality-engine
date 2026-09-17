'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import type { LogEntry, LogLevel } from '@/types/reality-engine';

const MODULES = [
  'colmap.mapper',
  'openmvs.dense',
  'world_ir.compiler',
  'perception.sam2',
  'system.gpu',
  'pipeline.orchestrator',
  'storage.writer',
  'fusion.open3d'
];

const LEVELS: LogLevel[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
const ALL_LEVEL_OPTIONS: ('ALL' | LogLevel)[] = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const MESSAGES = [
  'Initializing compute context',
  'Allocating GPU memory buffers',
  'Found 3215 feature matches',
  'Performing bundle adjustment step 1/4',
  'Failed to read sensor data stream',
  'Point cloud density: 4.2M points/m3',
  'Memory limit approaching (85%)',
  'Frame dropped due to latency spike',
  'Reconstruction pipeline started',
  'Writing cache chunk to disk',
  'Optimizing mesh topology',
  'CUDA kernel panic: Out of memory',
  'Loading model weights',
  'Camera pose estimation converged',
  'Semantic segmentation complete',
  'Unrecognized intrinsic parameters',
];

const generateFakeLogs = (): LogEntry[] => {
  const logs: LogEntry[] = [];
  const now = Date.now();
  for (let i = 0; i < 20; i++) {
    // Generate a random level biased towards INFO
    const rand = Math.random();
    let level: LogLevel = 'INFO';
    if (rand < 0.1) level = 'DEBUG';
    else if (rand < 0.7) level = 'INFO';
    else if (rand < 0.85) level = 'WARNING';
    else if (rand < 0.98) level = 'ERROR';
    else level = 'CRITICAL';

    const timestamp = new Date(now - (20 - i) * 1500).toISOString();
    const modName = MODULES[Math.floor(Math.random() * MODULES.length)];
    const message = MESSAGES[Math.floor(Math.random() * MESSAGES.length)];

    logs.push({
      id: `log-${i}-${Math.random().toString(36).slice(2, 11)}`,
      level,
      timestamp,
      module: modName,
      message,
    });
  }
  return logs;
};

const formatTime = (isoString: string) => {
  try {
    const d = new Date(isoString);
    const h = d.getHours().toString().padStart(2, '0');
    const m = d.getMinutes().toString().padStart(2, '0');
    const s = d.getSeconds().toString().padStart(2, '0');
    const ms = d.getMilliseconds().toString().padStart(3, '0');
    return `${h}:${m}:${s}.${ms}`;
  } catch {
    return '00:00:00.000';
  }
};

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
  const [logs, setLogs] = useState<LogEntry[]>(() => generateFakeLogs());
  const [selectedLevel, setSelectedLevel] = useState<'ALL' | LogLevel>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filteredLogs = useMemo(() => {
    if (selectedLevel === 'ALL') return logs;
    return logs.filter(log => log.level === selectedLevel);
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
        <div className="flex-1" />
        
        <select
          value={selectedLevel}
          onChange={(e) => setSelectedLevel(e.target.value as 'ALL' | LogLevel)}
          className="bg-[#111118] border border-[#1e1e24] text-[#8a8aaa] text-[10px] font-mono outline-none px-1 py-0.5 rounded focus:border-[#3d8ef7]"
        >
          {ALL_LEVEL_OPTIONS.map(lvl => (
            <option key={lvl} value={lvl}>{lvl}</option>
          ))}
        </select>
        
        <button
          onClick={() => setLogs([])}
          className="text-[#3a3a5a] text-[9px] hover:text-red-400 font-mono transition-colors"
        >
          CLEAR
        </button>
        
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className={`text-[9px] font-mono px-1.5 py-0.5 rounded transition-colors ${
            autoScroll 
              ? 'bg-[#3d8ef7]/20 text-[#3d8ef7]' 
              : 'bg-[#1e1e24] text-[#5a5a7a] hover:text-[#8a8aaa]'
          }`}
        >
          AUTO
        </button>
      </div>

      {/* Log list */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto overflow-x-hidden"
      >
        {filteredLogs.map((log) => (
          <div 
            key={log.id} 
            className="flex items-start gap-1 px-2 py-0.5 hover:bg-[#111118] border-b border-[#0d0d12] transition-colors"
          >
            <div className="text-[#3a3a5a] text-[10px] font-mono w-28 flex-shrink-0">
              {formatTime(log.timestamp)}
            </div>
            <div className={`text-[9px] font-mono font-bold px-1 rounded w-14 flex-shrink-0 text-center ${getLevelStyles(log.level)}`}>
              {log.level}
            </div>
            <div className="text-[#3a3a5a] text-[10px] font-mono w-32 flex-shrink-0 truncate" title={log.module}>
              {log.module}
            </div>
            <div className="text-[#c8c8e0] text-[10px] font-mono flex-1 break-words">
              {log.message}
            </div>
          </div>
        ))}
        {filteredLogs.length === 0 && (
          <div className="p-4 text-center text-[#3a3a5a] text-[10px] font-mono italic">
            No logs to display
          </div>
        )}
      </div>
    </div>
  );
}

export default LogStream;
