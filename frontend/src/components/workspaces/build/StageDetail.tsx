'use client';

import React from 'react';
import { PipelineStageInfo } from '@/types/reality-engine';

interface Props {
  stage: PipelineStageInfo;
  buildId: string;
}

export function StageDetail({ stage, buildId }: Props) {
  const formatDuration = (ms?: number) => {
    if (!ms) return '0s';
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
    if (m > 0) return `${m}m ${s % 60}s`;
    return `${s}s`;
  };

  const formatVRAM = (mb?: number) => {
    if (mb === undefined) return 'N/A';
    if (mb > 1000) return `${(mb / 1024).toFixed(2)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  const getStatusClasses = (status: string) => {
    switch (status) {
      case 'COMPLETE':
        return 'bg-green-900/40 text-green-400 border border-green-700';
      case 'RUNNING':
        return 'bg-blue-900/40 text-[#3d8ef7] border border-[#3d8ef7] animate-pulse';
      case 'FAILED':
        return 'bg-red-900/40 text-red-400 border border-red-700';
      case 'WARNING':
        return 'bg-yellow-900/40 text-yellow-400 border border-yellow-700';
      case 'PENDING':
      default:
        return 'bg-[#1e1e2e] text-[#5a5a7a] border border-[#3e3e4e]';
    }
  };

  const fakeLogs = [
    `[09:14:22.441] INFO colmap.mapper Initializing stage ${stage.stage}...`,
    `[09:14:23.102] DEBUG colmap.mapper Loading configuration parameters`,
    `[09:14:25.889] INFO colmap.mapper Allocating GPU memory (CUDA 4.2)`,
    `[09:14:30.001] WARN colmap.mapper Minor memory fragmentation detected`,
    `[09:14:32.441] INFO colmap.mapper Processing chunk 1/10...`,
    `[09:14:45.992] INFO colmap.mapper Processing chunk 5/10...`,
    `[09:14:58.210] INFO colmap.mapper Processing chunk 10/10...`,
    `[09:15:02.333] INFO colmap.mapper Optimizing parameters`,
    `[09:15:05.124] INFO colmap.mapper Finalizing stage output`,
    `[09:15:06.000] INFO colmap.mapper Stage completed successfully`,
  ];

  return (
    <div className="bg-[#0d0d0f] h-full flex flex-col overflow-y-auto">
      {/* Header section */}
      <div className="p-4 border-b border-[#1e1e24] flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-mono font-semibold text-[#e8e8f0] uppercase tracking-widest">
            {stage.stage}
          </h2>
          <span className={`px-2 py-0.5 text-xs font-mono rounded ${getStatusClasses(stage.status)}`}>
            {stage.status}
          </span>
        </div>
        <div className="text-xs font-mono text-[#5a5a7a]">
          BACKEND: {stage.backend || 'COLMAP CUDA 4.2'}
        </div>
        <div className="text-xs font-mono text-[#5a5a7a]">
          {`// WIRE: GET /api/builds/${buildId}/stages/${stage.stage}/logs`}
        </div>
      </div>

      <div className="p-4 flex flex-col gap-6">
        {/* Progress section */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs font-mono text-[#e8e8f0]">
            <span>PROGRESS</span>
            <span>{stage.progress}%</span>
          </div>
          <div className="h-2 w-full bg-[#1e1e24] rounded overflow-hidden">
            <div 
              className="h-full bg-[#3d8ef7] transition-all duration-500"
              style={{ width: `${stage.progress}%` }}
            />
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[#121215] p-3 rounded border border-[#1e1e24] flex flex-col gap-1">
            <span className="text-[10px] font-mono text-[#5a5a7a] uppercase tracking-wider">Runtime</span>
            <span className="text-sm font-mono text-[#e8e8f0]">{formatDuration(stage.durationMs)}</span>
          </div>
          <div className="bg-[#121215] p-3 rounded border border-[#1e1e24] flex flex-col gap-1">
            <span className="text-[10px] font-mono text-[#5a5a7a] uppercase tracking-wider">GPU / VRAM</span>
            <span className="text-sm font-mono text-[#e8e8f0]">
              {stage.gpuUsage !== undefined ? `${stage.gpuUsage}%` : 'N/A'} / {formatVRAM(stage.ramUsageMb)}
            </span>
          </div>
        </div>

        {/* Metrics table */}
        {stage.metrics && Object.keys(stage.metrics).length > 0 && (
          <div className="flex flex-col gap-2">
            <h3 className="text-[10px] font-mono text-[#5a5a7a] uppercase tracking-wider">Metrics</h3>
            <div className="bg-[#121215] rounded border border-[#1e1e24] overflow-hidden">
              <table className="w-full text-xs font-mono text-left">
                <thead className="bg-[#1a1a20] border-b border-[#1e1e24]">
                  <tr>
                    <th className="px-3 py-2 text-[#5a5a7a] font-normal">METRIC</th>
                    <th className="px-3 py-2 text-[#5a5a7a] font-normal">VALUE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e1e24]">
                  {Object.entries(stage.metrics).map(([key, value]) => (
                    <tr key={key}>
                      <td className="px-3 py-2 text-[#e8e8f0]">{key}</td>
                      <td className="px-3 py-2 text-[#e8e8f0]">{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Warnings */}
        {stage.warnings && stage.warnings.length > 0 && (
          <div className="flex flex-col gap-2">
            <h3 className="text-[10px] font-mono text-yellow-500 uppercase tracking-wider">WARNINGS</h3>
            <div className="flex flex-col gap-2">
              {stage.warnings.map((warning, idx) => (
                <div key={idx} className="bg-yellow-900/20 border-l-2 border-yellow-500 p-2 font-mono text-xs text-yellow-400/90 break-words">
                  {warning}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Errors */}
        {stage.errors && stage.errors.length > 0 && (
          <div className="flex flex-col gap-2">
            <h3 className="text-[10px] font-mono text-red-500 uppercase tracking-wider">ERRORS</h3>
            <div className="flex flex-col gap-2">
              {stage.errors.map((error, idx) => (
                <div key={idx} className="bg-red-900/20 border-l-2 border-red-500 p-2 font-mono text-xs text-red-400 break-words">
                  {error}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Logs */}
        <div className="flex flex-col gap-2 flex-grow">
          <h3 className="text-[10px] font-mono text-[#5a5a7a] uppercase tracking-wider">STAGE LOGS</h3>
          <div className="bg-[#080810] h-64 rounded border border-[#1e1e24] p-3 font-mono text-xs overflow-y-auto">
            {fakeLogs.map((log, idx) => {
              const isError = log.includes('ERROR');
              const isWarn = log.includes('WARN');
              const isInfo = log.includes('INFO');
              
              let levelColor = 'text-[#5a5a7a]';
              if (isError) levelColor = 'text-red-400';
              if (isWarn) levelColor = 'text-yellow-400';
              if (isInfo) levelColor = 'text-[#3d8ef7]';

              const parts = log.match(/^(\[\d{2}:\d{2}:\d{2}\.\d{3}\])\s+([A-Z]+)\s+(.*)$/);
              if (parts) {
                return (
                  <div key={idx} className="mb-1 flex gap-2">
                    <span className="text-[#5a5a7a] shrink-0">{parts[1]}</span>
                    <span className={`shrink-0 w-12 ${levelColor}`}>{parts[2]}</span>
                    <span className="text-[#c8c8e0] break-all">{parts[3]}</span>
                  </div>
                );
              }
              return <div key={idx} className="mb-1 text-[#c8c8e0]">{log}</div>;
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
