'use client';

import React from 'react';
import { PipelineStageInfo } from '@/types/reality-engine';

interface Props {
  stages: PipelineStageInfo[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

function formatDuration(ms?: number) {
  if (ms === undefined || ms === null) return '';
  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const m = Math.floor(totalSeconds / 60) % 60;
  const h = Math.floor(totalSeconds / 3600);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${totalSeconds % 60}s`;
}

function getStatusIcon(status: PipelineStageInfo['status']) {
  switch (status) {
    case 'COMPLETE': return <span className="text-green-400">✓</span>;
    case 'RUNNING': return <span className="text-[#3d8ef7] animate-pulse">►</span>;
    case 'PENDING': return <span className="text-[#3a3a50]">○</span>;
    case 'WARNING': return <span className="text-yellow-400">⚠</span>;
    case 'FAILED': return <span className="text-red-400">✗</span>;
    case 'QUEUED': return <span className="text-[#3a3a50]">◈</span>;
    case 'SKIPPED': return <span className="text-[#3a3a50]">—</span>;
    default: return <span className="text-[#3a3a50]">○</span>;
  }
}

function getProgressColor(status: PipelineStageInfo['status']) {
  switch (status) {
    case 'COMPLETE': return 'bg-green-500';
    case 'FAILED': return 'bg-red-500';
    case 'WARNING': return 'bg-yellow-500';
    default: return 'bg-[#3d8ef7]';
  }
}

export function PipelineStageList({ stages, selectedIndex, onSelect }: Props) {
  const overallProgress = stages.length > 0
    ? stages.reduce((acc, stage) => acc + (stage.progress || 0), 0) / stages.length
    : 0;
  const completedStages = stages.filter(s => s.status === 'COMPLETE').length;

  return (
    <div className="bg-[#121215] h-full flex flex-col font-mono">
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-between border-b border-[#1a1a22]">
        <div className="text-xs uppercase tracking-widest text-[#5a5a7a]">PIPELINE</div>
        <div className="text-[10px] text-[#4a4a6a]">{stages.length} STAGES</div>
      </div>
      
      {/* Overall Progress */}
      <div className="h-[2px] w-full bg-[#1e1e2e]">
        <div 
          className="h-full bg-[#3d8ef7] transition-all duration-300"
          style={{ width: `${overallProgress}%` }}
        />
      </div>

      {/* Stage List */}
      <div className="flex-1 overflow-y-auto">
        {stages.map((stage, i) => {
          const isSelected = i === selectedIndex;
          return (
            <div
              key={stage.stage || i}
              onClick={() => onSelect(i)}
              className={`h-10 flex flex-col justify-center px-3 cursor-pointer border-b border-[#1a1a22] hover:bg-[#1a1a22] transition-colors ${
                isSelected ? 'bg-[#1a1f35] border-l-2 border-l-[#3d8ef7]' : 'border-l-2 border-l-transparent'
              }`}
            >
              <div className="flex items-center w-full">
                <div className="w-4 text-[12px] flex justify-center mr-2">
                  {getStatusIcon(stage.status)}
                </div>
                <div className={`flex-1 text-xs uppercase tracking-wide truncate ${isSelected ? 'text-[#3d8ef7]' : 'text-[#e8e8f0]'}`}>
                  {stage.stage}
                </div>
                {stage.backend && (
                  <div className="text-[10px] text-[#4a4a6a] mr-2 truncate max-w-[80px]">
                    {stage.backend}
                  </div>
                )}
                <div className="text-[10px] text-[#4a4a6a]">
                  {formatDuration(stage.durationMs)}
                </div>
              </div>
              <div className="mt-1 h-[3px] w-full bg-[#1e1e2e] rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${getProgressColor(stage.status)}`}
                  style={{ width: `${stage.progress || 0}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-3 py-2 flex items-center justify-between">
        <div className="text-[10px] text-[#5a5a7a] tracking-wider">
          {completedStages} / {stages.length} COMPLETE
        </div>
      </div>
    </div>
  );
}

export default PipelineStageList;
