'use client';

import React, { useEffect, useState } from 'react';
import { useREStore } from '@/store/re-store';
import { PipelineStageList } from './PipelineStageList';
import { StageDetail } from './StageDetail';

export function BuildWorkspace() {
  const { builds, loadMockData } = useREStore();
  const [selectedStageIndex, setSelectedStageIndex] = useState(0);

  useEffect(() => {
    if (builds.length === 0) {
      loadMockData();
    }
  }, [builds.length, loadMockData]);

  if (builds.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[#0d0d0f] text-[#e8e8f0] font-mono text-sm">
        LOADING MOCK DATA...
      </div>
    );
  }

  const currentBuild = builds[0];
  const { status, stages } = currentBuild;

  // Calculate overall progress based on stages
  const overallProgress = stages.length > 0 
    ? stages.reduce((sum, stage) => sum + stage.progress, 0) / stages.length 
    : 0;

  const getStatusColor = (s: string) => {
    switch (s) {
      case 'COMPLETE': return 'text-green-400';
      case 'RUNNING': return 'text-[#3d8ef7]';
      case 'FAILED': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = (s: string) => {
    switch (s) {
      case 'COMPLETE': return 'bg-green-400/10 border-green-400/20';
      case 'RUNNING': return 'bg-[#3d8ef7]/10 border-[#3d8ef7]/20';
      case 'FAILED': return 'bg-red-400/10 border-red-400/20';
      default: return 'bg-gray-400/10 border-gray-400/20';
    }
  };

  return (
    <div className="flex flex-col h-full w-full bg-[#0d0d0f] text-[#e8e8f0] font-mono">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 h-10 border-b border-[#1e1e24] bg-[#121215] shrink-0 text-xs">
        <div className="flex items-center gap-2 text-gray-400">
          <span className="font-bold text-[#e8e8f0]">REALITY ENGINE</span>
          <span>/</span>
          <span>BUILD WORKSPACE</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-gray-400">PROJECT: {currentBuild.projectId}</span>
          <div className={`px-2 py-0.5 rounded border ${getStatusBg(status)} ${getStatusColor(status)}`}>
            {status}
          </div>
        </div>
      </div>

      {/* Header row */}
      <div className="flex flex-col px-4 py-3 border-b border-[#1e1e24] bg-[#0d0d0f] shrink-0 gap-2">
        <div className="text-sm">
          <span className="font-bold text-[#e8e8f0]">BUILD: SITE_042</span>
          <span className={`ml-2 ${getStatusColor(status)}`}>[{status}]</span>
        </div>
        <div className="h-1.5 w-full bg-[#1e1e24] rounded-full overflow-hidden">
          <div 
            className="h-full bg-[#3d8ef7] transition-all duration-300"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Main body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        <div className="w-[280px] border-r border-[#1e1e24] bg-[#121215] flex flex-col shrink-0 overflow-y-auto">
          <PipelineStageList 
            stages={stages}
            selectedIndex={selectedStageIndex}
            onSelect={setSelectedStageIndex}
          />
        </div>

        {/* Right detail panel */}
        <div className="flex-1 bg-[#0d0d0f] flex flex-col min-w-0 overflow-y-auto">
          {stages[selectedStageIndex] ? (
            <StageDetail 
              stage={stages[selectedStageIndex]} 
              buildId={currentBuild.id}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              No stage selected
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default BuildWorkspace;
