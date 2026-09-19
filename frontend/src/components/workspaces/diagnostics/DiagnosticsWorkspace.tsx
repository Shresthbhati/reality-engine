'use client';

import React, { useEffect } from 'react';
import { useREStore } from '@/store/re-store';
import { type ComputeMetrics, type Build, type PipelineStageInfo } from '@/types/reality-engine';
import ComputeMetricsPanel from './ComputeMetrics';
import LogStream from './LogStream';

const fakeEvents = [
  { time: '14:30:01', msg: 'Build build-001 completed' },
  { time: '14:28:45', msg: 'Stage COMPILATION finished in 60s' },
  { time: '14:25:10', msg: 'Started remote sync for build-001' },
  { time: '14:20:00', msg: 'User session started' },
  { time: '14:15:33', msg: 'Node cluster healthy' },
];

export default function DiagnosticsWorkspace() {
  const { builds, loadMockData, computeMetrics } = useREStore();

  useEffect(() => {
    if (builds.length === 0) {
      loadMockData();
    }
  }, [builds.length, loadMockData]);

  const activeBuild = builds[0];
  let currentStage: PipelineStageInfo | undefined = undefined;
  
  if (activeBuild) {
    currentStage = activeBuild.stages.find((s: PipelineStageInfo) => s.status === 'RUNNING') 
      || activeBuild.stages[activeBuild.stages.length - 1];
  }

  const overallProgress = activeBuild && activeBuild.stages.length > 0
    ? Math.round(activeBuild.stages.reduce((acc: number, s: PipelineStageInfo) => acc + (s.progress || 0), 0) / activeBuild.stages.length)
    : 0;

  return (
    <div className="flex flex-col flex-1 h-full bg-[#0c0c0e] text-[#e1e1e6] select-none">
      {/* Top Bar / Status */}
      <div className="h-10 border-b border-[#1e1e24] px-4 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="font-semibold text-white">ENGINE DIAGNOSTICS</span>
          <span className="text-[#8a8a93]">|</span>
          <span className="text-[#8a8a93]">NODE: local-cluster-0</span>
        </div>
        <div className="text-[#8a8a93]">
          STATUS: <span className="text-green-500 font-bold">OPTIMAL</span>
        </div>
      </div>

      {/* Main Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Column */}
        <div className="flex-1 flex flex-col border-r border-[#1e1e24] min-w-0">
          <ComputeMetricsPanel metrics={computeMetrics} />
        </div>

        {/* Center Column */}
        <div className="flex-1 flex flex-col border-r border-[#1e1e24] min-w-0 overflow-y-auto p-4">
          <div className="text-xs font-mono tracking-widest text-[#8a8a93] mb-4 uppercase">
            ACTIVE JOB
          </div>
          
          {!activeBuild ? (
            <div className="text-sm font-mono text-[#8a8a93]">NO ACTIVE JOB</div>
          ) : (
            <div className="flex flex-col space-y-6">
              <div className="flex justify-between items-start">
                <div>
                  <div className="font-mono text-lg text-white">{activeBuild.id}</div>
                  <div className="text-xs text-[#8a8a93] mt-1">Started: {activeBuild.startedAt}</div>
                </div>
                <div className={`text-[10px] px-2 py-1 rounded font-mono uppercase ${
                  activeBuild.status === 'RUNNING' ? 'bg-blue-900/30 text-blue-400 border border-blue-900' :
                  activeBuild.status === 'COMPLETE' ? 'bg-green-900/30 text-green-400 border border-green-900' :
                  activeBuild.status === 'FAILED' ? 'bg-red-900/30 text-red-400 border border-red-900' :
                  'bg-gray-800 text-gray-400 border border-gray-700'
                }`}>
                  {activeBuild.status}
                </div>
              </div>

              {currentStage && (
                <div className="bg-[#111115] border border-[#1e1e24] p-3 rounded">
                  <div className="text-xs text-[#8a8a93] mb-1 font-mono">CURRENT STAGE</div>
                  <div className="font-mono text-sm text-[#00ffcc]">{currentStage.stage}</div>
                </div>
              )}

              <div>
                <div className="flex justify-between text-xs mb-1 font-mono">
                  <span className="text-[#8a8a93]">OVERALL PROGRESS</span>
                  <span className="text-white">{overallProgress}%</span>
                </div>
                <div className="h-2 bg-[#1e1e24] rounded overflow-hidden">
                  <div 
                    className="h-full bg-[#00ffcc] transition-all duration-500"
                    style={{ width: `${overallProgress}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="text-xs text-[#8a8a93] mb-2 font-mono">STAGES</div>
                <div className="flex flex-col space-y-1">
                  {activeBuild.stages.map((stage: PipelineStageInfo, idx: number) => {
                    let dotClass = "bg-gray-600";
                    if (stage.status === 'COMPLETE') dotClass = "bg-green-500";
                    else if (stage.status === 'RUNNING') dotClass = "bg-blue-500 animate-pulse";
                    else if (stage.status === 'FAILED') dotClass = "bg-red-500";

                    return (
                      <div key={stage.stage || idx} className="flex items-center h-5 text-xs font-mono">
                        <div className={`w-1.5 h-1.5 rounded-full ${dotClass} mr-2`} />
                        <span className="flex-1 text-gray-300 truncate">{stage.stage}</span>
                        <span className={`text-[10px] ml-2 ${
                          stage.status === 'COMPLETE' ? 'text-green-500' :
                          stage.status === 'RUNNING' ? 'text-blue-500' :
                          stage.status === 'FAILED' ? 'text-red-500' :
                          'text-gray-500'
                        }`}>
                          {stage.status}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="text-xs font-mono text-[#8a8a93]">
                SESSIONS IN BUILD: <span className="text-white">{activeBuild.sessionIds?.length || 0}</span>
              </div>
            </div>
          )}

          <div className="mt-8 border-t border-[#1e1e24] pt-4">
            <div className="text-xs font-mono tracking-widest text-[#8a8a93] mb-3 uppercase">
              System Events
            </div>
            <div className="flex flex-col space-y-2">
              {fakeEvents.map((ev, i) => (
                <div key={i} className="flex items-start text-xs font-mono space-x-2">
                  <span className="text-[#8a8a93] shrink-0">[{ev.time}]</span>
                  <span className="text-gray-400 break-words">{ev.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="flex-1 flex flex-col min-w-0">
          <LogStream />
        </div>
      </div>
    </div>
  );
}
