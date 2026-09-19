'use client';

import React from 'react';
import { Check, CircleDot, Clock } from 'lucide-react';

export interface TimelineStage {
  id: string;
  label: string;
  status: 'COMPLETED' | 'IN_PROGRESS' | 'PENDING' | 'WARNING';
  timestamp?: string;
  description?: string;
}

export interface TimelinePrimitiveProps {
  stages: TimelineStage[];
  activeStageId?: string;
  onSelectStage?: (id: string) => void;
  className?: string;
  showTicks?: boolean;
}

export function TimelinePrimitive({
  stages,
  activeStageId,
  onSelectStage,
  className = '',
  showTicks = true,
}: TimelinePrimitiveProps) {
  return (
    <div className={`flex flex-col w-full select-none font-mono ${className}`}>
      {/* Stages Row with Connecting Track */}
      <div className="relative flex items-center justify-between w-full px-4 py-2">
        {/* Connecting Line Track */}
        <div className="absolute left-8 right-8 top-1/2 -translate-y-1/2 h-0.5 bg-[#1f222b] z-0" />

        {stages.map((stage, idx) => {
          const isSelected = stage.id === activeStageId;
          const isCompleted = stage.status === 'COMPLETED';
          const isInProgress = stage.status === 'IN_PROGRESS';
          const isWarning = stage.status === 'WARNING';

          return (
            <div
              key={stage.id}
              className="relative z-10 flex flex-col items-center group cursor-pointer"
              onClick={() => onSelectStage?.(stage.id)}
            >
              {/* Node Pip */}
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold border transition-all ${
                  isSelected
                    ? 'bg-[#00e5ff] text-[#08090b] border-[#00e5ff] shadow-[0_0_10px_rgba(0,229,255,0.4)] scale-110'
                    : isInProgress
                    ? 'bg-[#1a1d26] text-[#00e5ff] border-[#00e5ff] animate-pulse'
                    : isCompleted
                    ? 'bg-[#2ecc71]/20 text-[#2ecc71] border-[#2ecc71]/50'
                    : isWarning
                    ? 'bg-[#f5a623]/20 text-[#f5a623] border-[#f5a623]/50'
                    : 'bg-[#14161f] text-[#54596b] border-[#222633] group-hover:border-[#9296a6]'
                }`}
              >
                {isCompleted ? (
                  <Check className="w-3 h-3 stroke-[3]" />
                ) : isInProgress ? (
                  <CircleDot className="w-3 h-3" />
                ) : (
                  <span>{idx + 1}</span>
                )}
              </div>

              {/* Label Pill */}
              <span
                className={`mt-1.5 text-[10px] tracking-tight font-medium transition-colors ${
                  isSelected
                    ? 'text-[#00e5ff] font-bold'
                    : isInProgress
                    ? 'text-[#f0f1f6]'
                    : isCompleted
                    ? 'text-[#9296a6]'
                    : 'text-[#54596b] group-hover:text-[#9296a6]'
                }`}
              >
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Optional Continuous Time Axis Ticks */}
      {showTicks && (
        <div className="flex items-center justify-between px-6 pt-1 text-[8px] text-[#54596b] border-t border-[#1a1d26] mt-1">
          <span>0s [RAW INGEST]</span>
          <span>+2.4m [SFM POSE]</span>
          <span>+5.1m [DEPTH FUSION]</span>
          <span>+8.9m [SEMANTICS]</span>
          <span>LIVE [QUERYABLE]</span>
        </div>
      )}
    </div>
  );
}
