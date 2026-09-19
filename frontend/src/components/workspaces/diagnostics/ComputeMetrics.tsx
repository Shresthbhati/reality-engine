'use client';

import React from 'react';
import { ComputeMetrics } from '@/types/reality-engine';

interface Props {
  metrics: ComputeMetrics;
}

const formatGB = (mb: number): string => (mb / 1024).toFixed(1) + ' GB';

const BarGauge = ({
  label,
  value,
  max = 100,
  valueLabel,
  colorOverride,
}: {
  label: string;
  value: number;
  max?: number;
  valueLabel?: React.ReactNode;
  colorOverride?: string;
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  let color = 'bg-green-500';
  if (colorOverride) {
    color = colorOverride;
  } else {
    if (percentage >= 90) color = 'bg-red-500';
    else if (percentage >= 70) color = 'bg-yellow-500';
  }

  return (
    <div className="flex items-center gap-3 w-full my-1.5 text-[10px]">
      <div className="w-[40px] text-[#7a7a9a] uppercase">{label}</div>
      <div className="flex-1 h-1.5 bg-[#1e1e2e] overflow-hidden rounded-sm relative">
        <div
          className={`absolute top-0 left-0 h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="w-[75px] text-right text-[#c0c0d0] whitespace-nowrap">
        {valueLabel || `${Math.round(value)}%`}
      </div>
    </div>
  );
};

export default function ComputeMetricsPanel({ metrics }: Props) {
  const {
    gpuUsage,
    gpuVramUsed,
    gpuVramTotal,
    cpuUsage,
    ramUsed,
    ramTotal,
    diskRead,
    diskWrite,
    networkIn,
    networkOut,
  } = metrics;

  return (
    <div className="flex flex-col bg-[#121215] h-full font-mono text-xs text-[#c0c0d0] overflow-y-auto">
      {/* Header */}
      <div className="flex justify-between items-center px-3 py-2 border-b border-[#1e1e24]">
        <div className="uppercase tracking-widest text-[10px] text-[#5a5a7a]">Compute</div>
        <div className="text-[10px] text-[#3a3a5a]">RTX 4090</div>
      </div>

      {/* GPU Section */}
      <div className="px-3 py-2 border-b border-[#1a1a22]">
        <div className="flex justify-between items-center mb-2">
          <span className="text-[#5a5a7a]">GPU</span>
          <span className="text-[#3a3a5a] text-[9px]">NVIDIA RTX 4090</span>
        </div>
        <BarGauge label="USAGE" value={gpuUsage} />
        <BarGauge
          label="VRAM"
          value={gpuVramUsed}
          max={gpuVramTotal}
          valueLabel={`${formatGB(gpuVramUsed)} / ${formatGB(gpuVramTotal)}`}
        />
      </div>

      {/* CPU Section */}
      <div className="px-3 py-2 border-b border-[#1a1a22]">
        <div className="flex justify-between items-center mb-2">
          <span className="text-[#5a5a7a]">CPU</span>
          <span className="text-[#3a3a5a] text-[9px]">AMD Ryzen 9 7950X</span>
        </div>
        <BarGauge label="USAGE" value={cpuUsage} />
        <BarGauge
          label="RAM"
          value={ramUsed}
          max={ramTotal}
          valueLabel={`${formatGB(ramUsed)} / ${formatGB(ramTotal)}`}
        />
      </div>

      {/* STORAGE Section */}
      <div className="px-3 py-2 border-b border-[#1a1a22]">
        <div className="mb-2 text-[#5a5a7a]">STORAGE</div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">DISK READ</span>
          <span>{diskRead === 0 ? '--' : `${diskRead.toFixed(1)} MB/s`}</span>
        </div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">DISK WRITE</span>
          <span>{diskWrite === 0 ? '--' : `${diskWrite.toFixed(1)} MB/s`}</span>
        </div>
      </div>

      {/* NETWORK Section */}
      <div className="px-3 py-2 border-b border-[#1a1a22]">
        <div className="mb-2 text-[#5a5a7a]">NETWORK</div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">IN</span>
          <span>{networkIn.toFixed(1)} KB/s</span>
        </div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">OUT</span>
          <span>{networkOut.toFixed(1)} KB/s</span>
        </div>
      </div>

      {/* TEMPERATURE Section */}
      <div className="px-3 py-2 border-b border-[#1a1a22]">
        <div className="mb-2 text-[#5a5a7a]">TEMPERATURE</div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">GPU TEMP</span>
          <span className="text-yellow-500">71°C</span>
        </div>
        <div className="flex justify-between my-1">
          <span className="text-[#7a7a9a]">CPU TEMP</span>
          <span className="text-green-500">58°C</span>
        </div>
      </div>

      {/* STATUS Section */}
      <div className="px-3 py-2 text-[10px] text-[#3a3a5a] space-y-1">
        <div>DRIVER: 545.29.06</div>
        <div>CUDA: 12.3</div>
        <div>BACKEND: COLMAP CUDA 4.2</div>
      </div>
    </div>
  );
}
