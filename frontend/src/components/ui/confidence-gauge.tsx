'use client';

import React from 'react';

export interface ConfidenceGaugeProps {
  label: string;
  value: number; // 0 to 100
  unit?: string;
  displayValue?: string | number;
  variant?: 'confidence' | 'error' | 'uncertainty' | 'coverage';
  className?: string;
}

export function ConfidenceGauge({
  label,
  value,
  unit = '%',
  displayValue,
  variant = 'confidence',
  className = '',
}: ConfidenceGaugeProps) {
  const clamped = Math.min(100, Math.max(0, value));

  const colorMap = {
    confidence: {
      bar: 'bg-[#00e5ff]',
      text: 'text-[#00e5ff]',
      track: 'bg-[#152530]',
    },
    error: {
      bar: 'bg-[#f5a623]',
      text: 'text-[#f5a623]',
      track: 'bg-[#2a1d15]',
    },
    uncertainty: {
      bar: 'bg-[#a855f7]',
      text: 'text-[#a855f7]',
      track: 'bg-[#25182d]',
    },
    coverage: {
      bar: 'bg-[#2ecc71]',
      text: 'text-[#2ecc71]',
      track: 'bg-[#13281c]',
    },
  };

  const scheme = colorMap[variant];

  return (
    <div className={`flex flex-col gap-1 select-none font-mono ${className}`}>
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-[#9296a6]">{label}:</span>
        <span className={`font-bold ${scheme.text}`}>
          {displayValue !== undefined ? displayValue : `${clamped.toFixed(1)}${unit}`}
        </span>
      </div>

      <div className={`h-1.5 w-full rounded-full overflow-hidden ${scheme.track}`}>
        <div
          className={`h-full rounded-full transition-all duration-300 ${scheme.bar}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

export function UncertaintyInspectorGroup({
  geometricErrorCm = 2.1,
  semanticConfidencePct = 98.2,
  spatialCoveragePct = 94.0,
}: {
  geometricErrorCm?: number;
  semanticConfidencePct?: number;
  spatialCoveragePct?: number;
}) {
  return (
    <div className="space-y-2.5 p-3 rounded-lg bg-[#14161f] border border-[#1f222b]">
      <ConfidenceGauge
        label="Geometric Error"
        value={Math.min(100, (geometricErrorCm / 5.0) * 100)}
        displayValue={`${geometricErrorCm.toFixed(1)} cm`}
        variant="error"
      />

      <ConfidenceGauge
        label="Semantic Confidence"
        value={semanticConfidencePct}
        variant="confidence"
      />

      <ConfidenceGauge
        label="Spatial Coverage"
        value={spatialCoveragePct}
        variant="coverage"
      />
    </div>
  );
}
