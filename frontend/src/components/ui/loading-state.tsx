'use client';

import React from 'react';

export function Skeleton({
  className = '',
  width,
  height,
}: {
  className?: string;
  width?: string | number;
  height?: string | number;
}) {
  return (
    <div
      className={`rounded-md bg-white/5 animate-pulse ${className}`}
      style={{
        width: width ?? '100%',
        height: height ?? '1rem',
      }}
    />
  );
}

export function Spinner({
  size = 'md',
  className = '',
}: {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}) {
  const sizeMap = {
    sm: 'w-3 h-3 border',
    md: 'w-4 h-4 border-2',
    lg: 'w-6 h-6 border-2',
  };

  return (
    <div
      className={`rounded-full border-[#00e5ff] border-t-transparent animate-spin ${sizeMap[size]} ${className}`}
    />
  );
}

export function GeometricLoader({
  label = 'PROCESSING SPATIAL DATA...',
  sublabel,
}: {
  label?: string;
  sublabel?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center p-6 select-none">
      {/* Geometric Hexagon Pulse */}
      <div className="relative w-12 h-12 mb-3 flex items-center justify-center">
        <div className="absolute inset-0 rounded-lg border border-[#00e5ff]/40 animate-ping opacity-30" />
        <div className="w-8 h-8 rounded-md border-2 border-[#00e5ff] border-t-transparent animate-spin" />
        <div className="absolute w-2 h-2 rounded-full bg-[#00e5ff] shadow-[0_0_8px_#00e5ff]" />
      </div>

      <span className="text-[11px] font-mono font-semibold tracking-widest text-[#00e5ff] uppercase">
        {label}
      </span>

      {sublabel && (
        <span className="text-[10px] text-[#9296a6] font-mono mt-1">
          {sublabel}
        </span>
      )}
    </div>
  );
}

export function LoadingOverlay({
  label,
  sublabel,
}: {
  label?: string;
  sublabel?: string;
}) {
  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-[#08090b]/80 backdrop-blur-sm">
      <GeometricLoader label={label} sublabel={sublabel} />
    </div>
  );
}
