'use client';

import React from 'react';
import { Box, LucideIcon } from 'lucide-react';

export interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  icon: Icon = Box,
  title,
  description,
  action,
  className = '',
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center p-6 rounded-xl border border-dashed select-none ${className}`}
      style={{
        background: 'rgba(21, 24, 33, 0.4)',
        borderColor: 'var(--re-border-default, #1f222b)',
      }}
    >
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 border shadow-inner"
        style={{
          background: 'rgba(0, 229, 255, 0.05)',
          borderColor: 'rgba(0, 229, 255, 0.2)',
          color: 'var(--re-accent, #00e5ff)',
        }}
      >
        <Icon className="w-5 h-5 opacity-80" />
      </div>

      <h3 className="text-xs font-semibold text-[#f0f1f6] tracking-wide font-sans mb-1">
        {title}
      </h3>

      {description && (
        <p className="text-[11px] text-[#9296a6] max-w-xs leading-relaxed font-sans mb-3">
          {description}
        </p>
      )}

      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="px-3 py-1.5 rounded-md text-xs font-mono font-medium border transition-all hover:bg-[#00e5ff]/10 hover:border-[#00e5ff]/50"
          style={{
            background: 'var(--re-bg-surface, #151821)',
            borderColor: 'var(--re-border-default, #1f222b)',
            color: 'var(--re-accent, #00e5ff)',
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
