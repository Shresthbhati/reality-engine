'use client';

import React from 'react';
import { AlertOctagon, RefreshCw } from 'lucide-react';

export interface ErrorStateProps {
  title?: string;
  message: string;
  code?: string;
  details?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  title = 'Reconstruction Error',
  message,
  code,
  details,
  onRetry,
  className = '',
}: ErrorStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center p-6 rounded-xl border border-[#e54d4d]/30 bg-[#161014]/60 backdrop-blur-sm select-none ${className}`}
    >
      <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 bg-[#e54d4d]/10 border border-[#e54d4d]/30 text-[#e54d4d]">
        <AlertOctagon className="w-5 h-5" />
      </div>

      <h3 className="text-xs font-semibold text-[#fca5a5] tracking-wide font-sans mb-1">
        {title}
      </h3>

      <p className="text-[11px] text-[#9296a6] max-w-sm leading-relaxed font-sans mb-3">
        {message}
      </p>

      {code && (
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-black/40 border border-[#e54d4d]/20 text-[#e54d4d] mb-3">
          ERR_CODE: {code}
        </span>
      )}

      {details && (
        <pre className="text-[9px] font-mono text-left max-w-md w-full p-2.5 rounded bg-black/50 border border-white/5 text-[#9296a6] overflow-x-auto mb-4 whitespace-pre-wrap">
          {details}
        </pre>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-medium border border-[#e54d4d]/40 bg-[#e54d4d]/10 text-[#fca5a5] hover:bg-[#e54d4d]/20 transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Operation</span>
        </button>
      )}
    </div>
  );
}
