'use client';

/**
 * Generic mobile bottom sheet — used to host desktop side-panel content
 * (WorldNavPanel, ContextInspectorPanel, etc.) as a slide-up drawer on
 * phone-width viewports instead of a permanent sidebar.
 *
 * Reuses the same visual language as ContextualBottomDrawer (dark
 * surface, mono headers, cyan accent) but is generic over children so it
 * can host arbitrary panel content without duplicating each panel.
 */

import { useEffect } from 'react';
import { X } from 'lucide-react';

interface MobileSheetProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  /** Fraction of viewport height, e.g. 0.7 for 70vh. Default 0.75. */
  heightFraction?: number;
}

export default function MobileSheet({
  open,
  onClose,
  title,
  children,
  heightFraction = 0.75,
}: MobileSheetProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col justify-end" role="dialog" aria-modal="true" aria-label={title}>
      {/* Backdrop */}
      <button
        type="button"
        aria-label="Close panel"
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Sheet */}
      <div
        className="relative flex flex-col w-full bg-[#101217] border-t border-[#1f222b] rounded-t-2xl shadow-2xl overflow-hidden animate-in slide-in-from-bottom duration-200"
        style={{
          maxHeight: `${heightFraction * 100}vh`,
          paddingBottom: 'env(safe-area-inset-bottom, 0px)',
        }}
      >
        {/* Drag handle */}
        <div className="flex justify-center pt-2 pb-1 shrink-0">
          <div className="w-9 h-1 rounded-full bg-[#2d323f]" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-4 pb-2 shrink-0 border-b border-[#1f222b]">
          <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#9296a6]">
            {title}
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex items-center justify-center w-11 h-11 -mr-2 rounded-full text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto overscroll-contain">{children}</div>
      </div>
    </div>
  );
}
