'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import { X } from 'lucide-react';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showCloseButton?: boolean;
  closeOnBackdrop?: boolean;
}

const SIZE_CLASSES: Record<NonNullable<ModalProps['size']>, string> = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
  full: 'max-w-[95vw] h-[90vh]',
};

export function Modal({
  isOpen,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'md',
  showCloseButton = true,
  closeOnBackdrop = true,
}: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{
        background: 'rgba(5, 6, 8, 0.75)',
        backdropFilter: 'blur(8px)',
      }}
      onClick={closeOnBackdrop ? onClose : undefined}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={modalRef}
        className={`w-full ${SIZE_CLASSES[size]} flex flex-col rounded-xl overflow-hidden shadow-2xl border transition-all animate-in fade-in zoom-in-95 duration-150`}
        style={{
          background: 'var(--re-bg-elevated, #101217)',
          borderColor: 'var(--re-border-default, #1f222b)',
          color: 'var(--re-text-primary, #ededf2)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div
            className="flex items-center justify-between px-5 py-3.5 border-b select-none shrink-0"
            style={{ borderColor: 'var(--re-border-default, #1f222b)' }}
          >
            <div>
              {title && (
                <h2 className="text-sm font-semibold tracking-wide text-[#f0f1f6] font-sans">
                  {title}
                </h2>
              )}
              {description && (
                <p className="text-xs text-[#9296a6] mt-0.5 font-sans">
                  {description}
                </p>
              )}
            </div>

            {showCloseButton && (
              <button
                type="button"
                onClick={onClose}
                className="p-1 rounded-md text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
                aria-label="Close dialog"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 text-xs font-sans leading-relaxed">
          {children}
        </div>

        {/* Footer */}
        {footer && (
          <div
            className="flex items-center justify-end gap-2.5 px-5 py-3 border-t shrink-0 select-none bg-black/20"
            style={{ borderColor: 'var(--re-border-default, #1f222b)' }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
