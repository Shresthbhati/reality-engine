'use client';

import React, { useEffect } from 'react';
import { useREStore } from '@/store/re-store';
import type { UINotification, NotificationType } from '@/types/reality-engine';
import {
  Info,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  Workflow,
  X,
} from 'lucide-react';

const ICON_MAP: Record<NotificationType, React.ReactNode> = {
  info: <Info className="w-4 h-4 text-[#38bdf8] shrink-0" />,
  success: <CheckCircle2 className="w-4 h-4 text-[#2ecc71] shrink-0" />,
  warning: <AlertTriangle className="w-4 h-4 text-[#f5a623] shrink-0" />,
  error: <AlertOctagon className="w-4 h-4 text-[#e54d4d] shrink-0" />,
  pipeline: <Workflow className="w-4 h-4 text-[#00e5ff] shrink-0" />,
};

const BORDER_MAP: Record<NotificationType, string> = {
  info: 'border-[#38bdf8]/40',
  success: 'border-[#2ecc71]/40',
  warning: 'border-[#f5a623]/40',
  error: 'border-[#e54d4d]/40',
  pipeline: 'border-[#00e5ff]/50',
};

function ToastItem({ notification }: { notification: UINotification }) {
  const dismissNotification = useREStore((s) => s.dismissNotification);

  useEffect(() => {
    if (notification.durationMs && notification.durationMs > 0) {
      const timer = setTimeout(() => {
        dismissNotification(notification.id);
      }, notification.durationMs);
      return () => clearTimeout(timer);
    }
  }, [notification.id, notification.durationMs, dismissNotification]);

  return (
    <div
      className={`w-80 flex items-start gap-3 p-3.5 rounded-xl border bg-[#101217]/95 backdrop-blur-md shadow-2xl transition-all animate-in slide-in-from-bottom-2 duration-150 ${BORDER_MAP[notification.type]}`}
      role="alert"
    >
      <div className="pt-0.5">{ICON_MAP[notification.type]}</div>

      <div className="flex-1 min-w-0 font-sans">
        <div className="flex items-center justify-between gap-1">
          <span className="text-xs font-semibold text-[#f0f1f6] truncate">
            {notification.title}
          </span>
          <span className="text-[9px] font-mono text-[#54596b] shrink-0">
            {new Date(notification.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            })}
          </span>
        </div>

        <p className="text-[11px] text-[#9296a6] mt-0.5 leading-relaxed break-words">
          {notification.message}
        </p>

        {notification.actionLabel && notification.onAction && (
          <button
            type="button"
            onClick={() => {
              notification.onAction?.();
              dismissNotification(notification.id);
            }}
            className="mt-2 text-[10px] font-mono font-bold text-[#00e5ff] hover:underline"
          >
            {notification.actionLabel} →
          </button>
        )}
      </div>

      <button
        type="button"
        onClick={() => dismissNotification(notification.id)}
        className="text-[#54596b] hover:text-[#ededf2] transition-colors p-0.5"
        aria-label="Dismiss notification"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

export function ToastContainer() {
  const notifications = useREStore((s) => s.notifications);

  if (!notifications || notifications.length === 0) return null;

  return (
    <div
      className="fixed bottom-8 right-4 z-50 flex flex-col gap-2 pointer-events-auto max-h-[70vh] overflow-hidden"
      aria-live="polite"
    >
      {notifications.slice(-5).map((n) => (
        <ToastItem key={n.id} notification={n} />
      ))}
    </div>
  );
}
