"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Bell, Settings } from "lucide-react";
import NotificationsPanel from "@/components/shell/NotificationsPanel";
import { cn } from "@/lib/utils";

export interface WorkspaceBarProps {
  worldName: string;
  worldId: string;
  version?: string;
  coordinateSystem?: string;
  isProcessing?: boolean;
}

export default function WorkspaceBar({
  worldName,
  worldId,
  version = "v1",
  coordinateSystem = "ENU",
  isProcessing = false,
}: WorkspaceBarProps) {
  const router = useRouter();
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  return (
    <header
      className={cn(
        "h-9 shrink-0 flex items-center justify-between px-3 select-none border-b relative z-30",
        "bg-[var(--bg-base)]"
      )}
      style={{
        background: "var(--bg-base)",
        borderColor: "var(--border-subtle)",
      }}
    >
      {/* Left side: Back to /worlds + World Name */}
      <div className="flex items-center gap-2 min-w-0">
        <button
          type="button"
          onClick={() => router.push("/worlds")}
          title="Back to Worlds"
          aria-label="Back to Worlds"
          className="w-6 h-6 rounded flex items-center justify-center transition-colors hover:bg-[var(--bg-elevated)] shrink-0"
          style={{ color: "var(--text-secondary)" }}
        >
          <ArrowLeft className="w-3.5 h-3.5" />
        </button>
        <span
          className="text-xs font-medium truncate max-w-[200px] sm:max-w-[320px] md:max-w-[420px]"
          style={{ color: "var(--text-primary)" }}
          title={worldName}
        >
          {worldName}
        </span>
      </div>

      {/* Center: Version badge + Coordinate System indicator */}
      <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5 pointer-events-none sm:pointer-events-auto">
        <span
          className="text-[11px] font-mono-num px-1.5 py-0.5 rounded leading-none shrink-0"
          style={{
            background: "var(--bg-elevated)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border-subtle)",
          }}
          title={`Version: ${version}`}
        >
          {version}
        </span>
        <span
          className="text-[11px] font-mono-num px-1.5 py-0.5 rounded leading-none flex items-center gap-1 shrink-0"
          style={{
            background: "var(--bg-surface)",
            color: "var(--text-tertiary)",
            border: "1px solid var(--border-subtle)",
          }}
          title={`Coordinate System: ${coordinateSystem}`}
        >
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{ background: "var(--accent)" }}
          />
          <span>{coordinateSystem}</span>
        </span>
      </div>

      {/* Right side: Processing indicator + Notifications + Settings */}
      <div className="flex items-center gap-1.5 shrink-0">
        {isProcessing && (
          <div
            className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono-num mr-1"
            style={{
              background: "var(--accent-subtle)",
              color: "var(--accent)",
              border: "1px solid var(--accent-border)",
            }}
            title="Processing active jobs"
            aria-label="Processing active jobs"
          >
            <span className="relative flex h-2 w-2">
              <span
                className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ background: "var(--accent)" }}
              />
              <span
                className="relative inline-flex rounded-full h-2 w-2"
                style={{ background: "var(--accent)" }}
              />
            </span>
            <span className="text-[11px] hidden sm:inline">Processing</span>
          </div>
        )}

        <button
          type="button"
          title="Notifications"
          aria-label="Notifications"
          aria-haspopup="true"
          aria-expanded={notificationsOpen}
          onClick={() => setNotificationsOpen((v) => !v)}
          className="w-6 h-6 rounded flex items-center justify-center transition-colors hover:bg-[var(--bg-elevated)]"
          style={{ color: "var(--text-secondary)" }}
        >
          <Bell className="w-3.5 h-3.5" />
        </button>
        <NotificationsPanel open={notificationsOpen} onClose={() => setNotificationsOpen(false)} />

        <button
          type="button"
          onClick={() => router.push("/settings")}
          title="Settings"
          aria-label="Settings"
          className="w-6 h-6 rounded flex items-center justify-center transition-colors hover:bg-[var(--bg-elevated)]"
          style={{ color: "var(--text-secondary)" }}
        >
          <Settings className="w-3.5 h-3.5" />
        </button>
      </div>
    </header>
  );
}
