"use client";

import { useEffect, useState } from "react";
import WorkspaceBar from "./WorkspaceBar";
import CommandPalette from "@/components/shell/CommandPalette";

export interface WorkspaceShellProps {
  worldName: string;
  worldId: string;
  version?: string;
  coordinateSystem?: string;
  isProcessing?: boolean;
  children: React.ReactNode;
}

export default function WorkspaceShell({
  worldName,
  worldId,
  version,
  coordinateSystem,
  isProcessing,
  children,
}: WorkspaceShellProps) {
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Apply the user's persisted reduce-motion preference (set in Settings)
  // on every load, not just while the Settings page itself is mounted.
  useEffect(() => {
    try {
      if (window.localStorage.getItem("re-reduce-motion") === "true") {
        document.documentElement.setAttribute("data-reduce-motion", "true");
      }
    } catch {
      // Storage unavailable — falls back to the OS-level media query only.
    }
  }, []);

  return (
    <div
      className="flex flex-col h-screen w-screen overflow-hidden"
      style={{ background: "var(--bg-base)" }}
    >
      <WorkspaceBar
        worldName={worldName}
        worldId={worldId}
        version={version}
        coordinateSystem={coordinateSystem}
        isProcessing={isProcessing}
      />
      <main className="flex-1 min-h-0 min-w-0 overflow-hidden relative">
        {children}
      </main>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
